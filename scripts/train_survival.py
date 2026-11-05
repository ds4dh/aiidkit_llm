import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import yaml
import json
import gc
import torch
import wandb
import numpy as np
from pathlib import Path
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
from transformers.trainer_utils import get_last_checkpoint
from transformers.trainer_callback import PrinterCallback
from peft import LoraConfig, get_peft_model
from src.utils import apply_config_overrides
from src.data.patient_dataset import load_hf_data_and_metadata
from src.data.survival_labels import (
    resolve_survival_label_keys, derive_category_survival_labels, survival_keep_mask,
)
from src.model.patient_embedder import PatientEmbeddingModelFactory
from src.model.patient_embedder import PatientDataCollatorForSurvival
from src.model.survival_utils import make_survival_loss_func
from src.evaluation.evaluate_models import (
    DiscriminativeEmbeddingEvaluatorForSurvival as CustomEvaluator,
)
from scripts.train_classification import (
    PrefixAwareTrainer, preprocess_logits_for_metrics,
)
from scripts.script_utils import (
    scan_all_fups, prepare_dataset_fup_dict,
    save_predictions_table, extract_horizons_from_path,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune a patient transformer with a survival (Cox) objective."
    )
    parser.add_argument("--config", "-c", type=str, default="configs/survival_training.yaml")
    parser.add_argument("--reset_weights", "-r", action="store_true", help="Whether to reset model weights before fine-tuning.")
    parser.add_argument("--silent", "-s", action="store_true", help="Disable wandb logging.")
    parser.add_argument("--overrides", "-o", action="append", default=[], help="Overrides config (JSON string or key=value).")
    parser.add_argument("--tasks", "-t", type=str, default="", help="Comma-separated prediction_tasks keys to run.")
    return parser.parse_args()


def main():
    """
    Fine-tune survival models for the prediction tasks in the yaml config file.

    Instead of binary labels per horizon (train_classification.py), each patient
    provides a (time, status) pair measured from the fup observation point, and
    the model is trained with the Cox partial likelihood on a single log
    hazard-ratio output. Configured `horizons` are only used for reporting
    (predicted risk S(h), AUROC/AP/IPCW-Brier at h), keeping results directly
    comparable with the binary classification pipeline.
    """
    cli_args = parse_args()
    with open(cli_args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    cfg = apply_config_overrides(cfg, cli_args.overrides)
    if cli_args.tasks:
        wanted = [t.strip() for t in cli_args.tasks.split(",") if t.strip()]
        cfg["prediction_tasks"] = {t: cfg["prediction_tasks"][t] for t in wanted if t in cfg["prediction_tasks"]}

    # Identify pretraining and finetuning directories (same convention as classification)
    train_data_augment = cfg.get("train_data_augment", "all")
    mlm_masking_rules = cfg["data_collator"]["mlm_masking_rules"]
    pretrain_run_id = "-".join([f"{k[0]}{int(v * 100):02d}" for k, v in mlm_masking_rules.items()])
    finetune_run_id = pretrain_run_id if not cli_args.reset_weights else "no_pretrain"
    result_dir = Path(cfg["result_dir"]) / cfg["data_split_type"]
    # Pretraining checkpoint can live in another result tree (e.g. the one built
    # by the classification pipeline). Default: this config's own result_dir.
    pretrain_root = Path(cfg.get("pretrain_result_dir", cfg["result_dir"])) / cfg["data_split_type"]
    pretrained_dir = pretrain_root / pretrain_run_id / "pretraining"
    finetuning_subdir = cfg["finetuner"].pop("ft_subdir", "finetuning_survival")
    finetuning_dir = result_dir / finetune_run_id / finetuning_subdir

    # Iterate over prediction tasks (one endpoint outcome each)
    for task_key, task_specs in cfg["prediction_tasks"].items():

        # Reporting horizons (int or list of ints)
        horizons = task_specs.get("horizons", [30, 60, 90])
        horizons = horizons if isinstance(horizons, list) else [horizons]
        valid_fups = task_specs["fups"]

        # Define run configurations: list of tuples (train_fups_list, valid_fups_list)
        if train_data_augment == "none":  # one run per follow-up period
            run_configs = [([f], [f]) for f in valid_fups]
        else:  # single run with all (valid) follow-up periods
            if train_data_augment == "valid":
                train_fups = valid_fups  # all of interest
            elif train_data_augment == "all":
                data_dir = Path(cfg["data_dir"]) / cfg["data_split_type"]
                train_fups = scan_all_fups(data_dir)  # all available
            run_configs = [(train_fups, valid_fups)]

        for train_fups, valid_fups in run_configs:
            print(
                f"Starting survival fine-tuning: Task={task_key} | Horizons={horizons} | "
                f"Filtered={train_data_augment} | Train FUPs={train_fups} | Valid FUPs={valid_fups}"
            )
            finetune_survival_model(
                cfg=cfg,
                cli_args=cli_args,
                task_key=task_key,
                horizons=horizons,
                fup_train=train_fups,
                fup_valid=valid_fups,
                train_data_augment=train_data_augment,
                run_id=finetune_run_id,
                pretrained_dir=pretrained_dir,
                finetuning_dir=finetuning_dir,
            )


def finetune_survival_model(
    cfg: dict,
    cli_args: argparse.Namespace,
    task_key: str,
    horizons: list[int],
    fup_valid: list[int],
    fup_train: list[int],
    train_data_augment: str,
    run_id: str = "default_run",
    pretrained_dir: Path = None,
    finetuning_dir: Path = None,
):
    """
    Fine-tune one survival model (Cox PH) on a specific endpoint outcome.
    """
    data_root_dir = Path(cfg["data_dir"]) / cfg["data_split_type"]
    all_possible_fups = scan_all_fups(data_root_dir)

    # Resolve the survival label columns (derives category endpoints later)
    time_key, status_key, is_derived = resolve_survival_label_keys(task_key)
    label_keys = [time_key, status_key]

    # Load raw dataset and metadata (survival columns travel as any other label_*)
    undersampling_balancing = cfg.get("undersampling_balancing_label", None)
    if undersampling_balancing == "status":
        undersampling_balancing = status_key
    dataset, _, vocab = load_hf_data_and_metadata(
        data_dir=data_root_dir,
        fup_train=fup_train,
        fup_valid=all_possible_fups,
        fup_test=all_possible_fups,
        label_keys=label_keys,
        target_undersampling_ratio=cfg.get("target_undersampling_ratio", None),
        undersampling_balancing_label=undersampling_balancing,
        drop_unlabeled=cfg.get("drop_unlabeled", False),
        time_mapping=cfg["data_collator"]["time_mapping"],
        eav_mappings=cfg["data_collator"]["eav_mappings"],
    )

    # Derive category-level survival labels (bacteria/virus/fungi) if necessary
    for split in dataset.keys():
        if is_derived and status_key not in dataset[split].column_names:
            dataset[split] = derive_category_survival_labels(dataset[split], task_key)

    # Prepare split identification flags across arrays
    for split in dataset.keys():
        col_vals = [split] * len(dataset[split])
        dataset[split] = dataset[split].add_column("split", col_vals)

    # Filter out rows with unknown or zero-information survival labels
    for split in dataset.keys():
        times = np.asarray(dataset[split][time_key])
        statuses = np.asarray(dataset[split][status_key])
        keep_mask = survival_keep_mask(times, statuses)
        dataset[split] = dataset[split].select(np.where(keep_mask)[0])
        print(
            f"[{split}] survival rows kept: {keep_mask.sum()}/{len(keep_mask)} "
            f"({int((statuses[keep_mask] == 1).sum())} events)"
        )

    # Prepare datasets used for training and runtime evaluation.
    # During training we only score the pooled validation set ("all"): the
    # early-stopping / best-checkpoint metric is eval_all_early_stopping_metric,
    # so per-fup segments would only add cost and wandb noise here. The
    # per-fup breakdown is computed once, post-training, over ALL fups for
    # both validation and test (see test_survival_model).
    train_dataset = dataset["train"]
    eval_datasets = {"all": dataset["validation"]}

    # Auto-detect model sub-directory and best pre-trained model checkpoint
    pretrained_last_ckpt_dir = get_last_checkpoint(str(pretrained_dir))
    if pretrained_last_ckpt_dir is None:
        if (pretrained_dir / "config.json").exists():
            pretrained_last_ckpt_dir = str(pretrained_dir)
        else:
            sys.exit(f"Error: No checkpoint found in {pretrained_dir}")

    # Set up model configuration: single logit head = log hazard ratio
    model_cfg = cfg["model"].copy()
    model_cfg["pretrained_dir"] = pretrained_last_ckpt_dir
    model_cfg["embedding_layer_config"]["vocab_size"] = len(vocab)
    model_cfg["reset_weights"] = cli_args.reset_weights
    model_cfg["enforce_monotonicity"] = False
    model_cfg["task"] = "survival"
    model_cfg["model_args"]["num_labels"] = 1
    model_cfg["model_args"]["problem_type"] = "regression"

    # Initialize model
    model = PatientEmbeddingModelFactory.from_pretrained(**model_cfg)
    max_pos_embeddings = model.config.max_position_embeddings

    # Inject LoRA, if required
    if cfg.get("use_lora", False):
        peft_conf = cfg.get("lora_config", {})
        peft_config = LoraConfig(**peft_conf)
        model = get_peft_model(model, peft_config)
        print(">>> LoRA Enabled. Trainable parameters:")
        model.print_trainable_parameters()

    # Custom data collator: labels = [time, status] per patient
    collator_cfg = cfg["data_collator"].copy()
    collator_cfg.pop("mlm_label_keys", None)
    collator_cfg.pop("mlm_masking_rules", None)
    ft_collator = PatientDataCollatorForSurvival(
        **collator_cfg,
        time_label=time_key,
        status_label=status_key,
        max_position_embeddings=max_pos_embeddings,
    )

    # Evaluation pipeline (C-index, horizon-wise AUROC/AP/IPCW-Brier)
    ft_cfg = cfg["finetuner"].copy()
    patience = ft_cfg.pop("early_stopping_patience", 10)
    callbacks = [EarlyStoppingCallback(early_stopping_patience=patience)]
    evaluator = CustomEvaluator(
        horizons=horizons,
        early_stopping_metric=cfg["early_stopping_metric"],
    )

    # Survival loss (Cox NLL partial likelihood)
    loss_name = ft_cfg.pop("loss_name", "cox")
    loss_args = ft_cfg.pop("loss_args", {}) or {}  # kwargs for CoxNLLLoss, see config
    loss_func = make_survival_loss_func(loss_name, loss_args)

    # Training arguments, with the correct output directory
    fmt_fn = lambda x: "-".join(f"{i:04d}" for i in sorted(([x] if isinstance(x, int) else x or [])))
    fut_str = fmt_fn(fup_train) if train_data_augment == "none" else train_data_augment
    fuv_str = fmt_fn(fup_valid)
    hrz_str = fmt_fn(horizons)
    task_subdir = f"hrz({hrz_str})_fut({fut_str})_fuv({fuv_str})"
    run_subdir = str(Path(task_key) / task_subdir)
    ft_cfg["output_dir"] = str(finetuning_dir / run_subdir)
    if cli_args.silent:
        ft_cfg["report_to"] = "none"
    if not torch.cuda.is_available():
        ft_cfg["bf16"] = False
        ft_cfg["fp16"] = False
        ft_cfg["use_cpu"] = True
    ft_args = TrainingArguments(**ft_cfg)

    # Re-initialize a wandb run within the same workspace
    use_wandb = (not cli_args.silent) and (cfg.get("finetuner", {}).get("report_to") == "wandb")
    if use_wandb:
        workspace = Path(__file__).stem
        run_name = f"{run_id}_{run_subdir}"
        wandb.init(project=workspace, name=run_name, config=cfg)

    # Trainer (standard HuggingFace, prefix-aware for multi-FUP evaluation)
    trainer = PrefixAwareTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_datasets,
        args=ft_args,
        data_collator=ft_collator,
        compute_loss_func=loss_func,
        compute_metrics=evaluator,
        callbacks=callbacks,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )
    trainer.remove_callback(PrinterCallback)

    # Fine-tune the model
    trainer.train()

    # Post-hoc evaluation over ALL un-truncated elements
    test_survival_model(
        trainer=trainer,
        dataset_dict=dataset,  # Contains the full, filtered data splits
        all_possible_fups=all_possible_fups,
        output_dir=Path(ft_cfg["output_dir"]),
    )

    # Reset wandb and clean up CUDA memory for the next run
    if use_wandb:
        wandb.finish()
    del dataset, model, trainer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def test_survival_model(
    trainer: Trainer,
    dataset_dict: dict,
    all_possible_fups: list[int],
    output_dir: Path,
):
    """
    Post-training evaluation across ALL discovered FUP time-steps, on both the
    validation and test splits. Mirrors train_classification.test_model (same
    JSON/NPZ structure) and additionally saves the `patientid` of every scored
    row, so downstream analyses (e.g. comparison with classic ML / binary
    transformers) can align predictions with the survival ground truth.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.remove_callback(EarlyStoppingCallback)

    split_configs = []
    if "validation" in dataset_dict:
        split_configs.append({
            "split_label": "validation",
            "datasets": prepare_dataset_fup_dict(dataset_dict["validation"], all_possible_fups),
            "json_name": "validation_results.json",
        })
    if "test" in dataset_dict:
        split_configs.append({
            "split_label": "test",
            "datasets": prepare_dataset_fup_dict(dataset_dict["test"], all_possible_fups),
            "json_name": "test_results.json",
        })

    for config in split_configs:
        print(f"\n>>> Running final unified survival scoring pass over all FUPs for: [{config['split_label'].upper()}]")
        # Absolute test probabilities must not depend on test outcomes: reuse
        # the per-FUP Breslow baselines fitted on the validation split above.
        trainer.compute_metrics.freeze_baseline_for_test = config["split_label"] == "test"
        final_metrics = {}
        all_predictions = {}

        # Process the aggregated global view first (metrics only)
        mega_ds = config["datasets"].pop("all")
        results_mega = trainer.evaluate(mega_ds, metric_key_prefix=f"{config['split_label']}_all")
        final_metrics.update(results_mega)
        trainer.compute_metrics.saved_labels_and_probs = None

        # Process individual FUP segments sequentially
        for fup_key, fup_dataset in config["datasets"].items():
            prefix = f"{config['split_label']}_{fup_key}"
            results_fup = trainer.evaluate(fup_dataset, metric_key_prefix=prefix)
            final_metrics.update(results_fup)

            for key, array in trainer.compute_metrics.saved_labels_and_probs.items():
                all_predictions[f"{prefix}_{key}"] = array

            # Save row identities to align predictions with survival ground truth
            patient_ids = np.asarray(fup_dataset["patientid"])
            n_scored = len(trainer.compute_metrics.saved_labels_and_probs["labels"])
            if len(patient_ids) == n_scored:
                all_predictions[f"{prefix}_patientid"] = patient_ids
            else:
                print(f"    [WARN] {prefix}: {n_scored} scored rows vs {len(patient_ids)} samples; skipping ids")

            trainer.compute_metrics.saved_labels_and_probs = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Write metrics to disk (predictions go to the tidy CSV below)
        with open(output_dir / config["json_name"], "w") as f:
            json.dump(final_metrics, f, indent=4)
        print(f"    [SUCCESS] Saved {config['split_label']} stats under: {output_dir}")

        # Tidy long-format table: patient_id, fup, horizon, label, prob,
        # time, status, risk, pred_median_time
        # (label = event within horizon binarized from (time, status);
        #  prob = P(event within horizon) = 1 - S(horizon | patient);
        #  rows censored before the horizon get an unknown label and are skipped)
        save_predictions_table(
            all_predictions=all_predictions,
            split_label=config["split_label"],
            horizons=extract_horizons_from_path(output_dir),
            output_dir=output_dir,
            csv_name=f"{config['split_label']}_predictions.csv",
        )

        print(f"    [SUCCESS] Saved {config['split_label']} arrays and stats under: {output_dir}")


if __name__ == "__main__":
    main()
