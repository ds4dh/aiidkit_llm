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
from datasets import Dataset
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
from transformers.trainer_utils import get_last_checkpoint
from transformers.trainer_callback import PrinterCallback
from peft import LoraConfig, get_peft_model
from src.utils import apply_config_overrides
from src.data.patient_dataset import load_hf_data_and_metadata
from src.model.model_utils import make_loss_func, compute_loss_args, GradientClippingGuard
from src.model.patient_embedder import PatientEmbeddingModelFactory
from src.model.patient_embedder import PatientDataCollatorForClassification
from src.evaluation.plot_results import plot_task_results
from src.evaluation.evaluate_models import (
    DiscriminativeEmbeddingEvaluatorForClassification as CustomEvaluator,
)
from scripts.script_utils import (
    scan_all_fups, prepare_dataset_fup_dict,
    save_predictions_table, extract_horizons_from_path,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune a model to predict future infections.")
    parser.add_argument("--config", "-c", type=str, default="configs/discriminative_training.yaml")
    parser.add_argument("--reset_weights", "-r", action="store_true", help="Whether to reset model weights before fine-tuning.")
    parser.add_argument("--plot_only", "-p", action="store_true", help="Skip run and goes directly to the plot.")
    parser.add_argument("--silent", "-s", action="store_true", help="Disable wandb logging.")
    parser.add_argument("--overrides", "-o", action="append", default=[], help="Overrides config (JSON string or key=value).")
    parser.add_argument("--tasks", "-t", type=str, default="",
                        help="Comma-separated prediction_tasks keys to run (default: all active entries in the config). "
                             "Lets you run a subset (e.g. a single group task or the multi-task entry) without editing the config.")
    return parser.parse_args()


def main():
    """
    Fine-tune models for the prediction tasks in the yaml file from the CLI config
    """
    cli_args = parse_args()
    with open(cli_args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    cfg = apply_config_overrides(cfg, cli_args.overrides)

    # Optional run filter: run only a subset of the prediction_tasks entries
    # defined in the config (e.g. --tasks infection_grp_cmv or
    # --tasks multitask_bacteria,multitask_infections).
    if cli_args.tasks:
        wanted = [t.strip() for t in cli_args.tasks.split(",") if t.strip()]
        missing = [t for t in wanted if t not in cfg["prediction_tasks"]]
        if missing:
            sys.exit(f"Error: --tasks entries {missing} not found in config "
                     f"prediction_tasks {list(cfg['prediction_tasks'])}. "
                     f"(Entries must be active/uncommented in the YAML to be selectable.)")
        cfg["prediction_tasks"] = {t: cfg["prediction_tasks"][t] for t in wanted}
        print(f">>> --tasks filter active: running {list(cfg['prediction_tasks'])}")

    # Identify pretraining and finetuning directories
    train_data_augment = cfg.get("train_data_augment", "all")
    mlm_masking_rules = cfg["data_collator"]["mlm_masking_rules"]
    pretrain_run_id = "-".join([f"{k[0]}{int(v * 100):02d}" for k, v in mlm_masking_rules.items()])
    finetune_run_id = pretrain_run_id if not cli_args.reset_weights else "no_pretrain"
    result_dir = Path(cfg["result_dir"]) / cfg["data_split_type"]
    pretrained_dir = result_dir / pretrain_run_id / "pretraining"
    finetuning_subdir = cfg["finetuner"].pop("ft_subdir", "finetuning")
    finetuning_dir = result_dir / finetune_run_id / finetuning_subdir

    # Iterate over prediction tasks
    enforce_monotonicity = cfg["finetuner"].pop("enforce_monotonicity", True)
    for task_key, task_specs in cfg["prediction_tasks"].items():
        # A task entry may bundle several outcome tasks sharing the same horizon
        # grid into ONE multi-head run via a 'tasks' list. Without it, behavior
        # is unchanged: one single-task fine-tuning run per horizon group.
        task_keys = list(task_specs.get("tasks", [task_key]))
        if not cli_args.plot_only:
            
            # Iterate over all horizon configurations
            for horizons in task_specs["horizons"]:
                
                # Define run configurations: list of tuples (train_fups_list, valid_fups_list)
                valid_fups = task_specs["fups"]
                if train_data_augment == "none":  # one run per follow-up period
                    run_configs = [([f], [f]) for f in valid_fups]
                else:  # single run with all (valid) follow-up periods
                    if train_data_augment == "valid":
                        train_fups = valid_fups  # all of interest
                    elif train_data_augment == "all":
                        data_dir = Path(cfg["data_dir"]) / cfg["data_split_type"]
                        train_fups = scan_all_fups(data_dir)  # all available
                    run_configs = [(train_fups, valid_fups)]

                # Ensure horizons is a sorted list (even if only a single entry)
                horizons = sorted(horizons if isinstance(horizons, list) else [horizons])

                # Execute the runs
                for train_fups, valid_fups in run_configs:
                    print(
                        f"Starting fine-tuning: Tasks={task_key} | Horizons={horizons} | "
                        f"Filtered={train_data_augment} | "
                        f"Train FUPs={train_fups} | Valid FUPs={valid_fups}"
                    )
                    finetune_disciminative_model(
                        cfg=cfg,
                        cli_args=cli_args,
                        task_key=task_key,
                        task_keys=task_keys,
                        horizons=horizons,  
                        fup_train=train_fups,
                        fup_valid=valid_fups,
                        train_data_augment=train_data_augment,
                        enforce_monotonicity=enforce_monotonicity,
                        run_id=finetune_run_id,
                        pretrained_dir=pretrained_dir,
                        finetuning_dir=finetuning_dir,
                    )
        
        # After all runs for this task, generate plots (single-task entries only:
        # multi-task runs store per-(task, horizon) metrics in their JSON/CSVs)
        if len(task_keys) > 1:
            print(f"[plot] Skipping summary plots for multi-task entry '{task_key}'.")
            continue
        plot_task_results(
            task_key=task_key,
            task_specs=task_specs,
            finetuning_dir=finetuning_dir,
            train_data_augment=train_data_augment,
        )


def finetune_disciminative_model(
    cfg: dict,
    cli_args: argparse.Namespace,
    task_key: str,
    horizons: list[int],
    fup_valid: list[int],
    fup_train: list[int],
    train_data_augment: str,
    enforce_monotonicity: bool,
    task_keys: list[str] = None,
    run_id: str = "default_run",
    pretrained_dir: Path = None,
    finetuning_dir: Path = None,
):
    """
    Fine-tune one model on one or more prediction tasks (multi-head).

    Multi-task runs (len(task_keys) > 1) concatenate the per-task horizon label
    columns into one multi-label target, task-major ordered:
    [t1_h1..t1_hK, t2_h1..t2_hK, ...]. Each row trains all heads whose label
    is known (-100 entries are masked out by the loss and the metrics).
    """
    if task_keys is None:
        task_keys = [task_key]
    data_root_dir = Path(cfg["data_dir"]) / cfg["data_split_type"]
    all_possible_fups = scan_all_fups(data_root_dir)
    label_keys = [f"label_{t}_{h:04d}d" for t in task_keys for h in horizons]

    # Load raw dataset and metadata
    dataset, _, vocab = load_hf_data_and_metadata(
        data_dir=data_root_dir,
        fup_train=fup_train,
        fup_valid=all_possible_fups,
        fup_test=all_possible_fups,
        label_keys=label_keys,
        target_undersampling_ratio=cfg.get("target_undersampling_ratio", None),
        drop_unlabeled=cfg.get("drop_unlabeled", False),
        time_mapping=cfg["data_collator"]["time_mapping"],
        eav_mappings=cfg["data_collator"]["eav_mappings"],
    )

    # Prepare split identification flags across arrays
    for split in dataset.keys():
        col_vals = [split] * len(dataset[split])
        dataset[split] = dataset[split].add_column("split", col_vals)
    for split in dataset.keys():
        label_matrix = np.stack([dataset[split][k] for k in label_keys], axis=1)
        keep_mask = (label_matrix != -100).any(axis=1)
        dataset[split] = dataset[split].select(np.where(keep_mask)[0])
        
    # Prepare datasets used for training and runtime evaluation.
    # During training we only score the pooled validation set ("all"): the
    # early-stopping / best-checkpoint metric is eval_all_early_stopping_metric,
    # so per-fup segments would only add cost and wandb noise here. The
    # per-fup breakdown is computed once, post-training, over ALL fups for
    # both validation and test (see test_model).
    train_dataset = dataset["train"]
    eval_datasets = {"all": dataset["validation"]}

    # Auto-detect model sub-directory and best pre-trained model checkpoint
    pretrained_last_ckpt_dir = get_last_checkpoint(str(pretrained_dir))
    if pretrained_last_ckpt_dir is None:
        if (pretrained_dir / "config.json").exists():
            pretrained_last_ckpt_dir = str(pretrained_dir)
        else:
            sys.exit(f"Error: No checkpoint found in {pretrained_dir}")

    # Monotonicity: predicted probabilities must increase with the horizon. In
    # multi-task runs this ordering only holds WITHIN each task's horizon block,
    # so model and evaluator receive explicit group sizes (task-major column
    # order); with a single horizon per task there is nothing to order -> disable.
    monotonic_groups = None
    if enforce_monotonicity and len(task_keys) > 1:
        if len(horizons) > 1:
            monotonic_groups = [len(horizons)] * len(task_keys)
        else:
            enforce_monotonicity = False

    # Set up model configuration
    model_cfg = cfg["model"].copy()
    model_cfg["pretrained_dir"] = pretrained_last_ckpt_dir
    model_cfg["embedding_layer_config"]["vocab_size"] = len(vocab)
    model_cfg["reset_weights"] = cli_args.reset_weights
    model_cfg["enforce_monotonicity"] = monotonic_groups if monotonic_groups else enforce_monotonicity
    model_cfg["task"] = "classification"
    model_cfg["model_args"]["num_labels"] = len(label_keys)
    model_cfg["model_args"]["problem_type"] = "multi_label_classification"

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

    # Custom data collator dedicated to patient classification
    collator_cfg = cfg["data_collator"].copy()
    collator_cfg["label_keys"] = label_keys
    ft_collator = PatientDataCollatorForClassification(
        **collator_cfg,
        max_position_embeddings=max_pos_embeddings,
    )

    # Evaluation pipelines
    patience = cfg["finetuner"].pop("early_stopping_patience", 20)
    callbacks = [EarlyStoppingCallback(early_stopping_patience=patience)]
    evaluator = CustomEvaluator(
        do_clustering=False,
        label_names=label_keys,
        enforce_monotonicity=enforce_monotonicity,
        monotonic_groups=monotonic_groups,
        early_stopping_metric=cfg["early_stopping_metric"],
        early_stopping_subset=cfg.get("early_stopping_subset"),
        early_stopping_mean=cfg.get("early_stopping_mean", "harmonic"),
    )

    # Setup loss function
    ft_cfg = cfg["finetuner"].copy()
    loss_name = ft_cfg.pop("loss_name", "poly1")
    loss_args = compute_loss_args(train_dataset, label_keys)
    loss_args.update(ft_cfg.pop("loss_args", {}) or {})  # optional manual override
    loss_func = make_loss_func(loss_name, loss_args)  

    # Training arguments, with the correct output directory
    fmt_fn = lambda x: "-".join(f"{i:04d}" for i in sorted(([x] if isinstance(x, int) else x or [])))    
    fut_str = fmt_fn(fup_train) if train_data_augment == "none" else train_data_augment    
    fuv_str = fmt_fn(fup_valid)  
    hrz_str = fmt_fn(horizons)  
    task_subdir = f"hrz({hrz_str})_fut({fut_str})_fuv({fuv_str})"
    run_subdir = str(Path(task_key) / task_subdir)
    ft_cfg["output_dir"] = str(finetuning_dir / run_subdir)
    # Record the run layout so the test pass can map matrix columns back to
    # (task, horizon) pairs (essential for multi-task prediction CSVs).
    Path(ft_cfg["output_dir"]).mkdir(parents=True, exist_ok=True)
    with open(Path(ft_cfg["output_dir"]) / "run_metadata.json", "w") as f:
        json.dump(
            {"task_key": task_key, "tasks": list(task_keys),
             "horizons": list(horizons), "label_keys": label_keys},
            f, indent=2,
        )
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

    # Trainer (standard HuggingFace)
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
    test_model(
        trainer=trainer,
        dataset_dict=dataset, # Contains the full, un-truncated data splits
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


class PrefixAwareTrainer(GradientClippingGuard, Trainer):
    """
    Trainer that keeps per-FUP evaluation prefixes in sync and adds gradient
    clipping diagnostics + a non-finite gradient guard (see GradientClippingGuard).
    """
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        if hasattr(self.compute_metrics, "current_prefix"):
            self.compute_metrics.current_prefix = metric_key_prefix
            
        return super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)


def test_model(
    trainer: Trainer,
    dataset_dict: dict,  
    all_possible_fups: list[int],
    output_dir: Path,       
):
    """
    Aggregates post-training evaluation across ALL discovered FUP time-steps
    on both the validation and test splits, saving symmetrical JSON and NPZ structures.
    """    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    val_fup_datasets = prepare_dataset_fup_dict(dataset_dict["validation"], all_possible_fups)
    split_configs = []
    
    if "validation" in dataset_dict:
        split_configs.append({
            "split_label": "validation", 
            "datasets": val_fup_datasets, 
            "json_name": "validation_results.json",
        })
    if "test" in dataset_dict:
        test_fup_datasets = prepare_dataset_fup_dict(dataset_dict["test"], all_possible_fups)
        split_configs.append({
            "split_label": "test",       
            "datasets": test_fup_datasets, 
            "json_name": "test_results.json",
        })
    
    for config in split_configs:
        print(f"\n>>> Running final unified scoring pass over all FUPs for: [{config['split_label'].upper()}]")        
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
                
            # Save row identities so predictions can be traced back to patients
            if "patientid" in fup_dataset.column_names:
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

        # Tidy long-format table: patient_id, fup, [task,] horizon, label, prob,
        # prob_cal (the 'task' column appears for multi-task runs, where one
        # matrix column per (task, horizon) pair is scored jointly)
        task_keys = None
        meta_path = output_dir / "run_metadata.json"
        if meta_path.exists():
            task_keys = json.loads(meta_path.read_text()).get("tasks")
        save_predictions_table(
            all_predictions=all_predictions,
            split_label=config["split_label"],
            horizons=extract_horizons_from_path(output_dir),
            output_dir=output_dir,
            csv_name=f"{config['split_label']}_predictions.csv",
            task_keys=task_keys,
        )
            
        print(f"    [SUCCESS] Saved {config['split_label']} arrays and stats under: {output_dir}")


def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        return logits[0]  
    return logits


if __name__ == "__main__":
    main()
