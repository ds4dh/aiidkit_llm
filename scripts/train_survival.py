import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import gc
import json
import shutil
import numpy as np
import torch
import wandb
import yaml
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback, EvalPrediction
from transformers.trainer_callback import PrinterCallback
from peft import LoraConfig, get_peft_model
from src.utils import apply_config_overrides, load_optuna_json_overrides
from src.data.dynamic_store import make_dynamic_epoch_callback
from src.data.patient_dataset import load_hf_data_and_metadata
from src.data.survival_labels import (
    resolve_survival_label_keys, derive_category_survival_labels, survival_keep_mask,
)
from src.model.model_utils import resolve_best_checkpoint
from src.model.patient_embedder import PatientEmbeddingModelFactory
from src.model.patient_embedder import PatientDataCollatorForSurvival
from src.model.survival_utils import (
    build_survival_time_grid,
    compute_empirical_log_hazard_priors,
    make_survival_loss_func,
)
from src.evaluation.evaluate_models import (
    DiscriminativeEmbeddingEvaluatorForSurvival as CustomEvaluator,
)
from scripts.train_classification import (
    PrefixAwareTrainer, preprocess_logits_for_metrics,
)
from scripts.script_utils import (
    scan_all_fups,
    save_predictions_table, extract_horizons_from_path,
    format_pretrain_run_id, get_value_embedding_tag,
    resolve_run_id_with_val_tag,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune a patient transformer with a single-task or multi-task survival (Cox) objective."
    )
    parser.add_argument("--config", "-c", type=str, default="configs/survival_training.yaml")
    parser.add_argument("--reset_weights", "-r", action="store_true", help="Whether to reset model weights before fine-tuning.")
    parser.add_argument("--eval-only", "--eval_only", dest="eval_only", action="store_true",
                        help="Skip training, load existing fine-tuned checkpoint from output_dir, and run full test_survival_model.")
    parser.add_argument("--silent", "-s", action="store_true", help="Disable wandb logging.")
    parser.add_argument("--optuna-json", "-J", action="append", default=[],
                        help="Path(s) to best_{pretrain,finetune}_config.json written by train_optuna.py.")
    parser.add_argument("--overrides", "-o", action="append", default=[], help="Overrides config (JSON string or key=value).")
    parser.add_argument("--tasks", "-t", nargs="*", default=[], help="Comma-separated prediction_tasks keys to run.")
    return parser.parse_args()


def main():
    """
    Fine-tune single-task or multi-task Cox PH survival models for `prediction_tasks`.
    """
    cli_args = parse_args()
    with open(cli_args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    base_loss_name = cfg.get("finetuner", {}).get("loss_name", "surv_ehr")
    cfg = load_optuna_json_overrides(cfg, cli_args.optuna_json)
    if cli_args.optuna_json:
        # Preserve survival-specific loss if a binary Optuna JSON was supplied
        if str(cfg.get("finetuner", {}).get("loss_name", "")).lower() not in ("surv_ehr", "pc_hazard", "neural_cif", "cox"):
            cfg["finetuner"]["loss_name"] = base_loss_name
        if base_loss_name.lower() == "cox":
            cfg["target_undersampling_ratio"] = None
    cfg = apply_config_overrides(cfg, cli_args.overrides)

    if cli_args.tasks:
        wanted = [
            t.strip()
            for arg in (cli_args.tasks if isinstance(cli_args.tasks, list) else [cli_args.tasks])
            for t in arg.split(",")
            if t.strip()
        ]
        missing = [t for t in wanted if t not in cfg["prediction_tasks"]]
        if missing:
            sys.exit(f"Error: --tasks entries {missing} not found in survival config {list(cfg['prediction_tasks'])}.")
        cfg["prediction_tasks"] = {t: cfg["prediction_tasks"][t] for t in wanted}
        print(f">>> --tasks filter active: running {list(cfg['prediction_tasks'])}")

    train_data_augment = cfg.get("train_data_augment", "all")
    pretrain_run_id = format_pretrain_run_id(cfg)
    val_tag = get_value_embedding_tag(cfg)
    if cli_args.reset_weights:
        finetune_run_id = f"no_pretrain_{val_tag}"
    elif cfg.get("finetune_run_id"):
        finetune_run_id = resolve_run_id_with_val_tag(str(cfg["finetune_run_id"]), val_tag)
    else:
        finetune_run_id = pretrain_run_id
    result_dir = Path(cfg["result_dir"]) / cfg["data_split_type"]
    pretrain_root = Path(cfg.get("pretrain_result_dir", cfg["result_dir"])) / cfg["data_split_type"]
    pretrained_dir = pretrain_root / pretrain_run_id / "pretraining"
    if pretrained_dir.exists():
        pt_alias = result_dir / finetune_run_id / "pretraining"
        pt_alias.parent.mkdir(parents=True, exist_ok=True)
        if not pt_alias.exists() and not pt_alias.is_symlink():
            pt_alias.symlink_to(pretrained_dir.resolve(), target_is_directory=True)
    finetuning_subdir = cfg["finetuner"].pop("ft_subdir", "finetuning")
    finetuning_dir = result_dir / finetune_run_id / finetuning_subdir

    for task_key, task_specs in cfg["prediction_tasks"].items():
        task_keys = list(task_specs.get("tasks", [task_key]))
        raw_horizons = task_specs.get("horizons", [30, 60, 90])
        if isinstance(raw_horizons, list) and len(raw_horizons) == 1 and isinstance(raw_horizons[0], list):
            horizons = [int(h) for h in raw_horizons[0]]
        elif isinstance(raw_horizons, list):
            horizons = [int(h) for h in raw_horizons]
        else:
            horizons = [int(raw_horizons)]
        valid_fups = task_specs["fups"]

        if train_data_augment == "none":
            run_configs = [([f], [f]) for f in valid_fups]
        else:
            if train_data_augment == "valid":
                train_fups = valid_fups
            elif train_data_augment in ("all", "random"):
                data_dir = Path(cfg["data_dir"]) / cfg["data_split_type"]
                train_fups = scan_all_fups(data_dir)
            else:
                raise ValueError(f"Unsupported train_data_augment='{train_data_augment}' (expected 'all', 'random', 'valid', or 'none').")
            run_configs = [(train_fups, valid_fups)]

        for train_fups, valid_fups in run_configs:
            print(
                f"Starting survival fine-tuning: Group={task_key} | Tasks={task_keys} | Horizons={horizons} | "
                f"Filtered={train_data_augment} | Train FUPs={len(train_fups)} | Valid FUPs={len(valid_fups)}"
            )
            finetune_survival_model(
                cfg=cfg,
                cli_args=cli_args,
                task_key=task_key,
                task_keys=task_keys,
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
    task_keys: list[str],
    horizons: list[int],
    fup_valid: list[int],
    fup_train: list[int],
    train_data_augment: str,
    run_id: str = "default_run",
    pretrained_dir: Path = None,
    finetuning_dir: Path = None,
):
    """
    Fine-tune a single-task (K=1) or multi-task (K>=2) Cox PH survival Transformer.
    """
    data_root_dir = Path(cfg["data_dir"]) / cfg["data_split_type"]
    all_possible_fups = scan_all_fups(data_root_dir)

    resolved_specs = [resolve_survival_label_keys(tk) for tk in task_keys]
    survival_pairs = [(t_col, s_col) for t_col, s_col, _ in resolved_specs]
    label_keys = [col for t_col, s_col in survival_pairs for col in (t_col, s_col)]

    ft_cfg = cfg["finetuner"].copy()
    loss_name = str(ft_cfg.pop("loss_name", "surv_ehr")).lower()
    loss_args = dict(ft_cfg.pop("loss_args", {}) or {})
    is_pc_hazard = loss_name in ("surv_ehr", "pc_hazard", "neural_cif")

    # Load FULL unfiltered survival splits first (`target_undersampling_ratio=None`) so we:
    #   1) compute exact population empirical log-hazard priors on the full training split, and
    #   2) apply exact Horvitz-Thompson inverse-probability per-FUP undersampling wrt `max(horizons)`.
    max_seq_len = int(cfg.get("model", {}).get("config_args", {}).get("max_position_embeddings", 1024)) - 2
    dataset, _, vocab = load_hf_data_and_metadata(
        data_dir=data_root_dir,
        fup_train=fup_train,
        fup_valid=all_possible_fups,
        fup_test=all_possible_fups,
        label_keys=label_keys,
        target_undersampling_ratio=None,
        undersampling_balancing_label=None,
        drop_unlabeled=cfg.get("drop_unlabeled", False),
        time_mapping=cfg["data_collator"]["time_mapping"],
        eav_mappings=cfg["data_collator"]["eav_mappings"],
        landmark_mode=("random_landmark" if train_data_augment == "random" else "fixed_grid"),
        random_landmark_stride_days=int(cfg.get("random_landmark_stride_days", 30)),
        random_landmark_max_fup=int(cfg.get("random_landmark_max_fup", 3600)),
        random_landmark_seed=int(ft_cfg.get("seed", 1234)),
        max_length=max_seq_len,
        use_global_salience_truncation=bool(cfg.get("use_global_salience_truncation", True)),
    )

    for split in dataset.keys():
        for tk, (_, s_col, is_derived) in zip(task_keys, resolved_specs):
            if is_derived and s_col not in dataset[split].column_names:
                dataset[split] = derive_category_survival_labels(dataset[split], tk)

    # Keep rows where at least one task carries a valid, informative survival label
    for split in dataset.keys():
        n_before = len(dataset[split])
        any_keep = np.zeros(n_before, dtype=bool)
        ev_summary = []
        for tk, (t_col, s_col, _) in zip(task_keys, resolved_specs):
            t_arr = np.asarray(dataset[split].data.column(t_col))
            s_arr = np.asarray(dataset[split].data.column(s_col))
            km = survival_keep_mask(t_arr, s_arr)
            any_keep |= km
            ev_summary.append(f"{tk}:{int((s_arr[km] == 1).sum())}")
        if not any_keep.all():
            dataset[split] = dataset[split].select(np.where(any_keep)[0], keep_in_memory=True).flatten_indices(keep_in_memory=True)
        print(
            f"[{split}] survival rows kept: {int(any_keep.sum())}/{n_before} "
            f"(events -> {', '.join(ev_summary)})",
            flush=True,
        )

    time_grid = None
    empirical_log_hazards = None
    if is_pc_hazard:
        max_time_cfg = loss_args.pop("max_time", None)
        time_grid = build_survival_time_grid(horizons=horizons, max_time=max_time_cfg)
        tr_ds_full = dataset["train"]
        tr_times_full = np.stack([np.asarray(tr_ds_full.data.column(t_c)) for t_c, _ in survival_pairs], axis=1)
        tr_stats_full = np.stack([np.asarray(tr_ds_full.data.column(s_c)) for _, s_c in survival_pairs], axis=1)
        tr_fups_full = np.asarray(tr_ds_full.data.column("fup"))
        empirical_log_hazards = compute_empirical_log_hazard_priors(
            times_2d=tr_times_full,
            statuses_2d=tr_stats_full,
            fups_1d=tr_fups_full,
            time_grid=time_grid,
        )
        print(
            f">>> [SurvivEHR] Built {len(time_grid)-1}-interval time grid {time_grid} "
            f"and empirical ({len(task_keys)}x5x{len(time_grid)-1}) log-hazard priors from {len(tr_ds_full)} train rows.",
            flush=True,
        )

        # Horvitz-Thompson per-FUP importance-weighted undersampling on train split:
        # keeps 100% of rows with ANY observed event within `max(horizons)` (`w = 1.0`),
        # and subsamples remaining rows at `target_undersampling_ratio` with `w = N_rest / n_kept`.
        us_ratio = cfg.get("target_undersampling_ratio", None)
        if us_ratio is not None and float(us_ratio) > 0.0:
            max_h_case = float(max(horizons))
            is_acute_case = ((tr_stats_full == 1) & (tr_times_full >= 0) & (tr_times_full <= max_h_case)).any(axis=1)
            rng = np.random.default_rng(int(ft_cfg.get("seed", 1234)))
            # Group by 30-day bins (exact match on 30d grid, smooth on continuous random landmarks)
            fup_bins = tr_fups_full // 30
            order = np.argsort(fup_bins, kind="mergesort")
            uniq_fups, split_pts = np.unique(fup_bins[order], return_index=True)
            kept_idx_list, kept_ipw_list = [], []
            for grp in np.split(order, split_pts[1:]):
                pos_g = grp[is_acute_case[grp]]
                rest_g = grp[~is_acute_case[grp]]
                n_pos = len(pos_g)
                keep_rest_n = int(round(max(1, n_pos) * float(us_ratio)))
                if len(rest_g) > keep_rest_n and keep_rest_n > 0:
                    chosen_rest = rng.choice(rest_g, size=keep_rest_n, replace=False)
                    w_rest = float(len(rest_g)) / float(keep_rest_n)
                else:
                    chosen_rest = rest_g
                    w_rest = 1.0
                merged_g = np.concatenate([pos_g, chosen_rest])
                merged_w = np.concatenate([np.ones(len(pos_g), dtype=np.float32), np.full(len(chosen_rest), w_rest, dtype=np.float32)])
                perm = rng.permutation(len(merged_g))
                kept_idx_list.append(merged_g[perm])
                kept_ipw_list.append(merged_w[perm])
            final_tr_idx = np.concatenate(kept_idx_list)
            final_tr_ipw = np.concatenate(kept_ipw_list)
            n_full_tr = len(dataset["train"])
            dataset["train"] = (
                dataset["train"]
                .select(final_tr_idx, keep_in_memory=True)
                .flatten_indices(keep_in_memory=True)
                .add_column("ipw_weight", final_tr_ipw.tolist())
            )
            print(
                f">>> [SurvivEHR IPW] Per-FUP undersampling (ratio={us_ratio}, horizon<={int(max_h_case)}d): "
                f"kept {len(dataset['train'])}/{n_full_tr} train rows "
                f"(mean IPW={float(final_tr_ipw.mean()):.2f}, max IPW={float(final_tr_ipw.max()):.2f})",
                flush=True,
            )

    teav_cols = [*list(cfg["data_collator"]["eav_mappings"].keys()), "value_num", list(cfg["data_collator"]["time_mapping"].keys())[0]]
    for split in dataset.keys():
        col_vals = [split] * len(dataset[split])
        dataset[split] = dataset[split].add_column("split", col_vals)
        keep_cols = [
            c for c in [*teav_cols, *label_keys, "patientid", "fup", "ipw_weight", "split"]
            if c in dataset[split].column_names
        ]
        dataset[split].set_format(type="numpy", columns=keep_cols)

    train_dataset = dataset["train"]
    val_eval_ds = dataset["validation"]
    if fup_valid:
        fup_arr = np.asarray(val_eval_ds.data.column("fup"))
        val_idx = np.where(np.isin(fup_arr, fup_valid))[0]
        if len(val_idx) > 0:
            val_eval_ds = val_eval_ds.select(val_idx, keep_in_memory=True)
    eval_datasets = {"all": val_eval_ds}
    eval_fups = np.asarray(val_eval_ds.data.column("fup"))

    fmt_fn = lambda x: "-".join(f"{i:04d}" for i in sorted(([x] if isinstance(x, int) else x or [])))
    fut_str = fmt_fn(fup_train) if train_data_augment == "none" else train_data_augment
    fuv_str = fmt_fn(fup_valid)
    hrz_str = fmt_fn(horizons)
    task_subdir = f"hrz({hrz_str})_fut({fut_str})_fuv({fuv_str})"
    run_subdir = str(Path(task_key) / task_subdir)
    output_dir_path = finetuning_dir / run_subdir

    if cli_args.eval_only:
        resolved_ckpt = resolve_best_checkpoint(output_dir_path)
        if resolved_ckpt is None:
            sys.exit(f"Error: --eval-only found no fine-tuned checkpoint inside {output_dir_path}")
        init_ckpt_dir = str(resolved_ckpt)
    else:
        resolved_ckpt = resolve_best_checkpoint(pretrained_dir)
        if resolved_ckpt is None:
            sys.exit(f"Error: No valid pre-trained checkpoint found in {pretrained_dir}")
        init_ckpt_dir = str(resolved_ckpt)

    num_heads = len(task_keys)
    model_cfg = cfg["model"].copy()
    model_cfg["pretrained_dir"] = init_ckpt_dir
    model_cfg["embedding_layer_config"] = dict(model_cfg["embedding_layer_config"])
    model_cfg["embedding_layer_config"]["vocab_size"] = len(vocab)
    model_cfg["reset_weights"] = cli_args.reset_weights and (not cli_args.eval_only)
    model_cfg["enforce_monotonicity"] = False
    model_cfg["task"] = "survival"
    model_cfg["model_args"] = dict(model_cfg.get("model_args", {}))
    model_cfg["model_args"]["num_labels"] = num_heads
    model_cfg["model_args"]["problem_type"] = "regression"
    if is_pc_hazard:
        model_cfg["survival_head_config"] = {
            "num_tasks": num_heads,
            "time_grid": time_grid,
            "empirical_log_hazards": empirical_log_hazards,
        }

    model = PatientEmbeddingModelFactory.from_pretrained(**model_cfg)
    max_pos_embeddings = model.config.max_position_embeddings

    if cfg.get("use_lora", False):
        peft_conf = cfg.get("lora_config", {})
        peft_config = LoraConfig(**peft_conf)
        model = get_peft_model(model, peft_config)
        print(">>> LoRA Enabled. Trainable parameters:")
        model.print_trainable_parameters()

    collator_cfg = cfg["data_collator"].copy()
    collator_cfg.pop("mlm_label_keys", None)
    collator_cfg.pop("mlm_masking_rules", None)
    if "use_numerical_embedding" not in collator_cfg:
        collator_cfg["use_numerical_embedding"] = bool(model_cfg["embedding_layer_config"].get("use_numerical_embedding", True))
    if num_heads == 1:
        ft_collator = PatientDataCollatorForSurvival(
            **collator_cfg,
            time_label=survival_pairs[0][0],
            status_label=survival_pairs[0][1],
            max_position_embeddings=max_pos_embeddings,
        )
    else:
        ft_collator = PatientDataCollatorForSurvival(
            **collator_cfg,
            survival_label_pairs=survival_pairs,
            max_position_embeddings=max_pos_embeddings,
        )

    patience = ft_cfg.pop("early_stopping_patience", 7)
    callbacks = [EarlyStoppingCallback(early_stopping_patience=patience)]
    if hasattr(train_dataset, "configure_survival_resampling") and getattr(train_dataset, "mode", None) == "random_landmark":
        train_dataset.configure_survival_resampling(
            survival_pairs=survival_pairs,
            max_horizon=float(max(horizons)),
            us_ratio=(cfg.get("target_undersampling_ratio", None) if is_pc_hazard else None),
        )
        callbacks.append(make_dynamic_epoch_callback(train_dataset))
    evaluator = CustomEvaluator(
        horizons=horizons,
        task_keys=(task_keys if num_heads > 1 else None),
        eval_fups=eval_fups,
        early_stopping_metric=cfg.get("early_stopping_metric", "roc_auc"),
        early_stopping_mean=cfg.get("early_stopping_mean", "random_effects"),
        time_grid=time_grid,
    )

    if is_pc_hazard:
        loss_args["time_grid"] = time_grid
        loss_args["horizons"] = horizons
    loss_func = make_survival_loss_func(loss_name, loss_args)

    ft_cfg["output_dir"] = str(output_dir_path)
    if not cli_args.eval_only:
        for old_ckpt in output_dir_path.glob("checkpoint-*"):
            shutil.rmtree(old_ckpt, ignore_errors=True)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    head_pairs = [[t, int(h)] for t in task_keys for h in horizons]
    with open(output_dir_path / "run_metadata.json", "w") as f:
        json.dump(
            {
                "task_key": task_key,
                "tasks": list(task_keys),
                "horizons": list(horizons),
                "task_horizons": {t: list(horizons) for t in task_keys},
                "head_pairs": head_pairs,
                "label_keys": label_keys,
                "objective": loss_name,
                "time_grid": time_grid,
                "num_survival_tasks": num_heads,
                "value_embedding": get_value_embedding_tag(cfg),
                "use_numerical_embedding": bool(collator_cfg["use_numerical_embedding"]),
            },
            f, indent=2,
        )

    if cli_args.silent:
        ft_cfg["report_to"] = "none"
    if not torch.cuda.is_available():
        ft_cfg["bf16"] = False
        ft_cfg["fp16"] = False
        ft_cfg["use_cpu"] = True
    ft_args = TrainingArguments(**ft_cfg)

    use_wandb = (not cli_args.silent) and (cfg.get("finetuner", {}).get("report_to") == "wandb")
    if use_wandb:
        workspace = Path(__file__).stem
        run_name = f"{run_id}_{run_subdir}"
        wandb.init(project=workspace, name=run_name, config=cfg)

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

    if not cli_args.eval_only:
        trainer.train()

    test_survival_model(
        trainer=trainer,
        dataset_dict=dataset,
        all_possible_fups=all_possible_fups,
        output_dir=output_dir_path,
    )

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
    validation and test splits. Uses a SINGLE GPU forward pass per split (`mega_ds`),
    fits validation clinical-window/global Breslow priors before per-FUP validation
    scoring, and freezes validation baselines on the test split.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.remove_callback(EarlyStoppingCallback)

    meta_tasks = None
    meta_head_pairs = None
    meta_path = output_dir / "run_metadata.json"
    if meta_path.exists():
        meta_obj = json.loads(meta_path.read_text())
        if len(meta_obj.get("tasks") or []) > 1:
            meta_tasks = list(meta_obj["tasks"])
            if meta_obj.get("head_pairs"):
                meta_head_pairs = [(str(t), int(h)) for t, h in meta_obj["head_pairs"]]

    split_configs = []
    if "validation" in dataset_dict:
        split_configs.append({
            "split_label": "validation",
            "dataset": dataset_dict["validation"],
            "json_name": "validation_results.json",
        })
    if "test" in dataset_dict:
        split_configs.append({
            "split_label": "test",
            "dataset": dataset_dict["test"],
            "json_name": "test_results.json",
        })

    for config in split_configs:
        split_label = config["split_label"]
        mega_ds = config["dataset"]
        print(f"\n>>> Running final unified survival scoring pass over all FUPs for: [{split_label.upper()}]")
        trainer.compute_metrics.freeze_baseline_for_test = split_label == "test"
        final_metrics = {}
        all_predictions = {}

        # 1) Single GPU forward pass over the entire split (all FUPs pooled)
        results_mega = trainer.evaluate(mega_ds, metric_key_prefix=f"{split_label}_all")
        final_metrics.update(results_mega)
        mega_logits = np.asarray(trainer.compute_metrics.last_logits)
        mega_raw_labels = np.asarray(trainer.compute_metrics.last_raw_labels)
        mega_fups = np.asarray(mega_ds.data.column("fup"))
        mega_pids = np.asarray(mega_ds.data.column("patientid")) if "patientid" in mega_ds.column_names else None
        trainer.compute_metrics.saved_labels_and_probs = None

        if split_label == "validation":
            trainer.compute_metrics.fit_validation_priors_from_mega(mega_logits, mega_raw_labels, mega_fups)

        # 2) Compute per-FUP metrics & empirical-Bayes Breslow survival probabilities on CPU
        for fup in all_possible_fups:
            idx = np.where(mega_fups == fup)[0]
            if len(idx) == 0:
                continue
            fup_key = f"fup_{fup:04d}"
            prefix = f"{split_label}_{fup_key}"
            fup_logits = mega_logits[idx]
            fup_labels = mega_raw_labels[idx]

            trainer.compute_metrics.current_prefix = prefix
            raw_fup_metrics = trainer.compute_metrics(
                EvalPrediction(predictions=fup_logits, label_ids=fup_labels)
            )
            if trainer.compute_loss_func is not None:
                with torch.no_grad():
                    fup_loss = float(
                        trainer.compute_loss_func(
                            {"logits": torch.as_tensor(fup_logits, dtype=torch.float32)},
                            torch.as_tensor(fup_labels, dtype=torch.float32),
                        ).item()
                    )
                final_metrics[f"{prefix}_loss"] = fup_loss
            final_metrics.update({f"{prefix}_{k}": v for k, v in raw_fup_metrics.items()})

            for key, array in trainer.compute_metrics.saved_labels_and_probs.items():
                all_predictions[f"{prefix}_{key}"] = array

            if mega_pids is not None:
                patient_ids = mega_pids[idx]
                n_scored = len(trainer.compute_metrics.saved_labels_and_probs["labels"])
                if len(patient_ids) == n_scored:
                    all_predictions[f"{prefix}_patientid"] = patient_ids
                else:
                    print(f"    [WARN] {prefix}: {n_scored} scored rows vs {len(patient_ids)} samples; skipping ids")

            trainer.compute_metrics.saved_labels_and_probs = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        with open(output_dir / config["json_name"], "w") as f:
            json.dump(final_metrics, f, indent=4)
        print(f"    [SUCCESS] Saved {split_label} stats under: {output_dir}")

        save_predictions_table(
            all_predictions=all_predictions,
            split_label=split_label,
            horizons=extract_horizons_from_path(output_dir),
            output_dir=output_dir,
            csv_name=f"{split_label}_predictions.csv",
            task_keys=meta_tasks,
            head_pairs=meta_head_pairs,
        )

        print(f"    [SUCCESS] Saved {split_label} arrays and stats under: {output_dir}")


if __name__ == "__main__":
    main()
