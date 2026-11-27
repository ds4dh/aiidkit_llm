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
from datasets import Dataset
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback, EvalPrediction
from transformers.trainer_callback import PrinterCallback
from peft import LoraConfig, get_peft_model
from src.utils import apply_config_overrides, load_optuna_json_overrides
from src.data.dynamic_store import LandmarkSummaryPreprocessor, make_dynamic_epoch_callback
from src.data.patient_dataset import load_hf_data_and_metadata
from src.model.model_utils import (
    make_loss_func,
    compute_loss_args,
    GradientClippingGuard,
    resolve_best_checkpoint,
)
from src.model.patient_embedder import PatientEmbeddingModelFactory
from src.model.patient_embedder import PatientDataCollatorForClassification
from src.evaluation.plot_results import plot_task_results
from src.evaluation.evaluate_models import (
    DiscriminativeEmbeddingEvaluatorForClassification as CustomEvaluator,
)
from scripts.script_utils import (
    scan_all_fups,
    save_predictions_table,
    extract_horizons_from_path,
    format_pretrain_run_id,
    get_value_embedding_tag,
    resolve_run_id_with_val_tag,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune a model to predict future infections.")
    parser.add_argument("--config", "-c", type=str, default="configs/discriminative_training.yaml")
    parser.add_argument("--reset_weights", "-r", action="store_true", help="Whether to reset model weights before fine-tuning.")
    parser.add_argument("--plot_only", "-p", action="store_true", help="Skip run and goes directly to the plot.")
    parser.add_argument("--skip-test", "--skip_test", dest="skip_test", action="store_true",
                        help="Skip full 121-FUP validation/test scoring and plots (fast validation-only scoring for Optuna trials).")
    parser.add_argument("--eval-only", "--eval_only", dest="eval_only", action="store_true",
                        help="Skip training, load existing fine-tuned checkpoint from output_dir, and run full test_model + plots.")
    parser.add_argument("--silent", "-s", action="store_true", help="Disable wandb logging.")
    parser.add_argument("--optuna-json", "-J", action="append", default=[],
                        help="Path(s) to best_finetune_config.json written by train_optuna.py (applies portable_overrides before --overrides).")
    parser.add_argument("--overrides", "-o", action="append", default=[], help="Overrides config (JSON string or key=value).")
    parser.add_argument("--tasks", "-t", nargs="*", default=[],
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
    cfg = load_optuna_json_overrides(cfg, cli_args.optuna_json)
    cfg = apply_config_overrides(cfg, cli_args.overrides)

    # Optional run filter: run only a subset of the prediction_tasks entries
    # defined in the config (e.g. --tasks infection_grp_cmv or
    # --tasks multitask_bacteria,multitask_infections).
    if cli_args.tasks:
        wanted = [t.strip() for arg in (cli_args.tasks if isinstance(cli_args.tasks, list) else [cli_args.tasks]) for t in arg.split(",") if t.strip()]
        for t in list(wanted):
            if t not in cfg["prediction_tasks"]:
                for parent_specs in cfg["prediction_tasks"].values():
                    if isinstance(parent_specs, dict) and t in parent_specs.get("tasks", []):
                        ov_h = (parent_specs.get("task_horizons") or {}).get(t)
                        sub_horizons = [sorted(ov_h if isinstance(ov_h, list) else [ov_h])] if ov_h else parent_specs["horizons"]
                        cfg["prediction_tasks"][t] = {"fups": parent_specs["fups"], "horizons": sub_horizons}
                        break
        missing = [t for t in wanted if t not in cfg["prediction_tasks"]]
        if missing:
            sys.exit(f"Error: --tasks entries {missing} not found in config "
                     f"prediction_tasks {list(cfg['prediction_tasks'])}. "
                     f"(Entries must be active/uncommented in the YAML to be selectable.)")
        cfg["prediction_tasks"] = {t: cfg["prediction_tasks"][t] for t in wanted}
        print(f">>> --tasks filter active: running {list(cfg['prediction_tasks'])}")

    # Identify pretraining and finetuning directories
    train_data_augment = cfg.get("train_data_augment", "all")
    pretrain_run_id = format_pretrain_run_id(cfg)
    val_tag = get_value_embedding_tag(cfg)
    use_landmark_summary = bool(cfg.get("use_landmark_summary", False))
    if cli_args.reset_weights:
        finetune_run_id = f"no_pretrain_{val_tag}"
    elif cfg.get("finetune_run_id"):
        finetune_run_id = resolve_run_id_with_val_tag(str(cfg["finetune_run_id"]), val_tag)
        if use_landmark_summary and not finetune_run_id.endswith("_tab"):
            finetune_run_id = f"{finetune_run_id}_tab"
    elif use_landmark_summary:
        finetune_run_id = f"{pretrain_run_id}_tab"
    else:
        finetune_run_id = pretrain_run_id
    result_dir = Path(cfg["result_dir"]) / cfg["data_split_type"]
    pretrained_dir = result_dir / pretrain_run_id / "pretraining"
    if finetune_run_id != pretrain_run_id and pretrained_dir.exists():
        pt_alias = result_dir / finetune_run_id / "pretraining"
        pt_alias.parent.mkdir(parents=True, exist_ok=True)
        if not pt_alias.exists() and not pt_alias.is_symlink():
            pt_alias.symlink_to(pretrained_dir.resolve(), target_is_directory=True)
    finetuning_subdir = cfg["finetuner"].pop("ft_subdir", "finetuning")
    finetuning_dir = result_dir / finetune_run_id / finetuning_subdir

    # Iterate over prediction tasks
    enforce_monotonicity = cfg["finetuner"].pop("enforce_monotonicity", True)
    for task_key, task_specs in cfg["prediction_tasks"].items():
        # A task entry may bundle several outcome tasks into ONE multi-head run
        # via a 'tasks' list, with optional per-task horizon overrides in 'task_horizons'.
        task_keys = list(task_specs.get("tasks", [task_key]))
        task_horizons_override = dict(task_specs.get("task_horizons") or {})
        if not cli_args.plot_only:
            
            # Iterate over all horizon configurations
            for horizons in task_specs["horizons"]:
                
                # Define run configurations: list of tuples (train_fups_list, valid_fups_list)
                valid_fups = task_specs["fups"]
                if train_data_augment == "none":  # one run per follow-up period
                    run_configs = [([f], [f]) for f in valid_fups]
                else:  # single run with pooled follow-up periods
                    if train_data_augment == "valid":
                        train_fups = valid_fups  # 18 clinical landmarks only
                    elif train_data_augment in ("all", "random"):
                        data_dir = Path(cfg["data_dir"]) / cfg["data_split_type"]
                        train_fups = scan_all_fups(data_dir)  # all 121 eval FUPs (random mode samples continuously)
                    else:
                        raise ValueError(f"Unsupported train_data_augment='{train_data_augment}' (expected 'all', 'random', 'valid', or 'none').")
                    run_configs = [(train_fups, valid_fups)]

                # Ensure horizons is a sorted list (even if only a single entry)
                horizons = sorted(horizons if isinstance(horizons, list) else [horizons])

                # Execute the runs
                for train_fups, valid_fups in run_configs:
                    print(
                        f"Starting fine-tuning: Tasks={task_key} | DefaultHorizons={horizons} | "
                        f"TaskHorizonOverrides={task_horizons_override or None} | "
                        f"Filtered={train_data_augment} | "
                        f"Train FUPs={train_fups} | Valid FUPs={valid_fups}"
                    )
                    finetune_discriminative_model(
                        cfg=cfg,
                        cli_args=cli_args,
                        task_key=task_key,
                        task_keys=task_keys,
                        horizons=horizons,
                        task_horizons_override=task_horizons_override,
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
        if cli_args.skip_test:
            continue
        # multi-task runs: plot each sub-task individually with its own horizon grid
        if len(task_keys) > 1:
            for sub_task in task_keys:
                sub_specs = dict(task_specs)
                if sub_task in task_horizons_override:
                    ov_h = task_horizons_override[sub_task]
                    sub_specs["horizons"] = [sorted(ov_h if isinstance(ov_h, list) else [ov_h])]
                plot_task_results(
                    task_key=task_key,
                    task_specs=sub_specs,
                    finetuning_dir=finetuning_dir,
                    train_data_augment=train_data_augment,
                    sub_task=sub_task,
                )
            continue
        plot_task_results(
            task_key=task_key,
            task_specs=task_specs,
            finetuning_dir=finetuning_dir,
            train_data_augment=train_data_augment,
        )


def finetune_discriminative_model(
    cfg: dict,
    cli_args: argparse.Namespace,
    task_key: str,
    horizons: list[int],
    fup_valid: list[int],
    fup_train: list[int],
    train_data_augment: str,
    enforce_monotonicity: bool,
    task_keys: list[str] = None,
    task_horizons_override: dict[str, list[int]] | None = None,
    run_id: str = "default_run",
    pretrained_dir: Path = None,
    finetuning_dir: Path = None,
):
    """
    Fine-tune one model on one or more prediction tasks (multi-head).

    Multi-task runs (len(task_keys) > 1) concatenate the per-task horizon label
    columns into one multi-label target, task-major ordered:
    [t1_h1..t1_hK1, t2_h1..t2_hK2, ...]. Each task may optionally use its own
    horizon list via `task_horizons_override` (e.g. [30,60,90] for infections
    and [360,1080,3600] for death/graft_loss).
    """
    if task_keys is None:
        task_keys = [task_key]
    data_root_dir = Path(cfg["data_dir"]) / cfg["data_split_type"]
    all_possible_fups = scan_all_fups(data_root_dir)
    ov_map = task_horizons_override or {}
    per_task_horizons: dict[str, list[int]] = {}
    for t in task_keys:
        raw_h = ov_map.get(t, horizons)
        per_task_horizons[t] = sorted(raw_h if isinstance(raw_h, list) else [raw_h])
    head_pairs: list[tuple[str, int]] = [(t, int(h)) for t in task_keys for h in per_task_horizons[t]]
    label_keys = [f"label_{t}_{h:04d}d" for t, h in head_pairs]
    unique_all_horizons = sorted({h for _, h in head_pairs})

    # Compute output_dir early so --eval-only can locate existing checkpoint & landmark_scaler.json
    fmt_fn = lambda x: "-".join(f"{i:04d}" for i in sorted(([x] if isinstance(x, int) else x or [])))
    fut_str = fmt_fn(fup_train) if train_data_augment == "none" else train_data_augment
    fuv_str = fmt_fn(fup_valid)
    hrz_str = fmt_fn(unique_all_horizons)
    task_subdir = f"hrz({hrz_str})_fut({fut_str})_fuv({fuv_str})"
    run_subdir = str(Path(task_key) / task_subdir)
    output_dir_path = finetuning_dir / run_subdir

    # Load raw dataset and metadata (auto-switches to portable dynamic store when manifest.json is present)
    max_seq_len = int(cfg.get("model", {}).get("config_args", {}).get("max_position_embeddings", 1024)) - 2
    use_static_cond = bool(cfg.get("model", {}).get("embedding_layer_config", {}).get("use_static_conditioning", False))
    use_landmark_summary = bool(cfg.get("use_landmark_summary", False))
    classic_ml_split_dir = None
    saved_lm_prep = None
    if use_landmark_summary:
        cml_root = cfg.get("classic_ml_data_dir") or str(Path(cfg["data_dir"]).parent / "classic_ml")
        classic_ml_split_dir = Path(cml_root) / cfg["data_split_type"]
        scaler_path = output_dir_path / "landmark_scaler.json"
        if cli_args.eval_only and scaler_path.exists():
            saved_lm_prep = LandmarkSummaryPreprocessor.from_dict(json.loads(scaler_path.read_text(encoding="utf-8")))
    dataset, _, vocab = load_hf_data_and_metadata(
        data_dir=data_root_dir,
        fup_train=fup_train,
        fup_valid=all_possible_fups,
        fup_test=all_possible_fups,
        label_keys=label_keys,
        target_undersampling_ratio=cfg.get("target_undersampling_ratio", None),
        undersampling_mode=str(cfg.get("undersampling_mode", "global")),
        ipw_weight_cap=float(cfg.get("ipw_weight_cap", 10.0)),
        patient_weight_alpha=float(cfg.get("patient_weight_alpha", 0.0)),
        drop_unlabeled=cfg.get("drop_unlabeled", False),
        time_mapping=cfg["data_collator"]["time_mapping"],
        eav_mappings=cfg["data_collator"]["eav_mappings"],
        landmark_mode=("random_landmark" if train_data_augment == "random" else "fixed_grid"),
        random_landmark_stride_days=int(cfg.get("random_landmark_stride_days", 30)),
        random_landmark_max_fup=int(cfg.get("random_landmark_max_fup", 3600)),
        random_landmark_seed=int(cfg.get("finetuner", {}).get("seed", 1234)),
        max_length=max_seq_len,
        use_global_salience_truncation=bool(cfg.get("use_global_salience_truncation", True)),
        separate_static=use_static_cond,
        classic_ml_data_dir=classic_ml_split_dir,
        landmark_preprocessor=saved_lm_prep,
    )
    if "ipw_weight" in getattr(dataset["train"], "_extra_cols", {}):
        w_tr = dataset["train"]._extra_cols["ipw_weight"]
        print(
            f"    [Unbiased Training Weights] undersampling_mode={cfg.get('undersampling_mode', 'global')}, "
            f"patient_weight_alpha={cfg.get('patient_weight_alpha', 0.0)}, "
            f"w_min={float(w_tr.min()):.3f}, w_mean={float(w_tr.mean()):.3f}, w_max={float(w_tr.max()):.3f}"
        )

    # Prepare split identification flags across arrays
    for split in dataset.keys():
        col_vals = [split] * len(dataset[split])
        dataset[split] = dataset[split].add_column("split", col_vals)
    for split in dataset.keys():
        label_matrix = np.stack([np.asarray(dataset[split].data.column(k)) for k in label_keys], axis=1)
        keep_mask = (label_matrix != -100).any(axis=1)
        if not keep_mask.all():
            dataset[split] = dataset[split].select(np.where(keep_mask)[0], keep_in_memory=True).flatten_indices(keep_in_memory=True)
        
    # Prepare datasets used for training and runtime evaluation.
    # During training we only score the validation FUPs configured in `fup_valid`
    # (e.g. [0, 30, 60, 90, 180, 360]) rather than all 81 FUPs: this keeps
    # mid-training evaluations ~13x faster while matching the `_fuv(...)`
    # definition. The full 81-FUP validation and test breakdown is computed
    # post-training in a single forward pass per split (see test_model).
    train_dataset = dataset["train"]
    val_eval_ds = dataset["validation"]
    if fup_valid:
        fup_arr = np.asarray(val_eval_ds.data.column("fup"))
        val_idx = np.where(np.isin(fup_arr, fup_valid))[0]
        if len(val_idx) > 0:
            val_eval_ds = val_eval_ds.select(val_idx, keep_in_memory=True)
    eval_datasets = {"all": val_eval_ds}

    # Auto-detect model sub-directory and best checkpoint (fine-tuned when --eval-only, else pre-trained)
    ckpt_search_dir = output_dir_path if cli_args.eval_only else pretrained_dir
    resolved_ckpt = resolve_best_checkpoint(ckpt_search_dir)
    if resolved_ckpt is None:
        sys.exit(f"Error: No valid checkpoint found in {ckpt_search_dir}")
    pretrained_last_ckpt_dir = str(resolved_ckpt)

    # Monotonicity: predicted probabilities must increase with the horizon. In
    # multi-task runs this ordering only holds WITHIN each task's horizon block,
    # so model and evaluator receive explicit group sizes (task-major column
    # order); with a single horizon per task there is nothing to order -> disable.
    monotonic_groups = None
    if enforce_monotonicity:
        group_sizes = [len(per_task_horizons[t]) for t in task_keys]
        if len(task_keys) > 1:
            if any(g > 1 for g in group_sizes):
                monotonic_groups = group_sizes
            else:
                enforce_monotonicity = False
        elif group_sizes[0] <= 1:
            enforce_monotonicity = False

    # Set up model configuration
    model_cfg = cfg["model"].copy()
    model_cfg["embedding_layer_config"] = dict(cfg["model"]["embedding_layer_config"])
    model_cfg["pretrained_dir"] = pretrained_last_ckpt_dir
    model_cfg["embedding_layer_config"]["vocab_size"] = len(vocab)
    lm_prep_fitted = getattr(train_dataset, "landmark_preprocessor", None)
    if use_landmark_summary and lm_prep_fitted is not None and lm_prep_fitted.is_fitted:
        model_cfg["embedding_layer_config"]["landmark_summary_group_dims"] = list(lm_prep_fitted.group_dims)
        lm_bdim = cfg.get("landmark_bottleneck_dim", model_cfg["embedding_layer_config"].get("landmark_bottleneck_dim"))
        lm_drop = cfg.get("landmark_dropout", model_cfg["embedding_layer_config"].get("landmark_dropout"))
        lm_fup_gate = cfg.get("landmark_fup_gate", model_cfg["embedding_layer_config"].get("landmark_fup_gate", False))
        if cli_args.eval_only and (output_dir_path / "run_metadata.json").exists():
            prev_meta = json.loads((output_dir_path / "run_metadata.json").read_text(encoding="utf-8"))
            if lm_bdim is None and "landmark_bottleneck_dim" in prev_meta:
                lm_bdim = prev_meta["landmark_bottleneck_dim"]
            if lm_drop is None and "landmark_dropout" in prev_meta:
                lm_drop = prev_meta["landmark_dropout"]
            if "landmark_fup_gate" in prev_meta and "landmark_fup_gate" not in cfg:
                lm_fup_gate = prev_meta["landmark_fup_gate"]
        if lm_bdim is not None:
            model_cfg["embedding_layer_config"]["landmark_bottleneck_dim"] = int(lm_bdim)
        if lm_drop is not None:
            model_cfg["embedding_layer_config"]["landmark_dropout"] = float(lm_drop)
        model_cfg["embedding_layer_config"]["landmark_fup_gate"] = bool(lm_fup_gate)
        print(
            f"    [LandmarkSummaryEncoder] domains={dict(zip(lm_prep_fitted.group_names, lm_prep_fitted.group_dims))} "
            f"(total={sum(lm_prep_fitted.group_dims)}, bottleneck={lm_bdim}, dropout={lm_drop}, fup_gate={bool(lm_fup_gate)})"
        )
    model_cfg["reset_weights"] = cli_args.reset_weights
    model_cfg["enforce_monotonicity"] = monotonic_groups if monotonic_groups else enforce_monotonicity
    model_cfg["task"] = "classification"
    model_cfg["model_args"]["num_labels"] = len(label_keys)
    model_cfg["model_args"]["problem_type"] = "multi_label_classification"
    model_cfg["model_args"]["output_hidden_states"] = False

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
    collator_cfg.setdefault(
        "use_numerical_embedding",
        cfg["model"]["embedding_layer_config"].get("use_numerical_embedding", True),
    )
    collator_cfg["label_keys"] = label_keys
    ft_collator = PatientDataCollatorForClassification(
        **collator_cfg,
        max_position_embeddings=max_pos_embeddings,
    )

    # Evaluation pipelines
    ft_cfg = cfg["finetuner"].copy()
    patience = ft_cfg.pop("early_stopping_patience", 20)
    head_lr_multiplier = float(ft_cfg.pop("head_lr_multiplier", 1.0))
    callbacks = [EarlyStoppingCallback(early_stopping_patience=patience)]
    if hasattr(train_dataset, "set_epoch") and getattr(train_dataset, "mode", None) == "random_landmark":
        callbacks.append(make_dynamic_epoch_callback(train_dataset))
    evaluator = CustomEvaluator(
        do_clustering=False,
        label_names=label_keys,
        enforce_monotonicity=enforce_monotonicity,
        monotonic_groups=monotonic_groups,
        early_stopping_metric=cfg["early_stopping_metric"],
        early_stopping_subset=([t for t in ((cfg.get("prediction_tasks", {}).get(task_key, {}) or {}).get("early_stopping_subset") or cfg.get("early_stopping_subset") or []) if t in task_keys] or None),
        early_stopping_mean=cfg.get("early_stopping_mean", "harmonic"),
        n_bootstrap_eval=cfg.get("n_bootstrap_eval", 100),
    )

    # Setup loss function
    loss_name = ft_cfg.pop("loss_name", "poly1")
    loss_args = compute_loss_args(train_dataset, label_keys)
    loss_args.update(ft_cfg.pop("loss_args", {}) or {})  # optional manual override
    loss_func = make_loss_func(loss_name, loss_args)  

    # Training arguments, with the correct output directory
    ft_cfg["output_dir"] = str(output_dir_path)
    # Record the run layout so the test pass can map matrix columns back to
    # (task, horizon) pairs (essential for multi-task prediction CSVs).
    if not cli_args.eval_only:
        for old_ckpt in output_dir_path.glob("checkpoint-*"):
            shutil.rmtree(old_ckpt, ignore_errors=True)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    if use_landmark_summary and lm_prep_fitted is not None and lm_prep_fitted.is_fitted and not cli_args.eval_only:
        (output_dir_path / "landmark_scaler.json").write_text(
            json.dumps(lm_prep_fitted.to_dict(), indent=2), encoding="utf-8"
        )
    with open(output_dir_path / "run_metadata.json", "w") as f:
        json.dump(
            {
                "task_key": task_key,
                "tasks": list(task_keys),
                "horizons": list(unique_all_horizons),
                "task_horizons": per_task_horizons,
                "head_pairs": [[t, int(h)] for t, h in head_pairs],
                "label_keys": label_keys,
                "value_embedding": get_value_embedding_tag(cfg),
                "use_numerical_embedding": bool(collator_cfg["use_numerical_embedding"]),
                "undersampling_mode": str(cfg.get("undersampling_mode", "global")),
                "patient_weight_alpha": float(cfg.get("patient_weight_alpha", 0.0)),
                "landmark_bottleneck_dim": model_cfg["embedding_layer_config"].get("landmark_bottleneck_dim"),
                "landmark_dropout": model_cfg["embedding_layer_config"].get("landmark_dropout"),
                "landmark_fup_gate": bool(model_cfg["embedding_layer_config"].get("landmark_fup_gate", False)),
                "head_lr_multiplier": head_lr_multiplier,
                "backbone_lr": float(ft_cfg.get("learning_rate", 0.0)),
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

    # Re-initialize a wandb run within the same workspace
    use_wandb = (not cli_args.silent) and (cfg.get("finetuner", {}).get("report_to") == "wandb")
    if use_wandb:
        workspace = Path(__file__).stem
        run_name = f"{run_id}_{run_subdir}"
        wandb.init(project=workspace, name=run_name, config=cfg)

    # Trainer (standard HuggingFace + optional differential LR for fresh heads/tabular modules)
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
        head_lr_multiplier=head_lr_multiplier,
    )
    trainer.remove_callback(PrinterCallback)

    # Fine-tune the model (unless --eval-only was requested on an already-trained checkpoint)
    if not cli_args.eval_only:
        trainer.train()

    # Log learned FUP-gate profile if enabled
    lm_enc = getattr(getattr(trainer.model, "patient_embedder", None), "landmark_encoder", None)
    if lm_enc is not None and getattr(lm_enc, "use_fup_gate", False) and getattr(lm_enc, "fup_gate_lin", None) is not None:
        probe_fups = [0, 30, 90, 180, 360, 1080, 3600]
        with torch.no_grad():
            g_film_v, g_cls_v = lm_enc.compute_fup_gates(torch.tensor(probe_fups, dtype=torch.float32))
        gate_profile = {
            str(d): {"g_film": round(float(gf), 4), "g_cls": round(float(gc), 4)}
            for d, gf, gc in zip(probe_fups, g_film_v.squeeze(-1).cpu().tolist(), g_cls_v.squeeze(-1).cpu().tolist())
        }
        print(f"    [FupGate learned profile] {gate_profile}")
        meta_p = output_dir_path / "run_metadata.json"
        if meta_p.exists():
            meta_obj = json.loads(meta_p.read_text(encoding="utf-8"))
            meta_obj["learned_fup_gate"] = gate_profile
            meta_p.write_text(json.dumps(meta_obj, indent=2), encoding="utf-8")

    if cli_args.skip_test:
        # Fast validation-only evaluation on the 18 validation FUPs (~2s on GPU) for Optuna trials
        trainer.remove_callback(EarlyStoppingCallback)
        val_metrics = trainer.evaluate(eval_datasets["all"], metric_key_prefix="validation_all")
        with open(output_dir_path / "validation_results.json", "w") as f:
            json.dump(val_metrics, f, indent=4)
        print(f"    [SKIP-TEST] Saved fast validation_all metrics under: {output_dir_path}")
    else:
        # Post-hoc evaluation over ALL un-truncated elements
        test_model(
            trainer=trainer,
            dataset_dict=dataset, # Contains the full, un-truncated data splits
            all_possible_fups=all_possible_fups,
            output_dir=output_dir_path,
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
    Trainer that keeps per-FUP evaluation prefixes in sync, supports optional differential
    learning rates for freshly initialized classification/tabular modules vs. the pretrained
    backbone, and adds gradient clipping diagnostics + a non-finite gradient guard.
    """
    _HEAD_PARAM_PREFIXES = (
        "head.",
        "classifier.",
        "static_cls_fusion.",
        "landmark_cls_fusion.",
        "patient_embedder.landmark_encoder.",
    )

    def __init__(self, *args, head_lr_multiplier: float = 1.0, **kwargs):
        self.head_lr_multiplier = float(head_lr_multiplier)
        super().__init__(*args, **kwargs)
        if self.compute_loss_func is not None:
            self.model_accepts_loss_kwargs = False

    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer
        if abs(self.head_lr_multiplier - 1.0) < 1e-6:
            return super().create_optimizer()

        opt_model = self.model
        decay_parameters = set(self.get_decay_parameter_names(opt_model))
        base_lr = float(self.args.learning_rate)
        head_lr = base_lr * self.head_lr_multiplier
        wd = float(self.args.weight_decay)

        bb_decay, bb_nodecay, hd_decay, hd_nodecay = [], [], [], []
        for name, param in opt_model.named_parameters():
            if not param.requires_grad:
                continue
            is_head = any(name.startswith(pref) or (f".{pref}" in name) for pref in self._HEAD_PARAM_PREFIXES)
            use_decay = (name in decay_parameters) and ("fup_gate" not in name)
            if is_head:
                (hd_decay if use_decay else hd_nodecay).append(param)
            else:
                (bb_decay if use_decay else bb_nodecay).append(param)

        groups = []
        if bb_decay:
            groups.append({"params": bb_decay, "lr": base_lr, "weight_decay": wd})
        if bb_nodecay:
            groups.append({"params": bb_nodecay, "lr": base_lr, "weight_decay": 0.0})
        if hd_decay:
            groups.append({"params": hd_decay, "lr": head_lr, "weight_decay": wd})
        if hd_nodecay:
            groups.append({"params": hd_nodecay, "lr": head_lr, "weight_decay": 0.0})

        bb_n = sum(p.numel() for p in bb_decay) + sum(p.numel() for p in bb_nodecay)
        hd_n = sum(p.numel() for p in hd_decay) + sum(p.numel() for p in hd_nodecay)
        print(
            f"    [Differential LR] backbone_lr={base_lr:.2e} ({bb_n/1e6:.2f}M params) | "
            f"head_lr={head_lr:.2e} ({hd_n/1e6:.2f}M params, mult={self.head_lr_multiplier:.1f}x)"
        )

        optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(self.args)
        optimizer_kwargs.pop("lr", None)
        self.optimizer = optimizer_cls(groups, **optimizer_kwargs)
        return self.optimizer

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
    on both the validation and test splits, saving symmetrical JSON and CSV structures.
    Uses a SINGLE GPU forward pass per split (`mega_ds`) and slices logits/labels
    by FUP in memory on CPU, avoiding 81 redundant GPU DataLoader passes per split.
    """    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.remove_callback(EarlyStoppingCallback)
    
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
        print(f"\n>>> Running final unified scoring pass over all FUPs for: [{split_label.upper()}]")        
        final_metrics = {}
        all_predictions = {}
        
        # 1) Single GPU forward pass over the entire split (all FUPs pooled)
        results_mega = trainer.evaluate(mega_ds, metric_key_prefix=f"{split_label}_all")
        final_metrics.update(results_mega)
        mega_logits = np.asarray(trainer.compute_metrics.last_logits)
        mega_labels = np.asarray(trainer.compute_metrics.saved_labels_and_probs["labels"])
        mega_fups = np.asarray(mega_ds.data.column("fup"))
        mega_pids = np.asarray(mega_ds.data.column("patientid")) if "patientid" in mega_ds.column_names else None
        trainer.compute_metrics.saved_labels_and_probs = None
            
        # 2) Compute per-FUP metrics & prediction arrays on CPU by slicing mega_logits/mega_labels
        for fup in all_possible_fups:
            idx = np.where(mega_fups == fup)[0]
            if len(idx) == 0:
                continue
            fup_key = f"fup_{fup:04d}"
            prefix = f"{split_label}_{fup_key}"
            fup_logits = mega_logits[idx]
            fup_labels = mega_labels[idx]
            
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
                
            # Save row identities so predictions can be traced back to patients
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
            
        # Write metrics to disk (predictions go to the tidy CSV below)
        with open(output_dir / config["json_name"], "w") as f:
            json.dump(final_metrics, f, indent=4)
        print(f"    [SUCCESS] Saved {split_label} stats under: {output_dir}")

        # Tidy long-format table: patient_id, fup, [task,] horizon, label, prob,
        # prob_cal (the 'task' column appears for multi-task runs, where one
        # matrix column per (task, horizon) pair is scored jointly)
        task_keys = None
        head_pairs = None
        meta_path = output_dir / "run_metadata.json"
        if meta_path.exists():
            meta_obj = json.loads(meta_path.read_text())
            task_keys = meta_obj.get("tasks")
            if meta_obj.get("head_pairs") and task_keys:
                head_pairs = [(str(t), int(h)) for t, h in meta_obj["head_pairs"]]
        save_predictions_table(
            all_predictions=all_predictions,
            split_label=split_label,
            horizons=extract_horizons_from_path(output_dir),
            output_dir=output_dir,
            csv_name=f"{split_label}_predictions.csv",
            task_keys=task_keys,
            head_pairs=head_pairs,
        )
            
        print(f"    [SUCCESS] Saved {split_label} arrays and stats under: {output_dir}")


def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        return logits[0]  
    return logits


# Backwards-compatible alias for legacy imports
finetune_disciminative_model = finetune_discriminative_model


if __name__ == "__main__":
    main()
