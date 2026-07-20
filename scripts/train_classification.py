import argparse
import yaml
import json
import sys
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
from src.model.model_utils import make_loss_func, compute_loss_args
from src.model.patient_embedder import PatientEmbeddingModelFactory
from src.model.patient_embedder import PatientDataCollatorForClassification
from src.evaluation.plot_results import plot_task_results
from src.evaluation.evaluate_models import (
    DiscriminativeEmbeddingEvaluatorForClassification as CustomEvaluator,
)
from scripts.script_utils import scan_all_fups, prepare_dataset_fup_dict


CLI_CFG: dict[str, dict] = {}
SAFE_NUM_PROCS = 4  # max(1, len(os.sched_getaffinity(0)) - 2)

parser = argparse.ArgumentParser(description="Fine-tune a model to predict future infections.")
parser.add_argument("--config", "-c", type=str, default="configs/discriminative_training.yaml")
parser.add_argument("--reset_weights", "-r", action="store_true", help="Whether to reset model weights before fine-tuning.")
parser.add_argument("--plot_only", "-p", action="store_true", help="Skip run and goes directly to the plot.")
parser.add_argument("--silent", "-s", action="store_true", help="Disable wandb logging.")
parser.add_argument("--overrides", "-o", type=str, default="{}", help="Overrides config (JSON string).")
cli_args = parser.parse_args()


def main():
    """
    Fine-tune models for the prediction tasks in the yaml file from the CLI config
    """
    # Identify pretraining and finetuning directories
    train_data_augment = CLI_CFG["train_data_augment"]
    mlm_masking_rules = CLI_CFG["data_collator"]["mlm_masking_rules"]
    pretrain_run_id = "-".join([f"{k[0]}{int(v * 100):02d}" for k, v in mlm_masking_rules.items()])
    finetune_run_id = pretrain_run_id if not cli_args.reset_weights else "no_pretrain"
    result_dir = Path(CLI_CFG["result_dir"]) / CLI_CFG["data_split_type"]
    pretrained_dir = result_dir / pretrain_run_id / "pretraining"
    finetuning_subdir = CLI_CFG["finetuner"].pop("ft_subdir", "finetuning")
    finetuning_dir = result_dir / finetune_run_id / finetuning_subdir

    # Iterate over prediction tasks
    enforce_monotonicity = CLI_CFG["finetuner"].pop("enforce_monotonicity")
    for task_key, task_specs in CLI_CFG["prediction_tasks"].items():
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
                        data_dir = Path(CLI_CFG["data_dir"]) / CLI_CFG["data_split_type"]
                        train_fups = scan_all_fups(data_dir)  # all available
                    run_configs = [(train_fups, valid_fups)]

                # Ensure horizons is a list (even if only a single entry)
                if isinstance(horizons, int):
                    horizons = [horizons]

                # Execute the runs
                for train_fups, valid_fups in run_configs:
                    print(
                        f"Starting fine-tuning: Tasks={task_key} | Horizons={horizons} | "
                        f"Filtered={train_data_augment} | "
                        f"Train FUPs={train_fups} | Valid FUPs={valid_fups}"
                    )
                    finetune_disciminative_model(
                        task_key=task_key,
                        horizons=horizons,  
                        fup_train=train_fups,
                        fup_valid=valid_fups,
                        train_data_augment=train_data_augment,
                        enforce_monotonicity=enforce_monotonicity,
                        run_id=finetune_run_id,
                        pretrained_dir=pretrained_dir,
                        finetuning_dir=finetuning_dir,
                    )
        
        # After all runs for this task, generate plots
        plot_task_results(
            task_key=task_key,
            task_specs=task_specs,
            finetuning_dir=finetuning_dir,
            train_data_augment=train_data_augment,
        )


def finetune_disciminative_model(
    task_key: str,
    horizons: list[int],
    fup_valid: list[int],
    fup_train: list[int],
    train_data_augment: str,
    enforce_monotonicity: bool,
    run_id: str = "default_run",
    pretrained_dir: Path = None,
    finetuning_dir: Path = None,
):
    """
    Fine-tune one model on a specific infection prediction task
    """
    data_root_dir = Path(CLI_CFG["data_dir"]) / CLI_CFG["data_split_type"]
    all_possible_fups = scan_all_fups(data_root_dir)
    label_keys = [f"label_{task_key}_{h:04d}d" for h in horizons]

    # Load raw dataset and metadata
    dataset, _, vocab = load_hf_data_and_metadata(
        data_dir=data_root_dir,
        fup_train=fup_train,
        fup_valid=all_possible_fups,
        fup_test=all_possible_fups,
        label_keys=label_keys,
        target_undersampling_ratio=CLI_CFG.get("target_undersampling_ratio", None),
        time_mapping=CLI_CFG["data_collator"]["time_mapping"],
        eav_mappings=CLI_CFG["data_collator"]["eav_mappings"],
    )

    # Prepare split identification flags across arrays
    for split in dataset.keys():
        col_vals = [split] * len(dataset[split])
        dataset[split] = dataset[split].add_column("split", col_vals)
    for split in dataset.keys():
        label_matrix = np.stack([dataset[split][k] for k in label_keys], axis=1)
        keep_mask = (label_matrix != -100).any(axis=1)
        dataset[split] = dataset[split].select(np.where(keep_mask)[0])
        
    # Prepare datasets used for training and runtime evaluation (selected FUPs)
    train_dataset = dataset["train"]
    eval_datasets = prepare_dataset_fup_dict(dataset["validation"], fup_valid)

    # Auto-detect model sub-directory and best pre-trained model checkpoint
    pretrained_last_ckpt_dir = get_last_checkpoint(str(pretrained_dir))
    if pretrained_last_ckpt_dir is None:
        sys.exit(f"Error: No checkpoint found in {pretrained_dir}")

    # Set up model configuration
    CLI_CFG["model"]["pretrained_dir"] = pretrained_last_ckpt_dir
    CLI_CFG["model"]["embedding_layer_config"]["vocab_size"] = len(vocab)
    CLI_CFG["model"]["reset_weights"] = cli_args.reset_weights
    CLI_CFG["model"]["enforce_monotonicity"] = enforce_monotonicity
    CLI_CFG["model"]["task"] = "classification"
    CLI_CFG["model"]["model_args"]["num_labels"] = len(label_keys)
    CLI_CFG["model"]["model_args"]["problem_type"] = "multi_label_classification"

    # Initialize model
    model = PatientEmbeddingModelFactory.from_pretrained(**CLI_CFG["model"])
    max_pos_embeddings = model.config.max_position_embeddings  
    
    # Inject LoRA, if required
    if CLI_CFG.get("use_lora", False):
        peft_conf = CLI_CFG.get("lora_config", {})
        peft_config = LoraConfig(**peft_conf)
        model = get_peft_model(model, peft_config)
        print(">>> LoRA Enabled. Trainable parameters:")
        model.print_trainable_parameters()

    # Custom data collator dedicated to patient classification
    CLI_CFG["data_collator"]["label_keys"] = label_keys
    ft_collator = PatientDataCollatorForClassification(
        **CLI_CFG["data_collator"],
        max_position_embeddings=max_pos_embeddings,
    )

    # Evaluation pipelines
    patience = CLI_CFG["finetuner"].pop("early_stopping_patience", 20)
    callbacks = [EarlyStoppingCallback(early_stopping_patience=patience)]
    evaluator = CustomEvaluator(
        do_clustering=False,
        label_names=label_keys,
        enforce_monotonicity=enforce_monotonicity,
        early_stopping_metric=CLI_CFG["early_stopping_metric"],
    )

    # Setup loss function
    ft_cfg = CLI_CFG["finetuner"].copy()
    loss_name = ft_cfg.pop("loss_name", "poly1")
    loss_args = compute_loss_args(train_dataset, label_keys)
    loss_func = make_loss_func(loss_name, loss_args)  

    # Training arguments, with the correct output directory
    fmt_fn = lambda x: "-".join(f"{i:04d}" for i in sorted(([x] if isinstance(x, int) else x or [])))    
    fut_str = fmt_fn(fup_train) if train_data_augment == "none" else train_data_augment    
    fuv_str = fmt_fn(fup_valid)  
    hrz_str = fmt_fn(horizons)  
    task_subdir = f"hrz({hrz_str})_fut({fut_str})_fuv({fuv_str})"
    run_subdir = str(Path(task_key) / task_subdir)
    ft_cfg["output_dir"] = str(finetuning_dir / run_subdir)
    if cli_args.silent: ft_cfg["report_to"] = "none"
    ft_args = TrainingArguments(**ft_cfg)

    # Re-initialize a wandb run within the same workspace
    use_wandb = (not cli_args.silent) and (CLI_CFG.get("finetuner", {}).get("report_to") == "wandb")
    if use_wandb:
        workspace = Path(__file__).stem
        run_name = f"{run_id}_{run_subdir}"
        wandb.init(project=workspace, name=run_name, config=CLI_CFG)

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
    if use_wandb: wandb.finish()
    del dataset, model, trainer
    gc.collect()  
    torch.cuda.empty_cache()  


class PrefixAwareTrainer(Trainer):
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
            "npz_name": "validation_probs.npz"
        })
    if "test" in dataset_dict:
        test_fup_datasets = prepare_dataset_fup_dict(dataset_dict["test"], all_possible_fups)
        split_configs.append({
            "split_label": "test",       
            "datasets": test_fup_datasets, 
            "json_name": "test_results.json",       
            "npz_name": "test_probs.npz"
        })
    
    for config in split_configs:
        print(f"\n>>> Running final unified scoring pass over all FUPs for: [{config['split_label'].upper()}]")        
        final_metrics = {}
        all_predictions = {}
        
        # Process the aggregated global view first
        mega_ds = config["datasets"].pop("all")
        results_mega = trainer.evaluate(mega_ds, metric_key_prefix=f"{config['split_label']}_all")
        final_metrics.update(results_mega)
        for key, array in trainer.compute_metrics.saved_labels_and_probs.items():
            all_predictions[f"{config['split_label']}_all_{key}"] = array
            
        # Process individual FUP segments sequentially
        for fup_key, fup_dataset in config["datasets"].items():
            prefix = f"{config['split_label']}_{fup_key}"
            results_fup = trainer.evaluate(fup_dataset, metric_key_prefix=prefix)
            final_metrics.update(results_fup)
            
            for key, array in trainer.compute_metrics.saved_labels_and_probs.items():
                all_predictions[f"{prefix}_{key}"] = array
                
            trainer.compute_metrics.saved_labels_and_probs = None
            torch.cuda.empty_cache()
            
        # Write outputs to disk
        np.savez_compressed(output_dir / config["npz_name"], **all_predictions)
        with open(output_dir / config["json_name"], "w") as f:
            json.dump(final_metrics, f, indent=4)
            
        print(f"    [SUCCESS] Saved {config['split_label']} arrays and stats under: {output_dir}")


def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        return logits[0]  
    return logits


if __name__ == "__main__":
    with open(cli_args.config, 'r') as f:
        CLI_CFG = yaml.safe_load(f)
    CLI_CFG = apply_config_overrides(CLI_CFG, cli_args.overrides)
    main()