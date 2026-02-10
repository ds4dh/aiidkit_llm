import os
import argparse
import yaml
import json
import sys
import gc
import torch
import wandb
from pathlib import Path
from datasets import Dataset
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
from transformers.trainer_utils import get_last_checkpoint
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
    # Identify pretaining and finetuning directories
    train_data_augment = CLI_CFG["train_data_augment"]
    mlm_masking_rules = CLI_CFG["data_collator"]["mlm_masking_rules"]
    pretrain_run_id = "-".join([f"{k[0]}{int(v * 100):02d}" for k, v in mlm_masking_rules.items()])
    finetune_run_id = pretrain_run_id if not cli_args.reset_weights else "no_pretrain"
    pretrained_dir = Path(CLI_CFG["result_dir"]) / pretrain_run_id / "pretraining"
    finetuning_dir = Path(CLI_CFG["result_dir"]) / finetune_run_id / "finetuning"

    # Iterate over prediction tasks
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
                        data_dir = Path(CLI_CFG["hf_data_dir"])
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
                        horizons=horizons,  # can be int of list of int for multi-label classification
                        fup_train=train_fups,
                        fup_valid=valid_fups,
                        fup_test=valid_fups,  # same as validation
                        train_data_augment=train_data_augment,
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
    task_key: str,  # todo: include this as well in multi-label classification
    horizons: list[int],
    fup_valid: list[int],
    fup_train: list[int],
    fup_test: list[int],
    train_data_augment: str,
    run_id: str = "default_run",
    pretrained_dir: Path = None,
    finetuning_dir: Path = None,
):
    """
    Fine-tune one model on a specific infection prediction task
    """
    # Load data for classification task, using vocabulary from pretraining phase
    time_mapping = CLI_CFG["data_collator"]["time_mapping"]
    eav_mappings = CLI_CFG["data_collator"]["eav_mappings"]
    label_keys = [f"label_{task_key}_{h:04d}d" for h in horizons]
    dataset, _, vocab = load_hf_data_and_metadata(
        data_dir=Path(CLI_CFG["hf_data_dir"]),
        fup_train=fup_train, fup_valid=fup_valid, fup_test=fup_test,
        label_keys=label_keys,
        target_undersampling_ratio=CLI_CFG.get("target_undersampling_ratio", None),
        time_mapping=time_mapping,
        eav_mappings=eav_mappings,
    )

    # Prepare training dataset
    dataset = dataset.filter(
        lambda x: all(x[k] != -100 for k in label_keys), desc="Keeping valid labels",
        num_proc=SAFE_NUM_PROCS, load_from_cache_file=False,
    )
    train_dataset = dataset["train"].map(
        lambda x: {"split": "train"}, desc="Tagging split",
        num_proc=SAFE_NUM_PROCS, load_from_cache_file=False,
    )
    eval_datasets = prepare_dataset_fup_dict(dataset["validation"], "val", fup_valid)
    test_datasets = prepare_dataset_fup_dict(dataset["test"], "test", fup_test)

    # Auto-detect model sub-directory and best pre-trained model checkpoint
    pretrained_last_ckpt_dir = get_last_checkpoint(str(pretrained_dir))
    if pretrained_last_ckpt_dir is None:
        sys.exit(f"Error: No checkpoint found in {pretrained_dir}")

    # Set up model configuration
    CLI_CFG["model"]["pretrained_dir"] = pretrained_last_ckpt_dir
    CLI_CFG["model"]["embedding_layer_config"]["vocab_size"] = len(vocab)
    CLI_CFG["model"]["reset_weights"] = cli_args.reset_weights
    CLI_CFG["model"]["task"] = "classification"
    CLI_CFG["model"]["model_args"]["num_labels"] = len(label_keys)
    CLI_CFG["model"]["model_args"]["problem_type"] = "multi_label_classification"  # if len(label_keys) > 1 else "binary_classification"

    # Initialize model, with weights from the pre-training stage
    model = PatientEmbeddingModelFactory.from_pretrained(**CLI_CFG["model"])
    max_pos_embeddings = model.config.max_position_embeddings  # pre-compute
    
    # Inject LoRA, if required
    if CLI_CFG.get("use_lora", False):
        peft_conf = CLI_CFG.get("lora_config", {})
        peft_config = LoraConfig(**peft_conf)
        model = get_peft_model(model, peft_config)  # this hides max_pos_embeddings
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
        early_stopping_metric=CLI_CFG["early_stopping_metric"],
    )

    # Training arguments, with the correct output directory
    ft_cfg = CLI_CFG["finetuner"].copy()
    fmt_fn = lambda x: "-".join(f"{i:04d}" for i in sorted(([x] if isinstance(x, int) else x or [])))    
    fut_str = fmt_fn(fup_train) if train_data_augment == "none" else train_data_augment    # training: "fut(0090)" vs "fut(all)"
    fuv_str = fmt_fn(fup_valid)  # validation: "fuv(0090)" vs "fuv(0000-0030...)"
    hrz_str = fmt_fn(horizons)  # horizons: "hrz(0030-0090)"
    task_subdir = f"hrz({hrz_str})_fut({fut_str})_fuv({fuv_str})"
    run_subdir = str(Path(task_key) / task_subdir)
    ft_cfg["output_dir"] = str(finetuning_dir / run_subdir)
    if cli_args.silent: ft_cfg["report_to"] = "none"
    ft_args = TrainingArguments(**ft_cfg)

    # Re-initialize a wandb run within the same worspace
    use_wandb = (not cli_args.silent) and (CLI_CFG.get("finetuner", {}).get("report_to") == "wandb")
    if use_wandb:
        workspace = Path(__file__).stem
        run_name = f"{run_id}_{run_subdir}"
        wandb.init(project=workspace, name=run_name, config=CLI_CFG)

    # Setup loss function
    loss_args = compute_loss_args(train_dataset, label_keys)
    loss_func = make_loss_func('poly1', loss_args)  # ce, weighted_ce, focal, dice, poly1

    # Trainer (standard HuggingFace)
    trainer = PrefixAwareTrainer(
        model=model,
        train_dataset=train_dataset, 
        eval_dataset=eval_datasets,  # dictionary with different follow-up periods
        args=ft_args,
        data_collator=ft_collator,
        compute_loss_func=loss_func,
        compute_metrics=evaluator,
        callbacks=callbacks,
    )

    # Fine-tune the model and reset wandb for the next run
    trainer.train()  # best model saved automatically
    
    # Test best model, using calibration from validation set
    if "test" in dataset:
        output_path = Path(ft_cfg["output_dir"]) / "test_results.json"
        test_model(trainer, eval_datasets, test_datasets, output_path)
        print(f"Final test results saved to {output_path}")

    # Reset wandb and clean up CUDA memory for the next run
    if use_wandb: wandb.finish()
    del dataset, model, trainer
    gc.collect()  # ensure deleted objects are collected by the garbage collector
    torch.cuda.empty_cache()  # free deleted local objects from CUDA memory 


class PrefixAwareTrainer(Trainer):
    """
    Inject the current prefix (e.g., "eval_fup_0030") into the evaluator, to
    access it from the custom evaluator, which allows to have stratified plots
    """
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        if hasattr(self.compute_metrics, "current_prefix"):
            self.compute_metrics.current_prefix = metric_key_prefix
            
        return super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)


def prepare_dataset_fup_dict(
    dataset: Dataset,
    prefix_name: str,
    fup_list: list[int],
):
    """
    Creates a dictionary of datasets for different follow-up periods
    """
    out_dict = {"all": dataset.map(
        lambda x: {"split": f"{prefix_name}_all"},
        desc="Tagging split", num_proc=SAFE_NUM_PROCS,
    )}
    for fup in fup_list:
        subset = dataset.filter(lambda x: x["fup"] == fup, num_proc=SAFE_NUM_PROCS)
        if len(subset) > 0:
            out_dict[f"fup_{fup:04d}"] = subset.map(
                lambda x: {"split": f"{prefix_name}_{fup}"},
                desc="Tagging split", num_proc=SAFE_NUM_PROCS,
            )
        else:
            print(f"Warning: No labeled samples found for follow-up {fup} in {prefix_name}.")
            
    return out_dict


def test_model(
    trainer: Trainer,
    eval_datasets: dict[str, Dataset],
    test_datasets: dict[str, Dataset],
    output_path: str,
):
    """
    Aggregates evaluation on all subsplits of the test set and save to output file
    """
    final_metrics = {}

    # Fit calibration on validation dataset
    print("\nFitting calibration on validation set...")
    val_metrics = trainer.evaluate(eval_datasets["all"], metric_key_prefix="val_all")
    final_metrics.update(val_metrics)

    # Evaluate on test sets
    test_all = test_datasets.pop("all")
    results_all = trainer.evaluate(test_all, metric_key_prefix="test_all")
    final_metrics.update(results_all)
    for fup_key, fup_test_dataset in test_datasets.items():
        results = trainer.evaluate(fup_test_dataset, metric_key_prefix=f"test_{fup_key}")
        final_metrics.update(results)

    # Save to disk for later plotting
    with open(output_path, "w") as f:
        json.dump(final_metrics, f, indent=4)


def scan_all_fups(data_dir: Path) -> list[int]:
    """
    Find all available follow-up folders (fup_XXXX) in the data directory
    """
    fups = []
    for path in data_dir.iterdir():
        if path.is_dir() and path.name.startswith("fup_"):
            try:
                # Extract integer from "fup_0090" -> 90
                val = int(path.name.split("_")[-1])
                fups.append(val)
            except ValueError:
                continue  # skip fup_None or malformed folders
    return sorted(fups)


if __name__ == "__main__":
    with open(cli_args.config, 'r') as f:
        CLI_CFG = yaml.safe_load(f)
    CLI_CFG = apply_config_overrides(CLI_CFG, cli_args.overrides)
    main()