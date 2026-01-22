import argparse
import yaml
import json
import sys
import gc
import torch
import wandb
import itertools
from typing import Any
from pathlib import Path
from datasets import Dataset, DatasetDict
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
from transformers.trainer_utils import get_last_checkpoint
from src.data.patient_dataset import load_hf_data_and_metadata
from src.model.model_utils import make_loss_func
from src.model.patient_embedder import PatientEmbeddingModelFactory
from src.model.patient_embedder import PatientDataCollatorForClassification
from src.model.evaluate_models import (
    DiscriminativeEmbeddingEvaluatorForClassification as CustomEvaluator,
)

CLI_CFG: dict[str, dict] = {}
parser = argparse.ArgumentParser(description="Fine-tune a model to predict future infections.")
parser.add_argument("--config", "-c", type=str, default="configs/discriminative_training.yaml")
parser.add_argument("--reset_weights", "-r", action="store_true", help="Whether to reset model weights before fine-tuning.")
parser.add_argument("--debug", "-d", action="store_true", help="Disable wandb logging for debugging.")
cli_args = parser.parse_args()


def main():
    """
    Fine-tune models for the prediction tasks in the yaml file from the CLI config
    """
    for task_key, task_specs in CLI_CFG["prediction_tasks"].items():
        
        # Iterate over filtering options and horizons
        settings = itertools.product(
            [False, True],  # is_train_data_filtered -> [True, False] to run both
            task_specs["horizons"],
        )
        for is_train_data_filtered, horizon in settings:
            
            # Define run configurations: list of tuples (train_fups_list, valid_fups_list)
            valid_fups = task_specs["fups"]
            if is_train_data_filtered:  # one run per follow-up period
                run_configs = [([f], [f]) for f in valid_fups]
            else:  # single run with all follow-up periods
                data_dir = Path(CLI_CFG["hf_data_dir"])
                # train_fups = scan_all_fups(data_dir)  # all available
                train_fups = valid_fups  # all of interest
                run_configs = [(train_fups, valid_fups)]

            # Execute the runs
            for train_fups, valid_fups in run_configs:
                print(
                    f"Starting fine-tuning: Task={task_key} | Horizon={horizon} | "
                    f"Filtered={is_train_data_filtered} | "
                    f"Train FUPs={train_fups} | Valid FUPs={valid_fups}"
                )
                finetune_disciminative_model(
                    task_key=task_key,
                    horizon=horizon,
                    fup_train=train_fups,
                    fup_valid=valid_fups,
                    fup_test=valid_fups,  # same as validation
                    is_train_data_filtered=is_train_data_filtered,
                )


def finetune_disciminative_model(
    task_key: str,
    horizon: int,
    fup_valid: list[int],
    fup_train: list[int],
    fup_test: list[int],
    is_train_data_filtered: bool,
):
    """
    Fine-tune one model on a specific infection prediction task
    """
    # Load data for classification task, using vocabulary from pretraining phase
    time_mapping = CLI_CFG["data_collator"]["time_mapping"]
    eav_mappings = CLI_CFG["data_collator"]["eav_mappings"]
    label_key = f"label_{task_key}_{horizon:04d}d"
    do_undersampling = (not is_train_data_filtered)
    dataset, _, vocab = load_hf_data_and_metadata(
        data_dir=Path(CLI_CFG["hf_data_dir"]),
        fup_train=fup_train, fup_valid=fup_valid, fup_test=fup_test,
        label_key=label_key,  # used for undersampling
        target_undersampling_ratio=10.0 if do_undersampling else None,
        time_mapping=time_mapping,
        eav_mappings=eav_mappings,
    )
    
    # Prepare training dataset
    dataset = dataset.filter(lambda x: x[label_key] != -100)  # remove unlabelled samples
    train_dataset = dataset["train"].map(lambda x: {"split": "train"}, desc="Tagging split", num_proc=8)
    eval_datasets = prepare_dataset_fup_dict(dataset["validation"], "val", fup_valid)
    test_datasets = prepare_dataset_fup_dict(dataset["test"], "test", fup_test)

    # Auto-detect model sub-directory and best pre-trained model checkpoint
    mlm_masking_rules = CLI_CFG["data_collator"]["mlm_masking_rules"]
    run_id = "-".join([f"{k[0]}{int(v * 100):02d}" for k, v in mlm_masking_rules.items()])
    pretrained_dir = str(Path(CLI_CFG["result_dir"]) / run_id / "pretraining")
    pretrained_last_ckpt_dir = get_last_checkpoint(pretrained_dir)
    if pretrained_last_ckpt_dir is None:
        sys.exit(f"Error: No checkpoint found in {pretrained_dir}")

    # Create model and load weights from the pre-training stage
    CLI_CFG["model"]["pretrained_dir"] = pretrained_last_ckpt_dir
    CLI_CFG["model"]["embedding_layer_config"]["vocab_size"] = len(vocab)
    CLI_CFG["model"]["task"] = "classification"
    CLI_CFG["model"]["reset_weights"] = cli_args.reset_weights
    CLI_CFG["model"]["num_labels"] = 2
    model = PatientEmbeddingModelFactory.from_pretrained(**CLI_CFG["model"])

    # Custom data collator dedicated to patient classification
    CLI_CFG["data_collator"]["label_key"] = label_key
    ft_collator = PatientDataCollatorForClassification(
        **CLI_CFG["data_collator"],
        max_position_embeddings=model.config.max_position_embeddings,
    )

    # Evaluation pipelines
    patience = CLI_CFG["finetuner"].pop("early_stopping_patience", 20)
    callbacks = [EarlyStoppingCallback(early_stopping_patience=patience)]
    evaluator = CustomEvaluator(do_clustering=False)

    # Training arguments, with the correct output directory
    ft_cfg = CLI_CFG["finetuner"].copy()
    fmt_fn = lambda x: "-".join(f"{i:04d}" for i in sorted(([x] if isinstance(x, int) else x or [])))    
    fut_str = "all" if not is_train_data_filtered else fmt_fn(fup_train)  # training: "fut(0090)" vs "fut(all)"
    fuv_str = fmt_fn(fup_valid)  # validation: "fuv(0090)" vs "fuv(0000-0030...)"
    task_subdir = f"hrz({horizon:04d})_fut({fut_str})_fuv({fuv_str})"
    run_subdir = str(Path(task_key) / task_subdir)
    ft_cfg["output_dir"] = str(Path(CLI_CFG["result_dir"]) / run_id / "finetuning" / run_subdir)
    if cli_args.debug: ft_cfg["report_to"] = "none"
    ft_args = TrainingArguments(**ft_cfg)

    # Re-initialize a wandb run within the same worspace
    use_wandb = (not cli_args.debug) and (CLI_CFG.get("finetuner", {}).get("report_to") == "wandb")
    if use_wandb:
        workspace = Path(__file__).stem
        run_name = f"{run_id}_{run_subdir}"
        wandb.init(project=workspace, name=run_name, config=CLI_CFG)

    # Setup loss function
    loss_args = compute_loss_args(train_dataset, label_key)
    loss_func = make_loss_func('focal', loss_args)  # weighted_ce, focal, dice, poly1

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
    

def compute_loss_args(dataset: DatasetDict, label_key: str) -> dict[str, Any]:
    """ 
    Compute class weights for loss function based on training data distribution
    """
    train_labels = dataset[label_key]
    train_pos, train_tot = sum(train_labels), len(train_labels)
    train_neg = train_tot - train_pos
    print(f"Training samples: {train_tot} ({train_pos} +, {train_neg} -)")    
    pos_weight = train_neg / (train_pos if train_pos > 0 else 1)    
    return {
        "alpha": None,  # [1.0, pos_weight],  # controls volume (imbalance)
        "gamma": 2.0,  # controls focus on hard samples (standard = 2.0)
        "epsilon": 1.0,  # controls gradient flow (anti-flatline; standard = 1.0)
    }


def prepare_dataset_fup_dict(
    dataset: Dataset,
    prefix_name: str,
    fup_list: list[int],
):
    """
    Creates a dictionary of datasets for different follow-up periods
    """
    out_dict = {"all": dataset.map(
        lambda x: {"split": f"{prefix_name}_all"}, desc="Tagging split", num_proc=8,
    )}
    for fup in fup_list:
        subset = dataset.filter(lambda x: x["fup"] == fup)
        if len(subset) > 0:
            out_dict[f"fup_{fup:04d}"] = subset.map(
                lambda x: {"split": f"{prefix_name}_{fup}"}, desc="Tagging split", num_proc=8,
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
    main()