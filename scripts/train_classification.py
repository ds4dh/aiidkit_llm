import argparse
import yaml
import sys
import gc
import torch
import wandb
import itertools
from typing import Any
from pathlib import Path
from datasets import DatasetDict
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
cli_args = parser.parse_args()


def main():
    """
    Fine-tune models for the prediction tasks in the yaml file from the CLI config
    """
    # Iterate over the specific tasks we want to run
    for task_key, task_specs in CLI_CFG["prediction_tasks"].items():
        combinations = itertools.product(
            [False],  # is_train_data_filtered (only False for now)
            task_specs["horizons"],
            task_specs["fups"],
        )
        
        # Fine-tune model for all combinations
        for is_train_data_filtered, horizon, fup in combinations:
            print(
                f"Starting fine-tuning: Task={task_key} | "
                f"Horizon={horizon} | FUP={fup} | "
                f"Filtered={is_train_data_filtered}"
            )
            finetune_disciminative_model(
                task_key=task_key,
                horizon=horizon,
                fup_valid=fup,
                fup_train=fup if is_train_data_filtered else task_specs["fups"],
            )


def finetune_disciminative_model(
    task_key: str,
    horizon: int,
    fup_valid: list[int]|int,
    fup_train: list[int]|int|None=None,
):
    """
    Fine-tune one model on a specific infection prediction task
    """
    # Load data for classification task
    dataset, _, vocab = load_hf_data_and_metadata(
        data_dir=Path(CLI_CFG["hf_data_dir"]),
        fup_train=fup_train,
        fup_valid=fup_valid,
        overwrite_cache=False,
        metadata_cache_key="processed_train-None_val-None",  # aligns with pretraining
    )
    dataset = {k: v.map(lambda x: {"split": k}) for k, v in dataset.items()}
    
    # Filter unknown (censored) labels (-100)
    label_key = f"label_{task_key}_{horizon:04d}d"
    for split in dataset.keys():
        dataset[split] = dataset[split].filter(lambda x: x[label_key] != -100)

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
    evaluator = CustomEvaluator(do_clustering=True)

    # Training arguments, with the correct output directory
    ft_cfg = CLI_CFG["finetuner"].copy()
    fmt_fn = lambda x: "-".join(f"{i:04d}" for i in ([x] if isinstance(x, int) else x or []))
    task_subdir = f"hrz({fmt_fn(horizon)})_fut({fmt_fn(fup_train)})_fuv({fmt_fn(fup_valid)})"
    run_subdir = str(Path(task_key) / task_subdir)
    ft_cfg["output_dir"] = str(Path(CLI_CFG["result_dir"]) / run_id / "finetuning" / run_subdir)
    ft_args = TrainingArguments(**ft_cfg)

    # Re-initialize a wandb run within the same worspace
    use_wandb = CLI_CFG.get("finetuner", {}).get("report_to") == "wandb"
    if use_wandb:
        workspace = Path(__file__).stem
        run_name = f"{run_id}_{run_subdir}"
        wandb.init(project=workspace, name=run_name, config=CLI_CFG, reinit=True)

    # Setup loss function
    loss_args = compute_loss_args(dataset, label_key)
    focal_loss_func = make_loss_func("focal")  # weighted_ce, focal, dice, poly1

    # Trainer (standard HF)
    trainer = Trainer(
        model=model,
        train_dataset=dataset["train"], 
        eval_dataset=dataset["validation"],
        args=ft_args,
        data_collator=ft_collator,
        compute_loss_func=focal_loss_func,
        compute_metrics=evaluator,
        callbacks=callbacks,
    )

    # Fine-tune the model and reset wandb for the next run
    trainer.train()  # best model saved automatically

    # Reset wandb and clean up CUDA memory for the next run
    if use_wandb: wandb.finish()
    del dataset, model, trainer
    gc.collect()  # ensure deleted objects are collected by the garbage collector
    torch.cuda.empty_cache()  # free deleted local objects from CUDA memory 


def compute_loss_args(dataset: DatasetDict, label_key: str) -> dict[str, Any]:
    """ 
    Compute class weights for loss function based on training data distribution
    """
    train_labels = dataset["train"][label_key]
    valid_labels = dataset["validation"][label_key]
    train_tot, valid_tot = len(train_labels), len(valid_labels)
    train_pos, valid_pos = sum(train_labels), sum(valid_labels)
    train_neg, valid_neg = train_tot - train_pos, valid_tot - valid_pos
    
    print(f"Training samples: {train_tot} ({train_pos} +, {train_neg} -)")
    print(f"Validation samples: {valid_tot} ({valid_pos} +, {valid_neg} -)")
    
    return {
        "class_weights": [1.0, train_neg / (train_pos or 1)],
    }


if __name__ == "__main__":
    with open(cli_args.config, 'r') as f:
        CLI_CFG = yaml.safe_load(f)
    main()