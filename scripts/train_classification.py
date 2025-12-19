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
from src.data.process.patient_dataset import load_hf_data_and_metadata
from src.model.model_utils import make_loss_func
from src.model.patient_embedder import PatientEmbeddingModelFactory
from src.model.patient_embedder import PatientDataCollatorForClassification
from src.model.evaluate_models import (
    DiscriminativeEmbeddingEvaluatorForClassification as CustomEvaluator,
)

import src.constants as constants
CSTS = constants.ConstantsNamespace()
CLI_CFG: dict[str, dict] = {}
parser = argparse.ArgumentParser(description="Fine-tune a model to predict future infections.")
parser.add_argument("--config", "-c", type=str, default="configs/discriminative_training.yaml")

LABEL_FORMATTING_FUNCTION = lambda task_key, horizon: f"label_{task_key}_{horizon:04d}d"
CLASSIFICATION_HORIZONS = [30, 60, 90]
CLASSIFICATION_FUPS = [0, 30, 60, 90, 180, 365]  # elements can also be list[int]
CLASSIFICATION_TASK_KEYS = [
    # "infection",
    "infection_bacteria",
    # "infection_virus",
    # "infection_fungi",
    # "graft_loss",
    # "death",
]


def main():
    """
    Fine-tune models on different classification tasks
    """
    # Iterator of all possible model and task combinations
    bool_opts = [False, True]
    tasks = itertools.product(
        bool_opts,                  # use_pretrained_embeddings
        bool_opts,                  # training_data_filtered
        CLASSIFICATION_TASK_KEYS,   # which classification task is performed
        CLASSIFICATION_HORIZONS,    # horizon
        CLASSIFICATION_FUPS,        # follow-up
    )

    for use_pre, is_train_data_filtered, task_key, horizon, fup in tasks:
        print(
            f"Starting fine-tuning with: use_pre = {use_pre}, "
            f"is_train_data_filtered = {is_train_data_filtered}, task_key = {task_key}, "
            f"horizon = {horizon}, fup = {fup}"
        )
        finetune_disciminative_model(
            task_key=task_key,
            use_pretrained_embeddings=use_pre,
            horizon=horizon,
            fup_valid=fup,  # valid always uses current fup, but train can use all fups!
            fup_train=fup if is_train_data_filtered else CLASSIFICATION_FUPS,
        )


def finetune_disciminative_model(
    task_key: str,
    use_pretrained_embeddings: bool,
    horizon: int,
    fup_valid: list[int]|int,
    fup_train: list[int]|int|None=None,
):
    """
    Fine-tune one model on a specific infection prediction task
    """
    # Load data for classification task
    dataset, _, vocab = load_hf_data_and_metadata(
        data_dir=CSTS.HUGGINGFACE_DIR_PATH,
        cutoff_days_train=fup_train,
        cutoff_days_valid=fup_valid,
        overwrite_cache=True,  # to ensure wrong vocab is not used (remove if run once already)
        metadata_cache_key=None,  # to have the same vocab as pre-training!
    )

    # Auto-detect model sub-directory and best pre-trained model checkpoint
    model_subdir = f"pte({use_pretrained_embeddings})"
    pretrained_dir = str(Path(CLI_CFG["pretrainer"]["output_dir"]) / model_subdir)
    pretrained_last_ckpt_dir = get_last_checkpoint(pretrained_dir)
    if pretrained_last_ckpt_dir is None:
        sys.exit(f"Error: No checkpoint found in {pretrained_dir}")

    # Create model and load weights from the pre-training stage
    CLI_CFG["model"]["pretrained_dir"] = pretrained_last_ckpt_dir
    CLI_CFG["model"]["embedding_layer_config"]["use_pretrained_embeddings"] = use_pretrained_embeddings
    CLI_CFG["model"]["embedding_layer_config"]["vocab_size"] = len(vocab)
    CLI_CFG["model"]["task"] = "classification"
    CLI_CFG["model"]["num_labels"] = 2
    model = PatientEmbeddingModelFactory.from_pretrained(**CLI_CFG["model"])

    # Custom data collator dedicated to patient classification
    label_key = LABEL_FORMATTING_FUNCTION(task_key, horizon)
    CLI_CFG["data_collator"]["label_key"] = label_key
    ft_collator = PatientDataCollatorForClassification(
        **CLI_CFG["data_collator"],
        max_position_embeddings=model.config.max_position_embeddings,
        use_pretrained_embeddings=model.use_pretrained_embeddings,
    )

    # Evaluation pipelines
    evaluator = CustomEvaluator(do_clustering=True)
    callbacks = [EarlyStoppingCallback(early_stopping_patience=10)]

    # Training arguments, with the correct output directory
    ft_cfg = CLI_CFG["finetuner"].copy()
    train_str = str(fup_train) if isinstance(fup_train, int) else "-".join(map(str, fup_train or []))
    valid_str = str(fup_valid) if isinstance(fup_valid, int) else "-".join(map(str, fup_valid or []))
    task_subdir = f"hrz({horizon})_fut({train_str})_fuv({valid_str})"
    run_subdir = str(Path(model_subdir) / task_subdir)
    ft_cfg["output_dir"] = str(Path(ft_cfg["output_dir"]) / run_subdir)
    ft_cfg["logging_dir"] = str(Path(ft_cfg["logging_dir"]) / run_subdir)
    ft_args = TrainingArguments(**ft_cfg)

    # Re-initialize a wandb run within the same worspace
    use_wandb = CLI_CFG.get("finetuner", {}).get("report_to") == "wandb"
    if use_wandb:
        workspace = Path(__file__).stem
        wandb.init(project=workspace, name=run_subdir, config=CLI_CFG, reinit=True)

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
    """...
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
    cli_args = parser.parse_args()
    with open(cli_args.config, 'r') as f:
        CLI_CFG = yaml.safe_load(f)
    main()