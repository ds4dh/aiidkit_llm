import argparse
import yaml
import sys
import wandb
import itertools
from pathlib import Path
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
from transformers.trainer_utils import get_last_checkpoint
from peft import LoraConfig
from src.data.process.patient_dataset import load_hf_data_and_metadata
from src.model.patient_embedder import PatientEmbeddingModelFactory
from src.model.patient_embedder import PatientDataCollatorForClassification
from src.model.evaluate_models import CustomEmbeddingEvaluatorForClassification
from src.model.model_utils import make_focal_loss_func

import src.constants as constants
CSTS = constants.ConstantsNamespace()
CLI_CFG = {}
parser = argparse.ArgumentParser(description="Fine-tune a model to predict future infections.")
parser.add_argument("--config", "-c", type=str, default="configs/discriminative_training.yaml")

CLASSIFICATION_HORIZONS = [30, 60, 90]
CLASSIFICATION_FUPS = [0, 30, 60, 90, 180, 365]  # elements can also be list[int]
CLASSIFICATION_LABEL_KEYS = [
    "infection_label_binary_any",
    "infection_label_binary_bacterial",
    # "infection_label_binary_viral",
    # "infection_label_binary_fungal",
    # "infection_label_categorical",
    # "infection_label_one_hot",
]


def main():
    """
    Fine-tune models on different classification tasks
    """
    # Iterator of all possible model and task combinations
    bool_opts = [False, True]
    tasks = itertools.product(
        bool_opts,                  # use_positional_encoding
        bool_opts,                  # use_pretrained_embeddings
        bool_opts,                  # training_data_filtered
        CLASSIFICATION_LABEL_KEYS,  # which classification task is performed
        CLASSIFICATION_HORIZONS,    # horizon
        CLASSIFICATION_FUPS,        # follow-up
    )

    for use_pos, use_pre, is_train_data_filtered, label_key, horizon, fup in tasks:
        finetune_disciminative_model(
            label_key=label_key,
            use_positional_encoding=use_pos,
            use_pretrained_embeddings=use_pre,
            horizon=horizon,
            fup_valid=fup,  # fup_valid is always fup
            fup_train=fup if is_train_data_filtered else None,
        )


def finetune_disciminative_model(
    label_key: str,
    use_positional_encoding: bool,
    use_pretrained_embeddings: bool,
    horizon: int,
    fup_valid: list[int]|int,
    fup_train: list[int]|int|None=None,
):
    """
    Fine-tune one model on a specific infection prediction task
    """
    # Load data for classification task
    dataset, _, vocabs = load_hf_data_and_metadata(
        data_dir=CSTS.HUGGINGFACE_DIR_PATH,
        metadata_dir=CSTS.METADATA_DIR_PATH,
        prediction_horizon=horizon,
        cutoff_days_train=fup_train,
        cutoff_days_valid=fup_valid,
    )

    # Auto-detect model sub-directory and best pre-trained model checkpoint
    model_subdir = f"pse({use_positional_encoding})_pte({use_pretrained_embeddings})"
    pretrained_dir = str(Path(CLI_CFG["pretraining"]["output_dir"]) / model_subdir)
    pretrained_last_ckpt_dir = get_last_checkpoint(pretrained_dir)
    if pretrained_last_ckpt_dir is None:
        sys.exit(f"Error: No checkpoint found in {pretrained_dir}")

    # Model configuration
    CLI_CFG["model"]["pretrained_dir"] = pretrained_last_ckpt_dir
    CLI_CFG["model"]["embedding_layer_config"]["use_positional_encoding"] = use_positional_encoding
    CLI_CFG["model"]["embedding_layer_config"]["use_pretrained_embeddings"] = use_pretrained_embeddings
    CLI_CFG["model"]["embedding_layer_config"]["vocabs"] = vocabs
    CLI_CFG["model"]["task"] = CSTS.LABEL_KEY_TO_CONDITION_MAPPING[label_key]["task_type"]
    CLI_CFG["model"]["num_labels"] = CSTS.LABEL_KEY_TO_CONDITION_MAPPING[label_key]["num_labels"]

    # Create model and prepare it for LoRA-based fine-tuning
    clf_model = PatientEmbeddingModelFactory.from_pretrained(**CLI_CFG["model"])
    peft_config = LoraConfig(**CLI_CFG["peft"])

    # Custom data collator dedicated to patient classification
    CLI_CFG["data_collator"]["label_key"] = label_key
    ft_collator = PatientDataCollatorForClassification(
        **CLI_CFG["data_collator"],
        max_position_embeddings=clf_model.config.max_position_embeddings,
        use_pretrained_embeddings=clf_model.use_pretrained_embeddings,
    )

    # Evaluation pipeline
    evaluator = CustomEmbeddingEvaluatorForClassification(do_clustering=True)
    patience = CLI_CFG["finetuning"].pop("patience")
    callbacks = [EarlyStoppingCallback(early_stopping_patience=patience)]

    # Training arguments, with the correct output directory
    ft_cfg = CLI_CFG["finetuning"].copy()
    task_subdir = f"hrz({horizon})_fut({"-".join(map(str, fup_train or []))})_fuv({"-".join(map(str, fup_valid or []))})"
    ft_cfg["output_dir"] = str(Path(ft_cfg["output_dir"]) / model_subdir / task_subdir)
    ft_cfg["logging_dir"] = str(Path(ft_cfg["logging_dir"]) / model_subdir / task_subdir)
    ft_args = TrainingArguments(**ft_cfg)

    # Trainer (standard HF)
    focal_loss_func = make_focal_loss_func(alpha=0.1)
    ft_trainer = Trainer(
        model=clf_model,
        args=ft_args,
        train_dataset=dataset["train"], 
        eval_dataset=dataset["validation"],
        data_collator=ft_collator,
        compute_loss_func=focal_loss_func,
        compute_metrics=evaluator,
        callbacks=callbacks,
    )

    # Finetune the model (best checkpoint will be saved automatically)
    ft_trainer.train()


if __name__ == "__main__":
    # Parse command-line arguments
    cli_args = parser.parse_args()
    with open(cli_args.config, 'r') as f:
        CLI_CFG = yaml.safe_load(f)

    # Initialization needed for wandb
    if CLI_CFG["pretraining"]["report_to"] == "wandb":
        wandb.init(project=Path(__file__).stem)

    main()