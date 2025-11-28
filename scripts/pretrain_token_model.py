import os
import yaml
import torch
import argparse
from datasets import DatasetDict
from transformers import Trainer, TrainingArguments
from transformers.integrations import WandbCallback

from src.model.patient_embedder import (
    PatientTokenEmbeddingModel,a
    PatientDataCollatorForLanguageModelling,
)
from src.model.model_utils import (
    EarlyStoppingCallbackWithWarmup,
    SupervisedTaskWeightSchedulerCallback,
    WandbPlottingCallback,
    preprocess_logits_for_metrics,
)
from src.model.evaluate_models import CustomEmbeddingEvaluator
from src.data.process.patient_dataset import load_hf_data_and_metadata
import src.constants as constants
csts = constants.ConstantsNamespace()


def main(args):
    """
    Train a patient embedding model with MLM (or causal LM)
    """
    # Load model configuration
    with open(args.pretrain_config_path, "r") as f:
        run_cfg = yaml.safe_load(f)

    # Call the core training logic
    best_metric = run_training(run_cfg, args)
    print(f"\nStandalone training finished. Best metric achieved: {best_metric}")


def run_training(
    run_cfg: dict[str, str],
    args: argparse.Namespace,
) -> float:
    """
    Train a patient embedding model with MLM (using supervised labels if required)
    Return the best metric value, computed on the validation dataset
    """
    # Setup MLM / causal LM training dataset, and a special evaluation dataset if required
    dataset, vocabs = load_dataset_and_vocabs_for_patient_token_embedding(
        huggingface_dir_path=csts.HUGGINGFACE_DIR_PATH,
        metadata_dir_path=csts.METADATA_DIR_PATH,
        args=args,
    )

    # Load model and data collator for the MLM (or causal LM) task
    model = load_model_for_patient_token_embedding(
        model_cfg=run_cfg["MODEL_ARGUMENTS"],
        vocabs=vocabs,
    )
    data_collator = load_data_collator_for_patient_token_embedding(
        model=model,
        model_cfg=run_cfg["MODEL_ARGUMENTS"],
        vocabs=vocabs,
    )

    # Get trainer callbacks
    trainer_callbacks = get_trainer_callbacks(
        train_cfg=run_cfg["TRAINING_ARGUMENTS"],
        eval_cfg=run_cfg["EVALUATION_ARGUMENTS"],
        model_cfg=run_cfg["MODEL_ARGUMENTS"],
    )

    # Load trainer
    trainer = load_trainer_for_patient_token_embedding(
        model=model,
        dataset=dataset,
        run_cfg=run_cfg,
        data_collator=data_collator,
        trainer_callbacks=trainer_callbacks,
        debug_flag=args.debug,
    )

    # Train the model
    trainer.train()
    print("Training complete")

    return trainer.state.best_metric


def load_dataset_and_vocabs_for_patient_token_embedding(
    huggingface_dir_path: str,
    metadata_dir_path: str,
    args: argparse.Namespace,
):
    """
    Load the dataset and vocabularies for patient token embedding
    """
    # Partial patient sequences with infection labels (derived from future events)
    # Note: sequence are up to [0, 7, 30, 120, 365, 1000, 3000, None] after the
    # transplantation day, where None is full sequence minus prediction horizon
    if args.use_supervised_labels:
        dataset, _, vocabs = load_hf_data_and_metadata(
            data_dir=huggingface_dir_path,
            metadata_dir=metadata_dir_path,
            prediction_horizon=args.prediction_horizon,
            cutoff_days_train=args.cutoff_days_train,
            cutoff_days_valid=args.cutoff_days_valid,
        )

    # Full patient sequences without infection labels
    else:
        dataset, _, vocabs = load_hf_data_and_metadata(
            data_dir=huggingface_dir_path,
            metadata_dir=metadata_dir_path,
        )

    return dataset, vocabs


def load_model_for_patient_token_embedding(
    model_cfg: dict[str, str],
    vocabs: dict[str, dict[str, int]],
    for_sentence_embedding: bool=False,
) -> PatientTokenEmbeddingModel:
    """
    Initialize a patient embedding model with correct data type
    """
    # Huggingface will only understand the actual torch dtype object
    if "torch_dtype" in model_cfg["original_model_params"]:
        torch_dtype = getattr(
            torch,
            model_cfg["original_model_params"]["torch_dtype"],
        )
        model_cfg["original_model_params"]["torch_dtype"] = torch_dtype

    # Special case for when model will be finetuned to a sentence embedding model
    if for_sentence_embedding:
        model_cfg["send_hidden_states_to_cpu"] = False

    return PatientTokenEmbeddingModel(vocabs=vocabs, **model_cfg)


def load_data_collator_for_patient_token_embedding(
    model: PatientTokenEmbeddingModel,
    model_cfg: dict[str, str],
    vocabs: dict[str, dict[str, int]],
):
    """
    Load the data collator, which depends on the model and its configuration
    """
    data_collator = PatientDataCollatorForLanguageModelling(
        mlm=(model_cfg["original_model_task"] == "masked"),
        num_mlm_labels=len(vocabs["value_binned"]),
        num_tokens_max=model.num_tokens_max,
        use_supervised_task=model_cfg["use_supervised_task"],
        supervised_task_key=model_cfg["supervised_task_key"],
        use_pretrained_embeddings=model_cfg["use_pretrained_embeddings_for_input_layer"],
    )

    return data_collator


def get_trainer_callbacks(
    train_cfg: dict[str, str],
    eval_cfg: dict[str, str],
    model_cfg: dict[str, str],
) -> list:
    """
    Get the callbacks required by the trainer
    """
    # Initialize the list of callbacks
    callbacks = []

    # Add the supervised task weight scheduler if it's configured
    early_stopping_warmup_steps = train_cfg["warmup_steps"]
    if model_cfg.get("use_supervised_task") and "supervised_task_schedule" in model_cfg:
        schedule_params = model_cfg["supervised_task_schedule"]
        callbacks.append(SupervisedTaskWeightSchedulerCallback(**schedule_params))

        # Use a later start for activating early stopping if we have a supervised schedule
        early_stopping_warmup_steps = \
            (schedule_params.get("start_steps", 0) + schedule_params.get("end_steps", 0)) / 2
    
    # Add the early stopping callback
    callbacks.append(
        # EarlyStoppingCallback(early_stopping_patience=eval_cfg["early_stopping_patience"]),
        EarlyStoppingCallbackWithWarmup(
            warmup_steps=early_stopping_warmup_steps,
            early_stopping_patience=eval_cfg["early_stopping_patience"]
        )
    )

    # Add WandB callback if necessary, but last!
    if train_cfg["report_to"] == "wandb":
        callbacks.append(WandbCallback())

    return callbacks


def load_trainer_for_patient_token_embedding(
    model: PatientTokenEmbeddingModel,
    dataset: DatasetDict,
    run_cfg: dict[str, str],
    data_collator: PatientDataCollatorForLanguageModelling,
    trainer_callbacks: list=[],
    debug_flag: bool=False,
) -> Trainer:
    """
    Initialize a huggingface trainer object
    """
    # In debug mode, evaluation comes earlier for quick assessment
    cfg_training_args: dict = run_cfg["TRAINING_ARGUMENTS"]
    if debug_flag:
        cfg_training_args.update(
            {"eval_strategy": "steps", "eval_steps": 10, "report_to": "none"},
        )

    # Handle wandb specific settings
    wandb_project_name = cfg_training_args.pop("wandb_project", None)
    if run_cfg["TRAINING_ARGUMENTS"]["report_to"] == "wandb":
        os.environ["WANDB_PROJECT"] = wandb_project_name
        wandb_plotting_callback = WandbPlottingCallback()
        trainer_callbacks.append(wandb_plotting_callback)  # sharing callback
    else:
        wandb_plotting_callback = None

    # Load training arguments
    eval_cfg: dict = run_cfg["EVALUATION_ARGUMENTS"]
    training_arguments = TrainingArguments(
        remove_unused_columns=False,
        metric_for_best_model=eval_cfg["metric_for_best_model"],
        greater_is_better=eval_cfg["greater_is_better"],
        **cfg_training_args,
    )

    # Define model evaluation pipeline
    model_evaluator = CustomEmbeddingEvaluator(
        eval_dataset=dataset["validation"],
        embedding_mode="token",
        wandb_plotting_callback=wandb_plotting_callback,
        eval_label_key=run_cfg["MODEL_ARGUMENTS"]["supervised_task_key"],
        **eval_cfg,
    )

    # Define trainer object
    preprocess_fn = preprocess_logits_for_metrics if model.is_causal else None
    trainer = Trainer(
        model=model,
        args=training_arguments,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=data_collator,
        compute_metrics=model_evaluator,
        preprocess_logits_for_metrics=preprocess_fn,
        callbacks=trainer_callbacks,
    )

    return trainer


if __name__ == "__main__":

    # Common command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("-sl", "--use_supervised_labels", action="store_false", help="Use true labels to evaluate models")
    parser.add_argument("-ph", "--prediction_horizon", default=30, type=int, choices=csts.PREDICTION_HORIZONS, help="Horizon of the infection prediction task for model evaluation")
    parser.add_argument("-ct", "--cutoff_days_train", nargs="+", type=int, default=None, choices=csts.CUTOFF_DAYS, help="Cutoff days for the training dataset, take them all if None")
    parser.add_argument("-cv", "--cutoff_days_valid", nargs="+", type=int, default=None, choices=csts.CUTOFF_DAYS, help="Cutoff days for the evaluation/testing dataset, take them all if None")

    # Arguments that are specific to the current script
    parser.add_argument("-pc", "--pretrain_config_path", default="configs/patient_token_embedder.yaml", help="Path to the pretraining (masked LM) configuration file")

    # Run the pretraining script
    args = parser.parse_args()
    main(args)