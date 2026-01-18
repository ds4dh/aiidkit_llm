import argparse
import yaml
import gc
import torch
import wandb
from pathlib import Path
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
from src.data.patient_dataset import load_hf_data_and_metadata
from src.model.patient_embedder import PatientEmbeddingModelFactory
from src.model.patient_embedder import PatientDataCollatorForMaskedLanguageModelling
from src.model.evaluate_models import (
    DiscriminativeEmbeddingEvaluatorForMaskedLanguageModelling as CustomEvaluator,
    preprocess_logits_for_mlm_metrics,
)

CLI_CFG: dict[str, dict] = {}
parser = argparse.ArgumentParser(description="Train a UMLS normalization model.")
parser.add_argument("--config", "-c", type=str, default="configs/discriminative_training.yaml")


def main():
    """
    Pre-train an encoder-like model using masked language modelling
    """
    # Load whole patient dataset (all sequences) for masked language modelling
    # and associated entity-attribute-value vocabulary (required for encoding)
    dataset, _, vocab = load_hf_data_and_metadata(
        data_dir=Path(CLI_CFG["hf_data_dir"]),
        fup_train=None,  # look for folder 'fup_None'
        fup_valid=None,  # look for folder 'fup_None'
    )
    dataset = {k: v.map(lambda x: {"split": k}) for k, v in dataset.items()}

    # Initialize custom patient embedding model for masked language modelling
    CLI_CFG["model"]["task"] = "masked"
    CLI_CFG["model"]["config_args"]["vocab_size"] = len(vocab)
    CLI_CFG["model"]["embedding_layer_config"]["vocab_size"] = len(vocab)
    model = PatientEmbeddingModelFactory.create_from_backbone(**CLI_CFG["model"])

    # Use custom data collator for t-EAV formatted patient loading
    data_collator = PatientDataCollatorForMaskedLanguageModelling(
        **CLI_CFG["data_collator"],
        max_position_embeddings=model.config.max_position_embeddings,
    )

    # Evaluation pipelines
    patience = CLI_CFG["pretrainer"].pop("early_stopping_patience", 10)
    callbacks = [EarlyStoppingCallback(early_stopping_patience=patience)]
    evaluator = CustomEvaluator(do_clustering=True)

    # Training arguments, with the correct output directory
    mlm_masking_rules = CLI_CFG["data_collator"]["mlm_masking_rules"]
    run_id = "-".join([f"{k[0]}{int(v * 100):02d}" for k, v in mlm_masking_rules.items()])
    pt_config = CLI_CFG["pretrainer"].copy()
    pt_config["output_dir"] = str(Path(CLI_CFG["result_dir"]) / run_id / "pretraining")
    pt_args = TrainingArguments(**pt_config)

    # Re-initialize a wandb run within the same worspace
    use_wandb = CLI_CFG.get("pretrainer", {}).get("report_to") == "wandb"
    if use_wandb:
        workspace = Path(__file__).stem
        wandb.init(project=workspace, name=run_id, config=CLI_CFG)

    # Trainer (standard HuggingFace)
    trainer = Trainer(
        model=model,
        args=pt_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=data_collator,
        compute_metrics=evaluator,
        preprocess_logits_for_metrics=preprocess_logits_for_mlm_metrics,
        callbacks=callbacks,
    )

    # Pre-train the model and reset wandb for the next run
    trainer.train()  # best model saved automatically

    # Reset wandb and clean up CUDA memory for the next run
    if use_wandb: wandb.finish()
    del dataset, model, trainer
    gc.collect()  # ensure deleted objects are collected by the garbage collector
    torch.cuda.empty_cache()  # free deleted local objects from CUDA memory 


if __name__ == "__main__":
    cli_args = parser.parse_args()
    with open(cli_args.config, 'r') as f:
        CLI_CFG = yaml.safe_load(f)
    main()