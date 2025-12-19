import argparse
import yaml
import gc
import torch
import wandb
import itertools
from pathlib import Path
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
from src.data.process.patient_dataset import load_hf_data_and_metadata
from src.model.patient_embedder import PatientEmbeddingModelFactory
from src.model.patient_embedder import PatientDataCollatorForMaskedLanguageModelling
from src.model.evaluate_models import (
    DiscriminativeEmbeddingEvaluatorForMaskedLanguageModelling as CustomEvaluator,
    preprocess_logits_for_mlm_metrics,
)

import src.constants as constants
CSTS = constants.ConstantsNamespace()
CLI_CFG: dict[str, dict] = {}
parser = argparse.ArgumentParser(description="Train a UMLS normalization model.")
parser.add_argument("--config", "-c", type=str, default="configs/discriminative_training.yaml")


def main():
    """
    Fine-tune models on different classification tasks
    """
    for use_pre in [True]:
        print(f"Starting pre-training with use_pre = {use_pre}")
        pretrain_mlm_model(use_pretrained_embeddings=use_pre)


def pretrain_mlm_model(use_pretrained_embeddings: bool):
    """
    Pre-train an encoder-like model using masked language modelling
    """
    # Load whole patient dataset (all sequences) for masked language modelling
    # and associated entity-attribute-value vocabulary (required for encoding)
    dataset, _, vocab = load_hf_data_and_metadata(
        data_dir=CSTS.HUGGINGFACE_DIR_PATH,
        cutoff_days_train=None,  # no cutoff for MLM pre-training (full sequences)
        cutoff_days_valid=None,  # no cutoff for MLM pre-training (full sequences)
    )
    import ipdb; ipdb.set_trace()

    # Initialize custom patient embedding model for masked language modelling
    CLI_CFG["model"]["task"] = "masked"
    CLI_CFG["model"]["config_args"]["vocab_size"] = len(vocab)
    CLI_CFG["model"]["embedding_layer_config"]["vocab_size"] = len(vocab)
    CLI_CFG["model"]["embedding_layer_config"]["use_pretrained_embeddings"] = use_pretrained_embeddings
    model = PatientEmbeddingModelFactory.create_from_backbone(**CLI_CFG["model"])

    # Use custom data collator for t-EAV formatted patient loading
    data_collator = PatientDataCollatorForMaskedLanguageModelling.from_kwargs(
        **CLI_CFG["data_collator"],
        max_position_embeddings=model.config.max_position_embeddings,
        use_pretrained_embeddings=model.use_pretrained_embeddings,
    )

    # Evaluation pipelines
    evaluator = CustomEvaluator(do_clustering=True)
    callbacks = [EarlyStoppingCallback(early_stopping_patience=5)]

    # Training arguments, with the correct output directory
    pt_config = CLI_CFG["pretrainer"].copy()
    run_subdir = f"pte({use_pretrained_embeddings})"
    pt_config["output_dir"] = str(Path(pt_config["output_dir"]) / run_subdir)
    pt_args = TrainingArguments(**pt_config)

    # Re-initialize a wandb run within the same worspace
    use_wandb = CLI_CFG.get("pretrainer", {}).get("report_to") == "wandb"
    if use_wandb:
        workspace = Path(__file__).stem
        wandb.init(project=workspace, name=run_subdir, config=CLI_CFG, reinit=True)

    # Trainer (standard HF)
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