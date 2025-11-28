import argparse
import yaml
import wandb
import itertools
from pathlib import Path
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
from src.model.patient_embedder import PatientEmbeddingModelFactory
from src.data.process.patient_dataset import load_hf_data_and_metadata
from src.model.patient_embedder import PatientDataCollatorForMaskedLanguageModelling
from src.model.evaluate_models import (
    CustomEmbeddingEvaluatorForMaskedLanguageModelling,
    preprocess_logits_for_metrics,
)

import src.constants as constants
CSTS = constants.ConstantsNamespace()
CLI_CFG = {}
parser = argparse.ArgumentParser(description="Train a UMLS normalization model.")
parser.add_argument("--config", "-c", type=str, default="configs/discriminative_training.yaml")


def main():
    """
    Fine-tune models on different classification tasks
    """
    # Iterator of all possible model combinations
    bool_opts = [False, True]
    combinations = itertools.product(
        bool_opts,  # use_positional_encoding
        bool_opts,  # use_pretrained_embeddings
    )

    for use_pos, use_pre in combinations:
        print(f"Starting pre-training with use_pos = {use_pos}, use_pre = {use_pre}")
        pretrain_mlm_model(
            use_positional_encoding=use_pos,
            use_pretrained_embeddings=use_pre,
        )


def pretrain_mlm_model(
    use_positional_encoding: bool,
    use_pretrained_embeddings: bool,
):
    """
    Pre-train an encoder-like model using masked language modelling
    """
    # Load whole patient dataset (all sequences) for masked language modelling
    # and associated entity-attribute-value vocabularies (required for encoding)
    dataset, _, vocabs = load_hf_data_and_metadata(
        data_dir=CSTS.HUGGINGFACE_DIR_PATH,
        metadata_dir=CSTS.METADATA_DIR_PATH,
    )

    # Initialize custom patient embedding model for masked language modelling
    mlm_label_key = CLI_CFG["data_collator"]["mlm_label_key"]
    mlm_label_name = CSTS.VOCAB_ID_TO_KEY_MAPPING[mlm_label_key]
    num_mlm_labels = len(vocabs[mlm_label_name])
    CLI_CFG["model"]["task"] = "masked"
    CLI_CFG["model"]["config_args"]["vocab_size"] = num_mlm_labels
    CLI_CFG["model"]["embedding_layer_config"]["vocabs"] = vocabs
    CLI_CFG["model"]["embedding_layer_config"]["use_positional_encoding"] = use_positional_encoding
    CLI_CFG["model"]["embedding_layer_config"]["use_pretrained_embeddings"] = use_pretrained_embeddings
    model = PatientEmbeddingModelFactory.create_from_backbone(**CLI_CFG["model"])

    # Use custom data collator for t-EAV formatted patient loading
    data_collator = PatientDataCollatorForMaskedLanguageModelling.from_kwargs(
        **CLI_CFG["data_collator"],
        max_position_embeddings=model.config.max_position_embeddings,
        use_pretrained_embeddings=model.use_pretrained_embeddings,
    )

    # Evaluation pipelines
    evaluator = CustomEmbeddingEvaluatorForMaskedLanguageModelling(vocabs=vocabs, do_clustering=True)
    patience = CLI_CFG["pretraining"].pop("patience")
    callbacks = [EarlyStoppingCallback(early_stopping_patience=patience)]

    # Training arguments, with the correct output directory
    pt_config = CLI_CFG["pretraining"].copy()
    model_subdir = f"pse({use_positional_encoding})_pte({use_pretrained_embeddings})"
    pt_config["output_dir"] = str(Path(pt_config["output_dir"]) / model_subdir)
    pt_args = TrainingArguments(**pt_config)

    # Trainer (standard HF)
    trainer = Trainer(
        model=model,
        args=pt_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=data_collator,
        compute_metrics=evaluator,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        callbacks=callbacks,
    )

    # Train the model (best checkpoint will be saved automatically)
    trainer.train()


if __name__ == "__main__":
    # Parse command-line arguments
    cli_args = parser.parse_args()
    with open(cli_args.config, 'r') as f:
        CLI_CFG = yaml.safe_load(f)

    # Initialization needed for wandb
    if CLI_CFG["pretraining"]["report_to"] == "wandb":
        wandb.init(project=Path(__file__).stem)

    main()