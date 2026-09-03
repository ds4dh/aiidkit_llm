import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import yaml
import gc
import torch
import wandb
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
from src.utils import apply_config_overrides
from src.data.patient_dataset import load_hf_data_and_metadata
from src.model.patient_embedder import PatientEmbeddingModelFactory
from src.model.patient_embedder import PatientDataCollatorForMaskedLanguageModelling
from src.evaluation.evaluate_models import (
    DiscriminativeEmbeddingEvaluatorForMaskedLanguageModelling as CustomEvaluator,
    preprocess_logits_for_mlm_metrics,
)

def parse_args():
    parser = argparse.ArgumentParser(description="Pre-train patient sequence model via Masked Language Modeling.")
    parser.add_argument("--config", "-c", type=str, default="configs/discriminative_training.yaml")
    parser.add_argument("--silent", "-s", action="store_true", help="Disable wandb logging.")
    parser.add_argument("--overrides", "-o", action="append", default=[], help="Overrides config (JSON string or key=value).")
    return parser.parse_args()


def main():
    """
    Pre-train an encoder-like model using masked language modelling
    """
    cli_args = parse_args()
    with open(cli_args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    cfg = apply_config_overrides(cfg, cli_args.overrides)

    # Load whole patient dataset (all sequences) for masked language modelling
    # and associated entity-attribute-value vocabulary (required for encoding)
    time_mapping = cfg["data_collator"]["time_mapping"]
    eav_mappings = cfg["data_collator"]["eav_mappings"]
    data_dir = Path(cfg["data_dir"]) / cfg["data_split_type"]
    dataset, _, vocab = load_hf_data_and_metadata(
        data_dir=data_dir,
        fup_train=None,  # look for folder 'fup_None'
        fup_valid=None,  # look for folder 'fup_None'
        time_mapping=time_mapping,
        eav_mappings=eav_mappings,
    )
    dataset = {k: v.add_column("split", [k] * len(v)) for k, v in dataset.items()}

    # Initialize custom patient embedding model for masked language modelling
    cfg["model"]["task"] = "masked"
    cfg["model"]["config_args"]["vocab_size"] = len(vocab)
    cfg["model"]["embedding_layer_config"]["vocab_size"] = len(vocab)
    cfg["model"]["embedding_layer_config"]["time_mapping"] = time_mapping
    cfg["model"]["embedding_layer_config"]["eav_mappings"] = eav_mappings
    model = PatientEmbeddingModelFactory.create_from_backbone(**cfg["model"])

    # Use custom data collator for t-EAV formatted patient loading
    data_collator = PatientDataCollatorForMaskedLanguageModelling(
        **cfg["data_collator"],
        max_position_embeddings=model.config.max_position_embeddings,
    )

    # Evaluation pipelines
    patience = cfg["pretrainer"].pop("early_stopping_patience", 10)
    callbacks = [EarlyStoppingCallback(early_stopping_patience=patience)]
    evaluator = CustomEvaluator(do_clustering=True)

    # Training arguments, with the correct output directory
    mlm_masking_rules = cfg["data_collator"]["mlm_masking_rules"]
    run_id = "-".join([f"{k[0]}{int(v * 100):02d}" for k, v in mlm_masking_rules.items()])
    pt_cfg = cfg["pretrainer"].copy()
    result_dir = Path(cfg["result_dir"]) / cfg["data_split_type"]
    pt_cfg["output_dir"] = str(result_dir / run_id / "pretraining")
    if cli_args.silent:
        pt_cfg["report_to"] = "none"
    if not torch.cuda.is_available():
        pt_cfg["bf16"] = False
        pt_cfg["fp16"] = False
        pt_cfg["use_cpu"] = True
    pt_args = TrainingArguments(**pt_cfg)

    # Re-initialize a wandb run within the same workspace
    use_wandb = (not cli_args.silent) and (cfg.get("pretrainer", {}).get("report_to") == "wandb")
    if use_wandb:
        workspace = Path(__file__).stem
        wandb.init(project=workspace, name=run_id, config=cfg)

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

    # Reset wandb and clean up memory for the next run
    if use_wandb:
        wandb.finish()
    del dataset, model, trainer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()