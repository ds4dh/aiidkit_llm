import argparse
import yaml
import torch
import wandb
import itertools
from pathlib import Path
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    EarlyStoppingCallback,
    BitsAndBytesConfig as BnBConfig,
)
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
from src.data.process.patient_dataset import load_hf_data_and_metadata

import src.constants as constants
CSTS = constants.ConstantsNamespace()
CLI_CFG = {}
parser = argparse.ArgumentParser(description="Fine-tune a generative LLM on patient narratives.")
parser.add_argument("--config", "-c", type=str, default="configs/generative_training.yaml")

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
        bool_opts,                  # training_data_filtered
        CLASSIFICATION_LABEL_KEYS,  # which classification task is performed
        CLASSIFICATION_HORIZONS,    # horizon
        CLASSIFICATION_FUPS,        # follow-up
    )

    for is_train_data_filtered, label_key, horizon, fup in tasks:
        finetune_generative_model(
            label_key=label_key,
            horizon=horizon,
            fup_valid=fup,  # fup_valid is always fup
            fup_train=fup if is_train_data_filtered else None,
        )


def finetune_generative_model(
    label_key: str,
    horizon: int,
    fup_valid: list[int]|int,
    fup_train: list[int]|int|None=None,
):
    """
    Fine-tune generative model on a specific infection prediction task
    """    
    # Load custom patient narrative dataset
    dataset, _, _ = load_hf_data_and_metadata(
        data_dir=CSTS.HUGGINGFACE_DIR_PATH,
        metadata_dir=CSTS.METADATA_DIR_PATH,
        prediction_horizon=horizon,
        cutoff_days_train=fup_train,
        cutoff_days_valid=fup_valid,
    )

    # Prepare model arguments
    model_args = CLI_CFG["model"]["model_args"].copy()
    for key in ["dtype", "torch_dtype"]:
        if isinstance(model_args.get(key), str):
            model_args[key] = getattr(torch, model_args[key])

    # Prepare model and tokenizer
    model_id = CLI_CFG["model"]["model_id"]
    quant_cfg = CLI_CFG["model"].get("quantization")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.padding_side = "right"  # recommended for SFT
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        quantization_config=BnBConfig(**quant_cfg) if quant_cfg is not None else None,
        **model_args,
    )

    # Enable gradient checkpointing for memory efficiency
    model.gradient_checkpointing_enable()
    if quant_cfg is not None:
        model = prepare_model_for_kbit_training(model)

    # Define function to format the data into list of messages with labels
    task_dict = CSTS.LABEL_KEY_TO_CONDITION_MAPPING[label_key]
    target_col = label_key
    input_col = CLI_CFG["data"]["input_column"]
    system_prompt = CLI_CFG["data"]["system_prompt"].format(
        horizon=horizon,
        condition_description=task_dict["condition_description"],
    )
    # def apply_chat_structure(sample):
    #     """
    #     This would be useful if Qwen3 supports assistant_only_loss!
    #     """
    #     return {
    #         "messages": [
    #             {"role": "system", "content": system_prompt},
    #             {"role": "user", "content": sample[input_col]},
    #             {"role": "assistant", "content": str(sample[target_col])}
    #         ]
    #     }
    def format_prompt_completion(sample):
        prompt = f"{system_prompt}\n\n{sample[input_col]}\n\nResponse:"
        completion = str(sample[target_col])  # "0" or "1"
        return {"prompt": prompt, "completion": completion}

    # Map the dataset to create the "messages" column required by TRL
    dataset["train"] = dataset["train"].with_format("python").map(format_prompt_completion)
    dataset["validation"] = dataset["validation"].with_format("python").map(format_prompt_completion)

    # Evaluation pipeline
    patience = CLI_CFG["finetuning"].pop("patience")
    callbacks = [EarlyStoppingCallback(early_stopping_patience=patience)]

    # Setup training configuration, with correct output directory
    ft_cfg = CLI_CFG["finetuning"].copy()
    task_subdir = f"gen_hrz({horizon})"
    ft_cfg["output_dir"] = str(Path(ft_cfg["output_dir"]) / task_subdir)
    ft_cfg["logging_dir"] = str(Path(ft_cfg["logging_dir"]) / task_subdir)
    sft_config = SFTConfig(**ft_cfg)  # assistant_only_loss passed here
    peft_config = LoraConfig(**CLI_CFG["peft"])

    # Initialize generative fine-tuning trainer
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        args=sft_config,
        peft_config=peft_config,
        callbacks=callbacks,
    )

    # Finetune the generative model on the classification task
    trainer.train()


if __name__ == "__main__":
    # Parse command-line arguments
    cli_args = parser.parse_args()
    with open(cli_args.config, 'r') as f:
        CLI_CFG = yaml.safe_load(f)

    # Initialization needed for wandb
    if CLI_CFG["finetuning"]["report_to"] == "wandb":
        wandb.init(project=Path(__file__).stem)

    main()