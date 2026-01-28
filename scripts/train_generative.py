import argparse
import yaml
import gc
import torch
import wandb
import itertools
from pathlib import Path
from datasets import concatenate_datasets
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    EarlyStoppingCallback,
    BitsAndBytesConfig as BnBConfig,
)
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
from src.data.process.patient_dataset import load_hf_data_and_metadata
from src.evaluation.evaluate_models import (
    GenerativeEmbeddingEvaluatorForClassification as CustomEvaluator,
    make_preprocess_logits_for_metrics_generative,
)

import src.constants as constants
CSTS = constants.ConstantsNamespace()
CLI_CFG: dict[str, dict] = {}
parser = argparse.ArgumentParser(description="Fine-tune a generative LLM on patient narratives.")
parser.add_argument("--config", "-c", type=str, default="configs/generative_training.yaml")

CLASSIFICATION_HORIZONS = [30, 60, 90]
CLASSIFICATION_FUPS = [0, 30, 60, 90, 180, 365]  # elements can also be list[int]
CLASSIFICATION_LABEL_KEYS = [
    # "infection_label_binary_any",
    "infection_label_binary_bacterial",
    # "infection_label_binary_viral",
    # "infection_label_binary_fungal",
    # "infection_label_categorical",
    # "infection_label_one_hot",
]

TODO: UPDATE THIS SCRIPT LIKE THE NEW WAY OF TRAIN_CLASSIFICATION.PY

def main():
    """
    Fine-tune models on different classification tasks
    """
    # Iterator of all possible model and task combinations
    tasks = itertools.product(
        [False],  # [False, True]   # training_data_filtered
        [False],  # [False, True]   # with finetuning or not (eval-prompting)
        CLASSIFICATION_LABEL_KEYS,  # which classification task is performed
        CLASSIFICATION_HORIZONS,    # horizon
        CLASSIFICATION_FUPS,        # follow-up
    )

    for is_train_data_filtered, eval_only, label_key, horizon, fup in tasks:
        print(
            f"Starting fine-tuning with: "
            f"is_train_data_filtered = {is_train_data_filtered}, label_key = {label_key}, "
            f"horizon = {horizon}, fup = {fup}, eval_only = {eval_only}"
        )
        finetune_generative_model(
            label_key=label_key,
            horizon=horizon,
            fup_valid=fup,  # fup_valid is always fup
            fup_train=fup if is_train_data_filtered else None,
            eval_only=eval_only,
        )


def finetune_generative_model(
    label_key: str,
    horizon: int,
    fup_valid: list[int]|int,
    fup_train: list[int]|int|None=None,
    eval_only: bool=False,
):
    """
    Fine-tune generative model on a specific infection prediction task
    """    
    # Load custom patient narrative dataset
    dataset, _, vocabs = load_hf_data_and_metadata(
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

    # Memory efficiency parameters
    ft_args = CLI_CFG["trainer"].copy()
    if eval_only:
        model.config.use_cache = False
    else:
        if ft_args.get("gradient_checkpointing", False):
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
    def format_prompt_completion(sample):
        prompt = f"{system_prompt}\n\n{sample[input_col]}\n\nResponse:"
        completion = str(sample[target_col])  # "0" or "1"
        return {"prompt": prompt, "completion": completion}

    # Map the dataset to create the "messages" column required by TRL
    dataset["train"] = dataset["train"].with_format("python").map(format_prompt_completion)
    dataset["validation"] = dataset["validation"].with_format("python").map(format_prompt_completion)
    # if eval_only:  # just to try
    #     dataset["validation"] = concatenate_datasets([dataset["train"], dataset["validation"]])
    #     dataset["train"] = dataset["train"].select(range(1))

    # Evaluation pipeline
    preprocess_func = make_preprocess_logits_for_metrics_generative(tokenizer)
    evaluator = CustomEvaluator(tokenizer=tokenizer, do_clustering=False)
    callbacks = [EarlyStoppingCallback(early_stopping_patience=5)]

    # Setup training configuration, with correct output directory
    train_str = str(fup_train) if isinstance(fup_train, int) else "-".join(map(str, fup_train or []))
    valid_str = str(fup_valid) if isinstance(fup_valid, int) else "-".join(map(str, fup_valid or []))
    task_subdir = f"hrz({horizon})_fut({train_str})_fuv({valid_str})"
    run_subdir = str(Path(f"nft({eval_only})") / task_subdir)
    ft_args["output_dir"] = str(Path(ft_args["output_dir"]) / run_subdir)
    ft_args["logging_dir"] = str(Path(ft_args["logging_dir"]) / run_subdir)
    sft_config = SFTConfig(**ft_args)  # assistant_only_loss passed here
    peft_config = LoraConfig(**CLI_CFG["peft"])

    # Re-initialize a wandb run within the same worspace
    use_wandb = CLI_CFG.get("trainer", {}).get("report_to") == "wandb"
    if use_wandb:
        workspace = Path(__file__).stem
        wandb.init(project=workspace, name=run_subdir, config=CLI_CFG, reinit=True)

    # Initialize generative fine-tuning trainer
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        args=sft_config,
        peft_config=peft_config,
        compute_metrics=evaluator,
        preprocess_logits_for_metrics=preprocess_func,
        callbacks=callbacks,
    )

    # Train and/or evaluate generative model
    if not eval_only:
        trainer.train()  # best model and trainer_state.json saved automatically
    else:
        metrics = trainer.evaluate()           # using hugginface weights directly
        trainer.log_metrics("eval", metrics)   # record metric results
        trainer.save_metrics("eval", metrics)  # creates eval_results.json
        trainer.save_state()                   # creates trainer_state.json

    # Reset wandb and clean up CUDA memory for the next run
    if use_wandb: wandb.finish()
    del dataset, model, tokenizer, trainer
    gc.collect()  # ensure deleted objects are collected by the garbage collector
    torch.cuda.empty_cache()  # free deleted local objects from CUDA memory 


if __name__ == "__main__":
    cli_args = parser.parse_args()
    with open(cli_args.config, 'r') as f:
        CLI_CFG = yaml.safe_load(f)
    main()