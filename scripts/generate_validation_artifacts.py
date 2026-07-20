import yaml
import json
import gc
import torch
import numpy as np
from pathlib import Path
from transformers import Trainer, TrainingArguments
from scipy.special import logit
from src.data.patient_dataset import load_hf_data_and_metadata
from src.model.patient_embedder import PatientEmbeddingModelFactory, PatientDataCollatorForClassification
from src.evaluation.evaluate_models import DiscriminativeEmbeddingEvaluatorForClassification as CustomEvaluator
from scripts.train_classic_ml import build_model_pipeline, load_combined_data
from scripts.script_utils import scan_all_fups, prepare_dataset_fup_dict, get_best_optuna_run

CONFIG_PATH = Path("configs/discriminative_training.yaml")
ML_CONFIG_PATH = Path("configs/discriminative_classic_ml.yaml")
RESULTS_DIR_TF = Path("results_final/transformer")
RESULTS_DIR_ML = Path("results_final/classic_ml")
SPLIT_STRATEGIES = ["random_split", "temporal_split", "center_split"]
CLASSIC_ML_MODELS = ["logistic_regression", "random_forest", "xgboost"]
TARGET_TASKS = ["infection_bacteria", "infection_virus"]


def main():
    if not CONFIG_PATH.exists() or not ML_CONFIG_PATH.exists():
        raise FileNotFoundError("[CRITICAL] Missing core model path configuration scripts folder mappings.")

    with open(CONFIG_PATH, "r") as f: config_tf = yaml.safe_load(f)
    with open(ML_CONFIG_PATH, "r") as f: config_ml = yaml.safe_load(f)

    mlm_masking_rules = config_tf["data_collator"]["mlm_masking_rules"]
    run_id = "-".join([f"{k[0]}{int(v * 100):02d}" for k, v in mlm_masking_rules.items()])
    train_data_augment = config_tf.get("train_data_augment", "all")

    for split_strategy in SPLIT_STRATEGIES:
        print(f"\n=======================================================")
        print(f"LAUNCHING VALIDATION PROCESSING ENGINE: {split_strategy.upper()}")
        print(f"=======================================================")

        data_split_path = Path(config_tf["data_dir"]) / split_strategy
        if not data_split_path.exists():
            raise FileNotFoundError(f"[CRITICAL] Expected data split path missing from storage map: {data_split_path}")

        all_possible_fups = scan_all_fups(data_split_path)
        
        # run_transformer_validation(config_tf, split_strategy, data_split_path, run_id, train_data_augment, all_possible_fups)
        run_classic_ml_validation(config_ml, split_strategy, all_possible_fups)

    print(f"\n[SUCCESS] Pipeline validation execution finished. System outputs synchronized.")


class PrefixAwareTrainer(Trainer):
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        if hasattr(self.compute_metrics, "current_prefix"):
            self.compute_metrics.current_prefix = metric_key_prefix
        return super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)


def format_subdir_name(horizons, valid_fups, train_data_augment):
    fmt_fn = lambda x: "-".join(f"{i:04d}" for i in sorted(([x] if isinstance(x, int) else x or [])))
    hrz_str = fmt_fn(horizons)
    fut_str = "all" if train_data_augment == "all" else fmt_fn(valid_fups)
    fuv_str = fmt_fn(valid_fups)
    return f"hrz({hrz_str})_fut({fut_str})_fuv({fuv_str})"


def run_transformer_validation(config, split_strategy, data_split_path, run_id, train_data_augment, all_possible_fups):
    print(f"\n>>> [TRANSFORMER] Evaluating {split_strategy} across all discovered FUP time-steps...")

    for task_key in TARGET_TASKS:
        task_specs = config["prediction_tasks"][task_key]
        valid_fups_original = task_specs["fups"]
        
        for horizons in task_specs["horizons"]:
            if isinstance(horizons, int):
                horizons = [horizons]

            task_subdir = format_subdir_name(horizons, valid_fups_original, train_data_augment)
            
            if "optuna" in RESULTS_DIR_TF.name:
                trial_name, pt_config = get_best_optuna_run(RESULTS_DIR_TF, split_strategy, task_key)
                target_task_dir = RESULTS_DIR_TF / split_strategy / task_key / trial_name / split_strategy / pt_config / "finetuning" / task_key / task_subdir
            else:
                target_task_dir = RESULTS_DIR_TF / split_strategy / run_id / "finetuning" / task_key / task_subdir
            
            if not target_task_dir.exists():
                continue

            checkpoint_dirs = sorted(list(target_task_dir.glob("checkpoint-*")), key=lambda p: int(p.name.split("-")[-1]))
            if not checkpoint_dirs:
                raise FileNotFoundError(f"[CRITICAL] No training checkpoints available inside {target_task_dir}")
            checkpoint_dir = str(checkpoint_dirs[-1])

            print(f"  -> Processing Task: {task_key} | Horizon: {horizons} | Path: {task_subdir}")

            label_keys = [f"label_{task_key}_{h:04d}d" for h in horizons]
            
            dataset, _, vocab = load_hf_data_and_metadata(
                data_dir=data_split_path, fup_train=valid_fups_original, fup_valid=all_possible_fups, fup_test=all_possible_fups,
                label_keys=label_keys, target_undersampling_ratio=None,
                time_mapping=config["data_collator"]["time_mapping"], eav_mappings=config["data_collator"]["eav_mappings"],
            )

            validation_ds = dataset["validation"]
            validation_ds = validation_ds.add_column("split", ["validation"] * len(validation_ds))
            
            label_matrix = np.stack([validation_ds[k] for k in label_keys], axis=1)
            keep_mask = (label_matrix != -100).any(axis=1)
            validation_ds = validation_ds.select(np.where(keep_mask)[0])
            
            # Prepare separate dictionary cuts without holding onto the global "all" dataset view
            eval_datasets = prepare_dataset_fup_dict(validation_ds, all_possible_fups)
            if "all" in eval_datasets:
                del eval_datasets["all"] # Memory Optimization: Remove the mega-view immediately

            model_cfg = config["model"].copy()
            model_cfg["pretrained_dir"] = checkpoint_dir
            model_cfg["embedding_layer_config"]["vocab_size"] = len(vocab)
            model_cfg["reset_weights"] = False
            model_cfg["enforce_monotonicity"] = config["finetuner"].get("enforce_monotonicity", True)
            model_cfg["task"] = "classification"
            model_cfg["model_args"]["num_labels"] = len(label_keys)
            model_cfg["model_args"]["problem_type"] = "multi_label_classification"

            for key in ["dtype", "torch_dtype"]:
                if isinstance(model_cfg["model_args"].get(key), str):
                    model_cfg["model_args"][key] = getattr(torch, model_cfg["model_args"][key])

            model = PatientEmbeddingModelFactory.from_pretrained(**model_cfg)
            ft_collator = PatientDataCollatorForClassification(**config["data_collator"], label_keys=label_keys, max_position_embeddings=model.config.max_position_embeddings)
            evaluator = CustomEvaluator(do_clustering=False, label_names=label_keys, enforce_monotonicity=model_cfg["enforce_monotonicity"], early_stopping_metric=config["early_stopping_metric"])

            trainer = PrefixAwareTrainer(
                model=model, 
                args=TrainingArguments(
                    output_dir=str(target_task_dir), 
                    per_device_eval_batch_size=config["finetuner"].get("per_device_eval_batch_size", 32), 
                    bf16=config["finetuner"].get("bf16", True), 
                    remove_unused_columns=False,
                    eval_accumulation_steps=100,
                ), 
                data_collator=ft_collator, 
                compute_metrics=evaluator
            )

            final_metrics = {}
            all_val_predictions = {}

            # Sequentially process and clear memory slices one by one
            for fup_key, fup_val_dataset in eval_datasets.items():
                prefix = f"validation_{fup_key}"
                final_metrics.update(trainer.evaluate(fup_val_dataset, metric_key_prefix=prefix))
                
                # Extract predictions for the target subset
                for key, array in trainer.compute_metrics.saved_labels_and_probs.items():
                    all_val_predictions[f"{prefix}_{key}"] = array
                
                # Clear evaluation garbage profiles explicitly between time steps
                trainer.compute_metrics.saved_labels_and_probs = None
                torch.cuda.empty_cache()
                gc.collect()

            # Save artifacts matching main experiment schemas
            np.savez_compressed(target_task_dir / "validation_probs.npz", **all_val_predictions)
            with open(target_task_dir / "validation_results.json", "w") as f:
                json.dump(final_metrics, f, indent=4)

            del model, trainer, eval_datasets, validation_ds
            gc.collect()
            torch.cuda.empty_cache()


def run_classic_ml_validation(ml_config, split_strategy, all_possible_fups):
    print(f"\n>>> [CLASSIC ML] Evaluating baselines across all discovered FUP time-steps...")
    data_root = Path(ml_config['data_dir']) / split_strategy
    if not data_root.exists():
        raise FileNotFoundError(f"[CRITICAL] Missing ML split dataset space path: {data_root}")

    train_data_augment = ml_config.get("train_data_augment", "all")
    for model_type in CLASSIC_ML_MODELS:
        for task_key in TARGET_TASKS:
            task_specs = ml_config["prediction_tasks"][task_key]
            valid_fups_original = task_specs["fups"]
            
            if train_data_augment == "all":
                train_fups = all_possible_fups
            else:
                train_fups = valid_fups_original

            for horizon in task_specs["horizons"]:
                task_subdir = format_subdir_name(horizon, valid_fups_original, train_data_augment)
                target_task_dir = RESULTS_DIR_ML / split_strategy / model_type / task_key / task_subdir
                
                if not target_task_dir.exists():
                    raise FileNotFoundError(f"[CRITICAL] Expected ML experiment path missing: {target_task_dir}")

                best_params_path = target_task_dir / "best_params.json"
                if not best_params_path.exists():
                    raise FileNotFoundError(f"[CRITICAL] Parameter definitions array file missing: {best_params_path}")

                with open(best_params_path, "r") as f:
                    final_params = json.load(f)

                print(f"  -> Re-fitting Baseline: {model_type} | Task: {task_key} | Horizon: {horizon}d")

                label_key = f"label_{task_key}_{horizon:04d}d"
                ignore_cols = set(ml_config['ignore_columns'] + [label_key])

                X_train, y_train, features = load_combined_data(data_root, train_fups, "train.parquet", label_key, ignore_cols)

                target_us_ratio = None
                if ml_config.get('target_undersampling_ratio') is not None:
                    target_us_ratio = 1.0 / ml_config['target_undersampling_ratio']

                pipeline = build_model_pipeline(X_train, y_train, model_type, final_params, target_us_ratio)
                pipeline.fit(X_train, y_train)

                evaluator = CustomEvaluator(do_clustering=False, label_names=[label_key], early_stopping_metric="roc_auc")
                
                # Separate containers for Val and Test metadata
                final_val_metrics = {}
                all_val_predictions = {}
                all_test_predictions = {}

                def evaluate_pipeline_block(X_eval, y_eval, prefix, storage_dict, metrics_dict=None):
                    probs = pipeline.predict_proba(X_eval)[:, 1]
                    logits = logit(np.clip(probs, 1e-6, 1 - 1e-6))[:, None]
                    y_tem = y_eval[:, None]
                    
                    from transformers.trainer_utils import EvalPrediction
                    evaluator.current_prefix = prefix
                    metrics = evaluator(EvalPrediction(predictions=logits, label_ids=y_tem))
                    
                    if metrics_dict is not None:
                        metrics_dict.update({f"{prefix}_{k}": v for k, v in metrics.items()})
                    
                    for key, array in evaluator.saved_labels_and_probs.items():
                        storage_dict[f"{prefix}_{key}"] = array

                # Sequentially process validation and test data over every FUP folder
                for fup in all_possible_fups:
                    # Validation evaluation pass
                    X_val_fup, y_val_fup, _ = load_combined_data(data_root, [fup], "validation.parquet", label_key, ignore_cols, enforced_features=features)
                    if X_val_fup is not None and len(X_val_fup) > 0:
                        evaluate_pipeline_block(X_val_fup, y_val_fup, prefix=f"validation_fup_{fup:04d}", storage_dict=all_val_predictions, metrics_dict=final_val_metrics)
                    
                    # Test evaluation pass
                    X_test_fup, y_test_fup, _ = load_combined_data(data_root, [fup], "test.parquet", label_key, ignore_cols, enforced_features=features)
                    if X_test_fup is not None and len(X_test_fup) > 0:
                        evaluate_pipeline_block(X_test_fup, y_test_fup, prefix=f"test_fup_{fup:04d}", storage_dict=all_test_predictions, metrics_dict=None)
                        
                    # Proactive memory cleanup
                    evaluator.saved_labels_and_probs = None
                    gc.collect()

                # Save compressed artifacts for both splits separately
                np.savez_compressed(target_task_dir / "val_predictions.npz", **all_val_predictions)
                np.savez_compressed(target_task_dir / "test_predictions.npz", **all_test_predictions)
                with open(target_task_dir / "validation_results.json", "w") as f:
                    json.dump(final_val_metrics, f, indent=4)
                    

if __name__ == "__main__":
    main()