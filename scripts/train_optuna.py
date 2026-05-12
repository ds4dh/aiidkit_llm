import os
import sys
import yaml
import json
import shutil
import argparse
import optuna
import threading
import subprocess
import queue
from pathlib import Path
from scipy.stats import gmean
from optuna.samplers import TPESampler
from optuna.storages import JournalStorage, JournalFileStorage

# Retrieve configuration file (useful to differentiate simultaneous runs)
parser = argparse.ArgumentParser(description="Perform hyper-parameter optimization for pretraining and finetuning.")
parser.add_argument("--debug", "-d", action="store_true", help="Performa a small run with a few GPUs.")
parser.add_argument("--config", "-c", type=str, default="configs/discriminative_training.yaml")
cli_args = parser.parse_args()
DEBUG = cli_args.debug
CONFIG_PATH = cli_args.config
with open(cli_args.config, 'r') as f:
    CLI_CFG = yaml.safe_load(f)

# Extract run configuration
RESULTS_DIR = CLI_CFG["result_dir"]
N_TRIALS = CLI_CFG["optuna"]["num_trials"] if not DEBUG else CLI_CFG["optuna"]["num_trials_debug"]
DB_NAME = CLI_CFG["optuna"]["db_name"]
STUDY_NAME = CLI_CFG["optuna"]["study_name"]
METRIC_TO_OPTIMIZE = CLI_CFG["optuna"]["metric_to_optimize"]  # e.g., "val_all_roc_auc_label_infection_bacteria"
COMMON_TASK_PREFIX = os.path.commonprefix(list(CLI_CFG["prediction_tasks"].keys()))
if len(COMMON_TASK_PREFIX) > 0:
    METRIC_TO_OPTIMIZE = f"{METRIC_TO_OPTIMIZE}_label_{COMMON_TASK_PREFIX}"

# Lock to identify the current best model
BEST_MODEL_LOCK = threading.Lock()
BEST_MODEL_INFO = {"score": -float("inf"), "path": None}

# Create a thread-safe queue containing IDs "0", "1", ..., "#GPUS"
NUM_GPUS = len(os.environ.get("CUDA_VISIBLE_DEVICES").split(","))
GPU_QUEUE = queue.Queue()
for i in range(NUM_GPUS):
    GPU_QUEUE.put(str(i))


def main():
    """
    Runs the entire optuna study
    """
    # Initialize optuna study
    os.makedirs(RESULTS_DIR, exist_ok=True)
    log_file = os.path.join(RESULTS_DIR, f"{STUDY_NAME}_journal.log")
    storage = JournalStorage(JournalFileStorage(log_file))
    sampler = TPESampler(seed=None)
    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=storage,
        direction="maximize",
        load_if_exists=True,
        sampler=sampler,
    )
    
    # If resuming, prevents deleting a previous historical best
    if len(study.trials) > 0:
        try:
            BEST_MODEL_INFO["score"] = study.best_value
        except ValueError: 
            pass
    
    print(f"Starting optimization with {N_TRIALS} trials.")
    print(f"Results will be stored in {DB_NAME}")
    print(f"Number of GPUs available: {NUM_GPUS}")
    
    study.optimize(objective, n_trials=N_TRIALS, n_jobs=NUM_GPUS)
    
    print("\n" + "="*50)
    print("Optimization complete")
    print("="*50)
    print("Best parameters:")
    print(json.dumps(study.best_params, indent=4))
    print(f"Best aggregated metric: {study.best_value:.4f}")
    
    # Save best config to a file for easy reproduction
    best_config_path = os.path.join(RESULTS_DIR, "best_optimized_config.json")
    with open(best_config_path, "w") as f:
        json.dump(study.best_params, f, indent=4)
    print(f"Best parameters saved to {best_config_path}")


def get_run_id(masking_rules: dict) -> str:
    """
    Used to locate the results folder, like in train_mlm/train_classification
    """
    return "-".join([f"{k[0]}{int(v * 100):02d}" for k, v in masking_rules.items()])


def run_subprocess(script_path, overrides, env):
    """
    Runs a training script in a separate process to ensure memory cleanup
    """
    overrides_json = json.dumps(overrides)
    cmd = [
        sys.executable, str(script_path),
        "--config", CONFIG_PATH,
        "--overrides", overrides_json,
        "--silent",  # to disable wandb logging during optimization
    ]
    
    # Suppress stdout for clean optuna progress bar, but print stderr if it fails
    if not DEBUG:
        result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    else:
        result = subprocess.run(cmd, env=env)
    
    if result.returncode != 0:
        gpu_id = env.get("CUDA_VISIBLE_DEVICES", "Unknown")
        print(f"\n[!] Error running {script_path} on GPU {gpu_id}")
        print(result.stderr)
        raise RuntimeError(f"Script {script_path} failed.")


def aggregate_metrics(run_path):
    """
    Crawls results from a specific trial folder
    """
    finetuning_path = run_path / "finetuning"    
    if not finetuning_path.exists():
        print(f"Path not found: {finetuning_path}")
        return 0.0

    scores = []
    
    # Recursively find all test_results.json files
    for result_file in finetuning_path.rglob("test_results.json"):
        try:
            with open(result_file, 'r') as f:
                data = json.load(f)
            
            # Scan for all keys starting with the optimized metric
            for key, value in data.items():
                if key.startswith(METRIC_TO_OPTIMIZE):
                    if value != -1:  # -1 indicates impossible (e.g. only 1 class present)
                        scores.append(value)

        except Exception as e:
            print(f"Error reading {result_file}: {e}")

    if not scores:
        print(f"   [Warning] No valid ROC AUC scores found")
        return 0.0

    # Geometric mean aggregates performance across different tasks robustly
    return gmean(scores)


def objective(trial: optuna.Trial):
    """
    Compute objective based on full pre-training and finetuning run
    """
    # Make only one specific GPU visible to this trial
    gpu_id = GPU_QUEUE.get()
    try:
        trial_env = os.environ.copy()
        trial_env["CUDA_VISIBLE_DEVICES"] = gpu_id
        trial_dir = Path(RESULTS_DIR) / f"trial_{trial.number:03d}"

        # Task hyperparameters (making sure not everything is zero)
        p_entity = trial.suggest_float("mask_ent", 0.00, 0.25, step=0.05)
        p_attr   = trial.suggest_float("mask_att", 0.00, 0.50, step=0.05)
        p_value  = trial.suggest_float("mask_val", 0.05, 0.75, step=0.05)
        masking_rules = {"entity_id": p_entity, "attribute_id": p_attr, "value_id": p_value}

        # Learning hyperparameters
        lr_pre  = trial.suggest_float("lr_pre",  2e-5, 2e-4, log=True)
        lr_fine = trial.suggest_float("lr_fine", 1e-5, 1e-4, log=True)

        # Model architecture / regularization
        # Could add layer number / embedding dimension parameters?
        dropout_att = trial.suggest_float("dropout_att", 0.0, 0.2, step=0.05)
        dropout_mlp = trial.suggest_float("dropout_mlp", 0.0, 0.2, step=0.05)
        dropout_cls = trial.suggest_float("dropout_cls", 0.0, 0.5, step=0.10)

        # Construct overrides dictionary (structure must match the YAML hierarchy)
        overrides = {
            "result_dir": str(trial_dir),
            "data_collator": {
                "mlm_masking_rules": masking_rules,
            },
            "pretrainer": {
                "learning_rate": lr_pre,
                "max_steps": 100000 if not DEBUG else 100,
                "seed": trial.number,
            },
            "finetuner": {
                # "loss_name": loss_name, -> fixed to ce, for optimally calibrated model
                "learning_rate": lr_fine,
                "max_steps": 10000 if not DEBUG else 100,
                "seed": trial.number,  # each trial is run on an independent process
            },
            "model": {
                "config_args": {
                    "dropout_att": dropout_att,
                    "dropout_mlp": dropout_mlp,
                    "dropout_cls": dropout_cls,
                },
            }
        }

        # Execution (pretraining, finetuning), passing trial_env to run_subprocess
        try:
            print(f"\n[Trial {trial.number}] GPU {gpu_id}: Running pretraining...")
            run_subprocess("scripts/train_mlm.py", overrides, env=trial_env)
            
            print(f"[Trial {trial.number}] GPU {gpu_id}: Running finetuning...")
            run_subprocess("scripts/train_classification.py", overrides, env=trial_env)
            
        except RuntimeError as e:
            print(f"Trial failed: {e}")
            return 0.0

        # Evaluation
        run_id = get_run_id(masking_rules)
        full_run_path = trial_dir / CLI_CFG["data_split_type"] / run_id
        score = aggregate_metrics(full_run_path)
        
        # Cleanup
        path_to_delete = None
        with BEST_MODEL_LOCK:
            
            # If trial is the new global best
            if score > BEST_MODEL_INFO["score"]:
                
                # Mark previous best for deletion
                if BEST_MODEL_INFO["path"] is not None:
                    path_to_delete = BEST_MODEL_INFO["path"]
                
                # Update Best Info
                BEST_MODEL_INFO["score"] = score
                BEST_MODEL_INFO["path"] = full_run_path
            
            # If trial is not the best
            else:
                path_to_delete = full_run_path

        # Perform the actual deletion outside the lock (I/O is slow)
        if path_to_delete:
            cleanup_checkpoints(path_to_delete)

        return score
    
    finally:
        GPU_QUEUE.put(gpu_id)


def cleanup_checkpoints(run_path):
    """
    Deletes heavy checkpoints in the specified folder
    """
    if run_path and run_path.exists():
        print(f"   -> Cleaning up: {run_path}")
        for checkpoint_dir in run_path.rglob("checkpoint-*"):
            if checkpoint_dir.is_dir():
                try: shutil.rmtree(checkpoint_dir)
                except Exception as e: print(f"Failed to delete {checkpoint_dir}: {e}")


if __name__ == "__main__":
    main()