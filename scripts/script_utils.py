import re
import json
import numpy as np
from pathlib import Path
from datasets import Dataset


def scan_all_fups(data_dir: Path) -> list[int]:
    """
    Find all available follow-up folders (fup_XXXX) in the data directory
    """
    fups = []
    for path in data_dir.iterdir():
        if path.is_dir() and path.name.startswith("fup_"):
            try:
                # Extract integer from "fup_0090" -> 90
                val = int(path.name.split("_")[-1])
                fups.append(val)
            except ValueError:
                continue  # skip fup_None or malformed folders
    
    return sorted(fups)


def prepare_dataset_fup_dict(dataset: Dataset, fup_list: list[int]):
    """
    Creates a dictionary of datasets for different follow-up periods.
    """
    out_dict = {"all": dataset}
    fup_array = np.array(dataset["fup"])
    for fup in fup_list:
        indices = np.where(fup_array == fup)[0]
        if len(indices) > 0:
            subset = dataset.select(indices)  # dataset view
            out_dict[f"fup_{fup:04d}"] = subset
            
    return out_dict


def find_best_checkpoint(base_dir: Path, task_key: str, horizon: int) -> Path:
    """
    ...
    """
    task_dir = base_dir / "finetuning" / task_key
    if not task_dir.exists(): raise FileNotFoundError(f"Task directory not found: {task_dir}")
    
    h_str = f"{horizon:04d}"
    pattern = re.compile(rf"hrz\(([^)]*\b{h_str}\b[^)]*)\)")
    candidates = [p for p in task_dir.iterdir() if p.is_dir() and pattern.search(p.name)]
    if not candidates: raise FileNotFoundError(f"No run found for horizon {h_str} inside hrz() in {task_dir}")
    
    run_dir = candidates[0]
    checkpoint_dirs = sorted(
        list(run_dir.glob("checkpoint-*")),
        key=lambda p: int(p.name.split("-")[-1]),
    )
    if not checkpoint_dirs: raise FileNotFoundError(f"No checkpoints found in {run_dir}")
    
    return checkpoint_dirs[0]


def extract_horizons_from_path(checkpoint_path: Path) -> list[int]:
    """
    ...
    """
    run_dir = checkpoint_path
    while "hrz(" not in run_dir.name and run_dir.parent != run_dir:
        run_dir = run_dir.parent
    
    match = re.search(r"hrz\(([\d-]+)\)", run_dir.name)
    if not match:
        raise ValueError(f"Could not extract horizons from path: {run_dir.name}")
    
    return [int(h) for h in match.group(1).split("-")] 


def get_best_optuna_run(
    results_dir: Path,
    split_type: str,
    task_key: str,
) -> tuple[str, str]:
    """
    Parses the Optuna journal log to automatically find the best 
    completed trial folder and its pretraining configuration string.
    Works even if the optimization study was interrupted.
    """
    task_dir = results_dir / split_type / task_key
    log_files = list(task_dir.glob("*journal.log"))
    
    if not log_files:
        raise FileNotFoundError(f"Missing log file in {task_dir}")
        
    trial_scores = {}
    trial_params = {}
    
    # Read the log file line by line
    with open(log_files[0], "r") as f:
        for line in f:
            try:
                entry = json.loads(line.strip())
                op_code = entry.get("op_code")
                trial_id = entry.get("trial_id")
                
                if trial_id is None:
                    continue
                    
                # Initialize nested dict if new trial
                if trial_id not in trial_params:
                    trial_params[trial_id] = {}

                # op_code 5: A parameter was set for this trial
                if op_code == 5:
                    param_name = entry.get("param_name")
                    if param_name in ["mask_ent", "mask_att", "mask_val"]:
                        trial_params[trial_id][param_name] = entry.get("param_value_internal", 0.0)
                        
                # op_code 6: Trial finished (state 1 = COMPLETE)
                elif op_code == 6 and entry.get("state") == 1:
                    val = entry.get("values", [-float("inf")])[0]
                    if val is not None:
                        trial_scores[trial_id] = val
                        
            except json.JSONDecodeError:
                continue

    if not trial_scores:
        raise ValueError(f"No completed trials found in {log_files[0]}")

    # Find the trial ID with the highest recorded score
    best_trial_id = max(trial_scores, key=trial_scores.get)
    best_params = trial_params.get(best_trial_id, {})

    # Extract the masking parameters and format the config string
    # Using round() to handle floating point imprecision (e.g. 0.15000000002 -> 15)
    e_val = int(round(best_params.get("mask_ent", 0.0) * 100))
    a_val = int(round(best_params.get("mask_att", 0.0) * 100))
    v_val = int(round(best_params.get("mask_val", 0.0) * 100))
    
    pt_config = f"e{e_val:02d}-a{a_val:02d}-v{v_val:02d}"
    
    return f"trial_{best_trial_id:03d}", pt_config