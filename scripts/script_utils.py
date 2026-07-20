import re
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datasets import Dataset, load_from_disk
from scipy.special import logit
from sklearn.linear_model import LogisticRegression


def scan_all_fups(data_dir: Path) -> list[int]:
    """Finds all available follow-up folders (fup_XXXX) in the data directory."""
    fups = []
    if not data_dir.exists():
        return fups
    for path in data_dir.iterdir():
        if path.is_dir() and path.name.startswith("fup_"):
            try:
                # Strips trailing 'd' or similar suffix formatting characters cleanly
                val = int(path.name.split("_")[-1].replace("d", ""))
                fups.append(val)
            except ValueError:
                continue  # Skip unparseable directory signatures (e.g. fup_None)
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
    """Locates the final model training checkpoint inside structured execution logs."""
    task_dir = base_dir / "finetuning" / task_key
    if not task_dir.exists():
        raise FileNotFoundError(f"Task directory missing: {task_dir}")
    
    h_str = f"{horizon:04d}"
    pattern = re.compile(rf"hrz\(([^)]*\b{h_str}\b[^)]*)\)")
    candidates = [p for p in task_dir.iterdir() if p.is_dir() and pattern.search(p.name)]
    if not candidates:
        raise FileNotFoundError(f"No run found matching horizon configuration {h_str} inside {task_dir}")
    
    # Use the first directory matching your targeted horizon group assignment
    run_dir = candidates[0]
    checkpoint_dirs = sorted(
        list(run_dir.glob("checkpoint-*")),
        key=lambda p: int(p.name.split("-")[-1]),
    )
    if not checkpoint_dirs:
        raise FileNotFoundError(f"No valid checkpoints discovered inside structural path: {run_dir}")
    
    return checkpoint_dirs[-1]  # Return the last (highest step) checkpoint saved


def extract_horizons_from_path(checkpoint_path: Path) -> list[int]:
    """Extracts targeted validation prediction horizons from a checkpoint parent directory name."""
    run_dir = checkpoint_path
    while "hrz(" not in run_dir.name and run_dir.parent != run_dir:
        run_dir = run_dir.parent
    
    match = re.search(r"hrz\(([\d-]+)\)", run_dir.name)
    if not match:
        raise ValueError(f"Could not parse horizon keys from path layout string: {run_dir.name}")
    
    return [int(h) for h in match.group(1).split("-")] 


def calibrate_array_pair(
    y_val_true: np.ndarray, 
    y_val_prob: np.ndarray, 
    y_test_prob: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fits a regularized Platt-Scaling (Logistic Regression) model on validation outputs
    and transforms both validation and test probabilities.
    """
    if len(y_val_true) == 0 or len(np.unique(y_val_true)) < 2:
        return y_val_prob, y_test_prob

    # Clip probabilities to protect against logit numerical instability
    val_probs = np.clip(y_val_prob, 1e-7, 1.0 - 1e-7)
    test_probs = np.clip(y_test_prob, 1e-7, 1.0 - 1e-7)

    X_val_logits = logit(val_probs).reshape(-1, 1)
    X_test_logits = logit(test_probs).reshape(-1, 1)

    # L2 Regularized Platt Scaling (C=1.0 prevents parameter explosion on imbalanced labels)
    calibrator = LogisticRegression(C=1.0, penalty="l2", solver="lbfgs", random_state=42)
    calibrator.fit(X_val_logits, y_val_true)

    if hasattr(calibrator, "classes_"):
        pos_indices = np.where(calibrator.classes_ == 1)[0]
        if len(pos_indices) > 0:
            pos_idx = pos_indices[0]
            cal_val = calibrator.predict_proba(X_val_logits)[:, pos_idx]
            cal_test = calibrator.predict_proba(X_test_logits)[:, pos_idx]
            return cal_val, cal_test

    return y_val_prob, y_test_prob


def calibrate_dataframe_pair(
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    prob_col: str = "y_prob",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convenience wrapper around calibrate_array_pair for pandas DataFrames.
    """
    if df_val.empty or df_test.empty:
        return df_val, df_test

    df_val_out = df_val.copy()
    df_test_out = df_test.copy()

    cal_val, cal_test = calibrate_array_pair(
        y_val_true=df_val["y_true"].values,
        y_val_prob=df_val[prob_col].values,
        y_test_prob=df_test[prob_col].values,
    )

    df_val_out[prob_col] = cal_val
    df_test_out[prob_col] = cal_test

    return df_val_out, df_test_out


def get_best_optuna_run(results_dir: Path, split_type: str, task_key: str) -> tuple[str, str]:
    """Parses the Optuna journal log to find the best completed trial folder."""
    task_dir = results_dir / split_type / task_key
    log_files = list(task_dir.glob("*journal.log"))
    
    if not log_files:
        raise FileNotFoundError(f"Missing Optuna storage log inside target task space: {task_dir}")
        
    trial_scores = {}
    trial_params = {}
    
    with open(log_files[0], "r") as f:
        for line in f:
            try:
                entry = json.loads(line.strip())
                op_code = entry.get("op_code")
                trial_id = entry.get("trial_id")
                
                if trial_id is None:
                    continue
                    
                if trial_id not in trial_params:
                    trial_params[trial_id] = {}

                # op_code 5: Parameter tracking updates
                if op_code == 5:
                    param_name = entry.get("param_name")
                    if param_name in ["mask_ent", "mask_att", "mask_val"]:
                        trial_params[trial_id][param_name] = entry.get("param_value_internal", 0.0)
                        
                # op_code 6: Trial finalized lifecycle states (state 1 == COMPLETE)
                elif op_code == 6 and entry.get("state") == 1:
                    val = entry.get("values", [-float("inf")])[0]
                    if val is not None:
                        trial_scores[trial_id] = val
                        
            except json.JSONDecodeError:
                continue

    if not trial_scores:
        raise ValueError(f"Log parsing error: No completed trials found inside storage file {log_files[0]}")

    best_trial_id = max(trial_scores, key=trial_scores.get)
    best_params = trial_params.get(best_trial_id, {})

    e_val = int(round(best_params.get("mask_ent", 0.0) * 100))
    a_val = int(round(best_params.get("mask_att", 0.0) * 100))
    v_val = int(round(best_params.get("mask_val", 0.0) * 100))
    
    pt_config = f"e{e_val:02d}-a{a_val:02d}-v{v_val:02d}"
    return f"trial_{best_trial_id:03d}", pt_config


def extract_flat_split_csv(
    data_raw_path: Path,
    raw_pool: dict,
    model: str,
    task: str,
    split: str,
    horizon: int,
    dataset_split: str,
) -> pd.DataFrame:
    """
    Unpacks nested pool records and reconstructs tracking keys from disk.
    Guarantees index alignment across multi-label targets.
    """
    # Defensive slicing catch bounds
    if (split not in raw_pool or 
        model not in raw_pool[split] or 
        task not in raw_pool[split][model] or 
        horizon not in raw_pool[split][model][task]):
        return pd.DataFrame()
        
    fup_dict = raw_pool[split][model][task][horizon]
    flat_records = []
    
    for fup_day, records in fup_dict.items():
        y_true = np.asarray(records["labels"]).flatten()
        y_prob = np.asarray(records.get("probs_cal", records["probs"])).flatten()
        
        raw_fup_dir = data_raw_path / split / f"fup_{fup_day:04d}"
        if not raw_fup_dir.exists():
            raw_fup_dir = data_raw_path / split / f"fup_{fup_day:04d}d"
        if not raw_fup_dir.exists():
            continue
            
        try:
            raw_ds = load_from_disk(str(raw_fup_dir))[dataset_split]
            raw_keys = raw_ds["patientkey"]
            
            # Reconstruct keep_mask used during training/validation filtering steps
            target_col = f"label_{task}_{horizon:04d}d"
            if target_col in raw_ds.column_names:
                # If checking a single specific target label column directly
                keep_mask = np.array(raw_ds[target_col]) != -100
            else:
                # Multi-label task fallback matching full sequence array allocations
                hrz_cols = [c for c in raw_ds.column_names if c.startswith(f"label_{task}_") and c.endswith("d")]
                label_matrix = np.stack([raw_ds[c] for c in hrz_cols], axis=1)
                keep_mask = (label_matrix != -100).any(axis=1)
        except Exception:
            continue
            
        # Core Index Alignment Fix:
        # Filter patient keys down *first* to match true valid rows passed to the model
        valid_indices = np.where(keep_mask)[0]
        
        # Guard check to avoid length crashes if files are corrupted or cropped
        eval_len = min(len(valid_indices), len(y_prob), len(y_true))
        
        for idx in range(eval_len):
            row_ds_idx = valid_indices[idx]
            current_target_label = int(y_true[idx])
            
            # Skip records if the specific label array holds censored targets
            if current_target_label == -100:
                continue
                
            flat_records.append({
                "patientkey": raw_keys[row_ds_idx],
                "time_step": fup_day,
                "y_true": current_target_label,
                "y_prob": float(y_prob[idx])
            })
            
    if not flat_records:
        return pd.DataFrame()
        
    df = pd.DataFrame(flat_records)
    return df.sort_values(by=["patientkey", "time_step"]).reset_index(drop=True)    


def load_all_raw_predictions(
    transformer_base_dir: Path, 
    classic_ml_base_dir: Path, 
    from_optuna: bool, 
    split_types: list, 
    tasks: list, 
    classic_ml_models: list, 
    dataset_split: str = "test"
) -> dict:
    """
    Crawls output directories and builds a structured predictions dictionary pool.
    Isolates discrete follow-up tracking slices to prevent multi-label overwrites.
    """
    pool = {split: {} for split in split_types}
    prefix = f"{dataset_split}_" if dataset_split == "validation" else "test_"
    fup_pattern = re.compile(rf"^{prefix}fup_(\d+)_labels$")
    file_name = "validation_probs.npz" if dataset_split == "validation" else "test_probs.npz"
    ml_file_name = "val_predictions.npz" if dataset_split == "validation" else "test_predictions.npz"
    
    # 1. Gather Transformer Predictions
    for split in split_types:
        pool[split]["transformer"] = {t: {} for t in tasks}
            
        for task in tasks:
            if from_optuna:
                try:
                    trial_folder, pt_config = get_best_optuna_run(transformer_base_dir, split, task)
                    run_base_path = transformer_base_dir / split / task / trial_folder / split / pt_config / "finetuning" / task
                except Exception:
                    continue
            else:
                pt_config = "e00-a15-v60"
                run_base_path = transformer_base_dir / split / pt_config / "finetuning" / task
                
            if not run_base_path.exists():
                continue
                
            for hrz_folder in run_base_path.iterdir():
                if not hrz_folder.is_dir() or not hrz_folder.name.startswith("hrz("):
                    continue
                    
                hrz_match = re.search(r"hrz\(([\d-]+)\)", hrz_folder.name)
                if not hrz_match:
                    continue
                horizons = [int(h) for h in hrz_match.group(1).split("-")]
                
                npz_path = hrz_folder / file_name
                if not npz_path.exists():
                    continue
                    
                npz_data = np.load(npz_path, allow_pickle=True)
                
                for key in npz_data.files:
                    match = fup_pattern.match(key)
                    if not match:
                        continue
                    fup_day = int(match.group(1))
                    
                    fup_label_key = f"{prefix}fup_{fup_day:04d}_labels"
                    fup_probs_key = f"{prefix}fup_{fup_day:04d}_probs"
                    fup_probs_cal_key = f"{prefix}fup_{fup_day:04d}_probs_cal"
                    
                    for h in horizons:
                        if h not in pool[split]["transformer"][task]:
                            pool[split]["transformer"][task][h] = {}
                            
                        h_idx = horizons.index(h)
                        
                        record = {
                            "labels": npz_data[fup_label_key][:, h_idx],
                            "probs": npz_data[fup_probs_key][:, h_idx],
                            "row_mask_mode": "any_declared_horizon",
                            "horizons": horizons,
                            "source_archive": str(npz_path)
                        }
                        if fup_probs_cal_key in npz_data.files:
                            record["probs_cal"] = npz_data[fup_probs_cal_key][:, h_idx]
                            
                        pool[split]["transformer"][task][h][fup_day] = record

    # 2. Gather Classic ML Baseline Predictions
    for split in split_types:
        for model in classic_ml_models:
            pool[split][model] = {t: {} for t in tasks}
                
            for task in tasks:
                base_task_dir = classic_ml_base_dir / split / model / task
                if not base_task_dir.exists():
                    continue
                    
                for hrz_folder in base_task_dir.iterdir():
                    if not hrz_folder.is_dir() or not hrz_folder.name.startswith("hrz("):
                        continue
                        
                    hrz_match = re.search(r"hrz\(([\d-]+)\)", hrz_folder.name)
                    if not hrz_match:
                        continue
                    horizon = int(hrz_match.group(1))
                    
                    if horizon not in pool[split][model][task]:
                        pool[split][model][task][horizon] = {}
                        
                    npz_path = hrz_folder / ml_file_name
                    if not npz_path.exists():
                        continue
                        
                    npz_data = np.load(npz_path, allow_pickle=True)
                    
                    for key in npz_data.files:
                        match = fup_pattern.match(key)
                        if not match:
                            continue
                        fup_day = int(match.group(1))
                        
                        record = {
                            "labels": npz_data[f"{prefix}fup_{fup_day:04d}_labels"].flatten(),
                            "probs": npz_data[f"{prefix}fup_{fup_day:04d}_probs"].flatten(),
                            "row_mask_mode": "target_horizon_only",
                            "source_archive": str(npz_path)
                        }
                        fup_probs_cal_key = f"{prefix}fup_{fup_day:04d}_probs_cal"
                        if fup_probs_cal_key in npz_data.files:
                            record["probs_cal"] = npz_data[fup_probs_cal_key].flatten()
                            
                        pool[split][model][task][horizon][fup_day] = record

    return pool