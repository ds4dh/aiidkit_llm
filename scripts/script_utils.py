import re
import json
import src.utils  # activates UTF-8 log tee when AIIDKIT_LOG_FILE is set
import numpy as np
import pandas as pd
from pathlib import Path
from datasets import Dataset
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from concurrent.futures import ThreadPoolExecutor


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


def save_predictions_table(
    all_predictions: dict,
    split_label: str,
    horizons: list[int],
    output_dir: Path,
    csv_name: str | None = None,
    task_keys: list[str] | None = None,
) -> pd.DataFrame | None:
    """
    Flatten the per-FUP prediction arrays saved by the final test pass into a
    tidy (long-format) table with one row per (patient, fup, [task,] horizon):

        patient_id, fup, horizon, label, prob [, prob_cal, time, status,
                                               risk, pred_median_time]
        (multi-task runs add a 'task' column: one row per patient, fup, task,
         horizon, with the label/prob of that task's head)

    - Reads keys of the form '{split_label}_fup_{XXXX}_labels', '..._probs'
      and (optionally) '..._patientid', '..._probs_cal', '..._times',
      '..._statuses', '..._risks', '..._median_times' from the dict built by
      the test passes of train_classification.py / train_survival.py /
      train_classic_ml.py.
    - 'task_keys': for multi-task runs, the ordered list of tasks owning the
      matrix columns (task-major: [t1_h1..t1_hK, t2_h1..t2_hK, ...]). When
      given, every matrix column becomes a (task, horizon) row and a 'task'
      column is written. When None (classic single-task runs) the columns are
      interpreted as the horizon list, as before.
    - 'label' is 0/1 (binary tasks) or the horizon-binarized survival label
      (1 = event within horizon, 0 = known event-free at horizon). Rows whose
      label is unknown at that horizon (-100, i.e. censored before horizon)
      are skipped.
    - 'prob' is the model's predicted probability of the positive case
      (sigmoid output for binary tasks; 1 - S(horizon) for survival).
    - Survival runs additionally expose the raw ground truth and model output
      per row: 'time'/'status' = observed (censored) time to next event and
      event indicator from the fup observation point; 'risk' = Cox log hazard
      ratio; 'pred_median_time' = model-predicted median time to the next
      event (NaN when no median is reached within the observed baseline).
    - Writes a plain CSV (one file per split, per training run).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fup_re = re.compile(rf"^{re.escape(split_label)}_fup_(\d+)_labels$")

    # Optional per-row (horizon-independent) arrays: npz key suffix -> CSV column
    extras_map = {"times": "time", "statuses": "status", "risks": "risk",
                  "median_times": "pred_median_time"}

    frames = []
    for key in sorted(all_predictions.keys()):
        match = fup_re.match(key)
        if not match:
            continue
        fup_str, fup_val = match.group(1), int(match.group(1))
        prefix = f"{split_label}_fup_{fup_str}"
        if f"{prefix}_probs" not in all_predictions:
            continue

        labels = np.asarray(all_predictions[key])
        probs = np.asarray(all_predictions[f"{prefix}_probs"])
        if labels.ndim == 1:
            labels = labels[:, None]
        if probs.ndim == 1:
            probs = probs[:, None]
        n_rows = labels.shape[0]

        patient_ids = all_predictions.get(f"{prefix}_patientid", None)
        if patient_ids is not None:
            patient_ids = np.asarray(patient_ids)
            if len(patient_ids) != n_rows:
                print(f"    [WARN] {prefix}: patient ids ({len(patient_ids)}) "
                      f"misaligned with scored rows ({n_rows}); using row indices")
                patient_ids = None
        if patient_ids is None:
            patient_ids = np.arange(n_rows)

        probs_cal = all_predictions.get(f"{prefix}_probs_cal", None)
        extras = {}
        for suffix, col in extras_map.items():
            arr = all_predictions.get(f"{prefix}_{suffix}", None)
            if arr is not None and len(np.asarray(arr)) == n_rows:
                extras[col] = np.asarray(arr)

        # Map each matrix column to its (task, horizon) pair. Multi-task runs
        # store one column per (task, horizon), task-major ordered.
        n_h = len(horizons)
        if task_keys:
            if labels.shape[1] != len(task_keys) * n_h:
                print(f"    [WARN] {prefix}: {labels.shape[1]} label columns but "
                      f"{len(task_keys)} tasks x {n_h} horizons expected; skipping extras")
            col_map = [(task_keys[c // n_h], horizons[c % n_h])
                       for c in range(min(labels.shape[1], len(task_keys) * n_h))]
        else:
            col_map = [(None, horizons[c]) for c in range(min(labels.shape[1], n_h))]

        for c, (task_name, horizon_val) in enumerate(col_map):
            lab_col = labels[:, c]
            prob_col = probs[:, c] if probs.shape[1] > c else probs[:, -1]
            valid = lab_col != -100  # drop rows with unknown label at this horizon
            if valid.sum() == 0:
                continue
            n_valid = int(valid.sum())
            row = {
                "patient_id": patient_ids[valid],
                "fup": np.full(n_valid, fup_val, dtype=np.int32),
                "horizon": np.full(n_valid, int(horizon_val), dtype=np.int32),
                "label": lab_col[valid].astype(np.int8),
                "prob": prob_col[valid].astype(np.float32),
            }
            if task_name is not None:
                row["task"] = np.full(n_valid, task_name, dtype=object)
            if probs_cal is not None:
                probs_cal = np.asarray(probs_cal)
                cal_col = (probs_cal[:, c] if probs_cal.ndim > 1 and probs_cal.shape[1] > c
                           else probs_cal.reshape(-1))
                row["prob_cal"] = np.asarray(cal_col)[valid].astype(np.float32)
            for col, arr in extras.items():
                row[col] = arr[valid]
            frames.append(pd.DataFrame(row))

    if not frames:
        print(f"    [WARN] No per-FUP prediction arrays found for split '{split_label}'; "
              f"tidy table not written.")
        return None

    table = pd.concat(frames, ignore_index=True)
    try:  # patient ids are numeric strings in this dataset -> store as int
        table["patient_id"] = table["patient_id"].astype(np.int64)
    except (TypeError, ValueError):
        pass  # keep original dtype if not parseable as int
    sort_cols = [c for c in ["patient_id", "fup", "task", "horizon"] if c in table.columns]
    table = table.sort_values(sort_cols).reset_index(drop=True)

    if csv_name:
        table.to_csv(output_dir / csv_name, index=False, float_format="%.6f")
        print(f"    [SUCCESS] Saved tidy predictions CSV: {output_dir / csv_name} "
              f"({len(table)} rows, columns: {', '.join(table.columns)})")
    return table


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


def load_run_metadata(run_dir: Path) -> dict | None:
    """Read the run_metadata.json written by train_classification.py, if present.

    Contains {"task_key", "tasks", "horizons", "label_keys"} describing the head
    layout of the run (one classifier head per (task, horizon) pair, task-major).
    """
    meta_path = Path(run_dir) / "run_metadata.json"
    if meta_path.exists():
        return json.loads(meta_path.read_text())
    return None


def find_best_checkpoint_multitask(base_dir: Path, task_key: str, horizon: int) -> tuple[Path, dict]:
    """Locate a checkpoint of a MULTI-TASK run that contains heads for `task_key`.

    Multi-task fine-tuning stores all heads under one alias folder per config
    entry: base_dir/'finetuning'/<alias>/hrz(...)/ with a run_metadata.json
    listing every trained task. Scans those and returns
    (last checkpoint, metadata) for the first run containing `task_key` at the
    requested horizon. Raises FileNotFoundError otherwise (callers can then
    fall back to the classic single-task layout via find_best_checkpoint).
    """
    base_dir = Path(base_dir)
    ft_root = base_dir / "finetuning"
    if not ft_root.exists():
        raise FileNotFoundError(f"Task directory missing: {ft_root}")

    h_str = f"{horizon:04d}"
    pattern = re.compile(rf"hrz\(([^)]*\b{h_str}\b[^)]*)\)")
    matches: list[tuple[Path, dict]] = []
    for run_dir in sorted(p for p in ft_root.iterdir() if p.is_dir()):
        for hrz_dir in sorted(p for p in run_dir.iterdir() if p.is_dir() and pattern.search(p.name)):
            meta = load_run_metadata(hrz_dir)
            if not meta or task_key not in meta.get("tasks", []):
                continue
            checkpoint_dirs = sorted(
                hrz_dir.glob("checkpoint-*"),
                key=lambda p: int(p.name.split("-")[-1]),
            )
            if checkpoint_dirs:
                matches.append((checkpoint_dirs[-1], meta))
    if matches:
        return max(matches, key=lambda item: item[0].stat().st_mtime_ns)
    raise FileNotFoundError(
        f"No multi-task run with heads for task '{task_key}' (horizon {h_str}) under {ft_root}"
    )


def _pr_auc_diff_chunk(y_true, p_tf, p_base, index_chunk):
    """Compute PR-AUC diffs for a chunk of bootstrap index arrays.

    Standalone helper (not a closure) so it can be reused across threads
    without re-creating the function object on every call.
    """
    chunk_diffs = []
    for idx in index_chunk:
        yt_b = y_true[idx]
        if len(np.unique(yt_b)) < 2:
            continue
        auc_tf = average_precision_score(yt_b, p_tf[idx])
        auc_base = average_precision_score(yt_b, p_base[idx])
        chunk_diffs.append(auc_tf - auc_base)
    return chunk_diffs


def paired_bootstrap_pr_auc_pvalue(
    y_true: np.ndarray,
    p_tf: np.ndarray,
    p_base: np.ndarray,
    n_bootstraps: int = 1000,
    seed: int = 42,
    max_workers: int = 4,
) -> float:
    """
    Computes a one-tailed paired bootstrap p-value testing if Transformer PR-AUC
    is significantly higher than a baseline model's PR-AUC on the exact same test samples.
    """
    if len(np.unique(y_true)) < 2:
        return np.nan

    n_samples = len(y_true)
    rng = np.random.default_rng(seed)
    boot_indices = rng.choice(n_samples, size=(n_bootstraps, n_samples), replace=True)

    max_workers = max(1, max_workers)
    chunks = np.array_split(boot_indices, max_workers) if max_workers > 1 else [boot_indices]

    diffs = []
    if max_workers > 1 and len(chunks) > 1:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(_pr_auc_diff_chunk, y_true, p_tf, p_base, c)
                for c in chunks
            ]
            for f in futures:
                diffs.extend(f.result())
    else:
        diffs = _pr_auc_diff_chunk(y_true, p_tf, p_base, boot_indices)

    if not diffs:
        return np.nan

    # Empirical p-value for the hypothesis: Transformer PR-AUC > Baseline PR-AUC
    p_value = np.mean(np.array(diffs) <= 0)
    return float(p_value)


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
