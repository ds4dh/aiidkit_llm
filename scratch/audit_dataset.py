import os
import re
import sys
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np
import pandas as pd
from datasets import load_from_disk
from tqdm import tqdm

# Path configuration
BASE_DATA_PATH = Path("/home/shares/ds4dh/aiidkit_project/data_new/processed/v3.6_old/teav")
RESULTS_DIR = Path("results_final")
OUTPUT_DIR = RESULTS_DIR / "analysis" / "alignment_audit"
FUP_OUTPUT_DIR = OUTPUT_DIR / "by_fup"

TASKS = ["infection_bacteria"]
SPLIT_TYPES = ["random_split"]  # ["random_split", "temporal_split", "center_split"]
CLASSIC_MODELS = ["logistic_regression"]  # ["logistic_regression", "random_forest", "xgboost"]
TARGET_HORIZONS = [30, 60, 90]


# ==========================================
# WORKER DATA LOADERS
# ==========================================
def get_raw_patientkeys(split: str, fup_day: int, dataset_split: str) -> list:
    """Loads raw patient keys cleanly for a specific FUP folder."""
    raw_fup_dir = BASE_DATA_PATH / split / f"fup_{fup_day:04d}"
    if not raw_fup_dir.exists():
        raw_fup_dir = BASE_DATA_PATH / split / f"fup_{fup_day:04d}d"
    if not raw_fup_dir.exists():
        return None

    try:
        raw_ds = load_from_disk(str(raw_fup_dir))[dataset_split]
        return raw_ds["patientkey"]
    except Exception:
        return None


def load_transformer_flat_worker(task: str, split: str, dataset_split: str, target_horizon: int) -> pd.DataFrame:
    """Worker function to load Transformer predictions with reconstructed index alignment."""
    task_base_dir = RESULTS_DIR / "transformer" / split / "e00-a15-v60" / "finetuning" / task
    if not task_base_dir.exists():
        return pd.DataFrame()

    file_name = "validation_probs.npz" if dataset_split == "validation" else "test_probs.npz"
    candidates = [p for p in task_base_dir.iterdir() if p.is_dir() and p.name.startswith("hrz(")]
    
    matched_dir, target_idx, horizons = None, None, []
    for c in candidates:
        match = re.search(r"hrz\(([\d-]+)\)", c.name)
        if match:
            horizons = [int(h) for h in match.group(1).split("-")]
            if target_horizon in horizons:
                matched_dir = c
                target_idx = horizons.index(target_horizon)
                break

    if matched_dir is None or not (matched_dir / file_name).exists():
        return pd.DataFrame()

    npz_data = np.load(matched_dir / file_name, allow_pickle=True)
    prefix = f"{dataset_split}_" if dataset_split == "validation" else "test_"
    fup_pattern = re.compile(rf"^{prefix}fup_(\d+)_labels$")
    fup_days = sorted({int(fup_pattern.match(k).group(1)) for k in npz_data.files if fup_pattern.match(k)})

    flat_records = []
    prob_template = f"{prefix}fup_%04d_probs"

    for fup_day in fup_days:
        lbl_key = f"{prefix}fup_{fup_day:04d}_labels"
        if lbl_key not in npz_data.files:
            continue
        
        y_true_all = npz_data[lbl_key]
        target_prob_str = prob_template % fup_day
        if target_prob_str not in npz_data.files:
            target_prob_str = f"{prefix}fup_{fup_day:04d}_probs"
        y_prob_all = npz_data[target_prob_str]

        y_true = y_true_all if y_true_all.ndim == 1 else y_true_all[:, target_idx]
        y_prob = y_prob_all if y_prob_all.ndim == 1 else y_prob_all[:, target_idx]

        raw_fup_dir = BASE_DATA_PATH / split / f"fup_{fup_day:04d}"
        if not raw_fup_dir.exists():
            raw_fup_dir = BASE_DATA_PATH / split / f"fup_{fup_day:04d}d"
        if not raw_fup_dir.exists():
            continue

        raw_ds = load_from_disk(str(raw_fup_dir))[dataset_split]
        raw_keys = raw_ds["patientkey"]

        # RECONSTRUCT TRANSFORMER KEEP_MASK: .any(axis=1) across trained multi-label horizons
        label_cols = [f"label_{task}_{h:04d}d" for h in horizons if f"label_{task}_{h:04d}d" in raw_ds.column_names]
        if label_cols:
            label_matrix = np.stack([raw_ds[col] for col in label_cols], axis=1)
            keep_mask = (label_matrix != -100).any(axis=1)
        else:
            keep_mask = np.ones(len(raw_keys), dtype=bool)

        valid_indices = np.where(keep_mask)[0]
        eval_len = min(len(valid_indices), len(y_prob), len(y_true))

        for idx in range(eval_len):
            raw_ds_idx = valid_indices[idx]
            flat_records.append({
                "patientkey": raw_keys[raw_ds_idx],
                "time_step": fup_day,
                "horizon": int(target_horizon),
                "y_true_TF": int(y_true[idx]),
                "y_prob_TF": float(y_prob[idx]),
            })

    return pd.DataFrame(flat_records)


def load_classic_ml_worker(model_name: str, task: str, split: str, dataset_split: str, target_horizon: int) -> pd.DataFrame:
    """Worker function to load Classic ML predictions with reconstructed index alignment."""
    file_name = "val_predictions.npz" if dataset_split == "validation" else "test_predictions.npz"
    hrz_str = f"{target_horizon:04d}"
    task_dir = RESULTS_DIR / "classic_ml" / split / model_name / task

    if not task_dir.exists():
        return pd.DataFrame()

    matched_dir = None
    for p in task_dir.iterdir():
        if p.is_dir() and f"hrz({hrz_str})" in p.name:
            matched_dir = p
            break

    if matched_dir is None or not (matched_dir / file_name).exists():
        return pd.DataFrame()

    npz_data = np.load(matched_dir / file_name, allow_pickle=True)
    prefix = f"{dataset_split}_" if dataset_split == "validation" else "test_"
    fup_pattern = re.compile(rf"^{prefix}fup_(\d+)_labels$")
    fup_days = sorted({int(fup_pattern.match(k).group(1)) for k in npz_data.files if fup_pattern.match(k)})

    flat_records = []
    prob_template = f"{prefix}fup_%04d_probs"

    for fup_day in fup_days:
        lbl_key = f"{prefix}fup_{fup_day:04d}_labels"
        if lbl_key not in npz_data.files:
            continue
        
        y_true = npz_data[lbl_key].flatten()
        target_prob_str = prob_template % fup_day
        if target_prob_str not in npz_data.files:
            target_prob_str = f"{prefix}fup_{fup_day:04d}_probs"
        y_prob = npz_data[target_prob_str].flatten()

        raw_fup_dir = BASE_DATA_PATH / split / f"fup_{fup_day:04d}"
        if not raw_fup_dir.exists():
            raw_fup_dir = BASE_DATA_PATH / split / f"fup_{fup_day:04d}d"
        if not raw_fup_dir.exists():
            continue

        raw_ds = load_from_disk(str(raw_fup_dir))[dataset_split]
        raw_keys = raw_ds["patientkey"]

        # RECONSTRUCT CLASSIC ML KEEP_MASK: Single target column != -100
        target_col = f"label_{task}_{target_horizon:04d}d"
        if target_col in raw_ds.column_names:
            keep_mask = np.array(raw_ds[target_col]) != -100
        else:
            keep_mask = np.ones(len(raw_keys), dtype=bool)

        valid_indices = np.where(keep_mask)[0]
        eval_len = min(len(valid_indices), len(y_prob), len(y_true))

        for idx in range(eval_len):
            raw_ds_idx = valid_indices[idx]
            flat_records.append({
                "patientkey": raw_keys[raw_ds_idx],
                "time_step": fup_day,
                "horizon": int(target_horizon),
                "y_true_ML": int(y_true[idx]),
                "y_prob_ML": float(y_prob[idx]),
            })

    return pd.DataFrame(flat_records)


# ==========================================
# PARALLEL WORKER ENTRY POINT
# ==========================================
def _audit_task_tuple(args_tuple: tuple) -> dict:
    """Top-level worker function for multiprocessing."""
    task, split, dataset_split, horizon, model = args_tuple
    
    df_tf = load_transformer_flat_worker(task, split, dataset_split, horizon)
    df_ml = load_classic_ml_worker(model, task, split, dataset_split, horizon)

    if df_tf.empty or df_ml.empty:
        return None

    merge_keys = ["patientkey", "time_step", "horizon"]
    merged = pd.merge(df_tf, df_ml, on=merge_keys, how="outer", indicator=True)

    df_tf_valid = df_tf[df_tf["y_true_TF"] != -100]
    df_ml_valid = df_ml[df_ml["y_true_ML"] != -100]
    
    valid_merged = pd.merge(df_tf_valid, df_ml_valid, on=merge_keys, how="inner")
    label_mismatches = valid_merged[valid_merged["y_true_TF"] != valid_merged["y_true_ML"]]

    only_tf = merged[merged["_merge"] == "left_only"]
    tf_censored_count = (only_tf["y_true_TF"] == -100).sum() if not only_tf.empty else 0
    tf_uncensored_missing = only_tf[only_tf["y_true_TF"] != -100]
    only_ml = merged[merged["_merge"] == "right_only"]

    # PER-FUP DETAILED BREAKDOWN COMPUTATION & EXPORT
    all_fups = sorted(set(df_tf["time_step"].unique()).union(set(df_ml["time_step"].unique())))
    fup_rows = []

    for fup in all_fups:
        sub_tf = df_tf[df_tf["time_step"] == fup]
        sub_ml = df_ml[df_ml["time_step"] == fup]

        if sub_tf.empty or sub_ml.empty:
            fup_rows.append({
                "time_step": fup,
                "TF_total_rows": len(sub_tf),
                "ML_total_rows": len(sub_ml),
                "shared_patients": 0,
                "clean_matches": 0,
                "TF_censored_ML_valid": 0,
                "TF_valid_ML_missing": 0,
                "ML_valid_TF_missing": 0,
                "label_contradictions": 0,
            })
            continue

        m_fup = pd.merge(sub_tf, sub_ml, on=["patientkey", "time_step", "horizon"], how="outer")

        both_valid = m_fup[(m_fup["y_true_TF"] != -100) & (m_fup["y_true_ML"] != -100)]
        clean_matches = (both_valid["y_true_TF"] == both_valid["y_true_ML"]).sum()
        contradictions = (both_valid["y_true_TF"] != both_valid["y_true_ML"]).sum()

        tf_censored = (m_fup["y_true_TF"] == -100) & (m_fup["y_true_ML"] != -100)
        tf_valid_missing_ml = (m_fup["y_true_TF"] != -100) & (m_fup["y_true_ML"].isna())
        ml_valid_missing_tf = (m_fup["y_true_ML"] != -100) & (m_fup["y_true_TF"].isna())

        fup_rows.append({
            "time_step": fup,
            "TF_total_rows": len(sub_tf),
            "ML_total_rows": len(sub_ml),
            "shared_patients": len(m_fup.dropna(subset=["y_true_TF", "y_true_ML"])),
            "clean_matches": int(clean_matches),
            "TF_censored_ML_valid": int(tf_censored.sum()),
            "TF_valid_ML_missing": int(tf_valid_missing_ml.sum()),
            "ML_valid_TF_missing": int(ml_valid_missing_tf.sum()),
            "label_contradictions": int(contradictions),
        })

    df_fup_report = pd.DataFrame(fup_rows)
    fup_file_name = f"fup_audit_{task}_{split}_{dataset_split}_hrz{horizon}d_{model}.csv"
    
    FUP_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df_fup_report.to_csv(FUP_OUTPUT_DIR / fup_file_name, index=False)

    return {
        "Task": task,
        "Split": split,
        "Dataset Split": dataset_split,
        "Horizon (d)": horizon,
        "Baseline Model": model,
        "TF Total Rows": len(df_tf),
        "TF Active (uncensored)": len(df_tf_valid),
        "ML Active Total": len(df_ml_valid),
        "Synchronized Shared Overlap": len(valid_merged),
        "Label Contradictions": len(label_mismatches),
        "Dropped (Censored at Horizon)": tf_censored_count,
        "Discrepancy (Active TF Missing in ML)": len(tf_uncensored_missing),
        "Discrepancy (Active ML Missing in TF)": len(only_ml),
    }


# ==========================================
# MAIN EXECUTOR
# ==========================================
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FUP_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    param_combos = [
        (task, split, dataset_split, horizon, model)
        for task in TASKS
        for split in SPLIT_TYPES
        for dataset_split in ["validation", "test"]
        for horizon in TARGET_HORIZONS
        for model in CLASSIC_MODELS
    ]

    num_workers = min(16, os.cpu_count() or 1)
    print("=" * 80, flush=True)
    print(f" STARTING MULTIPROCESSING AUDIT ({len(param_combos)} Tasks | {num_workers} CPU Workers) ", flush=True)
    print("=" * 80, flush=True)
    results = []

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(_audit_task_tuple, combo): combo for combo in param_combos}
        
        with tqdm(total=len(param_combos), desc="Auditing configs", file=sys.stdout) as pbar:
            for future in as_completed(futures):
                res = future.result()
                if res is not None:
                    results.append(res)
                pbar.update(1)

    if not results:
        print("\n[!] No valid results returned.", flush=True)
        return

    df_report = pd.DataFrame(results)
    csv_path = OUTPUT_DIR / "parallel_sample_alignment_audit_report.csv"
    df_report.to_csv(csv_path, index=False)

    print(f"\n[SUCCESS] Multiprocessing audit finished!", flush=True)
    print(f" -> Main summary report saved to: {csv_path}", flush=True)
    print(f" -> Per-FUP detailed CSV files saved to: {FUP_OUTPUT_DIR}\n", flush=True)

    summary_cols = [
        "Task", "Split", "Dataset Split", "Horizon (d)", "Baseline Model",
        "TF Active (uncensored)", "ML Active Total", "Synchronized Shared Overlap",
        "Label Contradictions", "Dropped (Censored at Horizon)"
    ]
    print(df_report[summary_cols].to_markdown(index=False), flush=True)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()