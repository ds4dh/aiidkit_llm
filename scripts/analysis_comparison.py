import re
import argparse
import hashlib
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datasets import load_from_disk
from sklearn.metrics import (
    average_precision_score, precision_recall_curve, roc_auc_score, roc_curve,
)
from statsmodels.stats.contingency_tables import mcnemar
from scripts.script_utils import calibrate_array_pair, calibrate_dataframe_pair


# ==========================================
# SYSTEM CONFIGURATION AND PIPELINE TUNING
# ==========================================
parser = argparse.ArgumentParser(description="Run optimized evaluation pipeline with custom thresholding.")
parser.add_argument(
    "--threshold_mode",
    type=str,
    default="window_specific",
    choices=["global", "window_specific", "manual"],
    help="Select boundary constraint tuning mode (global, window_specific, or manual).",
)
parser.add_argument(
    "--target_recall",
    type=int,
    default=70,
    help="Target minimal recall used for tuning model decision threshold (e.g., 80).",
)
args = parser.parse_args()

THRESHOLD_MODE = args.threshold_mode
TARGET_RECALL_INPUT = args.target_recall
TARGET_RECALL_ANCHOR = float(TARGET_RECALL_INPUT) / 100.0
THRESHOLD_SUBFOLDER = f"rec{TARGET_RECALL_INPUT}"

USE_CALIBRATED_PROBS = {
    "Transformer": True,  # trained with CE, so no need of calibration?
    "logistic_regression": True,
    "random_forest": True,
    "xgboost": True,
}
THRESHOLD_TUNING_SET = "validation"  # validation (standard)), test (sanity checking)

BASE_DATA_PATH = Path("/home/shares/ds4dh/aiidkit_project/data_new/processed/v3.6/teav")
RESULTS_DIR = Path("results_final")
ANALYSIS_DIR = RESULTS_DIR / "analysis" / "comparison"
ANALYSIS_SUBDIR = ANALYSIS_DIR / THRESHOLD_MODE / THRESHOLD_SUBFOLDER
CACHE_DIR = ANALYSIS_DIR / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

CLASSIC_MODELS = ["logistic_regression", "random_forest", "xgboost"]
ALL_MODELS = ["Transformer"] + CLASSIC_MODELS

TASKS = ["infection_bacteria"]  # ["infection_bacteria", "infection_virus"]
SPLIT_TYPES = ["random_split", "temporal_split", "center_split"]
N_BOOTSTRAP = 1000
TARGET_HORIZONS = [30, 60, 90]
CLINICAL_WINDOWS = {
    "Perioperative (0-30 d)": (0, 30),
    "Opportunistic (31-180 d)": (31, 180),
    "Maintenance (181-360 d)": (181, 360),
    "Long-term (361-720 d)": (361, 720),
}

MODEL_DISPLAY_MAP = {
    "Transformer": "TF",
    "logistic_regression": "LR",
    "random_forest": "RF",
    "xgboost": "XGB",
}
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

METRIC_MAPPING = [
    ("ROC-AUC", "↑"), ("PR-AUC", "↑"), ("ECE", "↓"), 
    ("Sensitivity (recall)", "↑"), ("Precision", "↑"), ("Specificity", "↑")
]

# Create a unique parameter signature token for parameter-dependent caching
PARAM_SIGNATURE = {
    "THRESHOLD_MODE": THRESHOLD_MODE,
    "TARGET_RECALL_ANCHOR": TARGET_RECALL_ANCHOR,
    "THRESHOLD_TUNING_SET": THRESHOLD_TUNING_SET,
    "USE_CALIBRATED_PROBS": USE_CALIBRATED_PROBS
}
PARAM_HASH = hashlib.md5(json.dumps(PARAM_SIGNATURE, sort_keys=True).encode()).hexdigest()[:10]


# ==========================================
# MATHEMATICAL & PREDICTION LOADING UTILS
# ==========================================
def compute_expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n_samples = len(y_true)
    if n_samples == 0:
        return 0.0
    
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        in_bin = (y_prob >= bin_lower) & (y_prob < bin_upper) if i < n_bins - 1 else (y_prob >= bin_lower) & (y_prob <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(y_true[in_bin])
            avg_confidence_in_bin = np.mean(y_prob[in_bin])
            ece += prop_in_bin * np.abs(avg_confidence_in_bin - accuracy_in_bin)
            
    return ece


def load_transformer_flat_dataframe(task: str, split: str, dataset_split: str, target_horizon) -> pd.DataFrame:
    if target_horizon == "combined":
        dfs = [load_transformer_flat_dataframe(task, split, dataset_split, h) for h in TARGET_HORIZONS]
        valid_dfs = [d for d in dfs if not d.empty]
        return pd.concat(valid_dfs, ignore_index=True) if valid_dfs else pd.DataFrame()

    # Tier 1 Caching Check (Base Dataframe Storage)
    cache_file = CACHE_DIR / f"raw_tf_{task}_{split}_{dataset_split}_{target_horizon}.parquet"
    if cache_file.exists():
        return pd.read_parquet(cache_file)

    file_name = "validation_probs.npz" if dataset_split == "validation" else "test_probs.npz"
    task_base_dir = RESULTS_DIR / "transformer" / split / "e00-a15-v60" / "finetuning" / task
    
    if not task_base_dir.exists():
        return pd.DataFrame()
        
    candidates = [p for p in task_base_dir.iterdir() if p.is_dir() and p.name.startswith("hrz(")]
    matched_dir = None
    target_idx = None
    
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
    
    flat_records = []
    fup_days = {int(fup_pattern.match(k).group(1)) for k in npz_data.files if fup_pattern.match(k)}
    
    # Always pull raw uncalibrated probabilities for post-hoc analysis-time calibration
    prob_template = f"{prefix}fup_%04d_probs"
    
    for fup_day in sorted(fup_days):
        if f"{prefix}fup_{fup_day:04d}_labels" not in npz_data.files:
            continue
        y_true_all = npz_data[f"{prefix}fup_{fup_day:04d}_labels"]
        
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
        
        for idx in range(len(y_prob)):
            if int(y_true[idx]) == -100:
                continue
            flat_records.append({
                "patientkey": raw_keys[idx],
                "time_step": fup_day,
                "horizon": int(target_horizon),
                "y_true": int(y_true[idx]),
                "y_prob": float(y_prob[idx])
            })
            
    df = pd.DataFrame(flat_records)
    if not df.empty:
        df.to_parquet(cache_file, index=False)
    return df


def load_classic_ml_flat_dataframe(model_name: str, task: str, split: str, dataset_split: str, target_horizon) -> pd.DataFrame:
    if target_horizon == "combined":
        dfs = [load_classic_ml_flat_dataframe(model_name, task, split, dataset_split, h) for h in TARGET_HORIZONS]
        valid_dfs = [d for d in dfs if not d.empty]
        return pd.concat(valid_dfs, ignore_index=True) if valid_dfs else pd.DataFrame()

    # Tier 1 Caching Check (Base Dataframe Storage)
    cache_file = CACHE_DIR / f"raw_ml_{model_name}_{task}_{split}_{dataset_split}_{target_horizon}.parquet"
    if cache_file.exists():
        return pd.read_parquet(cache_file)

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
    
    flat_records = []
    fup_days = {int(fup_pattern.match(k).group(1)) for k in npz_data.files if fup_pattern.match(k)}
    
    # Always pull raw uncalibrated probabilities for post-hoc analysis-time calibration
    prob_template = f"{prefix}fup_%04d_probs"
    
    for fup_day in sorted(fup_days):
        if f"{prefix}fup_{fup_day:04d}_labels" not in npz_data.files:
            continue
        y_true = npz_data[f"{prefix}fup_{fup_day:04d}_labels"].flatten()
        
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
        
        for idx in range(len(y_prob)):
            if int(y_true[idx]) == -100:
                continue
            flat_records.append({
                "patientkey": raw_keys[idx],
                "time_step": fup_day,
                "horizon": int(target_horizon),
                "y_true": int(y_true[idx]),
                "y_prob": float(y_prob[idx])
            })
            
    df = pd.DataFrame(flat_records)
    if not df.empty:
        df.to_parquet(cache_file, index=False)
    return df


def bootstrap_metric_ci(sub_df: pd.DataFrame, model_name: str, thresh: float, metric_type: str) -> tuple:
    y_true_orig = sub_df['y_true'].values
    y_prob_orig = sub_df[f'y_prob_{model_name}'].values
    
    if metric_type == "ROC-AUC":
        pe = roc_auc_score(y_true_orig, y_prob_orig) if len(np.unique(y_true_orig)) >= 2 else 0.5
    elif metric_type == "PR-AUC":
        pe = average_precision_score(y_true_orig, y_prob_orig) if len(np.unique(y_true_orig)) >= 2 else 0.0
    elif metric_type == "ECE":
        pe = compute_expected_calibration_error(y_true_orig, y_prob_orig)
    else:
        preds_bin = (y_prob_orig >= thresh).astype(int)
        tp = np.sum((preds_bin == 1) & (y_true_orig == 1))
        fp = np.sum((preds_bin == 1) & (y_true_orig == 0))
        tn = np.sum((preds_bin == 0) & (y_true_orig == 0))
        fn = np.sum((preds_bin == 0) & (y_true_orig == 1))
        
        if metric_type == "Sensitivity (recall)":
            pe = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        elif metric_type == "Precision":
            pe = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        elif metric_type == "Specificity":
            pe = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    boot_stats = []
    np.random.seed(42)
    
    n_samples = len(sub_df)
    boot_indices = np.random.choice(n_samples, size=(N_BOOTSTRAP, n_samples), replace=True)
    
    for i in range(N_BOOTSTRAP):
        indices = boot_indices[i]
        yt = y_true_orig[indices]
        yp = y_prob_orig[indices]
        
        if len(np.unique(yt)) < 2:
            continue
            
        if metric_type == "ROC-AUC":
            boot_stats.append(roc_auc_score(yt, yp))
        elif metric_type == "PR-AUC":
            boot_stats.append(average_precision_score(yt, yp))
        elif metric_type == "ECE":
            boot_stats.append(compute_expected_calibration_error(yt, yp))
        else:
            p_bin = (yp >= thresh).astype(int)
            tp_b = np.sum((p_bin == 1) & (yt == 1))
            fp_b = np.sum((p_bin == 1) & (yt == 0))
            tn_b = np.sum((p_bin == 0) & (yt == 0))
            fn_b = np.sum((p_bin == 0) & (yt == 1))
            
            if metric_type == "Sensitivity (recall)":
                boot_stats.append(tp_b / (tp_b + fn_b) if (tp_b + fn_b) > 0 else 0.0)
            elif metric_type == "Precision":
                boot_stats.append(tp_b / (tp_b + fp_b) if (tp_b + fp_b) > 0 else 0.0)
            elif metric_type == "Specificity":
                boot_stats.append(tn_b / (tn_b + fp_b) if (tn_b + fp_b) > 0 else 0.0)
                
    if not boot_stats:
        return pe, f"{pe:.3f} (NaN - NaN)"
        
    low_bound = np.percentile(boot_stats, 2.5)
    high_bound = np.percentile(boot_stats, 97.5)
    return pe, f"{pe:.3f} ({low_bound:.3f} - {high_bound:.3f})"


def calculate_strict_threshold(df: pd.DataFrame, low: int, high: int, target_recall: float, label: str) -> float:
    sub = df[(df['time_step'] >= low) & ((df['time_step'] + df['horizon']) <= high)].copy()
    sub = sub[sub['y_true'] != -100]
    
    if sub.empty:
        raise RuntimeError(f"Strict Error: Zero tracking validation samples located within window frame {low}-{high}d for model setup: {label}")
    if len(np.unique(sub['y_true'])) < 2:
        raise RuntimeError(f"Strict Error: Class label space has zero variance in window {low}-{high}d for model setup: {label}")
        
    p, r, t = precision_recall_curve(sub['y_true'].values, sub['y_prob'].values)
    valid_indices = np.where(r[:-1] >= target_recall)[0]
    
    if len(valid_indices) == 0:
        raise RuntimeError(f"Strict Error: Target recall anchor of {target_recall:.2%} cannot be satisfied by model prediction arrays in slice {low}-{high}d for: {label}")
        
    return float(t[valid_indices[-1]])


# ==========================================
# REPORT STRUCTURING & CLEANING BREAKDOWNS
# ==========================================
def build_report_layout_and_mapping() -> tuple:
    METRIC_SORT_ORDER = ["Total evaluation frames", "Primary analysis"]
    ROW_CLEANING_MAP = {
        "Total evaluation frames": "Total evaluation frames",
        "Primary analysis": "Primary analysis",
        "Post-hoc analysis": "Post-hoc analysis"
    }
    
    for m in CLASSIC_MODELS:
        disp = MODEL_DISPLAY_MAP[m]
        key_head = f"TF vs {disp}"
        METRIC_SORT_ORDER.append(key_head)
        ROW_CLEANING_MAP[key_head] = key_head
        
        sub_metrics = [
            (f"  Discordant pairs (TF correct) ({disp})", "  Discordant pairs (TF correct)"),
            (f"  Discordant pairs ({disp} correct) ({disp})", f"  Discordant pairs ({disp} correct)"),
            (f"  McNemar p-value ({disp})", "  McNemar p-value"),
            (f"  Statistical winner ({disp})", "  Statistical winner")
        ]
        for internal_key, clean_name in sub_metrics:
            METRIC_SORT_ORDER.append(internal_key)
            ROW_CLEANING_MAP[internal_key] = clean_name
            
    METRIC_SORT_ORDER.append("Post-hoc analysis")
    
    for metric_prefix, arrow in METRIC_MAPPING:
        head_lbl = f"{metric_prefix} ({arrow})"
        METRIC_SORT_ORDER.append(head_lbl)
        ROW_CLEANING_MAP[head_lbl] = head_lbl
        
        for model in ["logistic_regression", "random_forest", "xgboost", "Transformer"]:
            disp = MODEL_DISPLAY_MAP[model]
            internal_key = f"  {metric_prefix} ({arrow}) ({disp})"
            METRIC_SORT_ORDER.append(internal_key)
            ROW_CLEANING_MAP[internal_key] = f"  {disp}"
            
    return METRIC_SORT_ORDER, ROW_CLEANING_MAP


# ==========================================
# STRATIFIED STRATEGY EVALUATION MODULES
# ==========================================
def process_split_strategy(task: str, split: str, horizon, analysis_dir: Path) -> dict:
    horizon_str = f"horizon_{horizon:04d}d" if isinstance(horizon, int) else "horizon_combined"
    
    # Parameter-dependent cache check
    report_out_path = analysis_dir / f"{split}_{horizon_str}_head_to_head_report_{PARAM_HASH}.csv"
    cache_data_path = CACHE_DIR / f"plot_cache_{task}_{split}_{horizon_str}_{PARAM_HASH}.npz"
    
    if report_out_path.exists() and cache_data_path.exists():
        print(f" [Cache Hit] Report and plotting cache recovered cleanly for split [{split}] under hash token: {PARAM_HASH}")
        cached_npz = np.load(cache_data_path, allow_pickle=True)
        plotted_window_cache = {}
        for w_name in CLINICAL_WINDOWS.keys():
            if f"{w_name}_y_true" in cached_npz:
                cols = {
                    'time_step': cached_npz[f"{w_name}_time_step"],
                    'horizon': cached_npz[f"{w_name}_horizon"],
                    'y_true': cached_npz[f"{w_name}_y_true"],
                }
                for m in ALL_MODELS:
                    cols[f'y_prob_{m}'] = cached_npz[f"{w_name}_y_prob_{m}"]
                
                sub_df = pd.DataFrame(cols)
                thresh_resolved = cached_npz[f"{w_name}_thresh"].item()
                plotted_window_cache[w_name] = (sub_df, thresh_resolved)
                
        df_report = pd.read_csv(report_out_path, index_col="Evaluation metric").fillna("")
        print(f"\n>>> REPORT SUMMARY (CACHED): STRATEGY SPLIT [{split.upper()}] | HORIZON: {str(horizon).upper()} <<<")
        print(df_report.to_markdown())
        return plotted_window_cache

    val_dfs = {"Transformer": load_transformer_flat_dataframe(task, split, "validation", horizon)}
    test_dfs = {"Transformer": load_transformer_flat_dataframe(task, split, "test", horizon)}
    
    for m in CLASSIC_MODELS:
        val_dfs[m] = load_classic_ml_flat_dataframe(m, task, split, "validation", horizon)
        test_dfs[m] = load_classic_ml_flat_dataframe(m, task, split, "test", horizon)
    
    # Global post-hoc calibration
    if THRESHOLD_MODE == "global":
        for m in ALL_MODELS:
            if USE_CALIBRATED_PROBS.get(m, False) and not val_dfs[m].empty and not test_dfs[m].empty:
                val_dfs[m], test_dfs[m] = calibrate_dataframe_pair(
                    df_val=val_dfs[m], 
                    df_test=test_dfs[m], 
                    prob_col="y_prob",
                )
        print(f" [Post-Hoc Global Calibration] Calibrated requested models on full Validation split [{split}].")
    
    if val_dfs["Transformer"].empty or test_dfs["Transformer"].empty:
        return None

    tuning_dfs = val_dfs if THRESHOLD_TUNING_SET == "validation" else test_dfs

    print(f" [Guard] Verifying alignment parity on set [{THRESHOLD_TUNING_SET}] for split [{split}]...")
    for m in CLASSIC_MODELS:
        if tuning_dfs[m].empty:
            raise RuntimeError(f"Alignment Error: Dataframe for baseline model '{m}' on set [{THRESHOLD_TUNING_SET}] is empty.")
            
        val_check = pd.merge(
            tuning_dfs["Transformer"][['patientkey', 'time_step', 'horizon', 'y_true']], 
            tuning_dfs[m][['patientkey', 'time_step', 'horizon', 'y_true']], 
            on=['patientkey', 'time_step', 'horizon'], 
            suffixes=('_TF', f'_{MODEL_DISPLAY_MAP[m]}')
        )
        
        mismatched_indices = val_check['y_true_TF'] != val_check[f'y_true_{MODEL_DISPLAY_MAP[m]}']
        count_mismatched_labels = np.sum(mismatched_indices)
        
        # Notification warning for samples removed
        if len(tuning_dfs["Transformer"]) != len(val_check) or len(tuning_dfs[m]) != len(val_check) or count_mismatched_labels > 0:
            
            # Isolate rows to drop from Transformer
            merged_keys = val_check[~mismatched_indices][['patientkey', 'time_step', 'horizon']]
            tf_dropped = tuning_dfs["Transformer"].merge(merged_keys, on=['patientkey', 'time_step', 'horizon'], how='left', indicator=True)
            tf_dropped_rows = tf_dropped[tf_dropped['_merge'] == 'left_only']
            
            # Isolate rows to drop from Classic Baseline model
            ml_dropped = tuning_dfs[m].merge(merged_keys, on=['patientkey', 'time_step', 'horizon'], how='left', indicator=True)
            ml_dropped_rows = ml_dropped[ml_dropped['_merge'] == 'left_only']
            
            # Extract clinical label distributions tracking dropped validation states
            pos_dropped_tf = np.sum(tf_dropped_rows['y_true'] == 1)
            neg_dropped_tf = np.sum(tf_dropped_rows['y_true'] == 0)
            pos_dropped_ml = np.sum(ml_dropped_rows['y_true'] == 1)
            neg_dropped_ml = np.sum(ml_dropped_rows['y_true'] == 0)
            
            print(f" [Guard Warning] Sample misalignment detected on set [{THRESHOLD_TUNING_SET}] for split [{split}] vs model [{m}].")
            print(f"    -> Transformer dropped total rows: {len(tf_dropped_rows)} (Positive cases: {pos_dropped_tf}, Negative cases: {neg_dropped_tf})")
            print(f"    -> Baseline ({m}) dropped total rows: {len(ml_dropped_rows)} (Positive cases: {pos_dropped_ml}, Negative cases: {neg_dropped_ml})")
            
            # Self-healing synchronization step
            clean_intersection_keys = val_check[~mismatched_indices][['patientkey', 'time_step', 'horizon']]
            tuning_dfs["Transformer"] = pd.merge(tuning_dfs["Transformer"], clean_intersection_keys, on=['patientkey', 'time_step', 'horizon'])
            tuning_dfs[m] = pd.merge(tuning_dfs[m], clean_intersection_keys, on=['patientkey', 'time_step', 'horizon'])
            
            tuning_dfs["Transformer"] = tuning_dfs["Transformer"].sort_values(by=["patientkey", "time_step", "horizon"]).reset_index(drop=True)
            tuning_dfs[m] = tuning_dfs[m].sort_values(by=["patientkey", "time_step", "horizon"]).reset_index(drop=True)

        final_verify = pd.merge(
            tuning_dfs["Transformer"][['patientkey', 'time_step', 'horizon', 'y_true']], 
            tuning_dfs[m][['patientkey', 'time_step', 'horizon', 'y_true']], 
            on=['patientkey', 'time_step', 'horizon'], 
            suffixes=('_TF', f'_{MODEL_DISPLAY_MAP[m]}')
        )
        assert len(tuning_dfs["Transformer"]) == len(tuning_dfs[m]) == len(final_verify), "Self-healing row reduction error."
            
    print(f" [Guard] Parity alignment verified and synchronized successfully on [{THRESHOLD_TUNING_SET}].")
        
    global_thresholds = {}
    if THRESHOLD_MODE == "global":
        for m in ALL_MODELS:
            global_thresholds[m] = calculate_strict_threshold(
                tuning_dfs[m], 0, 9999, TARGET_RECALL_ANCHOR, 
                f"{m} ({split} - Tuning Set: {THRESHOLD_TUNING_SET.upper()})"
            )
            
    split_results = {}
    plotted_window_cache = {}
    npz_save_payload = {}
    
    for w_idx, (window_name, (low, high)) in enumerate(CLINICAL_WINDOWS.items()):
        # Slice test frame for evaluation
        sub_test_df = test_dfs["Transformer"][
            (test_dfs["Transformer"]['time_step'] >= low) & 
            ((test_dfs["Transformer"]['time_step'] + test_dfs["Transformer"]['horizon']) <= high)
        ][['patientkey', 'time_step', 'horizon', 'y_true', 'y_prob']].rename(columns={'y_prob': 'y_prob_Transformer'}).copy()

        for m in CLASSIC_MODELS:
            if test_dfs[m].empty: continue
            m_sub = test_dfs[m][
                (test_dfs[m]['time_step'] >= low) & 
                ((test_dfs[m]['time_step'] + test_dfs[m]['horizon']) <= high)
            ][['patientkey', 'time_step', 'horizon', 'y_prob']].rename(columns={'y_prob': f'y_prob_{m}'})
            sub_test_df = pd.merge(sub_test_df, m_sub, on=['patientkey', 'time_step', 'horizon'])

        if sub_test_df.empty:
            continue
            
        y_true = sub_test_df['y_true'].values
        window_record = {"Total evaluation frames": len(sub_test_df), "Primary analysis": "", "Post-hoc analysis": ""}
        
        preds_map = {}
        thresholds_resolved = {}
        
        for m in ALL_MODELS:
            # Extract window-specific tuning slice
            val_m = tuning_dfs[m]
            val_win_mask = (val_m['time_step'] >= low) & ((val_m['time_step'] + val_m['horizon']) <= high)
            sub_val_df = val_m[val_win_mask].copy()

            # Window-specific post-hoc calibration
            if THRESHOLD_MODE != "global" and USE_CALIBRATED_PROBS.get(m, False):
                if not sub_val_df.empty and not sub_test_df.empty:
                    cal_win_val, cal_win_test = calibrate_array_pair(
                        y_val_true=sub_val_df["y_true"].values,
                        y_val_prob=sub_val_df["y_prob"].values,
                        y_test_prob=sub_test_df[f"y_prob_{m}"].values
                    )
                    sub_val_df["y_prob"] = cal_win_val
                    sub_test_df[f"y_prob_{m}"] = cal_win_test

            # Threhsold resolution
            if THRESHOLD_MODE == "global":
                thresholds_resolved[m] = global_thresholds[m]
            else:
                thresholds_resolved[m] = calculate_strict_threshold(
                    sub_val_df, low, high, TARGET_RECALL_ANCHOR, 
                    f"{m} [{window_name}] ({split} - Tuning Set: {THRESHOLD_TUNING_SET.upper()})"
                )
                
            preds_map[m] = (sub_test_df[f'y_prob_{m}'].values >= thresholds_resolved[m]).astype(int)

        for m in CLASSIC_MODELS:
            disp = MODEL_DISPLAY_MAP[m]
            correct_trans = (preds_map["Transformer"] == y_true)
            correct_baseline = (preds_map[m] == y_true)
            
            b = int(np.sum(correct_trans & ~correct_baseline))
            c = int(np.sum(~correct_trans & correct_baseline))
            
            contingency_table = [[np.sum(correct_trans & correct_baseline), b], [c, np.sum(~correct_trans & ~correct_baseline)]]
            try:
                res = mcnemar(contingency_table, exact=True)
                p_val = res.pvalue
                winner = "Tie" if p_val > 0.05 else ("TF" if b > c else disp)
            except Exception:
                p_val = np.nan
                winner = "NaN"
                
            window_record[f"TF vs {disp}"] = ""
            window_record[f"  Discordant pairs (TF correct) ({disp})"] = b
            window_record[f"  Discordant pairs ({disp} correct) ({disp})"] = c
            window_record[f"  McNemar p-value ({disp})"] = f"{p_val:.3e}" if not pd.isna(p_val) else "NaN"
            window_record[f"  Statistical winner ({disp})"] = winner

        for metric_prefix, arrow in METRIC_MAPPING:
            raw_estimates = {}
            string_outputs = {}
            
            window_record[f"{metric_prefix} ({arrow})"] = ""
            for target_model in ALL_MODELS:
                pe_val, formatted_str = bootstrap_metric_ci(sub_test_df, target_model, thresholds_resolved[target_model], metric_prefix)
                raw_estimates[target_model] = pe_val
                string_outputs[target_model] = formatted_str

            best_model = min(raw_estimates, key=raw_estimates.get) if metric_prefix == "ECE" else max(raw_estimates, key=raw_estimates.get)

            for target_model in ALL_MODELS:
                disp = MODEL_DISPLAY_MAP[target_model]
                final_cell_text = string_outputs[target_model]
                if target_model == best_model:
                    final_cell_text = f"**{final_cell_text}**"
                window_record[f"  {metric_prefix} ({arrow}) ({disp})"] = final_cell_text

        split_results[window_name] = window_record
        plotted_window_cache[window_name] = (sub_test_df, thresholds_resolved)
        
        npz_save_payload[f"{window_name}_time_step"] = sub_test_df['time_step'].values
        npz_save_payload[f"{window_name}_horizon"] = sub_test_df['horizon'].values
        npz_save_payload[f"{window_name}_y_true"] = sub_test_df['y_true'].values
        for m in ALL_MODELS:
            npz_save_payload[f"{window_name}_y_prob_{m}"] = sub_test_df[f'y_prob_{m}'].values
        npz_save_payload[f"{window_name}_thresh"] = thresholds_resolved

    if not split_results:
        return None

    METRIC_SORT_ORDER, ROW_CLEANING_MAP = build_report_layout_and_mapping()
    df_report = pd.DataFrame(split_results).reindex(METRIC_SORT_ORDER)
    df_report.index = df_report.index.map(ROW_CLEANING_MAP)
    df_report.index.name = "Evaluation metric"
    df_report = df_report.fillna("")
    df_report.to_csv(report_out_path)
    np.savez(cache_data_path, **npz_save_payload)
    
    print(f"\n>>> REPORT SUMMARY: STRATEGY SPLIT [{split.upper()}] | HORIZON: {str(horizon).upper()} <<<")
    with pd.option_context('display.max_colwidth', None, 'display.max_rows', None):
        print(df_report.to_markdown())
        
    return plotted_window_cache


# ==========================================
# VISUAL GRAPH PLOTTING ENGINES
# ==========================================
def render_matrix_visualization_grids(plotted_data_cache: dict, analysis_dir: Path, horizon_suffix: str):
    n_splits = len(SPLIT_TYPES)
    n_windows = len(CLINICAL_WINDOWS)
    
    fig_roc, axes_roc = plt.subplots(n_windows, n_splits, figsize=(6.5 * n_splits, 5.5 * n_windows), squeeze=False)
    fig_pr, axes_pr = plt.subplots(n_windows, n_splits, figsize=(6.5 * n_splits, 5.5 * n_windows), squeeze=False)
    fig_dca, axes_dca = plt.subplots(n_windows, n_splits, figsize=(6.5 * n_splits, 5.5 * n_windows), squeeze=False)
    
    model_fullname_map = {
        "logistic_regression": "Logistic regression",
        "random_forest": "Random forest",
        "xgboost": "XGBoost",
        "Transformer": "Transformer"
    }

    for w_idx, window_name in enumerate(CLINICAL_WINDOWS.keys()):
        max_dca_y_limit = 0.02
        for s_idx, split in enumerate(SPLIT_TYPES):
            if split not in plotted_data_cache or window_name not in plotted_data_cache[split]: continue
            sub_df, _ = plotted_data_cache[split][window_name]
            prevalence = np.sum(sub_df['y_true'].values == 1) / len(sub_df)
            max_dca_y_limit = max(max_dca_y_limit, prevalence * 1.05)

        for s_idx, split in enumerate(SPLIT_TYPES):
            ax_roc = axes_roc[w_idx, s_idx]
            ax_pr = axes_pr[w_idx, s_idx]
            ax_dca = axes_dca[w_idx, s_idx]

            if split not in plotted_data_cache or window_name not in plotted_data_cache[split]:
                for ax in [ax_roc, ax_pr, ax_dca]:
                    ax.text(0.5, 0.5, "(Not Applicable for this Horizon)", 
                            fontsize=14, color="darkred", ha='center', va='center', weight='bold')
                    ax.set_xlim([0.0, 1.0])
                    ax.set_ylim([0.0, 1.0])
                    ax.grid(False)
                    if w_idx == 0: ax.set_title(split.replace('_', ' ').title(), fontsize=20, fontweight='bold', pad=16)
                    if s_idx == 0: 
                        lbl = "Sensitivity" if ax == ax_roc else ("Precision" if ax == ax_pr else "Net benefit")
                        ax.set_ylabel(f"{window_name}\n\n{lbl}", fontsize=18, fontweight='bold')
                continue

            sub_df, thresholds_resolved = plotted_data_cache[split][window_name]
            y_true = sub_df['y_true'].values
            
            for m_idx, m in enumerate(["logistic_regression", "random_forest", "xgboost", "Transformer"]):
                p_arr = sub_df[f'y_prob_{m}'].values
                full_label = model_fullname_map[m]
                c_hex = COLORS[m_idx]
                
                fpr, tpr, _ = roc_curve(y_true, p_arr)
                ax_roc.plot(fpr, tpr, label=full_label, color=c_hex, lw=3.5)
                
                prec_arr, rec_arr, _ = precision_recall_curve(y_true, p_arr)
                ax_pr.plot(rec_arr, prec_arr, label=full_label, color=c_hex, lw=3.5)
                
                dca_thresh = np.linspace(0.01, 0.50, 50)
                net_benefit = []
                for t in dca_thresh:
                    tp = np.sum((p_arr >= t) & (y_true == 1))
                    fp = np.sum((p_arr >= t) & (y_true == 0))
                    net_benefit.append((tp / len(y_true)) - (fp / len(y_true)) * (t / (1.0 - t)))
                ax_dca.plot(dca_thresh, net_benefit, label=full_label, color=c_hex, lw=3.5)

            ax_roc.plot([0, 1], [0, 1], linestyle="--", color="gray", alpha=0.5)
            ax_roc.set_xlim([0.0, 1.0])
            ax_roc.set_ylim([0.0, 1.05])
            ax_roc.grid(True, linestyle=":", alpha=0.5)
            ax_roc.tick_params(labelsize=14)
            if w_idx == 0: ax_roc.set_title(split.replace('_', ' ').title(), fontsize=20, fontweight='bold', pad=16)
            if s_idx == 0: ax_roc.set_ylabel(f"{window_name}\n\nSensitivity", fontsize=18, fontweight='bold')
            if w_idx == n_windows - 1: ax_roc.set_xlabel("1.0 - Specificity", fontsize=18, fontweight='bold', labelpad=12)
            leg_roc = ax_roc.legend(loc="lower right", fontsize=15, frameon=True, framealpha=0.9)
            for handle in leg_roc.legend_handles: handle.set_linewidth(4.5)

            ax_pr.set_xlim([0.0, 1.0])
            ax_pr.set_ylim([0.0, 1.05])
            ax_pr.grid(True, linestyle=":", alpha=0.5)
            ax_pr.tick_params(labelsize=14)
            if w_idx == 0: ax_pr.set_title(split.replace('_', ' ').title(), fontsize=20, fontweight='bold', pad=16)
            if s_idx == 0: ax_pr.set_ylabel(f"{window_name}\n\nPrecision", fontsize=18, fontweight='bold')
            if w_idx == n_windows - 1: ax_pr.set_xlabel("Recall", fontsize=18, fontweight='bold', labelpad=12)
            leg_pr = ax_pr.legend(loc="upper right", fontsize=15, frameon=True, framealpha=0.9)
            for handle in leg_pr.legend_handles: handle.set_linewidth(4.5)

            prevalence = np.sum(y_true == 1) / len(y_true)
            ax_dca.plot(dca_thresh, np.zeros_like(dca_thresh), color="black", linestyle="-", label="Treat none")
            ax_dca.plot(dca_thresh, prevalence - (1.0 - prevalence) * (dca_thresh / (1.0 - dca_thresh)), color="darkgray", linestyle="--", label="Treat all")
            ax_dca.set_xlim([0.0, 0.5])
            ax_dca.set_ylim([-0.01 * (int(max_dca_y_limit * 10) + 1), max_dca_y_limit])
            ax_dca.grid(True, linestyle=":", alpha=0.5)
            ax_dca.tick_params(labelsize=14)
            if w_idx == 0: ax_dca.set_title(split.replace('_', ' ').title(), fontsize=20, fontweight='bold', pad=16)
            if s_idx == 0: ax_dca.set_ylabel(f"{window_name}\n\nNet benefit", fontsize=18, fontweight='bold')
            if w_idx == n_windows - 1: ax_dca.set_xlabel("Threshold probability", fontsize=18, fontweight='bold', labelpad=12)
            leg_dca = ax_dca.legend(loc="upper right", fontsize=13, frameon=True, framealpha=0.9)
            for handle in leg_dca.legend_handles: handle.set_linewidth(4.5)

    fig_roc.tight_layout()
    fig_roc.savefig(analysis_dir / f"matrix_roc_comparison_curves_{horizon_suffix}_{PARAM_HASH}.png", dpi=200, bbox_inches='tight')
    plt.close(fig_roc)

    fig_pr.tight_layout()
    fig_pr.savefig(analysis_dir / f"matrix_pr_comparison_curves_{horizon_suffix}_{PARAM_HASH}.png", dpi=200, bbox_inches='tight')
    plt.close(fig_pr)

    fig_dca.tight_layout()
    fig_dca.savefig(analysis_dir / f"matrix_dca_comparison_curves_{horizon_suffix}_{PARAM_HASH}.png", dpi=200, bbox_inches='tight')
    plt.close(fig_dca)


# ============================================
# CENTRAL CORE EXECUTIVE PIPELINE ORCHESTRATOR
# ============================================
def process_evaluation_pipeline(task: str, horizon):
    horizon_str = f"horizon_{horizon:04d}d" if isinstance(horizon, int) else "horizon_combined"
    task_horizon_dir = ANALYSIS_SUBDIR / task / horizon_str
    analysis_dir = task_horizon_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    
    plotted_data_cache = {}
    
    for split in SPLIT_TYPES:
        window_cache = process_split_strategy(task, split, horizon, analysis_dir)
        if window_cache is not None:
            plotted_data_cache[split] = window_cache
            
    if plotted_data_cache:
        plot_file_check = analysis_dir / f"matrix_roc_comparison_curves_{horizon_str}_{PARAM_HASH}.png"
        if plot_file_check.exists():
            print(f" -> Visualization plots match parameter signature hash [{PARAM_HASH}]. Skipping plot rendering.")
        else:
            print(f" -> Initializing large-format visual grid compilation maps for {horizon_str}...")
            render_matrix_visualization_grids(plotted_data_cache, analysis_dir, horizon_str)


if __name__ == "__main__":
    for target_task in TASKS:
        for target_horizon in ["combined"] + TARGET_HORIZONS:
            process_evaluation_pipeline(target_task, target_horizon)