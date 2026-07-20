import gc
import re
import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Dict
from pathlib import Path
from collections import defaultdict
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    brier_score_loss, precision_recall_curve, roc_curve,
)
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.contingency_tables import mcnemar  # Added for McNemar's test
from scripts.script_utils import get_best_optuna_run
from joblib import Parallel, delayed

# Configuration
RESULTS_DIR = Path("results_final")
OUTPUT_DIR = RESULTS_DIR / "analysis" / "comparison"
CLASSIC_ML_BASE_DIR = RESULTS_DIR / "classic_ml"
TRANSFORMER_BASE_DIR = RESULTS_DIR / "transformer"
FROM_OPTUNA = ("optuna" in TRANSFORMER_BASE_DIR.name)
BEST_CLASSIC_ML_MODEL = "xgboost"
LOAD_EVALUATION = True
PLOT_FUP_CURVES = False

# TASKS = ["infection_bacteria", "infection_virus", "death", "graft_loss"]
TASKS = ["infection_bacteria"]
SPLIT_TYPES = ["random_split", "temporal_split", "center_split"]
CLASSIC_ML_MODELS_TO_PLOT = ["logistic_regression", "random_forest", "xgboost"]
MODEL_NAME_MAP = {
    "logistic_regression": "Logistic regression",
    "random_forest": "Random forest",
    "xgboost": "XGBoost",
    "Transformer": "Transformer",
}

def get_phase_windows(start, end, horizons, step=30):
    return {h: w for h in horizons if (w := list(range(start, end + 1 - h, step)))}

CLINICAL_PERIODS_INFECTIONS = {
    "Perioperative\n(0-1 mo)": get_phase_windows(  0,  30, [30, 60, 90]),
    "Opportunistic\n(1-6 mo)": get_phase_windows( 30, 180, [30, 60, 90]),
    "Maintenance\n(6-12 mo)":  get_phase_windows(180, 360, [30, 60, 90]),
    "Long-term\n(1-2 yr)":     get_phase_windows(360, 720, [30, 60, 90]),
}
CLINICAL_PERIODS_OUTCOMES = {
    "Short-term\n(0-2 yr)":       get_phase_windows(   0,  360, [360, 720, 1080, 1800]),
    "Middle-term\n(1-3 yr)":      get_phase_windows( 360, 1080, [360, 720, 1080, 1800]),
    "Long-term\n(3-5 yr)":        get_phase_windows(1080, 1800, [360, 720, 1080, 1800]),
    "Very-long-term\n(5-10 yr)":  get_phase_windows(1800, 3600, [360, 720, 1080, 1800]),
}
CLINICAL_PERIOD_DICT = {
    "infection_bacteria": CLINICAL_PERIODS_INFECTIONS,
    "infection_virus": CLINICAL_PERIODS_INFECTIONS,
    "death": CLINICAL_PERIODS_OUTCOMES,
    "graft_loss": CLINICAL_PERIODS_OUTCOMES,
}
PROGNOSTIC_PERIODS_INFECTIONS = {
    "Full length\nhorizon (30 d)":  get_phase_windows(0, 3600, [30]),
    "Full length\nhorizon (60 d)":  get_phase_windows(0, 3600, [60]),
    "Full length\nhorizon (90 d)":  get_phase_windows(0, 3600, [90]),
}
PROGNOSTIC_PERIODS_OUTCOMES = {
    "Full length\nhorizon (360 d)":   get_phase_windows(0, 3600, [ 360]),
    "Full length\nhorizon (720 d)":   get_phase_windows(0, 3600, [ 720]),
    "Full length\nhorizon (1080 d)":  get_phase_windows(0, 3600, [1080]),
    "Full length\nhorizon (1800 d)":  get_phase_windows(0, 3600, [1800]),
}
PROGNOSTIC_PERIOD_DICT = {
    "infection_bacteria": PROGNOSTIC_PERIODS_INFECTIONS,
    "infection_virus": PROGNOSTIC_PERIODS_INFECTIONS,
    "death": PROGNOSTIC_PERIODS_OUTCOMES,
    "graft_loss": PROGNOSTIC_PERIODS_OUTCOMES,
}
EVALUATION_TYPES = {
    "clinical": CLINICAL_PERIOD_DICT,
    "prognostic": PROGNOSTIC_PERIOD_DICT,
}
N_BOOTSTRAP = 1000
MAX_FUP_CURVE_DAYS = 360

DEFAULT_THRESHOLD_SUFFIX = "rec70"  
THRESHOLD_SUFFIX = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_THRESHOLD_SUFFIX

# Parse the suffix into human-readable text
if THRESHOLD_SUFFIX.startswith("rec"):
    try:
        rec_val = int(THRESHOLD_SUFFIX.replace("rec", "")) / 100.0
        threshold_label = f" @ recall {rec_val:.2f}"
    except ValueError:
        threshold_label = f" ({THRESHOLD_SUFFIX})"
elif THRESHOLD_SUFFIX.startswith("t"):
    try:
        t_val = int(THRESHOLD_SUFFIX.replace("t", "")) / 100.0
        threshold_label = f" @ threshold {t_val:.2f}"
    except ValueError:
        threshold_label = f" ({THRESHOLD_SUFFIX})"
else:
    threshold_label = f" ({THRESHOLD_SUFFIX})"

METRICS_OF_INTEREST = {
    "roc_auc": "ROC AUC (→)",
    "pr_auc": "PR AUC (→)",
    "ece": "ECE (←)",
    f"specificity_{THRESHOLD_SUFFIX}": f"Specificity{threshold_label} (→)",
}
PAPER_TABLE_METRICS = [
    "roc_auc",
    "pr_auc",
    "ece",
    f"specificity_{THRESHOLD_SUFFIX}",
]
STAT_TEST_TARGET_METRIC = "pr_auc"
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
Y_LIM_DICT = {
    "roc_auc": (0.5, 1.0),
    "pr_auc": (0.0, 0.5),
    "brier": (0.0, 0.25),
    "ece": (0.0, 0.25),
    "recall": (0.0, 1.0),
    "specificity": (0.0, 1.0),
    "precision": (0.0, 1.0),
    "bal_acc": (0.0, 1.0),
    "f1": (0.0, 1.0),
    "nb": (0.0, 0.1),
    "delta_nb": (0.0, 0.1),
}

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR = OUTPUT_DIR / "cache"
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    
    PLOT_OUT_DIR = OUTPUT_DIR / THRESHOLD_SUFFIX
    PLOT_OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    PERIOD_CACHE_SIG = f"thresh_{THRESHOLD_SUFFIX}_boot_{N_BOOTSTRAP}"
    
    print(">>> Loading raw prediction data for FUP curves...")
    raw_data_pool = load_all_raw_predictions()
    if not raw_data_pool:
        print("No results found. Check your paths in the configuration!")
        return
        
    fup_boot_df = pd.DataFrame()  
    if PLOT_FUP_CURVES:
        fup_cache_path = CACHE_DIR / f"fup_bootstrapped_{PERIOD_CACHE_SIG}.pkl"
        if LOAD_EVALUATION and fup_cache_path.exists():
            print(f">>> Loading cached FUP bootstraps ({PERIOD_CACHE_SIG})...")
            fup_boot_df = pd.read_pickle(fup_cache_path)
        else:       
            print(">>> Computing FUP-specific bootstrapped metrics...")
            fup_boot_df = compute_fup_bootstrap_metrics(
                raw_pool=raw_data_pool, models_to_compare=[BEST_CLASSIC_ML_MODEL, "Transformer"], n_iterations=N_BOOTSTRAP
            )
            fup_boot_df.to_pickle(fup_cache_path)
    else:
        print(">>> Skipping FUP curve calculations as per configuration.")

    period_dfs = {}
    for eval_type, period_dict in EVALUATION_TYPES.items():
        period_cache_path = CACHE_DIR / f"period_df_{eval_type}_{PERIOD_CACHE_SIG}.pkl"
        if LOAD_EVALUATION and period_cache_path.exists():
            print(f">>> Loading cached {eval_type} period metrics ({PERIOD_CACHE_SIG})...")
            period_dfs[eval_type] = pd.read_pickle(period_cache_path)
        else:
            print(f">>> Aggregating into {eval_type} period metrics...")
            period_dfs[eval_type] = compute_period_metrics(raw_data_pool, period_dict, n_iterations=N_BOOTSTRAP)
            period_dfs[eval_type].to_pickle(period_cache_path)
        
    for task in TASKS:
        print(f"\n>>> Processing task: {task.upper()}")
        
        if PLOT_FUP_CURVES and not fup_boot_df.empty:
            task_fup_boot = fup_boot_df[fup_boot_df["Task"] == task].copy()
            if not task_fup_boot.empty:
                plot_fup_curves(task_fup_boot, BEST_CLASSIC_ML_MODEL, "Transformer", OUTPUT_DIR, task)

        for eval_type, p_df in period_dfs.items():
            for split in SPLIT_TYPES:
                compute_overall_prevalence(raw_data_pool, EVALUATION_TYPES[eval_type], task, split)

            task_period = p_df[p_df["Task"] == task].copy()
            if task_period.empty: continue
            
            plot_period_performance_bars(task_period, task, PLOT_OUT_DIR, eval_type)
            generate_performance_summaries(task_period, PLOT_OUT_DIR, task, eval_type)
            generate_paper_style_table(task_period, PLOT_OUT_DIR, task, eval_type)
            
            if raw_data_pool:
                print(f">>> Generating continuous ROC, PR, and net benefit curves for {eval_type} periods...")
                plot_curve_trajectories(raw_data_pool, EVALUATION_TYPES[eval_type], task, OUTPUT_DIR, eval_type, curve_type="ROC")
                plot_curve_trajectories(raw_data_pool, EVALUATION_TYPES[eval_type], task, OUTPUT_DIR, eval_type, curve_type="PR")
                plot_decision_curves(raw_data_pool, EVALUATION_TYPES[eval_type], task, OUTPUT_DIR, eval_type)

        gc.collect()
    
    # Existing Bootstrapped Superiority Test Engine
    significance_df = test_model_superiority(
        df=period_dfs["clinical"],
        baseline_name=BEST_CLASSIC_ML_MODEL,
        target_name="Transformer",
        target_metric=STAT_TEST_TARGET_METRIC,
    )
    significance_df.to_csv(PLOT_OUT_DIR / "statistical_significance.csv", index=False)

    # Added: Running instance-to-instance McNemar contingency analysis across cohorts
    print("\n>>> Executing Contingency Space Testing (McNemar's Test)...")
    mcnemar_df = run_mcnemar_test(
        raw_pool=raw_data_pool,
        period_dict=EVALUATION_TYPES["clinical"],
        baseline_name=BEST_CLASSIC_ML_MODEL,
        target_name="Transformer",
    )
    if not mcnemar_df.empty:
        mcnemar_df.to_csv(PLOT_OUT_DIR / "mcnemar_significance.csv", index=False)
        print(">>> McNemar tests saved to mcnemar_significance.csv")

    print("\nDone.")
    

# -------------------------------------------------
# --- Parallel chunk core task execution engine ---
# -------------------------------------------------

def get_vectorized_metrics_from_bootstraps(
    y_true_boot: np.ndarray,  
    y_prob_boot: np.ndarray,  
    y_cal_boot: np.ndarray,   
) -> dict:
    n_iters, min_len = y_true_boot.shape
    results = {}

    valid_rows_mask = np.any(y_true_boot == 0, axis=1) & np.any(y_true_boot == 1, axis=1)
    valid_indices = np.where(valid_rows_mask)[0]

    roc_aucs = np.full(n_iters, np.nan)
    pr_aucs = np.full(n_iters, np.nan)
    briers = np.full(n_iters, np.nan)
    eces = np.full(n_iters, np.nan)
    
    n_bins = 10
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    
    for idx in valid_indices:
        yt = y_true_boot[idx]
        yp = y_prob_boot[idx]
        yc = y_cal_boot[idx]
            
        roc_aucs[idx] = roc_auc_score(yt, yp)
        pr_aucs[idx] = average_precision_score(yt, yp)
        briers[idx] = brier_score_loss(yt, yc)
        
        bin_indices = np.digitize(yc, bin_boundaries[1:-1])
        ece_val = 0.0
        for b in range(n_bins):
            in_bin = (bin_indices == b)
            n_in_bin = np.sum(in_bin)
            if n_in_bin > 0:
                ece_val += np.abs(np.mean(yt[in_bin]) - np.mean(yc[in_bin])) * n_in_bin
        eces[idx] = ece_val / min_len

    results["roc_auc"] = roc_aucs
    results["pr_auc"] = pr_aucs
    results["brier"] = briers
    results["ece"] = eces

    any_suffix_requested = any(k.endswith(f"_{THRESHOLD_SUFFIX}") for k in METRICS_OF_INTEREST.keys())
    if any_suffix_requested:
        recalls = np.full(n_iters, np.nan)
        specificities = np.full(n_iters, np.nan)
        precisions = np.full(n_iters, np.nan)
        bal_accs = np.full(n_iters, np.nan)
        f1_scores = np.full(n_iters, np.nan)
        net_benefits = np.full(n_iters, np.nan)
        delta_net_benefits = np.full(n_iters, np.nan)

        if THRESHOLD_SUFFIX.startswith("rec"):
            target_rec = int(THRESHOLD_SUFFIX.replace("rec", "")) / 100.0
            rec_thresholds = np.zeros(n_iters)
            
            for idx in valid_indices:
                p, r, t = precision_recall_curve(y_true_boot[idx], y_cal_boot[idx])
                valid_idx = np.where(r[:-1] >= target_rec)[0]
                rec_thresholds[idx] = t[valid_idx[-1]] if len(valid_idx) > 0 else 0.0
                
            thresh_matrix = rec_thresholds[:, np.newaxis]
            preds = (y_cal_boot >= thresh_matrix).astype(int)
        else:
            t_val = int(THRESHOLD_SUFFIX.replace("t", "")) / 100.0
            preds = (y_cal_boot >= t_val).astype(int)
            rec_thresholds = np.full(n_iters, t_val)

        tp = np.sum((preds == 1) & (y_true_boot == 1), axis=1)
        fp = np.sum((preds == 1) & (y_true_boot == 0), axis=1)
        tn = np.sum((preds == 0) & (y_true_boot == 0), axis=1)
        fn = np.sum((preds == 0) & (y_true_boot == 1), axis=1)
        
        positives = tp + fn
        negatives = fp + tn
        
        with np.errstate(divide='ignore', invalid='ignore'):
            computed_recalls = np.where(positives > 0, tp / positives, 0.0)
            computed_specificities = np.where(negatives > 0, tn / negatives, 0.0)
            computed_precisions = np.where((tp + fp) > 0, tp / (tp + fp), 0.0)
            computed_bal_accs = (computed_recalls + computed_specificities) / 2.0
            
            f1_denom = computed_precisions + computed_recalls
            computed_f1_scores = np.where(f1_denom > 0, (2 * computed_precisions * computed_recalls) / f1_denom, 0.0)
            
            weight = rec_thresholds / (1.0 - rec_thresholds)
            weight = np.where(rec_thresholds < 1.0, weight, 0.0)
            computed_net_benefits = (tp / min_len) - (fp / min_len) * weight
            
            prevalence = positives / min_len
            nb_all = prevalence - (1.0 - prevalence) * weight
            computed_delta_net_benefits = computed_net_benefits - np.maximum(nb_all, 0.0)

        recalls[valid_rows_mask] = computed_recalls[valid_rows_mask]
        specificities[valid_rows_mask] = computed_specificities[valid_rows_mask]
        precisions[valid_rows_mask] = computed_precisions[valid_rows_mask]
        bal_accs[valid_rows_mask] = computed_bal_accs[valid_rows_mask]
        f1_scores[valid_rows_mask] = computed_f1_scores[valid_rows_mask]
        net_benefits[valid_rows_mask] = computed_net_benefits[valid_rows_mask]
        delta_net_benefits[valid_rows_mask] = computed_delta_net_benefits[valid_rows_mask]

        results[f"recall_{THRESHOLD_SUFFIX}"] = recalls
        results[f"specificity_{THRESHOLD_SUFFIX}"] = specificities
        results[f"precision_{THRESHOLD_SUFFIX}"] = precisions
        results[f"bal_acc_{THRESHOLD_SUFFIX}"] = bal_accs
        results[f"f1_{THRESHOLD_SUFFIX}"] = f1_scores
        results[f"nb_{THRESHOLD_SUFFIX}"] = net_benefits
        results[f"delta_nb_{THRESHOLD_SUFFIX}"] = delta_net_benefits

    return results


def _process_single_fup(split, task, h, fup, raw_pool, models_to_compare, n_iterations):
    models = raw_pool[split]
    model_arrays = {}
    
    for m in models_to_compare:
        if m in models and task in models[m] and h in models[m][task] and fup in models[m][task][h]:
            d = models[m][task][h][fup]
            y_t = d["labels"]
            mask = y_t != -100
            if mask.any():
                model_arrays[m] = {
                    "true": y_t[mask],
                    "prob": d["probs"][mask],
                    "cal": d["probs_cal"][mask] if "probs_cal" in d else d["probs"][mask]
                }
    
    if len(model_arrays) != len(models_to_compare):
        return []

    min_len = min(len(arr["true"]) for arr in model_arrays.values())
    for m in model_arrays:
        model_arrays[m]["true"] = model_arrays[m]["true"][:min_len]
        model_arrays[m]["prob"] = model_arrays[m]["prob"][:min_len]
        model_arrays[m]["cal"] = model_arrays[m]["cal"][:min_len]

    ref_model = models_to_compare[0]
    y_true_ref = model_arrays[ref_model]["true"]
    classes, counts = np.unique(y_true_ref, return_counts=True)
    
    rng = np.random.default_rng()
    boot_indices = np.empty((n_iterations, min_len), dtype=int)
    ptr = 0
    for cls, count in zip(classes, counts):
        cls_locs = np.where(y_true_ref == cls)[0]
        sampled_raw = rng.choice(cls_locs, size=(n_iterations, count), replace=True)
        boot_indices[:, ptr : ptr + count] = sampled_raw
        ptr += count

    local_records = []
    for model, arrays in model_arrays.items():
        y_true_boot = arrays["true"][boot_indices]
        y_prob_boot = arrays["prob"][boot_indices]
        y_cal_boot = arrays["cal"][boot_indices]
        
        metrics_dict = get_vectorized_metrics_from_bootstraps(y_true_boot, y_prob_boot, y_cal_boot)
        
        for m_name, iterations_array in metrics_dict.items():
            if m_name in METRICS_OF_INTEREST:
                valid_mask = ~np.isnan(iterations_array)
                indices = np.where(valid_mask)[0]
                vals = iterations_array[valid_mask]
                
                if len(vals) == 0: continue
                
                chunk_df = pd.DataFrame({
                    "Split": split, "Model": model, "Task": task,
                    "Horizon": h, "FUP": fup, "Metric": m_name,
                    "Value": vals, "Bootstrap_iter": indices
                })
                local_records.append(chunk_df)
                
    return local_records


def compute_fup_bootstrap_metrics(raw_pool: Dict, models_to_compare: list, n_iterations: int = 1000) -> pd.DataFrame:
    ref_model = models_to_compare[0]
    print(">>>> Collecting valid FUP coordinates...")
    fup_coordinates = []
    for split, models in raw_pool.items():
        if ref_model not in models: continue
        for task in TASKS:
            if task not in models[ref_model]: continue
            for h, fups in models[ref_model][task].items():
                for fup in fups.keys():
                    fup_coordinates.append((split, task, h, fup))

    print(f">>>> Found {len(fup_coordinates)} FUP groups within {MAX_FUP_CURVE_DAYS} days. Parallelizing Workflow...")
    
    results = Parallel(n_jobs=-1)(
        delayed(_process_single_fup)(split, task, h, fup, raw_pool, models_to_compare, n_iterations)
        for split, task, h, fup in tqdm(fup_coordinates, desc="Bootstrapping FUPs")
    )
    
    flattened_dfs = [df for sublist in results for df in sublist if isinstance(df, pd.DataFrame)]
    if not flattened_dfs:
        return pd.DataFrame()
    return pd.concat(flattened_dfs, ignore_index=True)


def compute_period_metrics(raw_pool: Dict, period_dict: Dict, do_bootstrapping: bool = True, n_iterations: int = 1000) -> pd.DataFrame:
    all_period_dfs = []
    
    for split, models in raw_pool.items():
        available_models = list(models.keys())
        if not available_models: continue
            
        for task, clinical_periods in period_dict.items():
            for period_name, h_fup_map in clinical_periods.items():
                model_arrays = {}
                for model in available_models:
                    if task not in models[model]: continue
                    horizons_data = models[model][task]
                    y_t_list, y_p_list, y_c_list = [], [], []
                    
                    for h, fups in h_fup_map.items():
                        if h not in horizons_data: continue
                        for fup in fups:
                            if fup in horizons_data[h]:
                                d = horizons_data[h][fup]
                                valid_mask = d["labels"] != -100
                                if valid_mask.any():
                                    y_t_list.append(d["labels"][valid_mask])
                                    y_p_list.append(d["probs"][valid_mask])
                                    y_c_list.append((d["probs_cal"] if "probs_cal" in d else d["probs"])[valid_mask])
                                
                    if y_t_list:
                        model_arrays[model] = {
                            "true": np.concatenate(y_t_list),
                            "prob": np.concatenate(y_p_list),
                            "cal": np.concatenate(y_c_list)
                        }

                if not model_arrays: continue
                
                min_len = min(len(arr["true"]) for arr in model_arrays.values())
                for m in model_arrays:
                    model_arrays[m]["true"] = model_arrays[m]["true"][:min_len]
                    model_arrays[m]["prob"] = model_arrays[m]["prob"][:min_len]
                    model_arrays[m]["cal"] = model_arrays[m]["cal"][:min_len]
                
                ref_model = list(model_arrays.keys())[0]
                y_true_ref = model_arrays[ref_model]["true"]
                classes, counts = np.unique(y_true_ref, return_counts=True)
                
                rng = np.random.default_rng()
                iters = n_iterations if do_bootstrapping else 1
                
                print(f">>>> Bootstrapping period: {split} - {task} - {period_name.splitlines()[0]}")
                
                boot_indices = np.empty((iters, min_len), dtype=int)
                if do_bootstrapping:
                    ptr = 0
                    for cls, count in zip(classes, counts):
                        cls_locs = np.where(y_true_ref == cls)[0]
                        boot_indices[:, ptr : ptr + count] = rng.choice(cls_locs, size=(iters, count), replace=True)
                        ptr += count
                else:
                    boot_indices[0, :] = np.arange(min_len)

                for model, arrays in model_arrays.items():
                    y_true_boot = arrays["true"][boot_indices]
                    y_prob_boot = arrays["prob"][boot_indices]
                    y_cal_boot = arrays["cal"][boot_indices]
                    
                    metrics_dict = get_vectorized_metrics_from_bootstraps(y_true_boot, y_prob_boot, y_cal_boot)
                    
                    for m_name, iterations_array in metrics_dict.items():
                        if m_name in METRICS_OF_INTEREST:
                            valid_mask = ~np.isnan(iterations_array)
                            indices = np.where(valid_mask)[0]
                            vals = iterations_array[valid_mask]
                            
                            if len(vals) == 0: continue
                            
                            chunk_df = pd.DataFrame({
                                "Split": split, "Model": model, "Period": period_name,
                                "Task": task, "Metric": m_name, "Value": vals
                            })
                            if do_bootstrapping:
                                chunk_df["Bootstrap_iter"] = indices
                            all_period_dfs.append(chunk_df)
                                
    if not all_period_dfs:
        return pd.DataFrame()
    return pd.concat(all_period_dfs, ignore_index=True)


# -------------------------------------------------------------------------
# --- Downstream Plotting and IO Scripts ----------------------------------
# -------------------------------------------------------------------------

def load_all_raw_predictions() -> Dict:
    raw_pool = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict))))
    for split in SPLIT_TYPES:
        for task in TASKS:
            if FROM_OPTUNA:
                try:
                    trial_name, pt_config = get_best_optuna_run(TRANSFORMER_BASE_DIR, split, task)
                    base_dir = TRANSFORMER_BASE_DIR / split / task / trial_name / split / pt_config
                except Exception as e:
                    print(f"[!] Skipping {split} / {task} due to error: {e}")
                    continue
            else:
                try:
                    [base_dir] = (d for d in (TRANSFORMER_BASE_DIR / split).iterdir() if d.is_dir())
                except ValueError:
                    continue
                
            t_path = base_dir / "finetuning" / task
            if t_path.exists():
                _extract_npz_to_pool(t_path, "Transformer", split, task, raw_pool)
                
            for ml_model in CLASSIC_ML_MODELS_TO_PLOT:
                c_path = CLASSIC_ML_BASE_DIR / split / ml_model / task
                if c_path.exists():
                    _extract_npz_to_pool(c_path, ml_model, split, task, raw_pool)
                    
    return raw_pool


def _extract_npz_to_pool(task_dir: Path, model_name: str, split: str, task: str, pool: Dict):
    for npz_file in task_dir.rglob("*.npz"):
        match = re.search(r"hrz\(([\d-]+)\)", npz_file.parent.name)
        if not match: continue
        horizons = [int(h) for h in match.group(1).split("-")]
        
        try: data = np.load(npz_file)
        except Exception: continue
            
        fup_keys = {int(re.match(r"^test_fup_(\d+)_", k).group(1)) for k in data.keys() if re.match(r"^test_fup_(\d+)_", k)}
        for fup in fup_keys:
            l_key, p_key, pc_key = f"test_fup_{fup:04d}_labels", f"test_fup_{fup:04d}_probs", f"test_fup_{fup:04d}_probs_cal"
            if l_key not in data or p_key not in data: continue
            
            labels, probs = data[l_key], data[p_key]
            probs_cal = data[pc_key] if pc_key in data else probs
            
            for col_idx, h in enumerate(horizons):
                l_col = labels if labels.ndim == 1 else labels[:, col_idx]
                p_col = probs if probs.ndim == 1 else probs[:, col_idx]
                pc_col = probs_cal if probs_cal.ndim == 1 else probs_cal[:, col_idx]
                    
                pool[split][model_name][task][h][fup] = {"labels": l_col, "probs": p_col, "probs_cal": pc_col}


def plot_fup_curves(df: pd.DataFrame, baseline_name: str, transformer_name: str, output_dir: Path, task_name: str):
    df_comp = df[df["Model"].isin([baseline_name, transformer_name])].copy()
    if df_comp.empty: return
    
    summary = df_comp.groupby(["Split", "Horizon", "FUP", "Metric", "Model"])["Value"].agg(
        mean="mean", ci_lower=lambda x: np.percentile(x, 2.5), ci_upper=lambda x: np.percentile(x, 97.5)
    ).reset_index()
    
    metrics = list(METRICS_OF_INTEREST.keys())
    horizons = sorted(summary["Horizon"].unique())
    colors = {baseline_name: "#ff7f0e", transformer_name: "#1f77b4"}
    
    for metric in metrics:
        metric_data = summary[summary["Metric"] == metric]
        if metric_data.empty: continue
        
        fig, axes = plt.subplots(
            len(horizons), len(SPLIT_TYPES), 
            figsize=(5 * len(SPLIT_TYPES), 3.5 * len(horizons)),
            sharex=True, sharey=True, squeeze=False
        )
        
        title_suffix = task_name.replace('_', ' ').capitalize()
        tf_title = MODEL_NAME_MAP.get(transformer_name, transformer_name)
        ml_title = MODEL_NAME_MAP.get(baseline_name, baseline_name)
        fig.suptitle(f"{METRICS_OF_INTEREST[metric]} across follow-up time {tf_title} vs {ml_title} ({title_suffix})", fontsize=16, fontweight='bold', y=1.02)
        
        y_limits = None
        for base_metric, limits in Y_LIM_DICT.items():
            if metric.startswith(base_metric):
                y_limits = limits
                break
                
        for row_idx, h in enumerate(horizons):
            for col_idx, split in enumerate(SPLIT_TYPES):
                ax = axes[row_idx, col_idx]
                subset = metric_data[(metric_data["Horizon"] == h) & (metric_data["Split"] == split)]
                if subset.empty:
                    ax.set_visible(False)
                    continue
                    
                for model in [baseline_name, transformer_name]:
                    model_data = subset[subset["Model"] == model].sort_values("FUP")
                    if model_data.empty: continue
                    
                    c = colors[model]
                    ax.plot(model_data["FUP"], model_data["mean"], label=MODEL_NAME_MAP.get(model, model), color=c, linewidth=2)
                    ax.fill_between(model_data["FUP"], model_data["ci_lower"], model_data["ci_upper"], color=c, alpha=0.2)
                
                if MAX_FUP_CURVE_DAYS is not None: ax.set_xlim(0, MAX_FUP_CURVE_DAYS)
                if y_limits: ax.set_ylim(y_limits)
                if row_idx == 0: ax.set_title(split.capitalize().replace('_', ' '), fontsize=15, fontweight='bold')
                if col_idx == 0: ax.set_ylabel(f"Horizon {h}d", fontsize=13, fontweight='bold')
                if row_idx == len(horizons) - 1:
                    ax.set_xlabel("Follow-Up Time (days)", fontsize=13, fontweight='bold')
                    ax.tick_params(labelbottom=True)
                    
        handles, labels = axes[0,0].get_legend_handles_labels()
        if handles: fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.05), ncol=2, fontsize=14, frameon=False)
            
        plt.tight_layout()
        out_file = output_dir / f"fup_curves_{task_name}_{metric}.png"
        plt.savefig(out_file, bbox_inches='tight', dpi=150)
        plt.close()


def test_model_superiority(
    df: pd.DataFrame, baseline_name: str,
    target_name: str = "Transformer",
    target_metric: str = None,
):
    print(f"\n>>> Statistical testing: {target_name} vs {baseline_name}")
    df_comp = df[df["Model"].isin([baseline_name, target_name])].copy()
    if df_comp.empty: return pd.DataFrame()
    
    if target_metric is not None:
        df_comp = df_comp[df_comp["Metric"] == target_metric]
        if df_comp.empty:
            print(f"[!] Warning: No data found for metric '{target_metric}'")
            return pd.DataFrame()
    
    pivot = df_comp.pivot_table(
        index=["Split", "Task", "Period", "Metric", "Bootstrap_iter"],
        columns="Model", values="Value",
    ).dropna().reset_index()
    if target_name not in pivot.columns or baseline_name not in pivot.columns:
        return pd.DataFrame()

    pivot["Delta"] = pivot[target_name] - pivot[baseline_name]
    
    stats = pivot.groupby(["Split", "Task", "Period", "Metric"]).agg(
        delta_mean=("Delta", "mean"),
        ci_lower=("Delta", lambda x: np.percentile(x, 2.5)),
        ci_upper=("Delta", lambda x: np.percentile(x, 97.5)),
        p_value_raw=("Delta", lambda x: 2 * min(np.mean(x <= 0), np.mean(x >= 0)))
    ).reset_index()
    
    stats["p_value_raw"] = stats["p_value_raw"].replace(0, 1 / df["Bootstrap_iter"].nunique())
    reject, pvals_corrected, _, _ = multipletests(stats["p_value_raw"], alpha=0.05, method='fdr_bh')
    
    stats["p_value_fdr"] = pvals_corrected
    stats["significant"] = reject

    # --- Print Winner Log Block ---
    metric_lbl = target_metric if target_metric else "All Metrics"
    print(f"\n=== Bootstrap Superiority Summary Results ({metric_lbl}) ===")
    for _, row in stats.iterrows():
        p_clean = row['Period'].replace('\n', ' ')
        
        if row['significant']:
            winner = target_name if row['delta_mean'] > 0 else baseline_name
        else:
            winner = "None (Not Significant)"
            
        print(f"[{row['Split']} - {p_clean} - {row['Metric']}] Winner: {winner} (Delta Mean={row['delta_mean']:.4f}, FDR p={row['p_value_fdr']:.4f})")
    print("=========================================================================")

    return stats


def run_mcnemar_test(raw_pool: Dict, period_dict: Dict, baseline_name: str, target_name: str = "Transformer") -> pd.DataFrame:
    """
    Computes McNemar's test from instance-to-instance binary predictions across all cohorts.
    Thresholding logic correctly adapts to the selected config token ('recXX' or 'tXX').
    Prints the winning model architecture for each clinical space.
    """
    mcnemar_records = []

    for split, models in raw_pool.items():
        if baseline_name not in models or target_name not in models:
            continue

        for task, clinical_periods in period_dict.items():
            for period_name, h_fup_map in clinical_periods.items():
                
                # Step 1: Align paired instance timelines for both candidate models
                paired_data = {baseline_name: {"true": [], "cal": []}, target_name: {"true": [], "cal": []}}
                
                for model in [baseline_name, target_name]:
                    horizons_data = models[model][task]
                    for h, fups in h_fup_map.items():
                        if h not in horizons_data: continue
                        for fup in fups:
                            if fup in horizons_data[h]:
                                d = horizons_data[h][fup]
                                mask = d["labels"] != -100
                                if mask.any():
                                    paired_data[model]["true"].append(d["labels"][mask])
                                    p_cal = d["probs_cal"][mask] if "probs_cal" in d else d["probs"][mask]
                                    paired_data[model]["cal"].append(p_cal)

                if not paired_data[baseline_name]["true"] or not paired_data[target_name]["true"]:
                    continue

                y_true_bl = np.concatenate(paired_data[baseline_name]["true"])
                y_cal_bl = np.concatenate(paired_data[baseline_name]["cal"])
                
                y_true_tg = np.concatenate(paired_data[target_name]["true"])
                y_cal_tg = np.concatenate(paired_data[target_name]["cal"])

                min_len = min(len(y_true_bl), len(y_true_tg))
                if min_len == 0: continue
                
                y_true = y_true_bl[:min_len]
                y_cal_bl = y_cal_bl[:min_len]
                y_cal_tg = y_cal_tg[:min_len]

                # Step 2: Resolve Decision Threshold spaces
                preds_bin = {}
                for m_name, y_cal_arr in [(baseline_name, y_cal_bl), (target_name, y_cal_tg)]:
                    if THRESHOLD_SUFFIX.startswith("rec"):
                        target_rec = int(THRESHOLD_SUFFIX.replace("rec", "")) / 100.0
                        p, r, t = precision_recall_curve(y_true, y_cal_arr)
                        valid_idx = np.where(r[:-1] >= target_rec)[0]
                        thresh = t[valid_idx[-1]] if len(valid_idx) > 0 else 0.5
                        preds_bin[m_name] = (y_cal_arr >= thresh).astype(int)
                    else:
                        t_val = int(THRESHOLD_SUFFIX.replace("t", "")) / 100.0
                        preds_bin[m_name] = (y_cal_arr >= t_val).astype(int)

                # Step 3: Map out Contingency Matrix elements
                bl_correct = (preds_bin[baseline_name] == y_true)
                tg_correct = (preds_bin[target_name] == y_true)

                a = np.sum(tg_correct & bl_correct)
                b = np.sum(tg_correct & ~bl_correct)  # Target right, Baseline wrong
                c = np.sum(~tg_correct & bl_correct)  # Baseline right, Target wrong
                d = np.sum(~tg_correct & ~bl_correct)

                table = [[a, b], [c, d]]
                
                try:
                    res = mcnemar(table, exact=True)
                    p_val = res.pvalue
                except Exception:
                    p_val = np.nan

                mcnemar_records.append({
                    "Split": split, "Task": task, "Period": period_name.replace('\n', ' '),
                    "Threshold_Used": THRESHOLD_SUFFIX, "Discordant_Target_Only(b)": b,
                    "Discordant_Baseline_Only(c)": c, "McNemar_p_value_raw": p_val
                })

    if not mcnemar_records:
        return pd.DataFrame()

    mcn_df = pd.DataFrame(mcnemar_records)
    valid_mask = ~mcn_df["McNemar_p_value_raw"].isna()
    
    if valid_mask.any():
        reject, pvals_corrected, _, _ = multipletests(mcn_df.loc[valid_mask, "McNemar_p_value_raw"], alpha=0.05, method='fdr_bh')
        mcn_df.loc[valid_mask, "McNemar_p_value_fdr"] = pvals_corrected
        mcn_df.loc[valid_mask, "Significant_McNemar"] = reject

    # --- Print Winner Log Block ---
    print("\n=== McNemar Test Summary Results ===")
    for _, row in mcn_df.iterrows():
        p_clean = row['Period']
        b_count = row['Discordant_Target_Only(b)']
        c_count = row['Discordant_Baseline_Only(c)']
        
        if row.get('Significant_McNemar', False):
            winner = target_name if b_count > c_count else baseline_name
        else:
            winner = "None (Not Significant)"
            
        print(f"[{row['Split']} - {p_clean}] Winner: {winner} (b={b_count}, c={c_count}, FDR p={row['McNemar_p_value_fdr']:.4f})")
    print("====================================")
    
    return mcn_df



def plot_period_performance_bars(df: pd.DataFrame, task_name: str, output_dir: Path, eval_type: str):
    metrics = list(METRICS_OF_INTEREST.keys())
    model_order = CLASSIC_ML_MODELS_TO_PLOT + ["Transformer"]
    col_keys = [m for m in model_order if m in df["Model"].unique()]
    if not metrics or not col_keys: return

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(len(metrics), len(SPLIT_TYPES), figsize=(6 * len(SPLIT_TYPES), 4.5 * len(metrics)), sharex=True, sharey="row", squeeze=False)

    for r, metric in enumerate(metrics):
        y_limits = None
        for base_metric, limits in Y_LIM_DICT.items():
            if metric.startswith(base_metric):
                y_limits = limits
                break
            
        for c, split in enumerate(SPLIT_TYPES):
            ax = axes[r, c]
            subset = df[(df["Metric"] == metric) & (df["Split"] == split)]
            if subset.empty:
                ax.set_visible(False)
                continue

            sns.barplot(
                data=subset, x="Period", y="Value", hue="Model",
                hue_order=col_keys, palette=COLORS[:len(col_keys)],
                ax=ax, edgecolor='black', linewidth=0.6, alpha=0.9,
            )
            if y_limits is not None: ax.set_ylim(y_limits)
            if r == 0: ax.set_title(split.replace('_', ' ').lower().capitalize(), fontsize=16, fontweight='bold')
            if c == 0: ax.set_ylabel(f"{METRICS_OF_INTEREST[metric]}", fontsize=14, fontweight='bold')
            else: ax.set_ylabel("")
            ax.set_xlabel(
                "Clinical period" if r == len(metrics) - 1 else "", 
                fontsize=14, fontweight='bold', labelpad=12,
            )
            ax.tick_params(labelbottom=(r == len(metrics)-1), labelsize=14)
            if ax.get_legend(): ax.get_legend().remove()

    handles, labels = axes[0,0].get_legend_handles_labels()
    pretty_labels = [MODEL_NAME_MAP.get(lbl, lbl) for lbl in labels]
    
    from matplotlib.patches import Patch
    title_proxy = Patch(color='none', label='') 
    title_text = f"Model architecture ({task_name.replace('_', ' ')}):"
    
    all_handles = [title_proxy] + handles
    all_labels = [title_text] + pretty_labels

    leg = fig.legend(
        all_handles, all_labels, 
        loc="lower center", 
        bbox_to_anchor=(0.5, 0.99), 
        ncol=len(all_labels),        
        fontsize=16,                
        frameon=False,
        columnspacing=1.6,          
        handletextpad=0.6            
    )
    
    handles_to_resize = leg.legend_handles if hasattr(leg, 'legend_handles') else leg.legendHandles
    
    for i, text in enumerate(leg.get_texts()):
        if i == 0:
            current_sz = 17
            text.set_weight('bold')
            text.set_fontsize(current_sz)   
            if hasattr(handles_to_resize[i], 'set_visible'):
                handles_to_resize[i].set_visible(False)
        else:
            if hasattr(handles_to_resize[i], 'set_sizes'): 
                handles_to_resize[i].set_sizes([250.0]) 

    plt.tight_layout()
    plt.savefig(output_dir / f"{eval_type}_bars_{task_name}.png", bbox_inches='tight', dpi=150)
    plt.close()


def plot_curve_trajectories(raw_pool: Dict, period_dict: Dict, task_name: str, output_dir: Path, eval_type: str, curve_type: str = "ROC"):
    clinical_periods = period_dict.get(task_name, {})
    if not clinical_periods: return
    
    model_order = CLASSIC_ML_MODELS_TO_PLOT + ["Transformer"]
    
    sns.set_theme(style="white")
    fig, axes = plt.subplots(
        len(clinical_periods), len(SPLIT_TYPES), 
        figsize=(5.5 * len(SPLIT_TYPES), 4.5 * len(clinical_periods)), 
        squeeze=False
    )
    
    legend_tracker = {}

    for r, (period_name, h_fup_map) in enumerate(clinical_periods.items()):
        for c, split in enumerate(SPLIT_TYPES):
            ax = axes[r, c]
            models_data = raw_pool.get(split, {})
            
            if curve_type == "ROC":
                ax.plot([0, 1], [0, 1], linestyle="--", color="gray", alpha=0.6, label="Baseline (0.50)")
            
            for m_idx, model in enumerate(model_order):
                if model not in models_data or task_name not in models_data[model]: continue
                
                horizons_data = models_data[model][task_name]
                y_t_list, y_p_list = [], []
                
                for h, fups in h_fup_map.items():
                    if h not in horizons_data: continue
                    for fup in fups:
                        if fup in horizons_data[h]:
                            d = horizons_data[h][fup]
                            valid_mask = d["labels"] != -100
                            if valid_mask.any():
                                y_t_list.append(d["labels"][valid_mask])
                                y_p_list.append(d["probs"][valid_mask])
                                
                if not y_t_list: continue
                
                y_true = np.concatenate(y_t_list)
                y_prob = np.concatenate(y_p_list)
                
                if len(np.unique(y_true)) < 2: continue
                
                if curve_type == "ROC":
                    fpr, tpr, _ = roc_curve(y_true, y_prob)
                    auc_val = roc_auc_score(y_true, y_prob)
                    lbl = f"{MODEL_NAME_MAP.get(model, model)} (AUC={auc_val:.3f})"
                    line, = ax.plot(fpr, tpr, color=COLORS[m_idx], linewidth=2, label=lbl)
                    legend_tracker[MODEL_NAME_MAP.get(model, model)] = line
                else:
                    precision, recall, _ = precision_recall_curve(y_true, y_prob)
                    auc_val = average_precision_score(y_true, y_prob)
                    lbl = f"{MODEL_NAME_MAP.get(model, model)} (PR-AUC={auc_val:.3f})"
                    line, = ax.plot(recall, precision, color=COLORS[m_idx], linewidth=2, label=lbl)
                    legend_tracker[MODEL_NAME_MAP.get(model, model)] = line
            
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.grid(True, linestyle=":", alpha=0.6)
            
            if r == 0: 
                ax.set_title(split.replace('_', ' ').capitalize(), fontsize=16, fontweight='bold')
            if c == 0: 
                ax.set_ylabel(f"{period_name}\n\n" + ("Sensitivity" if curve_type == "ROC" else "Precision"), fontsize=15, fontweight='bold')
            if r == len(clinical_periods) - 1: 
                ax.set_xlabel("1.0 - Specificity" if curve_type == "ROC" else "Recall", fontsize=15, fontweight='bold', labelpad=12)
            
            leg = ax.legend(loc="lower right" if curve_type=="ROC" else "upper right", fontsize=14, frameon=True, framealpha=0.8)
            for handle in leg.legend_handles:
                handle.set_linewidth(4.0)

    title_task = task_name.replace('_', ' ').title()
    fig.suptitle(f"Continuous Performance Spaces ({curve_type} Curves): {title_task}", fontsize=18, fontweight='bold', y=1.01)
    plt.tight_layout()
    
    out_name = output_dir / f"{eval_type}_{curve_type.lower()}_curves_{task_name}.png"
    plt.savefig(out_name, bbox_inches='tight', dpi=150)
    plt.close()


def plot_decision_curves(raw_pool: Dict, period_dict: Dict, task_name: str, output_dir: Path, eval_type: str):
    clinical_periods = period_dict.get(task_name, {})
    if not clinical_periods: return
    
    model_order = CLASSIC_ML_MODELS_TO_PLOT + ["Transformer"]
    thresholds = np.linspace(0.01, 0.99, 100)
    
    sns.set_theme(style="white")
    fig, axes = plt.subplots(
        len(clinical_periods), len(SPLIT_TYPES), 
        figsize=(5.5 * len(SPLIT_TYPES), 4.5 * len(clinical_periods)), 
        squeeze=False
    )
    
    for r, (period_name, h_fup_map) in enumerate(clinical_periods.items()):
        for c, split in enumerate(SPLIT_TYPES):
            ax = axes[r, c]
            models_data = raw_pool.get(split, {})
            
            y_true_ref = None
            for model in model_order:
                if model in models_data and task_name in models_data[model]:
                    horizons_data = models_data[model][task_name]
                    y_t_list = []
                    for h, fups in h_fup_map.items():
                        if h not in horizons_data: continue
                        for fup in fups:
                            if fup in horizons_data[h]:
                                mask = horizons_data[h][fup]["labels"] != -100
                                if mask.any():
                                    y_t_list.append(horizons_data[h][fup]["labels"][mask])
                    if y_t_list:
                        y_true_ref = np.concatenate(y_t_list)
                        break
            
            if y_true_ref is None or len(y_true_ref) == 0:
                ax.set_visible(False)
                continue
                
            n_samples = len(y_true_ref)
            n_positives = np.sum(y_true_ref == 1)
            prevalence = n_positives / n_samples
            
            ax.plot(thresholds, np.zeros_like(thresholds), color="black", linestyle="-", linewidth=1.2, label="Treat none")
            
            with np.errstate(divide='ignore', invalid='ignore'):
                nb_all = prevalence - (1.0 - prevalence) * (thresholds / (1.0 - thresholds))
            ax.plot(thresholds, nb_all, color="darkgray", linestyle="--", linewidth=1.5, label="Treat all")
            
            max_nb_plotted = 0.02
            
            for m_idx, model in enumerate(model_order):
                if model not in models_data or task_name not in models_data[model]: continue
                
                horizons_data = models_data[model][task_name]
                y_t_list, y_p_list = [], []
                
                for h, fups in h_fup_map.items():
                    if h not in horizons_data: continue
                    for fup in fups:
                        if fup in horizons_data[h]:
                            d = horizons_data[h][fup]
                            valid_mask = d["labels"] != -100
                            if valid_mask.any():
                                y_t_list.append(d["labels"][valid_mask])
                                y_p_list.append((d["probs_cal"] if "probs_cal" in d else d["probs"])[valid_mask])
                                
                if not y_t_list: continue
                
                y_true = np.concatenate(y_t_list)
                y_prob = np.concatenate(y_p_list)
                
                nb_model = []
                for t in thresholds:
                    tp = np.sum((y_prob >= t) & (y_true == 1))
                    fp = np.sum((y_prob >= t) & (y_true == 0))
                    val = (tp / n_samples) - (fp / n_samples) * (t / (1.0 - t))
                    nb_model.append(val)
                
                max_nb_plotted = max(max_nb_plotted, max(nb_model), prevalence)
                ax.plot(thresholds, nb_model, color=COLORS[m_idx], linewidth=2.2, label=MODEL_NAME_MAP.get(model, model))
                
            ax.set_xlim([0.0, 0.5])
            ax.set_ylim([-0.02, max_nb_plotted * 1.0])
            ax.grid(True, linestyle=":", alpha=0.5)
            
            if r == 0: 
                ax.set_title(split.replace('_', ' ').capitalize(), fontsize=16, fontweight='bold')
            if c == 0: 
                ax.set_ylabel(f"{period_name}\n\nNet benefit", fontsize=15, fontweight='bold')
            if r == len(clinical_periods) - 1: 
                ax.set_xlabel("Threshold probability", fontsize=15, fontweight='bold', labelpad=12)
            
            leg = ax.legend(loc="upper right", fontsize=14, frameon=True, framealpha=0.8)
            for handle in leg.legend_handles:
                handle.set_linewidth(3.5)

    title_task = task_name.replace('_', ' ').title()
    plt.tight_layout()
    
    out_name = output_dir / f"{eval_type}_decision_curves_{task_name}.png"
    plt.savefig(out_name, bbox_inches='tight', dpi=150)
    plt.close()
    
    
def generate_paper_style_table(df: pd.DataFrame, output_dir: Path, task_name: str, eval_type: str):
    valid_periods = list(EVALUATION_TYPES[eval_type][task_name].keys())
    df_filtered = df[(df["Metric"].isin(PAPER_TABLE_METRICS)) & (df["Period"].isin(valid_periods))].copy()
    if df_filtered.empty: return

    df_mean = df_filtered.groupby(["Split", "Model", "Metric", "Period"])["Value"].mean().reset_index() if "Bootstrap_iter" in df_filtered.columns else df_filtered.copy()
    clean_metric_map = {k: v.replace(" (→)", "").replace(" (←)", "") for k, v in METRICS_OF_INTEREST.items()}
    df_mean["Metric_Name"] = df_mean["Metric"].map(clean_metric_map)

    pivot_table = df_mean.pivot(index=["Split", "Model"], columns=["Period", "Metric_Name"], values="Value")
    clean_to_orig = {clean_metric_map[k]: k for k in PAPER_TABLE_METRICS}

    def format_and_bold_group(s, is_lower):
        valid_vals = s.dropna().unique()
        if len(valid_vals) > 0:
            sorted_vals = np.sort(valid_vals)
            if not is_lower: sorted_vals = sorted_vals[::-1] 
            best_val, second_best_val = sorted_vals[0], sorted_vals[1] if len(sorted_vals) > 1 else None
        else: best_val, second_best_val = None, None

        return s.apply(lambda x: "-" if pd.isnull(x) else f"<b>{x:.3f}</b>" if np.isclose(x, best_val, atol=1e-5) else f"<u>{x:.3f}</u>" if second_best_val is not None and np.isclose(x, second_best_val, atol=1e-5) else f"{x:.3f}")

    for col in pivot_table.columns:
        period, clean_metric = col
        orig_metric = clean_to_orig.get(clean_metric)
        is_lower_better = "←" in METRICS_OF_INTEREST.get(orig_metric, "") if orig_metric else False
        pivot_table[col] = pivot_table[col].groupby(level='Split').transform(format_and_bold_group, is_lower=is_lower_better)

    model_order = CLASSIC_ML_MODELS_TO_PLOT + ["Transformer"]
    ordered_metrics = [clean_metric_map[m] for m in PAPER_TABLE_METRICS if m in df_filtered["Metric"].unique()]
    existing_splits = [s for s in SPLIT_TYPES if s in pivot_table.index.get_level_values('Split')]
    existing_models = [m for m in model_order if m in pivot_table.index.get_level_values('Model')]
    existing_periods = [p for p in valid_periods if p in pivot_table.columns.get_level_values('Period')]
    
    pivot_table = pivot_table.reindex(
        index=pd.MultiIndex.from_product([existing_splits, existing_models], names=['Split', 'Model']),
        columns=pd.MultiIndex.from_product([existing_periods, ordered_metrics], names=['Period', 'Metric_Name'])
    ).dropna(how='all')

    center_css = "style='text-align: center; vertical-align: middle;'"
    html = ["<div style='overflow-x: auto; width: 100%;'>", "  <table border='1' style='border-collapse: collapse; table-layout: fixed; width: 166%; word-wrap: break-word;'>\n <thead><tr>"]
    html.append(f"      <th rowspan='2' {center_css}>Split<br>strategy</th><th rowspan='2' {center_css}>Model</th>")
    for period in existing_periods: html.append(f"      <th colspan='{len(ordered_metrics)}' {center_css}>{period.replace('\n', ' ')}</th>")
    html.append("    </tr><tr>")
    for _ in existing_periods:
        for metric in ordered_metrics: html.append(f"      <th {center_css}>{metric}</th>")
    html.append("    </tr></thead><tbody>")
    
    current_split = None
    for idx, row in pivot_table.iterrows():
        split, model = idx
        row_style = " style='border-top: 2px solid currentColor;'" if (split != current_split and current_split is not None) else ""
        html.append(f"    <tr{row_style}>")
        if split != current_split:
            html.append(f"      <td rowspan='{len(pivot_table.xs(split, level='Split'))}' {center_css}><b>{split.capitalize().replace('_', '<br>')}</b></td>")
            current_split = split
        html.append(f"      <td {center_css}>{MODEL_NAME_MAP.get(model, model)}</td>")
        for val in row: html.append(f"      <td {center_css}>{val}</td>")
        html.append("    </tr>")
        
    html.append("  </tbody></table></div>")
    with open(output_dir / f"{eval_type}_paper_table_{task_name}.md", "w", encoding='utf-8') as f:
        f.write(f"### Hierarchical performance comparison: {task_name.replace('_', ' ').capitalize()} ({eval_type.capitalize()})\n\n" + "\n".join(html))


def generate_performance_summaries(df: pd.DataFrame, output_dir: Path, task_name: str, eval_type: str):
    grouped = df.groupby(["Period", "Metric", "Model"])["Value"].agg(["mean", "std"])
    grouped["Formatted"] = grouped.apply(lambda r: f"{r['mean']:.3f} ± {r['std']:.3f}", axis=1)
    pivot_table = grouped.reset_index().pivot(index=["Period", "Metric"], columns="Model", values="Formatted").reset_index()
    cols = ["Period", "Metric"] + [m for m in (CLASSIC_ML_MODELS_TO_PLOT + ["Transformer"]) if m in pivot_table.columns]
    
    with open(output_dir / f"{eval_type}_summary_{task_name}.md", "w") as f:
        f.write(f"# Performance Summary: {task_name.replace('_', ' ').title()}\n\n" + pivot_table[cols].to_markdown(index=False))


def compute_overall_prevalence(raw_pool: Dict, period_dict: Dict, task_name: str, split_name: str):
    clinical_periods = period_dict.get(task_name, {})
    if not clinical_periods:
        print(f"No periods found for task: {task_name}")
        return

    available_models = list(raw_pool.get(split_name, {}).keys())
    if not available_models:
        print(f"No model data found for split: {split_name}")
        return
    ref_model = available_models[0]
    horizons_data = raw_pool[split_name][ref_model][task_name]

    print(f"=== Population Prevalence for {task_name.upper()} ({split_name}) ===")
    
    for period_name, h_fup_map in clinical_periods.items():
        y_t_list = []
        
        for h, fups in h_fup_map.items():
            if h not in horizons_data: continue
            for fup in fups:
                if fup in horizons_data[h]:
                    labels = horizons_data[h][fup]["labels"]
                    valid_mask = labels != -100
                    if valid_mask.any():
                        y_t_list.append(labels[valid_mask])
                        
        if not y_t_list:
            print(f"{period_name.replace('\n', ' ')}: No valid data points.")
            continue
            
        y_true = np.concatenate(y_t_list)
        total = len(y_true)
        positives = np.sum(y_true == 1)
        negatives = np.sum(y_true == 0)
        
        pos_rate = (positives / total) * 100
        neg_rate = (negatives / total) * 100
        
        print(f"{period_name.replace('\n', ' ')}:")
        print(f"   • Total Samples: {total}")
        print(f"   • Positives (Class 1): {positives} ({pos_rate:.2f}%)")
        print(f"   • Negatives (Class 0): {negatives} ({neg_rate:.2f}%)")
    
    print("=" * 50)


if __name__ == "__main__":
    main()