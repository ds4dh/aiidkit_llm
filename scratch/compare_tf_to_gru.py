import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datasets import load_from_disk
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve
from statsmodels.stats.contingency_tables import mcnemar
from scripts.script_utils import load_all_raw_predictions

BASE_DATA_PATH = Path("/home/shares/ds4dh/aiidkit_project/data_new/processed/v3.6/teav")
TRANSFORMER_BASE_DIR = Path("results_final/transformer")
COMP_DIR = Path("results_final/mhmmdrz_comp")
OUTPUT_DIR = COMP_DIR / "analysis"

MY_MODEL_TEST_CSV = COMP_DIR / "transformer.csv"
MY_MODEL_VAL_CSV = COMP_DIR / "transformer_val.csv"
COLLEAGUE_TEST_CSV = COMP_DIR / "mhmmdrz_Late-Gated_h64_seed42.csv"
COLLEAGUE_VAL_CSV = COMP_DIR / "mhmmdrz_Late-Gated_h64_seed42_val.csv"

TARGET_RECALL_ANCHOR = 0.80
CLINICAL_WINDOWS = {
    "Perioperative (0-30 d)": (0, 30),
    "Opportunistic (31-180 d)": (31, 180),
    "Maintenance (181-360 d)": (181, 360),
    "Long-term (361-720 d)": (361, 720),
}


def auto_generate_transformer_dependencies():
    """Compiles and builds flat CSV files for both test and validation splits using script_utils paths."""
    COMP_DIR.mkdir(parents=True, exist_ok=True)
    missing_test = not MY_MODEL_TEST_CSV.exists()
    missing_val = not MY_MODEL_VAL_CSV.exists()
    
    if missing_test or missing_val:
        print("[!] Missing flat transformer matrices. Compiling raw pool tracker...")
        
        # Configure parameters to match your directory structures precisely
        kwargs = {
            "transformer_base_dir": TRANSFORMER_BASE_DIR,
            "classic_ml_base_dir": COMP_DIR, # Dummy fallback path
            "from_optuna": False,
            "split_types": ["random_split"],
            "tasks": ["infection_bacteria"],
            "classic_ml_models": []
        }
        
        if missing_test:
            print(" -> Converting test split predictions...")
            raw_pool_test = load_all_raw_predictions(**kwargs, dataset_split="test")
            export_transformer_results_to_flat_format(
                raw_pool=raw_pool_test, target_model_name="transformer", task_name="infection_bacteria",
                split_name="random_split", horizon_days=30, output_dir=COMP_DIR, dataset_split="test"
            )
            generated_test = COMP_DIR / "transformer_infection_bacteria_random_split_hrz30.csv"
            if generated_test.exists():
                generated_test.rename(MY_MODEL_TEST_CSV)
                
        if missing_val:
            print(" -> Converting validation split predictions...")
            raw_pool_val = load_all_raw_predictions(**kwargs, dataset_split="validation")
            export_transformer_results_to_flat_format(
                raw_pool=raw_pool_val, target_model_name="transformer", task_name="infection_bacteria",
                split_name="random_split", horizon_days=30, output_dir=COMP_DIR, dataset_split="validation"
            )
            generated_val = COMP_DIR / "transformer_infection_bacteria_random_split_hrz30_val.csv"
            if generated_val.exists():
                generated_val.rename(MY_MODEL_VAL_CSV)


def export_transformer_results_to_flat_format(
    raw_pool: dict,
    target_model_name: str,
    task_name: str,
    split_name: str,
    horizon_days: int,
    output_dir: Path,
    dataset_split: str = "test",
    config_label: str = "Transformer_Base",
    seed: int = 0,
):
    """Flattens nested validation/test npz pool predictions and maps row patient keys."""
    if (split_name not in raw_pool or 
        target_model_name not in raw_pool[split_name] or 
        task_name not in raw_pool[split_name][target_model_name] or 
        horizon_days not in raw_pool[split_name][target_model_name][task_name]):
        raise KeyError(f"Slice {split_name} -> {target_model_name} -> {task_name} -> horizon {horizon_days}d not found.")
        
    fup_data_dict = raw_pool[split_name][target_model_name][task_name][horizon_days]
    flat_records = []
    
    for fup_day, records in fup_data_dict.items():
        y_true = records["labels"]
        y_prob = records["probs"]  # records["probs_cal"])
        
        raw_fup_dir = BASE_DATA_PATH / split_name / f"fup_{fup_day:04d}"
        if not raw_fup_dir.exists():
            raw_fup_dir = BASE_DATA_PATH / split_name / f"fup_{fup_day:04d}d"
            
        if not raw_fup_dir.exists():
            continue
            
        raw_ds = load_from_disk(str(raw_fup_dir))[dataset_split]
        raw_keys = raw_ds["patientkey"]
        raw_labels = raw_ds[f"label_{task_name}_{horizon_days:04d}d"]
        
        model_idx = 0
        for idx, true_val in enumerate(raw_labels):
            if true_val == -100:
                continue 
                
            if model_idx >= len(y_prob):
                break
                
            flat_records.append({
                "patientkey": raw_keys[idx],
                "time_step": fup_day,
                "y_true": int(y_true[model_idx]),
                "y_prob": float(y_prob[model_idx]),
                "config": config_label,
                "seed": seed
            })
            model_idx += 1
            
    df_flat = pd.DataFrame(flat_records)
    if df_flat.empty:
        print(f"[!] Warning: No matching rows found for split '{dataset_split}'.")
        return
        
    df_flat = df_flat.sort_values(by=["patientkey", "time_step"]).reset_index(drop=True)
    suffix = "_val" if dataset_split == "validation" else ""
    file_path = output_dir / f"{target_model_name}_{task_name}_{split_name}_hrz{horizon_days}{suffix}.csv"
    df_flat.to_csv(file_path, index=False)
    print(f">>> [{dataset_split.upper()}] Exported aligned table: {file_path}")


def get_window_specific_threshold(val_df: pd.DataFrame, low: int, high: int, target_recall: float, model_label: str) -> float:
    """Finds the decision threshold specifically pooled inside this clinical window frame."""
    sub_val = val_df[(val_df['time_step'] >= low) & (val_df['time_step'] <= high)].copy()
    sub_val = sub_val[sub_val['y_true'] != -100]
    
    if sub_val.empty or len(np.unique(sub_val['y_true'])) < 2:
        print(f"[!] Warning: Insufficient data in window {low}-{high}d for {model_label}. Falling back to 0.5")
        return 0.5
        
    p, r, t = precision_recall_curve(sub_val['y_true'].values, sub_val['y_prob'].values)
    valid_indices = np.where(r[:-1] >= target_recall)[0]
    if len(valid_indices) == 0: 
        return 0.5
    
    chosen_threshold = float(t[valid_indices[-1]])
    print(f"  -> [{model_label}] Window {low}-{high}d threshold fixed at >= {chosen_threshold:.4f} (Val Recall: {r[valid_indices[-1]]:.2%})")
    return chosen_threshold


def get_global_threshold(val_df: pd.DataFrame, target_recall: float, model_label: str) -> float:
    """Finds a single decision threshold across the entire validation set to hit target recall."""
    clean_val = val_df[val_df['y_true'] != -100]
    if clean_val.empty or len(np.unique(clean_val['y_true'])) < 2:
        print(f"[!] Warning: Insufficient data to calculate threshold for {model_label}. Falling back to 0.5")
        return 0.5
        
    p, r, t = precision_recall_curve(clean_val['y_true'].values, clean_val['y_prob'].values)
    valid_indices = np.where(r[:-1] >= target_recall)[0]
    if len(valid_indices) == 0: 
        return 0.5
    
    chosen_threshold = float(t[valid_indices[-1]])
    print(f"[GLOBAL THRESHOLD] {model_label} fixed at >= {chosen_threshold:.4f} (Validation Recall: {r[valid_indices[-1]]:.2%})")
    return chosen_threshold


def run_comparison():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    try:
        auto_generate_transformer_dependencies()
        
        print("\n=== STEP 1: CALCULATING GLOBAL OPERATIONAL THRESHOLDS ===")
        df_val_mine = pd.read_csv(MY_MODEL_VAL_CSV)
        df_val_coll = pd.read_csv(COLLEAGUE_VAL_CSV)
        
        # Calculate exactly one threshold per model using the full validation pool
        global_thresh_mine = get_global_threshold(df_val_mine, TARGET_RECALL_ANCHOR, "My Model")
        global_thresh_coll = get_global_threshold(df_val_coll, TARGET_RECALL_ANCHOR, "Colleague Model")
        
        if not COLLEAGUE_TEST_CSV.exists():
            raise FileNotFoundError(f"Colleague baseline matrix missing: {COLLEAGUE_TEST_CSV}")
            
        df_test_a = pd.read_csv(MY_MODEL_TEST_CSV)[['patientkey', 'time_step', 'y_true', 'y_prob']]
        df_test_b = pd.read_csv(COLLEAGUE_TEST_CSV)[['patientkey', 'time_step', 'y_true', 'y_prob']]
        df_test = pd.merge(df_test_a, df_test_b, on=['patientkey', 'time_step', 'y_true'], suffixes=('_my_model', '_colleague'))
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        return

    print("\n=== STEP 2: RUNNING STRATIFIED STATISTICAL EVALUATION ===")
    raw_results = {}
    sns.set_theme(style="whitegrid")
    
    n_windows = len(CLINICAL_WINDOWS)
    fig, axes = plt.subplots(2, n_windows, figsize=(5.5 * n_windows, 9.5), sharey='row')
    
    for idx, (window_name, (low, high)) in enumerate(CLINICAL_WINDOWS.items()):
        ax_pr = axes[0, idx]
        ax_roc = axes[1, idx]
        
        sub_df = df_test[(df_test['time_step'] >= low) & (df_test['time_step'] <= high)].copy()
        if sub_df.empty:
            print(f"[-] No overlapping frames found inside window: {window_name}")
            continue
            
        y_true = sub_df['y_true'].values
        p_mine = sub_df['y_prob_my_model'].values
        p_coll = sub_df['y_prob_colleague'].values
        
        pr_auc_mine = average_precision_score(y_true, p_mine)
        pr_auc_coll = average_precision_score(y_true, p_coll)
        roc_auc_mine = roc_auc_score(y_true, p_mine)
        roc_auc_coll = roc_auc_score(y_true, p_coll)
        
        # Apply the single global threshold uniformly across all windows
        preds_mine = (p_mine >= global_thresh_mine).astype(int)
        preds_coll = (p_coll >= global_thresh_coll).astype(int)
        
        # Diagnostic Matrix calculations
        tp_mine = np.sum((preds_mine == 1) & (y_true == 1))
        fp_mine = np.sum((preds_mine == 1) & (y_true == 0))
        tn_mine = np.sum((preds_mine == 0) & (y_true == 0))
        fn_mine = np.sum((preds_mine == 0) & (y_true == 1))

        tp_coll = np.sum((preds_coll == 1) & (y_true == 1))
        fp_coll = np.sum((preds_coll == 1) & (y_true == 0))
        tn_coll = np.sum((preds_coll == 0) & (y_true == 0))
        fn_coll = np.sum((preds_coll == 0) & (y_true == 1))

        rec_mine = tp_mine / (tp_mine + fn_mine) if (tp_mine + fn_mine) > 0 else 0.0
        rec_coll = tp_coll / (tp_coll + fn_coll) if (tp_coll + fn_coll) > 0 else 0.0
        
        prec_mine = tp_mine / (tp_mine + fp_mine) if (tp_mine + fp_mine) > 0 else 0.0
        prec_coll = tp_coll / (tp_coll + fp_coll) if (tp_coll + fp_coll) > 0 else 0.0
        
        spec_mine = tn_mine / (tn_mine + fp_mine) if (tn_mine + fp_mine) > 0 else 0.0
        spec_coll = tn_coll / (tn_coll + fp_coll) if (tn_coll + fp_coll) > 0 else 0.0
        
        # McNemar Test
        correct_mine = (preds_mine == y_true)
        correct_coll = (preds_coll == y_true)
        b = np.sum(correct_mine & ~correct_coll)
        c = np.sum(~correct_mine & correct_coll)
        
        contingency_table = [[np.sum(correct_mine & correct_coll), b], [c, np.sum(~correct_mine & ~correct_coll)]]
        
        try:
            mcn_res = mcnemar(contingency_table, exact=True)
            p_val = mcn_res.pvalue
        except Exception:
            p_val = np.nan
            
        winner = "Tie" if (pd.isna(p_val) or p_val > 0.05) else ("My Model" if b > c else "Colleague")
        
        raw_results[window_name] = {
            "Total Evaluation Frames": len(sub_df),
            "Global Threshold (Mine)": f"{global_thresh_mine:.4f}",
            "Global Threshold (Colleague)": f"{global_thresh_coll:.4f}",
            "PR-AUC (Mine)": f"{pr_auc_mine:.4f}",
            "PR-AUC (Colleague)": f"{pr_auc_coll:.4f}",
            "ROC-AUC (Mine)": f"{roc_auc_mine:.4f}",
            "ROC-AUC (Colleague)": f"{roc_auc_coll:.4f}",
            "Test Recall (Mine)": f"{rec_mine:.4f}",
            "Test Recall (Colleague)": f"{rec_coll:.4f}",
            "Test Precision (Mine)": f"{prec_mine:.4f}",
            "Test Precision (Colleague)": f"{prec_coll:.4f}",
            "Test Specificity (Mine)": f"{spec_mine:.4f}",
            "Test Specificity (Colleague)": f"{spec_coll:.4f}",
            "Discordant [Mine Correct] (b)": int(b),
            "Discordant [Coll Correct] (c)": int(c),
            "McNemar p-value": f"{p_val:.3e}" if not pd.isna(p_val) else "NaN",
            "STATISTICAL WINNER": winner
        }
        
        # Plotting remains the same
        pm_y, pm_x, _ = precision_recall_curve(y_true, p_mine)
        pc_y, pc_x, _ = precision_recall_curve(y_true, p_coll)
        ax_pr.plot(pm_x, pm_y, label=f"My Model (PR = {pr_auc_mine:.3f})", color="#1f77b4", lw=2.5)
        ax_pr.plot(pc_x, pc_y, label=f"Colleague (PR = {pr_auc_coll:.3f})", color="#ff7f0e", lw=2.5)
        ax_pr.axvline(TARGET_RECALL_ANCHOR, color="gray", linestyle=":", alpha=0.8, label="Anchor")
        ax_pr.set_title(window_name, fontsize=11, fontweight='bold', pad=10)
        ax_pr.set_xlabel("Recall")
        if idx == 0: ax_pr.set_ylabel("Precision", fontsize=12, fontweight='bold')
        ax_pr.legend(loc="lower left", fontsize=8, frameon=True)
        ax_pr.set_xlim([0.0, 1.0])
        ax_pr.set_ylim([0.0, 1.05])
        
        fpr_mine, tpr_mine, _ = roc_curve(y_true, p_mine)
        fpr_coll, tpr_coll, _ = roc_curve(y_true, p_coll)
        ax_roc.plot(fpr_mine, tpr_mine, label=f"My Model (ROC = {roc_auc_mine:.3f})", color="#1f77b4", lw=2.5)
        ax_roc.plot(fpr_coll, tpr_coll, label=f"Colleague (ROC = {roc_auc_coll:.3f})", color="#ff7f0e", lw=2.5)
        ax_roc.plot([0, 1], [0, 1], linestyle="--", color="gray", alpha=0.7)
        ax_roc.set_xlabel("False Positive Rate")
        if idx == 0: ax_roc.set_ylabel("True Positive Rate", fontsize=12, fontweight='bold')
        ax_roc.legend(loc="lower right", fontsize=8, frameon=True)
        ax_roc.set_xlim([0.0, 1.0])
        ax_roc.set_ylim([0.0, 1.05])

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "continuous_performance_comparison.png", dpi=200, bbox_inches='tight')
    plt.close()
    
    df_report = pd.DataFrame(raw_results)
    df_report.index.name = "Evaluation Metric"
    df_report.to_csv(OUTPUT_DIR / "head_to_head_report.csv")
    
    print("\n=== TRANSPOSED COHORT METRIC REPORT ===")
    print(df_report.to_markdown())
    

if __name__ == "__main__":
    run_comparison()