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
COMP_DIR = Path("results_final/mhmmdrz_comp")
OUTPUT_DIR = COMP_DIR / "mhmmdrz_comp" / "analysis"
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
    """Compiles and builds flat CSV files for both test and validation splits if missing."""
    COMP_DIR.mkdir(parents=True, exist_ok=True)
    missing_test = not MY_MODEL_TEST_CSV.exists()
    missing_val = not MY_MODEL_VAL_CSV.exists()
    
    if missing_test or missing_val:
        print("[!] Missing flat transformer matrices. Compiling raw pool tracker...")
        raw_pool = load_all_raw_predictions()
        
        if missing_test:
            print(" -> Converting Test split predictions...")
            export_transformer_results_to_flat_format(
                raw_pool=raw_pool, target_model_name="Transformer", task_name="infection_bacteria",
                split_name="random_split", horizon_days=30, output_dir=COMP_DIR, dataset_split="test"
            )
            generated_test = COMP_DIR / "Transformer_infection_bacteria_random_split_hrz30.csv"
            if generated_test.exists():
                generated_test.rename(MY_MODEL_TEST_CSV)
                
        if missing_val:
            print(" -> Converting Validation split predictions...")
            export_transformer_results_to_flat_format(
                raw_pool=raw_pool, target_model_name="Transformer", task_name="infection_bacteria",
                split_name="random_split", horizon_days=30, output_dir=COMP_DIR, dataset_split="validation"
            )
            generated_val = COMP_DIR / "Transformer_infection_bacteria_random_split_hrz30_val.csv"
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
    """
    Flattens nested npz pool predictions, recovers row-level patient keys 
    by mapping out tracking segment masks from disk, and saves a flat CSV.
    """
    if (split_name not in raw_pool or 
        target_model_name not in raw_pool[split_name] or 
        task_name not in raw_pool[split_name][target_model_name] or 
        horizon_days not in raw_pool[split_name][target_model_name][task_name]):
        raise KeyError(f"Slice {split_name} -> {target_model_name} -> {task_name} -> horizon {horizon_days}d not found.")
        
    fup_data_dict = raw_pool[split_name][target_model_name][task_name][horizon_days]
    flat_records = []
    
    # Base path pointing to your data space
    for fup_day, records in fup_data_dict.items():
        y_true = records["labels"]
        y_prob = records.get("probs_cal", records["probs"])
        
        # Resolve raw directory naming anomalies
        raw_fup_dir = BASE_DATA_PATH / split_name / f"fup_{fup_day:04d}"
        if not raw_fup_dir.exists():
            raw_fup_dir = BASE_DATA_PATH / split_name / f"fup_{fup_day:04d}d"
            
        if not raw_fup_dir.exists():
            continue
            
        # Dynamically load the requested dataset split chunk (test or validation)
        raw_ds = load_from_disk(str(raw_fup_dir))[dataset_split]
        raw_keys = raw_ds["patientkey"]
        raw_labels = raw_ds[f"label_{task_name}_{horizon_days:04d}d"]
        import ipdb; ipdb.set_trace()
        model_idx = 0
        for idx, true_val in enumerate(raw_labels):
            if true_val == -100:
                continue  # Filter out censored frames to maintain perfect parity alignment
                
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
        
    # Group and sort by patient timeline tracking states
    df_flat = df_flat.sort_values(by=["patientkey", "time_step"]).reset_index(drop=True)
    
    # Save file out with split-specific labeling identifiers
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_val" if dataset_split == "validation" else ""
    file_path = output_dir / f"{target_model_name}_{task_name}_{split_name}_hrz{horizon_days}{suffix}.csv"
    
    df_flat.to_csv(file_path, index=False)
    print(f">>> [{dataset_split.upper()}] Exported aligned records table: {file_path}")


def run_comparison():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    try:
        auto_generate_transformer_dependencies()
        
        print("\n=== STEP 1: EXTRACTING OPERATIONAL THRESHOLDS ===")
        thresh_mine = get_threshold_from_validation(MY_MODEL_VAL_CSV, TARGET_RECALL_ANCHOR, "My Model")
        thresh_coll = get_threshold_from_validation(COLLEAGUE_VAL_CSV, TARGET_RECALL_ANCHOR, "Colleague Model")
        
        print("\n=== STEP 2: LOADING ALIGNED COHORT TEST MATRICES ===")
        if not COLLEAGUE_TEST_CSV.exists():
            raise FileNotFoundError(f"Colleague baseline matrix missing: {COLLEAGUE_TEST_CSV}")
            
        df_test_a = pd.read_csv(MY_MODEL_TEST_CSV)[['patientkey', 'time_step', 'y_true', 'y_prob']]
        df_test_b = pd.read_csv(COLLEAGUE_TEST_CSV)[['patientkey', 'time_step', 'y_true', 'y_prob']]
        df_test = pd.merge(df_test_a, df_test_b, on=['patientkey', 'time_step', 'y_true'], suffixes=('_my_model', '_colleague'))
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        return

    print("\n=== STEP 3: RUNNING STRATIFIED STATISTICAL EVALUATION ===")
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
        
        # Calculate continuous AUROCs/AP scores
        pr_auc_mine = average_precision_score(y_true, p_mine)
        pr_auc_coll = average_precision_score(y_true, p_coll)
        roc_auc_mine = roc_auc_score(y_true, p_mine)
        roc_auc_coll = roc_auc_score(y_true, p_coll)
        
        # Deploy validation thresholds
        preds_mine = (p_mine >= thresh_mine).astype(int)
        preds_coll = (p_coll >= thresh_coll).astype(int)
        
        # Calculate Diagnostic Breakdowns
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
        
        # McNemar Significance Engine
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
        
        # Pack records keyed under the specific Clinical Window
        raw_results[window_name] = {
            "Total Evaluation Frames": len(sub_df),
            "PR-AUC (Mine)": f"{pr_auc_mine:.4f}",
            "PR-AUC (Colleague)": f"{pr_auc_coll:.4f}",
            "ROC-AUC (Mine)": f"{roc_auc_mine:.4f}",
            "ROC-AUC (Colleague)": f"{roc_auc_coll:.4f}",
            "Operational Recall (Mine)": f"{rec_mine:.4f}",
            "Operational Recall (Colleague)": f"{rec_coll:.4f}",
            "Operational Precision (Mine)": f"{prec_mine:.4f}",
            "Operational Precision (Colleague)": f"{prec_coll:.4f}",
            "Operational Specificity (Mine)": f"{spec_mine:.4f}",
            "Operational Specificity (Colleague)": f"{spec_coll:.4f}",
            "Discordant Frames [Mine Correct] (b)": int(b),
            "Discordant Frames [Coll Correct] (c)": int(c),
            "McNemar p-value": f"{p_val:.3e}" if not pd.isna(p_val) else "NaN",
            "STATISTICAL WINNER": winner
        }
        
        # Plot continuous curve profiles
        pm_y, pm_x, _ = precision_recall_curve(y_true, p_mine)
        pc_y, pc_x, _ = precision_recall_curve(y_true, p_coll)
        ax_pr.plot(pm_x, pm_y, label=f"My Model (PR = {pr_auc_mine:.3f})", color="#1f77b4", lw=2.5)
        ax_pr.plot(pc_x, pc_y, label=f"Colleague (PR = {pr_auc_coll:.3f})", color="#ff7f0e", lw=2.5)
        ax_pr.axvline(TARGET_RECALL_ANCHOR, color="gray", linestyle=":", alpha=0.8, label="Validation Anchor")
        ax_pr.set_title(window_name, fontsize=13, fontweight='bold', pad=10)
        ax_pr.set_xlabel("Recall")
        if idx == 0: ax_pr.set_ylabel("Precision", fontsize=12, fontweight='bold')
        ax_pr.legend(loc="lower left", fontsize=9, frameon=True)
        ax_pr.set_xlim([0.0, 1.0])
        ax_pr.set_ylim([0.0, 1.05])
        
        fpr_mine, tpr_mine, _ = roc_curve(y_true, p_mine)
        fpr_coll, tpr_coll, _ = roc_curve(y_true, p_coll)
        ax_roc.plot(fpr_mine, tpr_mine, label=f"My Model (ROC = {roc_auc_mine:.3f})", color="#1f77b4", lw=2.5)
        ax_roc.plot(fpr_coll, tpr_coll, label=f"Colleague (ROC = {roc_auc_coll:.3f})", color="#ff7f0e", lw=2.5)
        ax_roc.plot([0, 1], [0, 1], linestyle="--", color="gray", alpha=0.7, label="Chance")
        ax_roc.set_xlabel("False Positive Rate")
        if idx == 0: ax_roc.set_ylabel("True Positive Rate", fontsize=12, fontweight='bold')
        ax_roc.legend(loc="lower right", fontsize=9, frameon=True)
        ax_roc.set_xlim([0.0, 1.0])
        ax_roc.set_ylim([0.0, 1.05])

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "continuous_performance_comparison.png", dpi=200, bbox_inches='tight')
    plt.close()
    
    # Rotate the DataFrame 90 degrees
    df_report = pd.DataFrame(raw_results)
    df_report.index.name = "Evaluation Metric"
    df_report.to_csv(OUTPUT_DIR / "head_to_head_report.csv")
    
    print("\n=== TRANSPOSED COHORT METRIC REPORT (90° ROTATION) ===")
    print(df_report.to_markdown())
    print("======================================================")


def get_threshold_from_validation(val_path: Path, target_recall: float, model_label: str) -> float:
    """Finds the decision threshold on the validation split that yields target recall overall."""
    if not val_path.exists():
        raise FileNotFoundError(f"Validation matrix for {model_label} missing at: {val_path}")
    df_val = pd.read_csv(val_path)
    df_val = df_val[df_val['y_true'] != -100]
    p, r, t = precision_recall_curve(df_val['y_true'].values, df_val['y_prob'].values)
    valid_indices = np.where(r[:-1] >= target_recall)[0]
    if len(valid_indices) == 0: return 0.5
    chosen_threshold = float(t[valid_indices[-1]])
    print(f"[THRESHOLD SELECTION] {model_label} fixed at >= {chosen_threshold:.4f} (Validation Recall: {r[valid_indices[-1]]:.2%})")
    return chosen_threshold


if __name__ == "__main__":
    run_comparison()