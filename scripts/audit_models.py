import argparse
from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_recall_curve
from scipy.stats import norm
from datasets import load_from_disk

from scripts.script_utils import calibrate_dataframe_pair


# ==========================================
# Paths & Settings Defaults
# ==========================================
BASE_DATA_PATH = Path("/home/shares/ds4dh/aiidkit_project/data_new/processed/v3.6_old/teav")
RESULTS_DIR = Path("results_final")
OUTPUT_IMAGE = Path("./tmp/conf_mats.png")


# ==========================================
# Data Loading Functions (Single Step)
# ==========================================
def load_transformer_step(task, split, dataset_split, horizon, time_step):
    """Load Transformer predictions for a single follow-up time step."""
    fname = "validation_probs.npz" if dataset_split == "validation" else "test_probs.npz"
    root = RESULTS_DIR / "transformer" / split / "e00-a15-v60" / "finetuning" / task
    if not root.exists():
        return pd.DataFrame()

    found = None
    for d in root.iterdir():
        m = re.search(r"hrz\(([\d-]+)\)", d.name)
        if d.is_dir() and m and horizon in [int(v) for v in m.group(1).split("-")]:
            hs = [int(v) for v in m.group(1).split("-")]
            found = (d, hs.index(horizon), hs)
            break

    if found is None or not (found[0] / fname).exists():
        return pd.DataFrame()

    d, target_idx, hs = found
    z = np.load(d / fname, allow_pickle=True)
    pref = "validation_" if dataset_split == "validation" else "test_"
    lbl_key = f"{pref}fup_{time_step:04d}_labels"
    prob_key = f"{pref}fup_{time_step:04d}_probs"

    if lbl_key not in z.files or prob_key not in z.files:
        return pd.DataFrame()

    labels = z[lbl_key]
    probs = z[prob_key]
    y = labels if labels.ndim == 1 else labels[:, target_idx]
    p = probs if probs.ndim == 1 else probs[:, target_idx]

    dsdir = BASE_DATA_PATH / split / f"fup_{time_step:04d}"
    if not dsdir.exists():
        dsdir = BASE_DATA_PATH / split / f"fup_{time_step:04d}d"
    if not dsdir.exists():
        return pd.DataFrame()

    ds = load_from_disk(str(dsdir))[dataset_split]
    keys = ds["patientkey"]
    cols = [f"label_{task}_{h:04d}d" for h in hs if f"label_{task}_{h:04d}d" in ds.column_names]
    keep = (
        (np.stack([ds[c] for c in cols], 1) != -100).any(1)
        if cols else np.ones(len(keys), bool)
    )
    valid = np.where(keep)[0]

    records = []
    for i in range(min(len(valid), len(y), len(p))):
        if int(y[i]) != -100:
            records.append({
                "patientkey": keys[valid[i]],
                "time_step": time_step,
                "horizon": horizon,
                "y_true": int(y[i]),
                "y_prob": float(p[i]),
            })
    return pd.DataFrame(records)


def load_classic_step(model, task, split, dataset_split, horizon, time_step):
    """Load classic ML predictions for a single follow-up time step."""
    fname = "val_predictions.npz" if dataset_split == "validation" else "test_predictions.npz"
    root = RESULTS_DIR / "classic_ml" / split / model / task
    if not root.exists():
        return pd.DataFrame()

    d = next(
        (x for x in root.iterdir() if x.is_dir() and f"hrz({horizon:04d})" in x.name),
        None,
    )
    if d is None or not (d / fname).exists():
        return pd.DataFrame()

    z = np.load(d / fname, allow_pickle=True)
    pref = "validation_" if dataset_split == "validation" else "test_"
    lbl_key = f"{pref}fup_{time_step:04d}_labels"
    prob_key = f"{pref}fup_{time_step:04d}_probs"

    if lbl_key not in z.files or prob_key not in z.files:
        return pd.DataFrame()

    y = z[lbl_key].ravel()
    p = z[prob_key].ravel()

    dsdir = BASE_DATA_PATH / split / f"fup_{time_step:04d}"
    if not dsdir.exists():
        dsdir = BASE_DATA_PATH / split / f"fup_{time_step:04d}d"
    if not dsdir.exists():
        return pd.DataFrame()

    ds = load_from_disk(str(dsdir))[dataset_split]
    keys = ds["patientkey"]
    col = f"label_{task}_{horizon:04d}d"
    valid = np.where(
        np.asarray(ds[col]) != -100 if col in ds.column_names else np.ones(len(keys), bool)
    )[0]

    records = []
    for i in range(min(len(valid), len(y), len(p))):
        if int(y[i]) != -100:
            records.append({
                "patientkey": keys[valid[i]],
                "time_step": time_step,
                "horizon": horizon,
                "y_true": int(y[i]),
                "y_prob": float(p[i]),
            })
    return pd.DataFrame(records)


def load_model_data(model_name, task, split, dataset_split, horizon, time_step):
    if model_name == "Transformer":
        return load_transformer_step(task, split, dataset_split, horizon, time_step)
    return load_classic_step(model_name, task, split, dataset_split, horizon, time_step)


# ==========================================
# Helpers & Calculations
# ==========================================
def compute_threshold_for_recall(df_val, target_recall=0.80):
    """Find the threshold that achieves target recall on validation set."""
    if df_val.empty or df_val["y_true"].nunique() < 2:
        return 0.5
    _, recall, thresholds = precision_recall_curve(df_val["y_true"], df_val["y_prob"])
    idx = np.where(recall[:-1] >= target_recall)[0]
    if len(idx) == 0:
        return 0.5
    return float(thresholds[idx[-1]])


def calculate_mcnemar_sample_size(p10, p01, alpha=0.05, power=0.80, two_sided=True):
    """Calculate sample size for McNemar's test given discordant proportions (Connor 1987 / Machin)."""
    p_diff = abs(p10 - p01)
    p_disc = p10 + p01
    if p_diff == 0 or p_disc == 0:
        return np.nan

    z_alpha = norm.ppf(1.0 - alpha / 2.0) if two_sided else norm.ppf(1.0 - alpha)
    z_beta = norm.ppf(power)

    # Standard formula for paired binary proportions
    n_required = ((z_alpha * np.sqrt(p_disc) + z_beta * np.sqrt(p_disc - p_diff**2)) / p_diff) ** 2
    return int(np.ceil(n_required))


# ==========================================
# Main Execution Flow
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Two-Model Evaluation & McNemar Sample Size Calculator")
    parser.add_argument("--model_a", type=str, default="Transformer", choices=["Transformer", "logistic_regression", "random_forest", "xgboost"])
    parser.add_argument("--model_b", type=str, default="xgboost", choices=["Transformer", "logistic_regression", "random_forest", "xgboost"])
    parser.add_argument("--task", type=str, default="infection_bacteria", choices=["infection_bacteria", "infection_virus"])
    parser.add_argument("--split", type=str, default="random_split", choices=["random_split", "temporal_split", "center_split"])
    parser.add_argument("--time_step", "-T", type=int, default=180, help="Post-tpx follow-up day (e.g. 180)")
    parser.add_argument("--horizon", "-H", type=int, default=30, choices=[30, 60, 90], help="Prediction horizon in days")
    parser.add_argument("--target_recall", type=float, default=0.80, help="Recall target for thresholding (0.0 - 1.0)")
    parser.add_argument("--calibrate", action="store_true", default=True, help="Apply isotonic calibration")
    args = parser.parse_args()

    print(f"Loading data: Task={args.task} | Split={args.split} | FUP={args.time_step}d | Horizon={args.horizon}d")

    # Load validation and test sets
    val_a = load_model_data(args.model_a, args.task, args.split, "validation", args.horizon, args.time_step)
    test_a = load_model_data(args.model_a, args.task, args.split, "test", args.horizon, args.time_step)
    val_b = load_model_data(args.model_b, args.task, args.split, "validation", args.horizon, args.time_step)
    test_b = load_model_data(args.model_b, args.task, args.split, "test", args.horizon, args.time_step)

    if any(df.empty for df in [val_a, test_a, val_b, test_b]):
        raise RuntimeError("Could not find matching prediction files for the specified parameters.")

    # Align common patients
    keys = ["patientkey", "time_step", "horizon", "y_true"]
    common_val = val_a[keys].merge(val_b[keys], on=keys, how="inner").drop_duplicates()
    val_a = val_a.merge(common_val, on=keys).sort_values(keys).reset_index(drop=True)
    val_b = val_b.merge(common_val, on=keys).sort_values(keys).reset_index(drop=True)

    common_test = test_a[keys].merge(test_b[keys], on=keys, how="inner").drop_duplicates()
    test_a = test_a.merge(common_test, on=keys).sort_values(keys).reset_index(drop=True)
    test_b = test_b.merge(common_test, on=keys).sort_values(keys).reset_index(drop=True)

    if args.calibrate:
        val_a, test_a = calibrate_dataframe_pair(df_val=val_a, df_test=test_a, prob_col="y_prob")
        val_b, test_b = calibrate_dataframe_pair(df_val=val_b, df_test=test_b, prob_col="y_prob")

    # Compute classification thresholds
    thresh_a = compute_threshold_for_recall(val_a, target_recall=args.target_recall)
    thresh_b = compute_threshold_for_recall(val_b, target_recall=args.target_recall)

    y_test = test_a["y_true"].values
    pred_a = (test_a["y_prob"].values >= thresh_a).astype(int)
    pred_b = (test_b["y_prob"].values >= thresh_b).astype(int)

    # ------------------------------------------
    # Plot Confusion Matrices
    # ------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), dpi=150)

    cm_a = confusion_matrix(y_test, pred_a)
    cm_b = confusion_matrix(y_test, pred_b)

    disp_a = ConfusionMatrixDisplay(confusion_matrix=cm_a, display_labels=["Neg", "Pos"])
    disp_b = ConfusionMatrixDisplay(confusion_matrix=cm_b, display_labels=["Neg", "Pos"])

    disp_a.plot(ax=axes[0], cmap="Blues", colorbar=False)
    axes[0].set_title(f"{args.model_a}\n(Thresh: {thresh_a:.3f})", fontsize=12, fontweight="bold")

    disp_b.plot(ax=axes[1], cmap="Oranges", colorbar=False)
    axes[1].set_title(f"{args.model_b}\n(Thresh: {thresh_b:.3f})", fontsize=12, fontweight="bold")

    plt.suptitle(
        f"Confusion Matrices: {args.model_a} vs {args.model_b}\n"
        f"Task: {args.task} | FUP: {args.time_step}d | Horizon: {args.horizon}d | N={len(y_test)}",
        fontsize=13,
        y=1.05,
    )
    plt.tight_layout()
    plt.savefig(OUTPUT_IMAGE, bbox_inches="tight")
    plt.close()
    print(f"\n[Saved] Confusion matrices plot written to {OUTPUT_IMAGE.resolve()}")

    # ------------------------------------------
    # McNemar Contingency Table & Sample Size
    # ------------------------------------------
    correct_a = (pred_a == y_test)
    correct_b = (pred_b == y_test)

    n_total = len(y_test)
    n_pos_gt = int(np.sum(y_test == 1))
    n_neg_gt = int(np.sum(y_test == 0))

    # Overall Discordance
    n11 = int(np.sum(correct_a & correct_b))    # Both correct
    n10 = int(np.sum(correct_a & ~correct_b))   # Model A correct, Model B incorrect
    n01 = int(np.sum(~correct_a & correct_b))   # Model A incorrect, Model B correct
    n00 = int(np.sum(~correct_a & ~correct_b))  # Both incorrect

    p10 = n10 / n_total
    p01 = n01 / n_total
    p_disc = p10 + p01
    delta_acc = p10 - p01

    # Confusion matrix components
    tn_a, fp_a, fn_a, tp_a = cm_a.ravel()
    tn_b, fp_b, fn_b, tp_b = cm_b.ravel()

    # Stratified Discordance for Sensitivity & Specificity
    mask_neg = (y_test == 0)
    mask_pos = (y_test == 1)
    
    n10_neg = int(np.sum(correct_a[mask_neg] & ~correct_b[mask_neg]))  # A correctly identified TN, B called FP
    n01_neg = int(np.sum(~correct_a[mask_neg] & correct_b[mask_neg]))  # B correctly identified TN, A called FP

    n10_pos = int(np.sum(correct_a[mask_pos] & ~correct_b[mask_pos]))  # A correctly identified TP, B called FN
    n01_pos = int(np.sum(~correct_a[mask_pos] & correct_b[mask_pos]))  # B correctly identified TP, A called FN

    print("\n" + "=" * 65)
    print("             DATASET AND MODEL PREDICTION COUNTS")
    print("=" * 65)
    print(f"Task:                                   {args.task}")
    print(f"Split:                                  {args.split}")
    print(f"Follow-up period:                       {args.time_step}d")
    print(f"Prediction horizon:                     {args.horizon}d")
    print(f"Total test samples (N):                 {n_total}")
    print(f"Ground-truth positives (Y=1):           {n_pos_gt} ({n_pos_gt / n_total:.2%})")
    print(f"Ground-truth negatives (Y=0):           {n_neg_gt} ({n_neg_gt / n_total:.2%})")
    print("-" * 65)
    print(f"{args.model_a} Predictions:    {int(np.sum(pred_a == 1))} pos / {int(np.sum(pred_a == 0))} neg")
    print(f"  >>>    TP: {tp_a:<5} FP: {fp_a:<5} TN: {tn_a:<5} FN: {fn_a:<5}")
    print(f"{args.model_b} Predictions: {int(np.sum(pred_b == 1))} pos / {int(np.sum(pred_b == 0))} neg")
    print(f"  >>>    TP: {tp_b:<5} FP: {fp_b:<5} TN: {tn_b:<5} FN: {fn_b:<5}")
    
    print("\n" + "=" * 65)
    print("        PILOT PERFORMANCE AND DISCORDANCE BREAKDOWN")
    print("=" * 65)
    print(f"Both correct (n11):                     {n11} ({n11 / n_total:.2%})")
    print(f"Both incorrect (n00):                   {n00} ({n00 / n_total:.2%})")
    print(f"Discordant: {args.model_a} only (n10):     {n10} ({p10:.2%})")
    print(f"Discordant: {args.model_b} only (n01):         {n01} ({p01:.2%})")
    print(f"Total discordant proportion (p_disc):   {p_disc:.4f}")
    print(f"Accuracy difference (p10 - p01):        {delta_acc:+.4f}")
    print(f"Odds ratio (n10 / n01):                 {n10 / max(1, n01):.3f}")
    print("-" * 65)
    print("Stratified Discordant Pairs:")
    print(f"  On actual neg (Y=0, specificity): {args.model_a}={n10_neg} vs {args.model_b}={n01_neg}")
    print(f"  On actual pos (Y=1, sensitivity): {args.model_a}={n10_pos} vs {args.model_b}={n01_pos}")

    print("\n" + "=" * 65)
    print("   MCNEMAR SAMPLE SIZE REQUIREMENTS (Two-sided alpha = 0.05)")
    print("=" * 65)
    print(f"{'Power (1 - beta)':<20} | {'Required sample size (N)':<25}")
    print("-" * 65)

    for pwr in [0.80, 0.85, 0.90, 0.95]:
        req_n = calculate_mcnemar_sample_size(p10, p01, alpha=0.05, power=pwr, two_sided=True)
        req_str = f"{req_n:,}" if not np.isnan(req_n) else "Undefined (no difference)"
        print(f"{int(pwr * 100)}%{'':<17} | {req_str:<25}")
    print("=" * 65)


if __name__ == "__main__":
    main()