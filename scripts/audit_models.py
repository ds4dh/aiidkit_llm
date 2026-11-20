import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_recall_curve
from scipy.stats import norm

from src.tasks import UNIFIED_HORIZONS, UNIFIED_TASKS, normalize_task_key
from scripts.script_utils import calibrate_dataframe_pair, resolve_predictions_csv


# ==========================================
# Paths & Settings Defaults
# ==========================================
RESULTS_DIR = Path("results")
OUTPUT_IMAGE = Path("./tmp/conf_mats.png")

_LEGACY_MODEL_MAP = {
    "Transformer": "tf_raw_mt",
    "logistic_regression": "lr_raw",
    "random_forest": "rf_raw",
    "xgboost": "xgb_raw",
}


# ==========================================
# Data Loading Functions (Single Step)
# ==========================================
def _extract_step_from_csv_file(csv_path: Path | None, task: str, horizon: int, time_step: int) -> pd.DataFrame:
    """Reads one (task, horizon, fup) slice from a tidy '{split}_predictions.csv'."""
    if csv_path is None or not csv_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(csv_path)
    if "task" in df.columns:
        norm_t = normalize_task_key(task)
        df = df[df["task"].isin([task, norm_t])]
    df = df[(df["horizon"] == horizon) & (df["fup"] == time_step)]
    if df.empty:
        return pd.DataFrame()
    return pd.DataFrame({
        "patient_id": df["patient_id"].astype(str),
        "time_step": df["fup"].astype(int),
        "horizon": df["horizon"].astype(int),
        "y_true": df["label"].astype(int),
        "y_prob": df["prob"].astype(float),
    }).reset_index(drop=True)


def _find_csv_in_dir(root: Path, dataset_split: str, horizon: int) -> Path | None:
    if not root.exists():
        return None
    fname = f"{dataset_split}_predictions.csv"
    tag = f"{horizon:04d}"
    for x in sorted(root.iterdir(), key=lambda p: p.stat().st_mtime_ns if p.exists() else 0, reverse=True):
        if not x.is_dir() or not (x / fname).exists():
            continue
        m = re.match(r"hrz\(([0-9\-]+)\)", x.name)
        if m and tag in m.group(1).split("-"):
            return x / fname
    return None


def load_model_data(model_name: str, task: str, split: str, dataset_split: str, horizon: int, time_step: int, results_dir: Path = RESULTS_DIR) -> pd.DataFrame:
    """Loads single-(task, horizon, fup) predictions for any Transformer or Classic ML model."""
    norm_task = normalize_task_key(task)
    registry_candidates = [_LEGACY_MODEL_MAP[model_name], "tf_raw_mt", "tf_raw_pan", "tf_raw"] if model_name == "Transformer" else [
        _LEGACY_MODEL_MAP.get(model_name, model_name)
    ]
    for reg_key in registry_candidates:
        try:
            csv_path = resolve_predictions_csv(
                results_dir=results_dir,
                model_key=reg_key,
                split_type=split,
                task=norm_task,
                horizon=horizon,
                split_subset=dataset_split,
            )
            df = _extract_step_from_csv_file(csv_path, norm_task, horizon, time_step)
            if not df.empty:
                return df
        except KeyError:
            pass

    # Fallback for un-tagged classic_ml folders (e.g. classic_ml/<split>/xgboost/<task>)
    for t_cand in (norm_task, task):
        csv_path = _find_csv_in_dir(results_dir / "classic_ml" / split / model_name / t_cand, dataset_split, horizon)
        df = _extract_step_from_csv_file(csv_path, norm_task, horizon, time_step)
        if not df.empty:
            return df
    return pd.DataFrame()


# ==========================================
# Helpers & Calculations
# ==========================================
def compute_threshold_for_recall(df_val: pd.DataFrame, target_recall: float = 0.80) -> float:
    """Find the threshold that achieves target recall on validation set."""
    if df_val.empty or df_val["y_true"].nunique() < 2:
        return 0.5
    _, recall, thresholds = precision_recall_curve(df_val["y_true"], df_val["y_prob"])
    idx = np.where(recall[:-1] >= target_recall)[0]
    if len(idx) == 0:
        return float(thresholds[0]) if len(thresholds) else 0.5
    return float(thresholds[idx[-1]])


def calculate_mcnemar_sample_size(p10: float, p01: float, alpha: float = 0.05, power: float = 0.80, two_sided: bool = True):
    """Calculate sample size for McNemar's test given discordant proportions (Connor 1987 / Machin)."""
    p_diff = abs(p10 - p01)
    p_disc = p10 + p01
    if p_diff == 0 or p_disc == 0:
        return np.nan

    z_alpha = norm.ppf(1.0 - alpha / 2.0) if two_sided else norm.ppf(1.0 - alpha)
    z_beta = norm.ppf(power)
    n_required = ((z_alpha * np.sqrt(p_disc) + z_beta * np.sqrt(p_disc - p_diff**2)) / p_diff) ** 2
    return int(np.ceil(n_required))


# ==========================================
# Main Execution Flow
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Two-Model Evaluation & McNemar Sample Size Calculator")
    parser.add_argument("--model_a", type=str, default="tf_raw_mt",
                        help="First model (e.g. tf_raw_mt, tf_raw_unified, Transformer, xgboost, random_forest, logistic_regression)")
    parser.add_argument("--model_b", type=str, default="xgboost",
                        help="Second model to compare against model_a")
    parser.add_argument("--task", type=str, default="infection_bacteria",
                        help=f"Outcome task key (e.g. {', '.join(UNIFIED_TASKS[:4])}, death, graft_loss)")
    parser.add_argument("--split", type=str, default="random_split", choices=["random_split", "temporal_split", "center_split"])
    parser.add_argument("--time_step", "-T", type=int, default=180, help="Post-tpx follow-up day (e.g. 180)")
    parser.add_argument("--horizon", "-H", type=int, default=30, choices=UNIFIED_HORIZONS, help="Prediction horizon in days")
    parser.add_argument("--target_recall", type=float, default=0.80, help="Recall target for thresholding (0.0 - 1.0)")
    parser.add_argument("--calibrate", action=argparse.BooleanOptionalAction, default=True,
                        help="Apply validation-fitted L2 Platt scaling calibration (default: True; disable with --no-calibrate)")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR, help="Root results directory")
    parser.add_argument("--output-image", type=Path, default=OUTPUT_IMAGE, help="Output PNG path for confusion matrices")
    args = parser.parse_args()

    print(f"Loading data: Task={args.task} | Split={args.split} | FUP={args.time_step}d | Horizon={args.horizon}d")

    val_a = load_model_data(args.model_a, args.task, args.split, "validation", args.horizon, args.time_step, args.results_dir)
    test_a = load_model_data(args.model_a, args.task, args.split, "test", args.horizon, args.time_step, args.results_dir)
    val_b = load_model_data(args.model_b, args.task, args.split, "validation", args.horizon, args.time_step, args.results_dir)
    test_b = load_model_data(args.model_b, args.task, args.split, "test", args.horizon, args.time_step, args.results_dir)

    if any(df.empty for df in [val_a, test_a, val_b, test_b]):
        raise RuntimeError(
            f"Could not find matching prediction files for ({args.model_a} vs {args.model_b}, "
            f"task={args.task}, split={args.split}, fup={args.time_step}, horizon={args.horizon})."
        )

    keys = ["patient_id", "time_step", "horizon", "y_true"]
    common_val = val_a[keys].merge(val_b[keys], on=keys, how="inner").drop_duplicates()
    val_a = val_a.merge(common_val, on=keys).sort_values(keys).reset_index(drop=True)
    val_b = val_b.merge(common_val, on=keys).sort_values(keys).reset_index(drop=True)

    common_test = test_a[keys].merge(test_b[keys], on=keys, how="inner").drop_duplicates()
    test_a = test_a.merge(common_test, on=keys).sort_values(keys).reset_index(drop=True)
    test_b = test_b.merge(common_test, on=keys).sort_values(keys).reset_index(drop=True)

    if args.calibrate:
        val_a, test_a = calibrate_dataframe_pair(df_val=val_a, df_test=test_a, prob_col="y_prob")
        val_b, test_b = calibrate_dataframe_pair(df_val=val_b, df_test=test_b, prob_col="y_prob")

    thresh_a = compute_threshold_for_recall(val_a, target_recall=args.target_recall)
    thresh_b = compute_threshold_for_recall(val_b, target_recall=args.target_recall)

    y_test = test_a["y_true"].values
    pred_a = (test_a["y_prob"].values >= thresh_a).astype(int)
    pred_b = (test_b["y_prob"].values >= thresh_b).astype(int)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), dpi=150)

    cm_a = confusion_matrix(y_test, pred_a, labels=[0, 1])
    cm_b = confusion_matrix(y_test, pred_b, labels=[0, 1])

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
    args.output_image.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output_image, bbox_inches="tight")
    plt.close()
    print(f"\n[Saved] Confusion matrices plot written to {args.output_image.resolve()}")

    correct_a = (pred_a == y_test)
    correct_b = (pred_b == y_test)

    n_total = len(y_test)
    n_pos_gt = int(np.sum(y_test == 1))
    n_neg_gt = int(np.sum(y_test == 0))

    n11 = int(np.sum(correct_a & correct_b))
    n10 = int(np.sum(correct_a & ~correct_b))
    n01 = int(np.sum(~correct_a & correct_b))
    n00 = int(np.sum(~correct_a & ~correct_b))

    p10 = n10 / n_total
    p01 = n01 / n_total
    p_disc = p10 + p01
    delta_acc = p10 - p01

    tn_a, fp_a, fn_a, tp_a = cm_a.ravel()
    tn_b, fp_b, fn_b, tp_b = cm_b.ravel()

    mask_neg = (y_test == 0)
    mask_pos = (y_test == 1)

    n10_neg = int(np.sum(correct_a[mask_neg] & ~correct_b[mask_neg]))
    n01_neg = int(np.sum(~correct_a[mask_neg] & correct_b[mask_neg]))

    n10_pos = int(np.sum(correct_a[mask_pos] & ~correct_b[mask_pos]))
    n01_pos = int(np.sum(~correct_a[mask_pos] & correct_b[mask_pos]))

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
