import re
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Dict
from pathlib import Path
from collections import defaultdict
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    accuracy_score, balanced_accuracy_score, precision_score, 
    recall_score, f1_score, precision_recall_curve
)

# Configuration
TRANSFORMER_BASE_DIRS = {
    "infection_bacteria": Path("results_optuna/infection_bacteria/trial_033"),
    "infection_virus": Path("results_optuna/infection_virus/trial_034"),
    "death": Path("results_optuna/death/trial_040"),
    "graft_loss": Path("results_optuna/graft_loss/trial_000"),
}
TRANSFORMER_PT_CONFIGS = {
    "infection_bacteria": "e10-a10-v40",
    "infection_virus": "e10-a05-v45",
    "death": "e00-a20-v40",
    "graft_loss": "e10-a25-v30",
}
CLASSIC_ML_BASE_DIR = Path("results_classic_ml")
OUTPUT_DIR = Path("results_compared")
SPLIT_TYPES = [
    "random_split",
    "temporal_split",
    "center_split",
]
CLASSIC_ML_MODELS_TO_PLOT = [
    "logistic_regression",
    "random_forest",
    "xgboost",
]
BEST_CLASSIC_ML_MODEL = "xgboost"
TASKS = [
    "infection_bacteria", 
    "infection_virus",
    "death",
    "graft_loss",
]

# Clinical periods definition (horizon: [fups])
def get_phase_windows(start, end, horizons, step=30):
    return {h: w for h in horizons if (w := list(range(start, end + 1 - h, step)))}
CLINICAL_PERIODS_INFECTIONS = {
    "Perioperative\nphase (0-1 mo)": get_phase_windows(0, 30, [30, 60, 90]),
    "Opportunistic\nphase (1-6 mo)": get_phase_windows(30, 180, [30, 60, 90]),
    "Maintenance\nphase (6-12 mo)":  get_phase_windows(180, 360, [30, 60, 90]),
    "Long-term\nphase (1-2 yr)":     get_phase_windows(360, 720, [30, 60, 90]),
}
CLINICAL_PERIODS_OUTCOMES = {
    "Short-term\nphase (0-2 yr)":  get_phase_windows(0, 720, [360, 720, 1080, 1800]),
    "Middle-term\nphase (2-5 yr)": get_phase_windows(720, 1800, [360, 720, 1080, 1800]),
    "Long-term\nphase (5-10 yr)":  get_phase_windows(1800, 3600, [360, 720, 1080, 1800]),
}
CLINICAL_PERIOD_DICT = {
    "infection_bacteria": CLINICAL_PERIODS_INFECTIONS,
    "infection_virus": CLINICAL_PERIODS_INFECTIONS,
    "death": CLINICAL_PERIODS_OUTCOMES,
    "graft_loss": CLINICAL_PERIODS_OUTCOMES,
}

# Prognostic period definition (horizon: [fups])
PROGNOSTIC_PERIODS_INFECTIONS = {
    "Full length\n horizon (30 d)":  get_phase_windows(0, 3600, [30]),
    "Full length\n horizon (60 d)":  get_phase_windows(0, 3600, [60]),
    "Full length\n horizon (90 d)":  get_phase_windows(0, 3600, [90]),
}
PROGNOSTIC_PERIODS_OUTCOMES = {
    "Full length\n horizon (30 d)":  get_phase_windows(0, 3600, [30]),
    "Full length\n horizon (60 d)":  get_phase_windows(0, 3600, [60]),
    "Full length\n horizon (90 d)":  get_phase_windows(0, 3600, [90]),
}
PROGNOSTIC_PERIOD_DICT = {
    "infection_bacteria": PROGNOSTIC_PERIODS_INFECTIONS,
    "infection_virus": PROGNOSTIC_PERIODS_INFECTIONS,
    "death": PROGNOSTIC_PERIODS_OUTCOMES,
    "graft_loss": PROGNOSTIC_PERIODS_OUTCOMES,
}

# What is evaluated
# For threshold-fixed metrics, use "<metric_name>_t<level>", e.g., "f1_t10"
# For recall-fixed metrics, use "<metric_name>_rec<level>", e.g., "f1_rec90"
# For best-f1-score metrics, use "<metric_name>_best_f1", e.g., "f1_best_f1"
METRICS_OF_INTEREST = {
    "roc_auc": "ROC AUC (→)",
    "pr_auc": "PR AUC (→)",
    "brier": "Brier score (←)",
    "ece": "ECE (←)",
    "bal_acc_t10": "Balanced acc. (→)",
    # "recall_t10": "Recall (→)",
    # "precision_t10": "Precision (→)",
    "f1_t10": "F1-score (→)",
    "nb_t10": "Net benefit (→)",
}

# For plotting
COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", 
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
]
AGGREGATION_SCHEME = [(float('inf'), 1)]


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print(">>> Loading raw prediction data across all splits...")
    raw_data_pool = load_all_raw_predictions()
    
    if not raw_data_pool:
        print("No results found. Check your paths in the configuration!")
        return

    print(">>> Computing metrics...")
    granular_df = compute_granular_metrics(raw_data_pool)
    period_df = compute_period_metrics(raw_data_pool)

    # Process each task in its own dedicated figure
    for task in TASKS:
        print(f"\n{'='*50}")
        print(f">>> Processing Task: {task.upper()}")
        print(f"{'='*50}")
        
        # Filter data for the current task
        task_granular = granular_df[granular_df["Task"] == task].copy()
        task_period = period_df[period_df["Task"] == task].copy()

        if task_granular.empty or task_period.empty:
            print(f"No data found for task '{task}'. Skipping.")
            continue

        print(f"  Generating delta heatmap for {task}...")
        plot_delta_heatmap(
            df=task_granular, 
            baseline_name=BEST_CLASSIC_ML_MODEL, 
            transformer_name="Transformer",
            output_dir=OUTPUT_DIR,
            task_name=task,
        )

        print(f"  Generating period performance bar charts for {task}...")
        plot_period_performance_bars(
            df=task_period,
            task_name=task,
            output_dir=OUTPUT_DIR,
        )
        
        print(f"  Generating performance tables for {task}...")
        generate_performance_summaries(
            df=task_period, 
            output_dir=OUTPUT_DIR, 
            task_name=task,
        )
    
    print("\nDone.")


def expected_calibration_error(y_true, y_prob, n_bins=10):
    """
    Calculates the expected calibration error (ECE) across predefined bins
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_prob, bin_boundaries[1:-1])
    ece = 0.0
    for i in range(n_bins):
        in_bin = (bin_indices == i)
        n_in_bin = np.sum(in_bin)
        if n_in_bin > 0:
            acc_in_bin = np.mean(y_true[in_bin])
            conf_in_bin = np.mean(y_prob[in_bin])
            ece += np.abs(acc_in_bin - conf_in_bin) * n_in_bin
    return ece / len(y_true)


def get_metrics_from_arrays(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    y_cal: np.ndarray,
) -> dict:
    """
    Computes metrics dynamically based on the requested metric of interest
    """
    if len(np.unique(y_true)) < 2: 
        return {}

    results = {
        "roc_auc": roc_auc_score(y_true, y_prob),
        "pr_auc": average_precision_score(y_true, y_prob),
        "brier": brier_score_loss(y_true, y_cal),
        "ece": expected_calibration_error(y_true, y_cal),
    }

    # Helper to calculate Net Benefit
    def calc_net_benefit(preds, t):
        tp = np.sum((preds == 1) & (y_true == 1))
        fp = np.sum((preds == 1) & (y_true == 0))
        n = len(y_true)
        weight = t / (1 - t) if t < 1.0 else 0
        return (tp / n) - (fp / n) * weight

    # Parse requested thresholds
    requested_fixed_thresholds = set()
    requested_recall_targets = set()
    needs_best_f1 = False
    
    for key in METRICS_OF_INTEREST.keys():
        if "_best_f1" in key:
            needs_best_f1 = True
        else:
            # Check for fixed thresholds (e.g., _t05)
            match_t = re.search(r'_t(\d+)$', key)
            if match_t:
                requested_fixed_thresholds.add(match_t.group(1))
            
            # Check for target recall thresholds (e.g., _rec90)
            match_rec = re.search(r'_rec(\d+)$', key)
            if match_rec:
                requested_recall_targets.add(match_rec.group(1))

    # Compute PR curve just once if needed by Best F1 or Target Recall
    if needs_best_f1 or requested_recall_targets:
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)

    # Compute F1-optimal threshold if requested
    if needs_best_f1:
        denominator = precisions + recalls
        denominator[denominator == 0] = 1e-9
        f1_scores = (2 * precisions * recalls) / denominator
        f1_scores = f1_scores[:-1] 
        
        best_thresh = thresholds[np.argmax(f1_scores)] if len(f1_scores) > 0 else 0.5
        preds_best = (y_prob >= best_thresh).astype(int)
        
        results.update({
            "acc_best_f1": accuracy_score(y_true, preds_best),
            "bal_acc_best_f1": balanced_accuracy_score(y_true, preds_best),
            "precision_best_f1": precision_score(y_true, preds_best, zero_division=0),
            "recall_best_f1": recall_score(y_true, preds_best, zero_division=0),
            "f1_best_f1": f1_score(y_true, preds_best, zero_division=0),
            "nb_best_f1": calc_net_benefit(preds_best, best_thresh)
        })

    # Compute dynamically for target recall thresholds
    for rec_str in requested_recall_targets:
        target_recall = int(rec_str) / 100.0
        
        valid_idx = np.where(recalls[:-1] >= target_recall)[0]
        if len(valid_idx) > 0:
            rec_thresh = thresholds[valid_idx[-1]]
        else:
            rec_thresh = 0.0 
            
        preds_rec = (y_prob >= rec_thresh).astype(int)
        rec_suffix = f"rec{rec_str}"
        
        results.update({
            f"acc_{rec_suffix}": accuracy_score(y_true, preds_rec),
            f"bal_acc_{rec_suffix}": balanced_accuracy_score(y_true, preds_rec),
            f"precision_{rec_suffix}": precision_score(y_true, preds_rec, zero_division=0),
            f"recall_{rec_suffix}": recall_score(y_true, preds_rec, zero_division=0),
            f"f1_{rec_suffix}": f1_score(y_true, preds_rec, zero_division=0),
            f"nb_{rec_suffix}": calc_net_benefit(preds_rec, rec_thresh)
        })

    # Compute dynamically for fixed 'tXX' thresholds
    for t_str in requested_fixed_thresholds:
        t_float = int(t_str) / 100.0
        preds_t = (y_prob >= t_float).astype(int)
        t_suffix = f"t{t_str}"
        
        results.update({
            f"acc_{t_suffix}": accuracy_score(y_true, preds_t),
            f"bal_acc_{t_suffix}": balanced_accuracy_score(y_true, preds_t),
            f"precision_{t_suffix}": precision_score(y_true, preds_t, zero_division=0),
            f"recall_{t_suffix}": recall_score(y_true, preds_t, zero_division=0),
            f"f1_{t_suffix}": f1_score(y_true, preds_t, zero_division=0),
            f"nb_{t_suffix}": calc_net_benefit(preds_t, t_float)
        })

    return results


def load_all_raw_predictions() -> Dict:
    raw_pool = defaultdict(lambda: defaultdict(
        lambda: defaultdict(lambda: defaultdict(dict))
    ))
    
    for split in SPLIT_TYPES:
        for task in TASKS:
            # Transformer data
            base_dir = TRANSFORMER_BASE_DIRS[task]
            pt_config = TRANSFORMER_PT_CONFIGS[task]
            t_path = base_dir / split / pt_config / "finetuning" / task
            if t_path.exists():
                _extract_npz_to_pool(t_path, "Transformer", split, task, raw_pool)
                
            # Classic ML data
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


def compute_granular_metrics(raw_pool: Dict) -> pd.DataFrame:
    """
    Calculates metrics individually for every FUP/horizon
    """
    records = []
    for split, models in raw_pool.items():
        for model, tasks in models.items():
            for task, horizons in tasks.items():
                for h, fups in horizons.items():
                    for fup, arrays in fups.items():
                        
                        y_true, y_prob, y_cal = arrays["labels"], arrays["probs"], arrays["probs_cal"]
                        mask = y_true != -100
                        if not mask.any(): continue
                        
                        metrics = get_metrics_from_arrays(y_true[mask], y_prob[mask], y_cal[mask])
                        for m_name, m_val in metrics.items():
                            if m_name in METRICS_OF_INTEREST:
                                records.append({
                                    "Split": split, "Model": model, "Task": task, 
                                    "Horizon": h, "FUP": fup, "Metric": m_name, "Value": m_val
                                })
    
    return pd.DataFrame(records)


def compute_period_metrics(raw_pool: Dict) -> pd.DataFrame:
    """
    Concatenates arrays based on clinical periods before calculating metrics
    """
    records = []
    for split, models in raw_pool.items():
        for model, tasks in models.items():
            for task, horizons_data in tasks.items():
                clinical_periods = CLINICAL_PERIOD_DICT[task]
                for period_name, h_fup_map in clinical_periods.items():
                    
                    y_true_list, y_prob_list, y_cal_list = [], [], []
                    for h, fups in h_fup_map.items():
                        if h not in horizons_data: continue
                        for fup in fups:
                            if fup in horizons_data[h]:
                                d = horizons_data[h][fup]
                                y_true_list.append(d["labels"])
                                y_prob_list.append(d["probs"])
                                y_cal_list.append(d["probs_cal"])
                            
                    if not y_true_list: continue
                    y_true = np.concatenate(y_true_list)
                    y_prob = np.concatenate(y_prob_list)
                    y_cal = np.concatenate(y_cal_list)
                    
                    mask = y_true != -100
                    if not mask.any(): continue
                    
                    metrics = get_metrics_from_arrays(y_true[mask], y_prob[mask], y_cal[mask])
                    for m_name, m_val in metrics.items():
                        if m_name in METRICS_OF_INTEREST:
                            records.append({
                                "Split": split, "Model": model, "Task": task,
                                "Period": period_name, "Metric": m_name, "Value": m_val
                            })
                            
    return pd.DataFrame(records)


def plot_delta_heatmap(df: pd.DataFrame, baseline_name: str, transformer_name: str, output_dir: Path, task_name: str):
    df_comp = df[df["Model"].isin([baseline_name, transformer_name])].copy()
    if df_comp.empty: return

    pivot = df_comp.pivot_table(
        index=["Split", "Task", "Horizon", "FUP", "Metric"], 
        columns="Model", values="Value"
    ).dropna().reset_index()
    
    if transformer_name not in pivot.columns or baseline_name not in pivot.columns:
        return

    pivot["Delta"] = pivot[transformer_name] - pivot[baseline_name]
    metrics = list(METRICS_OF_INTEREST.keys())
    fig, axes = plt.subplots(
        len(metrics), len(SPLIT_TYPES), 
        figsize=(8 * len(SPLIT_TYPES), 1.5 * len(metrics)),
        squeeze=False,
    )
    
    title_suffix = task_name.replace('_', ' ').title()
    fig.suptitle(
        f"Performance Delta: {transformer_name} - {baseline_name} ({title_suffix})", 
        fontsize=20, fontweight='bold'
    )

    for row_idx, metric in enumerate(metrics):
        metric_data = pivot[pivot["Metric"] == metric].copy()
        
        if metric_data.empty: vmin, vmax = -0.1, 0.1 
        else:
            limit = metric_data["Delta"].abs().max() * 1.1 or 0.1
            vmin, vmax = -limit, limit

        for col_idx, split in enumerate(SPLIT_TYPES):
            ax = axes[row_idx, col_idx]
            subset = metric_data[metric_data["Split"] == split].copy()
            
            if subset.empty:
                ax.text(0.5, 0.5, "No Data", ha='center', va='center', color='gray')
                ax.set_axis_off()
                continue

            subset["Y_Label"] = "Horizon " + subset["Horizon"].astype(str) + "d"
            matrix = subset.pivot(index="Y_Label", columns="FUP", values="Delta").sort_index()
            
            fups = sorted(matrix.columns)
            y_labels = matrix.index.tolist()
            
            widths = [next((s for limit, s in AGGREGATION_SCHEME if f < limit), 1) for f in fups]
            x_edges = [0] + list(np.cumsum(widths))
            y_edges = range(len(y_labels) + 1)

            mesh = ax.pcolormesh(
                x_edges, y_edges, matrix.values, 
                cmap="RdBu", vmin=vmin, vmax=vmax, 
                edgecolors='white', linewidth=0.5 
            )
            
            ax.set_yticks([y + 0.5 for y in range(len(y_labels))])
            ax.set_yticklabels(y_labels, fontsize=11)
            step = max(1, len(fups) // 10)
            x_ticks_positions = [(x_edges[i] + x_edges[i+1])/2 for i in range(len(fups))]
            ax.set_xticks(x_ticks_positions[::step])
            ax.set_xticklabels(fups[::step], rotation=45, ha='right', fontsize=9)
            ax.invert_yaxis()
            
            if row_idx == 0: 
                ax.set_title(split.replace('_', ' ').title(), fontsize=14, fontweight='bold')
            if col_idx == 0: 
                ax.set_ylabel(METRICS_OF_INTEREST[metric], fontsize=13, fontweight='bold')
            
            cbar = fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('Delta')

    # Apply tight layout to compress whitespace, leaving room for the top title
    out_file = output_dir / f"delta_heatmap_{task_name}.png"
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(out_file, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"    [Plot Saved] {out_file.name}")


def plot_period_performance_bars(
    df: pd.DataFrame,
    task_name: str,
    period_type: str,
    output_dir: Path,
):
    metrics = list(METRICS_OF_INTEREST.keys())
    model_order = CLASSIC_ML_MODELS_TO_PLOT + ["Transformer"]
    col_keys = [m for m in model_order if m in df["Model"].unique()]
    
    row_keys = metrics
    if not row_keys or not col_keys: return

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(
        len(row_keys), len(SPLIT_TYPES),
        figsize=(6 * len(SPLIT_TYPES), 4.0 * len(row_keys)), 
        sharex=True, sharey="row", squeeze=False,
    )

    for r, metric in enumerate(row_keys):
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

            if r == 0:
                ax.set_title(
                    split.replace('_', ' ').title(),
                    fontsize=14, fontweight='bold',
                )
            if c == 0:
                y_label = f"{METRICS_OF_INTEREST[metric]}"
                ax.set_ylabel(y_label, fontsize=11, fontweight='bold')
            else:
                ax.set_ylabel("")
            
            ax.set_xlabel("Clinical Period" if r == len(row_keys)-1 else "")
            ax.tick_params(labelbottom=(r == len(row_keys)-1))
            if ax.get_legend(): ax.get_legend().remove()

    handles, labels = axes[0,0].get_legend_handles_labels()
    title_suffix = task_name.replace('_', ' ').title()
    fig.legend(
        handles, labels, loc="lower center", frameon=False,
        bbox_to_anchor=(0.5, 1.0), ncol=len(labels),
        fontsize=13, title_fontsize=14, title=f"Model Architecture ({title_suffix})",
    )

    plt.tight_layout(rect=[0, 0, 1, 1.0])
    out_file = output_dir / f"{period_type}_period_performance{task_name}.png"
    plt.savefig(out_file, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"    [Plot Saved] {out_file.name}")


def generate_performance_summaries(df: pd.DataFrame, output_dir: Path, task_name: str):
    grouped = df.groupby(["Period", "Metric", "Model"])["Value"].agg(["mean", "std"])
    grouped["Formatted"] = grouped.apply(lambda r: f"{r['mean']:.3f} ± {r['std']:.3f}", axis=1)
    
    pivot_table = grouped.reset_index().pivot(
        index=["Period", "Metric"], columns="Model", values="Formatted"
    ).reset_index()
    
    order = CLASSIC_ML_MODELS_TO_PLOT + ["Transformer"]
    cols = ["Period", "Metric"] + [m for m in order if m in pivot_table.columns]
    pivot_table = pivot_table[[c for c in cols if c in pivot_table.columns]]

    title_suffix = task_name.replace('_', ' ').title()
    out_file = output_dir / f"period_summary_{task_name}.md"
    
    with open(out_file, "w") as f:
        f.write(f"# Performance Summary: {title_suffix}\n")
        f.write("*Metrics computed by pooling predictions across specified clinical periods.*\n\n")
        f.write(pivot_table.to_markdown(index=False))
        
    print(f"    [Table Saved] {out_file.name}")


if __name__ == "__main__":
    main()