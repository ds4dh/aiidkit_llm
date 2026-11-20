import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import gc
import re
import yaml
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")

import argparse
from collections import Counter
from itertools import combinations
from concurrent.futures import ThreadPoolExecutor
from datasets import load_from_disk
from torch.utils.data import DataLoader
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from sklearn.calibration import calibration_curve
from scipy.stats import fisher_exact
from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test

from src.model.model_utils import build_monotonic_groups, resolve_best_checkpoint
from src.model.patient_embedder import PatientEmbeddingModelFactory, PatientDataCollatorForClassification
from src.evaluation.evaluate_models import UMAP_HDBSCAN_Clusterer, ModelInterpreter
from src.data.patient_dataset import load_hf_data_and_metadata
from src.data.survival_labels import derive_category_survival_labels, resolve_survival_label_keys
from src.tasks import abbreviate_feature_name, format_eav_feature_name, to_survival_task_key
from scripts.script_utils import (
    calibrate_array_pair,
    extract_horizons_from_path,
    find_best_checkpoint,
    find_best_checkpoint_multitask,
    resolve_base_dir_for_ckpt,
    scan_all_fups,
)

parser = argparse.ArgumentParser(description="Run patient stratification and survival analysis.")
parser.add_argument("--results-dir", type=Path, default=Path("results"), help="Base directory for results")
parser.add_argument("--data-split-type", "--data_split_type", type=str, default="random_split", help="Data split strategy")
parser.add_argument("--data-dir", "--data_dir", type=Path, default=Path(os.environ.get("TEAV_DATA_DIR", "/home/shares/ds4dh/aiidkit_project/data_new/processed/v3.9/teav")), help="Path to tEAV dataset directory")
parser.add_argument("--proposed-model", "--proposed_model", type=str, default="tf_raw_mt",
                    help="Model analyzed by this script (tf_raw_mt, tf_raw_st, tf_bin_mt, tf_bin_st).")
parser.add_argument("--run-id", "--run_id", type=str, default="",
                    help="Pre-training run id subfolder (e.g. e00-a15-v60). Default: the only folder present.")
parser.add_argument("--config", "-c", type=str, default="configs/discriminative_training.yaml",
                    help="Training config providing model/collator settings for checkpoint loading.")
parser.add_argument("--task", "-t", type=str, default=None,
                    help="Single outcome task to stratify patients for (e.g. infection_bacteria or infection_grp_cmv). Default: infection_bacteria.")
parser.add_argument("--tasks", nargs="+", default=None,
                    help="Outcome tasks to stratify patients for (one analysis per task). "
                         "Default: infection_bacteria. Multi-task checkpoints are supported: "
                         "the matching (task, horizon) head is selected per task.")
parser.add_argument("--horizon", type=int, default=None,
                    help="Prediction horizon in days for patient stratification (default: 360 for death/graft_loss, 90 for infections).")
args, _ = parser.parse_known_args()

# Model selection
RESULTS_DIR = args.results_dir
DATA_SPLIT_TYPE = args.data_split_type
PROPOSED_MODEL = args.proposed_model
RUN_ID = args.run_id.strip()

# Stratification clusters and scores patients with the fine-tuned transformer
# embeddings + classification head, so only transformer-family binary models
# are supported (classic ML contenders have no embeddings; the Cox survival
# variant uses a regression head, not wired in yet).
STRATIFIABLE_MODELS = {
    # model key -> (results subfolder, force multi-task head resolution, value embedding tag, mt_scope)
    "tf_raw": ("transformer", False, "raw", "any"),
    "tf_raw_st": ("transformer", False, "raw", "any"),
    "tf_raw_mt": ("transformer", True, "raw", "domain"),
    "tf_raw_unified": ("transformer", True, "raw", "unified"),
    "tf_bin": ("transformer", False, "bin", "any"),
    "tf_bin_st": ("transformer", False, "bin", "any"),
    "tf_bin_mt": ("transformer", True, "bin", "domain"),
    "tf_bin_unified": ("transformer", True, "bin", "unified"),
    "tfg_raw": ("generative_transformer", False, "raw", "any"),
    "tfg_raw_st": ("generative_transformer", False, "raw", "any"),
    "tfg_raw_mt": ("generative_transformer", True, "raw", "domain"),
    "tfg_raw_unified": ("generative_transformer", True, "raw", "unified"),
    "tfg_bin": ("generative_transformer", False, "bin", "any"),
    "tfg_bin_st": ("generative_transformer", False, "raw", "any"),
    "tfg_bin_mt": ("generative_transformer", True, "bin", "domain"),
    "tfg_bin_unified": ("generative_transformer", True, "bin", "unified"),
}
if PROPOSED_MODEL not in STRATIFIABLE_MODELS:
    print(f"[SKIP] Stratification analysis supports only {sorted(STRATIFIABLE_MODELS)}; "
          f"got proposed model '{PROPOSED_MODEL}'. Nothing to do.")
    sys.exit(0)
TRANSFORMER_BASE_DIR = RESULTS_DIR / STRATIFIABLE_MODELS[PROPOSED_MODEL][0]
MULTITASK_ONLY = STRATIFIABLE_MODELS[PROPOSED_MODEL][1]
MODEL_VAL_TAG = STRATIFIABLE_MODELS[PROPOSED_MODEL][2]
MT_SCOPE = STRATIFIABLE_MODELS[PROPOSED_MODEL][3]
# Split included in the output path so the three split types don't overwrite
# each other (consistent with the interpretability analysis layout).
OUTPUT_DIR = RESULTS_DIR / Path("analysis/stratification") / PROPOSED_MODEL / DATA_SPLIT_TYPE
CONFIG_PATH = Path(args.config) 
DATA_DIR = args.data_dir

# Configuration
FROM_OPTUNA = "optuna" in TRANSFORMER_BASE_DIR.name
DATA_SPLIT_TYPE = args.data_split_type
TASKS = [args.task] if args.task else (list(args.tasks) if args.tasks else ['infection_bacteria'])
MAX_FUP = 3600
if DATA_SPLIT_TYPE == "temporal_split":
    MAX_FUP = 2400
BASE_FUP_FOR_PREDICTION = 90
PREDICTION_HORIZON = args.horizon
SAFE_NUM_PROC = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Fontsize scaling configurations
FS_SCALE = 1.15
plt.rcParams.update({
    'font.size': 13 * FS_SCALE,
    'axes.labelsize': 15 * FS_SCALE,
    'axes.titlesize': 15 * FS_SCALE,
    'xtick.labelsize': 12 * FS_SCALE,
    'ytick.labelsize': 12 * FS_SCALE,
    'legend.fontsize': 11 * FS_SCALE,
    'legend.title_fontsize': 12 * FS_SCALE,
    'figure.titlesize': 20 * FS_SCALE,
})


def render_odds_ratio_subplot(ax, df_subset, title_suffix=""):
    """Helper rendering a single column forest plot for odds ratios on a given Axes."""
    if df_subset is None or df_subset.empty:
        ax.set_visible(False)
        return

    # Invert so highest ranking feature is at the top of the vertical axis
    plot_df = df_subset.iloc[::-1].reset_index(drop=True)
    y_positions = np.arange(len(plot_df))

    # Reference neutral vertical line at OR = 1.0
    ax.axvline(1.0, color="#4d4d4d", linewidth=1.8, zorder=2)

    # Clamped error bars (strictly non-negative)
    err_lower = np.maximum(0.0, plot_df["Odds_Ratio"].to_numpy() - plot_df["ci_lower"].to_numpy())
    err_upper = np.maximum(0.0, plot_df["ci_upper"].to_numpy() - plot_df["Odds_Ratio"].to_numpy())
    asymmetric_err = [err_lower, err_upper]

    ax.errorbar(
        plot_df["Odds_Ratio"],
        y_positions,
        xerr=asymmetric_err,
        fmt='o',
        color="#154b70",
        ecolor="#154b70",
        elinewidth=2.0,
        capsize=4.0,
        capthick=1.6,
        markersize=7.5,
        zorder=3
    )

    ax.set_yticks(y_positions)
    ax.set_yticklabels(
        plot_df["display_feature"],
        color="#222222",
        fontsize=13 * FS_SCALE,
        ha='right'
    )
    ax.tick_params(axis='y', which='major', pad=8)
    ax.set_xscale("log")

    min_val = max(0.1, np.nanmin(plot_df["ci_lower"]))
    max_val = max(2.0, np.nanmax(plot_df["ci_upper"]))
    ax.set_xlim(min_val * 0.7, max_val * 1.4)

    # Explicit sparse ticks to prevent crowding while retaining legible font size
    tick_candidates = [1.0, 5.0, 20.0, 50.0, 200.0, 500.0]
    ticks_to_use = [t for t in tick_candidates if (min_val * 0.7) <= t <= (max_val * 1.4)]
    if 1.0 not in ticks_to_use:
        ticks_to_use.append(1.0)
    ticks_to_use = sorted(ticks_to_use)

    ax.set_xticks(ticks_to_use)
    ax.set_xticklabels(
        [f"{t:g}x" if t != 1.0 else "1.0 (ref)" for t in ticks_to_use],
        fontsize=13 * FS_SCALE
    )
    ax.grid(True, axis='x', color="#e5e5e5", linestyle='--', linewidth=0.8, zorder=1)

    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#888888")
    ax.set_xlabel("Odds ratio (log scale)", labelpad=8, fontsize=15 * FS_SCALE)

    # Congruent title: centered above the subplot area, matching font weight and scale
    title_text = f"Enriched clinical features: high-risk cluster {title_suffix}"
    ax.text(
        0.22, 1.0, title_text,
        transform=ax.transAxes,
        ha='center', va='bottom',
        fontsize=15 * FS_SCALE,
        color="#111111"
    )


def plot_combined_stratification_grid(
    test_ds, patient_ids, probs, reduced_pt, reduced_ft, labels_pt, labels_ft,
    cluster_color_map, true_labels_horizon, task_key, label_key, df_drivers=None,
    surv_times=None, surv_statuses=None,
):
    survival_data = []
    if surv_times is not None and surv_statuses is not None:
        # Authoritative post-landmark survival times & right-censoring statuses from base_test_ds
        for i, pid in enumerate(patient_ids):
            t_val = float(surv_times[i])
            s_val = int(surv_statuses[i])
            if s_val >= 0 and t_val >= 0:
                duration = max(0.0, t_val)
                event = 1 if s_val == 1 else 0
                true_imminence = float(np.exp(-duration / 365.0)) if event == 1 else 0.0
            else:
                duration = 0.0
                event = 0
                true_imminence = 0.0
            survival_data.append({
                "pid": pid,
                "duration": duration,
                "event": event,
                "risk_score": probs[i],
                "true_imminence": true_imminence,
                "cluster_ft": labels_ft[i],
            })
    else:
        df_long = test_ds if isinstance(test_ds, pd.DataFrame) else test_ds.to_pandas()
        target_pids = set(patient_ids)
        df_long = df_long[(df_long["patientid"].isin(target_pids)) & (df_long["fup"] >= BASE_FUP_FOR_PREDICTION)].sort_values("fup")
        grouped_by_pid = {pid: grp for pid, grp in df_long.groupby("patientid", sort=False)}
        for i, pid in enumerate(patient_ids):
            p_data = grouped_by_pid.get(pid)
            if p_data is not None and not p_data.empty:
                future_events = p_data[p_data[label_key] == 1]
                if not future_events.empty:
                    first_event_fup = future_events["fup"].iloc[0]
                    duration = first_event_fup - BASE_FUP_FOR_PREDICTION
                    event = 1
                    true_imminence = np.exp(-duration / 365.0)
                else:
                    last_fup = p_data["fup"].iloc[-1]
                    duration = last_fup - BASE_FUP_FOR_PREDICTION
                    event = 0
                    true_imminence = 0.0
            else:
                duration = 0
                event = 0
                true_imminence = 0.0
            survival_data.append({
                "pid": pid,
                "duration": max(0, duration),
                "event": event,
                "risk_score": probs[i],
                "true_imminence": true_imminence,
                "cluster_ft": labels_ft[i],
            })

    df_surv = pd.DataFrame(survival_data)
    blue_to_red_cmap = plt.get_cmap("coolwarm")
    gray_to_red_cmap = LinearSegmentedColormap.from_list(name="blue_grey_red", colors=["lightgrey", "tab:red"])
    
    fig = plt.figure(figsize=(24, 30))
    outer_vertical = gridspec.GridSpec(2, 1, height_ratios=[4.0, 1.0], hspace=0.14)
    gs_top = gridspec.GridSpecFromSubplotSpec(3, 2, subplot_spec=outer_vertical[0], hspace=0.28, wspace=0.22)

    s_censored = 90
    s_event = 190
    censored_mask = df_surv["event"] == 0
    event_mask = df_surv["event"] == 1

    # Row 0: Discovered clusters
    ax_pt = fig.add_subplot(gs_top[0, 0])
    ax_ft = fig.add_subplot(gs_top[0, 1])
    plot_scatter_unsupervised(
        ax_pt, reduced_pt, labels_pt, true_labels_horizon, cluster_color_map, 
        "Discovered clusters: pre-trained embeddings"
    )
    plot_scatter_unsupervised(
        ax_ft, reduced_ft, labels_ft, true_labels_horizon, cluster_color_map, 
        "Discovered clusters: fine-tuned embeddings"
    )

    # Row 1, Col 0: Model risk score
    ax_risk = fig.add_subplot(gs_top[1, 0])
    max_risk_val = df_surv["risk_score"].max()
    ax_risk.scatter(
        reduced_ft[censored_mask, 0], reduced_ft[censored_mask, 1],
        c=df_surv[censored_mask]["risk_score"], cmap=blue_to_red_cmap, vmin=0.0, vmax=max_risk_val,
        marker="o", s=s_censored, edgecolor="black", linewidth=0.5, zorder=2, alpha=0.65  
    )
    if event_mask.sum() > 0:
        sc2 = ax_risk.scatter(
            reduced_ft[event_mask, 0], reduced_ft[event_mask, 1],
            c=df_surv[event_mask]["risk_score"], cmap=blue_to_red_cmap, vmin=0.0, vmax=max_risk_val,
            marker="X", s=s_event, edgecolor="black", linewidth=1.0, zorder=3, alpha=0.90  
        )
    sc2_ref = sc2 if event_mask.sum() > 0 else ax_risk.collections[0]
    cbar2 = plt.colorbar(sc2_ref, ax=ax_risk, ticks=np.arange(0.0, max_risk_val + 0.01, 0.05))
    cbar2.ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
    cbar2.set_label("Predicted risk score")
    ax_risk.set_title("Model risk score: fine-tuned embeddings")
    ax_risk.set_xlabel("UMAP 1")
    ax_risk.set_ylabel("UMAP 2")

    risk_legend_elements = [
        Line2D([0], [0], marker='o', color='w', label="No event (censored)",
               markerfacecolor='gray', markersize=10, alpha=0.65, markeredgecolor='black', markeredgewidth=0.5),
        Line2D([0], [0], marker='X', color='w', label="Event occurred (ever)",
               markerfacecolor='gray', markersize=12, alpha=0.90, markeredgecolor='black', markeredgewidth=1.0)
    ]
    ax_risk.legend(handles=risk_legend_elements, loc="upper right")

    # Row 1, Col 1: Event imminence
    ax_imm = fig.add_subplot(gs_top[1, 1])
    ax_imm.scatter(
        reduced_ft[censored_mask, 0], reduced_ft[censored_mask, 1],
        c="lightgrey", marker="o", s=s_censored, edgecolor="black", linewidth=0.8,
        zorder=2, alpha=0.4  
    )
    if event_mask.sum() > 0:
        sc1 = ax_imm.scatter(
            reduced_ft[event_mask, 0], reduced_ft[event_mask, 1],
            c=df_surv[event_mask]["true_imminence"], cmap=gray_to_red_cmap, vmin=0.0, vmax=1.0,
            marker="X", s=s_event, edgecolor="black", linewidth=0.8, zorder=3, alpha=0.75  
        )
        cbar1 = plt.colorbar(sc1, ax=ax_imm, ticks=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        cbar1.ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))
        cbar1.set_label("Event imminence (exponential decay)")
    ax_imm.set_title("Event imminence: fine-tuned embeddings")
    ax_imm.set_xlabel("UMAP 1")
    ax_imm.set_ylabel("UMAP 2")

    imminence_legend_elements = [
        Line2D([0], [0], marker='o', color='w', label="No event (censored)",
               markerfacecolor='gray', markersize=10, alpha=0.4, markeredgecolor='black', markeredgewidth=0.8),
        Line2D([0], [0], marker='X', color='w', label="Event occurred (ever)",
               markerfacecolor='gray', markersize=12, alpha=0.75, markeredgecolor='black', markeredgewidth=0.8)
    ]
    ax_imm.legend(handles=imminence_legend_elements, loc="upper right")

    # Row 2, Col 0: Stratified KM curves
    ax_km = fig.add_subplot(gs_top[2, 0])
    kmf = KaplanMeierFitter()
    unique_clusters = sorted(df_surv["cluster_ft"].unique())
    for cid in unique_clusters:
        if cid == -1: continue
        mask = df_surv["cluster_ft"] == cid
        if mask.sum() == 0: continue
            
        lbl = f"Cluster {cid}"
        clr = cluster_color_map.get(cid, "darkgrey")
        abs_duration = df_surv[mask]["duration"] + BASE_FUP_FOR_PREDICTION
        entry_t = np.full(mask.sum(), BASE_FUP_FOR_PREDICTION)
        kmf.fit(abs_duration, df_surv[mask]["event"], entry=entry_t, label=lbl)
        kmf.plot_survival_function(color=clr, ci_show=True, ax=ax_km, linewidth=2.5, alpha=1.0)

    df_stat = df_surv[df_surv["cluster_ft"] != -1]
    if len(df_stat["cluster_ft"].unique()) > 1:
        results = multivariate_logrank_test(
            event_durations=df_stat["duration"],
            groups=df_stat["cluster_ft"],
            event_observed=df_stat["event"]
        )
        p_val_text = "log-rank p-value: < 0.001" if results.p_value < 0.001 else f"log-rank p-value: {results.p_value:.2e}"
    else:
        p_val_text = "log-rank p-value: N/A"

    num_strips = 100
    g_start = -100
    x_edges = np.linspace(g_start, BASE_FUP_FOR_PREDICTION, num_strips + 1)
    alphas = np.linspace(0.0, 0.4, num_strips)  
    for i in range(num_strips):
        ax_km.axvspan(x_edges[i], x_edges[i+1], facecolor='black', alpha=alphas[i], zorder=1, lw=0)

    ax_km.set_xlim(g_start, MAX_FUP)
    ax_km.text(
        0, 0.5, f"Observation window (-∞; {BASE_FUP_FOR_PREDICTION} days]",
        rotation=90, va='center', ha='center', color='#333333', zorder=2,
        fontweight='bold', transform=ax_km.get_xaxis_transform()
    )
    ax_km.set_title(f"Longitudinal risk stratification ({p_val_text})")
    ax_km.set_xlabel("Days since transplantation")
    ax_km.set_ylabel("Probability of remaining event-free")
    ax_km.grid(True, axis='y', alpha=0.3)
    ax_km.legend(loc="upper right")

    # Row 2, Col 1: Correlation Plot
    ax_corr = fig.add_subplot(gs_top[2, 1])
    ax_corr.scatter(df_surv["risk_score"], df_surv["true_imminence"], color="tab:blue", alpha=0.4, s=45) 
    sns.regplot(
        data=df_surv, x="risk_score", y="true_imminence",
        scatter=False, line_kws={'color': 'tab:red', 'linewidth': 2.5}, ax=ax_corr
    )
    corr_val = df_surv["risk_score"].corr(df_surv["true_imminence"], method="spearman")
    ax_corr.set_title(f"Predicted risk vs event imminence ($r_s$: {corr_val:.2f})")
    ax_corr.set_xlabel("Model predicted risk")
    ax_corr.set_ylabel("Event imminence (exponential decay)")
    ax_corr.grid(True, alpha=0.3)

    # Align last row subplot
    gs_bottom_container = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer_vertical[1], wspace=0.22)
    gs_bottom_left = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs_bottom_container[0], width_ratios=[0.35, 0.70, -0.05])
    gs_bottom_right = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs_bottom_container[1], width_ratios=[0.35, 0.70, -0.05])
    ax_or_col1 = fig.add_subplot(gs_bottom_left[0, 1])
    ax_or_col2 = fig.add_subplot(gs_bottom_right[0, 1])

    if df_drivers is not None and not df_drivers.empty:
        plot_df = df_drivers.head(20).copy()

        ci_lower, ci_upper, or_corrected = [], [], []
        for _, row in plot_df.iterrows():
            a = row["a"]
            b = row["n_cluster"] - a
            c = row["c"]
            d = row["n_bg"] - c

            a_corr, b_corr, c_corr, d_corr = a + 0.5, b + 0.5, c + 0.5, d + 0.5
            log_or = np.log((a_corr * d_corr) / (b_corr * c_corr))
            se_log_or = np.sqrt(1.0 / a_corr + 1.0 / b_corr + 1.0 / c_corr + 1.0 / d_corr)

            or_corrected.append(np.exp(log_or))
            ci_lower.append(np.exp(log_or - 1.96 * se_log_or))
            ci_upper.append(np.exp(log_or + 1.96 * se_log_or))

        plot_df["Odds_Ratio"] = or_corrected
        plot_df["ci_lower"] = ci_lower
        plot_df["ci_upper"] = ci_upper

        # Exact abbreviation function
        plot_df["display_feature"] = plot_df["Feature"].apply(abbreviate_feature_name)

        df_part1 = plot_df.iloc[0:min(10, len(plot_df))]
        df_part2 = plot_df.iloc[10:20] if len(plot_df) > 10 else pd.DataFrame()

        render_odds_ratio_subplot(ax_or_col1, df_part1, title_suffix="(top 1–10)")
        render_odds_ratio_subplot(ax_or_col2, df_part2, title_suffix="(top 11–20)")
    else:
        ax_or_col1.set_visible(False)
        ax_or_col2.set_visible(False)

    out_path = OUTPUT_DIR / f"combined_stratification_{task_key}.png"
    plt.savefig(out_path, dpi=160, bbox_inches='tight')
    plt.close()
    print(f"Saved consolidated stratification summary to: {out_path.name}")


def plot_scatter_unsupervised(ax, reduced, hue_labels, style_labels, cluster_color_map, title_base):
    neg_mask = style_labels == 0
    pos_mask = style_labels == 1

    for cid in np.unique(hue_labels):
        cid_mask = hue_labels == cid
        c_color = cluster_color_map.get(cid, "lightgrey") if cid in cluster_color_map else "darkgrey"
        
        subset_neg = cid_mask & neg_mask
        if subset_neg.any():
            ax.scatter(
                reduced[subset_neg, 0], reduced[subset_neg, 1],
                color=c_color, marker="o", s=80,
                alpha=0.4, edgecolor="white", linewidth=0.5 
            )
            
        subset_pos = cid_mask & pos_mask
        if subset_pos.any():
            ax.scatter(
                reduced[subset_pos, 0], reduced[subset_pos, 1],
                color=c_color, marker="o", s=110,
                alpha=0.75, edgecolor="#1a1a1a", linewidth=2.2 
            )

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label="Event absent (horizon)",
               markerfacecolor='gray', markersize=10, alpha=0.4,
               markeredgecolor='white', markeredgewidth=0.5),
        Line2D([0], [0], marker='o', color='w', label="Event occurred (horizon)",
               markerfacecolor='gray', markersize=12, alpha=0.75,
               markeredgecolor='#1a1a1a', markeredgewidth=2.2)
    ]

    ax.legend(handles=legend_elements, loc='upper right')  
    ax.set_title(title_base)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")


def plot_test_calibration_curve(y_test_true, raw_probs, cal_probs, task_key, output_dir, n_bins=10):
    valid_m = y_test_true >= 0
    y_test_true, raw_probs, cal_probs = y_test_true[valid_m], raw_probs[valid_m], cal_probs[valid_m]
    fig, ax = plt.subplots(figsize=(8, 8))
    
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect Calibration")
    
    prob_true_raw, prob_pred_raw = calibration_curve(y_test_true, raw_probs, n_bins=n_bins)
    ax.plot(prob_pred_raw, prob_true_raw, marker="o", linewidth=2, color="tab:red", label="Uncalibrated (Raw)")
    
    prob_true_cal, prob_pred_cal = calibration_curve(y_test_true, cal_probs, n_bins=n_bins)
    ax.plot(prob_pred_cal, prob_true_cal, marker="s", linewidth=2, color="tab:blue", label="Calibrated (Platt)")
    
    ax.set_title(f"Calibration Diagram: {task_key.replace('_', ' ').title()}", fontsize=18, fontweight="bold", pad=15)
    ax.set_xlabel("Mean Predicted Probability", fontsize=15)
    ax.set_ylabel("Fraction of Positives (Observed)", fontsize=15)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper left", fontsize=13, frameon=True)
    
    plt.tight_layout()
    out_path = output_dir / f"calibration_curve_{task_key}.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved calibration curve to {out_path.name}")


def _extract_patient_feature_sets(base_test_ds, vocab) -> dict[str, set[str]]:
    """Extracts unique formatted clinical EAV feature strings per patient once."""
    id2token = {v: k for k, v in vocab.items()}
    patient_features: dict[str, set[str]] = {}
    for sample in base_test_ds:
        pid = sample["patientid"]
        ent_ids = sample.get("entity_id", [])
        attr_ids = sample.get("attribute_id", [])
        val_ids = sample.get("value_id", [])
        f_set: set[str] = set()
        for e_id, a_id, v_id in zip(ent_ids, attr_ids, val_ids):
            if a_id == 0:
                continue
            ent_name = id2token.get(e_id, f"Ent_{e_id}")
            attr_name = id2token.get(a_id, f"Attr_{a_id}")
            val_name = id2token.get(v_id, f"Val_{v_id}")
            full_feature_name = format_eav_feature_name(ent_name, attr_name, entity_id=int(e_id))
            f_set.add(f"{full_feature_name} : {val_name}")
        patient_features[pid] = f_set
    return patient_features


def compute_cluster_enrichment_profiles(
    base_test_ds, cluster_labels, true_labels, vocab, top_k_features=100, patient_features=None,
):
    print("\n>>> Computing cluster enrichment profiles...")
    df_patients = pd.DataFrame({
        "patientid": base_test_ds["patientid"],
        "cluster": cluster_labels,
        "outcome": true_labels
    })
    
    total_patients = len(df_patients)
    valid_clusters_df = df_patients[df_patients["cluster"].isin([0, 1]) & (df_patients["outcome"] >= 0)]
    c_counts = valid_clusters_df["cluster"].value_counts()
    
    global_stats = {"has_comparison": False}
    if 0 in c_counts.index and 1 in c_counts.index:
        n_c0 = (valid_clusters_df["cluster"] == 0).sum()
        n_c1 = (valid_clusters_df["cluster"] == 1).sum()
        events_c0 = valid_clusters_df[valid_clusters_df["cluster"] == 0]["outcome"].sum()
        events_c1 = valid_clusters_df[valid_clusters_df["cluster"] == 1]["outcome"].sum()
        
        contingency_table = [[events_c0, max(0, n_c0 - events_c0)], [events_c1, max(0, n_c1 - events_c1)]]
        _, p_two_tailed = fisher_exact(contingency_table, alternative='two-sided')
        _, p_one_tailed = fisher_exact(contingency_table, alternative='greater')
        
        global_stats.update({
            "has_comparison": True,
            "n_c0": n_c0, "n_c1": n_c1,
            "rate_c0": (events_c0 / n_c0) * 100 if n_c0 > 0 else 0, 
            "rate_c1": (events_c1 / n_c1) * 100 if n_c1 > 0 else 0,
            "p_diff": p_two_tailed, "p_greater": p_one_tailed
        })

    if patient_features is None:
        patient_features = _extract_patient_feature_sets(base_test_ds, vocab)
        
    unique_clusters = sorted([c for c in df_patients["cluster"].unique() if c != -1])
    cluster_summaries = {}
    
    for cluster_id in unique_clusters:
        cluster_mask = df_patients["cluster"] == cluster_id
        cluster_pids = df_patients[cluster_mask]["patientid"].tolist()
        n_cluster = len(cluster_pids)
        if n_cluster == 0: continue
        
        valid_outcome_mask = cluster_mask & (df_patients["outcome"] >= 0)
        n_cluster_valid = valid_outcome_mask.sum()
        cluster_events = df_patients[valid_outcome_mask]["outcome"].sum()
        event_rate = (cluster_events / n_cluster_valid) * 100 if n_cluster_valid > 0 else 0
        
        bg_pids = df_patients[~cluster_mask]["patientid"].tolist()
        n_bg = len(bg_pids)
        
        cluster_feat_counts = pd.Series([f for pid in cluster_pids for f in patient_features.get(pid, [])]).value_counts()
        bg_feat_counts = pd.Series([f for pid in bg_pids for f in patient_features.get(pid, [])]).value_counts()
        
        enrichment_results = []
        candidates = cluster_feat_counts[cluster_feat_counts > (n_cluster * 0.01)].index
        
        for feat in candidates:
            a = cluster_feat_counts.get(feat, 0)
            c = bg_feat_counts.get(feat, 0)
            odds, p_val = fisher_exact([[a, n_cluster - a], [c, n_bg - c]], alternative='greater')
            
            total_with_feat = a + c
            cohort_percentage = (total_with_feat / total_patients) * 100
            
            enrichment_results.append({
                "Feature": feat,
                "Cluster_%": (a / n_cluster) * 100,
                "Cohort_%": cohort_percentage, 
                "Odds_Ratio": odds,
                "P_Value": p_val,
                "a": a,
                "c": c,
                "n_cluster": n_cluster,
                "n_bg": n_bg
            })
            
        df_enrich = pd.DataFrame(enrichment_results)
        if not df_enrich.empty:
            df_enrich = df_enrich.sort_values(by=["P_Value", "Odds_Ratio"], ascending=[True, False])
            top_features = df_enrich.head(top_k_features)
        else:
            top_features = pd.DataFrame()
            
        cluster_summaries[cluster_id] = {
            "n_patients": n_cluster,
            "event_rate": event_rate,
            "top_drivers": top_features
        }
        
    return cluster_summaries, global_stats


def export_exact_cluster_comparison_csv(
    base_test_ds, cluster_labels, vocab, global_stats, task_key,
    space_type="fine-tuned", horizons_to_check=None, patient_features=None,
):
    out_csv = OUTPUT_DIR / f"cluster_comparison_{space_type}_{task_key}.csv"
    
    df_patients = pd.DataFrame({
        "patientid": base_test_ds["patientid"],
        "cluster": cluster_labels
    })
    
    total_patients = len(df_patients)
    c0_pids = set(df_patients[df_patients["cluster"] == 0]["patientid"])
    c1_pids = set(df_patients[df_patients["cluster"] == 1]["patientid"])
    
    n_c0 = len(c0_pids)
    n_c1 = len(c1_pids)
    
    if patient_features is None:
        patient_features = _extract_patient_feature_sets(base_test_ds, vocab)
    all_unique_features = {f for f_set in patient_features.values() for f in f_set}

    rows = []
    formatted_task_title = task_key.replace('_', ' ').capitalize()
    horizons_to_check = list(horizons_to_check) if horizons_to_check else [30, 60, 90]
    scalar_cols = ["patientid", *[c for c in ("fup", *[f"label_{task_key}_{h:04d}d" for h in horizons_to_check]) if c in base_test_ds.column_names]]
    df_test_full = base_test_ds.select_columns(scalar_cols).to_pandas()
    if "fup" in df_test_full.columns:
        df_test_base = df_test_full[df_test_full["fup"] == BASE_FUP_FOR_PREDICTION].drop_duplicates("patientid")
    else:
        df_test_base = df_test_full.drop_duplicates("patientid")

    for idx, h in enumerate(horizons_to_check):
        lbl_col = f"label_{task_key}_{h:04d}d"
        
        if lbl_col in df_test_base.columns:
            df_valid_c0 = df_test_base[(df_test_base["patientid"].isin(c0_pids)) & (df_test_base[lbl_col] >= 0)]
            df_valid_c1 = df_test_base[(df_test_base["patientid"].isin(c1_pids)) & (df_test_base[lbl_col] >= 0)]
            
            n_c0_valid = len(df_valid_c0)
            n_c1_valid = len(df_valid_c1)
            
            c0_events = int(df_valid_c0[lbl_col].sum()) if n_c0_valid > 0 else 0
            c1_events = int(df_valid_c1[lbl_col].sum()) if n_c1_valid > 0 else 0
            total_events = c0_events + c1_events
            
            pct_c0 = (c0_events / n_c0_valid) * 100.0 if n_c0_valid > 0 else 0.0
            pct_c1 = (c1_events / n_c1_valid) * 100.0 if n_c1_valid > 0 else 0.0
            pct_cohort = (total_events / (n_c0_valid + n_c1_valid)) * 100.0 if (n_c0_valid + n_c1_valid) > 0 else 0.0
            
            table = [[c0_events, max(0, n_c0_valid - c0_events)], [c1_events, max(0, n_c1_valid - c1_events)]]
            _, p_val = fisher_exact(table, alternative='two-sided')
            p_val_str = f"{p_val:.2e}" if p_val < 1e-2 else f"{p_val:.4f}"
            
            rows.append({
                "Feature": f"{formatted_task_title} within {h} days",
                "Value": "Yes",
                "Cohort %": f"{pct_cohort:.2f}%",
                f"Cluster 0 % (N={n_c0})": f"{pct_c0:.2f}%",
                f"Cluster 1 % (N={n_c1})": f"{pct_c1:.2f}%",
                "Fisher Exact P-value": p_val_str,
                "_raw_p": -3.0 + idx
            })

    counts_c0 = Counter(f for pid in c0_pids for f in patient_features.get(pid, ()))
    counts_c1 = Counter(f for pid in c1_pids for f in patient_features.get(pid, ()))

    for feat_str in all_unique_features:
        count_c0 = counts_c0.get(feat_str, 0)
        count_c1 = counts_c1.get(feat_str, 0)
        count_total = count_c0 + count_c1
        
        pct_c0 = (count_c0 / n_c0) * 100 if n_c0 > 0 else 0.0
        pct_c1 = (count_c1 / n_c1) * 100 if n_c1 > 0 else 0.0
        pct_cohort = (count_total / total_patients) * 100 if total_patients > 0 else 0.0
        
        if pct_cohort < 1.0 and pct_c0 < 1.0 and pct_c1 < 1.0:
            continue
            
        table = [[count_c0, max(0, n_c0 - count_c0)], [count_c1, max(0, n_c1 - count_c1)]]
        _, p_val = fisher_exact(table, alternative='two-sided')
        
        if " : " in feat_str:
            feat_name, val_name = feat_str.split(" : ", 1)
        else:
            feat_name, val_name = feat_str, ""

        rows.append({
            "Feature": feat_name,
            "Value": val_name,
            "Cohort %": f"{pct_cohort:.2f}%",
            f"Cluster 0 % (N={n_c0})": f"{pct_c0:.2f}%",
            f"Cluster 1 % (N={n_c1})": f"{pct_c1:.2f}%",
            "Fisher Exact P-value": f"{p_val:.2e}",
            "_raw_p": p_val
        })

    df_merged = pd.DataFrame(rows).sort_values("_raw_p").drop(columns=["_raw_p"])
    df_merged.to_csv(out_csv, index=False)
    print(f"Saved exact side-by-side cluster comparison CSV table to: {out_csv.name}")


def save_cluster_profiles_report(cluster_profiles, global_stats, task_key, space_type="fine-tuned"):
    out_file = OUTPUT_DIR / f"cluster_profiles_{space_type}_{task_key}.txt"
    with open(out_file, "w") as f:
        f.write(f"==============================================================\n")
        f.write(f"Cluster profile enrichment: {task_key.upper()} ({space_type.upper()} space)\n")
        f.write(f"==============================================================\n")
        
        if global_stats["has_comparison"]:
            f.write(f"### CROSS-CLUSTER ANALYTICAL HYPOTHESIS TESTING ###\n")
            f.write(f"  * Cluster 0 Baseline Sample Size : {global_stats['n_c0']} patients\n")
            f.write(f"  * Cluster 1 Baseline Sample Size : {global_stats['n_c1']} patients\n")
            f.write(f"  * Cluster 0 Infection / Event Rate: {global_stats['rate_c0']:.2f}%\n")
            f.write(f"  * Cluster 1 Infection / Event Rate: {global_stats['rate_c1']:.2f}%\n")
            f.write(f"  * [Two-Tailed Test] Rates are different: P-value = {global_stats['p_diff']:.4e}\n")
            f.write(f"  * [One-Tailed Test] Cluster 0 > Cluster 1: P-value = {global_stats['p_greater']:.4e}\n")
            f.write(f"  * Statistical Method: Fisher's Exact Test\n")
            f.write(f"==============================================================\n\n")
            
        for cid, profile in cluster_profiles.items():
            f.write(f"### Cluster {cid} ###\n")
            f.write(f"Size: {profile['n_patients']} patients\n")
            f.write(f"Infection/event rate: {profile['event_rate']:.2f}%\n")
            f.write(f"Distinguishing clinical characteristics:\n")
            f.write(f"{'-' * 100}\n")
            
            df_drivers = profile["top_drivers"]
            if df_drivers.empty:
                f.write("  No highly distinctive features met significance limits.\n")
            else:
                f.write(f"  {'Clinical EAV combination (feature : value)':<60} | {'Cluster %':<10} | {'Cohort %':<10} | {'P-value':<8}\n")
                f.write(f"  {'-' * 100}\n")
                for _, row in df_drivers.iterrows():
                    f.write(f"  {row['Feature']:<60} | {row['Cluster_%']:>8.1f}% | {row['Cohort_%']:>8.1f}% | {row['P_Value']:.2e}\n")
            f.write("\n" + "=" * 100 + "\n\n")
            
    print(f"Saved complete clinical cluster details report to: {out_file.name}")


def _partition_clusters_into_binary_risk_groups(
    cluster_labels: np.ndarray,
    risk_scores: np.ndarray,
    true_labels: np.ndarray | None = None,
    space_label: str = "",
) -> np.ndarray:
    """Partition discovered non-noise HDBSCAN clusters into at most 2 maximally risk-separated groups
    using ONLY model-predicted `risk_scores` (zero test-label leakage; `true_labels` is used strictly
    for post-hoc summary logging):
    - K == 0 or 1: keep as single non-noise cluster (Cluster 0).
    - K == 2: order by mean predicted risk so Cluster 0 is High-risk and Cluster 1 is Low-risk.
    - K >= 3: choose the bipartition maximizing Otsu / Fisher-Jenks between-group predicted-risk variance
      `N_A * N_B / (N_A + N_B) * (r_A - r_B)^2`, assigning higher predicted risk to Cluster 0.
    """
    unique_cids = sorted([int(c) for c in np.unique(cluster_labels) if c != -1])
    k = len(unique_cids)
    if k == 0:
        return cluster_labels.copy()
    if k == 1:
        out = np.copy(cluster_labels)
        out[cluster_labels == unique_cids[0]] = 0
        return out

    def _group_stats(cids: tuple[int, ...] | list[int]) -> tuple[int, float, float]:
        mask = np.isin(cluster_labels, cids)
        n_pts = int(np.sum(mask))
        if n_pts == 0:
            return 0, -1.0, -1.0
        mean_risk = float(np.mean(risk_scores[mask]))
        if true_labels is not None:
            valid_y = true_labels[mask & (true_labels >= 0)]
            obs_rate = float(np.mean(valid_y)) if len(valid_y) > 0 else -1.0
        else:
            obs_rate = -1.0
        return n_pts, mean_risk, obs_rate

    if k == 2:
        c0, c1 = unique_cids[0], unique_cids[1]
        _, r0, _ = _group_stats([c0])
        _, r1, _ = _group_stats([c1])
        high_group, low_group = ([c0], [c1]) if r0 >= r1 else ([c1], [c0])
    else:
        # By Fisher's exact 1D variance-partitioning theorem, optimal between-group variance
        # on scalar mean predicted risk is guaranteed among contiguous cuts of risk-sorted clusters;
        # for K <= 12 we still evaluate all non-empty bipartitions purely on `risk_sep`.
        candidate_partitions: list[tuple[list[int], list[int]]] = []
        if k <= 12:
            first_cid = unique_cids[0]
            rest_cids = unique_cids[1:]
            for r in range(len(rest_cids) + 1):
                for comb in combinations(rest_cids, r):
                    group_a = [first_cid, *comb]
                    if len(group_a) == k:
                        continue
                    set_a = set(group_a)
                    group_b = [c for c in unique_cids if c not in set_a]
                    candidate_partitions.append((group_a, group_b))
        else:
            sorted_cids = sorted(unique_cids, key=lambda c: _group_stats([c])[1], reverse=True)
            for split_idx in range(1, k):
                candidate_partitions.append((sorted_cids[:split_idx], sorted_cids[split_idx:]))

        best_score = -1.0
        best_pair: tuple[list[int], list[int]] = ([unique_cids[0]], unique_cids[1:])
        for group_a, group_b in candidate_partitions:
            na, ra, _ = _group_stats(group_a)
            nb, rb, _ = _group_stats(group_b)
            if na == 0 or nb == 0:
                continue
            weight = (na * nb) / float(na + nb)
            risk_sep = weight * ((ra - rb) ** 2)
            if risk_sep > best_score:
                best_score = risk_sep
                best_pair = (group_a, group_b)

        group_a, group_b = best_pair
        na, ra, ya = _group_stats(group_a)
        nb, rb, yb = _group_stats(group_b)
        if ra >= rb:
            high_group, low_group = group_a, group_b
            nh, rh, yh, nl, rl, yl = na, ra, ya, nb, rb, yb
        else:
            high_group, low_group = group_b, group_a
            nh, rh, yh, nl, rl, yl = nb, rb, yb, na, ra, ya

        prefix = f"[{space_label}] " if space_label else ""
        print(
            f"  {prefix}Binary risk partition from {k} raw clusters: "
            f"High-risk Cluster 0 <- {high_group} (n={nh}, mean_risk={rh:.3f}, event_rate={yh*100:.1f}%) vs "
            f"Low-risk Cluster 1 <- {low_group} (n={nl}, mean_risk={rl:.3f}, event_rate={yl*100:.1f}%)"
        )

    out = np.copy(cluster_labels)
    out[np.isin(cluster_labels, high_group)] = 0
    out[np.isin(cluster_labels, low_group)] = 1
    return out


def _load_longitudinal_test_labels(data_dir_split: Path, all_fups: list[int], label_cols: list[str]) -> pd.DataFrame:
    """Fast parallel loader for only the scalar (patientid, fup, label_*) columns across all FUPs.
    Supports both HuggingFace Arrow DatasetDict folders (load_from_disk) and legacy test.parquet files.
    """
    cols_to_read = ["patientid", "fup", *label_cols]

    def _read_fup(fup: int) -> pd.DataFrame | None:
        folder = data_dir_split / f"fup_{fup:04d}"
        if not folder.exists():
            folder = data_dir_split / f"fup_{fup:04d}d"
        if not folder.exists():
            return None

        parquet_file = folder / "test.parquet"
        if parquet_file.exists():
            df = pd.read_parquet(parquet_file, columns=cols_to_read)
            df["fup"] = fup
            return df

        try:
            ds_dict = load_from_disk(str(folder))
            if "test" not in ds_dict:
                return None
            test_ds = ds_dict["test"]
            avail_cols = [c for c in cols_to_read if c in test_ds.column_names]
            df = test_ds.select_columns(avail_cols).to_pandas()
            df["fup"] = fup
            return df
        except Exception:
            return None

    with ThreadPoolExecutor(max_workers=8) as ex:
        frames = [df for df in ex.map(_read_fup, all_fups) if df is not None]
    if not frames:
        return pd.DataFrame(columns=cols_to_read)
    return pd.concat(frames, ignore_index=True)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)
    use_num_emb = (MODEL_VAL_TAG == "raw")
    config.setdefault("model", {}).setdefault("embedding_layer_config", {})["use_numerical_embedding"] = use_num_emb
    config.setdefault("data_collator", {})["use_numerical_embedding"] = use_num_emb

    data_dir_split = DATA_DIR / DATA_SPLIT_TYPE
    all_fups = scan_all_fups(data_dir_split)
    if not all_fups:
        print(f"[Error] No fup_XXXX folders found in {data_dir_split}")
        return

    resolved_tasks = []
    union_label_keys: list[str] = []

    for task_key in TASKS:
        run_meta = None
        try:
            base_dir_for_ckpt, _ = resolve_base_dir_for_ckpt(
                TRANSFORMER_BASE_DIR, DATA_SPLIT_TYPE, RUN_ID, MODEL_VAL_TAG
            )

            req_horizon = (
                int(PREDICTION_HORIZON)
                if PREDICTION_HORIZON is not None
                else (360 if task_key in ("death", "graft_loss") else 90)
            )
            try:
                if MULTITASK_ONLY:
                    checkpoint_path_ft, run_meta = find_best_checkpoint_multitask(
                        base_dir_for_ckpt, task_key, req_horizon, scope=MT_SCOPE
                    )
                else:
                    try:
                        checkpoint_path_ft = find_best_checkpoint(base_dir_for_ckpt, task_key, req_horizon)
                        run_meta = None
                    except FileNotFoundError:
                        checkpoint_path_ft, run_meta = find_best_checkpoint_multitask(
                            base_dir_for_ckpt, task_key, req_horizon, scope=MT_SCOPE
                        )
            except FileNotFoundError:
                # Fallback when task uses long-term horizons (e.g. 360d) instead of default 90d
                checkpoint_path_ft, run_meta = find_best_checkpoint_multitask(
                    base_dir_for_ckpt, task_key, None, scope=MT_SCOPE
                )

            ckpt_horizons = (
                list(run_meta.get("task_horizons", {}).get(task_key, run_meta["horizons"]))
                if run_meta
                else extract_horizons_from_path(checkpoint_path_ft)
            )
            eff_horizon = req_horizon if req_horizon in ckpt_horizons else int(ckpt_horizons[0])
            if run_meta:
                _label_key = f"label_{task_key}_{eff_horizon:04d}d"
                if _label_key not in run_meta.get("label_keys", []):
                    raise FileNotFoundError(f"{_label_key} is not a head of multi-task run '{run_meta['task_key']}'")
                target_idx = run_meta["label_keys"].index(_label_key)
                all_required_labels = list(run_meta["label_keys"])
            else:
                target_idx = ckpt_horizons.index(eff_horizon)
                all_required_labels = [f"label_{task_key}_{h:04d}d" for h in ckpt_horizons]

            checkpoint_path_pt = resolve_best_checkpoint(base_dir_for_ckpt / "pretraining")
            if checkpoint_path_pt is None:
                raise FileNotFoundError(f"No pretraining checkpoint found under {base_dir_for_ckpt / 'pretraining'}")

            surv_task = to_survival_task_key(task_key)
            surv_time_col, surv_status_col, _ = resolve_survival_label_keys(surv_task)

            for lk in [
                *all_required_labels,
                *[f"label_{task_key}_{h:04d}d" for h in ckpt_horizons],
                surv_time_col,
                surv_status_col,
            ]:
                if lk not in union_label_keys:
                    union_label_keys.append(lk)

            resolved_tasks.append({
                "task_key": task_key,
                "surv_task": surv_task,
                "surv_time_col": surv_time_col,
                "surv_status_col": surv_status_col,
                "checkpoint_path_ft": Path(checkpoint_path_ft),
                "checkpoint_path_pt": Path(checkpoint_path_pt),
                "run_meta": run_meta,
                "ckpt_horizons": ckpt_horizons,
                "target_idx": target_idx,
                "label_key": f"label_{task_key}_{eff_horizon:04d}d",
                "all_required_labels": all_required_labels,
            })
        except Exception as e:
            print(f"[Skip] Could not find valid models for {task_key}: {e}")
            continue

    if not resolved_tasks:
        print("[Error] No valid tasks could be resolved for stratification.")
        sys.exit(1)

    print(f"\nLoading patient sequences at FUP {BASE_FUP_FOR_PREDICTION} once across {len(resolved_tasks)} task(s)...")
    dataset, _, vocab = load_hf_data_and_metadata(
        data_dir=data_dir_split,
        fup_train=[BASE_FUP_FOR_PREDICTION],
        fup_valid=[BASE_FUP_FOR_PREDICTION],
        fup_test=[BASE_FUP_FOR_PREDICTION],
        label_keys=union_label_keys,
        time_mapping=config["data_collator"].get("time_mapping", None),
        eav_mappings=config["data_collator"].get("eav_mappings", None),
    )
    base_val_ds = dataset["validation"].add_column("split", ["validation"] * len(dataset["validation"]))
    base_test_ds = dataset["test"].add_column("split", ["test"] * len(dataset["test"]))
    if len(base_test_ds) == 0:
        print(f"[Error] No patients found at FUP {BASE_FUP_FOR_PREDICTION}")
        sys.exit(1)

    for item in resolved_tasks:
        if item["surv_status_col"] not in base_test_ds.column_names:
            try:
                base_test_ds = derive_category_survival_labels(base_test_ds, item["surv_task"])
            except Exception:
                pass

    needs_longitudinal_fallback = any(
        item["surv_time_col"] not in base_test_ds.column_names
        or item["surv_status_col"] not in base_test_ds.column_names
        for item in resolved_tasks
    )
    if needs_longitudinal_fallback:
        long_label_cols = sorted({item["label_key"] for item in resolved_tasks})
        fups_from_base = [f for f in all_fups if f >= BASE_FUP_FOR_PREDICTION]
        print(f"Loading fallback longitudinal outcome labels across {len(fups_from_base)} FUPs...")
        df_long_all = _load_longitudinal_test_labels(data_dir_split, fups_from_base, long_label_cols)
    else:
        df_long_all = pd.DataFrame()

    model_cfg_base = config["model"].copy()
    model_cfg_base["model_args"] = model_cfg_base.get("model_args", {}).copy()
    for key in ["dtype", "torch_dtype"]:
        if isinstance(model_cfg_base["model_args"].get(key), str):
            model_cfg_base["model_args"][key] = getattr(torch, model_cfg_base["model_args"][key])
    target_dtype = model_cfg_base["model_args"].get(
        "torch_dtype", model_cfg_base["model_args"].get("dtype", torch.float32)
    )
    enforce_monotonicity = config.get("finetuner", {}).get("enforce_monotonicity", False)

    cached_pt_ckpt: Path | None = None
    cached_pt_cluster: tuple[np.ndarray, np.ndarray] | None = None
    cached_ft_ckpt: Path | None = None
    cached_ft_outputs: tuple[dict, dict | None, np.ndarray, np.ndarray] | None = None
    shared_patient_features = _extract_patient_feature_sets(base_test_ds, vocab)

    for item in resolved_tasks:
        task_key = item["task_key"]
        checkpoint_path_ft = item["checkpoint_path_ft"]
        checkpoint_path_pt = item["checkpoint_path_pt"]
        run_meta = item["run_meta"]
        ckpt_horizons = item["ckpt_horizons"]
        target_idx = item["target_idx"]
        label_key = item["label_key"]
        all_required_labels = item["all_required_labels"]

        print(f"\nStratifying task: {task_key}")
        print(f">>> Checkpoint: {checkpoint_path_ft}"
              + (f"  [multi-task run '{run_meta['task_key']}', head task '{task_key}']" if run_meta else ""))

        collator = PatientDataCollatorForClassification(
            **config["data_collator"],
            label_keys=all_required_labels,
            max_position_embeddings=model_cfg_base.get("config_args", {}).get(
                "max_position_embeddings", model_cfg_base["model_args"].get("max_position_embeddings", 1024)
            ),
        )
        loader_test = DataLoader(base_test_ds, batch_size=32, collate_fn=collator.torch_call)
        loader_val = DataLoader(base_val_ds, batch_size=32, collate_fn=collator.torch_call) if len(base_val_ds) > 0 else None

        if cached_pt_cluster is None or cached_pt_ckpt != checkpoint_path_pt:
            print(f"\nInitializing pre-trained model from {checkpoint_path_pt.name}...")
            pt_model_cfg = model_cfg_base.copy()
            pt_model_cfg["task"] = "masked"
            pt_model_cfg["pretrained_dir"] = str(checkpoint_path_pt)
            pt_model_cfg["embedding_layer_config"] = pt_model_cfg["embedding_layer_config"].copy()
            pt_model_cfg["embedding_layer_config"]["vocab_size"] = len(vocab)
            pt_model_cfg["model_args"] = pt_model_cfg["model_args"].copy()
            pt_model_cfg["model_args"].pop("num_labels", None)
            pt_model_cfg["model_args"].pop("problem_type", None)

            model_pt = PatientEmbeddingModelFactory.from_pretrained(**pt_model_cfg).to(device=DEVICE, dtype=target_dtype)
            interpreter_pt = ModelInterpreter(model_pt, device=DEVICE)

            print("Extracting pre-trained embeddings...")
            res_pt = interpreter_pt.get_embeddings_and_predictions(loader_test, extract_logits=False)
            embeddings_pt = res_pt["embeddings"]

            del model_pt, interpreter_pt
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            clusterer_pt = UMAP_HDBSCAN_Clusterer(n_optuna_trials=0)
            reduced_pt, labels_pt_raw = clusterer_pt.predict(
                embeddings_pt, n_components=2, min_cluster_size=15, min_samples=5
            )
            cached_pt_ckpt = checkpoint_path_pt
            cached_pt_cluster = (reduced_pt, labels_pt_raw)
        else:
            reduced_pt, labels_pt_raw = cached_pt_cluster

        if cached_ft_outputs is None or cached_ft_ckpt != checkpoint_path_ft:
            print(f"\nInitializing fine-tuned model from {checkpoint_path_ft.name}...")
            ft_model_cfg = model_cfg_base.copy()
            ft_model_cfg["task"] = "classification"
            ft_model_cfg["pretrained_dir"] = str(checkpoint_path_ft)
            ft_model_cfg["embedding_layer_config"] = ft_model_cfg["embedding_layer_config"].copy()
            ft_model_cfg["embedding_layer_config"]["vocab_size"] = len(vocab)
            ft_model_cfg["enforce_monotonicity"] = build_monotonic_groups(run_meta, enforce_monotonicity)
            ft_model_cfg["model_args"] = ft_model_cfg["model_args"].copy()
            ft_model_cfg["model_args"].pop("num_labels", None)
            ft_model_cfg["model_args"]["problem_type"] = "multi_label_classification"

            model_ft = PatientEmbeddingModelFactory.from_pretrained(**ft_model_cfg).to(device=DEVICE, dtype=target_dtype)
            interpreter_ft = ModelInterpreter(model_ft, device=DEVICE)

            print("Extracting fine-tuned embeddings and predictions...")
            res_ft = interpreter_ft.get_embeddings_and_predictions(loader_test)
            res_val = interpreter_ft.get_embeddings_and_predictions(loader_val) if loader_val is not None else None

            del model_ft, interpreter_ft
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            print("\nRunning UMAP reductions on fine-tuned embeddings...")
            clusterer_ft = UMAP_HDBSCAN_Clusterer(n_optuna_trials=0)
            reduced_ft, labels_ft_raw = clusterer_ft.predict(
                res_ft["embeddings"], n_components=2, min_cluster_size=15, min_samples=5
            )

            cached_ft_ckpt = checkpoint_path_ft
            cached_ft_outputs = (res_ft, res_val, reduced_ft, labels_ft_raw)
        else:
            res_ft, res_val, reduced_ft, labels_ft_raw = cached_ft_outputs

        embeddings_ft = res_ft["embeddings"]
        y_test_true = res_ft["labels"][:, target_idx]
        raw_probs_test = 1 / (1 + np.exp(-res_ft["logits"][:, target_idx]))
        patient_ids = base_test_ds["patientid"]

        if res_val is not None:
            y_val_true = res_val["labels"][:, target_idx]
            raw_probs_val = 1 / (1 + np.exp(-res_val["logits"][:, target_idx]))
            _, probs_ft = calibrate_array_pair(
                y_val_true=y_val_true,
                y_val_prob=raw_probs_val,
                y_test_prob=raw_probs_test,
            )
            print("Post-hoc Platt scaling calibration applied successfully.")
            plot_test_calibration_curve(
                y_test_true=y_test_true,
                raw_probs=raw_probs_test,
                cal_probs=probs_ft,
                task_key=task_key,
                output_dir=OUTPUT_DIR,
            )
        else:
            probs_ft = raw_probs_test

        # Partition discovered clusters into optimal binary High-risk (0) vs Low-risk (1) groups
        labels_pt = _partition_clusters_into_binary_risk_groups(
            labels_pt_raw, probs_ft, y_test_true, space_label="pre_trained"
        )
        labels_ft = _partition_clusters_into_binary_risk_groups(
            labels_ft_raw, probs_ft, y_test_true, space_label="fine_tuned"
        )

        # Compute cluster data alongside comparative cross-cluster risk statistics
        ft_profiles, global_stats = compute_cluster_enrichment_profiles(
            base_test_ds, labels_ft, y_test_true, vocab, patient_features=shared_patient_features,
        )
        save_cluster_profiles_report(ft_profiles, global_stats, task_key, space_type="fine_tuned")
        export_exact_cluster_comparison_csv(
            base_test_ds, labels_ft, vocab, global_stats, task_key,
            space_type="fine_tuned", horizons_to_check=ckpt_horizons, patient_features=shared_patient_features,
        )

        # Color palette (Cluster 0 = High-risk in red, Cluster 1 = Low-risk in blue)
        unique_all_clusters = sorted(
            {int(c) for c in np.concatenate([np.unique(labels_pt), np.unique(labels_ft)]) if c != -1}
        )
        fallback_palette = sns.color_palette("bright", n_colors=max(1, len(unique_all_clusters)))

        cluster_color_map = {0: "tab:red", 1: "tab:blue", -1: "lightgrey"}
        for idx, cid in enumerate(unique_all_clusters):
            if cid not in cluster_color_map:
                cluster_color_map[cid] = fallback_palette[idx % len(fallback_palette)]

        print("\nBuilding consolidated stratification performance canvas (including odds ratio subplots)...")
        top_drivers_c0 = ft_profiles[0]["top_drivers"] if 0 in ft_profiles else None

        surv_times = (
            np.asarray(base_test_ds[item["surv_time_col"]])
            if item["surv_time_col"] in base_test_ds.column_names
            else None
        )
        surv_statuses = (
            np.asarray(base_test_ds[item["surv_status_col"]])
            if item["surv_status_col"] in base_test_ds.column_names
            else None
        )

        plot_combined_stratification_grid(
            test_ds=df_long_all,
            patient_ids=patient_ids,
            probs=probs_ft,
            reduced_pt=reduced_pt,
            reduced_ft=reduced_ft,
            labels_pt=labels_pt,
            labels_ft=labels_ft,
            cluster_color_map=cluster_color_map, 
            true_labels_horizon=y_test_true,
            task_key=task_key,
            label_key=label_key,
            df_drivers=top_drivers_c0,
            surv_times=surv_times,
            surv_statuses=surv_statuses,
        )


if __name__ == "__main__":
    main()
