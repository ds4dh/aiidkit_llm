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

from pathlib import Path
from torch.utils.data import DataLoader
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from sklearn.calibration import calibration_curve
from scipy.stats import fisher_exact
from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test
from transformers.trainer_utils import get_last_checkpoint

from src.model.patient_embedder import PatientEmbeddingModelFactory, PatientDataCollatorForClassification
from src.evaluation.evaluate_models import UMAP_HDBSCAN_Clusterer, ModelInterpreter
from src.data.patient_dataset import load_hf_data_and_metadata
from scripts.analysis_interpretability import find_best_checkpoint, extract_horizons_from_path
from scripts.script_utils import scan_all_fups, get_best_optuna_run, calibrate_array_pair

import argparse
import os

parser = argparse.ArgumentParser(description="Run patient stratification and survival analysis.")
parser.add_argument("--results-dir", type=Path, default=Path("results_final"), help="Base directory for results")
parser.add_argument("--data-split-type", "--data_split_type", type=str, default="temporal_split", help="Data split strategy")
parser.add_argument("--data-dir", "--data_dir", type=Path, default=Path(os.environ.get("TEAV_DATA_DIR", "/home/shares/ds4dh/aiidkit_project/data_new/processed/v3.6/teav")), help="Path to tEAV dataset directory")
args, _ = parser.parse_known_args()

# Model selection
RESULTS_DIR = args.results_dir
TRANSFORMER_BASE_DIR = RESULTS_DIR / "transformer"
OUTPUT_DIR = RESULTS_DIR / Path("analysis/stratification")
CONFIG_PATH = Path("configs/discriminative_training.yaml") 
DATA_DIR = args.data_dir

# Configuration
FROM_OPTUNA = "optuna" in TRANSFORMER_BASE_DIR.name
DATA_SPLIT_TYPE = args.data_split_type
TASKS = [
    "infection_bacteria",
    # "infection_virus",
    # "death",
    # "graft_loss",
]
MAX_FUP = 3600
if DATA_SPLIT_TYPE == "temporal_split":
    MAX_FUP = 2400
BASE_FUP_FOR_PREDICTION = 90
PREDICTION_HORIZON = 30
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


def abbreviate_feature_name(text: str) -> str:
    """Replaces full clinical terms with concise abbreviations while preserving original casing."""
    replacements = {
        "patient": "pt.",
        "transplant": "tpx.",
        "immunology": "immu.",
        "medication": "med.",
        "hemoglobin": "hgb.",
        "previous": "prev.",
        "infection": "inf.",
        "diagnosis": "diag.",
        "event": "evt.",
        "resistance": "res.",
        "pathogen": "path.",
        "comorbidity": "comorb.",
        "pulmonary": "pulm.",
        "bacteria": "bact.",
        "clinically": "clin.",
        "clinical": "clin.",
    }
    
    def replace_match(match):
        word = match.group(0)
        lower_word = word.lower()
        replacement = replacements.get(lower_word, word)
        
        # Preserve Titlecase / UPPERCASE formatting
        if word.istitle():
            return replacement.capitalize()
        elif word.isupper():
            return replacement.upper()
        return replacement

    pattern = re.compile(r'\b(' + '|'.join(replacements.keys()) + r')\b', re.IGNORECASE)
    return pattern.sub(replace_match, text)


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
    cluster_color_map, true_labels_horizon, task_key, label_key, df_drivers=None
):
    df_long = test_ds.to_pandas()
    target_pids = set(patient_ids)
    df_long = df_long[(df_long["patientid"].isin(target_pids)) & (df_long["fup"] >= BASE_FUP_FOR_PREDICTION)]
    
    survival_data = []
    for i, pid in enumerate(patient_ids):
        p_data = df_long[df_long["patientid"] == pid].sort_values("fup")
        future_events = p_data[p_data[label_key] == 1]
        
        if not future_events.empty:
            first_event_fup = future_events["fup"].iloc[0]
            duration = first_event_fup - BASE_FUP_FOR_PREDICTION
            event = 1
            true_imminence = np.exp(-duration / 365.0) 
        else:
            last_fup = p_data["fup"].max() if not p_data.empty else BASE_FUP_FOR_PREDICTION
            duration = last_fup - BASE_FUP_FOR_PREDICTION
            event = 0
            true_imminence = 0.0
            
        survival_data.append({
            "pid": pid,
            "duration": max(0, duration),
            "event": event,
            "risk_score": probs[i],
            "true_imminence": true_imminence,
            "cluster_ft": labels_ft[i]  
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


def compute_cluster_enrichment_profiles(base_test_ds, cluster_labels, true_labels, vocab, top_k_features=100):
    print("\n>>> Computing cluster enrichment profiles...")
    id2token = {v: k for k, v in vocab.items()}
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

    patient_features = {}
    for idx, sample in enumerate(base_test_ds):
        pid = sample["patientid"]
        ent_ids  = sample.get("entity_id", [])
        attr_ids = sample.get("attribute_id", [])
        val_ids  = sample.get("value_id", [])
        
        f_set = set()
        for e_id, a_id, v_id in zip(ent_ids, attr_ids, val_ids):
            if a_id == 0: continue
            
            ent_name  = id2token.get(e_id, f"Ent_{e_id}")
            attr_name = id2token.get(a_id, f"Attr_{a_id}")
            val_name  = id2token.get(v_id, f"Val_{v_id}")
            
            if "infection" in ent_name.lower():
                if ent_name.strip().lower() == "infection":
                    ent_name = "Previous infection"
                elif not ent_name.lower().startswith("previous"):
                    ent_name = f"Previous {ent_name.lower()}"
            
            full_feature_name = f"{ent_name} - {attr_name}"
            f_set.add(f"{full_feature_name} : {val_name}")
            
        patient_features[pid] = f_set
        
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


def export_exact_cluster_comparison_csv(base_test_ds, cluster_labels, vocab, global_stats, task_key, space_type="fine-tuned"):
    out_csv = OUTPUT_DIR / f"cluster_comparison_{space_type}_{task_key}.csv"
    id2token = {v: k for k, v in vocab.items()}
    
    df_patients = pd.DataFrame({
        "patientid": base_test_ds["patientid"],
        "cluster": cluster_labels
    })
    
    total_patients = len(df_patients)
    c0_pids = set(df_patients[df_patients["cluster"] == 0]["patientid"])
    c1_pids = set(df_patients[df_patients["cluster"] == 1]["patientid"])
    
    n_c0 = len(c0_pids)
    n_c1 = len(c1_pids)
    
    patient_features = {}
    all_unique_features = set()
    
    for sample in base_test_ds:
        pid = sample["patientid"]
        ent_ids  = sample.get("entity_id", [])
        attr_ids = sample.get("attribute_id", [])
        val_ids  = sample.get("value_id", [])
        
        f_set = set()
        for e_id, a_id, v_id in zip(ent_ids, attr_ids, val_ids):
            if a_id == 0: continue
            
            ent_name  = id2token.get(e_id, f"Ent_{e_id}")
            attr_name = id2token.get(a_id, f"Attr_{a_id}")
            val_name  = id2token.get(v_id, f"Val_{v_id}")
            
            if "infection" in ent_name.lower():
                if ent_name.strip().lower() == "infection":
                    ent_name = "Previous infection"
                elif not ent_name.lower().startswith("previous"):
                    ent_name = f"Previous {ent_name.lower()}"
            
            full_feature_name = f"{ent_name} - {attr_name} : {val_name}"
            f_set.add(full_feature_name)
            all_unique_features.add(full_feature_name)
            
        patient_features[pid] = f_set

    rows = []
    formatted_task_title = task_key.replace('_', ' ').capitalize()
    horizons_to_check = [30, 60, 90]
    
    df_test_full = base_test_ds.to_pandas()
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

    for feat_str in all_unique_features:
        count_c0 = sum(1 for pid in c0_pids if feat_str in patient_features.get(pid, set()))
        count_c1 = sum(1 for pid in c1_pids if feat_str in patient_features.get(pid, set()))
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


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)
        
    for task_key in TASKS:
        print(f"\nStratifying task: {task_key}")
        
        try:
            if FROM_OPTUNA:
                trial_name, pt_config = get_best_optuna_run(TRANSFORMER_BASE_DIR, DATA_SPLIT_TYPE, task_key)
                base_dir_for_ckpt = TRANSFORMER_BASE_DIR / DATA_SPLIT_TYPE / task_key / trial_name / DATA_SPLIT_TYPE / pt_config
            else:
                [base_dir_for_ckpt] = (d for d in (TRANSFORMER_BASE_DIR / DATA_SPLIT_TYPE).iterdir() if d.is_dir())
            
            checkpoint_path_ft = find_best_checkpoint(base_dir_for_ckpt, task_key, PREDICTION_HORIZON)
            ckpt_horizons = extract_horizons_from_path(checkpoint_path_ft)
            target_idx = ckpt_horizons.index(PREDICTION_HORIZON)
            
            pretrain_dir = base_dir_for_ckpt / "pretraining"
            checkpoint_path_pt = get_last_checkpoint(str(pretrain_dir))
            if checkpoint_path_pt is None:
                ckpt_dirs = list(pretrain_dir.glob("checkpoint-*"))
                if ckpt_dirs:
                    checkpoint_path_pt = sorted(ckpt_dirs, key=lambda x: int(x.name.split("-")[-1]))[-1]
                else:
                    raise FileNotFoundError(f"No pretraining checkpoints found in {pretrain_dir}")
                    
        except Exception as e:
            print(f"[Skip] Could not find valid models for {task_key}: {e}")
            continue

        print("Loading patient sequences...")
        label_key = f"label_{task_key}_{PREDICTION_HORIZON:04d}d"
        all_required_labels = [f"label_{task_key}_{h:04d}d" for h in ckpt_horizons]
        
        data_dir_split = DATA_DIR / DATA_SPLIT_TYPE
        all_fups = scan_all_fups(data_dir_split)

        if not all_fups:
            print(f"[Error] No fup_XXXX folders found in {data_dir_split}")
            continue

        dataset, _, vocab = load_hf_data_and_metadata(
            data_dir=data_dir_split,
            fup_train=[BASE_FUP_FOR_PREDICTION],
            fup_valid=[BASE_FUP_FOR_PREDICTION],
            fup_test=all_fups, 
            label_keys=all_required_labels,
        )
        
        val_ds = dataset["validation"].add_column("split", ["validation"] * len(dataset["validation"]))
        test_ds = dataset["test"].add_column("split", ["test"] * len(dataset["test"]))

        model_cfg_base = config["model"].copy()
        if "model_args" not in model_cfg_base:
            model_cfg_base["model_args"] = {}
            
        for key in ["dtype", "torch_dtype"]:
            if isinstance(model_cfg_base["model_args"].get(key), str):
                model_cfg_base["model_args"][key] = getattr(torch, model_cfg_base["model_args"][key])
        
        target_dtype = model_cfg_base["model_args"].get("torch_dtype", torch.float32)
        enforce_monotonicity = config.get("finetuner", {}).get("enforce_monotonicity", False)
        
        collator = PatientDataCollatorForClassification(
            label_keys=all_required_labels,
            max_position_embeddings=config["data_collator"].get("max_position_embeddings", 512)
        )

        base_test_ds = test_ds.filter(lambda x: x["fup"] == BASE_FUP_FOR_PREDICTION)
        if len(base_test_ds) == 0:
            print(f"[Error] No patients found at FUP {BASE_FUP_FOR_PREDICTION}")
            continue
        loader_test = DataLoader(base_test_ds, batch_size=32, collate_fn=collator.torch_call)

        base_val_ds = val_ds.filter(lambda x: x["fup"] == BASE_FUP_FOR_PREDICTION)
        loader_val = DataLoader(base_val_ds, batch_size=32, collate_fn=collator.torch_call) if len(base_val_ds) > 0 else None

        print(f"\nInitializing pre-trained model from {Path(checkpoint_path_pt).name}...")
        pt_model_cfg = model_cfg_base.copy()
        pt_model_cfg["task"] = "masked"
        pt_model_cfg["pretrained_dir"] = str(checkpoint_path_pt)
        pt_model_cfg["embedding_layer_config"]["vocab_size"] = len(vocab)
        pt_model_cfg["model_args"] = pt_model_cfg["model_args"].copy()
        pt_model_cfg["model_args"].pop("num_labels", None)
        pt_model_cfg["model_args"].pop("problem_type", None)
        
        model_pt = PatientEmbeddingModelFactory.from_pretrained(**pt_model_cfg)
        model_pt = model_pt.to(device=DEVICE, dtype=target_dtype)
        interpreter_pt = ModelInterpreter(model_pt, device=DEVICE)
        
        print("Extracting pre-trained embeddings...")
        res_pt = interpreter_pt.get_embeddings_and_predictions(loader_test, extract_logits=False)
        embeddings_pt = res_pt["embeddings"]
        
        del model_pt, interpreter_pt
        gc.collect()
        torch.cuda.empty_cache()

        print(f"\nInitializing fine-tuned model from {Path(checkpoint_path_ft).name}...")
        ft_model_cfg = model_cfg_base.copy()
        ft_model_cfg["task"] = "classification"
        ft_model_cfg["pretrained_dir"] = str(checkpoint_path_ft)
        ft_model_cfg["embedding_layer_config"]["vocab_size"] = len(vocab)
        ft_model_cfg["enforce_monotonicity"] = enforce_monotonicity
        ft_model_cfg["model_args"]["num_labels"] = len(ckpt_horizons)
        ft_model_cfg["model_args"]["problem_type"] = "multi_label_classification"

        model_ft = PatientEmbeddingModelFactory.from_pretrained(**ft_model_cfg)
        model_ft = model_ft.to(device=DEVICE, dtype=target_dtype)
        interpreter_ft = ModelInterpreter(model_ft, device=DEVICE)
        
        print("Extracting fine-tuned embeddings and predictions...")
        res_ft = interpreter_ft.get_embeddings_and_predictions(loader_test)
        embeddings_ft = res_ft["embeddings"]
        
        y_test_true = res_ft["labels"][:, target_idx]
        raw_probs_test = 1 / (1 + np.exp(-res_ft["logits"][:, target_idx]))  
        patient_ids = base_test_ds["patientid"]

        if loader_val is not None:
            print("Extracting validation predictions for probability calibration...")
            res_val = interpreter_ft.get_embeddings_and_predictions(loader_val)
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

        del model_ft, interpreter_ft
        gc.collect()
        torch.cuda.empty_cache()

        print("\nRunning UMAP reductions...")
        clusterer_pt = UMAP_HDBSCAN_Clusterer(n_optuna_trials=0)
        reduced_pt, labels_pt = clusterer_pt.predict(embeddings_pt, n_components=2, min_cluster_size=15, min_samples=5)
        clusterer_ft = UMAP_HDBSCAN_Clusterer(n_optuna_trials=0)
        reduced_ft, labels_ft = clusterer_ft.predict(embeddings_ft, n_components=2, min_cluster_size=15, min_samples=5)

        # Compute cluster data alongside comparative cross-cluster risk statistics
        ft_profiles, global_stats = compute_cluster_enrichment_profiles(
            base_test_ds, labels_ft, res_ft["labels"][:, target_idx], vocab,
        )
        save_cluster_profiles_report(ft_profiles, global_stats, task_key, space_type="fine_tuned")
        export_exact_cluster_comparison_csv(base_test_ds, labels_ft, vocab, global_stats, task_key, space_type="fine_tuned")

        # Color palette
        unique_ft_clusters = sorted([c for c in np.unique(labels_ft) if c != -1])
        all_discovered_count = len(unique_ft_clusters)
        fallback_palette = sns.color_palette("bright", n_colors=max(1, all_discovered_count))
        
        cluster_color_map = {}
        for idx, cid in enumerate(unique_ft_clusters):
            if cid == 0:
                cluster_color_map[0] = "tab:red"
            elif cid == 1:
                cluster_color_map[1] = "tab:blue"
            else:
                cluster_color_map[cid] = fallback_palette[idx % len(fallback_palette)]
        cluster_color_map[-1] = "lightgrey"

        print("\nBuilding consolidated stratification performance canvas (including odds ratio subplots)...")
        top_drivers_c0 = ft_profiles[0]["top_drivers"] if 0 in ft_profiles else None

        plot_combined_stratification_grid(
            test_ds=test_ds,
            patient_ids=patient_ids,
            probs=raw_probs_test,
            reduced_pt=reduced_pt,
            reduced_ft=reduced_ft,
            labels_pt=labels_pt,
            labels_ft=labels_ft,
            cluster_color_map=cluster_color_map, 
            true_labels_horizon=res_ft["labels"][:, target_idx],
            task_key=task_key,
            label_key=label_key,
            df_drivers=top_drivers_c0
        )


if __name__ == "__main__":
    main()