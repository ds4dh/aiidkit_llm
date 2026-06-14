import gc
import yaml
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from scipy.stats import fisher_exact
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from transformers.trainer_utils import get_last_checkpoint

from src.model.patient_embedder import PatientEmbeddingModelFactory, PatientDataCollatorForClassification
from src.evaluation.evaluate_models import UMAP_HDBSCAN_Clusterer, ModelInterpreter
from src.data.patient_dataset import load_hf_data_and_metadata
from scripts.analysis_interpretability import find_best_checkpoint, extract_horizons_from_path
from scripts.script_utils import scan_all_fups, get_best_optuna_run

# Model selection
RESULTS_DIR = Path("results_final")
TRANSFORMER_BASE_DIR = RESULTS_DIR / "transformer"
OUTPUT_DIR = RESULTS_DIR / Path("analysis/stratification")
CONFIG_PATH = Path("configs/discriminative_training.yaml") 
DATA_DIR = Path("/home/shares/ds4dh/aiidkit_project/data_new/processed/v3.6/teav")

# Configuration
FROM_OPTUNA = "optuna" in TRANSFORMER_BASE_DIR.name
DATA_SPLIT_TYPE = "temporal_split"
TASKS = [
    "infection_bacteria",
    "infection_virus",
    "death",
    "graft_loss",
]
MAX_FUP = 3600                 # up to when the data is used and plotted
if DATA_SPLIT_TYPE == "temporal_split":
    MAX_FUP = 2400             # no test patients further than that
BASE_FUP_FOR_PREDICTION = 90   # post-tpx follow-up day where risk is evaluated
PREDICTION_HORIZON = 30        # model horizon to use for stratification
SAFE_NUM_PROC = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Fontsizes
FS_SCALE = 1.0
plt.rcParams.update({
    'font.size': 14 * FS_SCALE,              # general global font size
    'axes.labelsize': 16 * FS_SCALE,         # x/y axis label font size
    'xtick.labelsize': 14 * FS_SCALE,        # x tick font size
    'ytick.labelsize': 14 * FS_SCALE,        # y tick font size
    'legend.fontsize': 14 * FS_SCALE,        # legend font size (smaller for inside plot)
    'legend.title_fontsize': 14 * FS_SCALE,  # legend title font size
    'figure.titlesize': 20 * FS_SCALE,       # suptitle
})


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load the training configuration to get the proper model dtypes and args
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)
        
    for task_key in TASKS:
        print(f"\nStratifying task: {task_key}")
        
        # Locate checkpoints (pre-trained and fine-tuned)
        try:
            # Dynamically fetch the best trial and config OR load directly
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
            # import ipdb; ipdb.set_trace()
            print(f"[Skip] Could not find valid models for {task_key}: {e}")
            continue

        # Load dataset
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
        test_ds = dataset["test"]
        test_ds = test_ds.add_column("split", ["test"] * len(test_ds))

        # Setup robust config parsing
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
        loader = DataLoader(base_test_ds, batch_size=32, collate_fn=collator.torch_call)

        # Inference with pre-trained model
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
        res_pt = interpreter_pt.get_embeddings_and_predictions(loader, extract_logits=False)
        embeddings_pt = res_pt["embeddings"]
        
        del model_pt, interpreter_pt
        gc.collect()
        torch.cuda.empty_cache()

        # Inference with fine-tuned model
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
        res_ft = interpreter_ft.get_embeddings_and_predictions(loader)
        embeddings_ft = res_ft["embeddings"]
        
        probs_ft = 1 / (1 + np.exp(-res_ft["logits"][:, target_idx]))  
        true_labels_at_base = res_ft["labels"][:, target_idx]
        patient_ids = base_test_ds["patientid"]

        del model_ft, interpreter_ft
        gc.collect()
        torch.cuda.empty_cache()

        # Dimensionality reduction and clustering
        print("\nRunning UMAP + HDBSCAN on pre-trained embedding space...")
        clusterer_pt = UMAP_HDBSCAN_Clusterer(n_optuna_trials=0)
        reduced_pt, labels_pt = clusterer_pt.predict(embeddings_pt, n_components=2, min_cluster_size=15, min_samples=5)
        print("Running UMAP + HDBSCAN on fine-tuned embedding space...")
        clusterer_ft = UMAP_HDBSCAN_Clusterer(n_optuna_trials=0)
        reduced_ft, labels_ft = clusterer_ft.predict(embeddings_ft, n_components=2, min_cluster_size=15, min_samples=5)
        plot_clustering_comparison(reduced_pt, reduced_ft, labels_pt, labels_ft, true_labels_at_base, task_key)

        # Profile the fine-tuned cluster space to discover risk characteristics
        ft_profiles = compute_cluster_enrichment_profiles(
            base_test_ds, labels_ft, true_labels_at_base, vocab,
        )
        save_cluster_profiles_report(ft_profiles, task_key, space_type="fine_tuned")

        # Run survival analysis
        print("\nCalculating survival trajectories and calibration...")
        threshold = np.median(probs_ft)
        risk_map = {pid: (prob >= threshold) for pid, prob in zip(patient_ids, probs_ft)}
        
        run_survival_analysis(
            test_ds=test_ds,
            patient_ids=patient_ids,
            risk_map=risk_map,
            probs=probs_ft,
            reduced_embeddings=reduced_ft, 
            true_labels=true_labels_at_base,
            task_key=task_key,
            label_key=label_key
        )


def plot_scatter(ax, reduced, hue_labels, style_labels, title, is_outcome=False):
    # Prepare base DataFrame
    df = pd.DataFrame({
        "UMAP 1": reduced[:, 0], 
        "UMAP 2": reduced[:, 1], 
        "Hue_ID": hue_labels, # keep raw cluster IDs for noise filtering
        "Outcome": ["Positive" if y == 1 else "Negative" for y in style_labels],
    })
    
    # Separate Noise (-1) from clusters
    noise_mask = df["Hue_ID"] == -1
    df_noise = df[noise_mask].copy()
    df_clusters = df[~noise_mask].copy()

    # Plot noise first (underneath, smaller, gray, low alpha)
    if not df_noise.empty:
        sns.scatterplot(
            ax=ax, data=df_noise, x="UMAP 1", y="UMAP 2",
            color="lightgrey", s=25, alpha=0.4,
            markers={"Positive": "X", "Negative": "o"},
            style="Outcome", legend=False, linewidth=0,
        )

    # Plot valid clusters
    palette = {"Positive": "tab:red", "Negative": "tab:blue"} if is_outcome else "bright"
    df_clusters["Cluster_Group"] = "Cluster"  # single group name for legend grouping
    df_clusters = df_clusters.sort_values(by="Outcome", ascending=True)
    _ = sns.scatterplot(
        ax=ax, data=df_clusters, x="UMAP 1", y="UMAP 2", 
        hue="Outcome" if is_outcome else "Hue_ID", 
        hue_order=["Negative", "Positive"] if is_outcome else None,
        style="Outcome", style_order=["Negative", "Positive"],
        size="Outcome", sizes={"Positive": 140, "Negative": 60},
        size_order=["Negative", "Positive"],
        palette=palette, markers={"Positive": "X", "Negative": "o"},
        alpha=0.8, edgecolor="white", linewidth=0.6,
    )

    handles, labels = ax.get_legend_handles_labels()    
    new_handles = []
    new_labels = []
    seen = set()

    # For outcome plots, we only want "Positive" and "Negative" in legend
    if is_outcome:
        for h, l in zip(handles, labels):
            if l in ["Positive", "Negative"]:
                new_handles.append(h)
                new_labels.append(l)
                seen.add(l)
    
    # For unsupervised clusters, skip cluster IDs
    else:
        for h, l in zip(handles, labels):
            if l in ["Negative", "Positive"]:
                 new_handles.append(h)
                 new_labels.append(l)
                 seen.add(l)

    ax.legend(
        new_handles, new_labels, loc='best', facecolor='white',
        framealpha=0.9, title=None, frameon=True, shadow=False,
    )
    ax.set_title(title)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")


def plot_clustering_comparison(reduced_pt, reduced_ft, labels_pt, labels_ft, true_labels, task_key):
    fig, axes = plt.subplots(2, 2, figsize=(18, 16))
    task_str = task_key.replace('_', ' ').title().lower() + f", horizon {PREDICTION_HORIZON} days"
    fig.suptitle(f"Embedding space evolution: pre-trained vs fine-tuned ({task_str.lower()})", fontweight='bold', y=0.98)

    # Unsupervised clusters (color by Cluster ID, shape by true outcome)
    plot_scatter(axes[0, 0], reduced_pt, labels_pt, true_labels, "Pre-trained: discovered clusters", is_outcome=False)
    plot_scatter(axes[0, 1], reduced_ft, labels_ft, true_labels, "Fine-tuned: discovered clusters", is_outcome=False)

    # True clinical outcomes (color by true outcome, shape by true outcome)
    plot_scatter(axes[1, 0], reduced_pt, labels_pt, true_labels, "Pre-trained: true outcomes", is_outcome=True)
    plot_scatter(axes[1, 1], reduced_ft, labels_ft, true_labels, "Fine-tuned: true outcomes", is_outcome=True)

    out_path = OUTPUT_DIR / f"clustering_comparison_{task_key}.png"
    plt.tight_layout(h_pad=3.0, w_pad=3.0) 
    plt.subplots_adjust(top=0.94)
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved cluster comparison to {out_path.name}")


def run_survival_analysis(
    test_ds, patient_ids, risk_map, probs, reduced_embeddings, true_labels, task_key, label_key,
):
    """
    Produces a 3-row layout using GridSpec:
    Row 1 (spans both cols): KM survival curves
    Row 2: Calibration curve & Correlation plot (predicted vs true risk)
    Row 3: UMAP colored by predicted risk & UMAP colored by true time-to-event
    """
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
            true_risk = np.exp(-duration / 365.0)   # exp. decay: 1.0 at day 0, smooth decay for later years
        else:
            last_fup = p_data["fup"].max() if not p_data.empty else BASE_FUP_FOR_PREDICTION
            duration = last_fup - BASE_FUP_FOR_PREDICTION
            event = 0
            true_risk = 0.0
            
        survival_data.append({
            "duration": max(0, duration),
            "event": event,
            "is_high_risk": risk_map[pid],
            "risk_score": probs[i],
            "true_risk": true_risk
        })

    task_str = task_key.replace('_', ' ').title().lower() + f", horizon {PREDICTION_HORIZON} days"
    fig = plt.figure(figsize=(18, 22))
    fig.suptitle(f"Stratified survival analysis ({task_str})", fontweight='bold', y=0.98)
    
    gs = fig.add_gridspec(3, 2, height_ratios=[1.2, 1, 1])
    ax_km = fig.add_subplot(gs[0, :])
    ax_cal = fig.add_subplot(gs[1, 1])
    ax_corr = fig.add_subplot(gs[2, 1])
    ax_umap_pred = fig.add_subplot(gs[1, 0])
    ax_umap_true = fig.add_subplot(gs[2, 0])
    
    # Kaplan-Meier comparison
    df_surv = pd.DataFrame(survival_data)
    kmf = KaplanMeierFitter()
    for is_high in [True, False]:
        mask = df_surv["is_high_risk"] == is_high
        label = "High predicted risk" if is_high else "Low predicted risk"
        color = "tab:red" if is_high else "tab:blue"
        if mask.sum() > 0:
            # Shift the duration to absolute days and explicitly tell lifelines the entry time
            absolute_duration = df_surv[mask]["duration"] + BASE_FUP_FOR_PREDICTION
            entry_time = np.full(mask.sum(), BASE_FUP_FOR_PREDICTION)
            kmf.fit(absolute_duration, df_surv[mask]["event"], entry=entry_time, label=label)
            kmf.plot_survival_function(color=color, ci_show=True, ax=ax_km, linewidth=2)

    if (df_surv["is_high_risk"] == True).sum() > 0 and (df_surv["is_high_risk"] == False).sum() > 0:
        results = logrank_test(
            df_surv[df_surv["is_high_risk"]]["duration"],
            df_surv[~df_surv["is_high_risk"]]["duration"],
            df_surv[df_surv["is_high_risk"]]["event"],
            df_surv[~df_surv["is_high_risk"]]["event"]
        )
        if results.p_value < 0.001:
            p_val_text = "log-rank p-value: < 0.001"
        else:
            p_val_text = f"log-rank p-value: {results.p_value:.2e}"
    else:
        p_val_text = "log-rank p-value: N/A"
    
    # Dark rectangle going from left/transparent to right/darker
    num_strips = 100  # to simulate a smooth gradient
    gradient_start = -100  # rectangle starts further left
    x_edges = np.linspace(gradient_start, BASE_FUP_FOR_PREDICTION, num_strips + 1)
    alphas = np.linspace(0.0, 0.5, num_strips)  
    for i in range(num_strips):
        ax_km.axvspan(x_edges[i], x_edges[i+1], facecolor='black', alpha=alphas[i], zorder=1, lw=0)

    # Fix axis limit and add the vertical text label
    ax_km.set_xlim(gradient_start, MAX_FUP)
    ax_km.text(
        x=0, y=0.5, s=f"Observation window (-∞; {BASE_FUP_FOR_PREDICTION} days]",
        rotation=90, va='center', ha='center', color='#333333', zorder=2,
        fontweight='bold', transform=ax_km.get_xaxis_transform(),
    )
    
    ax_km.set_title(f"Longitudinal risk stratification ({p_val_text})")
    ax_km.set_xlabel("Days since transplantation")
    ax_km.set_ylabel("Probability of remaining event-free")
    ax_km.grid(axis='y', alpha=0.3)

    # Calibration plot
    try:
        df_surv['risk_bin'] = pd.qcut(df_surv['risk_score'], q=5, duplicates='drop')
    except ValueError:
        df_surv['risk_bin'] = pd.cut(df_surv['risk_score'], bins=5)
        
    cal_mean_preds = []
    cal_obs_risks = []
    
    kmf_cal = KaplanMeierFitter()
    for _, group in df_surv.groupby('risk_bin', observed=True):
        if len(group) == 0: continue
        cal_mean_preds.append(group['risk_score'].mean())
        
        kmf_cal.fit(group['duration'], group['event'])
        surv_prob = kmf_cal.predict(PREDICTION_HORIZON)
        cal_obs_risks.append(1.0 - surv_prob)

    max_val = max(max(cal_mean_preds), max(cal_obs_risks)) * 1.2 if cal_mean_preds else 1.0
    ax_cal.plot([0, max_val], [0, max_val], color='gray', linestyle='--', alpha=0.5, label="Perfect calibration")
    ax_cal.plot(
        cal_mean_preds, cal_obs_risks, marker='o', color='tab:red', 
        linestyle='-', linewidth=2, markersize=8, label="Model calibration"
    )
    
    title = f"Model calibration"
    ax_cal.set_title(title)
    ax_cal.set_xlabel("Mean predicted risk (grouped by quantiles)")
    ax_cal.set_ylabel(f"Observed event rate (KM estimate at {PREDICTION_HORIZON} days)")
    ax_cal.set_xlim(0, max_val)
    ax_cal.set_ylim(0, max_val)
    ax_cal.legend(loc='best')
    ax_cal.grid(axis='both', alpha=0.3)

    # Correlation plot
    sns.regplot(
        data=df_surv, x="risk_score", y="true_risk",
        scatter_kws={'alpha': 0.5, 's': 30},
        line_kws={'color': 'tab:red'}, ax=ax_corr
    )
    corr_val = df_surv["risk_score"].corr(df_surv["true_risk"], method="spearman")
    ax_corr.set_title(f"Predicted risk vs event imminence ($r_s$: {corr_val:.2f})")
    ax_corr.set_xlabel("Model predicted risk")
    ax_corr.set_ylabel("Event imminence (exponential decay)")
    ax_corr.grid(axis='both', alpha=0.3)

    # Risk-mapped UMAPs
    plot_df = pd.DataFrame({
        "UMAP 1": reduced_embeddings[:, 0], "UMAP 2": reduced_embeddings[:, 1],
        "Predicted risk": probs, "True risk": df_surv["true_risk"].values,
        "Outcome": ["Positive" if y == 1 else "Negative" for y in true_labels],
        "Duration": df_surv["duration"].values, "Event": df_surv["event"].values,
    })
    
    # Calculate the global min and max risk to keep colors consistent
    vmin_risk = plot_df["Predicted risk"].min()
    vmax_risk = plot_df["Predicted risk"].max()
    
    # Plot negatives (circles) first (zorder=2)
    subset_neg = plot_df[plot_df["Outcome"] == "Negative"]
    sc_neg = ax_umap_pred.scatter(
        subset_neg["UMAP 1"], subset_neg["UMAP 2"], c=subset_neg["Predicted risk"],
        cmap="coolwarm", vmin=vmin_risk, vmax=vmax_risk, marker="o", edgecolor="black",
        label="No event (censored)", zorder=2, s=60, linewidth=0.8, 
    )
    
    # Plot positives (crosses) last so they are on top
    subset_pos = plot_df[plot_df["Outcome"] == "Positive"]
    ax_umap_pred.scatter(
        subset_pos["UMAP 1"], subset_pos["UMAP 2"], c=subset_pos["Predicted risk"],
        cmap="coolwarm", vmin=vmin_risk, vmax=vmax_risk, marker="X", s=140,
        edgecolor="black", linewidth=0.8, label="Event occurred", zorder=3,
    )

    # Polish plot
    plt.colorbar(sc_neg, ax=ax_umap_pred, label="Predicted risk score")
    ax_umap_pred.set_title("UMAP: colored by predicted risk")
    ax_umap_pred.set_xlabel("UMAP 1")
    ax_umap_pred.set_ylabel("UMAP 2")
    handles, labels = ax_umap_pred.get_legend_handles_labels()
    ax_umap_pred.legend(handles, labels, title=None, loc="best")
    censored_mask = plot_df["Event"] == 0
    event_mask = plot_df["Event"] == 1
    
    # Risk-mapped UMAPs (with true future risk)
    censored_mask = plot_df["Event"] == 0
    event_mask = plot_df["Event"] == 1
    
    # Plot negative (censored) events first
    ax_umap_true.scatter(
        plot_df[censored_mask]["UMAP 1"], plot_df[censored_mask]["UMAP 2"],
        c="lightgrey", marker="o", s=60, edgecolor="black", linewidth=0.8,
        label="No event (censored)", zorder=2,
    )
    
    # Plot positives events last so they are on top
    if event_mask.sum() > 0:
        grey_to_red_cmap = LinearSegmentedColormap.from_list("grey_red", ["lightgrey", "tab:red"])
        sc_true = ax_umap_true.scatter(
            plot_df[event_mask]["UMAP 1"], plot_df[event_mask]["UMAP 2"],
            c=plot_df[event_mask]["True risk"], cmap=grey_to_red_cmap, vmin=0.0, vmax=1.0,
            marker="X", s=140, edgecolor="black", linewidth=0.8, label="Event occurred", zorder=3,
        )
        plt.colorbar(sc_true, ax=ax_umap_true, label="Event imminence (exponential decay)")

    # Polish true future risk plot
    ax_umap_true.set_title("UMAP: colored by event imminence")
    ax_umap_true.set_xlabel("UMAP 1")
    ax_umap_true.set_ylabel("UMAP 2")
    handles, labels = ax_umap_true.get_legend_handles_labels()
    ax_umap_true.legend(handles, labels, title=None, loc='best')
    
    # Save figure
    out_path = OUTPUT_DIR / f"combined_stratification_{task_key}.png"
    plt.tight_layout(h_pad=3.0, w_pad=3.0)
    plt.subplots_adjust(top=0.94)
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved combined analysis to {out_path.name}")


def compute_cluster_enrichment_profiles(base_test_ds, cluster_labels, true_labels, vocab, top_k_features=100):
    """
    Computes infection (event) rates per cluster and runs enrichment testing 
    to discover which clinical features dominate specific clusters.
    Unified feature normalization rules applied to match interpretation script outputs.
    """
    print("\n>>> Computing cluster enrichment profiles...")
    
    # Create an inverse vocabulary dictionary for decoding: ID -> String
    id2token = {v: k for k, v in vocab.items()}
    
    # Pack data into a processing DataFrame
    df_patients = pd.DataFrame({
        "patientid": base_test_ds["patientid"],
        "cluster": cluster_labels,
        "outcome": true_labels
    })
    
    # Calculate global baselines
    total_patients = len(df_patients)
    cluster_counts = df_patients["cluster"].value_counts()
    
    # Flatten patient data to count feature occurrences
    patient_features = {}
    all_feature_keys = set()
    
    for idx, sample in enumerate(base_test_ds):
        pid = sample["patientid"]
        
        # Access all three structural EAV token ID sequences
        ent_ids  = sample.get("entity_id", [])
        attr_ids = sample.get("attribute_id", [])
        val_ids  = sample.get("value_id", [])
        
        f_set = set()
        # Zip all three parallel token tracking sequences together
        for e_id, a_id, v_id in zip(ent_ids, attr_ids, val_ids):
            # Skip padding tokens (ID 0)
            if a_id == 0:
                continue
            
            # Decode all three components back into standard text strings
            ent_name  = id2token.get(e_id, f"Ent_{e_id}")
            attr_name = id2token.get(a_id, f"Attr_{a_id}")
            val_name  = id2token.get(v_id, f"Val_{v_id}")
            
            # Unified normalization logic from interpretation pipeline
            if "infection" in ent_name.lower():
                if ent_name.strip().lower() == "infection":
                    ent_name = "Previous infection"
                elif not ent_name.lower().startswith("previous"):
                    ent_name = f"Previous {ent_name.lower()}"
            
            full_feature_name = f"{ent_name} - {attr_name}"
            
            # Combine components into standardized profile triplet format
            f_set.add(f"{full_feature_name} : {val_name}")
            
        patient_features[pid] = f_set
        all_feature_keys.update(f_set)
        
    # Process each discovered cluster (skipping noise label -1)
    unique_clusters = sorted([c for c in df_patients["cluster"].unique() if c != -1])
    
    cluster_summaries = {}
    
    for cluster_id in unique_clusters:
        cluster_mask = df_patients["cluster"] == cluster_id
        cluster_pids = df_patients[cluster_mask]["patientid"].tolist()
        n_cluster = len(cluster_pids)
        
        if n_cluster == 0: continue
        
        # Calculate cluster specific event rate
        cluster_events = df_patients[cluster_mask]["outcome"].sum()
        event_rate = (cluster_events / n_cluster) * 100
        
        print(f" -> Analyzing Cluster {cluster_id} (N={n_cluster}, Event Rate={event_rate:.1f}%)")
        
        # Extract features present inside this cluster vs background rest of the cohort
        bg_pids = df_patients[~cluster_mask]["patientid"].tolist()
        n_bg = len(bg_pids)
        
        cluster_feat_counts = pd.Series([f for pid in cluster_pids for f in patient_features.get(pid, [])]).value_counts()
        bg_feat_counts = pd.Series([f for pid in bg_pids for f in patient_features.get(pid, [])]).value_counts()
        
        enrichment_results = []
        
        # Filter candidate features to avoid rare testing noise
        candidates = cluster_feat_counts[cluster_feat_counts > (n_cluster * 0.05)].index
        
        for feat in candidates:
            a = cluster_feat_counts.get(feat, 0)
            c = bg_feat_counts.get(feat, 0)
            
            # Fisher exact test (Alternative 'greater' to find over-represented items)
            odds, p_val = fisher_exact([[a, n_cluster - a], [c, n_bg - c]], alternative='greater')
            
            enrichment_results.append({
                "Feature": feat,
                "Cluster_%": (a / n_cluster) * 100,
                "Background_%": (c / n_bg) * 100 if n_bg > 0 else 0,
                "Odds_Ratio": odds,
                "P_Value": p_val
            })
            
        df_enrich = pd.DataFrame(enrichment_results)
        if not df_enrich.empty:
            # Sort by highest association strength (Odds Ratio) and statistical certainty
            df_enrich = df_enrich.sort_values(by=["P_Value", "Odds_Ratio"], ascending=[True, False])
            top_features = df_enrich.head(top_k_features)
        else:
            top_features = pd.DataFrame()
            
        cluster_summaries[cluster_id] = {
            "n_patients": n_cluster,
            "event_rate": event_rate,
            "top_drivers": top_features
        }
        
    return cluster_summaries


def save_cluster_profiles_report(cluster_profiles, task_key, space_type="fine-tuned"):
    """
    Formats the cluster profiling results into a clear text summary file.
    Extended padding allocated to prevent clipping of parsed triplet labels.
    """
    out_file = OUTPUT_DIR / f"cluster_profiles_{space_type}_{task_key}.txt"
    
    with open(out_file, "w") as f:
        f.write(f"==============================================================\n")
        f.write(f"CLUSTER PROFILE ENRICHMENT: {task_key.upper()} ({space_type.upper()} SPACE)\n")
        f.write(f"==============================================================\n\n")
        
        for cid, profile in cluster_profiles.items():
            f.write(f"### CLUSTER {cid} ###\n")
            f.write(f"Size: {profile['n_patients']} patients\n")
            f.write(f"Infection/Event Rate: {profile['event_rate']:.2f}%\n")
            f.write(f"Distinguishing Clinical Characteristics:\n")
            f.write(f"{'-' * 100}\n")
            
            df_drivers = profile["top_drivers"]
            if df_drivers.empty:
                f.write("  No highly distinctive features met significance limits.\n")
            else:
                f.write(f"  {'Clinical EAV Combination (Feature : Value)':<60} | {'Cluster %':<10} | {'Cohort %':<10} | {'P-Value':<8}\n")
                f.write(f"  {'-' * 100}\n")
                for _, row in df_drivers.iterrows():
                    f.write(f"  {row['Feature']:<60} | {row['Cluster_%']:>8.1f}% | {row['Background_%']:>8.1f}% | {row['P_Value']:.2e}\n")
            f.write("\n" + "=" * 100 + "\n\n")
            
    print(f"Saved complete clinical cluster profiles text report to: {out_file.name}")


if __name__ == "__main__":
    main()