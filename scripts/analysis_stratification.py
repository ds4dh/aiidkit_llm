import gc
import yaml
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from torch.utils.data import DataLoader
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from scipy.stats import fisher_exact, chi2_contingency
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test
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
MAX_FUP = 3600                  
if DATA_SPLIT_TYPE == "temporal_split":
    MAX_FUP = 2400
BASE_FUP_FOR_PREDICTION = 90   
PREDICTION_HORIZON = 30
SAFE_NUM_PROC = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Fontsize scaling configurations
FS_SCALE = 1.3
plt.rcParams.update({
    'font.size': 14 * FS_SCALE,              
    'axes.labelsize': 17 * FS_SCALE,         
    'axes.titlesize': 16 * FS_SCALE,         
    'xtick.labelsize': 14 * FS_SCALE,        
    'ytick.labelsize': 14 * FS_SCALE,        
    'legend.fontsize': 13 * FS_SCALE,        
    'legend.title_fontsize': 14 * FS_SCALE,  
    'figure.titlesize': 22 * FS_SCALE,       
})


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
        test_ds = dataset["test"]
        test_ds = test_ds.add_column("split", ["test"] * len(test_ds))

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
        patient_ids = base_test_ds["patientid"]

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

        # ---------------------------------------------------------------------------------
        # Build unified cluster color map ahead of generation to secure synergy
        # ---------------------------------------------------------------------------------
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

        print("\nBuilding consolidated stratification performance canvas...")
        plot_combined_stratification_grid(
            test_ds=test_ds,
            patient_ids=patient_ids,
            probs=probs_ft,
            reduced_pt=reduced_pt,
            reduced_ft=reduced_ft,
            labels_pt=labels_pt,
            labels_ft=labels_ft,
            cluster_color_map=cluster_color_map, 
            true_labels_horizon=res_ft["labels"][:, target_idx],
            task_key=task_key,
            label_key=label_key
        )


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

    # FIXED: Cleaned up trailing entries to optimize the legend box bounds
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


def plot_combined_stratification_grid(
    test_ds, patient_ids, probs, reduced_pt, reduced_ft, labels_pt, labels_ft, cluster_color_map, true_labels_horizon, task_key, label_key
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

    grey_to_red_cmap = LinearSegmentedColormap.from_list("grey_red", ["lightgrey", "tab:red"])
    
    fig, axes = plt.subplots(3, 2, figsize=(20, 22))

    s_censored = 90
    s_event = 190
    censored_mask = df_surv["event"] == 0
    event_mask = df_surv["event"] == 1

    # Pre-trained / fine-tuned UMAP panels
    plot_scatter_unsupervised(
        axes[0, 0], reduced_pt, labels_pt, true_labels_horizon, cluster_color_map, "Pre-trained UMAP: discovered clusters"
    )
    plot_scatter_unsupervised(
        axes[0, 1], reduced_ft, labels_ft, true_labels_horizon, cluster_color_map, "Fine-tuned UMAP: discovered clusters"
    )

    # Fine-tuned UMAP colored by model predicted risk
    ax = axes[1, 0]
    vmin_risk, vmax_risk = df_surv["risk_score"].min(), df_surv["risk_score"].max()
    ax.scatter(
        reduced_ft[censored_mask, 0], reduced_ft[censored_mask, 1],
        c=df_surv[censored_mask]["risk_score"], cmap="coolwarm", vmin=vmin_risk, vmax=vmax_risk,
        marker="o", s=s_censored, edgecolor="black", linewidth=0.8, label="No event (censored)", zorder=2, alpha=0.4  
    )
    if event_mask.sum() > 0:
        sc2 = ax.scatter(
            reduced_ft[event_mask, 0], reduced_ft[event_mask, 1],
            c=df_surv[event_mask]["risk_score"], cmap="coolwarm", vmin=vmin_risk, vmax=vmax_risk,
            marker="X", s=s_event, edgecolor="black", linewidth=0.8, label="Event occurred (ever)", zorder=3, alpha=0.75  
        )
    sc2_ref = sc2 if event_mask.sum() > 0 else ax.collections[0]
    cbar2 = plt.colorbar(sc2_ref, ax=ax)
    cbar2.set_label("Predicted risk score")
    ax.set_title("Fine-tuned UMAP: model risk score")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.legend(loc="upper right")

    # Fine-tuned UMAP with event imminence
    ax = axes[2, 0]
    ax.scatter(
        reduced_ft[censored_mask, 0], reduced_ft[censored_mask, 1],
        c="lightgrey", marker="o", s=s_censored, edgecolor="black", linewidth=0.8,
        label="No event (censored)", zorder=2, alpha=0.4  
    )
    if event_mask.sum() > 0:
        sc1 = ax.scatter(
            reduced_ft[event_mask, 0], reduced_ft[event_mask, 1],
            c=df_surv[event_mask]["true_imminence"], cmap=grey_to_red_cmap, vmin=0.0, vmax=1.0,
            marker="X", s=s_event, edgecolor="black", linewidth=0.8, label="Event occurred (ever)", zorder=3, alpha=0.75  
        )
        cbar1 = plt.colorbar(sc1, ax=ax)
        cbar1.set_label("Event imminence (exponential decay)")
    ax.set_title("Fine-tuned UMAP: event imminence")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.legend(loc="upper right")

    # Stratified Kaplan-Meier discovered cluster curves
    ax = axes[1, 1]
    kmf = KaplanMeierFitter()
    
    unique_clusters = sorted(df_surv["cluster_ft"].unique())
    for cid in unique_clusters:
        if cid == -1: continue
        mask = df_surv["cluster_ft"] == cid
        if mask.sum() == 0:
            continue
            
        lbl = f"Fine-tuned UMAP cluster {cid}"
        clr = cluster_color_map.get(cid, "darkgrey")
        abs_duration = df_surv[mask]["duration"] + BASE_FUP_FOR_PREDICTION
        entry_t = np.full(mask.sum(), BASE_FUP_FOR_PREDICTION)
        kmf.fit(abs_duration, df_surv[mask]["event"], entry=entry_t, label=lbl)
        kmf.plot_survival_function(color=clr, ci_show=True, ax=ax, linewidth=2.5, alpha=1.0)

    # Perform Multivariate Log-Rank Test across all valid non-noise clusters
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
        ax.axvspan(x_edges[i], x_edges[i+1], facecolor='black', alpha=alphas[i], zorder=1, lw=0)

    ax.set_xlim(g_start, MAX_FUP)
    ax.text(
        0, 0.5, f"Observation window (-∞; {BASE_FUP_FOR_PREDICTION} days]",
        rotation=90, va='center', ha='center', color='#333333', zorder=2,
        fontweight='bold', transform=ax.get_xaxis_transform()
    )
    # Preservation of your interactive runtime text label updates wrapper style
    ax.set_title(f"Longitudinal risk stratification ({p_val_text})")
    ax.set_xlabel("Days since transplantation")
    ax.set_ylabel("Probability of remaining event-free")
    ax.grid(True, axis='y', alpha=0.3)
    ax.legend(loc="upper right")  

    # Correlation plot
    ax = axes[2, 1]
    ax.scatter(df_surv["risk_score"], df_surv["true_imminence"], color="tab:blue", alpha=0.4, s=45) 
    sns.regplot(
        data=df_surv, x="risk_score", y="true_imminence",
        scatter=False, line_kws={'color': 'tab:red', 'linewidth': 2.5}, ax=ax
    )
    corr_val = df_surv["risk_score"].corr(df_surv["true_imminence"], method="spearman")
    ax.set_title(f"Predicted risk vs event imminence ($r_s$: {corr_val:.2f})")
    ax.set_xlabel("Model predicted risk")
    ax.set_ylabel("Event imminence (exponential decay)")
    ax.grid(True, alpha=0.3)

    plt.tight_layout(h_pad=2.0, w_pad=3.0)
    out_path = OUTPUT_DIR / f"combined_stratification_{task_key}.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved optimized stratification summary to {out_path.name}")
    

def compute_cluster_enrichment_profiles(base_test_ds, cluster_labels, true_labels, vocab, top_k_features=100):
    print("\n>>> Computing cluster enrichment profiles...")
    id2token = {v: k for k, v in vocab.items()}
    df_patients = pd.DataFrame({
        "patientid": base_test_ds["patientid"],
        "cluster": cluster_labels,
        "outcome": true_labels
    })
    
    total_patients = len(df_patients)
    
    valid_clusters_df = df_patients[
        df_patients["cluster"].isin([0, 1]) & (df_patients["outcome"] >= 0)
    ]
    c_counts = valid_clusters_df["cluster"].value_counts()
    
    global_stats = {"has_comparison": False}
    if 0 in c_counts.index and 1 in c_counts.index:
        n_c0 = (valid_clusters_df["cluster"] == 0).sum()
        n_c1 = (valid_clusters_df["cluster"] == 1).sum()
        events_c0 = valid_clusters_df[valid_clusters_df["cluster"] == 0]["outcome"].sum()
        events_c1 = valid_clusters_df[valid_clusters_df["cluster"] == 1]["outcome"].sum()
        
        non_events_c0 = max(0, n_c0 - events_c0)
        non_events_c1 = max(0, n_c1 - events_c1)
        
        contingency_table = [[events_c0, non_events_c0], [events_c1, non_events_c1]]
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
        candidates = cluster_feat_counts[cluster_feat_counts > (n_cluster * 0.05)].index
        
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
                "P_Value": p_val 
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


if __name__ == "__main__":
    main()