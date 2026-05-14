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
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from transformers.trainer_utils import get_last_checkpoint

from src.data.patient_dataset import load_hf_data_and_metadata
from src.model.patient_embedder import PatientEmbeddingModelFactory, PatientDataCollatorForClassification
from src.evaluation.evaluate_models import UMAP_HDBSCAN_Clusterer, ModelInterpreter
from scripts.interpret_models import find_best_checkpoint, extract_horizons_from_path
from scripts.script_utils import scan_all_fups


# Model selection
RESULTS_DIR = Path("results_optuna")
DATA_SPLIT_TYPE = "random_split"
BEST_TRIALS = {  
    "infection_bacteria": "trial_032",
    # "infection_virus": "trial_034",
    # "death": "trial_040",
    # "graft_loss": "trial_000",
}
TRANSFORMER_PT_CONFIGS = {  
    "infection_bacteria": "e20-a00-v60",
    # "infection_virus":    "e10-a05-v45",
    # "death":              "e00-a20-v40",
    # "graft_loss":         "e10-a25-v30",
}

# General settings
CONFIG_PATH = Path("configs/discriminative_training.yaml") 
DATA_DIR = Path("/home/shares/ds4dh/aiidkit_project/data_new/processed/v3.5/teav")
OUTPUT_DIR = Path("./results_stratification")
BASE_FUP_FOR_PREDICTION = 90    # post-tpx follow-up day where risk is evaluated
PREDICTION_HORIZON = 30         # model horizon to use for stratification
SAFE_NUM_PROC = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load the training configuration to get the proper model dtypes and args
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)
        
    for task_key, trial_name in BEST_TRIALS.items():
        print(f"\nStratifying task: {task_key}")
        
        # Locate checkpoints (pre-trained and fine-tuned)
        try:
            pt_config = TRANSFORMER_PT_CONFIGS[task_key]
            base_dir_for_ckpt = RESULTS_DIR / DATA_SPLIT_TYPE / task_key / trial_name / DATA_SPLIT_TYPE / pt_config
            
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
            import ipdb; ipdb.set_trace()
            print(f"  [Skip] Could not find valid models for {task_key}: {e}")
            continue

        # Load dataset
        print("  Loading patient sequences...")
        label_key = f"label_{task_key}_{PREDICTION_HORIZON:04d}d"
        all_required_labels = [f"label_{task_key}_{h:04d}d" for h in ckpt_horizons]
        
        data_dir_split = DATA_DIR / DATA_SPLIT_TYPE
        all_fups = scan_all_fups(data_dir_split)

        if not all_fups:
            print(f"  [Error] No fup_XXXX folders found in {data_dir_split}")
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
            print(f"  [Error] No patients found at FUP {BASE_FUP_FOR_PREDICTION}")
            continue
        loader = DataLoader(base_test_ds, batch_size=32, collate_fn=collator.torch_call)

        # Inference with pre-trained model
        print(f"\n  Initializing pre-trained model from {Path(checkpoint_path_pt).name}...")
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
        print("  Extracting pre-trained embeddings...")
        res_pt = interpreter_pt.get_embeddings_and_predictions(loader, extract_logits=False)
        embeddings_pt = res_pt["embeddings"]
        
        del model_pt, interpreter_pt
        gc.collect()
        torch.cuda.empty_cache()

        # Inference with fine-tuned model
        print(f"\n  Initializing fine-tuned model from {Path(checkpoint_path_ft).name}...")
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
        print("  Extracting fine-tuned embeddings and predictions...")
        res_ft = interpreter_ft.get_embeddings_and_predictions(loader)
        embeddings_ft = res_ft["embeddings"]
        
        probs_ft = 1 / (1 + np.exp(-res_ft["logits"][:, target_idx]))  
        true_labels_at_base = res_ft["labels"][:, target_idx]
        patient_ids = base_test_ds["patientid"]

        del model_ft, interpreter_ft
        gc.collect()
        torch.cuda.empty_cache()

        # Dimensionality reduction
        print("\n  Running UMAP + HDBSCAN on pre-trained embedding space...")
        clusterer_pt = UMAP_HDBSCAN_Clusterer(n_optuna_trials=0)
        reduced_pt, labels_pt = clusterer_pt.predict(embeddings_pt, n_components=2, min_cluster_size=15, min_samples=5)

        print("  Running UMAP + HDBSCAN on fine-tuned embedding space...")
        clusterer_ft = UMAP_HDBSCAN_Clusterer(n_optuna_trials=0)
        reduced_ft, labels_ft = clusterer_ft.predict(embeddings_ft, n_components=2, min_cluster_size=15, min_samples=5)

        plot_clustering_comparison(
            reduced_pt, reduced_ft, labels_pt, labels_ft, true_labels_at_base, task_key
        )

        # Run survival analysis
        print("\n  Calculating survival trajectories and calibration...")
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
    df = pd.DataFrame({
        "UMAP 1": reduced[:, 0], "UMAP 2": reduced[:, 1], "Hue": hue_labels,
        "Outcome": ["Positive" if y == 1 else "Negative" for y in style_labels],
    })
    
    palette = {"Positive": "tab:red", "Negative": "tab:blue"} if is_outcome else "bright"        
    sns.scatterplot(
        data=df, x="UMAP 1", y="UMAP 2", hue="Hue", style="Outcome", 
        palette=palette, markers={"Positive": "X", "Negative": "o"},
        alpha=0.7, edgecolor="white", s=60, ax=ax,
    )
    ax.set_title(title, fontsize=14, fontweight='normal')
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', title="Labels" if is_outcome else "Clusters")


def plot_clustering_comparison(reduced_pt, reduced_ft, labels_pt, labels_ft, true_labels, task_key):
    fig, axes = plt.subplots(2, 2, figsize=(18, 16))
    task_str = task_key.replace('_', ' ').title()
    fig.suptitle(
        f"Embedding space evolution: pre-trained vs fine-tuned ({task_str})", 
        fontsize=18, fontweight='bold', y=0.98,
    )

    c_labels_pt = [f"C{c}" if c != -1 else "Noise" for c in labels_pt]
    c_labels_ft = [f"C{c}" if c != -1 else "Noise" for c in labels_ft]
    out_labels = ["Positive" if y == 1 else "Negative" for y in true_labels]

    # Unsupervised clusters
    plot_scatter(axes[0, 0], reduced_pt, c_labels_pt, true_labels, "Pre-trained: discovered clusters", is_outcome=False)
    plot_scatter(axes[0, 1], reduced_ft, c_labels_ft, true_labels, "Fine-tuned: discovered clusters", is_outcome=False)

    # True clinical outcomes
    plot_scatter(axes[1, 0], reduced_pt, out_labels, true_labels, "Pre-trained: true outcomes", is_outcome=True)
    plot_scatter(axes[1, 1], reduced_ft, out_labels, true_labels, "Fine-tuned: true outcomes", is_outcome=True)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = OUTPUT_DIR / f"clustering_comparison_{task_key}.png"
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"    Saved cluster comparison to {out_path.name}")


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
            # Exponential decay: 1.0 at day 0, smooth decay out to multi-year horizons
            true_risk = np.exp(-duration / 365.0) 
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

    df_surv = pd.DataFrame(survival_data)
    
    fig = plt.figure(figsize=(18, 22))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.2, 1, 1])
    
    ax_km = fig.add_subplot(gs[0, :])
    ax_cal = fig.add_subplot(gs[1, 0])
    ax_corr = fig.add_subplot(gs[2, 0])
    ax_umap_pred = fig.add_subplot(gs[1, 1])
    ax_umap_true = fig.add_subplot(gs[2, 1])
    
    # Kaplan-Meier comparison
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
        p_val_text = f"Log-rank p-value: {results.p_value:.2e}"
    else:
        p_val_text = "Log-rank p-value: N/A"

    ax_km.set_title(f"Longitudinal risk stratification\n({p_val_text})", fontsize=16)
    ax_km.set_xlabel("Days since transplantation", fontsize=12)
    ax_km.set_ylabel("Probability of remaining event-free", fontsize=12)
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
    
    title = f"Model calibration at {PREDICTION_HORIZON} days\n(Predicted vs observed event rate)"
    ax_cal.set_title(title, fontsize=14, fontweight='normal')
    ax_cal.set_xlabel("Mean predicted risk (grouped by quantiles)")
    ax_cal.set_ylabel(f"Observed event rate (KM estimate at {PREDICTION_HORIZON} days)")
    ax_cal.set_xlim(0, max_val)
    ax_cal.set_ylim(0, max_val)
    ax_cal.legend(loc='upper left')
    ax_cal.grid(axis='both', alpha=0.3)

    # Correlation Plot
    sns.regplot(
        data=df_surv, x="risk_score", y="true_risk", 
        scatter_kws={'alpha': 0.5, 's': 30}, 
        line_kws={'color': 'tab:red'}, ax=ax_corr
    )
    corr_val = df_surv["risk_score"].corr(df_surv["true_risk"], method="spearman")
    ax_corr.set_title(f"Predicted risk vs true future risk\n(Spearman r: {corr_val:.2f})", fontsize=14, fontweight='normal')
    ax_corr.set_xlabel("Model predicted risk")
    ax_corr.set_ylabel("True future risk (exponential decay)")
    ax_corr.grid(axis='both', alpha=0.3)

    # Risk-mapped UMAPs
    plot_df = pd.DataFrame({
        "UMAP 1": reduced_embeddings[:, 0],
        "UMAP 2": reduced_embeddings[:, 1],
        "Predicted risk": probs,
        "True risk": df_surv["true_risk"].values,
        "Outcome": ["Positive" if y == 1 else "Negative" for y in true_labels],
        "Duration": df_surv["duration"].values,
        "Event": df_surv["event"].values
    })
    
    sc1 = ax_umap_pred.scatter(
        plot_df["UMAP 1"], plot_df["UMAP 2"],
        c=plot_df["Predicted risk"], cmap="coolwarm",
        s=50, alpha=0.6, edgecolor="white", linewidth=0.5
    )
    for outcome, marker in [("Positive", "X"), ("Negative", "o")]:
        subset = plot_df[plot_df["Outcome"] == outcome]
        ax_umap_pred.scatter(
            subset["UMAP 1"], subset["UMAP 2"], c=subset["Predicted risk"], cmap="coolwarm",
            marker=marker, s=70, edgecolor="black", linewidth=0.8, label=f"True {outcome}"
        )

    plt.colorbar(sc1, ax=ax_umap_pred, label="Predicted risk score")
    ax_umap_pred.set_title("UMAP: colored by predicted risk", fontsize=14, fontweight='normal')
    ax_umap_pred.set_xlabel("UMAP 1")
    ax_umap_pred.set_ylabel("UMAP 2")
    ax_umap_pred.legend(title="Outcome style")

    censored_mask = plot_df["Event"] == 0
    event_mask = plot_df["Event"] == 1
    
    ax_umap_true.scatter(
        plot_df[censored_mask]["UMAP 1"], plot_df[censored_mask]["UMAP 2"],
        c="lightgrey", s=50, alpha=0.6, edgecolor="white", linewidth=0.5, label="No event (censored)"
    )
    
    if event_mask.sum() > 0:
        grey_to_red_cmap = LinearSegmentedColormap.from_list("grey_red", ["lightgrey", "tab:red"])
        sc2 = ax_umap_true.scatter(
            plot_df[event_mask]["UMAP 1"], plot_df[event_mask]["UMAP 2"],
            c=plot_df[event_mask]["True risk"], cmap=grey_to_red_cmap, vmin=0.0, vmax=1.0,
            marker="X", s=70, edgecolor="black", linewidth=0.8, label="Event occurred",
        )
        plt.colorbar(sc2, ax=ax_umap_true, label="True future risk (exponential decay)")

    ax_umap_true.set_title("UMAP: colored by true future risk", fontsize=14, fontweight='normal')
    ax_umap_true.set_xlabel("UMAP 1")
    ax_umap_true.set_ylabel("UMAP 2")
    ax_umap_true.legend(loc='upper right')

    plt.tight_layout()
    out_path = OUTPUT_DIR / f"combined_stratification_{task_key}.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"    Saved combined analysis to {out_path.name}")


if __name__ == "__main__":
    main()