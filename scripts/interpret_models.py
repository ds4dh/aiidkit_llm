import re
import argparse
import yaml
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from tqdm.auto import tqdm
from adjustText import adjust_text
from matplotlib.lines import Line2D
from torch.utils.data import DataLoader, Subset
from scipy.stats import fisher_exact
from scipy.special import expit
from captum.attr import LayerIntegratedGradients

from src.data.patient_dataset import load_hf_data_and_metadata
from src.model.patient_embedder import PatientEmbeddingModelFactory, PatientDataCollatorForClassification
from src.evaluation.evaluate_models import UMAP_HDBSCAN_Clusterer

SAFE_NUM_PROC = 4
CLI_CFG = {}
DEFAULT_TASK_KEY = "graft_loss"
TASK_CONFIG = {
    "death": {
        "base_dir": "backups/2026_01/BACKUP - results_optuna_death/trial_002/e15-a15-v15",  # !!! careful: unoptimized run!!!
        "horizon": 1095,
        "follow-up_periods": [180, 365, 730, 1095, 1460, 1825, 2180, 2540, 2900, 3260, 3650],
    },
    "graft_loss": {
        "base_dir": "backups/2026_01/BACKUP - results_optuna_graft_loss/trial_000/e15-a35-v15",  # !!! careful: unoptimized run!!!
        "horizon": 1095,
        "follow-up_periods": [180, 365, 730, 1095, 1460, 1825, 2180, 2540, 2900, 3260, 3650],
    },
    "infection_bacteria": {
        "base_dir": "backups/2026_01/BACKUP - results_optuna_infection_bacteria/trial_032/e05-a35-v35",
        "horizon": 90,
        "follow-up_periods": [90, 120, 150, 180, 210, 240, 270, 300, 330, 365],
    },
}

# =================================
# GLOBAL MAPPINGS AND CONFIGURATION
# =================================

# Define medication roles
MED_ROLES = {
    'Med. CNI':                 ['Tacrolimus', 'Cyclosporine'],
    'Med. antimetabolite':      ['Mycophenolate Mofetil', 'Mycophenolic Acid', 'Azathioprine'],
    'Med. mTOR inhibitor':      ['Everolimus', 'Sirolimus'],
    'Med. steroids':            ['Glucocorticoid', 'Methylprednisolone', 'Prednisone'],
    'Med. induction':           ['Basiliximab', 'Rabbit ATG', 'Anti-thymocyte globulin'],
    'Med. rejection treat.':    ['Rituximab', 'IVIG', 'Human Immunoglobulin', 'Plasmapheresis'],
    'Med. antiviral proph.':    ['Valganciclovir', 'Valaciclovir', 'Lamivudine', 'Ganciclovir', 'Entecavir'],
    'Med. antibiotic proph.':   ['Cotrimoxazole', 'Atovaquone', 'Pentamidine', 'Dapsone'],
    'Med. antibiotic treat.':   ['Beta-Lactame', 'Quinolone', 'Cephalosporin', 'Metronidazole', 'Fosfomycin', 'Nitrofurantoin', 'Clarithromycin', 'Isoniazid'],
    'Med. antifungal treat.':   ['Amphotericin B', 'Itraconazole', 'Fluconazole', 'Voriconazole'],
    'Med. antihypertensive':    ['Calcium channel blocker', 'Beta-blocker', 'ACE inhibitor', 'Angiotensin receptor blocker'],
    'Med. antithrombotic':      ['Platelet aggregation inhibitor', 'Anticoagulation therapy'],
    'Med. diabetes treat.':     ['Insulin', 'Oral antidiabetic drug'],
    'Med. diuretic':            ['Torasemide', 'Furosemide'],
    'Med. lipid lowering':      ['Statin'],
    'Med. other':               ['Other drugs']
}

# Infection and event categories
EVENT_ROLES = {
    # Events / procedures
    'Event rejection':          ['Biops proven rj', 'Clinically suspected rj'],
    'Event procedure':          ['Nephrectomy native', 'Nephrectomy allograft', 'Nephrectomy allograft and native'],
    'Previous graft failure':   ['Previous GF'],

    # Infection type
    'Inf. type proven dis.':    ['Proven disease'],
    'Inf. type poss. dis.':     ['Possible disease', 'Probable disease'],
    'Inf. type viral synd.':    ['Viral syndrome'],
    'Inf. type asympt.':        ['Asymptomatic'],
    'Inf. type col.':           ['Colonization'],
    
    # Infection category
    'Inf. cat. bacterial':      ['Bacterial', 'Bacteria'],
    'Inf. cat. viral':          ['Viral', 'Virus'],
    'Inf. cat. fungal':         ['Fungal', 'Fungi'],
    'Inf. cat. parasite':       ['Parasite'],
    
    # Infection site
    'Inf. site UTI':            ['Urinary tract', 'UTI'],
    'Inf. site RT':             ['Respiratory', 'Lung', 'Pneumonia', 'RT'],
    'Inf. site GO':             ['Gastrointestinal', 'Abdominal', 'GI'],
    'Inf. site MC':             ['Skin', 'Mucocutaneous'],
    'Inf. site SSI':            ['SSI'],
    'Inf. site CNS':            ['CNS'],
    'Inf. site Eye':            ['Eye'],
    'Inf. site Liver':          ['Liver'],
    'Inf. site Kidney':         ['Kidney'],
    'Inf. site Heart':          ['Heart'],
    'Inf. site Bone':           ['Bone_Joint'],
    'Inf. site Blood':          ['Blood', 'Sepsis', 'Bacteremia', 'Catheter'],
}

# Clinical events / categories 
CLINICAL_ROLES = {
    # Primary kidney diseases
    'Kidney dis. GN':                     ['GN'],
    'Kidney dis. CKD':                    ['CKD'],
    'Kidney dis. PCKD':                   ['PCKD'],
    'Kidney dis. Congenital':             ['Congenital kidney'],
    'Kidney dis. Nephritis':              ['Interstitial nephritis'],
    'Kidney dis. Pyelo/Reflux':           ['Reflux/Pyelonephritis'],
    'Kidney dis. Nephrosclerosis':        ['Nephrosclerosis'],
    'Kidney dis. DM Nephropathy':         ['DM nephropathy'],
    'Kidney dis. Hereditary':             ['Hereditary non_PCKD'],
    'Kidney dis. ARF':                    ['ARF'],
    'Kidney dis. Acute on Chronic':       ['Acute on chronic RF'],

    # Comorbidities
    'Comorb. DM Type 1':           ['DM type1'],
    'Comorb. DM Type 2':           ['DM type2 treated'],
    'Comorb. HTN':                 ['HTN'],
    'Comorb. Lipids':              ['Hyperlipidemia'],
    'Comorb. Osteoporosis':        ['Osteoporosis'],
    'Comorb. PTDM':                ['PTDM'],
    'Comorb. Heart (CMP)':         ['Dilated CMP'],
    'Comorb. Metabolic':           ['Other metabolic'],
    'Comorb. Alcohol':             ['Alcohol'],
    'Comorb. Liver Injury':        ['Drug-induced liver injury'],

    # Serology and donor
    'Sero. D+/R-':                 ['D+/R-'],
    'Sero. D-/R+':                 ['D-/R+'],
    'Sero. D+/R+':                 ['D+/R+'],
    'Sero. D-/R-':                 ['D-/R-'],
    'Sero. D+/R?':                 ['D+/R?'],
    'Sero. D-/R?':                 ['D-/R?'],
    'Sero. D?/R+':                 ['D?/R+'],
    'Sero. D?/R?':                 ['D?/R?'],
    'Donor DBD':                   ['DBD'],
    'Donor DCD':                   ['DCD'],
    'Donor Living Related':        ['Living related'],
    'Donor Living Unrelated':      ['Living unrelated'],

    # Cancers
    'Cancer Skin (Basal)':         ['Basalioma'],
    'Cancer Skin (Spin)':          ['Spinalioma'],
    'Cancer Skin (Melanoma)':      ['Melanoma'],
    'Cancer Skin (Other)':         ['Other skin cancer'],
    'Cancer Breast':               ['Breast cancer'],
    'Cancer Lung':                 ['Lung cancer'],
    'Cancer Prostate':             ['Prostate cancer'],
    'Cancer Lymphoma (PTLD)':      ['PTLD'],
    'Cancer Cervix/Uterus':        ['Cervix - Uterus - Adnex ca'],
    'Cancer Colorectal':           ['Colorectal cancer'],
    'Cancer Bladder':              ['Uro_bladder cancer'],
    'Cancer Kidney':               ['Kidney cancer'],
    'Cancer Myeloid':              ['Myeloid neoplasm'],
    'Cancer Other':                ['Other neoplasia'],
}

# Numeric, ordinal, and unknown definitions
NUMERIC_ROLES = {
    '0 / False / No / M': ['0', 'No', 'False', 'Negative', 'Male', 'M'], 
    '1 / True / Yes / F': ['1', 'Yes', 'True', 'Occurred', 'Positive', 'Female', 'F'],
    '2': ['2'], 
    '3': ['3'],
}
ORDINAL_ROLES = {
    level: level 
    for level in ['Lowest', 'Lower', 'Low', 'Middle', 'High', 'Higher', 'Highest']
}
UNKNOWN_ROLES = {
    'Other/Unknown': [
        'Unknown', '[UNK]', 'Condition unknown',
        'Other', 'Other event or disease',
        'Missing', 'Site not identified', 'Undetermined',
    ]
}


def parse_args():
    parser = argparse.ArgumentParser(description="Interpret model predictions and embeddings on the test set.")
    parser.add_argument("--config", "-c", type=str, default="configs/discriminative_training.yaml", help="Path to the training config used.")
    parser.add_argument("--output_dir", "-o", type=str, default="results_interpretability", help="Folder to save plots and tables.")
    parser.add_argument("--task_key", type=str, default=DEFAULT_TASK_KEY, help="Task key.")
    parser.add_argument("--captum_samples", type=int, default=500, help="Maximum sample per label analyzed by captum")
    parser.add_argument("--agg_method", type=str, choices=["mean", "sum"], default="mean", help="Aggregation method: 'mean' (conditional impact) or 'sum' (population burden).")
    parser.add_argument("--top_k", type=int, default=20, help="Maximum number of features plotted in the attribution plots.")
    parser.add_argument("--min_freq", type=int, default=50, help="Minimum number of patients a feature must appear in to be plotted.")
    parser.add_argument("--add_noise", action="store_true", help="Inject a random noise feature for baseline comparison.")    
    parser.add_argument("--plot_only", "-p", action="store_true", help="Skip computation, load existing CSVs and replot.")
    return parser.parse_args()


def find_best_checkpoint(base_dir: Path, task_key: str, horizon: int) -> Path:
    task_dir = base_dir / "finetuning" / task_key
    if not task_dir.exists(): raise FileNotFoundError(f"Task directory not found: {task_dir}")
    
    h_str = f"{horizon:04d}"
    pattern = re.compile(rf"hrz\(([^)]*\b{h_str}\b[^)]*)\)")
    candidates = [p for p in task_dir.iterdir() if p.is_dir() and pattern.search(p.name)]
    if not candidates: raise FileNotFoundError(f"No run found for horizon {h_str} inside hrz() in {task_dir}")
    
    run_dir = candidates[0]
    checkpoint_dirs = sorted(
        list(run_dir.glob("checkpoint-*")),
        key=lambda p: int(p.name.split("-")[-1]),
    )
    if not checkpoint_dirs: raise FileNotFoundError(f"No checkpoints found in {run_dir}")
    
    return checkpoint_dirs[0]  # first, because early stopping might have kept an overfitted checkpoint


def extract_horizons_from_path(checkpoint_path: Path) -> list[int]:
    run_dir = checkpoint_path
    while "hrz(" not in run_dir.name and run_dir.parent != run_dir:
        run_dir = run_dir.parent
    
    match = re.search(r"hrz\(([\d-]+)\)", run_dir.name)
    if not match:
        raise ValueError(f"Could not extract horizons from path: {run_dir.name}")
    
    return [int(h) for h in match.group(1).split("-")] 


class ModelInterpreter:
    def __init__(self, model, device="cuda"):
        self.model = model
        self.model.eval()
        self.model.to(device)
        self.device = device

    def get_embeddings_and_predictions(self, dataloader):
        embeddings_list, logits_list, labels_list = [], [], []
        with torch.no_grad():
            for batch in tqdm(dataloader):
                input_dict = {k: v.to(self.device) for k, v in batch["input_dict"].items()}
    
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    outputs = self.model(input_dict=input_dict, output_hidden_states=True) 
                
                logits = outputs.logits
                last_hidden = outputs.hidden_states[-1] 
                mask = batch["attention_mask"].to(self.device).unsqueeze(-1)
                
                sum_embeddings = (last_hidden * mask).sum(dim=1)
                sum_mask = mask.sum(dim=1).clamp(min=1e-9)
                pooled = sum_embeddings / sum_mask
                
                embeddings_list.append(pooled.float().cpu().numpy())
                logits_list.append(logits.float().cpu().numpy())
                labels_list.append(batch["labels"].cpu().numpy())
        
        return {
            "embeddings": np.vstack(embeddings_list),
            "logits": np.vstack(logits_list),
            "labels": np.vstack(labels_list),
        }


class ForwardWrapperForCaptum(torch.nn.Module):
    def __init__(self, model, pad_id: int = 0):
        super().__init__()
        self.model = model
        self.pad_id = pad_id
    
    def forward(self, entity_id, attribute_id, value_id, days_since_tpx):
        input_dict = {
            "entity_id": entity_id,
            "attribute_id": attribute_id,
            "value_id": value_id,
            "days_since_tpx": days_since_tpx
        }
        attention_mask = (entity_id != self.pad_id).long()            
        outputs = self.model(input_dict=input_dict, attention_mask=attention_mask)
        return outputs.logits


def extract_attributions(
    model, dataset, collator, indices, vocab, device, target_idx=0,
    entity_filter=None, add_noise=False, csv_path=None, plot_only=False,
) -> pd.DataFrame:
    
    if plot_only:
        return pd.read_csv(csv_path)

    # Find layer from which attribution will be derived (usually input layer)
    try:
        target_layer = model.patient_embedder.token_embedding
    except AttributeError:
        print("Error: Could not find 'token_embedding'. Captum skipped.")
        return pd.DataFrame()

    # Prepare dataset and vocabulary
    subset = Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=32, collate_fn=collator.torch_call)
    id2word = {v: k for k, v in vocab.items()} if isinstance(vocab, dict) else vocab
    
    # Prepare noise values if required
    if add_noise:
        ordinal_keys = list(ORDINAL_ROLES.keys())  # ["Lowest", "Lower", ..., "Highest"]
        valid_ordinal_ids = [vocab[k] for k in ordinal_keys if k in vocab]
        noise_value_pool = torch.tensor(valid_ordinal_ids, device=device)
    
    # Prepare captum
    wrapper = ForwardWrapperForCaptum(model).to(device)
    lig = LayerIntegratedGradients(wrapper, target_layer)

    collected_data = []
    for batch in tqdm(loader, desc="Calculating attributions"):
        inp = batch["input_dict"]
        ent = inp["entity_id"].to(device)
        attr = inp["attribute_id"].to(device)
        val = inp["value_id"].to(device)
        days = inp["days_since_tpx"].to(device)

        # If required, add "static" noise token at the end of the sequence
        if add_noise:
            
            # Build noise timed entity-attribute-value vector
            unkown_token_id = 4  # original [UNK] token id
            batch_size = ent.shape[0]
            noise_ent = torch.full((batch_size, 1), unkown_token_id, device=device)  # unrecognizable entity
            noise_attr = torch.full((batch_size, 1), unkown_token_id, device=device)  # unrecognizable attribute
            rand_indices = torch.randint(0, len(noise_value_pool), (batch_size, 1), device=device)
            noise_val = noise_value_pool[rand_indices]  # sample value from ["Lowest", "Lower", ..., "Highest"]
            noise_days = torch.zeros((batch_size, 1), device=device)  # time = 0 days post-transplant (static)

            # Concatenate to the end of the sequence
            ent = torch.cat([ent, noise_ent], dim=1)
            attr = torch.cat([attr, noise_attr], dim=1)
            val = torch.cat([val, noise_val], dim=1)
            days = torch.cat([days, noise_days], dim=1)

        args = (ent, attr, val, days)
        try:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                attributions, delta = lig.attribute(
                    inputs=args,
                    target=target_idx,
                    n_steps=20,
                    return_convergence_delta=True,
                    internal_batch_size=64,
                )
        except RuntimeError as e:
            print(f"Skipping batch due to error: {e}")
            continue

        # If layer reused (E+A+V), captum returns a tuple. Sum for total event attribution.
        if isinstance(attributions, tuple):
            combined_attr = torch.stack(attributions).sum(dim=0)
        else: 
            combined_attr = attributions

        attrs_sum = combined_attr.sum(dim=-1).detach().float().cpu().numpy()
        
        ent_np = args[0].cpu().numpy()
        attr_np = args[1].cpu().numpy()
        val_np = args[2].cpu().numpy()
        mask_np = (ent_np != 0).astype(int)

        for i in range(ent_np.shape[0]):
            length = int(mask_np[i].sum())
            seq_ent = ent_np[i, :length]
            seq_attr = attr_np[i, :length]
            seq_val = val_np[i, :length]
            seq_score = attrs_sum[i, :length]
            
            for k, (e_id, a_id, v_id, score) in enumerate(
                zip(seq_ent, seq_attr, seq_val, seq_score)
            ):
                # Noise token (appended at the end)
                if add_noise and (k == length - 1):
                    full_feature_name = ">> RANDOM NOISE <<"  # flag name!
                    val_name = id2word.get(v_id, f"Val_{v_id}")
                
                # Normal token
                else:
                    if e_id < 5: continue
                    ent_name = id2word.get(e_id, f"Ent_{e_id}")
                    if entity_filter is not None:
                        if not ent_name.startswith(entity_filter): continue
                    
                    attr_name = id2word.get(a_id, f"Attr_{a_id}")
                    full_feature_name = f"{ent_name} - {attr_name}"
                    val_name = id2word.get(v_id, f"Val_{v_id}")
                
                collected_data.append({
                    "Feature": full_feature_name,
                    "Value": val_name,
                    "Score": float(score)
                })

    df = pd.DataFrame(collected_data)
    df.to_csv(csv_path, index=False)
    return df


def plot_feature_importance(df, output_dir, label_name, top_k=20, min_freq=10, agg_method="mean"):
    """
    Generates the Bar Chart.
    agg_method: "mean" (conditional impact) or "sum" (population burden).
    """
    if df.empty: return
    csv_path = output_dir / f"drivers_bar_{label_name}_{agg_method}.csv"

    counts = df["Feature"].value_counts()
    valid_features = counts[counts >= min_freq].index
    filtered_df = df[df["Feature"].isin(valid_features)]
    
    # Conditional aggregation logic
    agg_func = "sum" if agg_method == "sum" else "mean"
    agg_col = "Cumulative_Attribution" if agg_method == "sum" else "Avg_Attribution"
    
    agg_df = filtered_df.groupby("Feature")["Score"].agg(agg_func).reset_index()
    agg_df.rename(columns={"Score": agg_col}, inplace=True)
    agg_df = agg_df.sort_values(agg_col, ascending=False)
    agg_df.head(50).to_csv(csv_path, index=False)
    
    plt.figure(figsize=(10, 8))
    top_df = agg_df.head(top_k)
    plt.barh(top_df["Feature"], top_df[agg_col], color="teal")
    
    title_suffix = "cumulative impact" if agg_method == "sum" else "average conditional impact"
    plt.title(f"Main model drivers ({title_suffix}): {label_name}")
    plt.xlabel(f"{agg_col.replace('_', ' ')}")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(output_dir / f"drivers_bar_{label_name}_{agg_method}.png", dpi=300)
    plt.close()


def plot_contrastive_drivers(
    df_pos, df_neg, output_dir, label_name, feature_list=None, top_k=20, min_freq=10, agg_method="mean"
):
    """
    Generates a butterfly bar chart contrasting positive vs negative cohorts.
    """
    # Global frequency filter
    all_features = pd.concat([df_pos["Feature"], df_neg["Feature"]])
    counts = all_features.value_counts()
    valid_features = counts[counts >= min_freq].index
    df_pos_filt = df_pos[df_pos["Feature"].isin(valid_features)]
    df_neg_filt = df_neg[df_neg["Feature"].isin(valid_features)]

    # Aggregation
    agg_func = "sum" if agg_method == "sum" else "mean"
    pos_agg = df_pos_filt.groupby("Feature")["Score"].agg(agg_func).reset_index()
    neg_agg = df_neg_filt.groupby("Feature")["Score"].agg(agg_func).reset_index()
    
    pos_agg.rename(columns={"Score": "Score_Pos"}, inplace=True)
    neg_agg.rename(columns={"Score": "Score_Neg"}, inplace=True)
    merged = pd.merge(pos_agg, neg_agg, on="Feature", how="outer").fillna(0)
    
    # Sorting / selection
    if feature_list is not None:
        top_df = merged[merged["Feature"].isin(feature_list)].copy()
        top_df["Diff"] = (top_df["Score_Pos"] - top_df["Score_Neg"]).abs()
        top_df = top_df.sort_values("Diff", ascending=True) 
    else:
        merged["Divergence"] = merged["Score_Pos"] - merged["Score_Neg"] 
        merged["Abs_Divergence"] = merged["Divergence"].abs()
        top_df = merged.sort_values("Abs_Divergence", ascending=False).head(top_k)
        top_df = top_df.sort_values("Abs_Divergence", ascending=True)

    plt.figure(figsize=(12, 8))
    y_indices = np.arange(len(top_df))
    bar_height = 0.25
    offset = 0.125
    
    plt.barh(
        y_indices + offset, top_df["Score_Pos"], height=bar_height, 
        color="#d62728", label="High risk cohort", alpha=0.8,
    )
    plt.barh(
        y_indices - offset, top_df["Score_Neg"], height=bar_height, 
        color="#1f77b4", label="Low risk cohort", alpha=0.8,
    )
    
    plt.axvline(0, color="black", linewidth=0.8)
    plt.yticks(y_indices, top_df["Feature"])
    
    title_suffix = "cumulative impact" if agg_method == "sum" else "mean attribution"
    plt.title(f"Contrastive feature importance ({title_suffix}): {label_name}\n(right = higher risk)")
    plt.xlabel(f"{title_suffix.capitalize()} score")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f"drivers_contrastive_{label_name}_{agg_method}.png")
    plt.close()


def plot_frequency_vs_impact(df, output_dir, label_name, min_freq=5):
    """
    Produces a volcano-style plot: frequency (log scale) vs. mean impact
    """
    if df.empty: return
    
    # Aggregation
    stats = df.groupby("Feature").agg(
        Frequency=("Score", "count"),
        Mean_Impact=("Score", "mean")
    ).reset_index()
    stats = stats[stats["Frequency"] >= min_freq]
    
    # Create a "Total Importance" metric for bubble size
    stats["Abs_Mean_Impact"] = stats["Mean_Impact"].abs()
    stats["Total_Burden"] = stats["Frequency"] * stats["Abs_Mean_Impact"]
    
    # Normalize bubble size for plotting
    size_min, size_max = 20, 300
    stats["Bubble_Size"] = (
        (stats["Total_Burden"] - stats["Total_Burden"].min()) / 
        (stats["Total_Burden"].max() - stats["Total_Burden"].min() + 1e-9)
    ) * (size_max - size_min) + size_min

    # Plot centered at 0 impact
    plt.figure(figsize=(14, 10))
    sns.scatterplot(
        data=stats, x="Frequency", y="Mean_Impact", size="Bubble_Size",
        sizes=(size_min, size_max), hue="Mean_Impact", palette="vlag",
        alpha=0.7, edgecolor="black", legend=False,
    )
    
    # Add quadrant lines
    med_f = stats["Frequency"].median()
    plt.axvline(med_f, color="gray", linestyle="--", alpha=0.3, label=f"Median Freq: {int(med_f)}")
    plt.axhline(0, color="black", linestyle="-", linewidth=0.8)
    
    # Take top by frequency, impact (pos/neg), and total burden
    top_f = stats.nlargest(10, "Frequency"); top_p = stats.nlargest(10, "Mean_Impact")
    top_n = stats.nsmallest(10, "Mean_Impact"); top_b = stats.nlargest(10, "Total_Burden")
    to_label = pd.concat([top_f, top_p, top_n, top_b]).drop_duplicates(subset="Feature")
    
    # Create the text objects (but don't plot them finally yet)
    texts = []
    for _, row in to_label.iterrows():
        texts.append(
            plt.text(
                row["Frequency"], row["Mean_Impact"], row["Feature"], 
                fontsize=9, weight="bold", color="black",
            )
        )

    # Use adjust_text to fix overlaps and draw lines
    adjust_text(
        texts,
        arrowprops=dict(arrowstyle='-', color='gray', lw=0.5, shrinkA=5, shrinkB=5),
        only_move={'text':'xy'}, autoalign='xy', lim=1000,
        expand_points=(1.05, 1.2),  # do not push away too hard
        force_text=(0.05, 0.1),     # do not let texts repel each other too hard
        force_points=(0.05, 0.1),   # do not let dots repel text too hard
    )
    plt.xscale("log")
    plt.title(f"Feature frequency vs. severity: {label_name}", fontsize=14)
    plt.xlabel("Frequency (log scale)", fontsize=12)
    plt.ylabel("Conditional mean attribution (severity)", fontsize=12)
    plt.grid(True, which="both", linestyle="--", alpha=0.2)
    plt.tight_layout()
    plt.savefig(output_dir / f"drivers_scatter_{label_name}.png", dpi=300)
    plt.close()


def plot_feature_value_impact(df, output_dir, label_name, feature_list=None, top_k=20, min_freq=10):
    """
    Generates a SHAP-like plot with expanded medication and event categories.
    """
    if df.empty: return
    csv_path = output_dir / f"drivers_shap_{label_name}.csv"

    # Data selection
    if feature_list is not None:
        top_features = feature_list[::-1] 
    else:
        counts = df["Feature"].value_counts()
        valid = counts[counts >= min_freq].index
        df_filtered = df[df["Feature"].isin(valid)]
        df_filtered["Abs_Score"] = df_filtered["Score"].abs()
        feature_importance = df_filtered.groupby("Feature")["Abs_Score"].mean().sort_values(ascending=False)
        top_features = feature_importance.head(top_k).index.tolist()
    
    df_plot = df[df["Feature"].isin(top_features)].copy()
    df_plot.to_csv(csv_path, index=False)

    # Merge all definitions
    group_definitions = {
        **EVENT_ROLES, **MED_ROLES, **CLINICAL_ROLES, 
        **NUMERIC_ROLES, **ORDINAL_ROLES, **UNKNOWN_ROLES,
    }
    for o in ORDINAL_ROLES: group_definitions[o] = [o]

    # Create reverse map
    val_to_group = {}
    for group_name, members in group_definitions.items():
        for m in members: val_to_group[m] = group_name

    # Check for remaining unknowns
    unique_vals = df_plot["Value"].unique()
    unknowns = [v for v in unique_vals if v not in val_to_group]
    if unknowns:
        print(f"\n[Info] Still unmapped values (will be plotted as 'Other'):")
        print(f"       {unknowns}")

    # Ordinal colors (Viridis)
    vir_cmap = plt.get_cmap("viridis", len(ORDINAL_ROLES))
    ord_colors = {lvl: mcolors.to_hex(vir_cmap(i)) for i, lvl in enumerate(ORDINAL_ROLES)}
    
    # Numeric colors (YlOrRd)
    heat_cmap = plt.get_cmap("RdYlGn")
    color_values = [0.0, 0.8, 0.9, 1.0]  # close to 0: red, close to 1: dark green
    num_colors = {
        k: mcolors.to_hex(heat_cmap(color_values[i]))
        for i, k in enumerate(NUMERIC_ROLES)
    }

    # Categorical: re-use palette for meds vs events since shapes will differ
    cat_colors = {}
    base_palette = (plt.get_cmap("tab20").colors + plt.get_cmap("tab20b").colors)
    base_hex = [mcolors.to_hex(c) for c in base_palette]
        
    # Assign colors to medications (start index at 0)
    med_keys = list(MED_ROLES.keys())
    for i, key in enumerate(med_keys):
        cat_colors[key] = base_hex[i % len(base_hex)]

    # Assign colors to events/infections (restart index at 0 to re-use colours)
    event_keys = list(EVENT_ROLES.keys())
    for i, key in enumerate(event_keys):
        cat_colors[key] = base_hex[i % len(base_hex)]

    # Build master palette and size maps
    master_palette = {}
    for val in unique_vals:
        group = val_to_group.get(val, "Other/Unknown")
        
        if group in ord_colors:
            master_palette[val] = ord_colors[group]
        elif group in num_colors:
            master_palette[val] = num_colors[group]
        elif group in cat_colors:
            master_palette[val] = cat_colors[group]
        else:
            master_palette[val] = "#333333"  # dark grey for other

    def get_value_type(val):
        group = val_to_group.get(val, "Other/Unknown")
        if val in ORDINAL_ROLES: return "Ordinal"
        if group in num_colors: return "Numeric/Bool"
        if group in MED_ROLES: return "Medication"
        if group in EVENT_ROLES: return "Categorical"
        # if group in CLINICAL_ROLES: return "Clinical"
        return "Other"
    
    df_plot["Value_Type"] = df_plot["Value"].apply(get_value_type)
    markers_map = {
        "Ordinal": "o", "Numeric/Bool": "D",
        "Medication": "P", "Categorical": "s", "Other": "X",
    }
    sizes_map = {
        "Ordinal": 60, "Numeric/Bool": 50,
        "Medication": 85, "Categorical": 45, "Other": 70,
    }

    # Plotting
    feature_map = {name: i for i, name in enumerate(top_features)}
    df_plot["Feature_Y"] = df_plot["Feature"].map(feature_map)
    rng = np.random.default_rng(1234)
    df_plot["Y_Jittered"] = df_plot["Feature_Y"] + rng.uniform(-0.2, 0.2, size=len(df_plot))

    plt.figure(figsize=(14, 12))  # slightly taller for larger legend
    sns.scatterplot(
        data=df_plot, x="Score", y="Y_Jittered",
        hue="Value", style="Value_Type", size="Value_Type",
        sizes=sizes_map, markers=markers_map, palette=master_palette,
        alpha=0.6, edgecolor="white", linewidth=0.5, legend=False,
    )

    for y in range(len(top_features)):
        plt.axhline(y, color='gray', linestyle='-', linewidth=0.3, alpha=0.1)
    plt.axvline(0, color="black", linestyle="-", alpha=0.4)
    plt.yticks(ticks=range(len(top_features)), labels=top_features)
    plt.title(f"Feature value impact: {label_name}", fontsize=14)
    plt.xlabel("Attribution score (right = higher risk)")

    # Legend construction
    present_vals = set(df_plot["Value"].unique())
    present_groups = set()
    for v in present_vals:
        present_groups.add(val_to_group.get(v, "Other/Unknown"))

    # Add groups in the order they appear in the group definitions
    sorted_groups = []
    all_defined_keys = list(num_colors.keys()) + list(cat_colors.keys()) + list(ord_colors.keys())
    for g in all_defined_keys:
        if g in present_groups: sorted_groups.append(g)
    if "Other/Unknown" in present_groups: sorted_groups.append("Other/Unknown")

    legend_handles = []
    for group_name in sorted_groups:
        if group_name in num_colors: c = num_colors[group_name]
        elif group_name in ord_colors: c = ord_colors[group_name]
        elif group_name in cat_colors: c = cat_colors[group_name]
        else: c = "#333333"
        
        rep_val = group_definitions.get(group_name, ["Other"])[0]
        v_type = get_value_type(rep_val)
        m = markers_map.get(v_type, "X")
        s = sizes_map.get(v_type, 50) / 6.0 

        handle = Line2D(
            [0], [0], marker=m, color='w', markerfacecolor=c, 
            label=group_name, markersize=s, markeredgecolor='white'
        )
        legend_handles.append(handle)

    plt.legend(
        handles=legend_handles, 
        bbox_to_anchor=(1.02, 1), 
        loc='upper left', 
        title="Feature Values",
        frameon=False,
        ncol=1,
    )
    plt.tight_layout()
    plt.savefig(output_dir / f"drivers_shap_{label_name}.png", dpi=300)
    plt.close()
    

def run_captum_analysis(
    model, dataset, collator, vocab, device, output_dir, label_name,
    pos_indices, neg_indices, target_idx=0, entity_filter=None, add_noise=False,
    top_k=20, plot_only=None, agg_method="mean", min_freq=10,
):
    """
    Orchestrates the analysis and plotting pipeline.
    """
    print(f" -> Extracting attributions for {len(pos_indices)} positive samples...")
    csv_path = output_dir / f"attributions_pos.csv"
    df_pos = extract_attributions(
        model, dataset, collator, pos_indices, vocab, device,
        target_idx=target_idx, entity_filter=entity_filter,
        add_noise=add_noise, csv_path=csv_path, plot_only=plot_only,
    )
    df_pos["Cohort"] = "High risk"

    print(f" -> Extracting attributions for {len(neg_indices)} negative samples...")
    csv_path = output_dir / f"attributions_neg.csv"
    df_neg = extract_attributions(
        model, dataset, collator, neg_indices, vocab, device,
        target_idx=target_idx, entity_filter=entity_filter,
        add_noise=add_noise, csv_path=csv_path, plot_only=plot_only,
    )
    df_neg["Cohort"] = "Low risk"
    
    print(f"\n>>> Analyzing drivers (aggregation: {agg_method}, min freq: {min_freq})...")

    # Global frequency filtering (filter before ranking)
    all_features = pd.concat([df_pos["Feature"], df_neg["Feature"]])
    counts = all_features.value_counts()
    valid_features = counts[counts >= min_freq].index.tolist()
    
    df_pos_filt = df_pos[df_pos["Feature"].isin(valid_features)]
    df_neg_filt = df_neg[df_neg["Feature"].isin(valid_features)]

    # Ranking by chosen method (mean or sum)
    agg_func = "sum" if agg_method == "sum" else "mean"
    pos_scores = df_pos_filt.groupby("Feature")["Score"].agg(agg_func)
    neg_scores = df_neg_filt.groupby("Feature")["Score"].agg(agg_func)
    
    # Compare cohorts to find most divergent drivers
    combined = pd.DataFrame({"Pos": pos_scores, "Neg": neg_scores}).fillna(0)
    combined["Abs_Diff"] = (combined["Pos"] - combined["Neg"]).abs()
    shared_features = combined.sort_values("Abs_Diff", ascending=False).head(top_k).index.tolist()
    
    # Generate plots
    print(" -> Generating contrastive bar chart...")
    plot_contrastive_drivers(
        df_pos, df_neg, output_dir, label_name, 
        feature_list=shared_features, agg_method=agg_method
    )
    print(" -> Generating value impact strip chart (High Risk)...")
    plot_feature_value_impact(
        df_pos, output_dir, f"{label_name} - high risk", 
        feature_list=shared_features, min_freq=min_freq
    )
    print(" -> Generating volcano scatter plot...")
    plot_frequency_vs_impact(df_pos, output_dir, f"{label_name} - high risk", min_freq=min_freq)


def compute_feature_enrichment(target_indices, background_indices, dataset, top_k=20):
    """
    Fisher exact test to find features over-represented in the target group vs background.
    """
    print(f"   -> Analyzing features for {len(target_indices)} vs {len(background_indices)} samples...")
    
    def get_features(indices):
        features = []
        for idx in indices:
            sample = dataset[int(idx)]
            f_set = set([f"{a}_{v}" for a, v in zip(sample["attribute"], sample["value_binned"])])
            features.extend(list(f_set))
        return features

    target_counts = pd.Series(get_features(target_indices)).value_counts()
    bg_counts = pd.Series(get_features(background_indices)).value_counts()
    
    n_target = len(target_indices)
    n_bg = len(background_indices)
    results = []
    
    candidates = target_counts[target_counts > (n_target * 0.05)].index 
    
    for feat_key in candidates:
        a = target_counts.get(feat_key, 0)
        c = bg_counts.get(feat_key, 0)
        odds, p_val = fisher_exact([[a, n_target - a], [c, n_bg - c]], alternative='greater')
        
        try:
            a_id, v_id = map(int, feat_key.split("_"))
            feat_name = f"Attr({a_id}) - Val({v_id})" 
        except:
            feat_name = feat_key

        results.append({
            "Feature": feat_name,
            "Target_%": (a / n_target) * 100,
            "Background_%": (c / n_bg) * 100,
            "Odds_Ratio": odds,
            "P_Value": p_val
        })
        
    return pd.DataFrame(results).sort_values("Odds_Ratio", ascending=False).head(top_k)


def analyze_near_misses_by_risk(probs, labels, output_dir):
    """
    Identifies healthy patients with high predicted risk (False Positives / Near Misses).
    """
    # Ensure data type and shape
    if isinstance(labels, torch.Tensor): labels = labels.cpu().numpy()
    if isinstance(probs, torch.Tensor): probs = probs.cpu().numpy()
    if labels.ndim > 1: labels = labels.flatten()
    if probs.ndim > 1: probs = probs.flatten()
    
    is_healthy = (labels == 0)
    healthy_indices = np.where(is_healthy)[0]
    if len(healthy_indices) < 30:
        print("   -> Not enough healthy samples.")
        return None, None

    healthy_probs = probs[healthy_indices]    
    high_thresh = np.percentile(healthy_probs, 90)
    low_thresh = np.percentile(healthy_probs, 50)
    
    near_miss_idx = healthy_indices[healthy_probs >= high_thresh]
    safe_idx = healthy_indices[healthy_probs <= low_thresh]
    
    plt.figure(figsize=(10, 6))
    plt.hist(healthy_probs, bins=30, alpha=0.5, color="blue", label="Healthy (Label 0)", density=True)
    infected_probs = probs[labels == 1]
    if len(infected_probs) > 0:
        plt.hist(infected_probs, bins=30, alpha=0.5, color="red", label="Infected (Label 1)", density=True)
    
    plt.axvline(high_thresh, color='orange', linestyle='--', label='Near-miss threshold')
    plt.title("Risk distribution (Near-miss analysis)")
    plt.legend()
    plt.savefig(output_dir / "risk_distribution_near_miss.png", dpi=300)
    plt.close()
    
    return near_miss_idx, safe_idx


def main():
    # Load task configuration
    args = parse_args()
    base_dir = TASK_CONFIG[args.task_key]["base_dir"]
    hrz = TASK_CONFIG[args.task_key]["horizon"]
    fups = TASK_CONFIG[args.task_key]["follow-up_periods"]
    
    # Setup input model and output directories
    checkpoint_path = find_best_checkpoint(Path(base_dir), args.task_key, hrz)
    print(f"\n>>> Using Checkpoint: {checkpoint_path}")
    with open(args.config, 'r') as f: config = yaml.safe_load(f)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup keys for task of interest
    ckpt_hrzs = extract_horizons_from_path(checkpoint_path)
    if hrz not in ckpt_hrzs:
        raise ValueError(f"Requested horizon {hrz} not found in: {ckpt_hrzs}")
    target_idx = ckpt_hrzs.index(hrz)
    ckpt_label_keys = [f"label_{args.task_key}_{h:04d}d" for h in ckpt_hrzs]
    label_key = ckpt_label_keys[target_idx]
    print(f"\n>>> Target index: {target_idx} (horizon: {hrz}d, available: {ckpt_hrzs})")

    # Load dataset
    print("\n>>> Loading dataset...")
    data_dir = Path(config["hf_data_dir"])
    print(f"Follow-up considered in the analysis: {fups}")
    split_considered = "test"
    dataset, _, vocab = load_hf_data_and_metadata(
        data_dir=data_dir,
        fup_train=[365], fup_valid=[365], fup_test=fups,
        time_mapping=config["data_collator"]["time_mapping"],
        eav_mappings=config["data_collator"]["eav_mappings"],
        sanity_check_output_dir = "./results_interpretability/sanity_check",
    )
    test_ds = dataset[split_considered].filter(
        lambda x: x[label_key] != -100, num_proc=SAFE_NUM_PROC
    )
    test_ds = test_ds.map(
        lambda x: {"split": split_considered}, num_proc=SAFE_NUM_PROC
    )

    # Load model and data collator for test-set inference
    emb_cfg = config["model"]["embedding_layer_config"]
    emb_cfg["vocab_size"] = len(vocab)
    model = PatientEmbeddingModelFactory.from_pretrained(
        task="classification",
        pretrained_dir=str(checkpoint_path),
        embedding_layer_config=emb_cfg,
        model_args=config["model"]["model_args"]
    )
    collator = PatientDataCollatorForClassification(
        **config["data_collator"], 
        label_keys=ckpt_label_keys,
        max_position_embeddings=model.config.max_position_embeddings
    )

    # Run model inference and extract output probabilities
    print("\n>>> Running inference on test set...")
    interpreter = ModelInterpreter(model)
    loader = DataLoader(test_ds, batch_size=32, collate_fn=collator.torch_call)
    res = interpreter.get_embeddings_and_predictions(loader)
    probs = expit(res["logits"][:, target_idx])
    target_labels = res["labels"][:, target_idx]

    # Select positive and negative cohorts (according to model)
    model_strength = 0.9  # 0.5 for considering all samples
    q_high = np.quantile(probs, model_strength)
    q_low = np.quantile(probs, 1.0 - model_strength)
    pos_idx = np.where(probs >= q_high)[0] 
    neg_idx = np.where(probs <= q_low)[0] 
    print(f"\nCohort selection (extreme quantiles):")
    print(f" -> High-risk threshold: >= {q_high:.4f} | {len(pos_idx)} samples")
    print(f" -> Low-risk threshold: <= {q_low:.4f} | {len(neg_idx)} samples")
    if len(pos_idx) > args.captum_samples:
        pos_idx = np.random.choice(pos_idx, args.captum_samples, replace=False)
    if len(neg_idx) > args.captum_samples:
        neg_idx = np.random.choice(neg_idx, args.captum_samples, replace=False)

    # Run captum analysis (feature attribution using integrated gradients)
    label_name = label_key.replace("label_", "").replace("_", " ").capitalize()
    if len(pos_idx) > 0 and len(neg_idx) > 0:
        print("\n>>> Running captum analysis...")
        run_captum_analysis(
            model=model, dataset=test_ds, collator=collator, output_dir=output_dir,
            label_name=label_name, pos_indices=pos_idx, neg_indices=neg_idx,
            vocab=vocab, device="cuda", target_idx=target_idx, entity_filter=None,
            add_noise=args.add_noise, top_k=args.top_k, plot_only=args.plot_only,
            agg_method=args.agg_method, min_freq=args.min_freq,
        )

    # # Run clustering analysis
    # print("\n>>> Clustering with UMAP and HDBSCAN...")
    # clusterer = UMAP_HDBSCAN_Clusterer(n_optuna_trials=25)
    # _, fig = clusterer.perform_analysis(res["embeddings"], max_samples=5000)
    # if fig: fig.write_html(str(output_dir / "clusters_umap.html"))
    # _, cluster_ids = clusterer.fit_predict(res["embeddings"])
    # for cid in np.unique(cluster_ids):
    #     if cid == -1: continue
    #     idx = np.where(cluster_ids == cid)[0]
    #     bg_idx = np.where(cluster_ids != cid)[0]
    #     rate = target_labels[idx].mean()    
    #     enrich = compute_feature_enrichment(idx, bg_idx, test_ds, top_k=10)
    #     print(f"\n[Cluster {cid}] Size: {len(idx)} | Pos Rate: {rate:.2%}")
    #     print(enrich[["Feature", "Odds_Ratio"]].to_string(index=False))
    #     enrich.to_csv(output_dir / f"enrichment_cluster_{cid}.csv", index=False)

    # # Near miss analysis
    # print("\n>>> Analyzing near-misses (risk-based)...")
    # nm_idx, safe_idx = analyze_near_misses_by_risk(probs, target_labels, output_dir)
    # if nm_idx is not None:
    #     enrich = compute_feature_enrichment(nm_idx, safe_idx, test_ds, top_k=15)
    #     print("\n>>> Near-Miss Enrichment:")
    #     print(enrich.to_string(index=False))
    #     enrich.to_csv(output_dir / "near_miss_enrichment.csv", index=False)

    # Success!
    print(f"\nSuccess! All results can be found in {output_dir}")


if __name__ == "__main__":
    main()