import re
import argparse
import yaml
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.transforms as transforms
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from tqdm.auto import tqdm
from adjustText import adjust_text
from scipy.stats import fisher_exact
from torch.utils.data import DataLoader, Subset
from captum.attr import LayerIntegratedGradients

from src.data.patient_dataset import load_hf_data_and_metadata
from src.model.patient_embedder import PatientEmbeddingModelFactory, PatientDataCollatorForClassification

CLI_CFG = {}
SAFE_NUM_PROC = 4
GENERATE_SANITIZE_PLOTS = False
DEFAULT_TASK_KEY = "infection_bacteria"
TASK_CONFIG = {
    "infection_bacteria": {
        "base_dir": "results_monotonicity_yes/e05-a15-v35",
        "horizon": 30,
        "fup_min": 360,
        "fup_max": 720,
        "fup_step": 30,
    },
    # "infection_virus": {
    #     "base_dir": "results_monotonicity_yes/e05-a15-v35",
    #     "horizon": 90,
    #     "fup_min": 0,
    #     "fup_max": 360,
    #     "fup_step": 60,
    # },
    # "death": {
    #     "base_dir": "results_monotonicity_yes/e05-a15-v35",
    #     "horizon": 1080,
    #     "fup_min": 0,
    #     "fup_max": 1440,
    #     "fup_step": 180,
    # },
    # "graft_loss": {
    #     "base_dir": "results_monotonicity_yes/e05-a15-v35",
    #     "horizon": 1080,
    #     "fup_min": 0,
    #     "fup_max": 1440,
    #     "fup_step": 180,
    # },
}

# =================================
# GLOBAL MAPPINGS AND CONFIGURATION
# =================================

# Medications
MED_ROLES = {
    'Med. CNI':                     ['Tacrolimus', 'Cyclosporine'],
    'Med. antimetabolite':          ['Mycophenolate Mofetil', 'Mycophenolic Acid', 'Azathioprine'],
    'Med. mTOR inhibitor':          ['Everolimus', 'Sirolimus'],
    'Med. steroids':                ['Glucocorticoid', 'Methylprednisolone', 'Prednisone'],
    'Med. induction':               ['Basiliximab', 'Rabbit ATG', 'Anti-thymocyte globulin'],
    'Med. rejection treat.':        ['Rituximab', 'IVIG', 'Human Immunoglobulin', 'Plasmapheresis'],
    'Med. antiviral proph.':        ['Valganciclovir', 'Valaciclovir', 'Lamivudine', 'Ganciclovir', 'Entecavir'],
    'Med. antibiotic proph.':       ['Cotrimoxazole', 'Atovaquone', 'Pentamidine', 'Dapsone'],
    'Med. antibiotic treat.':       ['Beta-Lactame', 'Quinolone', 'Cephalosporin', 'Metronidazole', 'Fosfomycin', 'Nitrofurantoin', 'Clarithromycin', 'Isoniazid'],
    'Med. antifungal treat.':       ['Amphotericin B', 'Itraconazole', 'Fluconazole', 'Voriconazole'],
    'Med. antihypertensive':        ['Calcium channel blocker', 'Beta-blocker', 'ACE inhibitor', 'Angiotensin receptor blocker'],
    'Med. antithrombotic':          ['Platelet aggregation inhibitor', 'Anticoagulation therapy'],
    'Med. diabetes treat.':         ['Insulin', 'Oral antidiabetic drug'],
    'Med. diuretic':                ['Torasemide', 'Furosemide'],
    'Med. lipid lowering':          ['Statin'],
    'Med. other':                   ['Other drugs']
}

# Procedure event
INFECTION_ROLES = {
    # Infection categories
    'Bacteria':                     ['Bacterial', 'Bacteria'],
    'Virus':                        ['Viral', 'Virus'],
    'Fungal':                       ['Fungal', 'Fungi'],
    'Parasite':                     ['Parasite'],
    
    # Infection pathogens
    'Bacteria enteric':             ['E. coli', 'Klebsiella sp', 'Enterobacter', 'Other enterobacteria', 'Enterococcus', 'Other non-enteric GN bacteria', 'Clostridium sp', 'CDI'],
    'Bacteria resp./oral':          ['Pneumococcus', 'Streptococcus sp', 'Haemophilus influenzae', 'Legionella', 'Streptococcus  sp', 'Actinomyces'],
    'Bacteria skin/staph':          ['Staph aureus', 'MSSA', 'St. coagulase negative'],
    'Bacteria hospital':            ['Pseudomonas aeruginosa', 'Acinetobacter', 'Stenotrophomonas'],
    'Virus herpes group':           ['CMV', 'EBV', 'VZV', 'HSV'],
    'Virus respiratory':            ['SARS-CoV-2', 'Influenza', 'Rhinovirus', 'Metapneumovirus', 'Parainfluenza', 'Adenovirus', 'RSV'],
    'Virus hepatitis':              ['HBV', 'HCV', 'Noro', 'BKV', 'JCV', 'Parvo B19'],
    'Fungal yeast':                 ['Candida albicans', 'Candida non albicans'],
    'Fungal mold':                  ['Aspergillus non-fumigatus', 'Zygomycetes', 'Pneumocystis sp'],
    'Parasite':                     ['Cryptosporidium sp', 'Toxo', 'Other parasites'],

    # Infection types    
    'Proven':                       ['Proven disease', 'Primary infection'],
    'Possible':                     ['Possible disease', 'Probable disease'],
    'Viral syndrome':               ['Viral syndrome'],
    'Colonization':                 ['Colonization'],
    'Asymptomatic':                 ['Asymptomatic'],
    
    # Infection sites
    'UTI':                          ['Urinary tract', 'UTI'],
    'Resp':                         ['Respiratory', 'Lung', 'Pneumonia', 'RT'],
    'GI':                           ['Gastrointestinal', 'Abdominal', 'GI'],
    'Blood/Sepsis':                 ['Blood', 'Sepsis', 'Bacteremia', 'Catheter'],
    'Skin/Wound':                   ['Skin', 'Mucocutaneous', 'SSI'],
    'CNS':                          ['CNS'],
    'Eye':                          ['Eye'],
    'Liver':                        ['Liver'],
    'Kidney':                       ['Kidney'],
    'Heart':                        ['Heart'],
    'Bone joint':                   ['Bone_Joint'],
}

# Clinical events
CLINICAL_ROLES = {
    # Transplant-related events
    'Rejection event':              ['Biops proven rj', 'Clinically suspected rj', 'Clinical', 'Subclinical', 'SAR'],
    'Transplant procedure':         ['Kidney tpx', 'Heart tpx', 'Lung tpx', 'HSCT allo', 'HSCT auto'],
    'Surgery':                      ['Nephrectomy native', 'Nephrectomy allograft', 'Nephrectomy allograft and native', 'Non-Tx surgery'],
    'Pregnancy/birth':              ['Birth', 'Pregnancy', 'Abortion/miscarriage'],
    'Emergency/critical event':     ['MOF', 'Agranulocytosis', 'GI haemorrhage', 'Bone fracture'],
    'Previous graft failure':       ['Previous GF'],

    # Kidney disease diagnoses
    'Kidney dis. GN/nephritis':     ['GN', 'Interstitial nephritis', 'Reflux/Pyelonephritis'],
    'Kidney dis. PCKD':             ['PCKD'],
    'Kidney dis. hereditary':       ['Congenital kidney', 'Hereditary non_PCKD'],
    'Kidney dis. vascular/DM':      ['Nephrosclerosis', 'DM nephropathy'],
    'Kidney dis. failure':          ['CKD', 'ARF', 'Acute on chronic RF', 'ATN'],
    'Kidney dis. toxicity':         ['CNI nephrotoxicity'],
    'Kidney dis. other':            ['OTH', 'CTR', 'CTU'],  # not sure what these mean
    
    # Malignancy diagnoses
    'Cancer kidney':                ['Kidney cancer'],
    'Cancer skin':                  ['Melanoma', 'Spinalioma', 'Basalioma', 'Other skin cancer'],
    'Cancer GI':                    ['Colorectal cancer', 'Liver cancer', 'HCC'],
    'Cancer UTI':                   ['Uro_bladder cancer', 'Prostate cancer', 'Testicular cancer'],
    'Cancer gyneco':                ['Breast cancer', 'Cervix - Uterus - Adnex ca'],
    'Cancer lung':                  ['Lung cancer'],
    'Cancer endocrine':             ['Thyroid cancer', 'Neuroendocrine TU'],
    'Cancer sarkoma':               ['Sarkoma'],
    'Cancer blood/lymph':           ['PTLD', 'Myeloid neoplasm', 'Leukemia', 'Lymphoma'],
    
    # Other diagnoses
    'Comorb. diabetes':             ['DM type1', 'DM type2 treated', 'PTDM'],
    'Comorb. cardiac risk factors': ['HTN', 'Hyperlipidemia'],
    'Comorb. vascular':             ['CAD', 'PAD', 'CVD'],
    'Comorb. heart failure':        ['CHE', 'LVEF < 30%', 'Dilated CMP', 'HFpEF (Symptomatic Heart failure with preserved LVEF)'],
    'Comorb. arrhythmias':          ['AF', 'VT_PE', 'Pacemaker, ICD'],
    'Comorb. valvular':             ['Cardiac valvular disease'],
    'Comorb. respiratory':          ['COPD'],
    'Comorb. HIV':                  ['HIV'],
    'Comorb. liver/hep':            ['Hep C', 'Hep B', 'Drug-induced liver injury'],
    'Comorb. other':                ['Osteoporosis', 'Other metabolic', 'Alcohol', 'ANX', 'SUI'],
    
    # Donor-related info
    'Donor deceased':               ['DBD', 'DCD'],
    'Donor living':                 ['Living related', 'Living unrelated'],
    'Serology D+/R+':               ['D+/R+'],
    'Serology D+/R-':               ['D+/R-'],
    'Serology D-/R+':               ['D-/R+'],
    'Serology D-/R-':               ['D-/R-'],
    'Serology other':               ['D+/R?', 'D-/R?', 'D?/R+', 'D?/R?'],
}

# Numeric / boolean
NUMERIC_ROLES = {
    'No / Male':                    ['0', 'No', 'False', 'Negative', 'Male', 'M'], 
    'Yes / Female / Occured':       ['1', 'Yes', 'True', 'Occurred', 'Positive', 'Female', 'F'],
    'Value 2':                      ['2'], 
    'Value 3':                      ['3'],
}

# Ordinal (intensity)
ORDINAL_LEVELS_LIST = ['Below', 'Lowest', 'Lower', 'Low', 'Middle', 'High', 'Higher', 'Highest', 'Measurable']
ORDINAL_ROLES = {'Ordinal level': ORDINAL_LEVELS_LIST}

# Unknowns
UNKNOWN_ROLES = {
    'Other/Unknown': [
        'Unknown', '[UNK]', 'Condition unknown',
        'Other', 'Other event or disease',
        'Missing', 'Site not identified', 'Undetermined'
    ]
}


def parse_args():
    parser = argparse.ArgumentParser(description="Interpret model predictions and embeddings on the test set.")
    parser.add_argument("--config", "-c", type=str, default="configs/discriminative_training.yaml", help="Path to the training config used.")
    parser.add_argument("--output_dir", "-o", type=str, default="results_interpretability", help="Folder to save plots and tables.")
    parser.add_argument("--task_key", "-t", type=str, default=DEFAULT_TASK_KEY, help="Task key.")
    parser.add_argument("--plot_only", "-p", action="store_true", help="Skip computation, load existing CSVs and replot.")
    parser.add_argument("--captum_samples", type=int, default=1000, help="Number of samples to analyze (randomly selected from test set).")
    parser.add_argument("--max_delta", type=float, default=0.25, help="Maximum allowed convergence delta for attributions. Drops samples above this.")
    parser.add_argument("--add_noise", action="store_true", help="Inject a random noise feature for baseline comparison.")
    parser.add_argument("--agg_method", type=str, choices=["mean", "sum"], default="mean", help="Aggregation method: 'mean' (conditional impact) or 'sum' (population burden).")
    parser.add_argument("--top_k", type=int, default=20, help="Maximum number of features plotted.")
    parser.add_argument("--min_freq", type=int, default=20, help="Minimum number of patients a feature must appear in to be plotted.")

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
    
    return checkpoint_dirs[0]


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
    entity_filter=None, max_delta=None, add_noise=False,
    csv_path=None, plot_only=False,
) -> pd.DataFrame:
    
    if plot_only and csv_path.exists():
        return pd.read_csv(csv_path)

    # Find layer from which attribution will be derived
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
        ordinal_keys = ORDINAL_LEVELS_LIST # Use the list defined above
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
            unkown_token_id = 4 
            batch_size = ent.shape[0]
            noise_ent = torch.full((batch_size, 1), unkown_token_id, device=device) 
            noise_attr = torch.full((batch_size, 1), unkown_token_id, device=device) 
            rand_indices = torch.randint(0, len(noise_value_pool), (batch_size, 1), device=device)
            noise_val = noise_value_pool[rand_indices] 
            noise_days = torch.zeros((batch_size, 1), device=device) 

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
                    n_steps=100,
                    return_convergence_delta=True,
                    internal_batch_size=64,
                )
        except RuntimeError as e:
            print(f"Skipping batch due to error: {e}")
            continue

        if isinstance(attributions, tuple):
            combined_attr = torch.stack(attributions).sum(dim=0)
        else: 
            combined_attr = attributions

        attrs_sum = combined_attr.sum(dim=-1).detach().float().cpu().numpy()
        
        ent_np = args[0].cpu().numpy()
        attr_np = args[1].cpu().numpy()
        val_np = args[2].cpu().numpy()
        days_np = args[3].cpu().numpy()
        mask_np = (ent_np != 0).astype(int)
        delta_np = delta.detach().cpu().numpy()

        for i in range(ent_np.shape[0]):
            length = int(mask_np[i].sum())
            seq_ent = ent_np[i, :length]
            seq_attr = attr_np[i, :length]
            seq_val = val_np[i, :length]
            seq_score = attrs_sum[i, :length]
            seq_days = days_np[i, :length]
            patient_delta = float(abs(delta_np[i]))
            
            for k, (e_id, a_id, v_id, score, day) in enumerate(
                zip(seq_ent, seq_attr, seq_val, seq_score, seq_days)
            ):
                if add_noise and (k == length - 1):
                    full_feature_name = ">> RANDOM NOISE <<" 
                    val_name = id2word.get(v_id, f"Val_{v_id}")
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
                    "Score": float(score),
                    "Delta": patient_delta,
                    "Days": float(day)
                })

    df = pd.DataFrame(collected_data)
    if csv_path:
        df.to_csv(csv_path, index=False)
    return df


def plot_feature_importance(df, output_dir, label_name, top_k=20, min_freq=50, agg_method="mean"):
    """
    Generates a bidirectional Bar Chart.
    Red bars = Risk factors (positive score), Blue bars = Protective factors (negative score).
    """
    if df.empty: return
    csv_path = output_dir / f"drivers_bar_{label_name}_{agg_method}.csv"

    counts = df["Feature"].value_counts()
    valid_features = counts[counts >= min_freq].index
    if len(valid_features) == 0:
        print(f"      [!] Skipping Bar Chart: No features met min_freq={min_freq}.")
        return
    filtered_df = df[df["Feature"].isin(valid_features)]
    
    agg_func = "sum" if agg_method == "sum" else "mean"
    agg_col = "Attribution"
    
    agg_df = filtered_df.groupby("Feature")["Score"].agg(agg_func).reset_index()
    agg_df.rename(columns={"Score": agg_col}, inplace=True)
    
    # Calculate absolute importance for sorting
    agg_df["Abs_Attribution"] = agg_df[agg_col].abs()
    agg_df = agg_df.sort_values("Abs_Attribution", ascending=False)
    
    agg_df.head(50).to_csv(csv_path, index=False)
    
    plt.figure(figsize=(12, 10))
    top_df = agg_df.head(top_k).iloc[::-1] # Reverse for plotting top at top
    
    colors = ['#d62728' if x > 0 else '#1f77b4' for x in top_df[agg_col]]
    
    plt.barh(top_df["Feature"], top_df[agg_col], color=colors, alpha=0.8)
    
    plt.axvline(0, color="black", linewidth=0.8)
    title_suffix = "Cumulative Impact" if agg_method == "sum" else "Average Impact"
    plt.title(f"Top {top_k} Model Drivers: {label_name}\n(Red = Increases Risk, Blue = Reduces Risk)")
    plt.xlabel(f"{title_suffix} (Attribution Score)")
    plt.tight_layout()
    plt.savefig(output_dir / f"drivers_bar_{label_name}.png", dpi=300)
    plt.close()


def plot_frequency_vs_impact(df, output_dir, label_name, min_freq=50):
    """
    Produces a volcano-style plot: frequency (log scale) vs. mean impact
    """
    if df.empty: return
    
    stats = df.groupby("Feature").agg(
        Frequency=("Score", "count"),
        Mean_Impact=("Score", "mean")
    ).reset_index()
    stats = stats[stats["Frequency"] >= min_freq]
    if stats.empty:
        print(f"      [!] Skipping Volcano Plot: No features met min_freq={min_freq}.")
        return
    
    stats["Abs_Mean_Impact"] = stats["Mean_Impact"].abs()
    stats["Total_Burden"] = stats["Frequency"] * stats["Abs_Mean_Impact"]
    
    size_min, size_max = 20, 300
    stats["Bubble_Size"] = (
        (stats["Total_Burden"] - stats["Total_Burden"].min()) / 
        (stats["Total_Burden"].max() - stats["Total_Burden"].min() + 1e-9)
    ) * (size_max - size_min) + size_min

    plt.figure(figsize=(14, 10))
    sns.scatterplot(
        data=stats, x="Frequency", y="Mean_Impact", size="Bubble_Size",
        sizes=(size_min, size_max), hue="Mean_Impact", palette="vlag",
        alpha=0.7, edgecolor="black", legend=False,
    )
    
    med_f = stats["Frequency"].median()
    plt.axvline(med_f, color="gray", linestyle="--", alpha=0.3, label=f"Median Freq: {int(med_f)}")
    plt.axhline(0, color="black", linestyle="-", linewidth=0.8)
    
    # Label top features
    top_f = stats.nlargest(8, "Frequency"); top_p = stats.nlargest(8, "Mean_Impact")
    top_n = stats.nsmallest(8, "Mean_Impact"); top_b = stats.nlargest(8, "Total_Burden")
    to_label = pd.concat([top_f, top_p, top_n, top_b]).drop_duplicates(subset="Feature")
    
    texts = []
    for _, row in to_label.iterrows():
        texts.append(
            plt.text(
                row["Frequency"], row["Mean_Impact"], row["Feature"], 
                fontsize=9, weight="bold", color="black",
            )
        )

    adjust_text(
        texts,
        arrowprops=dict(arrowstyle='-', color='gray', lw=0.5, shrinkA=5, shrinkB=5),
        only_move={'text':'xy'}, autoalign='xy', lim=1000,
    )
    plt.xscale("log")
    plt.title(f"Frequency vs. Severity: {label_name}\n(Above 0 = Risk, Below 0 = Protective)", fontsize=14)
    plt.xlabel("Frequency (log scale)", fontsize=12)
    plt.ylabel("Conditional Mean Attribution", fontsize=12)
    plt.grid(True, which="both", linestyle="--", alpha=0.2)
    plt.tight_layout()
    plt.savefig(output_dir / f"drivers_volcano_{label_name}.png", dpi=300)
    plt.close()

def plot_feature_value_impact(df, output_dir, label_name, fup_max, max_delta=None, top_k=20, min_freq=50):
    """
    Generates a SHAP-like strip plot with aligned per-row legends on the left.
    Dynamic: Adjusts row height to prevent overlap and removes excess whitespace.
    Coloring: Unified palette for all categorical roles, specific palettes for numeric/ordinal.
    Grouping: Values are aggregated by Group in the legend to avoid clutter.
    """
    if df.empty: return
    csv_path = output_dir / f"drivers_shap_{label_name}.csv"

    # Select top-k features by absolute importance
    counts = df["Feature"].value_counts()
    valid = counts[counts >= min_freq].index
    if len(valid) == 0:
        print(f"      [!] Skipping Strip Plot: No features met min_freq={min_freq}.")
        return
    df_filtered = df[df["Feature"].isin(valid)]
    df_filtered["Abs_Score"] = df_filtered["Score"].abs()
    feature_importance = df_filtered.groupby("Feature")["Abs_Score"].mean().sort_values(ascending=False)
    
    # Reverse so most important is at the top of the plot
    top_features = feature_importance.head(top_k).index.tolist()[::-1]
    
    df_plot = df[df["Feature"].isin(top_features)].copy()
    
    # --- GLOBAL MAPPINGS & FLATTENING ---
    group_definitions = {
        **MED_ROLES, **CLINICAL_ROLES, **INFECTION_ROLES,
        **NUMERIC_ROLES, **ORDINAL_ROLES, **UNKNOWN_ROLES,
    }
    
    # Map raw value -> Group Name (e.g., 'E. coli' -> 'Bacteria - Enteric')
    val_to_group = {}
    for group_name, members in group_definitions.items():
        for m in members: val_to_group[m] = group_name

    # Check for unmapped values and auto-map "Other ..." to unknown
    unique_vals = df_plot["Value"].unique()
    unknowns = [v for v in unique_vals if v not in val_to_group]
    truly_unknown_groups = set()
    for v in unknowns:
        if str(v).lower().startswith("other "):
            val_to_group[v] = 'Other/Unknown'
        elif str(v) in ORDINAL_LEVELS_LIST: # Fallback if direct mapping missed
             val_to_group[v] = 'Ordinal level'
        else:
            # Create a dedicated warning label for the legend
            warning_label = f"{v}_"
            val_to_group[v] = warning_label
            truly_unknown_groups.add(warning_label)
            print(f"[WARNING] Missing mapping for: {v}")

    # Map Values to Groups for Plotting
    df_plot["Value_Grouped"] = df_plot["Value"].map(val_to_group).fillna("Other/Unknown")
    df_plot.to_csv(csv_path, index=False)

    # --- COLOR PALETTE GENERATION ---
    
    # 1. Ordinal Palette (Viridis is good, but Plasma is higher contrast against white)
    # We keep Viridis as requested, but ensure it uses the full range.
    vir_cmap = plt.get_cmap("viridis", len(ORDINAL_LEVELS_LIST))
    ord_colors = {lvl: mcolors.to_hex(vir_cmap(i)) for i, lvl in enumerate(ORDINAL_LEVELS_LIST)}
    
    # 2. Numeric/Boolean Palette (Explicit High Contrast)
    # Map specifically to match your Bar Chart logic (Blue=0/False, Red=1/True)
    # We define specific colors for the Numeric Role Keys found in GLOBAL MAPPINGS
    num_manual_map = {
        'No / Male': '#1f77b4',                # Muted Blue
        'Yes / Female / Occured': '#d62728',   # Brick Red
        'Value 2': '#ff7f0e',                  # Orange
        'Value 3': '#9467bd'                   # Purple
    }
    # Fallback generator if keys don't match exactly
    heat_cmap = plt.get_cmap("RdYlGn")
    num_colors = {}
    for i, k in enumerate(NUMERIC_ROLES.keys()):
        if k in num_manual_map:
            num_colors[k] = num_manual_map[k]
        else:
            num_colors[k] = mcolors.to_hex(heat_cmap(i / max(1, len(NUMERIC_ROLES))))

    unknown_color = "#333333"

    # Unified categorical palette (high contrast / curated)
    cat_hex_pool = [
        '#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231',  # Red, Green, Yellow, Blue, Orange
        '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe',  # Purple, Cyan, Magenta, Lime, Pink
        '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000',  # Teal, Lavender, Brown, Beige, Maroon
        # '#aaffc3', '#808000', '#ffd8b1', '#000075', '#808080',  # Mint, Olive, Apricot, Navy, Grey
    ]

    # Assign colors to groups
    master_palette = {}
    cat_counter = 0
    unique_groups = sorted(df_plot["Value_Grouped"].unique())    
    for grp in unique_groups:
        if grp == "Ordinal level": 
            pass
        elif grp in num_colors:
            master_palette[grp] = num_colors[grp]
        elif grp in UNKNOWN_ROLES or grp == 'Other/Unknown':
            master_palette[grp] = unknown_color
        else:
            # Categorical group cycling through high-contrast pool
            master_palette[grp] = cat_hex_pool[cat_counter % len(cat_hex_pool)]
            cat_counter += 1

    # Add specific colors for ordinal values
    for lvl in ORDINAL_LEVELS_LIST:
        master_palette[lvl] = ord_colors.get(lvl, unknown_color)

    # Helper function
    def get_value_type(group_name):
        if group_name == 'Ordinal level': return "Ordinal"
        if group_name in NUMERIC_ROLES: return "Numeric/Bool"
        if group_name in UNKNOWN_ROLES or group_name == "Other/Unknown": return "Other"
        return "Categorical"
    
    df_plot["Group_Type"] = df_plot["Value_Grouped"].apply(get_value_type)
    markers_map = {"Ordinal": "o", "Numeric/Bool": "D", "Categorical": "s", "Other": "X"}
    sizes_map = {"Ordinal": 60, "Numeric/Bool": 50, "Categorical": 55, "Other": 70}

    # Dynamic layout configuration
    START_X = -0.02
    WRAP_THRESHOLD = -1.35
    LINE_SPACING = 0.28
    CHAR_WIDTH = 0.013
    ROW_PADDING = 0.40
    
    # Plot rows one by one, with dedicated legends
    feature_y_map = {}
    row_heights = {}
    current_floor = 0.0
    for feature in top_features:
        
        # # Determine items to show in legend
        # subset = df_plot[df_plot["Feature"] == feature]
        # non_ord = sorted(subset[subset["Value_Grouped"] != "Ordinal level"]["Value_Grouped"].unique(), key=str)
        # ords = subset[subset["Value_Grouped"] == "Ordinal level"]["Value"].unique()
        # ords = sorted(ords, key=lambda x: ORDINAL_LEVELS_LIST.index(x) if x in ORDINAL_LEVELS_LIST else 999)
        # items_to_show = sorted(non_ord, key=str) + list(ords)
        
        # Determine items to show in legend
        subset = df_plot[df_plot["Feature"] == feature]
        cat_groups = subset[~subset["Group_Type"].isin(["Ordinal", "Numeric/Bool"])]["Value_Grouped"].unique()
        raw_vals = subset[subset["Group_Type"].isin(["Ordinal", "Numeric/Bool"])]["Value"].unique()
        cat_groups = sorted(cat_groups, key=str)
        ords = sorted([x for x in raw_vals if x in ORDINAL_LEVELS_LIST], key=lambda x: ORDINAL_LEVELS_LIST.index(x))
        nums = sorted([x for x in raw_vals if x not in ORDINAL_LEVELS_LIST], key=str)
        items_to_show = list(cat_groups) + nums + ords
        
        # Simulate legend wrapping to calculate height
        cursor_x = START_X
        n_lines = 1
        for item in reversed(items_to_show):
            item_width = (len(str(item)) * CHAR_WIDTH) + 0.06 
            if (cursor_x - item_width) < WRAP_THRESHOLD:
                n_lines += 1
                cursor_x = START_X
            cursor_x = cursor_x - item_width - 0.03
            
        # We want the scatter to be centered on the legend block
        legend_block_height = (n_lines * LINE_SPACING)
        row_height = max(0.5, legend_block_height + 0.15) 
        y_center = current_floor + (row_height / 2) + ROW_PADDING
        feature_y_map[feature] = y_center
        row_heights[feature] = row_height
        
        # Update floor for next row
        current_floor = y_center + (row_height / 2)

    df_plot["Feature_Y"] = df_plot["Feature"].map(feature_y_map)
    df_plot["Row_Height"] = df_plot["Feature"].map(row_heights)
    if max_delta is not None and max_delta > 0:
        df_plot["Alpha"] = 1.0 - (df_plot["Delta"] / max_delta)
        df_plot["Alpha"] = df_plot["Alpha"].clip(lower=0.0, upper=1.0)
    else:
        df_plot["Alpha"] = 1.0
    
    # Jitter (removed for time being mapped to y-value)
    # rng = np.random.default_rng(1234)
    # noise = rng.uniform(-0.42, 0.42, size=len(df_plot)) 
    # df_plot["Y_Value"] = df_plot["Feature_Y"] + (noise * df_plot["Row_Height"])
    
    # Use y-value within each row to show datapoint time (from 0 to fup_max)
    df_plot["Days_Clipped"] = df_plot["Days"].clip(lower=0.0)
    y_shift = (0.5 - (df_plot["Days_Clipped"] / fup_max)) * df_plot["Row_Height"]
    df_plot["Y_Value"] = df_plot["Feature_Y"] + y_shift

    # Plot setup
    total_y_max = current_floor + 0.5
    fig_height = max(6, total_y_max * 0.8)
    _, ax = plt.subplots(figsize=(16, fig_height))
    plt.subplots_adjust(left=0.60, right=0.97, top=0.98, bottom=0.04)
    
    # Set limits based on calculated floor/ceiling
    ax.set_ylim(feature_y_map[top_features[0]] - (row_heights[top_features[0]]/2) - 0.5, total_y_max)
    def get_color_key(row):
        return row["Value"] if row["Value_Grouped"] == "Ordinal level" else row["Value_Grouped"]
    df_plot["Color_Key"] = df_plot.apply(get_color_key, axis=1)
    
    sns.scatterplot(
        data=df_plot, x="Score", y="Y_Value",
        hue="Color_Key", style="Group_Type", size="Group_Type",
        sizes=sizes_map, markers=markers_map, palette=master_palette,
        alpha=0.6, edgecolor="white", linewidth=0.5, legend=False, ax=ax,
    )

    # Draw legends
    trans = transforms.blended_transform_factory(ax.transAxes, ax.transData)
    ax.set_yticks([feature_y_map[f] for f in top_features])
    ax.set_yticklabels([]) 
    ax.set_ylabel("")
    ax.set_xlabel("Attribution score (positive = risk, negative = protective)", fontsize=11)
    
    for f in top_features:
        y_center = feature_y_map[f]
        h = row_heights[f]
        ax.axhline(
            y_center - h/2 - (ROW_PADDING/2),
            color='gray', linestyle='-', linewidth=0.3, alpha=0.1,
        )
    ax.axvline(0, color="black", linestyle="-", alpha=0.4)

    for feature in top_features:
        y_center = feature_y_map[feature]
        h = row_heights[feature]
        
        # Calculate top of the legend block
        # Title goes slightly above the calculated "row top" visual area
        y_top = y_center + (h / 2) 
        ax.text(
            START_X, y_top + 0.00, feature, transform=trans, color='#222',
            ha='right', va='center', fontweight='bold', fontsize=11,
        )
        subset = df_plot[df_plot["Feature"] == feature]
        
        # # Re-build sorted items list
        # non_ord = sorted(subset[subset["Value_Grouped"] != "Ordinal level"]["Value_Grouped"].unique(), key=str)
        # ords = subset[subset["Value_Grouped"] == "Ordinal level"]["Value"].unique()
        # ords = sorted(ords, key=lambda x: ORDINAL_LEVELS_LIST.index(x) if x in ORDINAL_LEVELS_LIST else 999)
        # items_to_show = non_ord + list(ords)
        
        # Re-build sorted items list
        subset = df_plot[df_plot["Feature"] == feature]
        cat_groups = subset[~subset["Group_Type"].isin(["Ordinal", "Numeric/Bool"])]["Value_Grouped"].unique()
        raw_vals = subset[subset["Group_Type"].isin(["Ordinal", "Numeric/Bool"])]["Value"].unique()
        cat_groups = sorted(cat_groups, key=str)
        ords = sorted([x for x in raw_vals if x in ORDINAL_LEVELS_LIST], key=lambda x: ORDINAL_LEVELS_LIST.index(x))
        nums = sorted([x for x in raw_vals if x not in ORDINAL_LEVELS_LIST], key=str)
        items_to_show = list(cat_groups) + nums + ords

        # Start text below title
        cursor_y = y_top - 0.35 
        cursor_x = START_X
        for item in reversed(items_to_show):
            
            # Determine properties based on the resolved group name
            group_name = val_to_group.get(item, item)
            if group_name == "Ordinal level" or item in ORDINAL_LEVELS_LIST:
                g_type = "Ordinal"
                color = master_palette.get(item, unknown_color)
            elif group_name in NUMERIC_ROLES:
                g_type = "Numeric/Bool"
                color = master_palette.get(group_name, unknown_color)
            else: 
                g_type = get_value_type(group_name)
                color = master_palette.get(group_name, unknown_color)
            
            marker = markers_map.get(g_type, "X")
            ms = sizes_map.get(g_type, 50)
            text_str = str(item)
            
            item_width = (len(text_str) * CHAR_WIDTH) + 0.06
            if (cursor_x - item_width) < WRAP_THRESHOLD:
                cursor_x = START_X
                cursor_y -= LINE_SPACING

            ax.text(
                cursor_x, cursor_y, text_str, transform=trans,
                ha='right', va='center', fontsize=9, color='#555',
            )
            
            # Markers
            text_len_visual = len(text_str) * (CHAR_WIDTH * 0.9)
            marker_x_pos = cursor_x - text_len_visual - 0.025
            ax.scatter(
                [marker_x_pos], [cursor_y], marker=marker, s=ms, c=color, zorder=10,
                transform=trans, clip_on=False, edgecolors='white', linewidth=0.5,
            )
            
            cursor_x = marker_x_pos - 0.04

    plt.title(f"Detailed feature impact: {label_name}", fontsize=14, y=1.0)
    plt.savefig(output_dir / f"drivers_shap_{label_name}.png", dpi=300)
    plt.close()


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


def run_captum_analysis(
    model, dataset, collator, vocab, device, output_dir, label_name, fup_max,
    indices, target_idx=0, entity_filter=None, max_delta=None, add_noise=False,
    top_k=20, min_freq=20, plot_only=None, agg_method="mean",
):
    """
    Orchestrates the analysis and plotting pipeline on the selected indices.
    """
    print(f" -> Extracting attributions for {len(indices)} samples...")
    csv_path = output_dir / f"attributions_analysis_{label_name}.csv"
    
    df = extract_attributions(
        model, dataset, collator, indices, vocab, device,
        target_idx=target_idx, entity_filter=entity_filter,
        max_delta=max_delta, add_noise=add_noise,
        csv_path=csv_path, plot_only=plot_only,
    )
    
    if df.empty:
        print("No attributions extracted.")
        return

    print("\n>>> Generating plots...")

    print(" -> Generating Feature Importance Bar Chart...")
    plot_feature_importance(
        df, output_dir, label_name, 
        top_k=top_k, min_freq=min_freq, agg_method=agg_method
    )

    print(" -> Generating Volcano Plot (Frequency vs Impact)...")
    plot_frequency_vs_impact(df, output_dir, label_name, min_freq=min_freq)
    
    print(" -> Generating Detailed Strip Plot...")
    plot_feature_value_impact(
        df, output_dir, label_name, fup_max,
        max_delta=max_delta, top_k=top_k, min_freq=min_freq
    )


def main():
    # Load task configuration
    args = parse_args()
    base_dir = TASK_CONFIG[args.task_key]["base_dir"]
    hrz = TASK_CONFIG[args.task_key]["horizon"]
    fup_min = TASK_CONFIG[args.task_key]["fup_min"]
    fup_max = TASK_CONFIG[args.task_key]["fup_max"]
    fup_step = TASK_CONFIG[args.task_key]["fup_step"]
    fups = list(range(fup_min, fup_max + 1, fup_step))
    
    # Setup model and output directories
    checkpoint_path = find_best_checkpoint(Path(base_dir), args.task_key, hrz)
    print(f"\n>>> Using Checkpoint: {checkpoint_path}")
    with open(args.config, 'r') as f: config = yaml.safe_load(f)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup keys
    ckpt_hrzs = extract_horizons_from_path(checkpoint_path)
    if hrz not in ckpt_hrzs:
        raise ValueError(f"Requested horizon {hrz} not found in: {ckpt_hrzs}")
    target_idx = ckpt_hrzs.index(hrz)
    label_key = f"label_{args.task_key}_{hrz:04d}d"
    print(f"\n>>> Target index: {target_idx} (horizon: {hrz}d)")

    # Load dataset
    print("\n>>> Loading dataset...")
    data_dir = Path(config["data_dir"])
    sanity_dir = output_dir / "sanity_check" if GENERATE_SANITIZE_PLOTS else None
    dataset, _, vocab = load_hf_data_and_metadata(
        data_dir=data_dir,
        fup_train=[360], fup_valid=[360], fup_test=fups,
        time_mapping=config["data_collator"]["time_mapping"],
        eav_mappings=config["data_collator"]["eav_mappings"],
        sanity_check_output_dir=sanity_dir,
    )
    
    # Filter test set
    test_ds = dataset["test"].filter(
        lambda x: x[label_key] != -100, num_proc=SAFE_NUM_PROC
    ).map(
        lambda x: {"split": "test"}, num_proc=SAFE_NUM_PROC
    )

    # Load model
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
        label_keys=[f"label_{args.task_key}_{h:04d}d" for h in ckpt_hrzs],
        max_position_embeddings=model.config.max_position_embeddings
    )

    # Run full inference (necessary for clustering or to get probs)
    print("\n>>> Running inference on test set (for embeddings and selection)...")
    interpreter = ModelInterpreter(model)
    loader = DataLoader(test_ds, batch_size=32, collate_fn=collator.torch_call)
    res = interpreter.get_embeddings_and_predictions(loader)
    target_labels = res["labels"][:, target_idx]

    # Select samples for analysis (randomly sample N patients from the whole set)
    all_indices = np.arange(len(test_ds))
    if len(all_indices) > args.captum_samples:
        print(f"\n>>> Subsampling {args.captum_samples} patients from {len(all_indices)} total test samples.")
        selected_idx = np.random.choice(all_indices, args.captum_samples, replace=False)
    else:
        print(f"\n>>> Analyzing all {len(all_indices)} test samples.")
        selected_idx = all_indices

    # Run analysis
    label_name = f"{args.task_key}_hrz{hrz:04d}d"
    label_name = f"{label_name}_fup{fup_min}-{fup_max}-{fup_step}".lower()
    run_captum_analysis(
        model=model, dataset=test_ds, collator=collator, output_dir=output_dir,
        label_name=label_name, fup_max=fup_max, indices=selected_idx, vocab=vocab,
        device="cuda", target_idx=target_idx, max_delta=args.max_delta,
        add_noise=args.add_noise, top_k=args.top_k, plot_only=args.plot_only,
        agg_method=args.agg_method, min_freq=args.min_freq,
    )

    # # Run clustering analysis (Uncomment if needed)
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

    print(f"\nSuccess! All results can be found in {output_dir}")


if __name__ == "__main__":
    main()