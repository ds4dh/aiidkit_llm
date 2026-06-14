import yaml
import gc
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
from captum.attr import IntegratedGradients, LayerIntegratedGradients
from matplotlib.patches import PathPatch
from matplotlib.markers import MarkerStyle
from matplotlib.offsetbox import AnchoredOffsetbox, HPacker, VPacker, TextArea, DrawingArea

from src.data.patient_dataset import load_hf_data_and_metadata
from src.model.patient_embedder import PatientEmbeddingModelFactory, PatientDataCollatorForClassification
from src.evaluation.evaluate_models import ModelInterpreter
from scripts.script_utils import find_best_checkpoint, extract_horizons_from_path, get_best_optuna_run


CLI_CFG = {}
SAFE_NUM_PROC = 4
GENERATE_SANITIZE_PLOTS = False

# =================================
# GLOBAL MAPPINGS AND CONFIGURATION
# =================================

# Run configuration
RESULTS_DIR = Path("results_final")
TRANSFORMER_BASE_DIR = RESULTS_DIR / "transformer"
OUTPUT_DIR = RESULTS_DIR / Path("analysis/interpretability")
DATA_DIR = Path("/home/shares/ds4dh/aiidkit_project/data_new/processed/v3.6/teav")
CONFIG_PATH = Path("configs/discriminative_training.yaml")
FROM_OPTUNA = "optuna" in TRANSFORMER_BASE_DIR.name
DATA_SPLIT_TYPE = "temporal_split"
PLOT_ONLY = False

# Captum configuration
TOP_K = 20
MIN_FREQ = 20
MAX_DELTA = 0.05
AGG_METHOD = "mean"
NUM_CAPTUM_SAMPLES = 1000
ATTRIBUTIONS_TO_VALUES_ONLY = True  # captum sees value token input embeddings only

# Task configuration
TASK_CONFIG = {
    "bacteria_perioperative": {  # 0 -> 1 month post-tpx
        "task": "infection_bacteria", 
        "horizon": 30, "fup_min": 0, "fup_max": 0, "fup_step": 30,
    },
    "bacteria_opportunistic": {  # 1 -> 6 months post-tpx
        "task": "infection_bacteria",
        "horizon": 30, "fup_min": 30, "fup_max": 150, "fup_step": 30,
    },
    "bacteria_maintenance": {  # 6 -> 12 months post-tpx
        "task": "infection_bacteria",
        "horizon": 30, "fup_min": 180, "fup_max": 330, "fup_step": 30,
    },
    "bacteria_long_term": {  # 12 -> 24 months post-tpx
        "task": "infection_bacteria",
        "horizon": 30, "fup_min": 360, "fup_max": 690, "fup_step": 30,
    },
    "bacteria_very_long_term": {  # 24 -> 60 months post-tpx
        "task": "infection_bacteria",
        "horizon": 30, "fup_min": 720, "fup_max": 1770, "fup_step": 30,
    },
}

# Medications
MED_ROLES = {
    'Med. CNI':                     ['Tacrolimus', 'Cyclosporine'],
    'Med. costimulation blocker':   ['Belatacept'],
    'Med. antimetabolite':          ['Mycophenolate Mofetil', 'Mycophenolic Acid', 'Azathioprine'],
    'Med. mTOR inhibitor':          ['Everolimus', 'Sirolimus'],
    'Med. steroids':                ['Glucocorticoid', 'Methylprednisolone', 'Prednisone'],
    'Med. induction':               ['Basiliximab', 'Anti-thymocyte globulin', 'Rabbit ATG'],
    'Med. rejection treat.':        ['Rituximab', 'IVIG', 'Human Immunoglobulin'], 
    'Med. antiviral':               ['Valganciclovir', 'Valaciclovir', 'Lamivudine', 'Ganciclovir', 'Entecavir', 'Tenofovir', 'Emtricitabine', 'Dolutegravir'],
    'Med. antibiotic proph.':       ['Cotrimoxazole', 'Atovaquone', 'Pentamidine', 'Dapsone'],
    'Med. antibiotic treat.':       ['Beta-Lactame', 'Quinolone', 'Cephalosporin', 'Metronidazole', 'Fosfomycin', 'Nitrofurantoin', 'Clarithromycin', 'Isoniazid', 'Moxifloxacin', 'Ciprofloxacin', 'Azithromycin'],
    'Med. antifungal treat.':       ['Amphotericin B', 'Itraconazole', 'Fluconazole', 'Voriconazole', 'Posaconazole'],
    'Med. antihypertensive':        ['Calcium channel blocker', 'Beta-blocker', 'ACE inhibitor', 'Angiotensin receptor blocker', 'Moxonidine', 'Amlodipine', 'Sildenafil', 'Doxazosin', 'Minoxidil', 'Lisinopril'],
    'Med. cardiac/antiarrhythmic':  ['Amiodarone', 'Digoxin', 'Ranolazine'],
    'Med. antithrombotic':          ['Platelet aggregation inhibitor', 'Anticoagulation therapy', 'Phenprocoumon'],
    'Med. diabetes treat.':         ['Insulin', 'Oral antidiabetic drug', 'Linagliptin'],
    'Med. diuretic':                ['Torasemide', 'Furosemide', 'Chlortalidone', 'Hydrochlorothiazide', 'Spironolactone'],
    'Med. lipid lowering':          ['Statin', 'Rosuvastatin', 'Atorvastatin', 'Ezetimibe'],
    'Med. neuro/psych':             ['Pregabalin', 'Levetiracetam', 'Sertraline', 'Quetiapine'],
    'Med. other':                   ['Other drugs', 'Levothyroxine', 'Vitamin D', 'Calcium', 'Metoclopramide', 'Tamsulosin', 'Cinacalcet', 'Allopurinol', 'Pancreatin']
}

# Procedure & Infection events
INFECTION_ROLES = {
    'Bacteria':                     ['Bacterial', 'Bacteria'],
    'Virus':                        ['Viral', 'Virus'],
    'Fungal':                       ['Fungal', 'Fungi'],
    'Parasite':                     ['Parasite', 'Cryptosporidium sp', 'Toxo', 'Other parasites'], 
    'Bacteria enteric':             ['E. coli', 'Klebsiella sp', 'Enterobacter', 'Other enterobacteria', 'Enterococcus', 'Other non-enteric GN bacteria', 'Clostridium sp', 'CDI'],
    'Bacteria resp./oral':          ['Pneumococcus', 'Streptococcus sp', 'Haemophilus influenzae', 'Legionella', 'Streptococcus  sp', 'Actinomyces', 'M. tuberculosis group'],
    'Bacteria skin/staph':          ['Staph aureus', 'MSSA', 'St. coagulase negative', 'MRSA'],
    'Bacteria hospital':            ['Pseudomonas aeruginosa', 'Acinetobacter', 'Stenotrophomonas'],
    'Virus herpes group':           ['CMV', 'EBV', 'VZV', 'HSV'],
    'Virus respiratory':            ['SARS-CoV-2', 'Influenza', 'Rhinovirus', 'Metapneumovirus', 'Parainfluenza', 'Adenovirus', 'RSV'],
    'Virus hep/GI/other':           ['HBV', 'HCV', 'Noro', 'BKV', 'JCV', 'Parvo B19', 'Enterovirus'],
    'Fungal yeast':                 ['Candida albicans', 'Candida non albicans'],
    'Fungal mold':                  ['Aspergillus non-fumigatus', 'Zygomycetes', 'Pneumocystis sp', 'Aspergillus fumigatus'],
    'Proven':                       ['Proven disease', 'Primary infection'],
    'Possible':                     ['Possible disease', 'Probable disease'],
    'Viral syndrome':               ['Viral syndrome'],
    'Colonization':                 ['Colonization'],
    'Asymptomatic':                 ['Asymptomatic'],
    'UTI':                          ['Urinary tract', 'UTI'],
    'Resp':                         ['Respiratory', 'Lung', 'Pneumonia', 'RT'],
    'GI':                           ['Gastrointestinal', 'Abdominal', 'GI', 'Intraabdominal infection'],
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
    'Rejection event':              ['Biops proven rj', 'Clinically suspected rj', 'Clinical', 'Subclinical', 'SAR', 'CAN'],
    'Transplant procedure':         ['Kidney tpx', 'Heart tpx', 'Lung tpx', 'HSCT allo', 'HSCT auto', 'Kidney - Pancreas', 'Kidney - Liver', 'Kidney - Lung', 'Kidney - Heart', 'Kidney - Islets', 'Islets'],
    'Surgery':                      ['Nephrectomy native', 'Nephrectomy allograft', 'Nephrectomy allograft and native', 'Non-Tx surgery', 'Tpx-related re-surgery', 'Events/Surgery', 'Non-transplant surgery'], # <-- Included directly
    'Pregnancy/birth':              ['Birth', 'Pregnancy', 'Abortion/miscarriage'],
    'Emergency/critical event':     ['MOF', 'Agranulocytosis', 'GI haemorrhage', 'Bone fracture', 'Hemorrhagy', 'Thromboembolic disease'], # <-- Included directly
    'Complication surgical/uro':    ['Lymphocele', 'Biliary leak', 'Biliary stenosis', 'Urine leak', 'Obstruction', 'Prosthetic'],
    'Previous graft failure':       ['Previous GF', 'Previous graft failure'],
    'Kidney dis. GN/nephritis':     ['GN', 'Interstitial nephritis', 'Reflux/Pyelonephritis', 'Graft pyelonephritis'],
    'Kidney dis. PCKD':             ['PCKD'],
    'Kidney dis. hereditary':       ['Congenital kidney', 'Hereditary non_PCKD'],
    'Kidney dis. vascular/DM':      ['Nephrosclerosis', 'DM nephropathy', 'TMA', 'RA stenosis', 'RVT', 'RAT'],
    'Kidney dis. failure':          ['CKD', 'ARF', 'Acute on chronic RF', 'ATN', 'Acute kidney injury'],
    'Kidney dis. toxicity':         ['CNI nephrotoxicity', 'BKV nephropathy'],
    'Kidney dis. other':            ['OTH', 'CTR', 'CTU', 'Chronic lesions, not specified', 'De novo disease', 'Recurrence kidney disease', 'Metabolic/Kidney'],
    'Cancer kidney':                ['Kidney cancer'],
    'Cancer skin':                  ['Melanoma', 'Spinalioma', 'Basalioma', 'Other skin cancer'],
    'Cancer GI':                    ['Colorectal cancer', 'Liver cancer', 'HCC'],
    'Cancer UTI':                   ['Uro_bladder cancer', 'Prostate cancer', 'Testicular cancer'],
    'Cancer gyneco':                ['Breast cancer', 'Cervix - Uterus - Adnex ca'],
    'Cancer lung':                  ['Lung cancer'],
    'Cancer endocrine':             ['Thyroid cancer', 'Neuroendocrine TU'],
    'Cancer sarkoma':               ['Sarkoma'],
    'Cancer blood/lymph':           ['PTLD', 'Myeloid neoplasm', 'Leukemia', 'Lymphoma'],
    'Cancer other':                 ['Malignancy'],
    'Comorb. diabetes':             ['DM type1', 'DM type2 treated', 'PTDM', 'Diabetes mellitus'],
    'Comorb. cardiac risk factors': ['HTN', 'Hyperlipidemia', 'Hypertension'],
    'Comorb. vascular':             ['CAD', 'PAD', 'CVD', 'Arterial thrombosis', 'Cardiovascular'],
    'Comorb. heart failure':        ['CHE', 'LVEF < 30%', 'Dilated CMP', 'HFpEF (Symptomatic Heart failure with preserved LVEF)'],
    'Comorb. arrhythmias':          ['AF', 'VT_PE', 'Pacemaker, ICD'],
    'Comorb. valvular':             ['Cardiac valvular disease'],
    'Comorb. respiratory':          ['COPD', 'Bronchial stenosis'],
    'Comorb. HIV':                  ['HIV'],
    'Comorb. liver/hep':            ['Hep C', 'Hep B', 'Drug-induced liver injury', 'Cholangitis'],
    'Comorb. other':                ['Osteoporosis', 'Other metabolic', 'Alcohol', 'ANX', 'SUI'],
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
    'Value 4':                      ['4'],
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


def main():
    with open(CONFIG_PATH, 'r') as f:  config = yaml.safe_load(f)
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    for config_name, task_params in TASK_CONFIG.items():
        task_key = task_params["task"]
        hrz = task_params["horizon"]
        fup_min = task_params["fup_min"]
        fup_max = task_params["fup_max"]
        fup_step = task_params["fup_step"]
        fups = list(range(fup_min, fup_max + 1, fup_step))

        print(f"\n{'='*60}")
        print(f">>> Executing Config: '{config_name}' (Task: {task_key}, Horizon: {hrz})")
        print(f"{'='*60}")

        try:
            if FROM_OPTUNA:
                trial_name, pt_config = get_best_optuna_run(TRANSFORMER_BASE_DIR, DATA_SPLIT_TYPE, task_key)
                base_dir_for_ckpt = TRANSFORMER_BASE_DIR / DATA_SPLIT_TYPE / task_key / trial_name / DATA_SPLIT_TYPE / pt_config
            else:
                [base_dir_for_ckpt] = (d for d in (TRANSFORMER_BASE_DIR / DATA_SPLIT_TYPE).iterdir() if d.is_dir())
        except Exception as e:
            print(f"[Error] Could not resolve base directory for {task_key}: {e}")
            continue 
        
        try:
            checkpoint_path = find_best_checkpoint(base_dir_for_ckpt, task_key, hrz)
            print(f"\n>>> Using Checkpoint: {checkpoint_path}")
        except Exception as e:
            print(f"[Error] Checkpoint error: {e}")
            continue

        ckpt_hrzs = extract_horizons_from_path(checkpoint_path)
        if hrz not in ckpt_hrzs:
            print(f"[Error] Requested horizon {hrz} not found in: {ckpt_hrzs}")
            continue
            
        target_idx = ckpt_hrzs.index(hrz)
        label_key = f"label_{task_key}_{hrz:04d}d"
        all_required_labels = [f"label_{task_key}_{h:04d}d" for h in ckpt_hrzs]
        print(f"\n>>> Target index: {target_idx} (horizon: {hrz}d)")

        print("\n>>> Loading dataset...")
        data_dir_split = DATA_DIR / DATA_SPLIT_TYPE
        sanity_dir = output_dir / "sanity_check" if GENERATE_SANITIZE_PLOTS else None
        
        dataset, _, vocab = load_hf_data_and_metadata(
            data_dir=data_dir_split,
            fup_train=[360], 
            fup_valid=[360], 
            fup_test=fups,
            label_keys=all_required_labels,
            time_mapping=config["data_collator"].get("time_mapping", None),
            eav_mappings=config["data_collator"].get("eav_mappings", None),
            sanity_check_output_dir=sanity_dir,
        )
        
        test_ds = dataset["test"].filter(
            lambda x: x[label_key] != -100, num_proc=SAFE_NUM_PROC
        ).map(
            lambda x: {"split": "test"}, num_proc=SAFE_NUM_PROC
        )

        model_cfg_base = config["model"].copy()
        if "model_args" not in model_cfg_base:
            model_cfg_base["model_args"] = {}
            
        for key in ["dtype", "torch_dtype"]:
            if isinstance(model_cfg_base["model_args"].get(key), str):
                model_cfg_base["model_args"][key] = getattr(torch, model_cfg_base["model_args"][key])
        
        target_dtype = model_cfg_base["model_args"].get("torch_dtype", torch.float32)

        emb_cfg = model_cfg_base["embedding_layer_config"].copy()
        emb_cfg["vocab_size"] = len(vocab)
        
        model = PatientEmbeddingModelFactory.from_pretrained(
            task="classification",
            pretrained_dir=str(checkpoint_path),
            embedding_layer_config=emb_cfg,
            model_args=model_cfg_base["model_args"]
        )
        model = model.to(device="cuda", dtype=target_dtype)
        
        collator = PatientDataCollatorForClassification(
            **config["data_collator"], 
            label_keys=all_required_labels,
            max_position_embeddings=model.config.max_position_embeddings
        )

        print("\n>>> Running inference on test set (for embeddings and selection)...")
        interpreter = ModelInterpreter(model)
        loader = DataLoader(test_ds, batch_size=32, collate_fn=collator.torch_call)
        res = interpreter.get_embeddings_and_predictions(loader)
        target_labels = res["labels"][:, target_idx]

        all_indices = np.arange(len(test_ds))
        if len(all_indices) > NUM_CAPTUM_SAMPLES:
            print(f"\n>>> Subsampling {NUM_CAPTUM_SAMPLES} patients from {len(all_indices)} total test samples.")
            selected_idx = np.random.choice(all_indices, NUM_CAPTUM_SAMPLES, replace=False)
        else:
            print(f"\n>>> Analyzing all {len(all_indices)} test samples.")
            selected_idx = all_indices

        label_name = f"{config_name}_hrz{hrz:04d}d_fup{fup_min}-{fup_max}-{fup_step}".lower()
        run_captum_analysis(
            model=model, dataset=test_ds, collator=collator, output_dir=output_dir,
            label_name=label_name, fup_max=fup_max, indices=selected_idx, vocab=vocab,
            device="cuda", target_idx=target_idx, max_delta=MAX_DELTA, top_k=TOP_K,
            plot_only=PLOT_ONLY, agg_method=AGG_METHOD, min_freq=MIN_FREQ,
        )

        del model, interpreter, loader, res, dataset, test_ds
        gc.collect()
        torch.cuda.empty_cache()

    print(f"\nSuccess! All results can be found in {output_dir}")


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
    entity_filter=None, max_delta=None,
    csv_path=None, plot_only=False,
) -> pd.DataFrame:
    
    if plot_only and csv_path.exists():
        return pd.read_csv(csv_path)

    subset = Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=32, collate_fn=collator.torch_call)
    id2word = {v: k for k, v in vocab.items()} if isinstance(vocab, dict) else vocab
    wrapper = ForwardWrapperForCaptum(model).to(device)
    
    # Select the target hook module based on the global configuration flag
    if ATTRIBUTIONS_TO_VALUES_ONLY:
        target_layer = model.patient_embedder.value_embedding_hook
    else:
        target_layer = model.patient_embedder.aggregated_embedding_hook
    
    # Layer integrated gradient module itself
    lig = LayerIntegratedGradients(wrapper, target_layer)
    
    collected_data = []
    for batch in tqdm(loader, desc="Calculating attributions"):
        inp = batch["input_dict"]
        ent = inp["entity_id"].to(device)
        attr = inp["attribute_id"].to(device)
        val = inp["value_id"].to(device)
        days = inp["days_since_tpx"].to(device)
        args = (ent, attr, val, days)

        # Instead of zero arrays, we pass a structurally complete sequence matching the padding ID
        baseline_ent = torch.zeros_like(ent, device=device)  # assuming pad_id = 0
        baseline_attr = torch.zeros_like(attr, device=device)
        baseline_val = torch.zeros_like(val, device=device)
        baseline_days = torch.zeros_like(days, device=device)
        baseline_args = (baseline_ent, baseline_attr, baseline_val, baseline_days)

        try:
            # Underflow errors from bfloat16 tracking are completely avoided
            with torch.inference_mode(False): 
                attributions, delta = lig.attribute(
                    inputs=args,
                    baselines=baseline_args,  # pass the neutral sequence baseline reference
                    target=target_idx,
                    n_steps=200,              # Riemann integral path density increased (100 -> 200)
                    return_convergence_delta=True,
                    internal_batch_size=32,   # reduced batch slice size to mitigate FP32 VRAM overhead
                )
        except RuntimeError as e:
            print(f"Skipping batch due to error: {e}")
            continue
        
        attrs_sum = attributions.sum(dim=-1).detach().float().cpu().numpy()
        ent_np = args[0].cpu().numpy()
        attr_np = args[1].cpu().numpy()
        val_np = args[2].cpu().numpy()
        days_np = args[3].cpu().numpy()
        mask_np = (ent_np != 0).astype(int)
        delta_np = delta.detach().cpu().numpy()

        for i in range(ent_np.shape[0]):
            patient_delta = float(abs(delta_np[i]))
            
            if max_delta is not None and patient_delta > max_delta:
                continue  # if mathematical error threshold breaks max_delta, drop patient metrics
            
            length = int(mask_np[i].sum())
            seq_ent = ent_np[i, :length]
            seq_attr = attr_np[i, :length]
            seq_val = val_np[i, :length]
            seq_score = attrs_sum[i, :length]
            seq_days = days_np[i, :length]
            
            for k, (e_id, a_id, v_id, score, day) in enumerate(
                zip(seq_ent, seq_attr, seq_val, seq_score, seq_days)
            ):
                
                if e_id < 5: continue
                ent_name = id2word.get(e_id, f"Ent_{e_id}")
                if entity_filter is not None:
                    if not ent_name.startswith(entity_filter): continue
                
                if "infection" in ent_name.lower():
                    if ent_name.strip().lower() == "infection":
                        ent_name = "Previous infection"
                    elif not ent_name.lower().startswith("previous"):
                        ent_name = f"Previous {ent_name.lower()}"
                
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
    if df.empty: return
    csv_path = output_dir / f"drivers_bar_{label_name}_{agg_method}.csv"

    df = df.copy()
    df["Feature_Value"] = df["Feature"] + ": " + df["Value"].astype(str)

    counts = df["Feature_Value"].value_counts()
    valid_combinations = counts[counts >= min_freq].index
    if len(valid_combinations) == 0:
        print(f"[!] Skipping bar chart: No feature-value combinations met min_freq={min_freq}.")
        return
        
    filtered_df = df[df["Feature_Value"].isin(valid_combinations)]
    agg_func = "sum" if agg_method == "sum" else "mean"
    agg_col = "Attribution"
    agg_df = filtered_df.groupby("Feature_Value")["Score"].agg(agg_func).reset_index()
    agg_df.rename(columns={"Score": agg_col}, inplace=True)
    
    df_pos = agg_df[agg_df[agg_col] > 0].sort_values(agg_col, ascending=False).head(top_k)
    df_neg = agg_df[agg_df[agg_col] < 0].sort_values(agg_col, ascending=True).head(top_k)
    
    pd.concat([df_pos, df_neg]).to_csv(csv_path, index=False)

    pos_vals = df_pos[agg_col].tolist()
    pos_labels = df_pos["Feature_Value"].tolist()
    neg_vals = df_neg[agg_col].tolist()
    neg_labels = df_neg["Feature_Value"].tolist()

    max_rows = max(len(pos_vals), len(neg_vals))
    if max_rows == 0:
        return

    pos_vals += [0.0] * (max_rows - len(pos_vals))
    pos_labels += [""] * (max_rows - len(pos_labels))
    neg_vals += [0.0] * (max_rows - len(neg_vals))
    neg_labels += [""] * (max_rows - len(neg_labels))

    pos_vals = pos_vals[::-1]
    pos_labels = pos_labels[::-1]
    neg_vals = neg_vals[::-1]
    neg_labels = neg_labels[::-1]
    y_positions = np.arange(max_rows)

    fig, ax = plt.subplots(figsize=(16, max(6, max_rows * 0.4))) 
    
    ax.barh(y_positions, neg_vals, color='#1f77b4', alpha=0.8, height=0.6)
    ax.barh(y_positions, pos_vals, color='#d62728', alpha=0.8, height=0.6)
    ax.axvline(0, color="black", linewidth=1.2)

    max_abs = max([abs(v) for v in neg_vals + pos_vals])
    if max_abs == 0: max_abs = 1.0
    ax.set_xlim(-max_abs * 1.1, max_abs * 1.1)

    ax.set_yticks(y_positions)
    ax.set_yticklabels(neg_labels, fontsize=11)
    ax.tick_params(axis='y', left=False)
    
    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim())
    ax2.set_yticks(y_positions)
    ax2.set_yticklabels(pos_labels, fontsize=11)
    ax2.tick_params(axis='y', right=False)

    title_suffix = "Cumulative impact" if agg_method == "sum" else "Average impact"
    explanation = "(blue = reduces predicted risk, red = increases predicted risk)"
    ax.set_title(
        f"Top {top_k} risk-increasing and risk-reducing drivers: {label_name}\n{explanation}",
        pad=20, fontsize=15, fontweight='bold',
    )
    ax.set_xlabel(f"{title_suffix} (attribution score)", fontsize=12)
    
    ax.grid(axis='x', linestyle='--', alpha=0.4)
    for spine in ['top', 'right', 'left']:
        ax.spines[spine].set_visible(False)
        ax2.spines[spine].set_visible(False)
    ax2.spines['bottom'].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_dir / f"drivers_bar_{label_name}.png", dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_frequency_vs_impact(df, output_dir, label_name, min_freq=50):
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
    
    explanation = "(above 0 = risk-increasing, below 0 = risk-reducing)"
    plt.title(f"Frequency vs. severity: {label_name}\n{explanation}", fontsize=14)
    plt.xscale("log")
    plt.xlabel("Frequency (log scale)", fontsize=12)
    plt.ylabel("Conditional mean attribution", fontsize=12)
    plt.grid(True, which="both", linestyle="--", alpha=0.2)
    plt.tight_layout()
    plt.savefig(output_dir / f"drivers_volcano_{label_name}.png", dpi=300)
    plt.close()


def plot_feature_value_impact(
    df, output_dir, label_name, fup_max, max_delta=None, top_k=20, min_freq=50,
):
    """
    Generates a SHAP-like strip plot with aligned per-row legends on the left.
    - Transparent legend background boxes with distinct clearance spacing from scatterplot data.
    - Top-aligned legends flush with the row's upper border boundary.
    - Mathematically balanced marker sizes inside the legend vector canvas.
    """
    if df.empty: return
    csv_path = output_dir / f"drivers_shap_{label_name}.csv"

    counts = df["Feature"].value_counts()
    valid = counts[counts >= min_freq].index
    if len(valid) == 0:
        print(f"      [!] Skipping Strip Plot: No features met min_freq={min_freq}.")
        return
    df_filtered = df[df["Feature"].isin(valid)]
    df_filtered["Abs_Score"] = df_filtered["Score"].abs()
    feature_importance = df_filtered.groupby("Feature")["Abs_Score"].mean().sort_values(ascending=False)
    
    top_features = feature_importance.head(top_k).index.tolist()[::-1]
    df_plot = df[df["Feature"].isin(top_features)].copy()

    group_definitions = {
        **MED_ROLES, **CLINICAL_ROLES, **INFECTION_ROLES,
        **NUMERIC_ROLES, **ORDINAL_ROLES, **UNKNOWN_ROLES,
    }
    
    val_to_group = {}
    for group_name, members in group_definitions.items():
        for m in members: val_to_group[m] = group_name

    adjusted_rows = []
    for feature in top_features:
        feat_df = df_plot[df_plot["Feature"] == feature].copy()
        val_counts = feat_df["Value"].value_counts()
        
        if len(val_counts) > 9:
            top_9_vals = val_counts.head(9).index.tolist()
            feat_df["Value"] = feat_df["Value"].apply(lambda v: v if v in top_9_vals else "Other")
        
        adjusted_rows.append(feat_df)
        
    df_plot = pd.concat(adjusted_rows, ignore_index=True)

    unique_vals = df_plot["Value"].unique()
    unknowns = [v for v in unique_vals if v not in val_to_group]
    for v in unknowns:
        if str(v).lower().startswith("other ") or str(v) == "Other":
            val_to_group[v] = 'Other/Unknown'
        elif str(v) in ORDINAL_LEVELS_LIST:
             val_to_group[v] = 'Ordinal level'
        else:
            val_to_group[v] = f"{v}_"

    df_plot["Value_Grouped"] = df_plot["Value"].map(val_to_group).fillna("Other/Unknown")
    df_plot.to_csv(csv_path, index=False)

    vir_cmap = plt.get_cmap("viridis", len(ORDINAL_LEVELS_LIST))
    ord_colors = {lvl: mcolors.to_hex(vir_cmap(i)) for i, lvl in enumerate(ORDINAL_LEVELS_LIST)}
    
    num_manual_map = {
        'No / Male': '#1f77b4', 'Yes / Female / Occured': '#d62728',
        'Value 2': '#ff7f0e', 'Value 3': '#9467bd', 'Value 4': '#17becf'
    }
    heat_cmap = plt.get_cmap("RdYlGn")
    num_colors = {}
    for i, k in enumerate(NUMERIC_ROLES.keys()):
        num_colors[k] = num_manual_map.get(k, mcolors.to_hex(heat_cmap(i / max(1, len(NUMERIC_ROLES)))))

    unknown_color = "#333333"
    cat_hex_pool = [
        '#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231',
        '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe',
        '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000',
    ]

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
            master_palette[grp] = cat_hex_pool[cat_counter % len(cat_hex_pool)]
            cat_counter += 1

    for lvl in ORDINAL_LEVELS_LIST:
        master_palette[lvl] = ord_colors.get(lvl, unknown_color)

    def get_value_type(group_name):
        if group_name == 'Ordinal level': return "Ordinal"
        if group_name in NUMERIC_ROLES: return "Numeric/Bool"
        if group_name in UNKNOWN_ROLES or group_name == "Other/Unknown": return "Other"
        return "Categorical"
    
    df_plot["Group_Type"] = df_plot["Value_Grouped"].apply(get_value_type)
    markers_map = {"Ordinal": "o", "Numeric/Bool": "D", "Categorical": "s", "Other": "X"}

    ROW_HEIGHT = 1.0       
    ROW_PADDING = 0.45     
    
    feature_y_map = {}
    row_heights = {}
    current_floor = 0.0
    
    for feature in top_features:
        y_center = current_floor + (ROW_HEIGHT / 2) + ROW_PADDING
        feature_y_map[feature] = y_center
        row_heights[feature] = ROW_HEIGHT
        current_floor = y_center + (ROW_HEIGHT / 2)

    df_plot["Feature_Y"] = df_plot["Feature"].map(feature_y_map)
    df_plot["Row_Height"] = df_plot["Feature"].map(row_heights)
    
    df_plot["Days_Clipped"] = df_plot["Days"].fillna(0.0).clip(lower=0.0)
    if fup_max > 0:
        y_shift = (0.5 - (df_plot["Days_Clipped"] / fup_max)) * df_plot["Row_Height"]
    else:
        y_shift = 0.5 * df_plot["Row_Height"]
    df_plot["Y_Value"] = df_plot["Feature_Y"] + y_shift

    total_y_max = current_floor + 0.6
    fig_height = max(7, total_y_max * 0.75)
    fig, ax = plt.subplots(figsize=(16, fig_height))
    
    # Left margin of 0.54 isolates the legend canvas bounding overlaps entirely
    plt.subplots_adjust(left=0.54, right=0.97, top=0.95, bottom=0.06)
    ax.set_ylim(feature_y_map[top_features[0]] - (ROW_HEIGHT / 2) - ROW_PADDING, total_y_max)
    df_plot["Color_Key"] = df_plot.apply(
        lambda r: r["Value"] if r["Value_Grouped"] == "Ordinal level" else r["Value_Grouped"],
        axis=1,
    )
    
    # Balanced scatter sizes map
    sizes_map_scatter = {"Ordinal": 45, "Numeric/Bool": 35, "Categorical": 40, "Other": 50}
    combo_counts = df_plot.groupby(["Feature", "Value"]).size().reset_index(name="Combo_Frequency")
    df_plot = df_plot.merge(combo_counts, on=["Feature", "Value"], how="left")  # sort to show rarest events on top
    df_plot = df_plot.sort_values(by="Combo_Frequency", ascending=False).reset_index(drop=True)    
    
    # Scatter plot itself
    sns.scatterplot(
        data=df_plot, x="Score", y="Y_Value",
        hue="Color_Key", style="Group_Type", size="Group_Type",
        sizes=sizes_map_scatter, markers=markers_map, palette=master_palette,
        alpha=0.6, edgecolor="white", linewidth=0.5, legend=False, ax=ax,
    )

    ax.set_yticks([feature_y_map[f] for f in top_features])
    ax.set_yticklabels([]) 
    ax.set_ylabel("")
    ax.tick_params(axis='y', left=False)
    ax.set_xlabel("Attribution score (positive = risk-increasing, negative = risk-reducing)", fontsize=11)
    
    for f_idx, f in enumerate(top_features):
        y_center = feature_y_map[f]
        y_separator = y_center - (ROW_HEIGHT / 2) - (ROW_PADDING / 2)
        ax.axhline(y_separator, color='#d3d3d3', linestyle='-', linewidth=0.6, alpha=0.7)
        if f_idx == len(top_features) - 1:
            y_ceiling = y_center + (ROW_HEIGHT / 2) + (ROW_PADDING / 2)
            ax.axhline(y_ceiling, color='#d3d3d3', linestyle='-', linewidth=0.6, alpha=0.7)
        
    ax.axvline(0, color="black", linestyle="-", alpha=0.4)

    
    for feature in top_features:
        y_center = feature_y_map[feature]
        # Calculate the absolute upper ceiling coordinate of the row
        y_top_boundary = y_center + (ROW_HEIGHT / 2)
        
        subset = df_plot[df_plot["Feature"] == feature]
        cat_groups = subset[~subset["Group_Type"].isin(["Ordinal", "Numeric/Bool"])]["Value_Grouped"].unique()
        raw_vals = subset[subset["Group_Type"].isin(["Ordinal", "Numeric/Bool"])]["Value"].unique()
        
        cat_groups = sorted(cat_groups, key=str)
        ords = sorted([x for x in raw_vals if x in ORDINAL_LEVELS_LIST], key=lambda x: ORDINAL_LEVELS_LIST.index(x))
        nums = sorted([x for x in raw_vals if x not in ORDINAL_LEVELS_LIST], key=str)
        
        items_to_show = list(cat_groups) + nums + ords

        legend_boxes = []
        for item in items_to_show:
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
            
            marker_char = markers_map.get(g_type, "X")
            
            # Canvas setup
            da = DrawingArea(14, 10, 0, 0)
            marker_style = MarkerStyle(marker_char)
            marker_path = marker_style.get_path()
            marker_transform = marker_style.get_transform()
            
            # Extract the complete transformed path layout to get the true shape boundaries
            transformed_path = marker_path.transformed(marker_transform)
            bbox = transformed_path.get_extents()
            
            # Calculate true exact geometric center coordinates
            true_center_x = (bbox.x0 + bbox.x1) / 2.0
            true_center_y = (bbox.y0 + bbox.y1) / 2.0
                            
            # Build clean linear transformation matrix pipeline:
            scale_factor = 7.0 if marker_char in ['s', 'D'] else 8.0
            Y_VISUAL_OFFSET = 0.5
            final_transform = (
                transforms.Affine2D()
                .translate(-true_center_x, -true_center_y)
                .scale(scale_factor)
                .translate(7.0, 5.0 + Y_VISUAL_OFFSET)
            )
            
            # Apply the centering/scaling matrix to the fully shaped path
            final_path = transformed_path.transformed(final_transform)
            
            patch = PathPatch(final_path, facecolor=color, edgecolor='white', linewidth=0.5)
            da.add_artist(patch)
            
            ta = TextArea(f" {str(item)}", textprops=dict(fontsize=9, color='#444'))
            item_box = HPacker(children=[da, ta], align="center", pad=0, sep=0)
            legend_boxes.append(item_box)
            
        rows_children = []
        current_row_items = []
        accumulated_width = 0
        max_legend_width_pixels = 560
        
        for i, box in enumerate(legend_boxes):
            label_text = box.get_children()[1].get_text()
            approx_text_width = len(label_text) * 5.8  
            box_width = 14 + approx_text_width + 16 
            
            if accumulated_width + box_width > max_legend_width_pixels:
                if len(rows_children) >= 1:
                    break
                rows_children.append(HPacker(children=current_row_items, align="center", pad=0, sep=14))
                current_row_items = [box]
                accumulated_width = box_width
            else:
                current_row_items.append(box)
                accumulated_width += box_width
                
        if current_row_items and len(rows_children) < 2:
            rows_children.append(HPacker(children=current_row_items, align="center", pad=0, sep=14))
            
        title_box = TextArea(feature, textprops=dict(fontsize=10.5, fontweight='bold', color='#111'))
        content_box = VPacker(children=[title_box] + rows_children, align="right", pad=0, sep=4)
        
        # Create a blended transform: X is Axes fraction (0 to 1), Y is Data coordinates
        blended_legend_transform = transforms.blended_transform_factory(ax.transAxes, ax.transData)
        # Align the right edge of the legend exactly to the left of the plot rectangle
        anchored_box = AnchoredOffsetbox(
            loc='upper right', child=content_box, pad=0, borderpad=0, frameon=False,
            bbox_to_anchor=(-0.02, y_top_boundary), bbox_transform=blended_legend_transform,
        )
        anchored_box.set_clip_on(False)
        ax.add_artist(anchored_box)

    plt.title(f"Detailed feature impact: {label_name}", fontsize=14, y=1.0)
    plt.savefig(output_dir / f"drivers_shap_{label_name}.png", dpi=300)
    plt.close()
    
    
def compute_feature_enrichment(target_indices, background_indices, dataset, top_k=20):
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
    model, dataset, collator, vocab, device, output_dir, label_name,
    fup_max, indices, target_idx=0, entity_filter=None, max_delta=None,
    top_k=20, min_freq=20, plot_only=None, agg_method="mean",
):
    print(f" -> Extracting attributions for {len(indices)} samples...")
    csv_path = output_dir / f"attributions_analysis_{label_name}.csv"
    
    df = extract_attributions(
        model, dataset, collator, indices, vocab, device,
        target_idx=target_idx, entity_filter=entity_filter,
        max_delta=max_delta, csv_path=csv_path, plot_only=plot_only,
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


if __name__ == "__main__":
    main()