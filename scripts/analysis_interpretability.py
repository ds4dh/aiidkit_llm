import yaml
import gc
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.transforms as transforms
import seaborn as sns
import hashlib
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from tqdm.auto import tqdm
from adjustText import adjust_text
from scipy.stats import fisher_exact
from torch.utils.data import DataLoader, Subset
from captum.attr import LayerIntegratedGradients
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
OUTPUT_DIR_BASE_NAME = RESULTS_DIR / Path("analysis/interpretability")
DATA_DIR = Path("/home/shares/ds4dh/aiidkit_project/data_new/processed/v3.6/teav")
CONFIG_PATH = Path("configs/discriminative_training.yaml")
FROM_OPTUNA = "optuna" in TRANSFORMER_BASE_DIR.name
DATA_SPLIT_TYPE = "temporal_split"
PLOT_ONLY = True  # run downstream plots directly from cache
MAX_LEGEND_VALUES_TO_SHOW = 5  # threshold capping distinct legend item limits

# Captum configuration
TOP_K = 15
MIN_FREQ = 20
MAX_DELTA = 0.10  # 0.05
AGG_METHOD = "mean"
NUM_CAPTUM_SAMPLES = 1000
NUM_CAPTUM_STEPS = 100

# -------------------------------------------------------------------------------------------------------
# THEORETICAL INTERPRETABILITY FRAMEWORK CONSTANTS
# -------------------------------------------------------------------------------------------------------
# True  -> value_embedding_hook      -> Completeness w.r.t the value sub-pathway (Best for SHAP beeswarm)
# False -> aggregated_embedding_hook -> Completeness across full token sequence (Best for overall burden)
# -------------------------------------------------------------------------------------------------------
ATTRIBUTIONS_TO_VALUES_ONLY = True
USE_TIME_NEUTRAL_BASELINE = True
OUTPUT_DIR = f"{OUTPUT_DIR_BASE_NAME}_{NUM_CAPTUM_STEPS}-steps_{int(100 * MAX_DELTA):03d}-delta_{ATTRIBUTIONS_TO_VALUES_ONLY}-values"

# Task configuration
TASK_CONFIG = {
    "bacteria_perioperative": {  
        "task": "infection_bacteria", 
        "horizon": 30, "fup_min": 0, "fup_max": 0, "fup_step": 30,
    },
    "bacteria_opportunistic": {  
        "task": "infection_bacteria",
        "horizon": 30, "fup_min": 30, "fup_max": 150, "fup_step": 30,
    },
    "bacteria_maintenance": {  
        "task": "infection_bacteria",
        "horizon": 30, "fup_min": 180, "fup_max": 330, "fup_step": 30,
    },
    "bacteria_long_term": {  
        "task": "infection_bacteria",
        "horizon": 30, "fup_min": 360, "fup_max": 690, "fup_step": 30,
    },
    "bacteria_very_long_term": {  
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
    'Surgery':                      ['Nephrectomy native', 'Nephrectomy allograft', 'Nephrectomy allograft and native', 'Non-Tx surgery', 'Tpx-related re-surgery', 'Events/Surgery', 'Non-transplant surgery'], 
    'Pregnancy/birth':              ['Birth', 'Pregnancy', 'Abortion/miscarriage'],
    'Emergency/critical event':     ['MOF', 'Agranulocytosis', 'GI haemorrhage', 'Bone fracture', 'Hemorrhagy', 'Thromboembolic disease'], 
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
    'Other / Unknown': [
        'Unknown', '[UNK]', 'Condition unknown',
        'Other', 'Other event or disease',
        'Missing', 'Site not identified', 'Undetermined',
        'Other / Remaining', 'Other / Unknown'
    ]
}

_UNKNOWN_VALUE_STRINGS = set(UNKNOWN_ROLES['Other / Unknown'])


def _score_label() -> str:
    """Helper to return description based on the theoretical layer evaluation setting."""
    if ATTRIBUTIONS_TO_VALUES_ONLY:
        return "Value-level attribution (Δ Probability)"
    return "Token-level attribution (Δ Probability)"


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
        return torch.sigmoid(outputs.logits)


def _build_baseline(ent, attr, val, days, pad_id=0, bos_id=2, time_neutral=False):
    current_pad = pad_id if time_neutral else 0

    baseline_ent = torch.full_like(ent, current_pad)
    baseline_attr = torch.full_like(attr, current_pad)
    baseline_val = torch.full_like(val, current_pad)
    
    baseline_ent[:, 0] = bos_id
    baseline_attr[:, 0] = bos_id
    baseline_val[:, 0] = bos_id
    
    baseline_days = days.clone() if time_neutral else torch.zeros_like(days)
    
    return baseline_ent, baseline_attr, baseline_val, baseline_days


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
    
    if ATTRIBUTIONS_TO_VALUES_ONLY:
        target_layer = model.patient_embedder.value_embedding_hook
    else:
        target_layer = model.patient_embedder.aggregated_embedding_hook
    
    lig = LayerIntegratedGradients(wrapper, target_layer)
    
    collected_data = []

    for batch in tqdm(loader, desc="Calculating attributions"):
        inp = batch["input_dict"]
        ent = inp["entity_id"].to(device)
        attr = inp["attribute_id"].to(device)
        val = inp["value_id"].to(device)
        days = inp["days_since_tpx"].to(device)
        args = (ent, attr, val, days)

        bos_token_id = vocab.get("[BOS]", 2)
        pad_token_id = vocab.get("[PAD]", 0)
        baseline_ent, baseline_attr, baseline_val, baseline_days = _build_baseline(
            ent=ent, 
            attr=attr, 
            val=val, 
            days=days,
            pad_id=pad_token_id, 
            bos_id=bos_token_id,
            time_neutral=USE_TIME_NEUTRAL_BASELINE,
        )
        baseline_args = (baseline_ent, baseline_attr, baseline_val, baseline_days)

        try:
            with torch.inference_mode(False): 
                attributions, delta = lig.attribute(
                    inputs=args,
                    baselines=baseline_args,  
                    target=target_idx,
                    n_steps=NUM_CAPTUM_STEPS,              
                    return_convergence_delta=True,
                    internal_batch_size=32,   
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
                continue  
            
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

                val_name_str = str(val_name).strip()
                if val_name_str in _UNKNOWN_VALUE_STRINGS:
                    val_name = 'Other / Unknown'
            
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

    explanation = "(blue = reduces predicted risk, red = increases predicted risk)"
    ax.set_title(
        f"Top {top_k} risk-increasing and risk-reducing drivers: {label_name}\n{explanation}",
        pad=20, fontsize=15, fontweight='bold',
    )
    ax.set_xlabel(_score_label(), fontsize=12)
    
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
    
    top_f = stats.nlargest(8, "Frequency")
    top_p = stats.nlargest(8, "Mean_Impact")
    top_n = stats.nsmallest(8, "Mean_Impact")
    top_b = stats.nlargest(8, "Total_Burden")
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
    plt.ylabel(_score_label(), fontsize=12)
    plt.grid(True, which="both", linestyle="--", alpha=0.2)
    plt.tight_layout()
    plt.savefig(output_dir / f"drivers_volcano_{label_name}.png", dpi=300)
    plt.close()


def get_deterministic_color_with_context(feature_name: str, string_value: str, palette_list: list) -> str:
    """Hashes feature layout and category context together to prevent intra-row duplicate styling."""
    combined_string = f"{str(feature_name)}|||{str(string_value)}"
    hasher = hashlib.md5(combined_string.encode('utf-8'))
    hash_int = int(hasher.hexdigest(), 16)
    palette_index = hash_int % len(palette_list)
    return palette_list[palette_index]


def plot_feature_value_impact(
    df, output_dir, label_name, fup_max, max_delta=None, top_k=20, min_freq=50,
):
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
        
        feat_df["Value"] = feat_df["Value"].apply(
            lambda x: "Other / Unknown" if str(x).strip() in _UNKNOWN_VALUE_STRINGS else x
        )
        val_counts = feat_df["Value"].value_counts()
        
        is_ordinal_feature = any(val_to_group.get(v) == "Ordinal level" for v in val_counts.index)
        
        if not is_ordinal_feature and len(val_counts) > 9:
            top_9_vals = val_counts.head(9).index.tolist()
            feat_df["Value"] = feat_df["Value"].apply(lambda v: v if v in top_9_vals else "Other / Unknown")
        
        adjusted_rows.append(feat_df)
        
    df_plot = pd.concat(adjusted_rows, ignore_index=True)

    unique_vals = df_plot["Value"].unique()
    unknowns = [v for v in unique_vals if v not in val_to_group]
    for v in unknowns:
        v_str = str(v).strip()
        if v_str.lower().startswith("other ") or v_str == "Other" or v_str in ['Other / Remaining', 'Other / Unknown']:
            val_to_group[v] = 'Other / Unknown'
        elif v_str in ORDINAL_LEVELS_LIST:
             val_to_group[v] = 'Ordinal level'
        else:
            val_to_group[v] = f"{v}_"

    df_plot["Value_Grouped"] = df_plot["Value"].map(val_to_group).fillna("Other / Unknown")
    df_plot.to_csv(csv_path, index=False)

    categorical_colors_pool = [
        '#e6194b', '#3cb44b', '#4363d8', '#f58231', '#911eb4',
        '#26a69a', '#f032e6', '#b29e21', "#8C5B49", "#8D8D8D",
    ]
    
    num_manual_map = {
        'No / Male': "#D11D14", 'Yes / Female / Occured': "#203ed1",
        'Value 2': "#4082b7", 'Value 3': "#54a4a1", 'Value 4': "#88f3cf",
    }
    
    vir_cmap = plt.get_cmap("viridis", len(ORDINAL_LEVELS_LIST))
    ord_colors = {lvl: mcolors.to_hex(vir_cmap(i)) for i, lvl in enumerate(ORDINAL_LEVELS_LIST)}
    unknown_color = "#333333"

    # Contextual multi-index master palette to ensure uniqueness across variables inside a single row legend box
    master_palette = {}
    for feature in top_features:
        subset = df_plot[df_plot["Feature"] == feature]
        unique_vals_in_row = subset["Value"].unique()
        used_colors_in_row = set()
        
        for entry_value in unique_vals_in_row:
            grp = val_to_group.get(entry_value, entry_value)
            if grp == "Ordinal level" or entry_value in ORDINAL_LEVELS_LIST:
                color = ord_colors.get(entry_value, unknown_color)
            elif grp in NUMERIC_ROLES:
                color = num_manual_map.get(grp, unknown_color)
            elif grp in UNKNOWN_ROLES or grp == 'Other / Unknown':
                color = unknown_color
            else:
                color = get_deterministic_color_with_context(feature, entry_value, categorical_colors_pool)
                # Resolve intra-row matching color collisions through deterministic rotation shifts
                if color in used_colors_in_row:
                    try:
                        start_idx = categorical_colors_pool.index(color)
                        for offset in range(1, len(categorical_colors_pool)):
                            alt_idx = (start_idx + offset) % len(categorical_colors_pool)
                            alt_color = categorical_colors_pool[alt_idx]
                            if alt_color not in used_colors_in_row:
                                color = alt_color
                                break
                    except ValueError:
                        pass
                used_colors_in_row.add(color)
                
            master_palette[(feature, entry_value)] = color

    def get_value_type(group_name):
        if group_name == 'Ordinal level': return "Ordinal"
        if group_name in NUMERIC_ROLES: return "Numeric/Bool"
        if group_name in UNKNOWN_ROLES or group_name == "Other / Unknown": return "Other"
        return "Categorical"
    
    df_plot["Group_Type"] = df_plot["Value_Grouped"].apply(get_value_type)
    markers_map = {"Ordinal": "o", "Numeric/Bool": "D", "Categorical": "s", "Other": "X"}

    # --- ROW SPACING CONFIGURATION ---
    ROW_HEIGHT = 1.25
    ROW_PADDING = 0.65
    
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

    bottom_line_y = feature_y_map[top_features[0]] - (ROW_HEIGHT / 2) - (ROW_PADDING / 2)
    top_line_y = feature_y_map[top_features[-1]] + (ROW_HEIGHT / 2) + (ROW_PADDING / 2)
    
    SYMMETRY_PADDING = 0.30
    y_min_limit = bottom_line_y - SYMMETRY_PADDING
    total_y_max = top_line_y + SYMMETRY_PADDING
    
    fig_height = max(5, total_y_max * 0.40) 
    fig = plt.figure(figsize=(10.5, fig_height))
    
    gs = fig.add_gridspec(1, 1, left=0.55, right=0.85, top=0.98, bottom=0.14)
    ax = fig.add_subplot(gs[0, 0])
    ax.set_ylim(y_min_limit, total_y_max)
    
    # Map multi-index key coordinates configuration to extract distinct labels
    df_plot["Color_Key"] = list(zip(df_plot["Feature"], df_plot["Value"]))
    
    sizes_map_scatter = {"Ordinal": 45, "Numeric/Bool": 35, "Categorical": 40, "Other": 50}
    combo_counts = df_plot.groupby(["Feature", "Value"]).size().reset_index(name="Combo_Frequency")
    df_plot = df_plot.merge(combo_counts, on=["Feature", "Value"], how="left")  
    df_plot = df_plot.sort_values(by="Combo_Frequency", ascending=False).reset_index(drop=True)    
    
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
    ax.set_xlabel(_score_label(), fontsize=11, labelpad=4)
    
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
        y_top_boundary = y_center + (ROW_HEIGHT / 2)
        
        subset = df_plot[df_plot["Feature"] == feature]
        is_ordinal_feature = any(val_to_group.get(v) == "Ordinal level" for v in subset["Value"].unique())
        
        if is_ordinal_feature:
            named_items = sorted([v for v in subset["Value"].unique() if v in ORDINAL_LEVELS_LIST], key=lambda x: ORDINAL_LEVELS_LIST.index(x))
            has_leftovers = False
        else:
            item_counts = subset["Value"].value_counts()
            filtered_item_counts = item_counts.drop("Other / Unknown", errors="ignore")
            named_items = filtered_item_counts.head(MAX_LEGEND_VALUES_TO_SHOW).index.tolist()
            has_leftovers = (len(item_counts) > MAX_LEGEND_VALUES_TO_SHOW) or ("Other / Unknown" in item_counts.index)
        
        legend_boxes = []
        for item in named_items:
            group_name = val_to_group.get(item, item)
            g_type = get_value_type(group_name)
            
            # Fetch context isolated identifier coordinate
            color = master_palette.get((feature, item), unknown_color)
            
            marker_char = markers_map.get(g_type, "X")
            da = DrawingArea(14, 10, 0, 0)
            marker_style = MarkerStyle(marker_char)
            transformed_path = marker_style.get_path().transformed(marker_style.get_transform())
            bbox = transformed_path.get_extents()
            
            true_center_x = (bbox.x0 + bbox.x1) / 2.0
            true_center_y = (bbox.y0 + bbox.y1) / 2.0
                                            
            scale_factor = 7.0 if marker_char in ['s', 'D'] else 8.0
            final_transform = (
                transforms.Affine2D()
                .translate(-true_center_x, -true_center_y)
                .scale(scale_factor)
                .translate(7.0, 5.5)
            )
            
            patch = PathPatch(transformed_path.transformed(final_transform), facecolor=color, edgecolor='white', linewidth=0.5)
            da.add_artist(patch)
            
            ta = TextArea(f" {str(item)}", textprops=dict(fontsize=8.5, color='#444'))
            legend_boxes.append(HPacker(children=[da, ta], align="center", pad=0, sep=0))
            
        if has_leftovers:
            da = DrawingArea(14, 10, 0, 0)
            final_transform = transforms.Affine2D().translate(-0.0, -0.0).scale(7.0).translate(7.0, 5.5)
            patch = PathPatch(MarkerStyle("X").get_path().transformed(final_transform), facecolor="#777777", edgecolor='white', linewidth=0.5)
            da.add_artist(patch)
            ta = TextArea(" Other / Unknown", textprops=dict(fontsize=8.5, color='#666', fontstyle='italic'))
            legend_boxes.append(HPacker(children=[da, ta], align="center", pad=0, sep=0))

        rows_children = []
        n_elements = len(legend_boxes)
        
        if n_elements > 0:
            base_items_per_row = n_elements // 2
            rem = n_elements % 2
            
            idx_ptr = 0
            for r_sub in range(2):
                allocated_size = base_items_per_row + (1 if r_sub < rem else 0)
                if allocated_size == 0:
                    continue
                
                sub_row_chunk = legend_boxes[idx_ptr : idx_ptr + allocated_size]
                idx_ptr += allocated_size
                rows_children.append(HPacker(children=sub_row_chunk, align="center", pad=0, sep=10))
                
        while len(rows_children) < 2:
            rows_children.append(HPacker(children=[], align="center", pad=0, sep=0))
            
        title_box = TextArea(feature, textprops=dict(fontsize=10.5, fontweight='bold', color='#111'))
        content_box = VPacker(children=[title_box] + rows_children, align="right", pad=0, sep=3)
        
        blended_legend_transform = transforms.blended_transform_factory(ax.transAxes, ax.transData)
        anchored_box = AnchoredOffsetbox(
            loc='upper right', child=content_box, pad=0, borderpad=0, frameon=False,
            bbox_to_anchor=(-0.05, y_top_boundary), bbox_transform=blended_legend_transform,
        )
        anchored_box.set_clip_on(False)
        ax.add_artist(anchored_box)
    
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
        df, output_dir, label_name, top_k=top_k,
        min_freq=min_freq, agg_method=agg_method,
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