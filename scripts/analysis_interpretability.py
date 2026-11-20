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
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.transforms as transforms
import seaborn as sns
import hashlib
import warnings
warnings.filterwarnings("ignore")

import json
from tqdm.auto import tqdm
from adjustText import adjust_text
from torch.utils.data import DataLoader, Subset
from captum.attr import LayerIntegratedGradients
from matplotlib.patches import PathPatch
from matplotlib.markers import MarkerStyle
from matplotlib.offsetbox import AnchoredOffsetbox, HPacker, VPacker, TextArea, DrawingArea

from src.data.patient_dataset import load_hf_data_and_metadata
from src.model.model_utils import build_monotonic_groups
from src.model.patient_embedder import PatientEmbeddingModelFactory, PatientDataCollatorForClassification
from src.tasks import abbreviate_feature_name, format_eav_feature_name
from scripts.script_utils import (
    find_best_checkpoint,
    find_best_checkpoint_multitask,
    extract_horizons_from_path,
    resolve_base_dir_for_ckpt,
    scan_all_fups,
)

GENERATE_SANITIZE_PLOTS = False

# =================================
# GLOBAL MAPPINGS AND CONFIGURATION
# =================================

# Parse CLI arguments
parser = argparse.ArgumentParser(description="Run Captum attribution analysis.")
parser.add_argument("--results-dir", type=Path, default=Path("results"), help="Base directory for results")
parser.add_argument("--data-split-type", "--data_split_type", type=str, default="random_split", help="Data split strategy (e.g., random_split, temporal_split)")
parser.add_argument("--top-k", type=int, default=15, help="Top K features to analyze/display")
parser.add_argument("--min-freq", type=int, default=20, help="Minimum feature frequency threshold")
parser.add_argument("--max-delta", type=float, default=0.10, help="Maximum allowed Captum delta error")
parser.add_argument("--agg-method", type=str, default="mean", choices=["mean", "sum"], help="Aggregation method")
parser.add_argument("--num-samples", type=int, default=1000, help="Number of samples to analyze")
parser.add_argument("--num-steps", type=int, default=200, help="Number of Integrated Gradients steps")
parser.add_argument("--data-dir", "--data_dir", type=Path, default=Path(os.environ.get("TEAV_DATA_DIR", "/home/shares/ds4dh/aiidkit_project/data_new/processed/v3.9/teav")), help="Path to tEAV dataset directory")
parser.add_argument("--proposed-model", "--proposed_model", type=str, default="tf_raw_mt",
                    help="Model analyzed by this script (tf_raw_mt, tf_raw_st, tf_bin_mt, tf_bin_st).")
parser.add_argument("--run-id", "--run_id", type=str, default="",
                    help="Pre-training run id subfolder (e.g. e00-a15-v60). Default: the only folder present.")
parser.add_argument("--config", "-c", type=str, default="configs/discriminative_training.yaml",
                    help="Training config providing model/collator settings for checkpoint loading.")
parser.add_argument("--task", "-t", type=str, default=None,
                    help="Single outcome task to analyze (e.g. infection_bacteria or infection_grp_cmv). Default: infection_bacteria.")
parser.add_argument("--tasks", nargs="+", default=None,
                    help="Outcome tasks to attribute (each gets the five clinical-window "
                         "configs). Default: infection_bacteria. Multi-task checkpoints are "
                         "supported: the matching (task, horizon) head is selected per task. "
                         "E.g. --tasks infection_bacteria infection_virus infection_grp_cmv")
parser.add_argument("--horizon", type=int, default=None,
                    help="Prediction horizon in days to attribute (default: 30 for infections, 360 for death/graft_loss, or first available checkpoint horizon).")
parser.add_argument("--binning-mode", "--binning_mode", type=str, default="slice", choices=["slice", "global"],
                    help="How numerical feature values are binned for SHAP/bar plots: "
                         "'slice' (default) re-bins numerical values relative to the active clinical window slice; "
                         "'global' uses the whole-dataset pretraining quantile bins.")
args, _ = parser.parse_known_args()

# Run configuration
RESULTS_DIR = args.results_dir
DATA_SPLIT_TYPE = args.data_split_type
PROPOSED_MODEL = args.proposed_model
RUN_ID = args.run_id.strip()
BINNING_MODE = args.binning_mode

# Captum attribution runs directly on the model checkpoint, so only
# transformer-family models with a binary classification head can be
# interpreted here (classic ML contenders have no checkpoints/embeddings,
# and the Cox survival variant uses a regression head, not wired in yet).
INTERPRETABLE_MODELS = {
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
    "tfg_bin_st": ("generative_transformer", False, "bin", "any"),
    "tfg_bin_mt": ("generative_transformer", True, "bin", "domain"),
    "tfg_bin_unified": ("generative_transformer", True, "bin", "unified"),
}
if PROPOSED_MODEL not in INTERPRETABLE_MODELS:
    print(f"[SKIP] Interpretability analysis supports only {sorted(INTERPRETABLE_MODELS)}; "
          f"got proposed model '{PROPOSED_MODEL}'. Nothing to do.")
    sys.exit(0)
TRANSFORMER_BASE_DIR = RESULTS_DIR / INTERPRETABLE_MODELS[PROPOSED_MODEL][0]
MULTITASK_ONLY = INTERPRETABLE_MODELS[PROPOSED_MODEL][1]
MODEL_VAL_TAG = INTERPRETABLE_MODELS[PROPOSED_MODEL][2]
MT_SCOPE = INTERPRETABLE_MODELS[PROPOSED_MODEL][3]
OUTPUT_DIR_BASE_NAME = RESULTS_DIR / "analysis" / "interpretability" / PROPOSED_MODEL
DATA_DIR = args.data_dir
CONFIG_PATH = Path(args.config)
FROM_OPTUNA = "optuna" in TRANSFORMER_BASE_DIR.name
PLOT_ONLY = False  # run downstream plots directly from cache
MAX_LEGEND_VALUES_TO_SHOW = 5  # threshold capping distinct legend item limits

# Captum configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TOP_K = args.top_k
MIN_FREQ = args.min_freq
MAX_DELTA = args.max_delta
AGG_METHOD = args.agg_method
NUM_CAPTUM_SAMPLES = args.num_samples
NUM_CAPTUM_STEPS = args.num_steps

# -------------------------------------------------------------------------------------------------------
# THEORETICAL INTERPRETABILITY FRAMEWORK CONSTANTS
# -------------------------------------------------------------------------------------------------------
# True  -> value_embedding_hook      -> Completeness w.r.t the value sub-pathway (Best for SHAP beeswarm)
# False -> aggregated_embedding_hook -> Completeness across full token sequence (Best for overall burden)
# -------------------------------------------------------------------------------------------------------
ATTRIBUTIONS_TO_VALUES_ONLY = True
USE_TIME_NEUTRAL_BASELINE = True
OUTPUT_SUBDIR = f"{DATA_SPLIT_TYPE}-{NUM_CAPTUM_STEPS}-steps_{int(100 * MAX_DELTA):03d}-delta_{ATTRIBUTIONS_TO_VALUES_ONLY}-values"
OUTPUT_DIR = OUTPUT_DIR_BASE_NAME / OUTPUT_SUBDIR

# Task configuration (landmark FUP bounds aligned identically with analysis_comparison.py)
MAX_FUP = 2400 if DATA_SPLIT_TYPE == "temporal_split" else 3600
_WINDOWS = {
    "perioperative": (0, 30),
    "opportunistic": (31, 180),
    "maintenance": (181, 360),
    "long_term": (361, 1080),
    "very_long_term": (1081, MAX_FUP),
}
_selected_tasks = [args.task] if args.task else (list(args.tasks) if args.tasks else ["infection_bacteria"])
TASK_CONFIG = {
    f"{t.removeprefix('infection_').removeprefix('grp_') or t}_{w}": {
        "task": t,
        "horizon": int(args.horizon) if args.horizon is not None else (360 if t in ("death", "graft_loss") else 30),
        "fup_min": lo,
        "fup_max": hi,
        "fup_step": 30,
    }
    for t in _selected_tasks for w, (lo, hi) in _WINDOWS.items()
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
    'Yes / Female / Occurred':      ['1', 'Yes', 'True', 'Occurred', 'Positive', 'Female', 'F'],
    'Value 2':                      ['2'],
    'Value 3':                      ['3'],
    'Value 4':                      ['4'],
}

# Ordinal (intensity)
CORE_ORDINAL_7 = ['Lowest', 'Lower', 'Low', 'Middle', 'High', 'Higher', 'Highest']
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


def _resolve_base_dir_for_ckpt(task_key: str) -> Path:
    base_dir, _ = resolve_base_dir_for_ckpt(
        TRANSFORMER_BASE_DIR, DATA_SPLIT_TYPE, RUN_ID, MODEL_VAL_TAG
    )
    return base_dir


def main():
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)
    use_num_emb = (MODEL_VAL_TAG == "raw")
    config.setdefault("model", {}).setdefault("embedding_layer_config", {})["use_numerical_embedding"] = use_num_emb
    config.setdefault("data_collator", {})["use_numerical_embedding"] = use_num_emb
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    avail_fups = set(scan_all_fups(DATA_DIR / DATA_SPLIT_TYPE))
    resolved_runs = []
    union_fups: set[int] = set()
    union_label_keys: list[str] = []

    for config_name, task_params in TASK_CONFIG.items():
        task_key = task_params["task"]
        hrz = task_params["horizon"]
        fup_min = task_params["fup_min"]
        fup_max = task_params["fup_max"]
        fup_step = task_params["fup_step"]
        if avail_fups:
            fups = [f for f in sorted(avail_fups) if fup_min <= f <= fup_max and f % fup_step == 0]
        else:
            start_fup = ((fup_min + fup_step - 1) // fup_step) * fup_step
            fups = list(range(start_fup, fup_max + 1, fup_step))
        if not fups:
            print(f"[Skip] No available FUP folders in [{fup_min}, {fup_max}] for '{config_name}'")
            continue

        try:
            base_dir_for_ckpt = _resolve_base_dir_for_ckpt(task_key)
        except Exception as e:
            print(f"[Error] Could not resolve base directory for {task_key}: {e}")
            continue

        try:
            if MULTITASK_ONLY:
                checkpoint_path, run_meta = find_best_checkpoint_multitask(
                    base_dir_for_ckpt, task_key, hrz, scope=MT_SCOPE
                )
            else:
                try:
                    checkpoint_path = find_best_checkpoint(base_dir_for_ckpt, task_key, hrz)
                    run_meta = None
                except FileNotFoundError:
                    checkpoint_path, run_meta = find_best_checkpoint_multitask(
                        base_dir_for_ckpt, task_key, hrz, scope=MT_SCOPE
                    )
        except Exception as e:
            print(f"[Error] Checkpoint error for '{config_name}': {e}")
            continue

        ckpt_hrzs = (
            list(run_meta.get("task_horizons", {}).get(task_key, run_meta["horizons"]))
            if run_meta
            else extract_horizons_from_path(checkpoint_path)
        )
        if hrz not in ckpt_hrzs:
            if args.horizon is None and ckpt_hrzs:
                hrz = int(ckpt_hrzs[0])
            else:
                print(f"[Error] Requested horizon {hrz} not found in: {ckpt_hrzs}")
                continue

        label_key = f"label_{task_key}_{hrz:04d}d"
        if run_meta:
            if label_key not in run_meta.get("label_keys", []):
                print(f"[Error] {label_key} is not a head of multi-task run '{run_meta['task_key']}'")
                continue
            target_idx = run_meta["label_keys"].index(label_key)
            all_required_labels = list(run_meta["label_keys"])
        else:
            target_idx = ckpt_hrzs.index(hrz)
            all_required_labels = [f"label_{task_key}_{h:04d}d" for h in ckpt_hrzs]

        union_fups.update(fups)
        for lk in all_required_labels:
            if lk not in union_label_keys:
                union_label_keys.append(lk)

        resolved_runs.append({
            "config_name": config_name,
            "task_key": task_key,
            "hrz": hrz,
            "fup_min": fup_min,
            "fup_max": fup_max,
            "fup_step": fup_step,
            "fups": fups,
            "checkpoint_path": checkpoint_path,
            "run_meta": run_meta,
            "ckpt_hrzs": ckpt_hrzs,
            "label_key": label_key,
            "target_idx": target_idx,
            "all_required_labels": all_required_labels,
        })

    if not resolved_runs:
        print("[Error] No valid interpretability configurations could be resolved.")
        sys.exit(1)

    print(f"\n>>> Loading dataset once for {len(union_fups)} FUPs and {len(union_label_keys)} label keys...")
    data_dir_split = DATA_DIR / DATA_SPLIT_TYPE
    sanity_dir = OUTPUT_DIR / "sanity_check" if GENERATE_SANITIZE_PLOTS else None

    dataset, _, vocab = load_hf_data_and_metadata(
        data_dir=data_dir_split,
        fup_train=[360],
        fup_valid=[360],
        fup_test=sorted(union_fups),
        label_keys=union_label_keys,
        time_mapping=config["data_collator"].get("time_mapping", None),
        eav_mappings=config["data_collator"].get("eav_mappings", None),
        sanity_check_output_dir=sanity_dir,
    )
    full_test_ds = dataset["test"].add_column("split", ["test"] * len(dataset["test"]))
    test_fup_arr = np.asarray(full_test_ds.data.column("fup"))

    cached_ckpt_path: Path | None = None
    cached_model = None

    for item in resolved_runs:
        config_name = item["config_name"]
        task_key = item["task_key"]
        hrz = item["hrz"]
        fup_min = item["fup_min"]
        fup_max = item["fup_max"]
        fup_step = item["fup_step"]
        fups = item["fups"]
        checkpoint_path = item["checkpoint_path"]
        run_meta = item["run_meta"]
        ckpt_hrzs = item["ckpt_hrzs"]
        label_key = item["label_key"]
        target_idx = item["target_idx"]
        all_required_labels = item["all_required_labels"]

        print(f"\n{'='*60}")
        print(f">>> Executing Config: '{config_name}' (Task: {task_key}, Horizon: {hrz})")
        print(f"{'='*60}")
        print(f">>> Using Checkpoint: {checkpoint_path}"
              + (f"  [multi-task run '{run_meta['task_key']}', head task '{task_key}']" if run_meta else ""))
        print(f">>> Target index: {target_idx} (horizon: {hrz}d)")

        test_lbl_arr = np.asarray(full_test_ds.data.column(label_key))
        valid_raw_idx = np.flatnonzero(np.isin(test_fup_arr, fups) & (test_lbl_arr != -100))
        if len(valid_raw_idx) == 0:
            print(f"[Skip] No valid test samples in FUPs {fups} for '{config_name}'")
            continue

        if len(valid_raw_idx) > NUM_CAPTUM_SAMPLES:
            print(f"\n>>> Subsampling {NUM_CAPTUM_SAMPLES} patients from {len(valid_raw_idx)} valid window samples.")
            rng = np.random.default_rng(42)
            chosen_raw_idx = np.sort(rng.choice(valid_raw_idx, NUM_CAPTUM_SAMPLES, replace=False))
        else:
            print(f"\n>>> Analyzing all {len(valid_raw_idx)} valid window samples.")
            chosen_raw_idx = valid_raw_idx

        test_ds = full_test_ds.select(chosen_raw_idx.tolist(), keep_in_memory=True)
        selected_idx = np.arange(len(test_ds))

        if cached_model is None or cached_ckpt_path != checkpoint_path:
            if cached_model is not None:
                del cached_model
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            model_cfg_base = config["model"].copy()
            model_cfg_base["model_args"] = model_cfg_base.get("model_args", {}).copy()
            model_cfg_base["model_args"].pop("num_labels", None)
            model_cfg_base["model_args"].pop("problem_type", None)

            # Run Integrated Gradients in float32 so 200-step Riemann integration does not accumulate bfloat16 rounding error
            target_dtype = torch.float32
            model_cfg_base["model_args"]["dtype"] = torch.float32
            model_cfg_base["model_args"].pop("torch_dtype", None)

            emb_cfg = model_cfg_base["embedding_layer_config"].copy()
            emb_cfg["vocab_size"] = len(vocab)

            enforce_monotonicity = build_monotonic_groups(
                run_meta, config.get("finetuner", {}).get("enforce_monotonicity", False)
            )
            cached_model = PatientEmbeddingModelFactory.from_pretrained(
                enforce_monotonicity=enforce_monotonicity,
                task="classification",
                pretrained_dir=str(checkpoint_path),
                embedding_layer_config=emb_cfg,
                model_args=model_cfg_base["model_args"],
                config_args=model_cfg_base.get("config_args", {}),
                architecture_type=model_cfg_base.get("architecture_type"),
            ).to(device=DEVICE, dtype=target_dtype)
            cached_ckpt_path = checkpoint_path

        collator = PatientDataCollatorForClassification(
            **config["data_collator"],
            label_keys=all_required_labels,
            max_position_embeddings=cached_model.config.max_position_embeddings,
        )

        label_name = f"{config_name}_hrz{hrz:04d}d_fup{fup_min}-{fup_max}-{fup_step}".lower()
        ckpt_sig = f"{checkpoint_path}:{int(checkpoint_path.stat().st_mtime_ns)}"
        run_captum_analysis(
            model=cached_model, dataset=test_ds, collator=collator, OUTPUT_DIR=OUTPUT_DIR,
            label_name=label_name, fup_max=fup_max, indices=selected_idx, vocab=vocab,
            device=DEVICE, target_idx=target_idx, max_delta=MAX_DELTA, top_k=TOP_K,
            plot_only=PLOT_ONLY, agg_method=AGG_METHOD, min_freq=MIN_FREQ, ckpt_sig=ckpt_sig,
        )

    del cached_model, dataset, full_test_ds
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"\nSuccess! All results can be found in {OUTPUT_DIR}")


class ForwardWrapperForCaptum(torch.nn.Module):
    def __init__(self, model, pad_id: int = 0):
        super().__init__()
        self.model = model
        self.pad_id = pad_id
        self.fixed_attention_mask = None
    
    def forward(self, entity_id, attribute_id, value_id, days_since_tpx, value_num=None):
        input_dict = {
            "entity_id": entity_id,
            "attribute_id": attribute_id,
            "value_id": value_id,
            "days_since_tpx": days_since_tpx,
        }
        if value_num is not None:
            input_dict["value_num"] = value_num
        attention_mask = (
            self.fixed_attention_mask.repeat(max(1, entity_id.shape[0] // self.fixed_attention_mask.shape[0]), 1)[:entity_id.shape[0]]
            if self.fixed_attention_mask is not None
            else ((entity_id != self.pad_id) | (attribute_id != self.pad_id) | (value_id != self.pad_id)).long()
        )
        outputs = self.model(input_dict=input_dict, attention_mask=attention_mask)
        return torch.sigmoid(outputs.logits)


def _build_baseline(ent, attr, val, days, pad_id=0, bos_id=2, time_neutral=False, values_only=False):
    current_pad = pad_id if time_neutral else 0
    if values_only:
        # Preserve all special structural tokens (< 5: PAD, MASK, BOS, EOS, SEP) so EOS/last-token pooling stays intact
        baseline_val = torch.where(val < 5, val, torch.full_like(val, current_pad))
        return ent.clone(), attr.clone(), baseline_val, (days.clone() if time_neutral else torch.zeros_like(days))

    baseline_ent = torch.where(ent < 5, ent, torch.full_like(ent, current_pad))
    baseline_attr = torch.where(attr < 5, attr, torch.full_like(attr, current_pad))
    baseline_val = torch.where(val < 5, val, torch.full_like(val, current_pad))
    baseline_days = days.clone() if time_neutral else torch.zeros_like(days)

    return baseline_ent, baseline_attr, baseline_val, baseline_days


def apply_slice_relative_binning(df: pd.DataFrame, mode: str = "slice") -> pd.DataFrame:
    """
    Re-bins numerical attributes (`Value_Num` is finite) locally within the current
    attribution slice `df` (per `Feature`), mapping them onto `CORE_ORDINAL_7`
    (['Lowest', 'Lower', 'Low', 'Middle', 'High', 'Higher', 'Highest']):
      - If `mode == 'global'` or `Value_Num` is not present, preserves global bins.
      - If all values of a feature in the slice are identical (`n_unique == 1`, e.g.,
        transplant age at a single perioperative FUP=0), smoothly keeps the single
        global bin label without artificial splitting.
      - If a feature has `2 <= n_unique <= 7` distinct values in the slice (e.g., 5 FUP
        steps in the opportunistic window), maps each distinct ordered value to evenly
        spaced ordinal labels spanning `'Lowest'`..`'Highest'`.
      - If `n_unique > 7`, applies 7-quantile binning (`pd.qcut(..., duplicates='drop')`)
        within the slice and maps the resulting bins onto `'Lowest'`..`'Highest'`.
    """
    if df.empty or "Value" not in df.columns:
        return df
    df = df.copy()
    if "Value_Global" not in df.columns:
        df["Value_Global"] = df["Value"]
    if mode != "slice" or "Value_Num" not in df.columns:
        df["Value"] = df["Value_Global"]
        return df

    new_values = df["Value_Global"].astype(object).to_numpy().copy()
    val_nums = pd.to_numeric(df["Value_Num"], errors="coerce").to_numpy(dtype=np.float64)
    finite_mask = np.isfinite(val_nums)
    if not finite_mask.any():
        return df

    features = df["Feature"].to_numpy()
    for feat in pd.unique(features[finite_mask]):
        row_idx = np.flatnonzero((features == feat) & finite_mask)
        if len(row_idx) == 0:
            continue
        u = np.round(val_nums[row_idx], 6)
        uniq, inv = np.unique(u, return_inverse=True)
        n_uniq = len(uniq)

        # Smooth fallback when all values in this window slice are identical (e.g., FUP=0 only)
        if n_uniq <= 1:
            continue

        if n_uniq <= 7:
            # Map each distinct ordered value in the slice directly to evenly spaced ordinal labels
            label_indices = np.round(np.linspace(0, len(CORE_ORDINAL_7) - 1, n_uniq)).astype(int)
            mapped = np.asarray([CORE_ORDINAL_7[label_indices[k]] for k in inv], dtype=object)
            new_values[row_idx] = mapped
        else:
            try:
                cats = pd.qcut(u, q=len(CORE_ORDINAL_7), labels=False, duplicates="drop")
                cats_np = np.asarray(cats, dtype=int)
                n_bins_actual = int(cats_np.max()) + 1
                if n_bins_actual <= 1:
                    continue
                label_indices = np.round(np.linspace(0, len(CORE_ORDINAL_7) - 1, n_bins_actual)).astype(int)
                mapped = np.asarray([CORE_ORDINAL_7[label_indices[c]] for c in cats_np], dtype=object)
                new_values[row_idx] = mapped
            except ValueError:
                continue

    df["Value_Slice"] = new_values
    df["Value"] = df["Value_Slice"]
    return df


def extract_attributions(
    model, dataset, collator, indices, vocab, device, target_idx=0,
    entity_filter=None, max_delta=None,
    csv_path=None, plot_only=False, binning_mode: str = BINNING_MODE,
    ckpt_sig: str = "",
) -> pd.DataFrame:
    meta_sidecar = csv_path.with_suffix(".meta.json") if csv_path is not None else None
    if csv_path is not None and csv_path.exists():
        if plot_only:
            print(f" -> Found existing attribution CSV ({csv_path.name}); reusing cached attributions...")
            return apply_slice_relative_binning(pd.read_csv(csv_path), mode=binning_mode)
        if os.environ.get("RESUME_ATTRIBUTIONS", "1") == "1" and meta_sidecar is not None and meta_sidecar.exists():
            try:
                cached_meta = json.loads(meta_sidecar.read_text())
                if cached_meta.get("ckpt_sig") == ckpt_sig and cached_meta.get("target_idx") == target_idx:
                    print(f" -> Verified matching checkpoint cache ({csv_path.name}); reusing attributions...")
                    return apply_slice_relative_binning(pd.read_csv(csv_path), mode=binning_mode)
            except Exception:
                pass

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
    n_skipped_delta = 0
    n_total_patients = 0

    for batch in tqdm(loader, desc="Calculating attributions"):
        inp = batch["input_dict"]
        ent = inp["entity_id"].to(device)
        attr = inp["attribute_id"].to(device)
        val = inp["value_id"].to(device)
        days = inp["days_since_tpx"].to(device)
        val_num = inp["value_num"].to(device) if "value_num" in inp else None

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
            values_only=ATTRIBUTIONS_TO_VALUES_ONLY,
        )
        wrapper.fixed_attention_mask = ((ent != pad_token_id) | (attr != pad_token_id) | (val != pad_token_id)).long()
        if val_num is not None:
            args = (ent, attr, val, days, val_num)
            baseline_val_num = torch.full_like(val_num, float("nan"))
            baseline_args = (baseline_ent, baseline_attr, baseline_val, baseline_days, baseline_val_num)
        else:
            args = (ent, attr, val, days)
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
        val_num_np = args[4].detach().float().cpu().numpy() if len(args) > 4 else None
        mask_np = ((ent_np != 0) | (attr_np != 0) | (val_np != 0)).astype(int)
        delta_np = delta.detach().cpu().numpy()

        for i in range(ent_np.shape[0]):
            n_total_patients += 1
            patient_delta = float(abs(delta_np[i]))
            
            if max_delta is not None and patient_delta > max_delta:
                n_skipped_delta += 1
                continue  
            
            length = int(mask_np[i].sum())
            seq_ent = ent_np[i, :length]
            seq_attr = attr_np[i, :length]
            seq_val = val_np[i, :length]
            seq_score = attrs_sum[i, :length]
            seq_days = days_np[i, :length]
            seq_val_num = val_num_np[i, :length] if val_num_np is not None else [np.nan] * length
            
            for k, (e_id, a_id, v_id, score, day, v_num) in enumerate(
                zip(seq_ent, seq_attr, seq_val, seq_score, seq_days, seq_val_num)
            ):
                if max(e_id, a_id, v_id) < 5: continue
                ent_name = id2word.get(e_id, f"Ent_{e_id}")
                if entity_filter is not None and not ent_name.startswith(entity_filter):
                    continue
                attr_name = id2word.get(a_id, f"Attr_{a_id}")
                full_feature_name = format_eav_feature_name(ent_name, attr_name, entity_id=int(e_id))
                val_name = id2word.get(v_id, f"Val_{v_id}")

                val_name_str = str(val_name).strip()
                if val_name_str in _UNKNOWN_VALUE_STRINGS:
                    val_name = 'Other / Unknown'
            
                collected_data.append({
                    "Feature": full_feature_name,
                    "Value": val_name,
                    "Value_Num": float(v_num) if np.isfinite(v_num) else np.nan,
                    "Score": float(score),
                    "Delta": patient_delta,
                    "Days": float(day)
                })

    if n_skipped_delta > 0:
        print(f"   [Info] Skipped {n_skipped_delta}/{n_total_patients} patients exceeding max_delta={max_delta}")
    df = pd.DataFrame(collected_data)
    df = apply_slice_relative_binning(df, mode=binning_mode)
    if csv_path:
        df.to_csv(csv_path, index=False)
        if meta_sidecar is not None:
            meta_sidecar.write_text(json.dumps({"ckpt_sig": ckpt_sig, "target_idx": int(target_idx)}, indent=2))
    
    return df


def plot_feature_importance(df, OUTPUT_DIR, label_name, top_k=20, min_freq=50, agg_method="mean"):
    if df.empty: return
    csv_path = OUTPUT_DIR / f"drivers_bar_{label_name}_{agg_method}.csv"

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
    plt.savefig(OUTPUT_DIR / f"drivers_bar_{label_name}.png", dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_frequency_vs_impact(df, OUTPUT_DIR, label_name, min_freq=50):
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
    plt.savefig(OUTPUT_DIR / f"drivers_volcano_{label_name}.png", dpi=300)
    plt.close()


def get_deterministic_color_with_context(feature_name: str, string_value: str, palette_list: list) -> str:
    """Hashes feature layout and category context together to prevent intra-row duplicate styling."""
    combined_string = f"{str(feature_name)}|||{str(string_value)}"
    hasher = hashlib.md5(combined_string.encode('utf-8'))
    hash_int = int(hasher.hexdigest(), 16)
    palette_index = hash_int % len(palette_list)
    return palette_list[palette_index]


def plot_feature_value_impact(
    df, OUTPUT_DIR, label_name, fup_max, max_delta=None, top_k=20, min_freq=50,
):
    if df.empty: return
    csv_path = OUTPUT_DIR / f"drivers_shap_{label_name}.csv"

    counts = df["Feature"].value_counts()
    valid = counts[counts >= min_freq].index
    if len(valid) == 0:
        print(f"      [!] Skipping Strip Plot: No features met min_freq={min_freq}.")
        return
    df_filtered = df[df["Feature"].isin(valid)]
    df_filtered["Abs_Score"] = df_filtered["Score"].abs()
    feature_importance = df_filtered.groupby("Feature")["Abs_Score"].mean().sort_values(ascending=False)
    
    # Extract top feature names
    top_features_raw = feature_importance.head(top_k).index.tolist()[::-1]
    
    # Filter original dataframe using raw feature names
    df_plot = df[df["Feature"].isin(top_features_raw)].copy()

    # --- APPLY WORD ABBREVIATIONS ---
    # Apply abbreviation to the DataFrame feature column and map the feature list
    df_plot["Feature"] = df_plot["Feature"].apply(abbreviate_feature_name)
    top_features = [abbreviate_feature_name(f) for f in top_features_raw]

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
        'No / Male': "#D11D14", 'Yes / Female / Occurred': "#203ed1",
        'Value 2': "#4082b7", 'Value 3': "#54a4a1", 'Value 4': "#88f3cf",
    }
    
    vir_cmap = plt.get_cmap("viridis", len(ORDINAL_LEVELS_LIST))
    ord_colors = {lvl: mcolors.to_hex(vir_cmap(i)) for i, lvl in enumerate(ORDINAL_LEVELS_LIST)}
    unknown_color = "#333333"

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
    ROW_HEIGHT = 1.55 
    ROW_PADDING = 0.70 
    
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
    
    fig_height = max(6, total_y_max * 0.48) 
    fig = plt.figure(figsize=(11.5, fig_height))
    
    gs = fig.add_gridspec(1, 1, left=0.55, right=0.88, top=0.98, bottom=0.12)
    ax = fig.add_subplot(gs[0, 0])
    ax.set_ylim(y_min_limit, total_y_max)
    
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
    ax.set_xlabel(_score_label(), fontsize=13.0, labelpad=6)
    
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
            
            color = master_palette.get((feature, item), unknown_color)
            
            marker_char = markers_map.get(g_type, "X")
            da = DrawingArea(16, 12, 0, 0)
            marker_style = MarkerStyle(marker_char)
            transformed_path = marker_style.get_path().transformed(marker_style.get_transform())
            bbox = transformed_path.get_extents()
            
            true_center_x = (bbox.x0 + bbox.x1) / 2.0
            true_center_y = (bbox.y0 + bbox.y1) / 2.0
                                            
            scale_factor = 7.5 if marker_char in ['s', 'D'] else 8.5
            final_transform = (
                transforms.Affine2D()
                .translate(-true_center_x, -true_center_y)
                .scale(scale_factor)
                .translate(8.0, 6.0)
            )
            
            patch = PathPatch(transformed_path.transformed(final_transform), facecolor=color, edgecolor='white', linewidth=0.5)
            da.add_artist(patch)
            
            ta = TextArea(f" {str(item)}", textprops=dict(fontsize=10.5, color='#444'))
            legend_boxes.append(HPacker(children=[da, ta], align="center", pad=0, sep=0))
            
        if has_leftovers:
            da = DrawingArea(16, 12, 0, 0)
            final_transform = transforms.Affine2D().translate(-0.0, -0.0).scale(7.5).translate(8.0, 6.0)
            patch = PathPatch(MarkerStyle("X").get_path().transformed(final_transform), facecolor="#777777", edgecolor='white', linewidth=0.5)
            da.add_artist(patch)
            ta = TextArea(" Other / Unknown", textprops=dict(fontsize=10.5, color='#666', fontstyle='italic'))
            legend_boxes.append(HPacker(children=[da, ta], align="center", pad=0, sep=0))

        rows_children = []
        n_elements = len(legend_boxes)
        N_SUBROWS = 3 

        if n_elements > 0:
            base_items_per_row = n_elements // N_SUBROWS
            rem = n_elements % N_SUBROWS
            
            idx_ptr = 0
            for r_sub in range(N_SUBROWS):
                allocated_size = base_items_per_row + (1 if r_sub < rem else 0)
                if allocated_size == 0:
                    continue
                
                sub_row_chunk = legend_boxes[idx_ptr : idx_ptr + allocated_size]
                idx_ptr += allocated_size
                rows_children.append(HPacker(children=sub_row_chunk, align="center", pad=0, sep=10))
                
        while len(rows_children) < N_SUBROWS:
            rows_children.append(HPacker(children=[], align="center", pad=0, sep=0))
            
        title_box = TextArea(feature, textprops=dict(fontsize=13.0, fontweight='bold', color='#111'))
        content_box = VPacker(children=[title_box] + rows_children, align="right", pad=0, sep=3)
        
        blended_legend_transform = transforms.blended_transform_factory(ax.transAxes, ax.transData)
        anchored_box = AnchoredOffsetbox(
            loc='upper right', child=content_box, pad=0, borderpad=0, frameon=False,
            bbox_to_anchor=(-0.05, y_top_boundary), bbox_transform=blended_legend_transform,
        )
        anchored_box.set_clip_on(False)
        ax.add_artist(anchored_box)
    
    plt.savefig(OUTPUT_DIR / f"drivers_shap_{label_name}.png", dpi=300)
    plt.close()
    
    
def run_captum_analysis(
    model, dataset, collator, vocab, device, OUTPUT_DIR, label_name,
    fup_max, indices, target_idx=0, entity_filter=None, max_delta=None,
    top_k=20, min_freq=20, plot_only=None, agg_method="mean", ckpt_sig: str = "",
):
    print(f" -> Extracting attributions for {len(indices)} samples...")
    csv_path = OUTPUT_DIR / f"attributions_analysis_{label_name}.csv"
    
    df = extract_attributions(
        model, dataset, collator, indices, vocab, device,
        target_idx=target_idx, entity_filter=entity_filter,
        max_delta=max_delta, csv_path=csv_path, plot_only=plot_only, ckpt_sig=ckpt_sig,
    )
    
    if df.empty:
        print("No attributions extracted.")
        return

    print("\n>>> Generating plots...")

    print(" -> Generating Feature Importance Bar Chart...")
    plot_feature_importance(
        df, OUTPUT_DIR, label_name, top_k=top_k,
        min_freq=min_freq, agg_method=agg_method,
    )

    print(" -> Generating Volcano Plot (Frequency vs Impact)...")
    plot_frequency_vs_impact(df, OUTPUT_DIR, label_name, min_freq=min_freq)

    print(" -> Generating Detailed Strip Plot...")
    plot_feature_value_impact(
        df, OUTPUT_DIR, label_name, fup_max,
        max_delta=max_delta, top_k=top_k, min_freq=min_freq
    )


if __name__ == "__main__":
    main()