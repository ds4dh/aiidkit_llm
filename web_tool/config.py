"""
AIIDKIT Decision Support Tool — Configuration
==============================================

HOW TO CONNECT YOUR REAL MODEL
--------------------------------
Edit the paths in the "USER-EDITABLE PATHS" section below (or set
`AIIDKIT_MODEL_PATH` / `AIIDKIT_DATA_DIR` environment variables), then restart
the server. Tasks, horizons, and monotonic head groups are auto-detected from
`run_metadata.json`.
"""

import json
import os
import sys
import warnings
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.model.model_utils import resolve_best_checkpoint
from src.tasks import (
    UNIFIED_HORIZONS,
    UNIFIED_TASKS,
    build_available_tasks_list,
    build_horizon_labels_map,
)

# =============================================================================
# USER-EDITABLE PATHS
# =============================================================================

_UNIFIED_MODEL_CANDIDATES = [
    _ROOT
    / "results/transformer/random_split/e00-a15-v60_raw/finetuning/multitask_unified"
    / "hrz(0030-0060-0090-0180-0360-0720-1800)_fut(all)_fuv(0000-0030-0060-0090-0120-0150-0180-0240-0300-0360-0540-0720-0900-1080-1440-1800-2700-3600)",
    _ROOT
    / "results/transformer/random_split/e00-a15-v60_raw/finetuning/multitask_bacteria"
    / "hrz(0030-0060-0090)_fut(all)_fuv(0000-0030-0060-0090-0120-0150-0180-0240-0300-0360-0540-0720-0900-1080-1440-1800-2700-3600)",
]

_DEFAULT_MODEL_PATH = next((p for p in _UNIFIED_MODEL_CANDIDATES if p.exists()), _UNIFIED_MODEL_CANDIDATES[0])
MODEL_PATH: Path | None = (
    Path(os.environ["AIIDKIT_MODEL_PATH"])
    if os.environ.get("AIIDKIT_MODEL_PATH")
    else _DEFAULT_MODEL_PATH
)

_DEFAULT_DATA_DIR = Path("/home/shares/ds4dh/aiidkit_project/data_new/processed/v3.9/teav/random_split")
DATA_DIR: Path | None = (
    Path(os.environ["AIIDKIT_DATA_DIR"])
    if os.environ.get("AIIDKIT_DATA_DIR")
    else _DEFAULT_DATA_DIR
)

USE_GPU: bool = os.environ.get("AIIDKIT_USE_GPU", "true").lower() not in ("0", "false", "no")

VOCAB_PATH: Path | None = None
BIN_INTERVALS_PATH: Path | None = None


# =============================================================================
# PATH VALIDATION & METADATA AUTO-DISCOVERY
# =============================================================================

def _resolve(p: Path | None, label: str) -> Path | None:
    """Return `p` if it exists on disk, else `None` (with a warning)."""
    if p is None:
        return None
    if not p.exists():
        warnings.warn(
            f"[AIIDKIT] {label} not found: {p}\n  → Falling back to mock/demo mode.",
            stacklevel=2,
        )
        return None
    return p


def _discover_run_metadata(model_path: Path | None) -> dict | None:
    """Reads `run_metadata.json` from a checkpoint or its parent run directory if present."""
    if model_path is None:
        return None
    for cand in (model_path / "run_metadata.json", model_path.parent / "run_metadata.json"):
        if cand.exists():
            try:
                return json.loads(cand.read_text(encoding="utf-8"))
            except Exception:
                pass
    return None


def get_resolved_paths() -> dict:
    """Called once at server startup. Returns resolved paths and `use_mock` flag."""
    force_demo = (
        os.environ.get("AIIDKIT_DEMO_MODE", "").lower() in ("1", "true", "yes")
        or "--demo" in sys.argv
        or "--mock" in sys.argv
    )
    if force_demo:
        os.environ["AIIDKIT_DEMO_MODE"] = "1"
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        return {
            "use_mock":      True,
            "model_path":    None,
            "run_metadata":  None,
            "data_dir":      None,
            "test_dir":      None,
            "vocab_path":    None,
            "bin_intervals": None,
            "use_gpu":       False,
        }

    model = _resolve(MODEL_PATH, "MODEL_PATH")
    if model is not None and not (model / "model.safetensors").exists():
        try:
            model = resolve_best_checkpoint(model)
        except FileNotFoundError:
            warnings.warn(f"[AIIDKIT] No valid checkpoint inside {model}; falling back to demo mode.")
            model = None

    run_meta = _discover_run_metadata(model)
    data = _resolve(DATA_DIR, "DATA_DIR")

    cache_dir = Path(__file__).resolve().parent / ".cache"
    has_cached_embeddings = cache_dir.exists() and any(cache_dir.glob("cohort_umap_*.pkl"))
    use_mock = model is None or (data is None and not has_cached_embeddings)

    vocab = _resolve(VOCAB_PATH, "VOCAB_PATH") if VOCAB_PATH is not None else None
    bins = _resolve(BIN_INTERVALS_PATH, "BIN_INTERVALS_PATH") if BIN_INTERVALS_PATH is not None else None
    if data is not None and (vocab is None or bins is None):
        meta_candidates = []
        try:
            from src.data.patient_dataset import resolve_processed_cache_dir
            meta_candidates.append(resolve_processed_cache_dir(data) / "pretraining_metadata")
        except Exception:
            pass
        meta_candidates.append(data / "processed_cache" / "pretraining_metadata")
        for meta in meta_candidates:
            if vocab is None and (meta / "vocab.pkl").exists():
                vocab = _resolve(meta / "vocab.pkl", "vocab.pkl")
            if bins is None and (meta / "bin_intervals.pkl").exists():
                bins = _resolve(meta / "bin_intervals.pkl", "bin_intervals.pkl")

    test_dir = None
    if data is not None:
        candidate = data / f"fup_{DEFAULT_FUP:04d}" / "test"
        if not candidate.exists():
            candidate = data / "fup_0090" / "test"
        test_dir = _resolve(candidate, "test split")

    return {
        "use_mock":      use_mock,
        "model_path":    model,
        "run_metadata":  run_meta,
        "data_dir":      data,
        "test_dir":      test_dir,
        "vocab_path":    vocab,
        "bin_intervals": bins,
        "use_gpu":       USE_GPU,
    }


# =============================================================================
# TASKS & HORIZONS — Unified 13-Task × 7-Horizon Setup (auto-synced to checkpoint)
# =============================================================================

_STARTUP_META = _discover_run_metadata(_DEFAULT_MODEL_PATH if _DEFAULT_MODEL_PATH.exists() else None)
_ACTIVE_TASK_KEYS = (
    list(_STARTUP_META["tasks"])
    if (_STARTUP_META and isinstance(_STARTUP_META.get("tasks"), list) and _STARTUP_META["tasks"])
    else list(UNIFIED_TASKS)
)
_ACTIVE_HORIZONS = (
    sorted(int(h) for h in _STARTUP_META["horizons"])
    if (_STARTUP_META and isinstance(_STARTUP_META.get("horizons"), list) and _STARTUP_META["horizons"])
    else list(UNIFIED_HORIZONS)
)

TASK: str = _ACTIVE_TASK_KEYS[0] if "infection_bacteria" not in _ACTIVE_TASK_KEYS else "infection_bacteria"
AVAILABLE_TASKS: list[dict[str, str]] = build_available_tasks_list(_ACTIVE_TASK_KEYS)
AVAILABLE_HORIZONS: list[int] = _ACTIVE_HORIZONS
DEFAULT_HORIZON: int = 90 if 90 in AVAILABLE_HORIZONS else AVAILABLE_HORIZONS[min(2, len(AVAILABLE_HORIZONS) - 1)]
HORIZON_LABELS: dict[int, str] = build_horizon_labels_map(AVAILABLE_HORIZONS, style="long")
HORIZON_SHORT_LABELS: dict[int, str] = build_horizon_labels_map(AVAILABLE_HORIZONS, style="short")

DEFAULT_FUP: int = 30
PRECOMPUTED_FUPS: list[int] = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360]
AVAILABLE_FUPS: list[int] = list(PRECOMPUTED_FUPS)
MAX_COHORT_PATIENTS: int = 500
UMAP_FIT_SAMPLE_SIZE: int = 1950
UMAP_N_NEIGHBORS: int = 15
UMAP_MIN_DIST: float = 0.1
UMAP_RANDOM_STATE: int = 42
CACHE_DIR: Path = Path(__file__).resolve().parent / ".cache"
RISK_THRESHOLD: float = 0.50
CLUSTERING_METHOD: str = "clusterer"
CLUSTERING_UMAP_COMPONENTS: int = 2
HDBSCAN_MIN_CLUSTER_SIZE: int = 15
HDBSCAN_MIN_SAMPLES: int = 5

EAV_MAPPINGS: dict = {
    "entity_id":    "entity",
    "attribute_id": "attribute",
    "value_id":     "value_binned",
}
TIME_MAPPING: dict = {"days_since_tpx": "time"}

# =============================================================================
# MOCK DATA — used when use_mock=True
# =============================================================================

NUM_MOCK_PATIENTS: int = 250
MOCK_RANDOM_SEED: int = 42

# =============================================================================
# CLINICAL VOCABULARY (v3.9 t-EAV schema)
# =============================================================================

ORDINAL = ["Lowest", "Lower", "Low", "Middle", "High", "Higher", "Highest"]
BINARY = ["0", "1"]

VOCABULARY: dict[str, dict[str, list[str]]] = {
    "Biopsy": {
        "Adequacy": ["Satisfactory"],
        "Banff - aah": ["0"],
        "Banff - ah": ["0"],
        "Banff - c4d": ["0"],
        "Banff - cg": ["0"],
        "Banff - ct": ["0"],
        "Banff - cv": ["0"],
        "Banff - mm": ["2"],
        "Banff - ptc": ["1"],
        "Banff - ti": ["0", "1"],
        "Banff - v": ["0"],
        "Presentation": ["Missing"],
        "Rejection diagnosis": ["No", "Other"],
        "Type": ["Diag biops", "PP biops"],
    },
    "Comorbidity": {
        "Category": ["Events/Surgery", "Malignancy", "Metabolic/Kidney"],
        "Diagnosis": ["Diabetes mellitus", "Hypertension", "Other", "Other metabolic disease"],
    },
    "Donor info": {
        "Age at transplant (years)": ORDINAL,
        "Altruistic": ["0"],
        "Blood group": ["0", "A", "AB", "B"],
        "Cause of death": ["ANX", "CVA", "Trauma"],
        "History of CAD": BINARY,
        "History of myocardial contusion": ["0"],
        "Identical twin": ["0"],
        "KPD": ["0"],
        "Sex": ["Female", "Male"],
        "Smoker": BINARY,
        "Type": ["DBD", "DCD", "Living related", "Living unrelated"],
        "Valvular disease": ["0"],
    },
    "Graft Loss": {
        "Cause": ["Other causes"],
    },
    "History": {
        "Pregnancy": BINARY,
        "Transfusion": BINARY,
    },
    "Immunology": {
        "Calculated PRA (%)": ["0"],
        "Latest cPRA (%)": ORDINAL,
        "Peak cPRA (%)": ORDINAL,
    },
    "Infection": {
        "Category": ["Bacteria", "Fungi", "Virus"],
        "Clinically significant": ["False", "True"],
        "Microbiology specimen": ["Blood", "Urine"],
        "Pathogen": ["BKV", "CMV", "Candida albicans", "E. coli", "Klebsiella pneumoniae", "Pseudomonas aeruginosa", "Staphylococcus aureus"],
        "Severity grade": ["Grade 1", "Grade 2", "Grade 3"],
        "Site": ["Abdominal", "Bloodstream", "Respiratory tract", "Urinary tract"],
        "Source": ["Community acquired", "Hospital acquired"],
        "Type": ["Possible disease", "Probable disease", "Proven disease"],
    },
    "Malignancy": {
        "Status": ["Incident", "Newly detected"],
    },
    "Medication": {
        "Start (immunosuppression)": [
            "Belatacept",
            "Cyclosporine",
            "Glucocorticoid",
            "Mycophenolate Mofetil",
            "Mycophenolic Acid",
            "Tacrolimus",
        ],
        "Stop (immunosuppression)": [
            "Belatacept",
            "Cyclosporine",
            "Glucocorticoid",
            "Mycophenolate Mofetil",
            "Mycophenolic Acid",
            "Tacrolimus",
        ],
        "Start (induction)": ["Basiliximab", "Other drugs", "Rabbit ATG"],
        "Stop (induction)": ["Basiliximab", "Other drugs", "Rabbit ATG"],
        "Start (infection prophylaxis)": [
            "Atovaquone",
            "Beta-Lactame",
            "Cotrimoxazole",
            "Fluconazole",
            "Quinolone",
            "Valaciclovir",
            "Valganciclovir",
        ],
        "Stop (infection prophylaxis)": [
            "Atovaquone",
            "Beta-Lactame",
            "Cotrimoxazole",
            "Fluconazole",
            "Quinolone",
            "Valaciclovir",
            "Valganciclovir",
        ],
        "Start (other)": [
            "Angiotensin receptor blocker",
            "Anticoagulation therapy",
            "Beta-blocker",
            "Calcium channel blocker",
            "Human Immunoglobulin",
            "Insulin",
            "Oral antidiabetic drug",
            "Other drugs",
            "Platelet aggregation inhibitor",
            "Statin",
        ],
        "Stop (other)": [
            "Angiotensin receptor blocker",
            "Anticoagulation therapy",
            "Beta-blocker",
            "Calcium channel blocker",
            "Human Immunoglobulin",
            "Insulin",
            "Oral antidiabetic drug",
            "Other drugs",
            "Platelet aggregation inhibitor",
            "Statin",
        ],
    },
    "Mismatch info": {
        "Mismatch HLA sum (minimum)": ORDINAL,
        "Mismatch HLA-A": ["0", "1", "2"],
        "Mismatch HLA-B": ["0", "1", "2"],
        "Mismatch HLA-DQB1": ["0", "1", "2"],
        "Mismatch HLA-DRB1": ["0", "1", "2"],
        "Mismatch HLA-DRB35": ["0", "1", "2"],
    },
    "Organ event": {
        "Diagnosis": ["Biops proven rj", "CNI nephrotoxicity", "Graft pyelonephritis", "Lymphocele", "Other"],
    },
    "Organ lab": {
        "Albumin (g/l)": ORDINAL,
        "Factor V (%)": ORDINAL,
        "Fibrinogen (g/l)": ORDINAL,
        "Proteinuria (mg/mmol)": ORDINAL,
        "Triglycerides (mmol/l)": ORDINAL,
    },
    "Patient info": {
        "Age at transplant (years)": ORDINAL,
        "Blood group": ["0", "A", "AB", "B"],
        "Ethnicity": ["Caucasian"],
        "Etiology": ["Condition unknown", "GN", "Hereditary non_PCKD", "PCKD"],
        "Pre-transplant immuno-suppressed": ["No", "Yes"],
        "Previous transplant count (any organ)": ["0", "1"],
        "Previous transplanted organ count": ["0", "1"],
        "Sex": ["Female", "Male"],
    },
    "Patient lab": {
        "CRP (mg/l)": ORDINAL,
        "Creatinine (µmol/l)": ORDINAL,
        "Glucose (mmol/l)": ORDINAL,
        "Glycated hemoglobin (HbA1c) (%)": ORDINAL,
        "HDL cholesterol (mmol/l)": ORDINAL,
        "LDL cholesterol (mmol/l)": ORDINAL,
        "Leukocytes (G/l)": ORDINAL,
        "Lymphocytes (G/l)": ORDINAL,
        "Tacrolimus level (µg/l)": ORDINAL,
        "Total bilirubin (µmol/l)": ORDINAL,
        "Total cholesterol (mmol/l)": ORDINAL,
        "eGFR (mL/min/1.73m²)": ORDINAL,
    },
    "Patient serology": {
        "CMV": BINARY,
        "EBV": BINARY,
        "HBcAb": BINARY,
        "HBsAb": BINARY,
        "HBsAg": BINARY,
        "HCV": BINARY,
        "HIV": BINARY,
        "HSV": BINARY,
        "Syphilis": BINARY,
        "Toxoplasmosis": BINARY,
        "VZV": BINARY,
    },
    "Patient vitals": {
        "Patient BMI (kg/m²)": ORDINAL,
        "Patient age (years)": ORDINAL,
        "Patient diastolic BP (mmHg)": ORDINAL,
        "Patient height (cm)": ORDINAL,
        "Patient systolic BP (mmHg)": ORDINAL,
        "Patient weight (kg)": ORDINAL,
        "Transplant age (years)": ORDINAL,
    },
    "Serology info": {
        "CMV": ["D+/R+", "D+/R-", "D-/R+", "D-/R-"],
        "EBV": ["D+/R+", "D+/R-", "D-/R+", "D-/R-"],
        "HBcAb": ["D-/R+"],
        "HBsAb": ["D+/R+", "D-/R+"],
        "HBsAg": ["D-/R-"],
        "HCV": ["D-/R-"],
        "HIV": ["D-/R-"],
        "HSV": ["D+/R+"],
        "Toxoplasmosis": ["D+/R+", "D+/R-"],
        "VZV": ["D+/R+"],
    },
    "Transplant info": {
        "24h Urine Collection": ["0"],
        "ABO compatible": BINARY,
        "Asystolic ischemia": ORDINAL,
        "Cold ischemia time (hours)": ORDINAL,
        "Delayed graft function": BINARY,
        "Dialysis Type (Pre-Tx)": ["HD", "PD"],
        "Hosp. stay duration so far (days)": ORDINAL,
        "Hospital discharge": ["Kidney"],
        "Listing Organ Type": ["Kidney"],
        "Previous pregnancy": BINARY,
        "Previous transfusion": BINARY,
        "Previous transplant count (kidney)": ["0", "1"],
        "Procedure provider": ["Insel"],
        "Re-transplant": ["First"],
        "Total Organ Count": ["1"],
        "Transplant Order": ["1"],
        "Transplant Resection": ["Re Tpx"],
        "Transplant event": ["Kidney"],
        "Transplantation year": ORDINAL,
        "Virtual crossmatch": BINARY,
        "Waitlist removal": ["Kidney"],
        "Waitlist start": ["Kidney", "Kidney - Islets", "Kidney - Pancreas"],
        "urgentlisting": BINARY,
    },
    "Virology": {
        "Detection": ["BKV", "CMV", "HBV", "HCV"],
        "Dynamics": ["Monitoring"],
        "Viral load (copies/ml)": ORDINAL,
    },
}

# =============================================================================
# SERVER
# =============================================================================

HOST: str = os.environ.get("AIIDKIT_HOST", "0.0.0.0")
PORT: int = int(os.environ.get("AIIDKIT_PORT", "5000"))
DEBUG: bool = os.environ.get("AIIDKIT_DEBUG", "false").lower() in ("1", "true", "yes")
EXCLUDE_PATTERNS: list[str] = ["*/scripts/*", "*/.cache/*"]
