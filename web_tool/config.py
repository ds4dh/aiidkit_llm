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
    ACUTE_HORIZONS,
    LONG_TERM_HORIZONS,
    UNIFIED_HORIZONS,
    UNIFIED_TASKS,
    build_available_tasks_list,
    build_horizon_labels_map,
)

# =============================================================================
# SPECIALIST MULTI-TASK DOMAIN SUITE (B_CAT_MT — 5 Specialist Models)
# =============================================================================

_DEFAULT_FINETUNING_ROOT = _ROOT / "results/transformer/random_split/e00-a15-v60_raw/finetuning"
FINETUNING_ROOT: Path = (
    Path(os.environ["AIIDKIT_FINETUNING_ROOT"])
    if os.environ.get("AIIDKIT_FINETUNING_ROOT")
    else _DEFAULT_FINETUNING_ROOT
)

SPECIALIST_DOMAIN_DEFS: list[dict] = [
    {
        "domain": "bacteria",
        "label": "Bacterial Specialist",
        "folder": "multitask_bacteria",
        "tasks": [
            "infection_bacteria",
            "infection_grp_enterobacterales",
            "infection_grp_gram_positive",
            "infection_grp_gn_nonfermenters",
            "infection_grp_clostridium",
        ],
        "horizons": list(ACUTE_HORIZONS),
    },
    {
        "domain": "viruses",
        "label": "Viral Specialist",
        "folder": "multitask_viruses",
        "tasks": [
            "infection_virus",
            "infection_grp_herpes_viruses",
            "infection_grp_cmv",
            "infection_grp_bkpyv",
        ],
        "horizons": list(ACUTE_HORIZONS),
    },
    {
        "domain": "fungi",
        "label": "Fungal Specialist",
        "folder": "multitask_fungi",
        "tasks": [
            "infection_fungi",
            "infection_grp_candida",
        ],
        "horizons": list(ACUTE_HORIZONS),
    },
    {
        "domain": "graft_loss",
        "label": "Graft Survival Specialist",
        "folder": "graft_loss",
        "tasks": ["graft_loss"],
        "horizons": list(LONG_TERM_HORIZONS),
    },
    {
        "domain": "death",
        "label": "Mortality Specialist",
        "folder": "death",
        "tasks": ["death"],
        "horizons": list(LONG_TERM_HORIZONS),
    },
]

# Explicit single-model override only when AIIDKIT_MODEL_PATH is provided
MODEL_PATH: Path | None = (
    Path(os.environ["AIIDKIT_MODEL_PATH"])
    if os.environ.get("AIIDKIT_MODEL_PATH")
    else None
)

_DEFAULT_DATA_DIR = Path("/home/shares/ds4dh/aiidkit_project/data_new/processed/v3.9/teav/random_split")
DATA_DIR: Path | None = (
    Path(os.environ["AIIDKIT_DATA_DIR"])
    if os.environ.get("AIIDKIT_DATA_DIR")
    else _DEFAULT_DATA_DIR
)

# Portable embeddings-only server bundle (used inside Docker or when AIIDKIT_BUNDLE_DIR is set)
_DOCKER_BUNDLE_CANDIDATES = [
    Path(os.environ["AIIDKIT_BUNDLE_DIR"]) if os.environ.get("AIIDKIT_BUNDLE_DIR") else None,
    Path("/data/bundle"),
    Path(__file__).resolve().parent / "server_bundle",
]
BUNDLE_DIR: Path | None = next(
    (p for p in _DOCKER_BUNDLE_CANDIDATES if p is not None and (p / "bundle_manifest.json").exists()),
    Path(os.environ["AIIDKIT_BUNDLE_DIR"]) if os.environ.get("AIIDKIT_BUNDLE_DIR") else None,
)
PREFER_BUNDLE: bool = os.environ.get("AIIDKIT_PREFER_BUNDLE", "").lower() in ("1", "true", "yes") or bool(
    os.environ.get("AIIDKIT_BUNDLE_DIR") or Path("/data/bundle/bundle_manifest.json").exists()
)

USE_GPU: bool = os.environ.get("AIIDKIT_USE_GPU", "true").lower() not in ("0", "false", "no")
IG_STEPS: int | None = int(os.environ["AIIDKIT_IG_STEPS"]) if os.environ.get("AIIDKIT_IG_STEPS") else None


def _default_max_models_in_ram() -> int:
    if os.environ.get("AIIDKIT_MAX_MODELS_IN_RAM"):
        return max(1, int(os.environ["AIIDKIT_MAX_MODELS_IN_RAM"]))
    try:
        mem_bytes = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        if mem_bytes >= 14 * (1024 ** 3):
            return 5
    except Exception:
        pass
    return 5 if USE_GPU else 2


MAX_MODELS_IN_RAM: int = _default_max_models_in_ram()

VOCAB_PATH: Path | None = None
BIN_INTERVALS_PATH: Path | None = None

# Runtime derived-UMAP cache directory (writable tmpfs inside read-only Docker containers)
RUNTIME_CACHE_DIR: Path = Path(
    os.environ.get("AIIDKIT_RUNTIME_CACHE_DIR", str(Path(__file__).resolve().parent / ".cache"))
)

# Optional HTTP Basic Auth (RFC 7617) & Reverse-Proxy settings
BASIC_AUTH_USER: str = os.environ.get("AIIDKIT_BASIC_AUTH_USER", "aiidkit")
BASIC_AUTH_PASS: str = os.environ.get("AIIDKIT_BASIC_AUTH_PASS", "")
BASIC_AUTH_REALM: str = os.environ.get("AIIDKIT_BASIC_AUTH_REALM", "AIIDKIT Clinical Decision Support")
TRUST_PROXY: bool = os.environ.get("AIIDKIT_TRUST_PROXY", "true").lower() not in ("0", "false", "no")


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


def _resolve_latest_domain_checkpoint(finetuning_root: Path, folder_name: str) -> Path | None:
    folder = finetuning_root / folder_name
    if not folder.exists():
        return None
    run_dirs = sorted(
        (p for p in folder.glob("hrz(*)*") if p.is_dir()),
        key=lambda p: p.stat().st_mtime_ns,
        reverse=True,
    )
    for rd in run_dirs:
        ckpt = resolve_best_checkpoint(rd)
        if ckpt is not None and (ckpt / "model.safetensors").exists():
            return ckpt
    ckpt = resolve_best_checkpoint(folder)
    return ckpt if (ckpt is not None and (ckpt / "model.safetensors").exists()) else None


def discover_specialist_checkpoints(finetuning_root: Path | None = None) -> dict[str, dict]:
    """Discover all trained specialist checkpoints under `finetuning_root`."""
    ft_root = finetuning_root or FINETUNING_ROOT
    discovered: dict[str, dict] = {}
    for d_def in SPECIALIST_DOMAIN_DEFS:
        ckpt = _resolve_latest_domain_checkpoint(ft_root, d_def["folder"])
        if ckpt is not None:
            run_meta = _discover_run_metadata(ckpt) or {
                "tasks": d_def["tasks"],
                "horizons": d_def["horizons"],
            }
            discovered[d_def["domain"]] = {
                "domain": d_def["domain"],
                "label": d_def["label"],
                "model_path": ckpt,
                "run_metadata": run_meta,
            }
    return discovered


def get_resolved_paths() -> dict:
    """Called once at server startup. Returns resolved single/multi-domain paths and `use_mock` flag."""
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
            "bundle_dir":    None,
            "domains":       {},
            "model_path":    None,
            "run_metadata":  None,
            "data_dir":      None,
            "test_dir":      None,
            "vocab_path":    None,
            "bin_intervals": None,
            "use_gpu":       False,
        }

    # Priority 1: Explicit single-model override via AIIDKIT_MODEL_PATH
    if MODEL_PATH is not None:
        model = _resolve(MODEL_PATH, "MODEL_PATH")
        if model is not None and not (model / "model.safetensors").exists():
            model = resolve_best_checkpoint(model)
        run_meta = _discover_run_metadata(model)
        data = _resolve(DATA_DIR, "DATA_DIR")
        vocab = _resolve(VOCAB_PATH, "VOCAB_PATH") if VOCAB_PATH is not None else None
        bins = _resolve(BIN_INTERVALS_PATH, "BIN_INTERVALS_PATH") if BIN_INTERVALS_PATH is not None else None
        if data is not None and (vocab is None or bins is None):
            try:
                from src.data.patient_dataset import resolve_processed_cache_dir
                m_dir = resolve_processed_cache_dir(data) / "pretraining_metadata"
                if vocab is None and (m_dir / "vocab.pkl").exists():
                    vocab = m_dir / "vocab.pkl"
                if bins is None and (m_dir / "bin_intervals.pkl").exists():
                    bins = m_dir / "bin_intervals.pkl"
            except Exception:
                pass
        domains = (
            {"default": {"domain": "default", "label": "Active Model", "model_path": model, "run_metadata": run_meta}}
            if model is not None
            else {}
        )
        return {
            "use_mock":      model is None,
            "bundle_dir":    None,
            "domains":       domains,
            "model_path":    model,
            "run_metadata":  run_meta,
            "data_dir":      data,
            "test_dir":      (data / f"fup_{DEFAULT_FUP:04d}" / "test") if data is not None else None,
            "vocab_path":    vocab,
            "bin_intervals": bins,
            "use_gpu":       USE_GPU,
        }

    # Priority 2: De-identified Server Bundle (Multi-Domain v2 or Single-Domain v1)
    bundle = BUNDLE_DIR if (BUNDLE_DIR is not None and (BUNDLE_DIR / "bundle_manifest.json").exists()) else None
    data = None if (bundle is not None and PREFER_BUNDLE) else _resolve(DATA_DIR, "DATA_DIR")

    if bundle is not None:
        manifest = json.loads((bundle / "bundle_manifest.json").read_text(encoding="utf-8"))
        bundle_domains: dict[str, dict] = {}
        if isinstance(manifest.get("domains"), dict) and manifest["domains"]:
            for d_key, d_info in manifest["domains"].items():
                b_mod = bundle / "domains" / d_key / "model"
                if (b_mod / "model.safetensors").exists():
                    bundle_domains[d_key] = {
                        "domain": d_key,
                        "label": d_info.get("label", d_key.replace("_", " ").title()),
                        "model_path": b_mod,
                        "run_metadata": _discover_run_metadata(b_mod),
                        "platt_path": bundle / "domains" / d_key / "platt_params.json",
                        "embeddings_path": bundle / "domains" / d_key / "cohort_embeddings.npz",
                        "domain_fingerprint": d_info.get("domain_fingerprint", manifest.get("bundle_fingerprint", "v2")),
                    }
        elif (bundle / "model" / "model.safetensors").exists():
            b_mod = bundle / "model"
            bundle_domains["default"] = {
                "domain": "default",
                "label": "Unified Model",
                "model_path": b_mod,
                "run_metadata": _discover_run_metadata(b_mod),
                "platt_path": bundle / "metadata" / "platt_params.json",
                "embeddings_path": bundle / "embeddings" / "cohort_embeddings.npz",
                "domain_fingerprint": manifest.get("bundle_fingerprint", "v1"),
            }
        if bundle_domains:
            first_dom = next(iter(bundle_domains.values()))
            return {
                "use_mock":      False,
                "bundle_dir":    bundle,
                "domains":       bundle_domains,
                "model_path":    first_dom["model_path"],
                "run_metadata":  first_dom["run_metadata"],
                "data_dir":      None,
                "test_dir":      None,
                "vocab_path":    bundle / "metadata" / "vocab.json",
                "bin_intervals": bundle / "metadata" / "bin_intervals.json",
                "use_gpu":       USE_GPU,
            }

    # Priority 3: Direct HPC Multi-Domain discovery from results/ + DATA_DIR
    discovered = discover_specialist_checkpoints(FINETUNING_ROOT)
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

    first_dom = next(iter(discovered.values()), None)
    return {
        "use_mock":      (not discovered) or (data is None),
        "bundle_dir":    None,
        "domains":       discovered,
        "model_path":    first_dom["model_path"] if first_dom else None,
        "run_metadata":  first_dom["run_metadata"] if first_dom else None,
        "data_dir":      data,
        "test_dir":      (data / f"fup_{DEFAULT_FUP:04d}" / "test") if data is not None else None,
        "vocab_path":    vocab,
        "bin_intervals": bins,
        "use_gpu":       USE_GPU,
    }


# =============================================================================
# TASKS & HORIZONS — 5-Domain Specialist Suite (13 Tasks, Domain-Matched Horizons)
# =============================================================================

TASK_TO_DOMAIN: dict[str, str] = {
    t: d["domain"] for d in SPECIALIST_DOMAIN_DEFS for t in d["tasks"]
}
DOMAIN_LABELS: dict[str, str] = {
    d["domain"]: d["label"] for d in SPECIALIST_DOMAIN_DEFS
}
TASK_HORIZONS: dict[str, list[int]] = {
    t: list(d["horizons"]) for d in SPECIALIST_DOMAIN_DEFS for t in d["tasks"]
}

_ACTIVE_TASK_KEYS: list[str] = [t for d in SPECIALIST_DOMAIN_DEFS for t in d["tasks"]]
_ACTIVE_HORIZONS: list[int] = sorted({h for d in SPECIALIST_DOMAIN_DEFS for h in d["horizons"]})

TASK: str = "infection_bacteria"
AVAILABLE_TASKS: list[dict] = [
    {
        **item,
        "domain": TASK_TO_DOMAIN.get(item["key"], "bacteria"),
        "domain_label": DOMAIN_LABELS.get(TASK_TO_DOMAIN.get(item["key"], "bacteria"), "Specialist Model"),
        "horizons": TASK_HORIZONS.get(item["key"], list(ACUTE_HORIZONS)),
    }
    for item in build_available_tasks_list(_ACTIVE_TASK_KEYS)
]
AVAILABLE_HORIZONS: list[int] = _ACTIVE_HORIZONS
DEFAULT_HORIZON: int = 90
HORIZON_LABELS: dict[int, str] = build_horizon_labels_map(sorted(set(UNIFIED_HORIZONS) | set(_ACTIVE_HORIZONS)), style="long")
HORIZON_SHORT_LABELS: dict[int, str] = build_horizon_labels_map(sorted(set(UNIFIED_HORIZONS) | set(_ACTIVE_HORIZONS)), style="short")

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
