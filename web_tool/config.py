"""
AIIDKIT Decision Support Tool — Configuration
==============================================

HOW TO CONNECT YOUR REAL MODEL
--------------------------------
Edit the two paths in the "USER-EDITABLE PATHS" section below, then restart
the server.  Everything else is auto-detected.

STEP 1 — Point to the model checkpoint directory
    MODEL_PATH = Path("/home/shares/ds4dh/aiidkit_project/results_final/"
                      "transformer/temporal_split/e00-a15-v60/finetuning/"
                      "infection_bacteria/"
                      "hrz(0030-0060-0090)_fut(all)_fuv(0000-0030-0060-0090-0180-0360)/"
                      "checkpoint-2500")
    The directory must contain: model.safetensors, config.json, training_args.bin

STEP 2 — Point to the evaluation data split
    DATA_DIR = Path("/home/shares/ds4dh/aiidkit_project/data_new/"
                    "processed/v3.6/teav/temporal_split")
    The test split is auto-discovered as DATA_DIR / "fup_0090/test/"

STEP 3 — GPU
    USE_GPU = True   (the tool auto-selects the first available CUDA device)

FALLBACK BEHAVIOUR
    If either MODEL_PATH or DATA_DIR is None, or the path does not exist on
    disk, the tool transparently falls back to mock/demo data — no changes
    needed anywhere else.  The header banner will say "DEMO MODE" or "LIVE MODEL"
    accordingly.
"""

from pathlib import Path

# =============================================================================
# USER-EDITABLE PATHS
# Change these when running on the GPU server with real data.
# Leave as None to keep DEMO MODE (mock data + mock model).
# =============================================================================

MODEL_PATH: Path | None = Path(
    "/home/users/b/borneta/aiidkit_llm/results_final/"
    "transformer/temporal_split/e00-a15-v60/finetuning/"
    "infection_bacteria/"
    "hrz(0030-0060-0090)_fut(all)_fuv(0000-0030-0060-0090-0180-0360)/"
    "checkpoint-2500"
)
# Set to None to force DEMO MODE regardless of whether the path exists:
# MODEL_PATH = None

DATA_DIR: Path | None = Path(
    "/home/shares/ds4dh/aiidkit_project/data_new/"
    "processed/v3.6_old/teav/temporal_split"
)
# Set to None to force DEMO MODE regardless of whether the path exists:
# DATA_DIR = None

USE_GPU: bool = True   # GPU is available on HPC; set False for CPU-only testing

# Optional overrides (auto-resolved from DATA_DIR when left as None)
VOCAB_PATH: Path | None = None
# Auto: DATA_DIR / "processed_cache/pretraining_metadata/vocab.pkl"

BIN_INTERVALS_PATH: Path | None = None
# Auto: DATA_DIR / "processed_cache/pretraining_metadata/bin_intervals.pkl"


# =============================================================================
# PATH VALIDATION — auto-switches demo/live at startup
# =============================================================================

def _resolve(p: Path | None, label: str) -> Path | None:
    """Return p if it exists on disk, else None (with a warning)."""
    if p is None:
        return None
    if not p.exists():
        import warnings
        warnings.warn(
            f"[AIIDKIT] {label} not found: {p}\n"
            "  → Falling back to mock/demo mode.",
            stacklevel=2,
        )
        return None
    return p


def get_resolved_paths() -> dict:
    """
    Called once at server startup.
    Returns a dict with resolved (existing) paths and a 'use_mock' flag.
    """
    model = _resolve(MODEL_PATH, "MODEL_PATH")
    data  = _resolve(DATA_DIR,   "DATA_DIR")

    use_mock = model is None or data is None

    # Auto-resolve vocab / bin_intervals from DATA_DIR if not overridden
    vocab = VOCAB_PATH
    bins  = BIN_INTERVALS_PATH
    if data is not None:
        meta = data / "processed_cache" / "pretraining_metadata"
        if vocab is None:
            vocab = _resolve(meta / "vocab.pkl", "vocab.pkl")
        if bins is None:
            bins  = _resolve(meta / "bin_intervals.pkl", "bin_intervals.pkl")

    # Test split: DATA_DIR / "fup_0090/test/"
    test_dir = None
    if data is not None:
        candidate = data / "fup_0090" / "test"
        test_dir  = _resolve(candidate, "test split")

    return {
        "use_mock":      use_mock,
        "model_path":    model,
        "data_dir":      data,
        "test_dir":      test_dir,
        "vocab_path":    vocab,
        "bin_intervals": bins,
        "use_gpu":       USE_GPU,
    }


# =============================================================================
# TASK — infection_bacteria only for now
# =============================================================================

TASK: str = "infection_bacteria"
DEFAULT_HORIZON: int = 90
AVAILABLE_HORIZONS: list[int] = [30, 60, 90]
DEFAULT_FUP: int = 30
AVAILABLE_FUPS: list[int] = [30 * i for i in range(1, 25)]
RISK_THRESHOLD: float = 0.50
CLUSTERING_METHOD: str = "clusterer"  # "model_risk" (threshold-based) or "clusterer" (HDBSCAN on UMAP)
CLUSTERING_UMAP_COMPONENTS: int = 2   # Matches analysis_stratification.py UMAP reduction components
HDBSCAN_MIN_CLUSTER_SIZE: int = 15    # Matches analysis_stratification.py min_cluster_size
HDBSCAN_MIN_SAMPLES: int = 5          # Matches analysis_stratification.py min_samples
HORIZON_LABELS: dict[int, str] = {30: "30 days", 60: "60 days", 90: "90 days"}

# EAV column mapping (must match discriminative_training.yaml)
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
MOCK_RANDOM_SEED:  int = 42

# =============================================================================
# CLINICAL VOCABULARY
# Drives the structured-input form dropdowns.
# Reflects the AIIDKIT t-EAV entity/attribute vocabulary.
# =============================================================================

ORDINAL = ["Lowest", "Lower", "Low", "Middle", "High", "Higher", "Highest"]
BINARY  = ["0", "1"]

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
        "Diagnosis": ["Diabetes mellitus", "Other", "Other metabolic disease"],
    },
    "Donor info": {
        "Age at transplant (years)": ["Lowest", "Lower", "Low"],
        "Altruistic": ["0"],
        "Blood group": ["0", "A", "AB"],
        "Cause of death": ["ANX"],
        "History of CAD": ["0"],
        "History of myocardial contusion": ["0"],
        "Identical twin": ["0"],
        "KPD": ["0"],
        "Sex": ["Female", "Male"],
        "Smoker": ["0"],
        "Type": ["Living related"],
        "Valvular disease": ["0"],
    },
    "Graft Loss": {
        "Cause": ["Other causes"],
    },
    "History": {
        "Pregnancy": ["0"],
        "Transfusion": ["0"],
    },
    "Immunology": {
        "Calculated PRA (%)": ["0"],
        "Latest cPRA (%)": ["Lowest"],
        "Peak cPRA (%)": ["Lowest"],
    },
    "Infection": {
        "Category": ["Bacteria", "Fungi", "Virus"],
        "Clinically significant": ["False", "True"],
        "Microbiology specimen": ["Blood", "Urine"],
        "Pathogen": ["E. coli", "Klebsiella pneumoniae", "Pseudomonas aeruginosa", "Staphylococcus aureus"],
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
            "Quinolone",
            "Valaciclovir",
            "Valganciclovir",
        ],
        "Stop (infection prophylaxis)": [
            "Atovaquone",
            "Beta-Lactame",
            "Cotrimoxazole",
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
        "Mismatch HLA sum (minimum)": ["Lowest", "Middle", "High"],
        "Mismatch HLA-A": ["1", "2"],
        "Mismatch HLA-B": ["1", "2"],
        "Mismatch HLA-DQB1": ["0"],
        "Mismatch HLA-DRB1": ["1"],
        "Mismatch HLA-DRB35": ["1"],
    },
    "Organ event": {
        "Diagnosis": ["CNI nephrotoxicity", "Graft pyelonephritis", "Lymphocele", "Other"],
    },
    "Organ lab": {
        "Albumin (g/l)": ["Highest"],
        "Factor V (%)": ["Highest"],
        "Fibrinogen (g/l)": ["Highest"],
        "Proteinuria (mg/mmol)": ORDINAL,
        "Triglycerides (mmol/l)": ["Lowest", "Low"],
    },
    "Patient info": {
        "Age at transplant (years)": ["Lowest", "Middle", "Higher", "Highest"],
        "Blood group": ["0", "A"],
        "Ethnicity": ["Caucasian"],
        "Etiology": ["Condition unknown", "GN", "Hereditary non_PCKD", "PCKD"],
        "Pre-transplant immuno-suppressed": ["No", "Yes"],
        "Previous transplant count (any organ)": ["0"],
        "Previous transplanted organ count": ["0", "1"],
        "Sex": ["Female", "Male"],
    },
    "Patient lab": {
        "CRP (mg/l)": ORDINAL,
        "Creatinine (µmol/l)": ORDINAL,
        "Glucose (mmol/l)": ORDINAL,
        "Glycated hemoglobin (HbA1c) (%)": ORDINAL,
        "HDL cholesterol (mmol/l)": ["Lowest", "Lower", "Low", "High", "Higher", "Highest"],
        "LDL cholesterol (mmol/l)": ["Lowest", "Lower", "Middle", "High", "Higher", "Highest"],
        "Leukocytes (G/l)": ORDINAL,
        "Lymphocytes (G/l)": ORDINAL,
        "Tacrolimus level (µg/l)": ORDINAL,
        "Total bilirubin (µmol/l)": ["Middle", "High"],
        "Total cholesterol (mmol/l)": ORDINAL,
        "eGFR (mL/min/1.73m²)": ORDINAL,
    },
    "Patient serology": {
        "CMV": BINARY,
        "EBV": ["1"],
        "HBcAb": ["0"],
        "HBsAb": ["1"],
        "HBsAg": ["0"],
        "HCV": ["0"],
        "HIV": ["0"],
        "HSV": ["1"],
        "Syphilis": ["0"],
        "Toxoplasmosis": BINARY,
        "VZV": ["1"],
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
        "CMV": ["D+/R+", "D-/R-"],
        "EBV": ["D+/R+"],
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
        "ABO compatible": ["1"],
        "Asystolic ischemia": ["Middle"],
        "Cold ischemia time (hours)": ["Low", "High"],
        "Delayed graft function": ["0"],
        "Dialysis Type (Pre-Tx)": ["HD", "PD"],
        "Hosp. stay duration so far (days)": ["Lowest", "High", "Higher", "Highest"],
        "Hospital discharge": ["Kidney"],
        "Listing Organ Type": ["Kidney"],
        "Previous pregnancy": ["0"],
        "Previous transfusion": BINARY,
        "Previous transplant count (kidney)": ["0"],
        "Procedure provider": ["Insel"],
        "Re-transplant": ["First"],
        "Total Organ Count": ["1"],
        "Transplant Order": ["1"],
        "Transplant Resection": ["Re Tpx"],
        "Transplant event": ["Kidney"],
        "Transplantation year": ["Lowest"],
        "Virtual crossmatch": ["0"],
        "Waitlist removal": ["Kidney"],
        "Waitlist start": ["Kidney", "Kidney - Islets", "Kidney - Pancreas"],
        "urgentlisting": ["0"],
    },
    "Virology": {
        "Detection": ["BKV", "HBV", "HCV"],
        "Dynamics": ["Monitoring"],
        "Viral load (copies/ml)": ["Low", "Higher", "Highest"],
    },
}

# =============================================================================
# SERVER
# =============================================================================

HOST:  str  = "0.0.0.0"
PORT:  int  = 5000
DEBUG: bool = True

# Reloader exclude patterns (fnmatch wildcards) to ignore when debug=True.
# Prevents Flask/Werkzeug from restarting the app when files in these folders are modified.
EXCLUDE_PATTERNS: list[str] = ["*/scripts/*"]
