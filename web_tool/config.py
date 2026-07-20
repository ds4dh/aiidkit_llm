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
    "processed/v3.6/teav/temporal_split"
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
DEFAULT_HORIZON: int = 30
AVAILABLE_HORIZONS: list[int] = [30, 60, 90]
DEFAULT_FUP: int = 90
AVAILABLE_FUPS: list[int] = [0, 30, 90, 180, 360, 720]
RISK_THRESHOLD: float = 0.75
CLUSTERING_METHOD: str = "clusterer"  # "model_risk" (threshold-based) or "clusterer" (HDBSCAN on UMAP)
CLUSTERING_UMAP_COMPONENTS: int = 15  # Dimensionality reduction components before clustering
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
    "Medication": {
        "Tacrolimus":               ORDINAL,
        "Cyclosporine":             ORDINAL,
        "Mycophenolate Mofetil":    BINARY,
        "Mycophenolic Acid":        BINARY,
        "Everolimus":               BINARY,
        "Sirolimus":                BINARY,
        "Glucocorticoid":           ORDINAL,
        "Methylprednisolone":       BINARY,
        "Belatacept":               BINARY,
        "Basiliximab":              BINARY,
        "Anti-thymocyte globulin":  BINARY,
        "Valganciclovir":           BINARY,
        "Valaciclovir":             BINARY,
        "Cotrimoxazole":            BINARY,
        "Rituximab":                BINARY,
    },
    "Lab result": {
        "Creatinine":               ORDINAL,
        "eGFR":                     ORDINAL,
        "Tacrolimus level":         ORDINAL,
        "WBC":                      ORDINAL,
        "CRP":                      ORDINAL,
        "Hemoglobin":               ORDINAL,
        "Platelet count":           ORDINAL,
        "Lymphocyte count":         ORDINAL,
        "Albumin":                  ORDINAL,
    },
    "Infection": {
        "Bacterial infection":      BINARY,
        "CMV":                      BINARY,
        "BKV":                      BINARY,
        "UTI":                      BINARY,
        "Respiratory infection":    BINARY,
        "Viral syndrome":           BINARY,
        "Fungal infection":         BINARY,
        "Sepsis/Bacteremia":        BINARY,
    },
    "Clinical event": {
        "Rejection event":          BINARY,
        "Transplant procedure":     ["Kidney tpx", "Kidney - Pancreas", "Kidney - Liver"],
        "Non-transplant surgery":   BINARY,
        "Transplantation event":    ["1"],
    },
    "Comorbidity": {
        "Diabetes mellitus":        BINARY,
        "Hypertension":             BINARY,
        "CAD":                      BINARY,
        "COPD":                     BINARY,
        "HIV":                      BINARY,
    },
    "Donor": {
        "Donor type":               ["DBD", "DCD", "Living related", "Living unrelated"],
        "CMV D/R status":           ["D+/R+", "D+/R-", "D-/R+", "D-/R-"],
    },
    "Patient": {
        "Age at transplant":        ORDINAL,
        "Sex":                      ["Male", "Female"],
        "BMI":                      ORDINAL,
        "Dialysis duration":        ORDINAL,
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
