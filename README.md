# Patient sequence modeling project

This project trains transformer-based models on sequential electronic health records (EHR) of kidney transplant recipients. The pipeline supports learning patient representations via masked language modeling (MLM), fine-tuning for downstream clinical predictions (bacterial and viral infections across multiple horizons), training classic machine learning baselines, and extracting clinical interpretability.

---

## Installation

Install uv and configure the environment:

```bash
# Install uv
curl -LsSf [https://astral.sh/uv/install.sh](https://astral.sh/uv/install.sh) | sh

# Create and activate environment
uv venv --python 3.14
source .venv/bin/activate

# Install PyTorch with CUDA support
uv pip install torch torchvision --index-url [https://download.pytorch.org/whl/cu132](https://download.pytorch.org/whl/cu132)

# Install package in editable mode with development dependencies
uv pip install -e ".[dev]"
```

---

## Data preparation

### Option A: Preprocessing the STCS dataset

Access to the <a href="https://www.stcs.ch/for-investigators/#general_rules" target="_blank" rel="noopener noreferrer">Swiss Transplant Cohort Study (STCS)</a> dataset is necessary to reproduce the reported clinical findings.

Upon receiving data access approval from the STCS, process the raw cohort tables into timed EAV sequences and classic ML tabular matrices using the preprocessing pipeline in [`aiidkit`](https://github.com/mhmmdrz92/aiidkit).

1. Clone the preprocessing repository:

```bash
git clone [https://github.com/mhmmdrz92/aiidkit.git](https://github.com/mhmmdrz92/aiidkit.git) aiidkit_data_preprocessing
cd aiidkit_data_preprocessing
```

2. Export your base data directory and convert raw CSV tables to Parquet:

```bash
# Set base data directory containing raw CSV exports and future processed outputs
export STCS_DATA_DIR="/path/to/stcs_data"

# Convert raw CSV tables to Parquet
python scripts/convert_files_to_parquet.py \
    --raw-csv-dir "${STCS_DATA_DIR}/csv" \
    --output-dir "${STCS_DATA_DIR}/parquet"
```

3. Generate the downstream modeling datasets:

```bash
# Build timed EAV sequence datasets (for MLM and Transformer fine-tuning)
python scripts/build_teav_datasets.py --data-dir "$STCS_DATA_DIR"

# Build tabular aggregated datasets (for Logistic Regression, Random Forest, XGBoost)
python scripts/build_classic_ml_dataset.py --data-dir "$STCS_DATA_DIR"
```

Once completed, versioned modeling datasets will be available under `${STCS_DATA_DIR}/processed/v3.9/teav` and `${STCS_DATA_DIR}/processed/v3.9/classic_ml`.

### Option B: Generating synthetic data (code verification only)

If you do not have STCS data access and simply want to verify that the code and environment run end-to-end, generate lightweight synthetic data:

```bash
python scripts/generate_synthetic_data.py --output_dir data/synthetic --samples 100
```

*(Note: Synthetic data is solely for pipeline verification and will not produce meaningful clinical results.)*

---

## Modelling pipeline execution

The entire workflow (pre-training, fine-tuning, classic ML baselines, and downstream evaluation) is orchestrated by `scripts/train_all.sh`.

### 1. Configure the pipeline

Before running the orchestrator, open `scripts/train_all.sh` and set the configuration flags at the top of the file to match your setup:

* **Data source (`USE_SYNTHETIC`):** Set to `false` for the real STCS dataset, or `true` if using the synthetic verification dataset.
* **Data paths:** If using real data, set `STCS_DATA_ROOT` to your processed directory (e.g., `${STCS_DATA_DIR}/processed/v3.9`). If using synthetic data, set `SYNTHETIC_DATA_ROOT` to the synthetic data output location.
* **Stage toggles:** Enable or disable individual stages as needed:
  * `RUN_STEP_1_MLM`: Transformer MLM pre-training
  * `RUN_STEP_2_FINETUNING`: Transformer infection classification fine-tuning
  * `RUN_STEP_3_CLASSIC_ML`: LR, RF, and XGBoost training with Optuna
  * `RUN_STEP_4_COMPARISON`: Decision curve analysis (DCA) and statistical comparisons
  * `RUN_STEP_5_INTERPRET`: Layer integrated gradient feature attribution
  * `RUN_STEP_6_SURVIVAL`: Patient clustering and Kaplan-Meier survival analysis

### 2. Run the pipeline

Once configured, launch the complete execution script:

```bash
bash scripts/train_all.sh
```