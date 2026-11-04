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

Access to the [Swiss Transplant Cohort Study (STCS)](https://www.stcs.ch/for-investigators/#general_rules) dataset is necessary to reproduce the reported clinical findings.

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

The entire workflow (pre-training, survival and binary fine-tuning, classic ML baselines, and downstream evaluation) is orchestrated by `scripts/train_all.sh`.

### 1. Configure the pipeline

Before running the orchestrator, open `scripts/train_all.sh` and set the configuration flags at the top of the file to match your setup:

* **Data source (`USE_SYNTHETIC`):** Set to `false` for the real STCS dataset, or `true` if using the synthetic verification dataset.
* **Data paths:** If using real data, set `STCS_DATA_ROOT` to your processed directory (e.g., `${STCS_DATA_DIR}/processed/v3.9`). If using synthetic data, set `SYNTHETIC_DATA_ROOT` to the synthetic data output location.
* **Stage toggles:** Enable or disable individual stages as needed:
  * `RUN_STEP_1_MLM`: Transformer MLM pre-training
  * `RUN_STEP_2_SURVIVAL_FT`: Transformer survival (Cox) fine-tuning — runs before binary fine-tuning
  * `RUN_STEP_3_FINETUNING`: Transformer infection classification fine-tuning (binary)
  * `RUN_STEP_4_CLASSIC_ML`: LR, RF, and XGBoost training with Optuna
  * `RUN_STEP_5_COMPARISON`: Decision curve analysis (DCA) and statistical comparisons
  * `RUN_STEP_6_SURVIVAL_COMP`: Cross-family comparison (classic ML vs binary vs survival)
  * `RUN_STEP_7_INTERPRET`: Layer integrated gradient feature attribution
  * `RUN_STEP_8_STRATIFICATION`: Patient clustering and Kaplan-Meier survival analysis

Within the transformer branch, the stages run in sequence: MLM pre-training → survival fine-tuning → binary classification fine-tuning, so the survival models are always trained first after pre-training (they reuse the Step 1 checkpoint). The classic ML branch (CPU/Optuna) runs in parallel.

### 2. Run the pipeline

Once configured, launch the complete execution script:

```bash
bash scripts/train_all.sh
```

### 3. Survival fine-tuning (alternative to binary classification)

Survival fine-tuning is part of the pipeline (Step 2, executed before the binary
fine-tuning of Step 3), but it can also be run standalone with the command below.
Instead of binary labels per horizon (`label_{task}_{H}d`), the raw datasets also contain
survival labels per endpoint: `label_{outcome}_time` (time-to-event or censoring, in days
from the fup observation point) and `label_{outcome}_status` (1 = event, 0 = right-censored),
computed in the preprocessing repo (`../aiidkit_mhmmdrz`). Endpoints available out of the
box: `infection`, `death`, `graft_loss`, and one per pathogen group (`grp_{pathogen_group}`,
21 groups). Category endpoints (`infection_bacteria`, `infection_virus`, `infection_fungi`)
are stored as `label_infection_{category}_{time,status}` columns as of dataset v3.9 and are
used as-is; the fallback that pools pathogen-group labels on the fly (`src/data/survival_labels.py`)
only runs for pre-v3.9 data, prints a `WARN`, and is not identical to the stored labels (~1% of
rows, ~3% fewer events: episodes without a pathogen-group assignment are missing from the groups).
Watch the naming: a group endpoint is `grp_{group}` for survival but `infection_grp_{group}` for
the binary pipeline.

The model keeps the same architecture, but the head outputs a single log hazard ratio per
patient, trained with a Cox partial likelihood (Breslow ties, `src/model/survival_utils.py`):

```bash
# Fine-tune survival models (config: configs/survival_training.yaml)
# Reuses the MLM checkpoint from results/transformer (pretrain_result_dir)
python scripts/train_survival.py -c configs/survival_training.yaml
```

Evaluation during training reports Harrell's C-index plus horizon-wise AUROC / AP / IPCW-Brier
(computed by converting Cox risks into survival curves via a Breslow baseline), and each run
saves the same JSON/NPZ structure as the classification pipeline (plus `patientid` alignment
arrays) for downstream analyses.

### 4. Cross-family comparison (classic ML vs binary vs survival)

Run automatically by the pipeline as Step 6, `scripts/analysis_survival_comparison.py`
evaluates all model families on the shared survival
ground truth (aligned per patient and fup observation point), using metrics that apply to both
binary-risk and survival models:

* AUROC / AP at each horizon (binary targets derived by binarizing survival labels; rows
  censored before the horizon are masked out),
* IPCW (Graf) Brier score at each horizon and IPCW Integrated Brier Score over the grid,
* C-index stratified by fup (comparable pairs only within the same observation point),
* stratified paired bootstrap p-values (reference model vs each competitor).

```bash
# Compare all enabled families (config: configs/survival_training.yaml, "comparison:" section;
# this is the same config used by train_survival.py, which ignores that section)
python scripts/analysis_survival_comparison.py -c configs/survival_training.yaml
# Outputs: metrics_by_horizon.csv, metrics_survival_level.csv, bootstrap_pvalues.csv
```
