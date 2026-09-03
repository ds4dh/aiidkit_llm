# Patient sequence modeling project

This project trains transformer-based models on sequential electronic health records (EHR) of kidney transplant recipients. The pipeline supports learning robust patient representations via masked language modeling (MLM), fine-tuning for downstream clinical predictions (e.g., infections, graft loss, or death), training classic machine learning baselines, and extracting clinical interpretability.

Note: This pipeline requires access to the AIIDKIT dataset. You must place the raw data files in the appropriate data directories before running the code.

## Installation

First, install the `uv` tool and set up the virtual environment with all required packages.

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create and activate the virtual environment
uv venv --python 3.11
source .venv/bin/activate

# Install PyTorch with CUDA 12.4 support (for GPU systems)
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Install the project and dependencies
uv pip install -e ".[dev]"
uv pip install flash-attn --no-build-isolation
```

## Usage and paper replication

Follow these step-by-step instructions to replicate all experimental results, figures, and analyses presented in the paper (*Infection Risk Prediction After Kidney Transplantation Using Timed Entity-Attribute-Value Transformer Encoders*).

---

### (Optional) Generate synthetic data for local testing

Optionally, generate synthetic datasets for local code verification and pipeline testing:

```bash
python scripts/generate_synthetic_data.py --output_dir data/synthetic --samples 100
```

> **Note**: Synthetic data is created purely for verifying code functionality, data loading, and end-to-end pipeline execution. Training and evaluating models on synthetic data will not produce clinically meaningful results.

---

### Step 1: Pre-train the t-EAV Transformer model

Pre-train the backbone model using self-supervised Masked Language Modeling (MLM) on full patient Entity-Attribute-Value sequences:

```bash
# Choose overrides: synthetic data for fast local verification vs real STCS dataset
OVERRIDES='{"data_dir": "data/synthetic/teav", "pretrainer": {"max_steps": 5, "eval_steps": 5, "save_steps": 5}}'  # Synthetic local test (default)
# OVERRIDES='{"data_dir": "path/to/stcs/teav"}'  # Real STCS dataset

# Temporal split (earliest 85% train/val, prospective 15% test)
python scripts/train_mlm.py -c configs/discriminative_training.yaml --overrides "$OVERRIDES" --overrides '{"data_split_type": "random_split"}'

# Random split (70% train / 15% val / 15% test)
python scripts/train_mlm.py -c configs/discriminative_training.yaml --overrides "$OVERRIDES" --overrides '{"data_split_type": "temporal_split"}'

# Center split (CHUV center held out as test set)
python scripts/train_mlm.py -c configs/discriminative_training.yaml --overrides "$OVERRIDES" --overrides '{"data_split_type": "center_split"}'
```

---

### Step 2: Fine-tune for infection prediction tasks

Fine-tune the pre-trained Transformer model on downstream binary prediction tasks (`infection_bacteria` and `infection_virus`) across prediction horizons (30, 60, 90 days):

```bash
# Choose overrides: synthetic data for fast local verification vs real STCS dataset
OVERRIDES='{"data_dir": "data/synthetic/teav", "finetuner": {"max_steps": 5, "eval_steps": 5, "save_steps": 5}}'  # Synthetic local test (default)
# OVERRIDES='{"data_dir": "path/to/stcs/teav"}'  # Real STCS dataset

# Fine-tune on temporal split
python scripts/train_classification.py -c configs/discriminative_training.yaml --overrides "$OVERRIDES" --overrides '{"data_split_type": "random_split"}'

# Fine-tune on random split
python scripts/train_classification.py -c configs/discriminative_training.yaml --overrides "$OVERRIDES" --overrides '{"data_split_type": "temporal_split"}'

# Fine-tune on center split
python scripts/train_classification.py -c configs/discriminative_training.yaml --overrides "$OVERRIDES" --overrides '{"data_split_type": "center_split"}'
```

---

### Step 3: Train classic ML baselines with Optuna

Train traditional machine learning baselines (Logistic Regression, Random Forest, XGBoost) using Optuna hyperparameter optimization across all split strategies:

```bash
# Choose overrides: synthetic data for fast local verification vs real STCS dataset
OVERRIDES='{"data_dir": "data/synthetic/classic_ml", "models": {"logistic_regression": {"n_optuna_trials": 2}, "random_forest": {"n_optuna_trials": 2}, "xgboost": {"n_optuna_trials": 2}}}'  # Synthetic local test (default)
# OVERRIDES='{"data_dir": "path/to/stcs/classic_ml"}'  # Real STCS dataset

# Train ML baselines on temporal split
python scripts/train_classic_ml.py -c configs/discriminative_classic_ml.yaml --overrides "$OVERRIDES" --overrides '{"data_split_type": "random_split"}'

# Train ML baselines on random split
python scripts/train_classic_ml.py -c configs/discriminative_classic_ml.yaml --overrides "$OVERRIDES" --overrides '{"data_split_type": "temporal_split"}'

# Train ML baselines on center split
python scripts/train_classic_ml.py -c configs/discriminative_classic_ml.yaml --overrides "$OVERRIDES" --overrides '{"data_split_type": "center_split"}'
```

---

### Step 4: Model performance and statistical comparisons (Figures 1 and 2)

Evaluate discriminative performance (ROC-AUC, PR-AUC), operational utility (Sensitivity and Specificity at 0.80 recall anchor), Decision Curve Analysis (DCA), McNemar's exact tests, and paired bootstrap PR-AUC tests across five post-transplant clinical phases:

1. **Perioperative (POP)**: 0–30 days
2. **Opportunistic (OPT)**: 31–180 days
3. **Maintenance (MTN)**: 181–360 days
4. **Long-term (LT)**: 361–1,080 days
5. **Very Long-term (VLT)**: 1,081–3,600 days

```bash
# Choose dataset directory: synthetic data for fast local verification vs real STCS dataset
DATA_DIR="data/synthetic/teav"  # Synthetic local test (default)
# DATA_DIR="path/to/stcs/teav"  # Real STCS dataset

python scripts/analysis_comparison.py --data-dir "$DATA_DIR"
```

---

### Step 5: Clinical feature attribution and interpretability (Figure 3)

Extract sequence-level feature attributions using Layer Integrated Gradients via Captum across clinical post-transplant timelines:

```bash
# Choose dataset directory: synthetic data for fast local verification vs real STCS dataset
DATA_DIR="data/synthetic/teav"  # Synthetic local test (default)
# DATA_DIR="path/to/stcs/teav"  # Real STCS dataset

python scripts/analysis_interpretability.py --data-dir "$DATA_DIR"
```

---

### Step 6: Patient stratification and survival analysis (Figure 4)

Extract 2D UMAP visualizations of the learned embedding space, perform HDBSCAN clustering, and calculate Kaplan-Meier infection-free survival curves:

```bash
# Choose dataset directory: synthetic data for fast local verification vs real STCS dataset
DATA_DIR="data/synthetic/teav"  # Synthetic local test (default)
# DATA_DIR="path/to/stcs/teav"  # Real STCS dataset

python scripts/analysis_stratification.py --data-dir "$DATA_DIR" --data-split-type temporal_split
```
