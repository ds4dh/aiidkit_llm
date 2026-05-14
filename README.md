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

# Install the project and dependencies
uv pip install -e ".[dev]"
uv pip install flash-attn --no-build-isolation
```

## Usage

The pipeline is split into modular scripts. Follow these steps to replicate the end-to-end workflow.

### Step 1: Pre-train the model

This script pre-trains the backbone model using masked language modeling (MLM) on the patient Entity-Attribute-Value (EAV) sequences.

```bash
python scripts/train_mlm.py -c configs/discriminative_training.yaml
```

### Step 2: Fine-tune for classification

Once pre-trained, fine-tune the model on specific predictive tasks using specialized loss functions (e.g., Poly1, Focal Loss, Weighted CE) and continuous follow-up window evaluations.

```bash
python scripts/train_classification.py -c configs/discriminative_training.yaml
```

### Step 3: Hyperparameter optimization (optional)

To run a distributed hyperparameter search across available GPUs, use the Optuna orchestration script. This manages both pre-training and fine-tuning trials.

```bash
python scripts/train_optuna.py -c configs/discriminative_training.yaml
```

### Step 4: Train classic ML baselines

Train traditional machine learning models (XGBoost, Random Forest, Logistic Regression) on aggregated feature sets to establish performance baselines.

```bash
python scripts/train_classic_ml.py -c configs/discriminative_classic_ml.yaml
```

### Step 5: Generate interpretability and comparative results

Evaluate the fine-tuned models to understand feature importance and clinical drivers using Captum (Integrated Gradients), compare the performance of the Transformer model to the classic baselines, or group the population based on predicted risk, for stratified survival analysis.

```bash
python scripts/interpret_models.py
python scripts/compare_models.py
python scripts/stratify_models.py
```

## Scripts overview

* `scripts/train_mlm.py`: Pre-trains the patient sequence model using MLM on EAV events.
* `scripts/train_classification.py`: Fine-tunes the pretrained model for multi-label or binary clinical prediction tasks with continuous follow-up evaluations.
* `scripts/train_optuna.py`: Orchestrates hyperparameter optimization using Optuna, distributing trials across multiple GPUs.
* `scripts/train_classic_ml.py`: Builds pipelines and trains classic ML baselines with automated feature selection and imputation.
* `scripts/train_generative.py`: Fine-tunes generative LLMs on text-formatted patient narratives (using TRL).
* `scripts/interpret_models.py`: Runs interpretability analysis to map model predictions back to specific clinical events.
* `scripts/compare_models.py` & `src/evaluation/plot_results.py`: Aggregates evaluation metrics (ROC AUC, PR AUC, Net Benefit, Brier) and generates comparative visual reports.
