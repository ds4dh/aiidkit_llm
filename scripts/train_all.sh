#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# DATA MODE AND ROOT DIRECTORIES
# ==============================================================================
# If true, the pipeline will run quick tests with small synthetic data
USE_SYNTHETIC=false

# Root base directories (subfolders are 'teav' and 'classic_ml')
REAL_DATA_ROOT="/home/shares/ds4dh/aiidkit_project/data_new/processed/v3.8"
SYNTHETIC_DATA_ROOT="$(pwd)/data/synthetic"

# ==============================================================================
# STEP EXECUTION TOGGLES
# ==============================================================================
RUN_STEP_1_MLM=true          # Step 1: Pre-train t-EAV Transformer
RUN_STEP_2_FINETUNING=true   # Step 2: Fine-tune for Infection Prediction Tasks
RUN_STEP_3_CLASSIC_ML=true   # Step 3: Classic ML Baselines (LR, RF, XGBoost)
RUN_STEP_4_COMPARISON=true   # Step 4: DCA, McNemar, PR-AUC
RUN_STEP_5_INTERPRET=true    # Step 5: Captum Feature Attribution
RUN_STEP_6_SURVIVAL=true     # Step 6: Patient Stratification & Survival

# ==============================================================================
# ACTIVE PATH SELECTION AND DIRECTORY VERIFICATION
# ==============================================================================
if [ "$USE_SYNTHETIC" = true ]; then
    echo "=== Running pipeline in SYNTHETIC verification mode ==="
    DATA_ROOT="$SYNTHETIC_DATA_ROOT"
    
    TEAV_DATA_DIR="${DATA_ROOT}/teav"
    CLASSIC_DATA_DIR="${DATA_ROOT}/classic_ml"

    MLM_OVERRIDES='{"data_dir": "'"$TEAV_DATA_DIR"'", "pretrainer": {"max_steps": 5, "eval_steps": 5, "save_steps": 5}}'
    FT_OVERRIDES='{"data_dir": "'"$TEAV_DATA_DIR"'", "finetuner": {"max_steps": 5, "eval_steps": 5, "save_steps": 5}}'
    ML_OVERRIDES='{"data_dir": "'"$CLASSIC_DATA_DIR"'", "models": {"logistic_regression": {"n_optuna_trials": 2}, "random_forest": {"n_optuna_trials": 2}, "xgboost": {"n_optuna_trials": 2}}}'
else
    echo "=== Running pipeline in REAL dataset mode ==="
    DATA_ROOT="$REAL_DATA_ROOT"
    
    TEAV_DATA_DIR="${DATA_ROOT}/teav"
    CLASSIC_DATA_DIR="${DATA_ROOT}/classic_ml"

    MLM_OVERRIDES="{\"data_dir\": \"$TEAV_DATA_DIR\"}"
    FT_OVERRIDES="{\"data_dir\": \"$TEAV_DATA_DIR\"}"
    ML_OVERRIDES="{\"data_dir\": \"$CLASSIC_DATA_DIR\"}"
fi

# Verify root directory
if [ ! -d "$DATA_ROOT" ]; then
    echo "Error: Data root directory does not exist: $DATA_ROOT" >&2
    exit 1
fi

# Verify subdirectories for active stages
if [ "$RUN_STEP_1_MLM" = true ] || [ "$RUN_STEP_2_FINETUNING" = true ] || [ "$RUN_STEP_4_COMPARISON" = true ] || [ "$RUN_STEP_5_INTERPRET" = true ] || [ "$RUN_STEP_6_SURVIVAL" = true ]; then
    if [ ! -d "$TEAV_DATA_DIR" ]; then
        echo "Error: TEAV data directory does not exist: $TEAV_DATA_DIR" >&2
        exit 1
    fi
fi

if [ "$RUN_STEP_3_CLASSIC_ML" = true ]; then
    if [ ! -d "$CLASSIC_DATA_DIR" ]; then
        echo "Error: Classic ML data directory does not exist: $CLASSIC_DATA_DIR" >&2
        exit 1
    fi
fi

export DATA_DIR="$TEAV_DATA_DIR"
export TEAV_DATA_DIR="$TEAV_DATA_DIR"

SPLITS=("temporal_split" "random_split" "center_split")

# Track background process PIDs
PIDS=()

# ==============================================================================
# PARALLEL BRANCH A: Transformer pipeline (MLM -> Finetuning in sequence)
# ==============================================================================
if [ "$RUN_STEP_1_MLM" = true ] || [ "$RUN_STEP_2_FINETUNING" = true ]; then
    (
        set -euo pipefail
        if [ "$RUN_STEP_1_MLM" = true ]; then
            echo -e "\n[Transformer-Branch] Starting Step 1: MLM Pre-training..."
            for split in "${SPLITS[@]}"; do
                echo "[Transformer-Branch] Running MLM for split: $split"
                python scripts/train_mlm.py \
                    -c configs/discriminative_training.yaml \
                    --overrides "$MLM_OVERRIDES" \
                    --overrides "{\"data_split_type\": \"$split\"}"
            done
        fi

        if [ "$RUN_STEP_2_FINETUNING" = true ]; then
            echo -e "\n[Transformer-Branch] Starting Step 2: Fine-tuning..."
            for split in "${SPLITS[@]}"; do
                echo "[Transformer-Branch] Fine-tuning for split: $split"
                python scripts/train_classification.py \
                    -c configs/discriminative_training.yaml \
                    --overrides "$FT_OVERRIDES" \
                    --overrides "{\"data_split_type\": \"$split\"}"
            done
        fi
        echo -e "\n[Transformer-Branch] Completed successfully."
    ) &
    PID_TRANSFORMER=$!
    PIDS+=("$PID_TRANSFORMER")
    echo "Spawned Transformer background job (PID: $PID_TRANSFORMER)"
fi

# ==============================================================================
# PARALLEL BRANCH B: Classic ML baselines (CPU/Optuna)
# ==============================================================================
if [ "$RUN_STEP_3_CLASSIC_ML" = true ]; then
    (
        set -euo pipefail
        echo -e "\n[ClassicML-Branch] Starting Step 3: Classic ML Baselines..."
        for split in "${SPLITS[@]}"; do
            echo "[ClassicML-Branch] Training classic ML for split: $split"
            python scripts/train_classic_ml.py \
                -c configs/discriminative_classic_ml.yaml \
                --overrides "$ML_OVERRIDES" \
                --overrides "{\"data_split_type\": \"$split\"}"
        done
        echo -e "\n[ClassicML-Branch] Completed successfully."
    ) &
    PID_CLASSIC=$!
    PIDS+=("$PID_CLASSIC")
    echo "Spawned Classic ML background job (PID: $PID_CLASSIC)"
fi

# ==============================================================================
# BARRIER: Wait for parallel model training branches to finish
# ==============================================================================
if [ ${#PIDS[@]} -gt 0 ]; then
    echo -e "\n>>> Waiting for model training branches to finish..."
    for pid in "${PIDS[@]}"; do
        wait "$pid"
    done
    echo -e ">>> All model trainings complete! Proceeding to analysis.\n"
fi

# ==============================================================================
# ANALYSIS: Model performance and statistical comparisons
# ==============================================================================
if [ "$RUN_STEP_4_COMPARISON" = true ]; then
    echo -e "\n>>> [Step 4/6] Running Model Comparison Analysis (DCA, McNemar, PR-AUC)..."
    python scripts/analysis_comparison.py --data-dir "$TEAV_DATA_DIR"
else
    echo -e "\n>>> [Step 4/6] Skipped (RUN_STEP_4_COMPARISON=false)"
fi

# ==============================================================================
# ANALYSIS: Feature attribution and clinical interpretability
# ==============================================================================
if [ "$RUN_STEP_5_INTERPRET" = true ]; then
    echo -e "\n>>> [Step 5/6] Running Captum Feature Attribution..."
    python scripts/analysis_interpretability.py \
        --data-dir "$TEAV_DATA_DIR" \
        --data-split-type "temporal_split"
else
    echo -e "\n>>> [Step 5/6] Skipped (RUN_STEP_5_INTERPRET=false)"
fi

# ==============================================================================
# ANALYSIS: Patient stratification and survival analysis
# ==============================================================================
if [ "$RUN_STEP_6_SURVIVAL" = true ]; then
    echo -e "\n>>> [Step 6/6] Running Stratification and Survival Analysis..."
    python scripts/analysis_stratification.py \
        --data-dir "$TEAV_DATA_DIR" \
        --data-split-type "temporal_split"
else
    echo -e "\n>>> [Step 6/6] Skipped (RUN_STEP_6_SURVIVAL=false)"
fi

echo -e "\n=== Pipeline run completed! ==="