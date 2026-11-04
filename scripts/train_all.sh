#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# DATA MODE AND ROOT DIRECTORIES
# ==============================================================================
# If true, the pipeline will run quick tests with small synthetic data
USE_SYNTHETIC=false

# Root base directories (subfolders are 'teav' and 'classic_ml')
STCS_DATA_ROOT="/home/shares/ds4dh/aiidkit_project/data_new/processed/v3.9"
SYNTHETIC_DATA_ROOT="$(pwd)/data/synthetic"

# ==============================================================================
# DEBUG MODE (smoke test: same pipeline, but every training run is capped to a
# small number of steps so it fits quickly within the HPC walltime).
# Enable with:  DEBUG_MODE=true bash scripts/train_all.sh   (or flip below)
# Optional:     DEBUG_MAX_STEPS=50 bash scripts/train_all.sh
# ==============================================================================
DEBUG_MODE=${DEBUG_MODE:-false}
DEBUG_MAX_STEPS=${DEBUG_MAX_STEPS:-100}

# ==============================================================================
# RESULTS AND LOGGING
# Python script output is NOT printed to the terminal. Each step logs to its own
# file under LOG_DIR (a timestamped directory inside the results folder). The
# script itself prints progress and how to "peek" into the logs.
# Override with: RESULTS_DIR=/path bash scripts/train_all.sh
#           or:  LOG_DIR=/path    bash scripts/train_all.sh
# ==============================================================================
RESULTS_DIR=${RESULTS_DIR:-"$(pwd)/results"}
LOG_DIR=${LOG_DIR:-"${RESULTS_DIR}/logs/train_all_$(date +%Y%m%d_%H%M%S)"}
mkdir -p "$LOG_DIR"

# ==============================================================================
# CONFIG FILES PER STEP  --  edit here (all also env-overridable)
# ==============================================================================
MLM_CONFIG=${MLM_CONFIG:-"configs/discriminative_training.yaml"}                  # Step 1: MLM pre-training
SURVIVAL_FT_CONFIG=${SURVIVAL_FT_CONFIG:-"configs/survival_training.yaml"}        # Step 2: Survival (Cox) fine-tuning
FT_CONFIG=${FT_CONFIG:-"configs/discriminative_training_multitask.yaml"}          # Single-task: configs/discriminative_training.yaml
CLASSIC_ML_CONFIG=${CLASSIC_ML_CONFIG:-"configs/discriminative_classic_ml.yaml"}  # Step 4: LR / RF / XGBoost baselines
SURV_COMP_CONFIG=${SURV_COMP_CONFIG:-"configs/survival_training.yaml"}            # Step 6: Cross-family comparison
ANALYSIS_CONFIG=${ANALYSIS_CONFIG:-"configs/discriminative_training.yaml"}        # Steps 7 & 8: model/collator settings for interpretability + stratification

# ==============================================================================
# STEP EXECUTION TOGGLES (env-overridable, e.g.:
#   RUN_STEP_3_FINETUNING=true RUN_STEP_5_COMPARISON=false bash scripts/train_all.sh)
# ==============================================================================
RUN_STEP_1_MLM=${RUN_STEP_1_MLM:-false}                                           # Step 1: Pre-train t-EAV-Transformer
RUN_STEP_2_SURVIVAL_FT=${RUN_STEP_2_SURVIVAL_FT:-false}                           # Step 2: Survival (Cox) fine-tuning (runs before binary fine-tuning)
RUN_STEP_3_FINETUNING=${RUN_STEP_3_FINETUNING:-true}                              # Step 3: Fine-tune for infection prediction tasks (binary)
RUN_STEP_4_CLASSIC_ML=${RUN_STEP_4_CLASSIC_ML:-true}                              # Step 4: Classic ML baselines (LR, RF, XGBoost)
RUN_STEP_5_COMPARISON=${RUN_STEP_5_COMPARISON:-true}                              # Step 5: DCA, McNemar, PR-AUC
RUN_STEP_6_SURVIVAL_COMP=${RUN_STEP_6_SURVIVAL_COMP:-false}                       # Step 6: Cross-family comparison (classic ML vs binary vs survival)
RUN_STEP_7_INTERPRET=${RUN_STEP_7_INTERPRET:-true}                                # Step 7: Captum feature attribution
RUN_STEP_8_STRATIFICATION=${RUN_STEP_8_STRATIFICATION:-true}                      # Step 8: Patient stratification and Kaplan-Meier survival
RUN_SPLITS=(${RUN_SPLITS:-"random_split" "center_split" "temporal_split"})        # Which data splits to run (space-separated). Default: all three.

# ==============================================================================
# ANALYSIS MODEL SELECTION (Steps 5, 7, 8)
# The PROPOSED MODEL is the central comparator in Step 5 (comparison) and the
# model being analyzed in Steps 7 (Captum attribution) and 8 (stratification).
# The CONTENDER MODELS are the baselines compared against it in Step 5.
# Model keys (see MODEL_REGISTRY in scripts/analysis_comparison.py):
#   transformer, transformer_multitask, survival_transformer,
#   logistic_regression, random_forest, xgboost
# Note: Steps 7/8 load transformer checkpoints (Captum / embeddings), so they
# only support transformer-family proposed models and skip with a message for
# classic ML or unsupported models. The survival model is an optional bonus.
# ==============================================================================
ANALYSIS_PROPOSED_MODEL=${ANALYSIS_PROPOSED_MODEL:-"transformer_multitask"}       # Single-task analyses: "transformer"
# Baselines compared against the proposed model. Only LR is currently trained
# on the group tasks; add random_forest / xgboost after Step 4 runs with them:
ANALYSIS_CONTENDER_MODELS=${ANALYSIS_CONTENDER_MODELS:-"logistic_regression"}     # Full: "logistic_regression random_forest xgboost"
# Pre-training run id used to locate transformer-family result folders
# (must match run_id in configs / results layout)
ANALYSIS_RUN_ID=${ANALYSIS_RUN_ID:-"e00-a15-v60"}
# Tasks to analyze (space-separated), one report per task. Default: the 9
# prevalence-vetted tasks (bacteria + virus general classes and their
# actionable groups; fungi excluded as too rare — see prevalence annotations
# in configs/discriminative_training_multitask.yaml).
# "all_pathogens" = auto-read from the multi-task run's run_metadata.json
# (works once the multi-task run has trained at least one checkpoint).
# Empty = script defaults (comparison: bacteria+virus; steps 7/8: bacteria).
ANALYSIS_TASKS=${ANALYSIS_TASKS:-"infection_bacteria infection_virus infection_grp_enterobacterales infection_grp_gram_positive infection_grp_gn_nonfermenters infection_grp_clostridium infection_grp_herpes_viruses infection_grp_cmv infection_grp_bkpyv"}
# Per-analysis overrides (default: ANALYSIS_TASKS). Steps 7/8 are expensive
# per task -> curated to the two showcase groups (one bacterial, one viral):
COMPARISON_TASKS=${COMPARISON_TASKS:-"$ANALYSIS_TASKS"}
INTERPRET_TASKS=${INTERPRET_TASKS:-"infection_grp_enterobacterales infection_grp_cmv"}
STRATIFICATION_TASKS=${STRATIFICATION_TASKS:-"infection_grp_enterobacterales infection_grp_cmv"}
if [ "$ANALYSIS_TASKS" = "all_pathogens" ]; then
    ANALYSIS_TASKS=$(python - <<PY 2>/dev/null
import json
from pathlib import Path
for m in sorted(Path("$RESULTS_DIR").glob("*/**/finetuning/*/hrz(*)/run_metadata.json")):
    try:
        print(" ".join(json.loads(m.read_text()).get("tasks", [])))
        break
    except Exception:
        pass
PY
)
    echo "=== ANALYSIS_TASKS=all_pathogens -> resolved to: ${ANALYSIS_TASKS:-<none found, using defaults>} ==="
fi
# Step 5 specific options
COMPARISON_THRESHOLD_MODE=${COMPARISON_THRESHOLD_MODE:-"window_specific"}   # global | window_specific
COMPARISON_TARGET_RECALL=${COMPARISON_TARGET_RECALL:-80}
COMPARISON_WORKERS=${COMPARISON_WORKERS:-14}

# ==============================================================================
# STEP 3 TASK SELECTION (config file itself: FT_CONFIG in CONFIG FILES above)
# FT_TASKS: comma-separated prediction_tasks keys to run (default: all active
# entries of the config), e.g.:  FT_TASKS=multitask_bacteria bash scripts/train_all.sh
# ==============================================================================
FT_TASKS=${FT_TASKS:-""}

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

# Optional shared --tasks flags for the analysis scripts (empty -> defaults)
make_task_args() {
    local -n _arr=$1; local _list="$2"
    _arr=()
    if [ -n "$_list" ]; then read -r -a _arr <<< "--tasks $_list"; fi
}
make_task_args COMPARISON_TASK_ARGS "$COMPARISON_TASKS"
make_task_args INTERPRET_TASK_ARGS "$INTERPRET_TASKS"
make_task_args STRAT_TASK_ARGS "$STRATIFICATION_TASKS"

# run_logged <log_name> <command...> : run a command with stdout+stderr appended
# to ${LOG_DIR}/<log_name>.log instead of the terminal.
run_logged() {
    local name="$1"; shift
    local logfile="${LOG_DIR}/${name}.log"
    echo "      log: ${logfile}"
    echo "      peek: tail -f ${logfile}"
    "$@" >>"$logfile" 2>&1
}

# run_step <log_name> <header> <command...> : prints step header, logs quietly,
# reports success/failure (with log path) instead of dumping output.
run_step() {
    local name="$1"; shift
    local header="$1"; shift
    echo -e "\n>>> ${header}"
    if run_logged "$name" "$@"; then
        echo "    [OK] ${name} (log: ${LOG_DIR}/${name}.log)"
    else
        echo "    [FAILED] ${name} - see log for details: ${LOG_DIR}/${name}.log" >&2
        exit 1
    fi
}

# ==============================================================================
# ACTIVE PATH SELECTION AND DIRECTORY VERIFICATION
# ==============================================================================
echo "=== All python script output is redirected to log files ==="
echo "=== Logs directory: ${LOG_DIR} ==="
echo "=== Peek at a live log in another terminal, e.g.: ==="
echo "      tail -f ${LOG_DIR}/step1_mlm_temporal_split.log"
echo "      ls ${LOG_DIR}/*.log"
echo "      tail -n 50 ${LOG_DIR}/step1_mlm_temporal_split.log"
echo ""

if [ "$USE_SYNTHETIC" = true ]; then
    echo "=== Running pipeline in SYNTHETIC verification mode ==="
    DATA_ROOT="$SYNTHETIC_DATA_ROOT"
    
    TEAV_DATA_DIR="${DATA_ROOT}/teav"
    CLASSIC_DATA_DIR="${DATA_ROOT}/classic_ml"

    MLM_OVERRIDES='{"data_dir": "'"$TEAV_DATA_DIR"'", "pretrainer": {"max_steps": 5, "eval_steps": 5, "save_steps": 5}}'
    FT_OVERRIDES='{"data_dir": "'"$TEAV_DATA_DIR"'", "finetuner": {"max_steps": 5, "eval_steps": 5, "save_steps": 5}}'
    SURV_FT_OVERRIDES='{"data_dir": "'"$TEAV_DATA_DIR"'", "finetuner": {"max_steps": 5, "eval_steps": 5, "save_steps": 5}}'
    ML_OVERRIDES='{"data_dir": "'"$CLASSIC_DATA_DIR"'", "models": {"logistic_regression": {"n_optuna_trials": 2}, "random_forest": {"n_optuna_trials": 2}, "xgboost": {"n_optuna_trials": 2}}}'
    SURV_COMP_OVERRIDES='{"data_dir": "'"$TEAV_DATA_DIR"'", "n_bootstrap": 10}'
else
    echo "=== Running pipeline in REAL dataset mode ==="
    DATA_ROOT="$STCS_DATA_ROOT"
    
    TEAV_DATA_DIR="${DATA_ROOT}/teav"
    CLASSIC_DATA_DIR="${DATA_ROOT}/classic_ml"

    MLM_OVERRIDES="{\"data_dir\": \"$TEAV_DATA_DIR\"}"
    FT_OVERRIDES="{\"data_dir\": \"$TEAV_DATA_DIR\"}"
    SURV_FT_OVERRIDES="{\"data_dir\": \"$TEAV_DATA_DIR\"}"
    ML_OVERRIDES="{\"data_dir\": \"$CLASSIC_DATA_DIR\"}"
    SURV_COMP_OVERRIDES="{\"data_dir\": \"$TEAV_DATA_DIR\"}"
fi

# Verify root directory
if [ ! -d "$DATA_ROOT" ]; then
    echo "Error: Data root directory does not exist: $DATA_ROOT" >&2
    exit 1
fi

# ==============================================================================
# DEBUG OVERRIDES (applied on top of the mode-specific overrides above; skipped
# in synthetic mode, which is already capped to 5 steps). Warmup is shortened
# and early stopping disabled so that debug runs reliably execute exactly
# DEBUG_MAX_STEPS steps and always reach their final test/CSV-writing pass.
# ==============================================================================
if [ "$DEBUG_MODE" = true ] && [ "$USE_SYNTHETIC" != true ]; then
    DEBUG_EVAL_STEPS=$(( DEBUG_MAX_STEPS / 4 ))
    echo "=== DEBUG_MODE enabled: capping training to ${DEBUG_MAX_STEPS} steps (eval/save every ${DEBUG_EVAL_STEPS}) ==="
    DEBUG_MLM_OVERRIDES="{\"pretrainer\": {\"max_steps\": ${DEBUG_MAX_STEPS}, \"eval_steps\": ${DEBUG_EVAL_STEPS}, \"save_steps\": ${DEBUG_EVAL_STEPS}, \"warmup_steps\": 10, \"logging_steps\": 10, \"early_stopping_patience\": 100000}}"
    DEBUG_FT_OVERRIDES="{\"finetuner\": {\"max_steps\": ${DEBUG_MAX_STEPS}, \"eval_steps\": ${DEBUG_EVAL_STEPS}, \"save_steps\": ${DEBUG_EVAL_STEPS}, \"warmup_steps\": 10, \"logging_steps\": 10, \"early_stopping_patience\": 100000}}"
    DEBUG_SURV_FT_OVERRIDES="$DEBUG_FT_OVERRIDES"
    DEBUG_ML_OVERRIDES='{"models": {"logistic_regression": {"n_optuna_trials": 2}, "random_forest": {"n_optuna_trials": 2}, "xgboost": {"n_optuna_trials": 2}}}'
    DEBUG_SURV_COMP_OVERRIDES='{"n_bootstrap": 10}'
else
    DEBUG_MLM_OVERRIDES="{}"
    DEBUG_FT_OVERRIDES="{}"
    DEBUG_SURV_FT_OVERRIDES="{}"
    DEBUG_ML_OVERRIDES="{}"
    DEBUG_SURV_COMP_OVERRIDES="{}"
fi

# Verify subdirectories for active stages
if [ "$RUN_STEP_1_MLM" = true ] || [ "$RUN_STEP_2_SURVIVAL_FT" = true ] || [ "$RUN_STEP_3_FINETUNING" = true ] || [ "$RUN_STEP_6_SURVIVAL_COMP" = true ] || [ "$RUN_STEP_7_INTERPRET" = true ] || [ "$RUN_STEP_8_STRATIFICATION" = true ]; then
    if [ ! -d "$TEAV_DATA_DIR" ]; then
        echo "Error: TEAV data directory does not exist: $TEAV_DATA_DIR" >&2
        exit 1
    fi
fi

if [ "$RUN_STEP_4_CLASSIC_ML" = true ]; then
    if [ ! -d "$CLASSIC_DATA_DIR" ]; then
        echo "Error: Classic ML data directory does not exist: $CLASSIC_DATA_DIR" >&2
        exit 1
    fi
fi

export DATA_DIR="$TEAV_DATA_DIR"
export TEAV_DATA_DIR="$TEAV_DATA_DIR"

# Track background process PIDs
PIDS=()

# ==============================================================================
# PARALLEL BRANCH A: Transformer pipeline (MLM -> Survival FT -> Binary FT)
# Survival fine-tuning runs before binary fine-tuning (it reuses the MLM
# checkpoint from Step 1, so it can only start after pre-training is done).
# ==============================================================================
if [ "$RUN_STEP_1_MLM" = true ] || [ "$RUN_STEP_2_SURVIVAL_FT" = true ] || [ "$RUN_STEP_3_FINETUNING" = true ]; then
    (
        set -euo pipefail
        if [ "$RUN_STEP_1_MLM" = true ]; then
            echo -e "\n[Transformer-Branch] Starting Step 1: MLM Pre-training..."
            for split in "${RUN_SPLITS[@]}"; do
                run_step "step1_mlm_${split}" "[Transformer-Branch] Running MLM for split: $split" \
                    python scripts/train_mlm.py \
                    -c "$MLM_CONFIG" \
                    --overrides "$MLM_OVERRIDES" \
                    --overrides "{\"data_split_type\": \"$split\"}" \
                    --overrides "$DEBUG_MLM_OVERRIDES"
            done
        fi

        if [ "$RUN_STEP_2_SURVIVAL_FT" = true ]; then
            echo -e "\n[Transformer-Branch] Starting Step 2: Survival (Cox) fine-tuning..."
            for split in "${RUN_SPLITS[@]}"; do
                run_step "step2_survival_${split}" "[Transformer-Branch] Survival fine-tuning for split: $split" \
                    python scripts/train_survival.py \
                    -c "$SURVIVAL_FT_CONFIG" \
                    --overrides "$SURV_FT_OVERRIDES" \
                    --overrides "{\"data_split_type\": \"$split\"}" \
                    --overrides "$DEBUG_SURV_FT_OVERRIDES"
            done
        fi

        if [ "$RUN_STEP_3_FINETUNING" = true ]; then
            echo -e "\n[Transformer-Branch] Starting Step 3: Binary classification fine-tuning..."
            for split in "${RUN_SPLITS[@]}"; do
                run_step "step3_finetune_${split}" "[Transformer-Branch] Fine-tuning for split: $split" \
                    python scripts/train_classification.py \
                    -c "$FT_CONFIG" \
                    --tasks "$FT_TASKS" \
                    --overrides "$FT_OVERRIDES" \
                    --overrides "{\"data_split_type\": \"$split\"}" \
                    --overrides "$DEBUG_FT_OVERRIDES"
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
if [ "$RUN_STEP_4_CLASSIC_ML" = true ]; then
    (
        set -euo pipefail
        echo -e "\n[ClassicML-Branch] Starting Step 4: Classic ML Baselines..."
        for split in "${RUN_SPLITS[@]}"; do
            run_step "step4_classic_ml_${split}" "[ClassicML-Branch] Training classic ML for split: $split" \
                python scripts/train_classic_ml.py \
                -c "$CLASSIC_ML_CONFIG" \
                --overrides "$ML_OVERRIDES" \
                --overrides "{\"data_split_type\": \"$split\"}" \
                --overrides "$DEBUG_ML_OVERRIDES"
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
if [ "$RUN_STEP_5_COMPARISON" = true ]; then
    # shellcheck disable=SC2086  # intentional word splitting of the contender list
    run_step "step5_comparison" "[Step 5/8] Running Model Comparison Analysis (DCA, McNemar, PR-AUC)..." \
        python scripts/analysis_comparison.py \
        --results-dir "$RESULTS_DIR" \
        --run-id "$ANALYSIS_RUN_ID" \
        --proposed-model "$ANALYSIS_PROPOSED_MODEL" \
        --contender-models $ANALYSIS_CONTENDER_MODELS \
        --threshold_mode "$COMPARISON_THRESHOLD_MODE" \
        --target_recall "$COMPARISON_TARGET_RECALL" \
        --workers "$COMPARISON_WORKERS" \
        ${COMPARISON_TASK_ARGS[@]+"${COMPARISON_TASK_ARGS[@]}"}
else
    echo -e "\n>>> [Step 5/8] Skipped (RUN_STEP_5_COMPARISON=false)"
fi

# ==============================================================================
# ANALYSIS: Cross-family comparison (classic ML vs binary vs survival)
# ==============================================================================
if [ "$RUN_STEP_6_SURVIVAL_COMP" = true ]; then
    for split in "${RUN_SPLITS[@]}"; do
        run_step "step6_survival_comparison_${split}" "[Step 6/8] Cross-Family Comparison (classic ML vs binary vs survival) - $split" \
            python scripts/analysis_survival_comparison.py \
            -c "$SURV_COMP_CONFIG" \
            --overrides "$SURV_COMP_OVERRIDES" \
            --overrides "{\"data_split_type\": \"$split\"}" \
            --overrides "$DEBUG_SURV_COMP_OVERRIDES"
    done
else
    echo -e "\n>>> [Step 6/8] Skipped (RUN_STEP_6_SURVIVAL_COMP=false)"
fi

# ==============================================================================
# ANALYSIS: Feature attribution and clinical interpretability
# ==============================================================================
if [ "$RUN_STEP_7_INTERPRET" = true ]; then
    for split in "${RUN_SPLITS[@]}"; do
        run_step "step7_interpretability_${split}" "[Step 7/8] Captum Feature Attribution - $split" \
            python scripts/analysis_interpretability.py \
            --data-dir "$TEAV_DATA_DIR" \
            --config "$ANALYSIS_CONFIG" \
            --data-split-type "$split" \
            --results-dir "$RESULTS_DIR" \
            --proposed-model "$ANALYSIS_PROPOSED_MODEL" \
            --run-id "$ANALYSIS_RUN_ID" \
            ${INTERPRET_TASK_ARGS[@]+"${INTERPRET_TASK_ARGS[@]}"}
    done
else
    echo -e "\n>>> [Step 7/8] Skipped (RUN_STEP_7_INTERPRET=false)"
fi

# ==============================================================================
# ANALYSIS: Patient stratification and survival analysis
# ==============================================================================
if [ "$RUN_STEP_8_STRATIFICATION" = true ]; then
    for split in "${RUN_SPLITS[@]}"; do
        run_step "step8_stratification_${split}" "[Step 8/8] Stratification and Kaplan-Meier Survival Analysis - $split" \
            python scripts/analysis_stratification.py \
            --data-dir "$TEAV_DATA_DIR" \
            --config "$ANALYSIS_CONFIG" \
            --data-split-type "$split" \
            --results-dir "$RESULTS_DIR" \
            --proposed-model "$ANALYSIS_PROPOSED_MODEL" \
            --run-id "$ANALYSIS_RUN_ID" \
            ${STRAT_TASK_ARGS[@]+"${STRAT_TASK_ARGS[@]}"}
    done
else
    echo -e "\n>>> [Step 8/8] Skipped (RUN_STEP_8_STRATIFICATION=false)"
fi

echo -e "\n=== Pipeline run completed! ==="
echo "All logs for this run are in: ${LOG_DIR}"
echo "  ls ${LOG_DIR}/*.log"
echo "  tail -n 50 ${LOG_DIR}/<step>.log"