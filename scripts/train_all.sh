#!/usr/bin/env bash
# ==============================================================================
# End-to-End Slurm Pipeline Orchestrator for AIIDKIT (Baobab HPC)
#
# Submits parallel Slurm jobs with automatic `afterok` dependency chaining:
#   1a) Pre-training (MLM):      1 GPU job per split in parallel
#   1b) Classical ML (LR/RF/XGB): 1 CPU job per (split, task) in parallel (starts concurrently with 1a)
#   2)  Binary Fine-Tuning:      1 GPU job per (split, task) in parallel (both MT and non-MT; depends on 1a per split)
#   3)  Survival Fine-Tuning:    1 GPU job per (split, task) in parallel (default: false; depends on 1a per split)
#   4)  Downstream Analyses:     Comparison (CPU), Survival Comparison (GPU, default: false),
#                                Interpretability (GPU), and Stratification (GPU)
#
# Usage:
#   bash scripts/train_all.sh                     # submit full pipeline across all 3 splits
#   bash scripts/train_all.sh --dry-run           # preview all sbatch commands & dependencies without submitting
#   RUN_SPLITS="random_split" bash scripts/train_all.sh
#   RUN_STEP_1_MLM=false RUN_STEP_2_SURVIVAL_FT=true bash scripts/train_all.sh
# ==============================================================================
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SHARED_DATA_DIR="/home/shares/ds4dh/"

# Locate Apptainer image
if [ -f "${PROJECT_DIR}/research-env.sif" ]; then
    SOURCE_SIF="${PROJECT_DIR}/research-env.sif"
elif [ -f "/home/users/b/borneta/aiidkit_llm/research-env.sif" ]; then
    SOURCE_SIF="/home/users/b/borneta/aiidkit_llm/research-env.sif"
else
    SOURCE_SIF="${PROJECT_DIR}/research-env.sif"
fi

# ==============================================================================
# CLI FLAGS (optional overrides on top of environment variables)
# ==============================================================================
DRY_RUN=${DRY_RUN:-false}
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run|-n)
            DRY_RUN=true
            shift
            ;;
        --splits|-s)
            IFS=',' read -r -a RUN_SPLITS <<< "$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

# ==============================================================================
# DATA MODE AND ROOT DIRECTORIES
# ==============================================================================
USE_SYNTHETIC=${USE_SYNTHETIC:-false}
STCS_DATA_ROOT="/home/shares/ds4dh/aiidkit_project/data_new/processed/v3.9"
SYNTHETIC_DATA_ROOT="${PROJECT_DIR}/data/synthetic"

# ==============================================================================
# DEBUG MODE (smoke test: caps training steps for quick verification)
# ==============================================================================
DEBUG_MODE=${DEBUG_MODE:-false}
DEBUG_MAX_STEPS=${DEBUG_MAX_STEPS:-100}

# ==============================================================================
# RESULTS AND LOGGING
# ==============================================================================
RESULTS_DIR=${RESULTS_DIR:-"${PROJECT_DIR}/results"}
LOG_DIR=${LOG_DIR:-"${RESULTS_DIR}/logs/train_all_$(date +%Y%m%d_%H%M%S)"}
mkdir -p "$LOG_DIR"

# ==============================================================================
# SLURM HARDWARE & WALLTIME SETTINGS
# ==============================================================================
GPU_PARTITION=${GPU_PARTITION:-"shared-gpu"}
CPU_PARTITION=${CPU_PARTITION:-"shared-cpu"}
GPU_SPEC=${GPU_SPEC:-"nvidia_a100-pcie-40gb:1"}
GPU_CONSTRAINT=${GPU_CONSTRAINT:-""}
SLURM_CPUS_GPU=${SLURM_CPUS_GPU:-${SLURM_CPUS:-8}}
SLURM_CPUS_CPU=${SLURM_CPUS_CPU:-${SLURM_CPUS:-16}}
SLURM_MEM_GPU=${SLURM_MEM_GPU:-"32G"}
SLURM_MEM_CPU=${SLURM_MEM_CPU:-"64G"}

# Reasonable walltimes per stage (based on empirical Baobab runtimes + safety margin; shared-gpu max is 12h)
TIME_PREWARM=${TIME_PREWARM:-"00-02:00:00"}       # CPU dataset cache prewarming (~3-8m cold, ~2s warm on 16 CPU cores)
TIME_OPTUNA_FT=${TIME_OPTUNA_FT:-"00-11:30:00"}   # Optuna fine-tune worker (~40-90m/trial on RTX 3090 / A100)
TIME_MLM=${TIME_MLM:-"00-06:00:00"}               # Pre-training (~1-2h per split)
TIME_CLASSIC_ML=${TIME_CLASSIC_ML:-"00-06:00:00"} # Classic ML (~12-90m per task on 16 CPU cores)
TIME_FINETUNE=${TIME_FINETUNE:-"00-06:00:00"}     # Binary fine-tuning (~1.5-2.5h for multitask_all on RTX 3090)
TIME_SURVIVAL_FT=${TIME_SURVIVAL_FT:-"00-04:00:00"} # Survival fine-tuning (~20-45m per task on RTX 3090)
TIME_COMPARISON=${TIME_COMPARISON:-"00-06:00:00"} # Model comparison (~45-90m on 16 CPU cores)
TIME_SURV_COMP=${TIME_SURV_COMP:-"00-03:00:00"}   # Survival comparison (~30-60m on GPU)
TIME_INTERPRET=${TIME_INTERPRET:-"00-06:00:00"}   # Captum attribution (~60-130m per task on RTX 3090)
TIME_STRAT=${TIME_STRAT:-"00-02:00:00"}           # UMAP + HDBSCAN stratification (~3-5m per split on RTX 3090)

# ==============================================================================
# MODEL FAMILY & CONFIG FILES PER STEP (all env-overridable)
# Set MODEL_FAMILY=tfg (or generative) to switch the pipeline to the generative
# causal decoder-only Transformer (`configs/generative_training.yaml`, `tfg_*`).
# ==============================================================================
MODEL_FAMILY=${MODEL_FAMILY:-"tf"}
if [[ "$MODEL_FAMILY" == "tfg" || "$MODEL_FAMILY" == "generative" ]]; then
    DEFAULT_TF_CONFIG="configs/generative_training.yaml"
    DEFAULT_MODEL_PREFIX="tfg"
    DEFAULT_FAMILY_DIR="generative_transformer"
else
    DEFAULT_TF_CONFIG="configs/discriminative_training.yaml"
    DEFAULT_MODEL_PREFIX="tf"
    DEFAULT_FAMILY_DIR="transformer"
fi

MLM_CONFIG=${MLM_CONFIG:-"$DEFAULT_TF_CONFIG"}                                    # Step 1a: Pre-training (MLM or Causal Next-Event)
CLASSIC_ML_CONFIG=${CLASSIC_ML_CONFIG:-"configs/discriminative_classic_ml.yaml"}  # Step 1b: LR / RF / XGBoost baselines
FT_CONFIG=${FT_CONFIG:-"$DEFAULT_TF_CONFIG"}                                      # Step 2: Binary fine-tuning (MT + non-MT)
SURVIVAL_FT_CONFIG=${SURVIVAL_FT_CONFIG:-"configs/survival_training.yaml"}        # Step 3: Survival (Cox) fine-tuning
SURV_COMP_CONFIG=${SURV_COMP_CONFIG:-"configs/survival_training.yaml"}            # Step 6: Cross-family survival comparison
ANALYSIS_CONFIG=${ANALYSIS_CONFIG:-"$DEFAULT_TF_CONFIG"}                          # Steps 7 & 8: Interpretability + Stratification

# Optional Optuna fine-tuning HPO JSON & canonical run-id override for final multi-split training
OPTUNA_FINETUNE_JSON=${OPTUNA_FINETUNE_JSON:-""}
OPTUNA_PRETRAINED_DIR=${OPTUNA_PRETRAINED_DIR:-""}
RUN_ID_OVERRIDE=${RUN_ID_OVERRIDE:-""}
TRAIN_DATA_AUGMENT=${TRAIN_DATA_AUGMENT:-""}
EXTRA_MLM_OVERRIDES=${EXTRA_MLM_OVERRIDES:-"{}"}
EXTRA_FT_OVERRIDES=${EXTRA_FT_OVERRIDES:-"{}"}

# Optuna fine-tuning HPO settings (disabled by default; enable with RUN_STEP_0_OPTUNA_FINETUNE=true)
# Pre-training uses fixed canonical config (e00-a15-v60_raw); Optuna optimizes fine-tuning only.
RUN_STEP_0_OPTUNA_FINETUNE=${RUN_STEP_0_OPTUNA_FINETUNE:-false}
OPTUNA_SPLIT=${OPTUNA_SPLIT:-"random_split"}
OPTUNA_WORKERS=${OPTUNA_WORKERS:-12}
OPTUNA_FT_TOTAL_TRIALS=${OPTUNA_FT_TOTAL_TRIALS:-50}
DEFAULT_FT_WORKER_CAP=$(( (OPTUNA_FT_TOTAL_TRIALS + OPTUNA_WORKERS - 1) / OPTUNA_WORKERS + 2 ))
OPTUNA_FT_TRIALS_PER_WORKER=${OPTUNA_FT_TRIALS_PER_WORKER:-$DEFAULT_FT_WORKER_CAP}
OPTUNA_REAP_RUNNING=${OPTUNA_REAP_RUNNING:-false}
OPTUNA_TASKS=${OPTUNA_TASKS:-"multitask_bacteria"}
OPTUNA_EXPORT_RUN_ID=${OPTUNA_EXPORT_RUN_ID:-"optuna_ft"}

# ==============================================================================
# STEP EXECUTION TOGGLES (env-overridable)
# ==============================================================================
RUN_STEP_0_PREWARM=${RUN_STEP_0_PREWARM:-true}                                    # Step 0: Prewarm t-EAV processed caches on CPU nodes before GPU jobs
RUN_STEP_1_MLM=${RUN_STEP_1_MLM:-true}                                            # Step 1a: Pre-train t-EAV-Transformer (parallel across splits)
RUN_STEP_4_CLASSIC_ML=${RUN_STEP_4_CLASSIC_ML:-true}                              # Step 1b: Classic ML baselines (LR, RF, XGBoost; parallel across splits & tasks)
RUN_STEP_3_FINETUNING=${RUN_STEP_3_FINETUNING:-true}                              # Step 2: Binary fine-tuning (MT and non-MT; parallel across splits & tasks)
RUN_STEP_2_SURVIVAL_FT=${RUN_STEP_2_SURVIVAL_FT:-false}                           # Step 3: Survival (Cox) fine-tuning (default: false)
RUN_STEP_5_COMPARISON=${RUN_STEP_5_COMPARISON:-true}                              # Step 5: Head-to-head comparison (TFMT vs LR, RF, XGBoost)
RUN_STEP_6_SURVIVAL_COMP=${RUN_STEP_6_SURVIVAL_COMP:-false}                       # Step 6: Cross-family survival comparison (default: false)
RUN_STEP_7_INTERPRET=${RUN_STEP_7_INTERPRET:-true}                                # Step 7: Captum feature attribution
RUN_STEP_8_STRATIFICATION=${RUN_STEP_8_STRATIFICATION:-true}                      # Step 8: Patient stratification and Kaplan-Meier survival

if [ -z "${RUN_SPLITS+x}" ]; then
    RUN_SPLITS=("random_split" "temporal_split" "center_split")
elif [[ "$(declare -p RUN_SPLITS 2>/dev/null)" != "declare -a"* ]]; then
    read -r -a RUN_SPLITS <<< "${RUN_SPLITS//,/ }"
fi

# ==============================================================================
# TASK LISTS FOR PARALLEL SLURM SUBMISSION
# ==============================================================================
# All 13 well-powered canonical tasks (5 bacterial + 4 viral + 2 fungal + 2 long-term)
DEFAULT_SINGLE_TASKS=(
    "infection_bacteria"
    "infection_grp_enterobacterales"
    "infection_grp_gram_positive"
    "infection_grp_gn_nonfermenters"
    "infection_grp_clostridium"
    "infection_virus"
    "infection_grp_herpes_viruses"
    "infection_grp_cmv"
    "infection_grp_bkpyv"
    "infection_fungi"
    "infection_grp_candida"
    "death"
    "graft_loss"
)

# Category-grouped multi-task entries (B_CAT_MT: 5 fine-tuning runs per split)
DEFAULT_MULTITASK_TASKS=(
    "multitask_bacteria"
    "multitask_viruses"
    "multitask_fungi"
    "death"
    "graft_loss"
)

# Step 1b (Classic ML) tasks: comma- or space-separated override via CLASSIC_ML_TASKS
if [ -n "${CLASSIC_ML_TASKS:-}" ]; then
    read -r -a CML_TASK_LIST <<< "${CLASSIC_ML_TASKS//,/ }"
else
    CML_TASK_LIST=("${DEFAULT_SINGLE_TASKS[@]}")
fi
CLASSIC_ML_MODELS=${CLASSIC_ML_MODELS:-"logistic_regression,random_forest,xgboost"}
CLASSIC_ML_FIXED_HP=${CLASSIC_ML_FIXED_HP:-false}

# Step 2 (Binary Fine-Tuning) tasks: defaults to unified multi-task (override FT_TASKS for domain/single-task sweeps)
if [ -n "${FT_TASKS:-}" ]; then
    read -r -a FT_TASK_LIST <<< "${FT_TASKS//,/ }"
else
    FT_TASK_LIST=("${DEFAULT_MULTITASK_TASKS[@]}")
fi

# Step 3 (Survival Fine-Tuning) tasks (when RUN_STEP_2_SURVIVAL_FT=true)
if [ -n "${SURVIVAL_TASKS:-}" ]; then
    read -r -a SURV_TASK_LIST <<< "${SURVIVAL_TASKS//,/ }"
else
    SURV_TASK_LIST=(
        "infection_bacteria"
        "grp_enterobacterales"
        "grp_gram_positive"
        "grp_gn_nonfermenters"
        "grp_clostridium"
        "infection_virus"
        "grp_herpes_viruses"
        "grp_cmv"
        "grp_bkpyv"
        "infection_fungi"
        "grp_candida"
        "death"
        "graft_loss"
    )
fi

# ==============================================================================
# ANALYSIS MODEL SELECTION (Steps 5, 7, 8)
# Supported transformer variants (no un-suffixed aliases):
#   - tf_raw_mt  (TFRM):  discriminative multi-task + raw continuous value embeddings
#   - tf_raw_st  (TFRS):  discriminative single-task + raw continuous value embeddings
#   - tf_bin_mt  (TFBM):  discriminative multi-task + 7-level binned value embeddings
#   - tf_bin_st  (TFBS):  discriminative single-task + 7-level binned value embeddings
#   - tfg_raw_mt (TFGRM): generative causal multi-task + raw continuous value embeddings
#   - tfg_raw_st (TFGRS): generative causal single-task + raw continuous value embeddings
#   - tfg_bin_mt (TFGBM): generative causal multi-task + 7-level binned value embeddings
#   - tfg_bin_st (TFGBS): generative causal single-task + 7-level binned value embeddings
# Default proposed model is `${DEFAULT_MODEL_PREFIX}_{raw|bin}_mt` compared against
# classical ML baselines (`logistic_regression random_forest xgboost`).
# ==============================================================================
USE_NUMERICAL_EMBEDDING=${USE_NUMERICAL_EMBEDDING:-true}
if [ "$USE_NUMERICAL_EMBEDDING" = "false" ]; then
    DEFAULT_VAL_TAG="bin"
else
    DEFAULT_VAL_TAG="raw"
fi
ANALYSIS_PROPOSED_MODEL=${ANALYSIS_PROPOSED_MODEL:-"${DEFAULT_MODEL_PREFIX}_${DEFAULT_VAL_TAG}_mt"}
ANALYSIS_CONTENDER_MODELS=${ANALYSIS_CONTENDER_MODELS:-"logistic_regression random_forest xgboost"}
if [ -n "${RUN_ID_OVERRIDE}" ]; then
    DEFAULT_ANALYSIS_RUN_ID="${RUN_ID_OVERRIDE}_${DEFAULT_VAL_TAG}"
elif [ "$RUN_STEP_0_OPTUNA_FINETUNE" = true ] || [ -n "${OPTUNA_FINETUNE_JSON:-}" ]; then
    DEFAULT_ANALYSIS_RUN_ID="${OPTUNA_EXPORT_RUN_ID}_${DEFAULT_VAL_TAG}"
else
    DEFAULT_ANALYSIS_RUN_ID="e00-a15-v60_${DEFAULT_VAL_TAG}"
fi
ANALYSIS_RUN_ID=${ANALYSIS_RUN_ID:-"${DEFAULT_ANALYSIS_RUN_ID}"}

ANALYSIS_TASKS=${ANALYSIS_TASKS:-"infection_bacteria infection_grp_enterobacterales infection_grp_gram_positive infection_grp_gn_nonfermenters infection_grp_clostridium infection_virus infection_grp_herpes_viruses infection_grp_cmv infection_grp_bkpyv infection_fungi infection_grp_candida death graft_loss"}
COMPARISON_TASKS=${COMPARISON_TASKS:-"$ANALYSIS_TASKS"}
COMPARISON_HORIZONS=${COMPARISON_HORIZONS:-""}
INTERPRET_TASKS=${INTERPRET_TASKS:-"infection_bacteria"}
STRATIFICATION_TASKS=${STRATIFICATION_TASKS:-"$ANALYSIS_TASKS"}
STRATIFICATION_PROPOSED_MODEL=${STRATIFICATION_PROPOSED_MODEL:-"$ANALYSIS_PROPOSED_MODEL"}
STRATIFICATION_HORIZON=${STRATIFICATION_HORIZON:-90}

COMPARISON_THRESHOLD_MODE=${COMPARISON_THRESHOLD_MODE:-"window_specific"}
COMPARISON_TARGET_RECALL=${COMPARISON_TARGET_RECALL:-80}
COMPARISON_WORKERS=${COMPARISON_WORKERS:-14}

# ==============================================================================
# ACTIVE PATH SELECTION AND OVERRIDES
# ==============================================================================
if [ "$USE_SYNTHETIC" = true ]; then
    DATA_ROOT="$SYNTHETIC_DATA_ROOT"
    TEAV_DATA_DIR="${DATA_ROOT}/teav"
    CLASSIC_DATA_DIR="${DATA_ROOT}/classic_ml"

    MLM_OVERRIDES="{\"data_dir\": \"$TEAV_DATA_DIR\", \"model\": {\"embedding_layer_config\": {\"use_numerical_embedding\": ${USE_NUMERICAL_EMBEDDING}}}, \"data_collator\": {\"use_numerical_embedding\": ${USE_NUMERICAL_EMBEDDING}}, \"pretrainer\": {\"max_steps\": 5, \"eval_steps\": 5, \"save_steps\": 5}}"
    FT_OVERRIDES="{\"data_dir\": \"$TEAV_DATA_DIR\", \"model\": {\"embedding_layer_config\": {\"use_numerical_embedding\": ${USE_NUMERICAL_EMBEDDING}}}, \"data_collator\": {\"use_numerical_embedding\": ${USE_NUMERICAL_EMBEDDING}}, \"finetuner\": {\"max_steps\": 5, \"eval_steps\": 5, \"save_steps\": 5}}"
    SURV_FT_OVERRIDES="{\"data_dir\": \"$TEAV_DATA_DIR\", \"model\": {\"embedding_layer_config\": {\"use_numerical_embedding\": ${USE_NUMERICAL_EMBEDDING}}}, \"data_collator\": {\"use_numerical_embedding\": ${USE_NUMERICAL_EMBEDDING}}, \"finetuner\": {\"max_steps\": 5, \"eval_steps\": 5, \"save_steps\": 5}}"
    ML_OVERRIDES="{\"data_dir\": \"$CLASSIC_DATA_DIR\", \"models\": {\"logistic_regression\": {\"n_optuna_trials\": 2}, \"random_forest\": {\"n_optuna_trials\": 2}, \"xgboost\": {\"n_optuna_trials\": 2}}}"
    SURV_COMP_OVERRIDES="{\"data_dir\": \"$TEAV_DATA_DIR\", \"n_bootstrap\": 10}"
else
    DATA_ROOT="$STCS_DATA_ROOT"
    TEAV_DATA_DIR="${DATA_ROOT}/teav"
    CLASSIC_DATA_DIR="${DATA_ROOT}/classic_ml"

    MLM_OVERRIDES="{\"data_dir\": \"$TEAV_DATA_DIR\", \"model\": {\"embedding_layer_config\": {\"use_numerical_embedding\": ${USE_NUMERICAL_EMBEDDING}}}, \"data_collator\": {\"use_numerical_embedding\": ${USE_NUMERICAL_EMBEDDING}}}"
    FT_OVERRIDES="{\"data_dir\": \"$TEAV_DATA_DIR\", \"model\": {\"embedding_layer_config\": {\"use_numerical_embedding\": ${USE_NUMERICAL_EMBEDDING}}}, \"data_collator\": {\"use_numerical_embedding\": ${USE_NUMERICAL_EMBEDDING}}}"
    SURV_FT_OVERRIDES="{\"data_dir\": \"$TEAV_DATA_DIR\", \"model\": {\"embedding_layer_config\": {\"use_numerical_embedding\": ${USE_NUMERICAL_EMBEDDING}}}, \"data_collator\": {\"use_numerical_embedding\": ${USE_NUMERICAL_EMBEDDING}}}"
    ML_OVERRIDES="{\"data_dir\": \"$CLASSIC_DATA_DIR\"}"
    SURV_COMP_OVERRIDES="{\"data_dir\": \"$TEAV_DATA_DIR\"}"
fi

if [ "$DEBUG_MODE" = true ] && [ "$USE_SYNTHETIC" != true ]; then
    DEBUG_EVAL_STEPS=$(( DEBUG_MAX_STEPS / 4 ))
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

if [ "$DRY_RUN" != true ]; then
    if [ ! -d "$DATA_ROOT" ]; then
        echo "Error: Data root directory does not exist: $DATA_ROOT" >&2
        exit 1
    fi
    if [ ! -f "$SOURCE_SIF" ]; then
        echo "Error: Apptainer image not found at $SOURCE_SIF" >&2
        exit 1
    fi
fi

# ==============================================================================
# SLURM WORKER TEMPLATE (generated inside LOG_DIR)
# Copies research-env.sif to node-local /scratch/${USER}/${SLURM_JOB_ID} and
# executes the requested step inside the container.
# ==============================================================================
WORKER_SCRIPT="${LOG_DIR}/_slurm_pipeline_worker.sbatch"
cat << 'EOF' > "$WORKER_SCRIPT"
#!/bin/bash
#SBATCH --open-mode=append

set -euo pipefail

SCRATCH_DIR="/scratch/${USER}/${SLURM_JOB_ID}"
TARGET_SIF="${SCRATCH_DIR}/research-env.sif"
mkdir -p -m 700 "/scratch/${USER}" "${SCRATCH_DIR}" "${SCRATCH_DIR}/torch"
chmod 700 "/scratch/${USER}" "${SCRATCH_DIR}" 2>/dev/null || true

cleanup() {
    rm -rf "$SCRATCH_DIR" 2>/dev/null || true
}
trap cleanup EXIT SIGTERM SIGINT

echo "============================================================"
echo "[$(date -Is)] Slurm Job ${SLURM_JOB_ID} (${SLURM_JOB_NAME}) on $(hostname)"
echo "Stage: ${JOB_STAGE} | Split: ${JOB_SPLIT:-all} | Task: ${JOB_TASK:-all}"
echo "Copying SIF (${SOURCE_SIF}) to node-local scratch (${TARGET_SIF})..."
echo "============================================================"
cp "$SOURCE_SIF" "$TARGET_SIF"

PERSISTENT_CACHE_DIR="$HOME/.cache"
mkdir -p "$PERSISTENT_CACHE_DIR/huggingface" \
         "$PERSISTENT_CACHE_DIR/triton" \
         "$PERSISTENT_CACHE_DIR/matplotlib"

export APPTAINERENV_VIRTUAL_ENV="/opt/venv"
export APPTAINERENV_PATH="/opt/venv/bin:$PATH"
export APPTAINERENV_XDG_CACHE_HOME="$PERSISTENT_CACHE_DIR"
export APPTAINERENV_HF_HOME="$PERSISTENT_CACHE_DIR/huggingface"
export APPTAINERENV_TRITON_CACHE_DIR="$PERSISTENT_CACHE_DIR/triton"
export APPTAINERENV_TORCH_HOME="${SCRATCH_DIR}/torch"
export APPTAINERENV_MPLCONFIGDIR="$PERSISTENT_CACHE_DIR/matplotlib"
export APPTAINERENV_REQUESTS_CA_BUNDLE="/mnt/host_cert.crt"
export APPTAINERENV_SSL_CERT_FILE="/mnt/host_cert.crt"
export APPTAINERENV_PYTHONUNBUFFERED="1"
export APPTAINERENV_PYTORCH_ALLOC_CONF="expandable_segments:True"
export APPTAINERENV_AIIDKIT_LOG_FILE="${JOB_LOG_FILE}"
export APPTAINERENV_AIIDKIT_SCRATCH_DIR="/tmp"
export APPTAINERENV_HF_DATASETS_CACHE="/tmp/hf_datasets_cache"

cd "$PROJECT_DIR"
START_TS=$(date +%s)

BIND_MOUNTS="$HOME:$HOME,${SCRATCH_DIR}:/tmp,${SHARED_DATA_DIR}:${SHARED_DATA_DIR},/etc/pki/tls/certs/ca-bundle.crt:/mnt/host_cert.crt:ro"
if [ -d "/srv/beegfs/scratch" ]; then
    BIND_MOUNTS="${BIND_MOUNTS},/srv/beegfs/scratch:/srv/beegfs/scratch"
fi

case "${JOB_STAGE}" in
    prewarm)
        apptainer exec --bind "$BIND_MOUNTS" "$TARGET_SIF" \
            /opt/venv/bin/python scripts/prewarm_caches.py \
                -c "${FT_CONFIG}" \
                --data-dir "${TEAV_DATA_DIR}" \
                --splits "${JOB_SPLIT}"
        ;;
    optuna_ft_worker)
        OPTUNA_TASKS_COMMA="${OPTUNA_TASKS_COLON//:/,}"
        EXTRA_OPT_FLAGS=()
        [ "${DEBUG_MODE}" = "true" ] && EXTRA_OPT_FLAGS+=(--debug)
        [ -n "${OPTUNA_PRETRAINED_DIR:-}" ] && EXTRA_OPT_FLAGS+=(--pretrained-dir "${OPTUNA_PRETRAINED_DIR}")
        apptainer exec --nv --bind "$BIND_MOUNTS" "$TARGET_SIF" \
            /opt/venv/bin/python scripts/train_optuna.py \
                --stage finetune \
                -c "${FT_CONFIG}" \
                --split "${JOB_SPLIT}" \
                --tasks "${OPTUNA_TASKS_COMMA}" \
                --target-trials "${OPTUNA_FT_TOTAL_TRIALS}" \
                --n-trials "${OPTUNA_FT_TRIALS_PER_WORKER}" \
                --export-run-id "${OPTUNA_EXPORT_RUN_ID}" \
                --skip-final-eval \
                --overrides "${FT_OVERRIDES}" \
                --overrides "${EXTRA_FT_OVERRIDES}" \
                "${EXTRA_OPT_FLAGS[@]}"
        ;;
    optuna_ft_finalize)
        OPTUNA_TASKS_COMMA="${OPTUNA_TASKS_COLON//:/,}"
        EXTRA_FIN_FLAGS=(--compact-journal --reap-running)
        [ -n "${OPTUNA_PRETRAINED_DIR:-}" ] && EXTRA_FIN_FLAGS+=(--pretrained-dir "${OPTUNA_PRETRAINED_DIR}")
        apptainer exec --nv --bind "$BIND_MOUNTS" "$TARGET_SIF" \
            /opt/venv/bin/python scripts/train_optuna.py \
                --stage finetune \
                -c "${FT_CONFIG}" \
                --split "${JOB_SPLIT}" \
                --tasks "${OPTUNA_TASKS_COMMA}" \
                --n-trials 0 \
                --export-run-id "${OPTUNA_EXPORT_RUN_ID}" \
                --overrides "${FT_OVERRIDES}" \
                --overrides "${EXTRA_FT_OVERRIDES}" \
                "${EXTRA_FIN_FLAGS[@]}"
        ;;
    mlm)
        EXTRA_MLM_ARGS=()
        [ -n "${RUN_ID_OVERRIDE:-}" ] && EXTRA_MLM_ARGS+=(--overrides "{\"run_id\": \"${RUN_ID_OVERRIDE}\"}")
        apptainer exec --nv --bind "$BIND_MOUNTS" "$TARGET_SIF" \
            /opt/venv/bin/python scripts/train_mlm.py \
                -c "${MLM_CONFIG}" \
                -s \
                "${EXTRA_MLM_ARGS[@]}" \
                --overrides "${MLM_OVERRIDES}" \
                --overrides "${EXTRA_MLM_OVERRIDES}" \
                --overrides "{\"data_split_type\": \"${JOB_SPLIT}\"}" \
                --overrides "${DEBUG_MLM_OVERRIDES}"
        ;;
    classic_ml)
        MODELS_COMMA="${CLASSIC_ML_MODELS_COLON//:/,}"
        EXTRA_CML_ARGS=()
        [ "${CLASSIC_ML_FIXED_HP:-false}" = "true" ] && EXTRA_CML_ARGS+=(--fixed_hp)
        apptainer exec --bind "$BIND_MOUNTS" "$TARGET_SIF" \
            /opt/venv/bin/python scripts/train_classic_ml.py \
                -c "${CLASSIC_ML_CONFIG}" \
                --tasks "${JOB_TASK}" \
                --models "${MODELS_COMMA}" \
                "${EXTRA_CML_ARGS[@]}" \
                --overrides "${ML_OVERRIDES}" \
                --overrides "{\"data_split_type\": \"${JOB_SPLIT}\"}" \
                --overrides "${DEBUG_ML_OVERRIDES}"
        ;;
    finetune)
        EXTRA_FT_ARGS=()
        [ -n "${OPTUNA_FINETUNE_JSON:-}" ] && EXTRA_FT_ARGS+=(--optuna-json "${OPTUNA_FINETUNE_JSON}")
        [ -n "${RUN_ID_OVERRIDE:-}" ] && EXTRA_FT_ARGS+=(--overrides "{\"run_id\": \"${RUN_ID_OVERRIDE}\"}")
        [ -n "${TRAIN_DATA_AUGMENT:-}" ] && EXTRA_FT_ARGS+=(--overrides "{\"train_data_augment\": \"${TRAIN_DATA_AUGMENT}\"}")
        apptainer exec --nv --bind "$BIND_MOUNTS" "$TARGET_SIF" \
            /opt/venv/bin/python scripts/train_classification.py \
                -c "${FT_CONFIG}" \
                -s \
                --tasks "${JOB_TASK}" \
                "${EXTRA_FT_ARGS[@]}" \
                --overrides "${FT_OVERRIDES}" \
                --overrides "${EXTRA_FT_OVERRIDES}" \
                --overrides "{\"data_split_type\": \"${JOB_SPLIT}\"}" \
                --overrides "${DEBUG_FT_OVERRIDES}"
        ;;
    survival_ft)
        apptainer exec --nv --bind "$BIND_MOUNTS" "$TARGET_SIF" \
            /opt/venv/bin/python scripts/train_survival.py \
                -c "${SURVIVAL_FT_CONFIG}" \
                -s \
                --tasks "${JOB_TASK}" \
                --overrides "${SURV_FT_OVERRIDES}" \
                --overrides "{\"data_split_type\": \"${JOB_SPLIT}\"}" \
                --overrides "${DEBUG_SURV_FT_OVERRIDES}"
        ;;
    comparison)
        IFS=':' read -r -a CONTENDER_ARR <<< "${CONTENDERS_COLON}"
        IFS=':' read -r -a SPLIT_ARR <<< "${SPLITS_COLON}"
        if [ -n "${JOB_TASK:-}" ]; then
            COMP_TASK_ARR=("${JOB_TASK}")
        else
            IFS=':' read -r -a COMP_TASK_ARR <<< "${COMP_TASKS_COLON}"
        fi
        EXTRA_COMP_ARGS=()
        if [ -n "${COMP_HORIZONS_COLON:-}" ]; then
            IFS=':' read -r -a COMP_HRZ_ARR <<< "${COMP_HORIZONS_COLON}"
            EXTRA_COMP_ARGS+=(--horizons "${COMP_HRZ_ARR[@]}")
        fi
        apptainer exec --bind "$BIND_MOUNTS" "$TARGET_SIF" \
            /opt/venv/bin/python scripts/analysis_comparison.py \
                --results-dir "${RESULTS_DIR}" \
                --run-id "${ANALYSIS_RUN_ID}" \
                --proposed-model "${ANALYSIS_PROPOSED_MODEL}" \
                --contender-models "${CONTENDER_ARR[@]}" \
                --splits "${SPLIT_ARR[@]}" \
                --threshold_mode "${COMPARISON_THRESHOLD_MODE}" \
                --target_recall "${COMPARISON_TARGET_RECALL}" \
                --workers "${COMPARISON_WORKERS}" \
                --tasks "${COMP_TASK_ARR[@]}" \
                "${EXTRA_COMP_ARGS[@]}"
        if [ -z "${JOB_TASK:-}" ]; then
            apptainer exec --bind "$BIND_MOUNTS" "$TARGET_SIF" \
                /opt/venv/bin/python scripts/generate_summary_report.py \
                    --results-dir "${RESULTS_DIR}" \
                    --splits "${SPLIT_ARR[@]}"
        fi
        ;;
    summary_report)
        IFS=':' read -r -a SPLIT_ARR <<< "${SPLITS_COLON}"
        apptainer exec --bind "$BIND_MOUNTS" "$TARGET_SIF" \
            /opt/venv/bin/python scripts/generate_summary_report.py \
                --results-dir "${RESULTS_DIR}" \
                --run-id "${ANALYSIS_RUN_ID}" \
                --splits "${SPLIT_ARR[@]}"
        ;;
    survival_comp)
        apptainer exec --bind "$BIND_MOUNTS" "$TARGET_SIF" \
            /opt/venv/bin/python scripts/analysis_survival_comparison.py \
                -c "${SURV_COMP_CONFIG}" \
                --overrides "${SURV_COMP_OVERRIDES}" \
                --overrides "{\"data_split_type\": \"${JOB_SPLIT}\"}" \
                --overrides "${DEBUG_SURV_COMP_OVERRIDES}"
        ;;
    interpretability)
        apptainer exec --nv --bind "$BIND_MOUNTS" "$TARGET_SIF" \
            /opt/venv/bin/python scripts/analysis_interpretability.py \
                --data-dir "${TEAV_DATA_DIR}" \
                --config "${ANALYSIS_CONFIG}" \
                --data-split-type "${JOB_SPLIT}" \
                --results-dir "${RESULTS_DIR}" \
                --proposed-model "${ANALYSIS_PROPOSED_MODEL}" \
                --run-id "${ANALYSIS_RUN_ID}" \
                --tasks "${JOB_TASK}"
        ;;
    stratification)
        IFS=':' read -r -a STRAT_TASK_ARR <<< "${STRAT_TASKS_COLON}"
        apptainer exec --nv --bind "$BIND_MOUNTS" "$TARGET_SIF" \
            /opt/venv/bin/python scripts/analysis_stratification.py \
                --data-dir "${TEAV_DATA_DIR}" \
                --config "${ANALYSIS_CONFIG}" \
                --data-split-type "${JOB_SPLIT}" \
                --results-dir "${RESULTS_DIR}" \
                --proposed-model "${STRATIFICATION_PROPOSED_MODEL}" \
                --run-id "${ANALYSIS_RUN_ID}" \
                --horizon "${STRATIFICATION_HORIZON}" \
                --tasks "${STRAT_TASK_ARR[@]}"
        ;;
    *)
        echo "Error: Unknown JOB_STAGE='${JOB_STAGE}'" >&2
        exit 1
        ;;
esac

END_TS=$(date +%s)
echo "============================================================"
echo "[$(date -Is)] Stage ${JOB_STAGE} (${JOB_SPLIT:-all} / ${JOB_TASK:-all}) COMPLETED in $((END_TS - START_TS))s"
echo "============================================================"
EOF
chmod +x "$WORKER_SCRIPT"

# ==============================================================================
# HELPER: Submit a Slurm job and return its numeric Job ID
# ==============================================================================
FAKE_JOB_COUNTER=900000
ALL_SUBMITTED_JOBS=()

short_split() {
    case "$1" in
        random_split)   echo "rnd" ;;
        temporal_split) echo "tmp" ;;
        center_split)   echo "ctr" ;;
        *)              echo "${1%_split}" ;;
    esac
}

short_task() {
    local t="${1#infection_}"
    t="${t#grp_}"
    echo "$t"
}

join_by_colon() {
    local IFS=':'
    echo "$*"
}

submit_slurm_job() {
    local job_name="$1"
    local partition="$2"
    local walltime="$3"
    local mem="$4"
    local use_gpu="$5"        # "true" or "false"
    local dependency="$6"     # "" or "afterok:123:456"
    local stage="$7"
    local split="$8"
    local task="$9"
    local log_stem="${10}"

    local log_file="${LOG_DIR}/${log_stem}.log"
    local slurm_out="${LOG_DIR}/slurm_${log_stem}_%j.out"

    local job_cpus="${SLURM_CPUS_CPU}"
    if [ "$use_gpu" = "true" ]; then
        job_cpus="${SLURM_CPUS_GPU}"
    fi

    local sbatch_args=(
        --parsable
        --job-name="${job_name}"
        --partition="${partition}"
        --nodes=1
        -c "${job_cpus}"
        --mem="${mem}"
        --time="${walltime}"
        --output="${slurm_out}"
        --error="${slurm_out}"
    )

    if [ "$use_gpu" = "true" ]; then
        sbatch_args+=(--gpus="${GPU_SPEC}")
        if [ -n "${GPU_CONSTRAINT:-}" ]; then
            sbatch_args+=(--constraint="${GPU_CONSTRAINT}")
        fi
    fi

    if [ -n "$dependency" ]; then
        sbatch_args+=(--dependency="${dependency}")
    fi

    # Pass stage parameters via environment variables (avoid commas inside --export values)
    export PROJECT_DIR SHARED_DATA_DIR SOURCE_SIF RESULTS_DIR TEAV_DATA_DIR
    export MLM_CONFIG CLASSIC_ML_CONFIG FT_CONFIG SURVIVAL_FT_CONFIG SURV_COMP_CONFIG ANALYSIS_CONFIG
    export MLM_OVERRIDES FT_OVERRIDES SURV_FT_OVERRIDES ML_OVERRIDES SURV_COMP_OVERRIDES EXTRA_MLM_OVERRIDES EXTRA_FT_OVERRIDES
    export DEBUG_MLM_OVERRIDES DEBUG_FT_OVERRIDES DEBUG_SURV_FT_OVERRIDES DEBUG_ML_OVERRIDES DEBUG_SURV_COMP_OVERRIDES
    export ANALYSIS_PROPOSED_MODEL STRATIFICATION_PROPOSED_MODEL ANALYSIS_RUN_ID COMPARISON_THRESHOLD_MODE COMPARISON_TARGET_RECALL COMPARISON_WORKERS STRATIFICATION_HORIZON
    export OPTUNA_FINETUNE_JSON OPTUNA_PRETRAINED_DIR RUN_ID_OVERRIDE TRAIN_DATA_AUGMENT DEBUG_MODE
    export OPTUNA_FT_TOTAL_TRIALS OPTUNA_FT_TRIALS_PER_WORKER OPTUNA_EXPORT_RUN_ID OPTUNA_REAP_RUNNING
    export OPTUNA_TASKS_COLON="${OPTUNA_TASKS//,/:}"
    export CLASSIC_ML_MODELS_COLON="${CLASSIC_ML_MODELS//,/:}"
    export CLASSIC_ML_FIXED_HP
    export CONTENDERS_COLON="${ANALYSIS_CONTENDER_MODELS// /:}"
    export SPLITS_COLON="$(join_by_colon "${RUN_SPLITS[@]}")"
    export COMP_TASKS_COLON="${COMPARISON_TASKS// /:}"
    export COMP_HORIZONS_COLON="${COMPARISON_HORIZONS// /:}"
    export STRAT_TASKS_COLON="${STRATIFICATION_TASKS// /:}"
    export JOB_STAGE="$stage"
    export JOB_SPLIT="$split"
    export JOB_TASK="$task"
    export JOB_LOG_FILE="$log_file"

    if [ "$DRY_RUN" = true ]; then
        FAKE_JOB_COUNTER=$((FAKE_JOB_COUNTER + 1))
        LAST_JOB_ID="$FAKE_JOB_COUNTER"
        echo "  [DRY-RUN -> ${LAST_JOB_ID}] sbatch ${sbatch_args[*]} ${WORKER_SCRIPT}  # stage=${stage} split=${split:-all} task=${task:-all}"
    else
        rm -f "$log_file"
        local jid
        jid=$(sbatch "${sbatch_args[@]}" --export=ALL "$WORKER_SCRIPT")
        jid="${jid%%;*}"
        LAST_JOB_ID="$jid"
        ALL_SUBMITTED_JOBS+=("$jid")
        echo "  -> Submitted ${job_name} (JobID=${jid}, partition=${partition}, time=${walltime}${dependency:+, dep=${dependency}}) | log: ${log_file}"
    fi
}

echo "=============================================================================="
echo "AIIDKIT Slurm Pipeline Orchestrator"
echo "  Logs directory : ${LOG_DIR}"
echo "  Splits         : ${RUN_SPLITS[*]}"
echo "  GPU Partition  : ${GPU_PARTITION} (${GPU_SPEC}, ${SLURM_CPUS_GPU} CPUs, ${SLURM_MEM_GPU})"
echo "  CPU Partition  : ${CPU_PARTITION} (${SLURM_CPUS_CPU} CPUs, ${SLURM_MEM_CPU})"
echo "  Proposed Model : ${ANALYSIS_PROPOSED_MODEL} vs [${ANALYSIS_CONTENDER_MODELS}]"
echo "  Step Toggles   : MLM=${RUN_STEP_1_MLM} | ClassicML=${RUN_STEP_4_CLASSIC_ML} | BinaryFT=${RUN_STEP_3_FINETUNING} | SurvivalFT=${RUN_STEP_2_SURVIVAL_FT}"
echo "  Analysis       : Comparison=${RUN_STEP_5_COMPARISON} | SurvivalComp=${RUN_STEP_6_SURVIVAL_COMP} | Interpret=${RUN_STEP_7_INTERPRET} | Strat=${RUN_STEP_8_STRATIFICATION} (horizon=${STRATIFICATION_HORIZON}d)"
echo "=============================================================================="

declare -A PREWARM_JOB_BY_SPLIT=()
declare -A MLM_JOB_BY_SPLIT=()
declare -A FT_JOBS_BY_SPLIT=()
declare -A SURV_JOBS_BY_SPLIT=()
declare -A CML_JOBS_BY_SPLIT=()
ALL_FT_JOBS=()
ALL_SURV_JOBS=()
ALL_CML_JOBS=()
OPTUNA_FT_FINAL_JOB=""

# ==============================================================================
# STEP 0-PREWARM: Build shared slim processed caches on CPU nodes upfront so GPU
#                 jobs (Optuna, MLM, Fine-Tuning, Survival) spend 0s building caches
# ==============================================================================
NEED_TF_CACHE=false
if [ "$RUN_STEP_0_OPTUNA_FINETUNE" = true ] || \
   [ "$RUN_STEP_1_MLM" = true ] || [ "$RUN_STEP_3_FINETUNING" = true ] || [ "$RUN_STEP_2_SURVIVAL_FT" = true ]; then
    NEED_TF_CACHE=true
fi

if [ "$RUN_STEP_0_PREWARM" = true ] && [ "$NEED_TF_CACHE" = true ]; then
    declare -A _PREWARM_SEEN=()
    PREWARM_SPLITS=()
    if [ "$RUN_STEP_0_OPTUNA_FINETUNE" = true ]; then
        PREWARM_SPLITS+=("$OPTUNA_SPLIT")
        _PREWARM_SEEN["$OPTUNA_SPLIT"]=1
    fi
    if [ "$RUN_STEP_1_MLM" = true ] || [ "$RUN_STEP_3_FINETUNING" = true ] || [ "$RUN_STEP_2_SURVIVAL_FT" = true ]; then
        for split in "${RUN_SPLITS[@]}"; do
            if [ -z "${_PREWARM_SEEN[$split]:-}" ]; then
                PREWARM_SPLITS+=("$split")
                _PREWARM_SEEN["$split"]=1
            fi
        done
    fi
    echo -e "\n>>> [Step 0-Prewarm] Submitting CPU cache prewarm jobs across ${#PREWARM_SPLITS[@]} split(s) in parallel (zero GPU time used)..."
    for split in "${PREWARM_SPLITS[@]}"; do
        sp=$(short_split "$split")
        submit_slurm_job \
            "prewarm-${sp}" \
            "$CPU_PARTITION" \
            "$TIME_PREWARM" \
            "$SLURM_MEM_CPU" \
            "false" \
            "" \
            "prewarm" \
            "$split" \
            "" \
            "step0_prewarm_${split}"
        PREWARM_JOB_BY_SPLIT["$split"]="$LAST_JOB_ID"
    done
else
    echo -e "\n>>> [Step 0-Prewarm] Skipped CPU Cache Prewarm (RUN_STEP_0_PREWARM=${RUN_STEP_0_PREWARM})"
fi

# ==============================================================================
# STEP 0 (Optional): Parallel Fine-Tuning Optuna HPO on OPTUNA_SPLIT
#   Seeds from canonical pre-trained backbone (e00-a15-v60_raw/pretraining)
#   or Step-1a MLM if RUN_STEP_1_MLM=true on OPTUNA_SPLIT.
# ==============================================================================
# ==============================================================================
# STEP 1a: MLM Pre-Training (1 GPU job per split, launched in parallel)
# ==============================================================================
if [ "$RUN_STEP_1_MLM" = true ]; then
    echo -e "\n>>> [Step 1a] Submitting MLM Pre-Training jobs across ${#RUN_SPLITS[@]} split(s) in parallel..."
    for split in "${RUN_SPLITS[@]}"; do
        sp=$(short_split "$split")
        mlm_deps=()
        [ -n "${PREWARM_JOB_BY_SPLIT[$split]:-}" ] && mlm_deps+=("${PREWARM_JOB_BY_SPLIT[$split]}")
        mlm_dep=""
        [ ${#mlm_deps[@]} -gt 0 ] && mlm_dep="afterok:$(join_by_colon "${mlm_deps[@]}")"
        submit_slurm_job \
            "pt-${sp}" \
            "$GPU_PARTITION" \
            "$TIME_MLM" \
            "$SLURM_MEM_GPU" \
            "true" \
            "$mlm_dep" \
            "mlm" \
            "$split" \
            "" \
            "step1a_mlm_${split}"
        MLM_JOB_BY_SPLIT["$split"]="$LAST_JOB_ID"
    done
else
    echo -e "\n>>> [Step 1a] Skipped MLM Pre-Training (RUN_STEP_1_MLM=false)"
    if [ -n "${EXTERNAL_MLM_JOB_ID:-}" ]; then
        for split in "${RUN_SPLITS[@]}"; do
            MLM_JOB_BY_SPLIT["$split"]="$EXTERNAL_MLM_JOB_ID"
        done
        echo "  -> Chaining downstream jobs to existing MLM JobID=${EXTERNAL_MLM_JOB_ID} (afterok:${EXTERNAL_MLM_JOB_ID})"
    fi
fi

if [ "$RUN_STEP_0_OPTUNA_FINETUNE" = true ]; then
    sp=$(short_split "$OPTUNA_SPLIT")
    opt_ft_deps=()
    [ -n "${PREWARM_JOB_BY_SPLIT[$OPTUNA_SPLIT]:-}" ] && opt_ft_deps+=("${PREWARM_JOB_BY_SPLIT[$OPTUNA_SPLIT]}")
    [ -n "${MLM_JOB_BY_SPLIT[$OPTUNA_SPLIT]:-}" ] && opt_ft_deps+=("${MLM_JOB_BY_SPLIT[$OPTUNA_SPLIT]}")
    dep=""
    [ ${#opt_ft_deps[@]} -gt 0 ] && dep="afterok:$(join_by_colon "${opt_ft_deps[@]}")"
    echo -e "\n>>> [Step 0-Optuna] Submitting Fine-Tuning Optuna (global target=${OPTUNA_FT_TOTAL_TRIALS} trials across ${OPTUNA_WORKERS} workers, cap=${OPTUNA_FT_TRIALS_PER_WORKER}/worker; tasks=${OPTUNA_TASKS}) on ${OPTUNA_SPLIT}..."
    opt_ft_workers=()
    for ((w=1; w<=OPTUNA_WORKERS; w++)); do
        submit_slurm_job \
            "opt-ft-${sp}-w${w}" \
            "$GPU_PARTITION" \
            "$TIME_OPTUNA_FT" \
            "$SLURM_MEM_GPU" \
            "true" \
            "$dep" \
            "optuna_ft_worker" \
            "$OPTUNA_SPLIT" \
            "w${w}" \
            "step0_optuna_ft_${OPTUNA_SPLIT}_w${w}"
        opt_ft_workers+=("$LAST_JOB_ID")
    done
    submit_slurm_job \
        "opt-ft-${sp}-fin" \
        "$GPU_PARTITION" \
        "00-02:30:00" \
        "$SLURM_MEM_GPU" \
        "true" \
        "afterany:$(join_by_colon "${opt_ft_workers[@]}")" \
        "optuna_ft_finalize" \
        "$OPTUNA_SPLIT" \
        "finalize" \
        "step0_optuna_ft_${OPTUNA_SPLIT}_finalize"
    OPTUNA_FT_FINAL_JOB="$LAST_JOB_ID"
    if [ -z "${OPTUNA_FINETUNE_JSON:-}" ]; then
        SAFE_OPT_TASK_TAG="${OPTUNA_TASKS//,/-}"
        OPTUNA_FINETUNE_JSON="${RESULTS_DIR}/${DEFAULT_FAMILY_DIR}/optuna_finetune/${OPTUNA_SPLIT}_${DEFAULT_VAL_TAG}/${SAFE_OPT_TASK_TAG}/best_finetune_config.json"
    fi
fi

# ==============================================================================
# STEP 1b: Classical ML Baselines (1 CPU job per (split, task), launched in
#          parallel at the same time as Step 1a — no GPU or MLM dependency)
# ==============================================================================
if [ "$RUN_STEP_4_CLASSIC_ML" = true ]; then
    echo -e "\n>>> [Step 1b] Submitting Classical ML jobs (${CLASSIC_ML_MODELS}) across ${#RUN_SPLITS[@]} split(s) x ${#CML_TASK_LIST[@]} task(s) in parallel..."
    for split in "${RUN_SPLITS[@]}"; do
        sp=$(short_split "$split")
        split_cml_jobs=()
        for task in "${CML_TASK_LIST[@]}"; do
            tk=$(short_task "$task")
            submit_slurm_job \
                "cml-${sp}-${tk}" \
                "$CPU_PARTITION" \
                "$TIME_CLASSIC_ML" \
                "$SLURM_MEM_CPU" \
                "false" \
                "" \
                "classic_ml" \
                "$split" \
                "$task" \
                "step1b_classic_ml_${split}_${task}"
            split_cml_jobs+=("$LAST_JOB_ID")
            ALL_CML_JOBS+=("$LAST_JOB_ID")
        done
        CML_JOBS_BY_SPLIT["$split"]="$(join_by_colon "${split_cml_jobs[@]}")"
    done
else
    echo -e "\n>>> [Step 1b] Skipped Classical ML Baselines (RUN_STEP_4_CLASSIC_ML=false)"
fi

# ==============================================================================
# STEP 2: Binary Classification Fine-Tuning — both Single-Task (non-MT) and
#         Multi-Task (MT), 1 GPU job per (split, task) in parallel.
#         Waits for Step 1a's split-specific MLM job if Step 1a was enabled.
# ==============================================================================
if [ "$RUN_STEP_3_FINETUNING" = true ]; then
    echo -e "\n>>> [Step 2] Submitting Binary Fine-Tuning (non-MT + MT) across ${#RUN_SPLITS[@]} split(s) x ${#FT_TASK_LIST[@]} task(s) in parallel..."
    for split in "${RUN_SPLITS[@]}"; do
        sp=$(short_split "$split")
        ft_deps=()
        [ -n "${PREWARM_JOB_BY_SPLIT[$split]:-}" ] && ft_deps+=("${PREWARM_JOB_BY_SPLIT[$split]}")
        [ -n "${MLM_JOB_BY_SPLIT[$split]:-}" ] && ft_deps+=("${MLM_JOB_BY_SPLIT[$split]}")
        [ -n "${OPTUNA_FT_FINAL_JOB:-}" ] && ft_deps+=("${OPTUNA_FT_FINAL_JOB}")
        dep=""
        if [ ${#ft_deps[@]} -gt 0 ]; then
            dep="afterok:$(join_by_colon "${ft_deps[@]}")"
        fi
        split_ft_jobs=()
        for task in "${FT_TASK_LIST[@]}"; do
            if [ "$RUN_STEP_0_OPTUNA_FINETUNE" = true ] && [ "$split" = "$OPTUNA_SPLIT" ] && [ -n "${OPTUNA_FT_FINAL_JOB:-}" ] && [[ ",${OPTUNA_TASKS}," == *",${task},"* ]]; then
                split_ft_jobs+=("$OPTUNA_FT_FINAL_JOB")
                ALL_FT_JOBS+=("$OPTUNA_FT_FINAL_JOB")
                echo "  -> Reusing Stage-0b Optuna winning fine-tuning on ${split}/${task} (JobID=${OPTUNA_FT_FINAL_JOB})"
                continue
            fi
            tk=$(short_task "$task")
            submit_slurm_job \
                "ft-${sp}-${tk}" \
                "$GPU_PARTITION" \
                "$TIME_FINETUNE" \
                "$SLURM_MEM_GPU" \
                "true" \
                "$dep" \
                "finetune" \
                "$split" \
                "$task" \
                "step2_finetune_${split}_${task}"
            split_ft_jobs+=("$LAST_JOB_ID")
            ALL_FT_JOBS+=("$LAST_JOB_ID")
        done
        FT_JOBS_BY_SPLIT["$split"]="$(join_by_colon "${split_ft_jobs[@]}")"
    done
else
    echo -e "\n>>> [Step 2] Skipped Binary Fine-Tuning (RUN_STEP_3_FINETUNING=false)"
fi

# ==============================================================================
# STEP 3: Survival (Cox) Fine-Tuning — 1 GPU job per (split, task) in parallel.
#         Kept FALSE by default (enable with RUN_STEP_2_SURVIVAL_FT=true).
#         Waits for Step 1a's split-specific MLM job if Step 1a was enabled.
# ==============================================================================
if [ "$RUN_STEP_2_SURVIVAL_FT" = true ]; then
    echo -e "\n>>> [Step 3] Submitting Survival (Cox) Fine-Tuning across ${#RUN_SPLITS[@]} split(s) x ${#SURV_TASK_LIST[@]} task(s) in parallel..."
    for split in "${RUN_SPLITS[@]}"; do
        sp=$(short_split "$split")
        ft_deps=()
        [ -n "${PREWARM_JOB_BY_SPLIT[$split]:-}" ] && ft_deps+=("${PREWARM_JOB_BY_SPLIT[$split]}")
        [ -n "${MLM_JOB_BY_SPLIT[$split]:-}" ] && ft_deps+=("${MLM_JOB_BY_SPLIT[$split]}")
        [ -n "${OPTUNA_FT_FINAL_JOB:-}" ] && ft_deps+=("${OPTUNA_FT_FINAL_JOB}")
        dep=""
        if [ ${#ft_deps[@]} -gt 0 ]; then
            dep="afterok:$(join_by_colon "${ft_deps[@]}")"
        fi
        split_surv_jobs=()
        for task in "${SURV_TASK_LIST[@]}"; do
            tk=$(short_task "$task")
            submit_slurm_job \
                "surv-${sp}-${tk}" \
                "$GPU_PARTITION" \
                "$TIME_SURVIVAL_FT" \
                "$SLURM_MEM_GPU" \
                "true" \
                "$dep" \
                "survival_ft" \
                "$split" \
                "$task" \
                "step3_survival_${split}_${task}"
            split_surv_jobs+=("$LAST_JOB_ID")
            ALL_SURV_JOBS+=("$LAST_JOB_ID")
        done
        SURV_JOBS_BY_SPLIT["$split"]="$(join_by_colon "${split_surv_jobs[@]}")"
    done
else
    echo -e "\n>>> [Step 3] Skipped Survival Fine-Tuning (RUN_STEP_2_SURVIVAL_FT=false)"
fi

# ==============================================================================
# STEP 5: Model Comparison Analysis (DCA, McNemar, PR-AUC) on CPU
#         Compares ANALYSIS_PROPOSED_MODEL (default: tf_raw_mt)
#         against ANALYSIS_CONTENDER_MODELS (logistic_regression, random_forest, xgboost)
# ==============================================================================
if [ "$RUN_STEP_5_COMPARISON" = true ]; then
    read -r -a COMP_TASK_LIST <<< "${COMPARISON_TASKS//,/ }"
    echo -e "\n>>> [Step 5] Submitting Parallel Model Comparison Analysis (${ANALYSIS_PROPOSED_MODEL} vs ${ANALYSIS_CONTENDER_MODELS}) across ${#COMP_TASK_LIST[@]} task(s)..."
    comp_deps=()
    [ ${#ALL_FT_JOBS[@]} -gt 0 ]   && comp_deps+=("${ALL_FT_JOBS[@]}")
    [ ${#ALL_SURV_JOBS[@]} -gt 0 ] && comp_deps+=("${ALL_SURV_JOBS[@]}")
    [ ${#ALL_CML_JOBS[@]} -gt 0 ]  && comp_deps+=("${ALL_CML_JOBS[@]}")
    dep=""
    if [ ${#comp_deps[@]} -gt 0 ]; then
        dep="afterok:$(join_by_colon "${comp_deps[@]}")"
    fi
    comp_task_jobs=()
    for ctask in "${COMP_TASK_LIST[@]}"; do
        tk=$(short_task "$ctask")
        submit_slurm_job \
            "anl-cmp-${tk}" \
            "$CPU_PARTITION" \
            "$TIME_COMPARISON" \
            "$SLURM_MEM_CPU" \
            "false" \
            "$dep" \
            "comparison" \
            "" \
            "$ctask" \
            "step5_comparison_${ctask}"
        comp_task_jobs+=("$LAST_JOB_ID")
    done
    sum_dep="afterok:$(join_by_colon "${comp_task_jobs[@]}")"
    submit_slurm_job \
        "anl-summary" \
        "$CPU_PARTITION" \
        "00-00:30:00" \
        "16G" \
        "false" \
        "$sum_dep" \
        "summary_report" \
        "" \
        "" \
        "step5_summary_report"
else
    echo -e "\n>>> [Step 5] Skipped Model Comparison Analysis (RUN_STEP_5_COMPARISON=false)"
fi

# ==============================================================================
# STEP 6: Cross-Family Survival Comparison (default: false)
# ==============================================================================
if [ "$RUN_STEP_6_SURVIVAL_COMP" = true ]; then
    echo -e "\n>>> [Step 6] Submitting Cross-Family Survival Comparison..."
    for split in "${RUN_SPLITS[@]}"; do
        sp=$(short_split "$split")
        s_deps=()
        [ -n "${FT_JOBS_BY_SPLIT[$split]:-}" ]   && s_deps+=("${FT_JOBS_BY_SPLIT[$split]}")
        [ -n "${SURV_JOBS_BY_SPLIT[$split]:-}" ] && s_deps+=("${SURV_JOBS_BY_SPLIT[$split]}")
        [ -n "${CML_JOBS_BY_SPLIT[$split]:-}" ]  && s_deps+=("${CML_JOBS_BY_SPLIT[$split]}")
        dep=""
        if [ ${#s_deps[@]} -gt 0 ]; then
            dep="afterok:$(join_by_colon "${s_deps[@]}")"
        fi
        submit_slurm_job \
            "anl-surv-${sp}" \
            "$CPU_PARTITION" \
            "$TIME_SURV_COMP" \
            "$SLURM_MEM_CPU" \
            "false" \
            "$dep" \
            "survival_comp" \
            "$split" \
            "" \
            "step6_survival_comparison_${split}"
    done
else
    echo -e "\n>>> [Step 6] Skipped Cross-Family Survival Comparison (RUN_STEP_6_SURVIVAL_COMP=false)"
fi

# ==============================================================================
# STEP 7: Captum Feature Attribution (1 GPU job per (split, task) in parallel)
# ==============================================================================
if [ "$RUN_STEP_7_INTERPRET" = true ]; then
    read -r -a INT_TASK_LIST <<< "${INTERPRET_TASKS//,/ }"
    echo -e "\n>>> [Step 7] Submitting Captum Feature Attribution across ${#RUN_SPLITS[@]} split(s) x ${#INT_TASK_LIST[@]} task(s)..."
    for split in "${RUN_SPLITS[@]}"; do
        sp=$(short_split "$split")
        dep=""
        if [ -n "${FT_JOBS_BY_SPLIT[$split]:-}" ]; then
            dep="afterok:${FT_JOBS_BY_SPLIT[$split]}"
        fi
        for task in "${INT_TASK_LIST[@]}"; do
            tk=$(short_task "$task")
            submit_slurm_job \
                "anl-int-${sp}-${tk}" \
                "$GPU_PARTITION" \
                "$TIME_INTERPRET" \
                "$SLURM_MEM_GPU" \
                "true" \
                "$dep" \
                "interpretability" \
                "$split" \
                "$task" \
                "step7_interpretability_${split}_${task}"
        done
    done
else
    echo -e "\n>>> [Step 7] Skipped Captum Feature Attribution (RUN_STEP_7_INTERPRET=false)"
fi

# ==============================================================================
# STEP 8: Patient Stratification & Kaplan-Meier Survival (1 GPU job per split)
# ==============================================================================
if [ "$RUN_STEP_8_STRATIFICATION" = true ]; then
    echo -e "\n>>> [Step 8] Submitting Patient Stratification (horizon=${STRATIFICATION_HORIZON}d) across ${#RUN_SPLITS[@]} split(s)..."
    for split in "${RUN_SPLITS[@]}"; do
        sp=$(short_split "$split")
        dep=""
        if [ -n "${FT_JOBS_BY_SPLIT[$split]:-}" ]; then
            dep="afterok:${FT_JOBS_BY_SPLIT[$split]}"
        fi
        submit_slurm_job \
            "anl-strat-${sp}" \
            "$GPU_PARTITION" \
            "$TIME_STRAT" \
            "$SLURM_MEM_GPU" \
            "true" \
            "$dep" \
            "stratification" \
            "$split" \
            "" \
            "step8_stratification_${split}"
    done
else
    echo -e "\n>>> [Step 8] Skipped Patient Stratification (RUN_STEP_8_STRATIFICATION=false)"
fi

echo ""
echo "=============================================================================="
if [ "$DRY_RUN" = true ]; then
    echo "DRY-RUN complete. No Slurm jobs were submitted."
else
    echo "All ${#ALL_SUBMITTED_JOBS[@]} Slurm jobs submitted successfully!"
    echo "  Job IDs : ${ALL_SUBMITTED_JOBS[*]}"
    echo "  Monitor : squeue -u \$USER"
    echo "  Logs    : ${LOG_DIR}"
fi
echo "=============================================================================="
