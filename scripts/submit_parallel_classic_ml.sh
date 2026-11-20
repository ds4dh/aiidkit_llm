#!/usr/bin/env bash
# ==============================================================================
# Thin CLI wrapper around `scripts/train_all.sh` for Step 1b (Classic ML).
# Delegates all Slurm setup and 0700 scratch permissions to `scripts/train_all.sh`.
#
# Usage:
#   bash scripts/submit_parallel_classic_ml.sh                     # all 13 unified tasks on random_split
#   bash scripts/submit_parallel_classic_ml.sh --split center_split --tasks infection_bacteria,death
#   bash scripts/submit_parallel_classic_ml.sh --fixed-hp --dry-run
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SPLIT="random_split"
CONFIG="configs/discriminative_classic_ml.yaml"
WALLTIME="00-06:00:00"
PARTITION_OVERRIDE="${CPU_PARTITION:-shared-cpu}"
CUSTOM_TASKS=""
CUSTOM_MODELS=""
FIXED_HP=false
PASSTHROUGH_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --split|-s) SPLIT="$2"; shift 2 ;;
        --config|-c) CONFIG="$2"; shift 2 ;;
        --tasks|-t) CUSTOM_TASKS="$2"; shift 2 ;;
        --models|-m) CUSTOM_MODELS="$2"; shift 2 ;;
        --all-tasks) shift ;;
        --fixed-hp|--fixed_hp|-f) FIXED_HP=true; shift ;;
        --partition|-p) PARTITION_OVERRIDE="$2"; shift 2 ;;
        --time) WALLTIME="$2"; shift 2 ;;
        --dry-run|-n) PASSTHROUGH_ARGS+=(--dry-run); shift ;;
        *)
            CUSTOM_TASKS="${CUSTOM_TASKS:+${CUSTOM_TASKS},}$1"
            shift
            ;;
    esac
done

export CLASSIC_ML_CONFIG="$CONFIG"
export TIME_CLASSIC_ML="$WALLTIME"
export CPU_PARTITION="$PARTITION_OVERRIDE"
export CLASSIC_ML_FIXED_HP="$FIXED_HP"
[ -n "$CUSTOM_TASKS" ] && export CLASSIC_ML_TASKS="$CUSTOM_TASKS"
[ -n "$CUSTOM_MODELS" ] && export CLASSIC_ML_MODELS="$CUSTOM_MODELS"
export RUN_STEP_0_PREWARM=false
export RUN_STEP_1_MLM=false
export RUN_STEP_4_CLASSIC_ML=true
export RUN_STEP_3_FINETUNING=false
export RUN_STEP_2_SURVIVAL_FT=false
export RUN_STEP_5_COMPARISON=false
export RUN_STEP_6_SURVIVAL_COMP=false
export RUN_STEP_7_INTERPRET=false
export RUN_STEP_8_STRATIFICATION=false

exec bash "${SCRIPT_DIR}/train_all.sh" --splits "$SPLIT" "${PASSTHROUGH_ARGS[@]}"
