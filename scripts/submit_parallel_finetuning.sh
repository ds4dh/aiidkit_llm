#!/usr/bin/env bash
# ==============================================================================
# Thin CLI wrapper around `scripts/train_all.sh` for Step 2 (Binary Fine-Tuning).
# Delegates all Slurm setup, 0700 scratch permissions, and CPU cache prewarming
# to `scripts/train_all.sh`.
#
# Usage:
#   bash scripts/submit_parallel_finetuning.sh
#   bash scripts/submit_parallel_finetuning.sh --split temporal_split --tasks multitask_unified
#   bash scripts/submit_parallel_finetuning.sh --dry-run
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SPLIT="random_split"
CONFIG="configs/discriminative_training.yaml"
WALLTIME="00-06:00:00"
GPU_SPEC_OVERRIDE="${GPU_SPEC:-nvidia_geforce_rtx_3090:1}"
PARTITION_OVERRIDE="${GPU_PARTITION:-shared-gpu}"
CUSTOM_TASKS=""
PASSTHROUGH_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --split|-s) SPLIT="$2"; shift 2 ;;
        --config|-c) CONFIG="$2"; shift 2 ;;
        --tasks|-t) CUSTOM_TASKS="$2"; shift 2 ;;
        --time) WALLTIME="$2"; shift 2 ;;
        --gpu|--gpus) GPU_SPEC_OVERRIDE="$2"; shift 2 ;;
        --partition|-p) PARTITION_OVERRIDE="$2"; shift 2 ;;
        --dry-run|-n) PASSTHROUGH_ARGS+=(--dry-run); shift ;;
        *)
            CUSTOM_TASKS="${CUSTOM_TASKS:+${CUSTOM_TASKS},}$1"
            shift
            ;;
    esac
done

export FT_CONFIG="$CONFIG"
export TIME_FINETUNE="$WALLTIME"
export GPU_SPEC="$GPU_SPEC_OVERRIDE"
export GPU_PARTITION="$PARTITION_OVERRIDE"
export FT_TASKS="${CUSTOM_TASKS:-multitask_bacteria,multitask_viruses,multitask_fungi,death,graft_loss}"
export RUN_STEP_0_PREWARM="${RUN_STEP_0_PREWARM:-false}"
export RUN_STEP_1_MLM=false
export RUN_STEP_4_CLASSIC_ML=false
export RUN_STEP_3_FINETUNING=true
export RUN_STEP_2_SURVIVAL_FT=false
export RUN_STEP_5_COMPARISON=false
export RUN_STEP_6_SURVIVAL_COMP=false
export RUN_STEP_7_INTERPRET=false
export RUN_STEP_8_STRATIFICATION=false

exec bash "${SCRIPT_DIR}/train_all.sh" --splits "$SPLIT" "${PASSTHROUGH_ARGS[@]}"
