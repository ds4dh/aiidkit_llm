#!/usr/bin/env bash
# ==============================================================================
# Thin CLI wrapper around `scripts/train_all.sh` for Steps 5, 7, 8 (Analyses).
# Delegates all Slurm setup and 0700 scratch permissions to `scripts/train_all.sh`.
#
# Usage:
#   bash scripts/submit_parallel_analyses.sh
#   bash scripts/submit_parallel_analyses.sh --split random_split --modes comparison stratification
#   bash scripts/submit_parallel_analyses.sh --dry-run
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SPLIT="random_split"
CONFIG="configs/discriminative_training.yaml"
GPU_SPEC_OVERRIDE="${GPU_SPEC:-nvidia_geforce_rtx_3090:1}"
MODES=("comparison" "stratification" "interpretability")
PASSTHROUGH_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --split|--splits|-s) SPLIT="$2"; shift 2 ;;
        --config|-c) CONFIG="$2"; shift 2 ;;
        --gpu|--gpus) GPU_SPEC_OVERRIDE="$2"; shift 2 ;;
        --proposed-model) export ANALYSIS_PROPOSED_MODEL="$2"; shift 2 ;;
        --run-id) export ANALYSIS_RUN_ID="$2"; shift 2 ;;
        --tasks|-t) export ANALYSIS_TASKS="${2//,/ }"; export COMPARISON_TASKS="$ANALYSIS_TASKS"; export INTERPRET_TASKS="$ANALYSIS_TASKS"; export STRATIFICATION_TASKS="$ANALYSIS_TASKS"; shift 2 ;;
        --modes|-m)
            shift
            MODES=()
            while [[ $# -gt 0 && ! "$1" =~ ^- ]]; do
                MODES+=("$1")
                shift
            done
            ;;
        --dry-run|-n) PASSTHROUGH_ARGS+=(--dry-run); shift ;;
        *)
            echo "Error: Unknown argument '$1' in submit_parallel_analyses.sh" >&2
            exit 1
            ;;
    esac
done

has_mode() {
    local target="$1"
    for m in "${MODES[@]}"; do
        [[ "$m" == "$target" || "$m" == "all" ]] && return 0
    done
    return 1
}

export ANALYSIS_CONFIG="$CONFIG"
export GPU_SPEC="$GPU_SPEC_OVERRIDE"
export RUN_STEP_0_PREWARM=false
export RUN_STEP_1_MLM=false
export RUN_STEP_4_CLASSIC_ML=false
export RUN_STEP_3_FINETUNING=false
export RUN_STEP_2_SURVIVAL_FT=false
export RUN_STEP_6_SURVIVAL_COMP=false
export RUN_STEP_5_COMPARISON=$(has_mode "comparison" && echo true || echo false)
export RUN_STEP_7_INTERPRET=$(has_mode "interpretability" && echo true || echo false)
export RUN_STEP_8_STRATIFICATION=$(has_mode "stratification" && echo true || echo false)

exec bash "${SCRIPT_DIR}/train_all.sh" --splits "$SPLIT" "${PASSTHROUGH_ARGS[@]}"
