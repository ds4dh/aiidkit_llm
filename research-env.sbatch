#!/bin/bash

#SBATCH --partition=shared-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --output=build/logs/%x_%j.out
#SBATCH --error=build/logs/%x_%j.err

set -euo pipefail

# Directory paths
WORK_DIR=$(pwd)
DEF_FILE_PATH="${WORK_DIR}/research-env.def"
FINAL_SIF_PATH="${WORK_DIR}/research-env.sif"
LOG_DIR="${WORK_DIR}/build/logs"
mkdir -p "$LOG_DIR"

# Use node-local fast scratch storage for build intermediate files & logging
LOCAL_SCRATCH="/scratch/${USER}/${SLURM_JOB_ID}"
BUILD_TMP_DIR="${LOCAL_SCRATCH}/tmp"
BUILD_CACHE_DIR="${LOCAL_SCRATCH}/cache"
LOCAL_SIF_PATH="${LOCAL_SCRATCH}/research-env.sif"
LOCAL_BUILD_LOG="${LOCAL_SCRATCH}/build_internal.log"

# Create local scratch directories
mkdir -p "$BUILD_TMP_DIR" "$BUILD_CACHE_DIR"

# Ensure log copying and cleanup happen regardless of success or failure
cleanup() {
    echo "=========================================="
    if [ -f "$LOCAL_BUILD_LOG" ]; then
        echo "Copying internal build log to ${LOG_DIR}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_build.log..."
        cp "$LOCAL_BUILD_LOG" "${LOG_DIR}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_build.log" || true
    fi
    echo "Cleaning up local scratch directory..."
    rm -rf "$LOCAL_SCRATCH"
    echo "=========================================="
}
trap cleanup EXIT

# Export environment variables for Apptainer
export APPTAINER_TMPDIR="$BUILD_TMP_DIR"
export APPTAINER_CACHEDIR="$BUILD_CACHE_DIR"

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID on node $(hostname)"
echo "Using Local Scratch: $LOCAL_SCRATCH"
echo "Definition file: $DEF_FILE_PATH"
echo "=========================================="

# Remove existing final .sif file if present
if [ -f "$FINAL_SIF_PATH" ]; then
    rm "$FINAL_SIF_PATH"
    echo "Removed existing $FINAL_SIF_PATH"
fi

# Build image directly onto fast local scratch with I/O isolated to local scratch disk
echo "Building Apptainer image to local scratch (logging to local disk)..."
if apptainer build --fakeroot "$LOCAL_SIF_PATH" "$DEF_FILE_PATH" > "$LOCAL_BUILD_LOG" 2>&1; then
    echo "Build process completed successfully."
else
    BUILD_EXIT_CODE=$?
    echo "ERROR: Apptainer build failed with exit code $BUILD_EXIT_CODE."
    cat "$LOCAL_BUILD_LOG" | tail -n 60
    exit $BUILD_EXIT_CODE
fi

# Copy completed .sif back to project directory
if [ -f "$LOCAL_SIF_PATH" ]; then
    echo "Copying SIF to $FINAL_SIF_PATH..."
    cp "$LOCAL_SIF_PATH" "$FINAL_SIF_PATH"
    echo "SUCCESS: Apptainer image ready at $FINAL_SIF_PATH."
else
    echo "ERROR: No SIF image found at $LOCAL_SIF_PATH."
    exit 1
fi