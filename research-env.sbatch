#!/bin/bash

#SBATCH --partition=shared-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --output=build/logs/%x_%j.out
#SBATCH --error=build/logs/%x_%j.err

# Apptainer file paths
WORK_DIR=$(pwd)
DEF_FILE_PATH="${WORK_DIR}/research-env.def"
SIF_FILE_PATH="${WORK_DIR}/research-env.sif"
BUILD_TMP_DIR="${WORK_DIR}/build/tmp"
BUILD_CACHE_DIR="${WORK_DIR}/build/cache"

# Remove existing .sif file if it exists
if [ -f "$SIF_FILE_PATH" ]; then
    rm "$SIF_FILE_PATH"
    echo "Removed existing $SIF_FILE_PATH"
fi

# Create directories
mkdir -p "$BUILD_TMP_DIR"
mkdir -p "$BUILD_CACHE_DIR"

# Export the variables so Apptainer sees them
export APPTAINER_TMPDIR="$BUILD_TMP_DIR"
export APPTAINER_CACHEDIR="$BUILD_CACHE_DIR"

# Build the apptainer image
echo "Building Apptainer image..."
echo "Using Temp Dir: $APPTAINER_TMPDIR"
apptainer build --fakeroot "$SIF_FILE_PATH" "$DEF_FILE_PATH"

# Verify if the build was successful
if [ $? -eq 0 ] && [ -f "$SIF_FILE_PATH" ]; then
    echo "SUCCESS: Apptainer image built at $SIF_FILE_PATH."
else
    echo "ERROR: Failed to build Apptainer image."
    exit 1
fi