#!/usr/bin/env bash
# ==============================================================================
# Launch the AIIDKIT Clinical Decision Support Web Tool inside research-env.sif
#
# Usage (directly on a GPU/CPU node such as gpu037):
#   bash web_tool/run_web_tool.sh
#   bash web_tool/run_web_tool.sh --port 5005
#   bash web_tool/run_web_tool.sh --model results/transformer/random_split/e00-a15-v60_raw/finetuning/multitask_bacteria
# ==============================================================================
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SIF_IMAGE="${PROJECT_DIR}/research-env.sif"
PORT="${AIIDKIT_PORT:-5000}"
MODEL_OVERRIDE="${AIIDKIT_MODEL_PATH:-}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --port|-p) PORT="$2"; shift 2 ;;
        --model|-m) MODEL_OVERRIDE="$2"; shift 2 ;;
        --demo|--mock) export APPTAINERENV_AIIDKIT_DEMO_MODE="1"; shift ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

if [[ ! -f "$SIF_IMAGE" ]]; then
    echo "Error: Apptainer image not found at $SIF_IMAGE" >&2
    exit 1
fi

PERSISTENT_CACHE_DIR="$HOME/.cache"
mkdir -p "$PERSISTENT_CACHE_DIR/huggingface" "$PERSISTENT_CACHE_DIR/matplotlib"

export APPTAINERENV_VIRTUAL_ENV="/opt/venv"
export APPTAINERENV_PATH="/opt/venv/bin:$PATH"
export APPTAINERENV_HF_HOME="$PERSISTENT_CACHE_DIR/huggingface"
export APPTAINERENV_MPLCONFIGDIR="$PERSISTENT_CACHE_DIR/matplotlib"
export APPTAINERENV_REQUESTS_CA_BUNDLE="/mnt/host_cert.crt"
export APPTAINERENV_SSL_CERT_FILE="/mnt/host_cert.crt"
export APPTAINERENV_PYTHONUNBUFFERED="1"
export APPTAINERENV_AIIDKIT_PORT="$PORT"
if [[ -n "$MODEL_OVERRIDE" ]]; then
    export APPTAINERENV_AIIDKIT_MODEL_PATH="$MODEL_OVERRIDE"
fi

BIND_MOUNTS="$HOME:$HOME,/home/shares/ds4dh:/home/shares/ds4dh,/etc/pki/tls/certs/ca-bundle.crt:/mnt/host_cert.crt:ro"
if [[ -d "/srv/beegfs/scratch" ]]; then
    BIND_MOUNTS="${BIND_MOUNTS},/srv/beegfs/scratch:/srv/beegfs/scratch"
fi

NV_FLAG=()
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    NV_FLAG+=(--nv)
fi

cd "$PROJECT_DIR"
echo "=============================================================================="
echo "Starting AIIDKIT Web Tool on $(hostname):${PORT}"
echo "  Local URL (IDE port-forwarding): http://localhost:${PORT}"
echo "=============================================================================="
exec apptainer exec "${NV_FLAG[@]}" --bind "$BIND_MOUNTS" "$SIF_IMAGE" \
    /opt/venv/bin/python web_tool/app.py
