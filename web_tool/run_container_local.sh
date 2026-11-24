#!/usr/bin/env bash
# ==============================================================================
# AIIDKIT Web Tool — Isolated Container Local Test Launcher
# ==============================================================================
# Reproduces the exact isolated `ds4dh.unige.ch` deployment experience locally:
#   1. Exports (or verifies) the de-identified embeddings-only `server_bundle/`
#      containing ZERO raw patient data, ZERO ground-truth labels, and ZERO pickles.
#   2. Enables HTTP Basic Auth (`AIIDKIT_BASIC_AUTH_USER` / `AIIDKIT_BASIC_AUTH_PASS`).
#   3. Runs inside an isolated read-only container mounting ONLY code + `/data/bundle:ro`:
#      - Uses `docker compose` automatically on workstations/servers with Docker.
#      - Uses `apptainer exec --containall --cleanenv --no-home` on UNIGE Baobab HPC
#        (where the Docker daemon is disabled on compute nodes), guaranteeing zero
#        access to `/home/shares/ds4dh` or `$HOME` inside the container!
#
# Usage:
#   bash web_tool/run_container_local.sh
#   bash web_tool/run_container_local.sh --port 5005 --user clinician --password demo-secret
# ==============================================================================
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUNDLE_DIR="${PROJECT_DIR}/web_tool/server_bundle"
ENV_FILE="${PROJECT_DIR}/web_tool/.env"

if [[ -f "$ENV_FILE" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    set +a
fi

PORT="${AIIDKIT_PORT:-5000}"
AUTH_USER="${AIIDKIT_BASIC_AUTH_USER:-clinician}"
AUTH_PASS="${AIIDKIT_BASIC_AUTH_PASS:-aiidkit-preview}"
FORCE_EXPORT=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --port|-p) PORT="$2"; shift 2 ;;
        --user|-u) AUTH_USER="$2"; shift 2 ;;
        --password|--pass) AUTH_PASS="$2"; shift 2 ;;
        --no-auth) AUTH_PASS=""; shift ;;
        --reexport) FORCE_EXPORT=1; shift ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

# Step 1: Ensure de-identified 5-domain embeddings-only server_bundle exists & passes privacy audit
if [[ "$FORCE_EXPORT" -eq 1 || ! -d "${BUNDLE_DIR}/domains" ]]; then
    echo "[1/2] Exporting de-identified 5-domain specialist server_bundle..."
    if [[ -f "${PROJECT_DIR}/research-env.sif" ]] && command -v apptainer >/dev/null 2>&1; then
        EXP_NV=()
        if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then EXP_NV+=(--nv); fi
        EXP_BIND="$HOME:$HOME,/home/shares/ds4dh:/home/shares/ds4dh"
        if [[ -d "/srv/beegfs/scratch" ]]; then EXP_BIND="${EXP_BIND},/srv/beegfs/scratch:/srv/beegfs/scratch"; fi
        apptainer exec "${EXP_NV[@]}" --bind "$EXP_BIND" \
            "${PROJECT_DIR}/research-env.sif" \
            /opt/venv/bin/python "${PROJECT_DIR}/web_tool/export_server_bundle.py" --output "$BUNDLE_DIR"
    else
        python3 "${PROJECT_DIR}/web_tool/export_server_bundle.py" --output "$BUNDLE_DIR"
    fi
fi

echo "=============================================================================="
echo "Starting Isolated AIIDKIT Container on $(hostname):${PORT}"
echo "  Mode       : Embeddings-Only Bundle (/data/bundle:ro — ZERO raw patient data)"
echo "  Local URL  : http://localhost:${PORT}"
if [[ -n "$AUTH_PASS" ]]; then
    echo "  Basic Auth : ENABLED (username: ${AUTH_USER} | password: ${AUTH_PASS})"
else
    echo "  Basic Auth : DISABLED"
fi
echo "=============================================================================="

# Step 2A: Native Docker Compose path (workstations / ds4dh.unige.ch)
if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
    cd "$PROJECT_DIR"
    AIIDKIT_PORT="$PORT" \
    AIIDKIT_BASIC_AUTH_USER="$AUTH_USER" \
    AIIDKIT_BASIC_AUTH_PASS="$AUTH_PASS" \
    exec docker compose up --build
fi

# Step 2B: HPC isolated container path (`apptainer --containall --cleanenv --no-home`)
SIF_IMAGE="${PROJECT_DIR}/research-env.sif"
if [[ ! -f "$SIF_IMAGE" ]]; then
    echo "Error: Neither running Docker daemon nor ${SIF_IMAGE} found." >&2
    exit 1
fi

TMP_RUNTIME_DIR="$(mktemp -d "${TMPDIR:-/tmp}/aiidkit_container_XXXXXX")"
trap 'rm -rf "$TMP_RUNTIME_DIR"' EXIT
PERSIST_RUNTIME_CACHE="${TMPDIR:-/tmp}/aiidkit_isolated_cache_${USER:-uid$(id -u)}"
mkdir -p -m 700 "$PERSIST_RUNTIME_CACHE/runtime" "$PERSIST_RUNTIME_CACHE/numba"

NV_FLAG=()
USE_GPU_VAL="false"
if [[ "${AIIDKIT_USE_GPU:-true}" != "false" ]] && command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    NV_FLAG+=(--nv)
    USE_GPU_VAL="true"
fi

# Stage ONLY the exact files that Dockerfile copies into /app (read-only mount, no .cache or results!)
STAGE_APP="${TMP_RUNTIME_DIR}/app"
mkdir -p "${STAGE_APP}/src/model"
cp "${PROJECT_DIR}/src/__init__.py" "${STAGE_APP}/src/__init__.py"
cp "${PROJECT_DIR}/src/tasks.py" "${STAGE_APP}/src/tasks.py"
cp "${PROJECT_DIR}/src/model/__init__.py" "${STAGE_APP}/src/__init__.py"
cp "${PROJECT_DIR}/src/model/model_utils.py" "${STAGE_APP}/src/model/model_utils.py"
cp "${PROJECT_DIR}/src/model/patient_embedder.py" "${STAGE_APP}/src/model/patient_embedder.py"
mkdir -p "${STAGE_APP}/web_tool"
cp "${PROJECT_DIR}/web_tool/"*.py "${STAGE_APP}/web_tool/"
cp -r "${PROJECT_DIR}/web_tool/static" "${STAGE_APP}/web_tool/static"
cp -r "${PROJECT_DIR}/web_tool/templates" "${STAGE_APP}/web_tool/templates"
cp -r "${PROJECT_DIR}/web_tool/synthetic_patients" "${STAGE_APP}/web_tool/synthetic_patients"

exec apptainer exec \
    --containall \
    --cleanenv \
    --no-home \
    "${NV_FLAG[@]}" \
    --pwd /app/web_tool \
    --bind "${STAGE_APP}:/app:ro,${BUNDLE_DIR}:/data/bundle:ro,${PERSIST_RUNTIME_CACHE}:/var/cache/aiidkit,${TMP_RUNTIME_DIR}:/tmp" \
    --env VIRTUAL_ENV=/opt/venv \
    --env PATH="/opt/venv/bin:/usr/bin:/bin" \
    --env PYTHONUNBUFFERED=1 \
    --env NUMBA_CACHE_DIR=/var/cache/aiidkit/numba \
    --env AIIDKIT_HOST=0.0.0.0 \
    --env AIIDKIT_PORT="$PORT" \
    --env AIIDKIT_BUNDLE_DIR=/data/bundle \
    --env AIIDKIT_RUNTIME_CACHE_DIR=/var/cache/aiidkit/runtime \
    --env AIIDKIT_USE_GPU="$USE_GPU_VAL" \
    --env AIIDKIT_TRUST_PROXY=true \
    --env AIIDKIT_BASIC_AUTH_USER="$AUTH_USER" \
    --env AIIDKIT_BASIC_AUTH_PASS="$AUTH_PASS" \
    "$SIF_IMAGE" \
    /opt/venv/bin/python -c "
import app as web_app, config as CONFIG
web_app.app.run(host=CONFIG.HOST, port=CONFIG.PORT, debug=False, threaded=True)
"
