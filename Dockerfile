# ==============================================================================
# AIIDKIT Clinical Decision Support Web Tool — Isolated Production Dockerfile
# ==============================================================================
# Zero patient data or model checkpoints are baked into this image.
# At runtime, mount ONLY the de-identified embeddings-only `server_bundle/`
# read-only at `/data/bundle:ro`.
# ==============================================================================
FROM python:3.12-slim

ARG PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    AIIDKIT_HOST=0.0.0.0 \
    AIIDKIT_PORT=5000 \
    AIIDKIT_BUNDLE_DIR=/data/bundle \
    AIIDKIT_RUNTIME_CACHE_DIR=/tmp/aiidkit_runtime_cache \
    AIIDKIT_TRUST_PROXY=true \
    NUMBA_CACHE_DIR=/tmp/numba_cache

# Minimal build tools only for hdbscan Cython wheels if prebuilt wheel unavailable
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install lean CPU PyTorch first (override PYTORCH_INDEX_URL for GPU servers if desired),
# then remaining web + inference packages.
COPY web_tool/requirements.txt /app/web_tool/requirements.txt
RUN pip install --upgrade pip \
    && pip install --index-url "${PYTORCH_INDEX_URL}" "torch>=2.2.0" \
    && pip install -r /app/web_tool/requirements.txt \
    && apt-get purge -y --auto-remove build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy strictly the minimal model inference modules and web_tool application code
COPY src/__init__.py /app/src/__init__.py
COPY src/tasks.py /app/src/tasks.py
COPY src/model/__init__.py /app/src/model/__init__.py
COPY src/model/model_utils.py /app/src/model/model_utils.py
COPY src/model/patient_embedder.py /app/src/model/patient_embedder.py
COPY web_tool/ /app/web_tool/

# Run as unprivileged user (UID 1000)
RUN useradd --create-home --uid 1000 --shell /usr/sbin/nologin aiidkit \
    && chown -R aiidkit:aiidkit /app

USER aiidkit
WORKDIR /app/web_tool

EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=5s --start-period=300s --retries=3 \
    CMD curl -fsS http://127.0.0.1:5000/api/health || exit 1

# Single Gunicorn worker + 4 threads shares the specialist model pool & UMAP manifolds in RAM
# across concurrent requests (guarded by thread-safe _MODEL_LOCK in app.py).
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "1", "--threads", "4", "--timeout", "300", "--access-logfile", "-", "app:app"]
