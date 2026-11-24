# Deploying AIIDKIT Web Tool with Docker (`ds4dh.unige.ch`)

This guide explains how to package and run the AIIDKIT Clinical Decision Support Tool inside an isolated **Docker** container powered by the **5-Domain Specialist Multi-Task Suite (`B_CAT_MT`)** using **strictly de-identified patient embeddings** (zero raw clinical records on the web server) and **HTTP Basic Auth**.

---

## 1. Specialist Multi-Model Architecture (`B_CAT_MT`) & Privacy Bundle

To maximize predictive accuracy and match clinical decision time-scales across all 13 post-transplant outcomes, the web tool routes predictions and UMAP cohort manifolds across **5 domain-specialist models**:
1. **`bacteria`** (`multitask_bacteria`): 5 bacterial pathogen groups @ **30d, 60d, 90d**
2. **`viruses`** (`multitask_viruses`): 4 viral pathogen groups (CMV, BKPyV, Herpes, General) @ **30d, 60d, 90d**
3. **`fungi`** (`multitask_fungi`): All Fungi & *Candida* spp. @ **30d, 60d, 90d**
4. **`graft_loss`** (`graft_loss`): Allograft Failure @ **1 year, 2 years, 5 years** (`360d, 720d, 1800d`)
5. **`death`** (`death`): Patient Mortality @ **1 year, 2 years, 5 years** (`360d, 720d, 1800d`)

Raw transplant cohort data (`/home/shares/ds4dh/...`) **never leaves the HPC cluster** and is **never mounted** into the web container. Instead, `web_tool/export_server_bundle.py` exports `server_bundle/` (~`1.14 GB` total across all 5 models + 5 embedding sets):

```text
server_bundle/
├── bundle_manifest.json              # v2.0.0 multi-domain SHA-256 manifest
├── metadata/
│   ├── vocab.json                    # Shared clinical concept token -> ID dictionary
│   └── bin_intervals.json            # Shared population quantile cutoffs per lab test
└── domains/
    ├── bacteria/   (model/, platt_params.json, cohort_embeddings.npz)
    ├── viruses/    (model/, platt_params.json, cohort_embeddings.npz)
    ├── fungi/      (model/, platt_params.json, cohort_embeddings.npz)
    ├── graft_loss/ (model/, platt_params.json, cohort_embeddings.npz)
    └── death/      (model/, platt_params.json, cohort_embeddings.npz)
```

### Key Privacy & Systems Guarantees
- **Zero raw patient records**: No `.arrow`, `.parquet`, `.csv`, clinical event histories, lab values, demographics, or patient IDs exist in `server_bundle/`.
- **Unlinkable across follow-up landmarks**: Rows in every `cohort_embeddings.npz` are independently permuted with a cryptographic RNG (`secrets.randbits(64)`) at each FUP landmark (`0d..360d`).
- **Zero Python `pickle` files**: Model weights use `safetensors`, metadata uses pure JSON, and embeddings use `np.savez_compressed` loaded with `allow_pickle=False`.
- **LRU RAM Pool (`AIIDKIT_MAX_MODELS_IN_RAM`)**: Pre-computed UMAP cohorts for all 5 domains sit permanently in RAM (~`75 MB` total for `0 ms` instant cohort switching), while PyTorch weights use a configurable LRU pool (`5` models on GPU / `>=14 GB` servers, or `2` models on small CPU VMs).

---

## 2. Step-by-Step Export & Local Preview (on UNIGE Baobab HPC)

```bash
# Export and privacy-audit all 5 specialist domains into web_tool/server_bundle/
apptainer exec --nv --bind "$HOME:$HOME,/home/shares/ds4dh:/home/shares/ds4dh,/srv/beegfs/scratch:/srv/beegfs/scratch" \
    research-env.sif /opt/venv/bin/python web_tool/export_server_bundle.py

# Launch isolated local container preview with HTTP Basic Auth
bash web_tool/run_container_local.sh --port 5000 --user clinician --password aiidkit-preview
```

---

## 3. Deploying on `ds4dh.unige.ch` (for the Server Admin)

1. **Clone the repository** and copy `server_bundle/` into `./web_tool/server_bundle/`:
   ```bash
   git clone https://github.com/ds4dh/aiidkit_llm.git aiidkit
   cd aiidkit
   # Copy server_bundle/ (~1.14 GB) into ./web_tool/server_bundle/
   ```

2. **Configure environment (`web_tool/.env`)**:
   ```bash
   cp web_tool/.env.example web_tool/.env
   ```
   - Leave `AIIDKIT_BASIC_AUTH_PASS=` empty if Nginx/Traefik enforces Basic Auth in front of `127.0.0.1:5000`, or set `AIIDKIT_BASIC_AUTH_USER` / `AIIDKIT_BASIC_AUTH_PASS` for built-in app Basic Auth (`GET /api/health` remains unauthenticated for `HEALTHCHECK`).

3. **Build & start the isolated container**:
   ```bash
   docker compose up -d --build
   docker compose ps
   curl -fsS http://127.0.0.1:5000/api/health
   ```
