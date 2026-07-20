"""
AIIDKIT Decision Support Tool — Flask Backend
=============================================
Serves the single-page frontend and three REST API endpoints.

Endpoints
---------
GET  /                  Serve index.html
GET  /api/config        Return vocabulary + task config
GET  /api/cohort        Return pre-computed cohort data (250 patients)
POST /api/predict       Accept patient EAV sequence, return risk analysis

Real-model integration
----------------------
Edit MODEL_PATH and DATA_DIR in config.py and restart the server.
The tool auto-detects whether the paths exist and switches between
DEMO MODE (mock data) and LIVE MODEL (real inference) accordingly.

To run (from the web_tool/ directory):
    pip install -r requirements.txt
    python app.py
Then open http://localhost:5000
"""

import sys
import os
from pathlib import Path
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS

# ---------------------------------------------------------------------------
# Ensure src/ is importable: web_tool/ lives inside the modelling code root
# (locally: modelling_code/web_tool/  |  HPC: aiidkit_llm/web_tool/)
# parent       = web_tool/
# parent.parent = modelling_code/  (or aiidkit_llm/)  ← this goes on sys.path
#   ↳ allows:  from src.model...  from src.data...
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent.parent   # modelling_code/ (or aiidkit_llm/)
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import config as CONFIG
from mock_data import generate_cohort_data, mock_predict

# ---------------------------------------------------------------------------
# Optional: real model imports (only activated when MODEL_PATH is set)
# ---------------------------------------------------------------------------
_REAL_MODEL    = None
_REAL_COLLATOR = None
_REAL_VOCAB    = None
_REAL_BIN_INT  = None
_REAL_LOGITS_MEDIAN = None
_REAL_LOGITS_STD    = None


def _get_event_info_from_row(row, task_key):
    """
    Scans a dataset row for label keys and computes approximate days_to_event and event_imminence.
    """
    import numpy as np
    prefix = f"label_{task_key}_"
    horizons = []
    for k in row.keys():
        if k.startswith(prefix) and k.endswith("d"):
            try:
                day_str = k[len(prefix):-1]
                day = int(day_str)
                horizons.append((day, k))
            except ValueError:
                pass
    if not horizons:
        return None, 0.0
        
    horizons.sort(key=lambda x: x[0])
    
    first_event_h = None
    prev_h = 0
    for day, k in horizons:
        if row.get(k, 0) == 1:
            first_event_h = day
            break
        prev_h = day
        
    if first_event_h is not None:
        days_to_event = int((prev_h + first_event_h) / 2)
        event_imminence = float(np.exp(-days_to_event / 365.0))
        return days_to_event, event_imminence
    else:
        return None, 0.0


def _generate_real_cohort_data(test_dir_path) -> dict:
    """
    Load the real test split, run the real model over it to compute risk scores,
    and run UMAP + HDBSCAN to compute coordinates and clusters.
    """
    global _REAL_MODEL, _REAL_COLLATOR, _REAL_VOCAB, _REAL_BIN_INT, _REAL_UMAP_REDUCER
    global _REAL_LOGITS_MEDIAN, _REAL_LOGITS_STD
    import torch
    import numpy as np
    from collections import defaultdict
    from datasets import load_from_disk
    from torch.utils.data import DataLoader
    import umap
    import hdbscan
    from scipy.special import expit
    from lifelines import KaplanMeierFitter
    from lifelines.statistics import logrank_test
    from src.data.patient_dataset import BIN_LABELS

    device = next(_REAL_MODEL.parameters()).device
    model_dtype = next(_REAL_MODEL.parameters()).dtype
    device_type = "cuda" if "cuda" in str(device) else "cpu"

    print("[AIIDKIT] Loading real test dataset for cohort analysis...")
    test_ds = load_from_disk(str(test_dir_path))
    
    # Subsample to keep startup fast (e.g. max 500 patients)
    n_patients = min(len(test_ds), 500)
    print(f"[AIIDKIT] Found {len(test_ds)} patients. Using subset of {n_patients} for cohort analysis.")
    subset_ds = test_ds.select(range(n_patients))
    subset_ds = subset_ds.add_column("split", ["test"] * len(subset_ds))
    
    # Preprocess all patients in the subset using the dataset tokenization logic
    from src.data.patient_dataset import preprocess_batch
    processed = preprocess_batch(
        batch=subset_ds,
        vocab=_REAL_VOCAB,
        bin_intervals=_REAL_BIN_INT,
        bin_labels=BIN_LABELS,
        time_mapping=CONFIG.TIME_MAPPING,
        eav_mappings=CONFIG.EAV_MAPPINGS,
    )

    samples = []
    patient_ids = []
    for i in range(n_patients):
        row = subset_ds[i]
        pid = row.get("patientid", f"PAT_{i:03d}")
        patient_ids.append(pid)
        
        sample = {
            "entity_id":      np.array(processed["entity_id"][i]),
            "attribute_id":   np.array(processed["attribute_id"][i]),
            "value_id":       np.array(processed["value_id"][i]),
            "days_since_tpx": np.array(processed["days_since_tpx"][i]),
            "split":          "test",
        }
        for k in _REAL_COLLATOR.label_keys:
            sample[k] = 0.0
        samples.append(sample)

    # Run batch inference in mini-batches to avoid GPU Out Of Memory
    batch_size = 16
    all_logits = []
    all_embeddings = []
    
    print("[AIIDKIT] Extracting embeddings and predictions for real cohort...")
    for idx in range(0, n_patients, batch_size):
        batch_samples = samples[idx : idx + batch_size]
        batch_collated = _REAL_COLLATOR.torch_call(batch_samples)
        
        input_dict = {}
        for k, v in batch_collated["input_dict"].items():
            v = v.to(device)
            if v.is_floating_point():
                v = v.to(dtype=model_dtype)
            input_dict[k] = v
        attn_mask = batch_collated["attention_mask"].to(device)
        
        with torch.no_grad():
            with torch.autocast(device_type=device_type, dtype=model_dtype):
                outputs = _REAL_MODEL(input_dict=input_dict, attention_mask=attn_mask, output_hidden_states=True)
                
                # Apply monotonicity logic to logits
                is_monotonic = getattr(_REAL_MODEL.config, "enforce_monotonicity", False)
                is_multilabel = getattr(_REAL_MODEL.config, "problem_type", None) == "multi_label_classification"
                logits = outputs.logits
                if is_monotonic and is_multilabel:
                    if logits.shape[-1] > 1:
                        monotonic_logits = [logits[:, 0]]
                        for i in range(1, logits.shape[-1]):
                            next_logit = monotonic_logits[-1] + torch.nn.functional.softplus(logits[:, i])
                            monotonic_logits.append(next_logit)
                        logits = torch.stack(monotonic_logits, dim=1)
                        
                all_logits.append(logits.float().cpu().numpy())
                
                # Average pool embeddings
                last_hidden = outputs.hidden_states[-1]
                mask = attn_mask.unsqueeze(-1)
                sum_embeddings = (last_hidden * mask).sum(dim=1)
                sum_mask = mask.sum(dim=1).clamp(min=1e-9)
                pooled = sum_embeddings / sum_mask
                all_embeddings.append(pooled.float().cpu().numpy())
                
    raw_logits_all = np.vstack(all_logits)  # (n_patients, num_horizons)
    
    # Compute cohort-wide logit statistics (median and standard deviation) for Z-score normalization
    _REAL_LOGITS_MEDIAN = np.median(raw_logits_all, axis=0)
    _REAL_LOGITS_STD    = np.std(raw_logits_all, axis=0)
    std_clamped = np.clip(_REAL_LOGITS_STD, a_min=1e-6, a_max=None)
    
    # Compute normalized cohort "probs" representing Infection Score (sigmoid of normalized logits)
    probs = np.zeros_like(raw_logits_all)
    for idx_h in range(raw_logits_all.shape[1]):
        probs[:, idx_h] = expit((raw_logits_all[:, idx_h] - _REAL_LOGITS_MEDIAN[idx_h]) / std_clamped[idx_h])
            
    embeddings = np.vstack(all_embeddings)  # (n_patients, hidden_dim)

    # Fit UMAP
    print("[AIIDKIT] Fitting UMAP on real cohort embeddings...")
    _REAL_UMAP_REDUCER = umap.UMAP(n_components=2, random_state=42)
    coords_2d = _REAL_UMAP_REDUCER.fit_transform(embeddings)

    cluster_colors = {0: "#FF4757", 1: "#00C9A7"}
    cluster_names = {0: "High risk", 1: "Low risk"}

    horizon_idx = CONFIG.AVAILABLE_HORIZONS.index(CONFIG.DEFAULT_HORIZON)

    # Run clustering depending on CONFIG.CLUSTERING_METHOD
    if getattr(CONFIG, "CLUSTERING_METHOD", "model_risk") == "clusterer":
        print("[AIIDKIT] Running HDBSCAN clustering...")
        
        # Reduce embeddings specifically for clustering (independent of visualization UMAP)
        n_comp = getattr(CONFIG, "CLUSTERING_UMAP_COMPONENTS", 15)
        if n_comp > 0 and n_comp < embeddings.shape[1]:
            print(f"[AIIDKIT] Reducing embeddings to {n_comp} dimensions via UMAP before clustering...")
            reducer = umap.UMAP(n_components=n_comp, random_state=42)
            cluster_input = reducer.fit_transform(embeddings)
        else:
            cluster_input = coords_2d  # fallback to 2D UMAP coordinates
            
        clusterer = hdbscan.HDBSCAN(min_cluster_size=15, min_samples=5)
        cluster_labels = clusterer.fit_predict(cluster_input)
        
        unique_labels = [l for l in np.unique(cluster_labels) if l != -1]
        label_map = {}
        for l in unique_labels:
            mask_l = cluster_labels == l
            mean_risk = float(probs[mask_l, horizon_idx].mean())
            if mean_risk >= CONFIG.RISK_THRESHOLD:
                label_map[l] = 0
            else:
                label_map[l] = 1
        
        ui_cluster_labels = []
        for i, l in enumerate(cluster_labels):
            if l == -1:
                ui_cluster_labels.append(0 if probs[i, horizon_idx] >= CONFIG.RISK_THRESHOLD else 1)
            else:
                ui_cluster_labels.append(label_map[l])
    else:
        # model_risk mode: partition cohort purely based on risk threshold
        print("[AIIDKIT] Clustering defined by model risk score threshold...")
        ui_cluster_labels = [0 if probs[i, horizon_idx] >= CONFIG.RISK_THRESHOLD else 1 for i in range(n_patients)]

    # Generate clinical summaries
    id2vocab = {v: k for k, v in _REAL_VOCAB.items()}
    patient_summaries = []
    for i in range(n_patients):
        ents = [id2vocab.get(id_val, "") for id_val in processed["entity_id"][i] if id_val != 0]
        attrs = [id2vocab.get(id_val, "") for id_val in processed["attribute_id"][i] if id_val != 0]
        vals = [id2vocab.get(id_val, "") for id_val in processed["value_id"][i] if id_val != 0]
        
        summary_terms = []
        for ent, attr, val in zip(ents, attrs, vals):
            if ent == "Medication" and attr == "Tacrolimus" and val in ["High", "Higher", "Highest"]:
                summary_terms.append("Tacrolimus↑")
            elif ent == "Lab result" and attr == "WBC" and val in ["Lowest", "Lower"]:
                summary_terms.append("Leucopenia")
            elif ent == "Infection" and val == "1":
                clean_attr = attr.replace(" infection", "").replace("Infection", "")
                summary_terms.append(f"{clean_attr}+" if clean_attr else "Infection+")
            elif ent == "Clinical event" and attr == "Rejection event" and val == "1":
                summary_terms.append("Rejection")
            elif ent == "Comorbidity" and attr == "Diabetes mellitus" and val == "1":
                summary_terms.append("Diabetes")
            elif ent == "Comorbidity" and attr == "Hypertension" and val == "1":
                summary_terms.append("Hypertension")
            elif ent == "Lab result" and attr == "Creatinine" and val in ["High", "Higher", "Highest"]:
                summary_terms.append("Creatinine↑")
            elif ent == "Lab result" and attr == "eGFR" and val in ["Lowest", "Lower"]:
                summary_terms.append("Low eGFR")
                
        summary_terms = list(dict.fromkeys(summary_terms))
        if not summary_terms:
            summary_terms = ["Standard profile"]
        patient_summaries.append(", ".join(summary_terms[:3]))

    # Generate cohort data for each horizon dynamically
    horizons_data = {}
    
    # Calculate global features frequencies in high (0) and low (2) clusters to construct global attributions
    high_counts = defaultdict(int)
    low_counts = defaultdict(int)
    all_features = set()
    n_high = sum(1 for label in ui_cluster_labels if label == 0)
    n_low = sum(1 for label in ui_cluster_labels if label == 1)
    
    for i in range(n_patients):
        c_id = ui_cluster_labels[i]
        ents = [id2vocab.get(id_val, "") for id_val in processed["entity_id"][i] if id_val != 0]
        attrs = [id2vocab.get(id_val, "") for id_val in processed["attribute_id"][i] if id_val != 0]
        vals = [id2vocab.get(id_val, "") for id_val in processed["value_id"][i] if id_val != 0]
        
        seen_pat_features = set()
        for ent, attr, val in zip(ents, attrs, vals):
            if ent in ["Medication", "Lab result", "Infection"]:
                feat = (ent, attr, val)
                if feat not in seen_pat_features:
                    seen_pat_features.add(feat)
                    all_features.add(feat)
                    if c_id == 0:
                        high_counts[feat] += 1
                    elif c_id == 1:
                        low_counts[feat] += 1
                        
    global_attrs = []
    for (ent, attr, val) in all_features:
        f_high = (high_counts[(ent, attr, val)] / n_high) if n_high > 0 else 0.0
        f_low = (low_counts[(ent, attr, val)] / n_low) if n_low > 0 else 0.0
        score = (f_high - f_low) * 0.2
        if abs(score) > 0.01:
            global_attrs.append({
                "feature": f"{ent} - {attr}",
                "value": val,
                "score": round(score, 4)
            })
            
    global_attributions = sorted(global_attrs, key=lambda x: abs(x["score"]), reverse=True)[:15]

    for idx_h, h in enumerate([30, 60, 90]):
        h_patients = []
        for i in range(n_patients):
            c_id = ui_cluster_labels[i]
            patient_risk = float(probs[i, idx_h])
            # true label for this specific horizon (more events will occur at longer horizons)
            t_lbl = int(subset_ds[i].get(f"label_{CONFIG.TASK}_{h:04d}d", 0) == 1)
            
            days_to_event, event_imminence = _get_event_info_from_row(subset_ds[i], CONFIG.TASK)
            
            h_patients.append({
                "id":            patient_ids[i],
                "umap_x":        round(float(coords_2d[i, 0]), 3),
                "umap_y":        round(float(coords_2d[i, 1]), 3),
                "risk_score":    round(patient_risk, 4),
                "calibrated_risk": round(float(expit(raw_logits_all[i, idx_h])), 4),
                "cluster":       c_id,
                "cluster_name":  cluster_names[c_id],
                "cluster_color": cluster_colors[c_id],
                "true_label":    t_lbl,
                "summary":       patient_summaries[i],
                "days_to_event": days_to_event,
                "event_imminence": round(event_imminence, 4),
            })
            
        rv = probs[:, idx_h]
        risk_distribution = {
            "values":      [round(float(x), 4) for x in rv],
            "mean":        round(float(rv.mean()), 4),
            "std":         round(float(rv.std()),  4),
            "percentiles": {
                "p25": round(float(np.percentile(rv, 25)), 4),
                "p50": round(float(np.percentile(rv, 50)), 4),
                "p75": round(float(np.percentile(rv, 75)), 4),
                "p90": round(float(np.percentile(rv, 90)), 4),
            },
        }

        # Count patient-level features in each cluster for prevalence calculations
        cluster_sizes = {0: 0, 1: 0}
        for label in ui_cluster_labels:
            cluster_sizes[label] = cluster_sizes.get(label, 0) + 1
            
        static_counts = {0: defaultdict(int), 1: defaultdict(int)}
        recent_counts = {0: defaultdict(int), 1: defaultdict(int)}
        all_static_features = set()
        all_recent_features = set()
        
        for i in range(n_patients):
            c_id = ui_cluster_labels[i]
            patient_static = set()
            patient_recent = set()
            
            # Find the max time in the sequence (excluding padding)
            seq_len = len(processed["entity_id"][i])
            valid_times = []
            for idx in range(seq_len):
                ent_id = processed["entity_id"][i][idx]
                attr_id = processed["attribute_id"][i][idx]
                val_id = processed["value_id"][i][idx]
                if ent_id == 0 or attr_id == 0 or val_id == 0:
                    continue
                valid_times.append(float(processed["days_since_tpx"][i][idx]))
            
            max_t = max(valid_times) if valid_times else 0.0
            
            for idx in range(seq_len):
                ent_id = processed["entity_id"][i][idx]
                attr_id = processed["attribute_id"][i][idx]
                val_id = processed["value_id"][i][idx]
                t_val = float(processed["days_since_tpx"][i][idx])
                
                if ent_id == 0 or attr_id == 0 or val_id == 0:
                    continue
                    
                ent = id2vocab.get(ent_id, "")
                attr = id2vocab.get(attr_id, "")
                val = id2vocab.get(val_id, "")
                
                if not ent or not attr or not val:
                    continue
                if attr in ["Transplantation event", "Transplant procedure"]:
                    continue
                    
                feat = (ent, attr, val)
                
                # Check if static: Patient/Donor entity or time <= 0.0
                is_static = (ent in ["Patient", "Donor"] or t_val <= 0.0)
                # Check if recent: not static and occurred within the last month of the observation window
                is_recent = (not is_static and t_val > 0.0 and t_val >= (max_t - 30.0))
                
                if is_static:
                    patient_static.add(feat)
                elif is_recent:
                    patient_recent.add(feat)
                    
            for feat in patient_static:
                static_counts[c_id][feat] += 1
                all_static_features.add(feat)
            for feat in patient_recent:
                recent_counts[c_id][feat] += 1
                all_recent_features.add(feat)

        cluster_profiles = {}
        for c_id in [0, 1]:
            other_c_id = 1 - c_id
            c_size = cluster_sizes.get(c_id, 0)
            other_size = cluster_sizes.get(other_c_id, 0)
            
            c_labels = [h_patients[i]["true_label"] for i in range(n_patients) if ui_cluster_labels[i] == c_id]
            event_rate = float(np.mean(c_labels)) if c_labels else 0.0
            
            # Static diffs
            static_diffs = []
            for feat in all_static_features:
                p_c = (static_counts[c_id][feat] / c_size) if c_size > 0 else 0.0
                p_other = (static_counts[other_c_id][feat] / other_size) if other_size > 0 else 0.0
                diff = p_c - p_other
                if diff > 0.0:
                    static_diffs.append((feat, diff))
            sorted_static = sorted(static_diffs, key=lambda x: x[1], reverse=True)[:10]
            top_static = []
            for (ent, attr, val), diff in sorted_static:
                score = diff if c_id == 0 else -diff
                top_static.append({
                    "feature": f"{ent} - {attr}",
                    "value": val,
                    "score": score
                })

            # Recent diffs
            recent_diffs = []
            for feat in all_recent_features:
                p_c = (recent_counts[c_id][feat] / c_size) if c_size > 0 else 0.0
                p_other = (recent_counts[other_c_id][feat] / other_size) if other_size > 0 else 0.0
                diff = p_c - p_other
                if diff > 0.0:
                    recent_diffs.append((feat, diff))
            sorted_recent = sorted(recent_diffs, key=lambda x: x[1], reverse=True)[:10]
            top_recent = []
            for (ent, attr, val), diff in sorted_recent:
                score = diff if c_id == 0 else -diff
                top_recent.append({
                    "feature": f"{ent} - {attr}",
                    "value": val,
                    "score": score
                })
                
            cluster_profiles[str(c_id)] = {
                "name":        cluster_names[c_id],
                "n_patients":  c_size,
                "event_rate":  round(event_rate, 4),
                "color":       cluster_colors[c_id],
                "top_features": {
                    "static": top_static,
                    "recent": top_recent
                },
            }
            
        horizons_data[str(h)] = {
            "patients": h_patients,
            "risk_distribution": risk_distribution,
            "cluster_profiles": cluster_profiles,
        }

    return {
        "horizons":          horizons_data,
        "global_attributions": global_attributions,
    }






def _load_real_model(paths: dict):
    """
    Load the actual trained transformer model for inference.
    Called once at startup when CONFIG.MODEL_PATH is set.
    """
    global _REAL_MODEL, _REAL_COLLATOR, _REAL_VOCAB, _REAL_BIN_INT, _COHORT_DATA
    import pickle, torch
    from src.model.patient_embedder import (
        PatientEmbeddingModelFactory,
        PatientDataCollatorForClassification,
    )
    from src.data.patient_dataset import BIN_LABELS

    vocab_path = paths["vocab_path"]
    bin_int_path = paths["bin_intervals"]

    with open(vocab_path, "rb") as f:
        _REAL_VOCAB = pickle.load(f)
    with open(bin_int_path, "rb") as f:
        _REAL_BIN_INT = pickle.load(f)

    device = "cuda" if (paths["use_gpu"] and torch.cuda.is_available()) else "cpu"

    emb_cfg = {
        "vocab_size":    len(_REAL_VOCAB),
        "eav_mappings":  CONFIG.EAV_MAPPINGS,   # dict  e.g. {"entity_id": "entity", ...}
        "time_mapping":  CONFIG.TIME_MAPPING,   # dict  e.g. {"days_since_tpx": "time"}
    }
    model_args = {
        "num_labels":   len(CONFIG.AVAILABLE_HORIZONS),
        "problem_type": "multi_label_classification",
    }
    _REAL_MODEL = PatientEmbeddingModelFactory.from_pretrained(
        task="classification",
        pretrained_dir=str(paths["model_path"]),
        embedding_layer_config=emb_cfg,
        model_args=model_args,
        enforce_monotonicity=True,
    )
    _REAL_MODEL.eval()
    _REAL_MODEL.to(device)

    _REAL_COLLATOR = PatientDataCollatorForClassification(
        eav_mappings=CONFIG.EAV_MAPPINGS,
        time_mapping=CONFIG.TIME_MAPPING,
        label_keys=[
            f"label_{CONFIG.TASK}_{h:04d}d"
            for h in CONFIG.AVAILABLE_HORIZONS
        ],
        max_position_embeddings=1024,
    )
    print(f"[AIIDKIT] Real model loaded from {paths['model_path']} on {device}")



    # Generate real cohort data
    try:
        real_cohort = _generate_real_cohort_data(paths["test_dir"])
        _COHORT_CACHE[CONFIG.DEFAULT_FUP] = real_cohort
        global _COHORT_DATA
        _COHORT_DATA = real_cohort
        print(f"[AIIDKIT] Real cohort data successfully generated from {paths['test_dir']}")
    except Exception as exc:
        print(f"[AIIDKIT] WARNING: Could not generate real cohort data: {exc}")
        import traceback; traceback.print_exc()
        print("[AIIDKIT] Falling back to mock cohort data.")


def _deterministic_seed(events: list[dict]) -> int:
    import hashlib
    # Sort events to ensure order-independence
    sorted_events = sorted(
        events,
        key=lambda e: (
            float(e.get("days_since_tpx", 0.0)),
            str(e.get("entity", "")),
            str(e.get("attribute", "")),
            str(e.get("value", ""))
        )
    )
    event_str = "".join(
        f"{e.get('entity')}-{e.get('attribute')}-{e.get('value')}-{e.get('days_since_tpx')}"
        for e in sorted_events
    )
    return int(hashlib.md5(event_str.encode("utf-8")).hexdigest(), 16) & 0xffffffff


def _real_predict(events: list[dict], horizon: int, cohort: dict) -> dict:
    """
    Run inference with the actual trained model.
    Converts the raw EAV event list to model input, runs forward pass,
    computes attributions via Input * Gradient, and returns prediction dict.
    """
    import torch, numpy as np
    from src.data.patient_dataset import BIN_LABELS, preprocess_batch
    from scipy.special import expit

    device = next(_REAL_MODEL.parameters()).device
    model_dtype = next(_REAL_MODEL.parameters()).dtype
    device_type = "cuda" if "cuda" in str(device) else "cpu"

    # Preprocess events into token IDs
    batch_raw = {
        "entity":    [e["entity"]    for e in events],
        "attribute": [e["attribute"] for e in events],
        "value":     [e["value"]     for e in events],
        "time":      [e.get("days_since_tpx", 0.0) for e in events],
    }
    processed = preprocess_batch(
        {k: [v] for k, v in batch_raw.items()},
        vocab=_REAL_VOCAB,
        bin_intervals=_REAL_BIN_INT,
        bin_labels=BIN_LABELS,
        time_mapping=CONFIG.TIME_MAPPING,
        eav_mappings=CONFIG.EAV_MAPPINGS,
    )

    sample = {
        "entity_id":      np.array(processed["entity_id"][0]),
        "attribute_id":   np.array(processed["attribute_id"][0]),
        "value_id":       np.array(processed["value_id"][0]),
        "days_since_tpx": np.array(processed["days_since_tpx"][0]),
        "split":          "test",   # required by collator to disable augmentation
    }

    # Add dummy label keys to satisfy the classification collator's expectation
    for k in _REAL_COLLATOR.label_keys:
        sample[k] = 0.0

    # Collate to batch of 1
    batch = _REAL_COLLATOR.torch_call([sample])
    
    # Cast tensors to correct device and model parameter dtype (e.g. bfloat16/float16 for float inputs)
    input_dict = {}
    for k, v in batch["input_dict"].items():
        v = v.to(device)
        if v.is_floating_point():
            v = v.to(dtype=model_dtype)
        input_dict[k] = v
        
    attn_mask = batch["attention_mask"].to(device)

    is_monotonic = getattr(_REAL_MODEL.config, "enforce_monotonicity", False)
    is_multilabel = getattr(_REAL_MODEL.config, "problem_type", None) == "multi_label_classification"
    horizon_idx = CONFIG.AVAILABLE_HORIZONS.index(horizon)

    # 1. Run clean forward pass (without gradient tracking) to compute predictions and hidden states
    with torch.no_grad():
        with torch.autocast(device_type=device_type, dtype=model_dtype):
            outputs = _REAL_MODEL(input_dict=input_dict, attention_mask=attn_mask, output_hidden_states=True)
            logits = outputs.logits
            
            # Apply monotonicity logic to logits
            if is_monotonic and is_multilabel:
                if logits.shape[-1] > 1:
                    monotonic_logits = [logits[:, 0]]
                    for idx in range(1, logits.shape[-1]):
                        next_logit = monotonic_logits[-1] + torch.nn.functional.softplus(logits[:, idx])
                        monotonic_logits.append(next_logit)
                    logits = torch.stack(monotonic_logits, dim=1)

            pass

    # 2. Compute feature attribution scores
    attr_scores = None

    # Try utilizing Captum's LayerIntegratedGradients
    try:
        from captum.attr import LayerIntegratedGradients

        class ForwardWrapperForCaptum(torch.nn.Module):
            def __init__(self, model, pad_id: int = 0, enforce_monotonicity: bool = False, is_multilabel: bool = False, device_type="cpu", model_dtype=torch.float32):
                super().__init__()
                self.model = model
                self.pad_id = pad_id
                self.enforce_monotonicity = enforce_monotonicity
                self.is_multilabel = is_multilabel
                self.device_type = device_type
                self.model_dtype = model_dtype
            
            def forward(self, entity_id, attribute_id, value_id, days_since_tpx):
                input_dict_c = {
                    "entity_id": entity_id,
                    "attribute_id": attribute_id,
                    "value_id": value_id,
                    "days_since_tpx": days_since_tpx
                }
                attention_mask = (entity_id != self.pad_id).long()            
                
                with torch.autocast(device_type=self.device_type, dtype=self.model_dtype):
                    outputs_c = self.model(input_dict=input_dict_c, attention_mask=attention_mask)
                    logits_c = outputs_c.logits
                    
                    # Apply monotonicity logic to logits
                    if self.enforce_monotonicity and self.is_multilabel:
                        if logits_c.shape[-1] > 1:
                            monotonic_logits = [logits_c[:, 0]]
                            for idx in range(1, logits_c.shape[-1]):
                                next_logit = monotonic_logits[-1] + torch.nn.functional.softplus(logits_c[:, idx])
                                monotonic_logits.append(next_logit)
                            logits_c = torch.stack(monotonic_logits, dim=1)

                    pass
                    
                return logits_c.float()

        wrapper = ForwardWrapperForCaptum(
            _REAL_MODEL,
            pad_id=0,
            enforce_monotonicity=is_monotonic,
            is_multilabel=is_multilabel,

            device_type=device_type,
            model_dtype=model_dtype
        ).to(device)

        ent = input_dict["entity_id"]
        attr = input_dict["attribute_id"]
        val = input_dict["value_id"]
        days = input_dict["days_since_tpx"]
        args = (ent, attr, val, days)

        baseline_ent = torch.zeros_like(ent, device=device)
        baseline_attr = torch.zeros_like(attr, device=device)
        baseline_val = torch.zeros_like(val, device=device)
        baseline_days = torch.zeros_like(days, device=device)
        baseline_args = (baseline_ent, baseline_attr, baseline_val, baseline_days)

        target_layer = _REAL_MODEL.patient_embedder.value_embedding_hook
        lig = LayerIntegratedGradients(wrapper, target_layer)

        print("[AIIDKIT] Calculating attributions via Captum LayerIntegratedGradients...")
        with torch.inference_mode(False):
            attributions, delta = lig.attribute(
                inputs=args,
                baselines=baseline_args,
                target=horizon_idx,
                n_steps=100,
                return_convergence_delta=True,
                internal_batch_size=1,
            )
        
        attrs_sum = attributions.sum(dim=-1).squeeze(0)
        attr_scores = attrs_sum.detach().float().cpu().numpy()

    except Exception as exc:
        print(f"[AIIDKIT] Captum LayerIntegratedGradients failed or not available: {exc}")
        print("[AIIDKIT] Falling back to manual Input * Gradient approximation...")
        
        # Fallback to Input * Gradient
        with torch.set_grad_enabled(True):
            with torch.autocast(device_type=device_type, dtype=model_dtype):
                inputs_embeds = _REAL_MODEL.patient_embedder(**input_dict)
                inputs_embeds.requires_grad_()
                
                outputs_g = _REAL_MODEL.original_forward(inputs_embeds=inputs_embeds, attention_mask=attn_mask)
                logits_g = outputs_g.logits
                if is_monotonic and is_multilabel:
                    if logits_g.shape[-1] > 1:
                        monotonic_logits = [logits_g[:, 0]]
                        for idx in range(1, logits_g.shape[-1]):
                            next_logit = monotonic_logits[-1] + torch.nn.functional.softplus(logits_g[:, idx])
                            monotonic_logits.append(next_logit)
                        logits_g = torch.stack(monotonic_logits, dim=1)

                pass

                logit_g = logits_g[0, horizon_idx]
                
            grads = torch.autograd.grad(logit_g, inputs_embeds)[0]
            
        input_grad_attr = (inputs_embeds * grads).sum(dim=-1).squeeze(0)
        attr_scores = input_grad_attr.detach().cpu().numpy()

    # Scale attribution scores proportionally so that the maximum absolute value is 0.2
    # This highlights relative event contributions in the clinical timeline
    max_abs_attr = np.max(np.abs(attr_scores))
    if max_abs_attr > 1e-8:
        attr_scores = attr_scores * (0.2 / max_abs_attr)

    # Extract risk predictions
    logits_np = logits.detach().float().cpu().numpy()[0]
    
    # Event score (Infection score) is the Z-score normalized logit run through sigmoid
    std_clamped = np.clip(_REAL_LOGITS_STD, a_min=1e-6, a_max=None)
    event_scores = expit((logits_np - _REAL_LOGITS_MEDIAN) / std_clamped)
    
    # Calibrated probability (Actual risk) is the raw sigmoid model probability
    calibrated_probs = expit(logits_np)
            
    selected_risk = float(np.clip(event_scores[horizon_idx], 0.01, 0.99))
    selected_calibrated_risk = float(np.clip(calibrated_probs[horizon_idx], 0.01, 0.99))

    risk_scores = {
        f"{h}d": round(float(np.clip(event_scores[i], 0.01, 0.99)), 4)
        for i, h in enumerate(CONFIG.AVAILABLE_HORIZONS)
    }
    
    calibrated_risks = {
        f"{h}d": round(float(np.clip(calibrated_probs[i], 0.01, 0.99)), 4)
        for i, h in enumerate(CONFIG.AVAILABLE_HORIZONS)
    }

    from mock_data import _risk_category, CLUSTER_PROFILES
    category, color = _risk_category(selected_risk)

    # UMAP projection
    umap_x, umap_y = 0.0, 0.0
    # Assign cluster based on risk threshold from config
    if selected_risk >= CONFIG.RISK_THRESHOLD:
        best_cluster = CLUSTER_PROFILES[0]
    else:
        best_cluster = CLUSTER_PROFILES[1]
    
    if _REAL_UMAP_REDUCER is not None:
        try:
            # Get the pooled embedding of this new patient sequence
            last_hidden = outputs.hidden_states[-1] if hasattr(outputs, "hidden_states") else None
            if last_hidden is None:
                last_hidden = inputs_embeds # approximation
            mask = attn_mask.unsqueeze(-1)
            sum_embeddings = (last_hidden * mask).sum(dim=1)
            sum_mask = mask.sum(dim=1).clamp(min=1e-9)
            pooled = sum_embeddings / sum_mask
            new_emb = pooled.detach().float().cpu().numpy()
            
            new_coords = _REAL_UMAP_REDUCER.transform(new_emb)[0]
            umap_x = round(float(new_coords[0]), 3)
            umap_y = round(float(new_coords[1]), 3)
        except Exception as exc:
            print(f"[AIIDKIT] WARNING: UMAP projection failed: {exc}")
            import random
            seed_val = _deterministic_seed(events)
            local_random = random.Random(seed_val)
            cx, cy = best_cluster["umap_center"]
            umap_x = round(cx + local_random.gauss(0, 0.3), 3)
            umap_y = round(cy + local_random.gauss(0, 0.3), 3)
    else:
        import random
        seed_val = _deterministic_seed(events)
        local_random = random.Random(seed_val)
        cx, cy = best_cluster["umap_center"]
        umap_x = round(cx + local_random.gauss(0, 0.3), 3)
        umap_y = round(cy + local_random.gauss(0, 0.3), 3)

    # Find the cluster of the patient in the cohort
    assigned_cluster_id = best_cluster["id"]
    assigned_cluster_name = best_cluster["name"]
    assigned_cluster_color = best_cluster["color"]
    
    # Calculate similar patients and cohort percentile
    active_cohort = cohort["horizons"][str(horizon)]
    cohort_risks = np.array([p["risk_score"] for p in active_cohort["patients"]])
    percentile = int(np.round((cohort_risks < selected_risk).mean() * 100))
    
    similar_patients = []
    try:
        # Distance in UMAP space to cohort patients
        dists = []
        for p in active_cohort["patients"]:
            d = (p["umap_x"] - umap_x)**2 + (p["umap_y"] - umap_y)**2
            dists.append((d, p))
        dists = sorted(dists, key=lambda x: x[0])
        for d, p in dists[:3]:
            similar_patients.append({
                "id": p["id"],
                "risk_score": p["risk_score"],
                "summary": p["summary"],
            })
            
        # If similar patients found, update assigned cluster in clusterer mode
        if dists and getattr(CONFIG, "CLUSTERING_METHOD", "model_risk") == "clusterer":
            nearest = dists[0][1]
            assigned_cluster_id = nearest["cluster"]
            assigned_cluster_name = nearest["cluster_name"]
            assigned_cluster_color = nearest["cluster_color"]
    except Exception:
        pass

    # Map back to event fields for feature attributions
    attributions = []
    events_with_scores = []
    
    for i, e in enumerate(events):
        score = float(attr_scores[i + 1]) if (i + 1) < len(attr_scores) else 0.0
        feat_name = f"{e['entity']} - {e['attribute']}"
        attributions.append({
            "feature": feat_name,
            "value": str(e["value"]),
            "score": round(score, 4),
        })
        events_with_scores.append({
            **e,
            "score": round(score, 4),
        })
        
    attributions = sorted(attributions, key=lambda x: abs(x["score"]), reverse=True)[:15]

    # Generate detailed narrative based on attributions and assigned cluster
    pos_feats = [a for a in attributions if a["score"] > 0.0]
    neg_feats = [a for a in attributions if a["score"] < 0.0]

    risk_txt = (
        f"This patient's predicted {horizon}-day infection score is "
        f"{selected_risk*100:.1f}/100 ({category}), placing them at the "
        f"{percentile}th percentile of the reference cohort."
    )
    if pos_feats:
        names = ", ".join(
            f"{f['feature'].replace('Infection', 'Previous infection')}: {f['value']}" for f in pos_feats[:3]
        )
        risk_txt += f" Key risk drivers: {names}."
    if neg_feats:
        names = ", ".join(
            f"{f['feature'].replace('Infection', 'Previous infection')}: {f['value']}" for f in neg_feats[:2]
        )
        risk_txt += f" Protective factors present: {names}."

    matched_profile = next((p for p in CLUSTER_PROFILES if str(p["id"]) == str(assigned_cluster_id)), best_cluster)
    event_rate_pct = int(matched_profile.get("event_rate", 0) * 100)
    risk_txt += (
        f" The patient clusters with the '{assigned_cluster_name}' phenotype "
        f"(event rate ≈ {event_rate_pct}%)."
    )

    return {
        "risk_scores":       risk_scores,
        "calibrated_risks":  calibrated_risks,
        "selected_horizon":  horizon,
        "risk_score":        selected_risk,
        "calibrated_risk":   selected_calibrated_risk,
        "risk_category":     category,
        "risk_color":        color,
        "percentile":        percentile,
        "cluster":           assigned_cluster_id,
        "cluster_name":      assigned_cluster_name,
        "cluster_color":     assigned_cluster_color,
        "umap_x":            umap_x,
        "umap_y":            umap_y,
        "days_to_event":     None,
        "event_imminence":   0.0,
        "attributions":      attributions,
        "events_with_scores":events_with_scores,
        "similar_patients":  similar_patients,
        "narrative":         risk_txt,
    }


# ---------------------------------------------------------------------------
# App initialisation
# ---------------------------------------------------------------------------

app = Flask(__name__)
CORS(app)

# ---------------------------------------------------------------------------
# Startup: validate paths, load model or fall back to mock
# ---------------------------------------------------------------------------
_PATHS = CONFIG.get_resolved_paths()

# ---------------------------------------------------------------------------
# Dynamically enrich vocabulary from synthetic patients if present
# ---------------------------------------------------------------------------
def _enrich_vocabulary_from_synthetic(vocab):
    import csv
    base_dir = Path(__file__).resolve().parent
    synthetic_dir = base_dir / "synthetic_patients"
    if not synthetic_dir.exists():
        return vocab
        
    filenames = [
        "low_risk.csv",
        "medium_risk.csv",
        "high_risk.csv"
    ]
    
    enriched = dict(vocab)
    for filename in filenames:
        file_path = synthetic_dir / filename
        if not file_path.exists():
            continue
        try:
            with open(file_path, mode="r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    entity = row.get("entity")
                    attribute = row.get("attribute")
                    val = row.get("value_binned")
                    if entity and attribute and val is not None:
                        # Ensure string keys/values
                        entity = str(entity).strip()
                        attribute = str(attribute).strip()
                        val = str(val).strip()
                        if not entity or not attribute:
                            continue
                        if entity not in enriched:
                            enriched[entity] = {}
                        if attribute not in enriched[entity]:
                            enriched[entity][attribute] = []
                        if val not in enriched[entity][attribute]:
                            enriched[entity][attribute].append(val)
        except Exception as e:
            print(f"[AIIDKIT] Failed to read {filename} for vocabulary enrichment: {e}")
            
    # Sort values for consistency
    ordinals = ["Lowest", "Lower", "Low", "Middle", "High", "Higher", "Highest"]
    for entity in enriched:
        for attribute in enriched[entity]:
            vals = enriched[entity][attribute]
            try:
                # check if all values can be numeric
                if all(v.replace('.', '', 1).isdigit() or (v.startswith('-') and v[1:].replace('.', '', 1).isdigit()) for v in vals):
                    enriched[entity][attribute] = sorted(vals, key=float)
                    continue
            except Exception:
                pass
                
            # Sort with ordinals custom logic
            if any(o in vals for o in ordinals):
                enriched[entity][attribute] = sorted(vals, key=lambda x: ordinals.index(x) if x in ordinals else 99)
            else:
                enriched[entity][attribute] = sorted(vals)
                
    return enriched

# Cache for pre-generated cohort data payloads (key: FUP, value: cohort dict)
_COHORT_CACHE = {}
_COHORT_DATA = None
_USING_REAL_MODEL = False

# Avoid running dynamic vocabulary enrichment and model/cohort loading twice in Flask's debug reloader
if not CONFIG.DEBUG or os.environ.get("WERKZEUG_RUN_MAIN") == "true":
    print("[AIIDKIT] Dynamic vocabulary enrichment...")
    CONFIG.VOCABULARY = _enrich_vocabulary_from_synthetic(CONFIG.VOCABULARY)

    if not _PATHS["use_mock"]:
        try:
            _load_real_model(_PATHS)
            _USING_REAL_MODEL = True
            print("[AIIDKIT] LIVE MODEL mode active.")
        except Exception as exc:
            print(f"[AIIDKIT] WARNING: Could not load real model: {exc}")
            print("[AIIDKIT] Falling back to DEMO MODE.")
    else:
        print("[AIIDKIT] DEMO MODE — edit MODEL_PATH / DATA_DIR in config.py to use real model.")

    if not _USING_REAL_MODEL:
        print(f"[AIIDKIT] Pre-generating default FUP {CONFIG.DEFAULT_FUP} mock cohort data …")
        _COHORT_CACHE[CONFIG.DEFAULT_FUP] = generate_cohort_data(fup=CONFIG.DEFAULT_FUP, seed=CONFIG.MOCK_RANDOM_SEED)
        _COHORT_DATA = _COHORT_CACHE[CONFIG.DEFAULT_FUP]
        print("[AIIDKIT] DEMO MODE active.")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/config")
def api_config():
    """Return task configuration and clinical vocabulary."""
    return jsonify({
        "task":               CONFIG.TASK,
        "default_horizon":    CONFIG.DEFAULT_HORIZON,
        "available_horizons": CONFIG.AVAILABLE_HORIZONS,
        "default_fup":        CONFIG.DEFAULT_FUP,
        "available_fups":     CONFIG.AVAILABLE_FUPS,
        "using_mock_data":    not _USING_REAL_MODEL,
        "vocabulary":         CONFIG.VOCABULARY,
        "risk_threshold":     CONFIG.RISK_THRESHOLD,
    })


@app.route("/api/cohort")
def api_cohort():
    """Return pre-computed cohort data for the requested FUP."""
    fup_str = request.args.get("fup")
    try:
        fup = int(fup_str) if fup_str is not None else CONFIG.DEFAULT_FUP
    except ValueError:
        return jsonify({"error": "Invalid FUP value. Must be an integer."}), 400

    if fup not in CONFIG.AVAILABLE_FUPS:
        return jsonify({"error": f"FUP must be one of {CONFIG.AVAILABLE_FUPS}"}), 400

    # Retrieve from cache if already loaded
    if fup in _COHORT_CACHE:
        return jsonify(_COHORT_CACHE[fup])

    try:
        if _USING_REAL_MODEL:
            # Dynamically resolve directory for this FUP test dataset split
            fup_dir = _PATHS["data_dir"] / f"fup_{fup:04d}" / "test"
            if fup_dir.exists():
                print(f"[AIIDKIT] Generating real cohort data for FUP {fup} from {fup_dir}...")
                cohort = _generate_real_cohort_data(fup_dir)
                _COHORT_CACHE[fup] = cohort
                return jsonify(cohort)
            else:
                print(f"[AIIDKIT] WARNING: FUP {fup} directory {fup_dir} not found. Falling back to mock data.")

        # Fallback / mock mode
        print(f"[AIIDKIT] Generating mock cohort data for FUP {fup}...")
        cohort = generate_cohort_data(fup=fup, seed=CONFIG.MOCK_RANDOM_SEED)
        _COHORT_CACHE[fup] = cohort
        return jsonify(cohort)
    except Exception as exc:
        print(f"[AIIDKIT] Error generating cohort for FUP {fup}: {exc}")
        import traceback; traceback.print_exc()
        return jsonify({"error": f"Failed to generate cohort: {exc}"}), 500


@app.route("/api/examples/<example_type>")
def api_example_patient(example_type):
    """Return parsed event sequence from one of the synthetic patient CSV files."""
    mapping = {
        "low": "low_risk.csv",
        "mod": "medium_risk.csv",
        "medium": "medium_risk.csv",
        "high": "high_risk.csv",
    }
    
    filename = mapping.get(example_type.lower())
    if not filename:
        return jsonify({"error": f"Invalid example type. Choose from: {list(mapping.keys())}"}), 400
        
    base_dir = Path(__file__).resolve().parent
    file_path = base_dir / "synthetic_patients" / filename
    
    if not file_path.exists():
        return jsonify({"error": f"Example file {filename} not found."}), 404
        
    import csv
    events = []
    try:
        with open(file_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                days_str = row.get("days_since_tpx")
                entity = row.get("entity")
                attribute = row.get("attribute")
                value = row.get("value_binned")
                if days_str is not None and entity and attribute and value is not None:
                    try:
                        days = float(days_str)
                        if days.is_integer():
                            days = int(days)
                    except ValueError:
                        days = 0.0
                    events.append({
                        "entity": entity,
                        "attribute": attribute,
                        "value": value,
                        "days_since_tpx": days
                    })
        return jsonify(events)
    except Exception as exc:
        return jsonify({"error": f"Failed to load or parse example patient: {exc}"}), 500



@app.route("/api/predict", methods=["POST"])
def api_predict():
    """Accept a patient EAV sequence and return risk analysis."""
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Request body must be JSON"}), 400

    events  = data.get("events", [])
    horizon = int(data.get("horizon", CONFIG.DEFAULT_HORIZON))
    fup     = int(data.get("fup", CONFIG.DEFAULT_FUP))

    if not events:
        return jsonify({"error": "No events provided"}), 400

    if horizon not in CONFIG.AVAILABLE_HORIZONS:
        return jsonify({
            "error": f"Horizon must be one of {CONFIG.AVAILABLE_HORIZONS}"
        }), 400

    if fup not in CONFIG.AVAILABLE_FUPS:
        return jsonify({
            "error": f"FUP must be one of {CONFIG.AVAILABLE_FUPS}"
        }), 400

    # Validate event schema
    required_keys = {"entity", "attribute", "value", "days_since_tpx"}
    for i, ev in enumerate(events):
        missing = required_keys - set(ev.keys())
        if missing:
            return jsonify({
                "error": f"Event {i} missing fields: {missing}"
            }), 400
        try:
            ev["days_since_tpx"] = float(ev["days_since_tpx"])
        except (ValueError, TypeError):
            return jsonify({
                "error": f"Event {i}: days_since_tpx must be numeric"
            }), 400

    # Ensure the cohort is in our cache
    if fup not in _COHORT_CACHE:
        try:
            if _USING_REAL_MODEL:
                fup_dir = _PATHS["data_dir"] / f"fup_{fup:04d}" / "test"
                if fup_dir.exists():
                    _COHORT_CACHE[fup] = _generate_real_cohort_data(fup_dir)
                else:
                    _COHORT_CACHE[fup] = generate_cohort_data(fup=fup, seed=CONFIG.MOCK_RANDOM_SEED)
            else:
                _COHORT_CACHE[fup] = generate_cohort_data(fup=fup, seed=CONFIG.MOCK_RANDOM_SEED)
        except Exception as exc:
            print(f"[AIIDKIT] Failed to auto-generate cohort for FUP {fup} on predict: {exc}")
            # fallback to default
            _COHORT_CACHE[fup] = _COHORT_CACHE[CONFIG.DEFAULT_FUP]

    cohort = _COHORT_CACHE[fup]

    try:
        if _USING_REAL_MODEL:
            result = _real_predict(events, horizon, cohort)
        else:
            result = mock_predict(events, horizon, cohort)
        return jsonify(result)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Prediction failed: {exc}"}), 500


# ---------------------------------------------------------------------------
# Dev server
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(
        host=CONFIG.HOST,
        port=CONFIG.PORT,
        debug=CONFIG.DEBUG,
        exclude_patterns=getattr(CONFIG, "EXCLUDE_PATTERNS", None)
    )
