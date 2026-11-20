"""
AIIDKIT Decision Support Tool — Flask Backend
=============================================
Serves the single-page frontend and REST API endpoints:
  GET  /                       Serve index.html
  GET  /api/config             Return vocabulary, tasks, horizons, and labels
  GET  /api/cohort             Return pre-computed shaded cohort data
  GET  /api/examples/<type>    Return synthetic reference patient sequence
  POST /api/predict            Score patient t-EAV sequence + Integrated Gradients
"""

from collections import OrderedDict, defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path
import pickle
import random
import sys
import threading
import time

from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
import numpy as np
from scipy.special import expit

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import config as CONFIG
from mock_data import (
    STATIC_ENTITY_SET,
    build_clinical_narrative,
    generate_cohort_data,
    mock_predict,
)
from src.model.model_utils import build_monotonic_groups
from src.tasks import build_available_tasks_list, build_horizon_labels_map

# ---------------------------------------------------------------------------
# Runtime State & Thread-Safe Caches
# ---------------------------------------------------------------------------
_REAL_MODEL = None
_REAL_COLLATOR = None
_REAL_VOCAB = None
_REAL_BIN_INT = None
_REAL_UMAP_REDUCER = None
_REAL_GLOBAL_UMAP_BOUNDS = None
_REAL_LOGITS_MEDIAN = None
_REAL_LOGITS_STD = None
_REAL_TASK_COL_INDICES: list[int] = []
_REAL_TASK_COLS_MAP: dict[str, list[int]] = {}
_REAL_TASK_HORIZONS_MAP: dict[str, list[int]] = {}
_REAL_PLATT_PARAMS_BY_COL: dict[int, tuple[float, float]] = {}
_REAL_LOGITS_MEDIAN_BY_TASK: dict[str, np.ndarray] = {}
_REAL_LOGITS_STD_BY_TASK: dict[str, np.ndarray] = {}
_EMBEDDINGS_CACHE_BY_FUP: dict[int, dict] = {}

_MODEL_LOCK = threading.Lock()
_CACHE_LOCK = threading.Lock()
_PREDICTION_CACHE: OrderedDict = OrderedDict()
_TRAJECTORY_CACHE: OrderedDict = OrderedDict()
_IG_TASK_CACHE: OrderedDict = OrderedDict()
_COHORT_CACHE: dict[int, dict] = {}
_COHORT_DATA: dict | None = None
_USING_REAL_MODEL = False


def _lru_get(cache: OrderedDict, key):
    with _CACHE_LOCK:
        if key not in cache:
            return None
        cache.move_to_end(key)
        return cache[key]


def _lru_put(cache: OrderedDict, key, value, maxsize: int = 128):
    with _CACHE_LOCK:
        cache[key] = value
        cache.move_to_end(key)
        while len(cache) > maxsize:
            cache.popitem(last=False)


def _ensure_private_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        os.chmod(path, 0o700)
    except OSError:
        pass
    return path


def _chmod_private_file(path: Path) -> None:
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass


def _apply_platt_vector(logits_vec: np.ndarray, cols: list[int]) -> np.ndarray:
    """Applies validation-fitted L2 Platt scaling `(coef * logit + intercept)` per column."""
    out = np.empty(len(cols), dtype=np.float64)
    for i, col_idx in enumerate(cols):
        coef, intercept = _REAL_PLATT_PARAMS_BY_COL.get(int(col_idx), (1.0, 0.0))
        out[i] = expit(coef * float(logits_vec[i]) + intercept)
    return out


def _apply_platt_matrix(logits_mat: np.ndarray, cols: list[int]) -> np.ndarray:
    """Vectorized validation-fitted L2 Platt scaling across a `(N, H)` logit matrix."""
    coefs = np.array([_REAL_PLATT_PARAMS_BY_COL.get(int(c), (1.0, 0.0))[0] for c in cols], dtype=np.float64)
    intercepts = np.array([_REAL_PLATT_PARAMS_BY_COL.get(int(c), (1.0, 0.0))[1] for c in cols], dtype=np.float64)
    return expit(logits_mat * coefs[None, :] + intercepts[None, :])


def _get_event_info_from_row(row: dict, task_key: str) -> tuple[int | None, float]:
    """Approximates `days_to_event` and `event_imminence` from multi-horizon binary labels."""
    prefix = f"label_{task_key}_"
    horizons = []
    for k in row.keys():
        if k.startswith(prefix) and k.endswith("d"):
            try:
                horizons.append((int(k[len(prefix):-1]), k))
            except ValueError:
                pass
    if not horizons:
        return None, 0.0

    horizons.sort(key=lambda x: x[0])
    prev_h = 0
    for day, k in horizons:
        if row.get(k, 0) == 1:
            days_to_event = int((prev_h + day) / 2)
            return days_to_event, float(np.exp(-days_to_event / 365.0))
        prev_h = day
    return None, 0.0


def _pool_and_project_embeddings(outputs, attn_mask):
    """Pools sequence representations directly from the shared Transformer trunk (`outputs.hidden_states[-1]`),
    prior to any task-specific classification head or decision-layer projection, so patient geometry stays
    task-agnostic and comparable with pre-trained representations when switching across tasks.
    """
    import torch

    last_hidden = outputs.hidden_states[-1]
    pooling_mode = getattr(getattr(_REAL_MODEL, "config", None), "classifier_pooling", "mean")
    if pooling_mode == "last":
        seq_lens = attn_mask.sum(dim=1).clamp(min=1) - 1
        batch_idx = torch.arange(last_hidden.size(0), device=last_hidden.device)
        pooled = last_hidden[batch_idx, seq_lens]
    else:
        mask = attn_mask.unsqueeze(-1)
        pooled = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)

    return pooled.float()


def parse_clusters_into_risk_groups(cluster_rates: list) -> dict:
    """Partitions HDBSCAN clusters into binary High-risk (0) vs Low-risk (1) groups via Otsu/Fisher-Jenks."""
    if not cluster_rates:
        return {}

    normalized = [
        (item[0], float(item[1]), max(1, int(item[2])) if len(item) >= 3 else 1)
        for item in cluster_rates
    ]
    sorted_clusters = sorted(normalized, key=lambda x: x[1], reverse=True)
    k = len(sorted_clusters)
    if k == 1:
        return {sorted_clusters[0][0]: 0}
    if k == 2:
        return {sorted_clusters[0][0]: 0, sorted_clusters[1][0]: 1}

    best_score = -1.0
    best_split = 1
    for split_idx in range(1, k):
        high_part = sorted_clusters[:split_idx]
        low_part = sorted_clusters[split_idx:]
        n_a = sum(c for _, _, c in high_part)
        n_b = sum(c for _, _, c in low_part)
        if n_a == 0 or n_b == 0:
            continue
        r_a = sum(r * c for _, r, c in high_part) / float(n_a)
        r_b = sum(r * c for _, r, c in low_part) / float(n_b)
        score = ((n_a * n_b) / float(n_a + n_b)) * ((r_a - r_b) ** 2)
        if score > best_score:
            best_score = score
            best_split = split_idx

    return {cid: (0 if rank < best_split else 1) for rank, (cid, _, _) in enumerate(sorted_clusters)}


def _fit_startup_platt_calibration(data_dir: Path) -> dict[int, tuple[float, float]]:
    """Fits L2-regularized Platt scaling `(coef, intercept)` per head on validation FUPs at startup."""
    import torch
    from datasets import disable_caching, load_from_disk
    from sklearn.linear_model import LogisticRegression
    from src.data.patient_dataset import BIN_LABELS, preprocess_batch

    disable_caching()
    device = next(_REAL_MODEL.parameters()).device
    model_dtype = next(_REAL_MODEL.parameters()).dtype
    device_type = "cuda" if "cuda" in str(device) else "cpu"
    val_fups = [0, 30, 60, 90, 120, 150, 180, 240, 300, 360]
    label_keys = list(_REAL_COLLATOR.label_keys)

    logits_chunks = []
    labels_chunks = []

    for fup in val_fups:
        val_dir = data_dir / f"fup_{fup:04d}" / "validation"
        if not val_dir.exists():
            continue
        val_ds = load_from_disk(str(val_dir))
        n_rows = min(len(val_ds), 400)
        if n_rows == 0:
            continue
        sub_ds = val_ds.select(range(n_rows), keep_in_memory=True)
        processed = preprocess_batch(
            batch=sub_ds[:],
            vocab=_REAL_VOCAB,
            bin_intervals=_REAL_BIN_INT,
            bin_labels=BIN_LABELS,
            time_mapping=CONFIG.TIME_MAPPING,
            eav_mappings=CONFIG.EAV_MAPPINGS,
        )
        samples = []
        lbl_mat = np.full((n_rows, len(label_keys)), -100, dtype=np.int64)
        for col_j, lk in enumerate(label_keys):
            if lk in sub_ds.column_names:
                lbl_mat[:, col_j] = np.asarray(sub_ds[lk], dtype=np.int64)

        for i in range(n_rows):
            sample = {
                "entity_id":      np.array(processed["entity_id"][i]),
                "attribute_id":   np.array(processed["attribute_id"][i]),
                "value_id":       np.array(processed["value_id"][i]),
                "value_num":      np.array(processed["value_num"][i], dtype=np.float32),
                "days_since_tpx": np.array(processed["days_since_tpx"][i]),
                "split":          "validation",
            }
            for lk in label_keys:
                sample[lk] = 0.0
            samples.append(sample)

        with _MODEL_LOCK:
            for idx in range(0, n_rows, 48):
                b_coll = _REAL_COLLATOR.torch_call(samples[idx : idx + 48])
                inp = {
                    k: (v.to(device, dtype=model_dtype) if v.is_floating_point() else v.to(device))
                    for k, v in b_coll["input_dict"].items()
                }
                am = b_coll["attention_mask"].to(device)
                with torch.no_grad():
                    with torch.autocast(device_type=device_type, dtype=model_dtype):
                        out = _REAL_MODEL(input_dict=inp, attention_mask=am)
                        logits_chunks.append(out.logits.float().cpu().numpy())
        labels_chunks.append(lbl_mat)
        del val_ds, sub_ds, processed, samples

    if not logits_chunks:
        return {}

    all_logits = np.vstack(logits_chunks)
    all_labels = np.vstack(labels_chunks)
    platt_params: dict[int, tuple[float, float]] = {}
    for col_j in range(all_logits.shape[1]):
        y_col = all_labels[:, col_j]
        valid = (y_col == 0) | (y_col == 1)
        if valid.sum() < 10 or np.unique(y_col[valid]).size < 2:
            platt_params[col_j] = (1.0, 0.0)
            continue
        y_bin = y_col[valid].astype(np.int64)
        x_col = np.clip(all_logits[valid, col_j], -15.0, 15.0).reshape(-1, 1)
        n_pos = int(np.sum(y_bin == 1))
        slope, intercept, fitted = 1.0, 0.0, False
        if n_pos >= 20:
            try:
                clf = LogisticRegression(C=1.0, solver="lbfgs", max_iter=500)
                clf.fit(x_col, y_bin)
                if float(clf.coef_[0, 0]) > 0.05:
                    slope, intercept = float(clf.coef_[0, 0]), float(clf.intercept_[0])
                    fitted = True
            except Exception:
                pass
        if not fitted:
            p_bar = float(np.clip(np.mean(y_bin), 1e-6, 1.0 - 1e-6))
            slope = 1.0
            intercept = float(np.log(p_bar / (1.0 - p_bar)) - np.mean(x_col))
        platt_params[col_j] = (slope, intercept)
    return platt_params


def _extract_fup_shaded_representations(
    test_dir_path: Path,
    fup: int | None = None,
    anon_id_map: dict[str, str] | None = None,
) -> dict:
    """
    STARTUP-ONLY: Embeds a single FUP test split and distills de-identified ("shaded")
    feature sets for cohort cluster enrichment. Raw clinical sequences and real patient
    identifiers are discarded before returning.
    """
    import torch
    from datasets import disable_caching, load_from_disk
    from src.data.patient_dataset import BIN_LABELS, preprocess_batch

    disable_caching()
    device = next(_REAL_MODEL.parameters()).device
    model_dtype = next(_REAL_MODEL.parameters()).dtype
    device_type = "cuda" if "cuda" in str(device) else "cpu"

    test_ds = load_from_disk(str(test_dir_path))
    max_patients = int(getattr(CONFIG, "MAX_COHORT_PATIENTS", 500))
    n_patients = min(len(test_ds), max_patients)
    subset_ds = test_ds.select(range(n_patients), keep_in_memory=True)
    subset_ds = subset_ds.add_column("split", ["test"] * len(subset_ds))

    # Map real patientid -> ephemeral anonymous PAT_XXX so IDs stay consistent across FUPs,
    # then discard anon_id_map before saving anything to disk.
    patient_ids = []
    has_pid = "patientid" in subset_ds.column_names
    for i in range(n_patients):
        if anon_id_map is not None and has_pid:
            raw_pid = str(subset_ds[i]["patientid"])
            if raw_pid not in anon_id_map:
                anon_id_map[raw_pid] = f"PAT_{len(anon_id_map) + 1:03d}"
            patient_ids.append(anon_id_map[raw_pid])
        else:
            patient_ids.append(f"PAT_{i + 1:03d}")

    processed = preprocess_batch(
        batch=subset_ds[:],
        vocab=_REAL_VOCAB,
        bin_intervals=_REAL_BIN_INT,
        bin_labels=BIN_LABELS,
        time_mapping=CONFIG.TIME_MAPPING,
        eav_mappings=CONFIG.EAV_MAPPINGS,
    )

    samples = []
    for i in range(n_patients):
        sample = {
            "entity_id":      np.array(processed["entity_id"][i]),
            "attribute_id":   np.array(processed["attribute_id"][i]),
            "value_id":       np.array(processed["value_id"][i]),
            "value_num":      np.array(processed["value_num"][i], dtype=np.float32),
            "days_since_tpx": np.array(processed["days_since_tpx"][i]),
            "split":          "test",
        }
        for k in _REAL_COLLATOR.label_keys:
            sample[k] = 0.0
        samples.append(sample)

    all_full_logits = []
    all_embeddings = []
    with _MODEL_LOCK:
        for idx in range(0, n_patients, 32):
            batch_collated = _REAL_COLLATOR.torch_call(samples[idx : idx + 32])
            input_dict = {
                k: (v.to(device, dtype=model_dtype) if v.is_floating_point() else v.to(device))
                for k, v in batch_collated["input_dict"].items()
            }
            attn_mask = batch_collated["attention_mask"].to(device)
            with torch.no_grad():
                with torch.autocast(device_type=device_type, dtype=model_dtype):
                    outputs = _REAL_MODEL(input_dict=input_dict, attention_mask=attn_mask, output_hidden_states=True)
                    all_full_logits.append(outputs.logits.float().cpu().numpy())
                    all_embeddings.append(_pool_and_project_embeddings(outputs, attn_mask).float().cpu().numpy())

    full_logits_all = np.vstack(all_full_logits)
    embeddings = np.vstack(all_embeddings)

    id2vocab = {v: k for k, v in _REAL_VOCAB.items()}
    patient_summaries = []
    patient_static_sets = []
    patient_recent_sets = []

    v39_static_entities = STATIC_ENTITY_SET | {"Patient", "Donor"}
    v39_lab_entities = {"Patient lab", "Organ lab", "Lab result"}

    for i in range(n_patients):
        summary_terms = []
        p_static = set()
        p_recent = set()
        seq_len = len(processed["entity_id"][i])
        valid_times = [
            float(processed["days_since_tpx"][i][idx])
            for idx in range(seq_len)
            if processed["entity_id"][i][idx] >= 5 and processed["attribute_id"][i][idx] >= 5 and processed["value_id"][i][idx] >= 5
        ]
        max_t = max(valid_times) if valid_times else 0.0

        for idx in range(seq_len):
            ent_id = processed["entity_id"][i][idx]
            attr_id = processed["attribute_id"][i][idx]
            val_id = processed["value_id"][i][idx]
            if ent_id < 5 or attr_id < 5 or val_id < 5:
                continue
            t_val = float(processed["days_since_tpx"][i][idx])
            ent = id2vocab.get(ent_id, "")
            attr = id2vocab.get(attr_id, "")
            val = id2vocab.get(val_id, "")
            if not ent or not attr or not val or attr in ("Transplantation event", "Transplant procedure", "Transplant event"):
                continue

            # v3.9-aware summary extraction
            if ("Tacrolimus" in attr and val in ("High", "Higher", "Highest")) or (ent == "Medication" and val == "Tacrolimus"):
                if "Tacrolimus" in attr:
                    summary_terms.append("Tacrolimus↑")
            elif ent in v39_lab_entities and ("Leukocytes" in attr or attr == "WBC") and val in ("Lowest", "Lower"):
                summary_terms.append("Leukopenia")
            elif ent == "Infection" and attr in ("Pathogen", "Category") and val not in ("0", "False", "No"):
                summary_terms.append(f"{val}+")
            elif ent in ("Organ event", "Biopsy", "Clinical event") and ("rj" in val.lower() or "Rejection" in attr) and val not in ("0", "No"):
                summary_terms.append("Rejection")
            elif ent == "Comorbidity" and "Diabetes" in val:
                summary_terms.append("Diabetes")
            elif ent == "Comorbidity" and "Hypertension" in val:
                summary_terms.append("Hypertension")
            elif ent in v39_lab_entities and "Creatinine" in attr and val in ("High", "Higher", "Highest"):
                summary_terms.append("Creatinine↑")
            elif ent in v39_lab_entities and "eGFR" in attr and val in ("Lowest", "Lower"):
                summary_terms.append("Low eGFR")

            feat = (ent, attr, val)
            is_static = ent in v39_static_entities or t_val <= 0.0
            if is_static:
                p_static.add(feat)
            elif t_val > 0.0 and t_val >= (max_t - 30.0):
                p_recent.add(feat)

        dedup_terms = list(dict.fromkeys(summary_terms))
        patient_summaries.append(", ".join(dedup_terms[:3]) if dedup_terms else "Standard profile")
        patient_static_sets.append(p_static)
        patient_recent_sets.append(p_recent)

    available_tasks = getattr(CONFIG, "AVAILABLE_TASKS", [{"key": CONFIG.TASK, "label": CONFIG.TASK}])
    horizons = list(getattr(CONFIG, "AVAILABLE_HORIZONS", [30, 60, 90, 180, 360, 720, 1800]))
    task_outcomes = {}
    for t_info in available_tasks:
        t_key = t_info["key"]
        t_horizons = _REAL_TASK_HORIZONS_MAP.get(t_key, horizons)
        event_info = [_get_event_info_from_row(subset_ds[i], t_key) for i in range(n_patients)]
        labels_by_h = {}
        for h in t_horizons:
            col_name = f"label_{t_key}_{h:04d}d"
            if col_name in subset_ds.column_names:
                raw_col = np.asarray(subset_ds[col_name], dtype=np.int64)
                labels_by_h[h] = [(-100 if int(v) == -100 else int(v == 1)) for v in raw_col]
            else:
                labels_by_h[h] = [-100] * n_patients
        task_outcomes[t_key] = {"event_info": event_info, "labels_by_h": labels_by_h}

    del test_ds, subset_ds, processed, samples

    return {
        "fup": fup,
        "patient_ids": patient_ids,
        "full_logits_all": full_logits_all,
        "embeddings": embeddings,
        "n_patients": n_patients,
        "patient_summaries": patient_summaries,
        "patient_static_sets": patient_static_sets,
        "patient_recent_sets": patient_recent_sets,
        "task_outcomes": task_outcomes,
    }


def _build_cohort_payload_from_shaded(shaded: dict, coords_2d: np.ndarray, global_umap_bounds: dict | None = None) -> dict:
    """Assembles the multi-task, multi-horizon cohort payload for one FUP landmark."""
    global _REAL_LOGITS_MEDIAN, _REAL_LOGITS_STD, _REAL_LOGITS_MEDIAN_BY_TASK, _REAL_LOGITS_STD_BY_TASK
    import hdbscan
    from sklearn.cluster import KMeans
    import umap

    patient_ids = shaded["patient_ids"]
    full_logits_all = shaded["full_logits_all"]
    embeddings = shaded["embeddings"]
    n_patients = shaded["n_patients"]
    fup_val = shaded.get("fup")
    patient_summaries = shaded["patient_summaries"]
    patient_static_sets = shaded["patient_static_sets"]
    patient_recent_sets = shaded["patient_recent_sets"]
    task_outcomes = shaded["task_outcomes"]

    cluster_names_lookup = {0: "High risk", 1: "Low risk", 2: "Medium risk"}
    cluster_colors_lookup = {0: "#FF4757", 1: "#00C9A7", 2: "#FFA502"}

    use_clusterer = getattr(CONFIG, "CLUSTERING_METHOD", "clusterer") == "clusterer"
    cluster_input = coords_2d
    cluster_labels = None
    unique_labels = []
    if use_clusterer and n_patients >= 4:
        n_comp = getattr(CONFIG, "CLUSTERING_UMAP_COMPONENTS", 2)
        if n_comp > 0 and n_comp != 2 and n_comp < embeddings.shape[1]:
            cluster_input = umap.UMAP(n_components=n_comp, random_state=42).fit_transform(embeddings)
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=getattr(CONFIG, "HDBSCAN_MIN_CLUSTER_SIZE", 15),
            min_samples=getattr(CONFIG, "HDBSCAN_MIN_SAMPLES", 5),
            cluster_selection_method="leaf",
        )
        cluster_labels = clusterer.fit_predict(cluster_input)
        unique_labels = [int(l) for l in np.unique(cluster_labels) if l != -1]
        if len(unique_labels) < 2 and n_patients >= 4:
            # Guarantee both High-risk (0) and Low-risk (1) reference subgroups exist
            km = KMeans(n_clusters=2, n_init=10, random_state=42)
            cluster_labels = km.fit_predict(cluster_input)
            unique_labels = [0, 1]

    tasks_data = {}
    available_tasks = getattr(CONFIG, "AVAILABLE_TASKS", [{"key": CONFIG.TASK, "label": CONFIG.TASK}])
    global_horizons = list(getattr(CONFIG, "AVAILABLE_HORIZONS", [30, 60, 90, 180, 360, 720, 1800]))

    for t_info in available_tasks:
        task_key = t_info["key"]
        cols = _REAL_TASK_COLS_MAP.get(task_key, _REAL_TASK_COL_INDICES)
        t_horizons = _REAL_TASK_HORIZONS_MAP.get(task_key, global_horizons)
        ref_h = CONFIG.DEFAULT_HORIZON if CONFIG.DEFAULT_HORIZON in t_horizons else t_horizons[min(2, len(t_horizons) - 1)]
        horizon_idx = t_horizons.index(ref_h)

        raw_logits_all = full_logits_all[:, cols]
        t_median = np.median(raw_logits_all, axis=0)
        t_std = np.std(raw_logits_all, axis=0)
        _REAL_LOGITS_MEDIAN_BY_TASK[task_key] = t_median
        _REAL_LOGITS_STD_BY_TASK[task_key] = t_std
        std_clamped = np.clip(t_std, a_min=1e-6, a_max=None)

        probs = expit((raw_logits_all - t_median[None, :]) / std_clamped[None, :])
        cal_probs = _apply_platt_matrix(raw_logits_all, cols)

        if use_clusterer and cluster_labels is not None and len(unique_labels) >= 1:
            cluster_rates = [
                (l, float(probs[cluster_labels == l, horizon_idx].mean()), int((cluster_labels == l).sum()))
                for l in unique_labels
            ]
            label_map = parse_clusters_into_risk_groups(cluster_rates)
            non_noise_idx = np.flatnonzero(cluster_labels != -1)
            ui_cluster_labels = []
            for i, l in enumerate(cluster_labels):
                if l == -1:
                    if len(non_noise_idx) > 0:
                        dists = np.sum((cluster_input[non_noise_idx] - cluster_input[i]) ** 2, axis=1)
                        nearest_idx = non_noise_idx[int(np.argmin(dists))]
                        ui_cluster_labels.append(label_map[int(cluster_labels[nearest_idx])])
                    else:
                        ui_cluster_labels.append(0 if probs[i, horizon_idx] >= 0.50 else 1)
                else:
                    ui_cluster_labels.append(label_map[int(l)])
        else:
            ui_cluster_labels = [0 if probs[i, horizon_idx] >= 0.50 else 1 for i in range(n_patients)]

        unique_cids = sorted(set(ui_cluster_labels))
        cluster_sizes = defaultdict(int)
        static_counts = defaultdict(lambda: defaultdict(int))
        recent_counts = defaultdict(lambda: defaultdict(int))
        all_static_features = set()
        all_recent_features = set()

        for i in range(n_patients):
            c_id = ui_cluster_labels[i]
            cluster_sizes[c_id] += 1
            for feat in patient_static_sets[i]:
                static_counts[c_id][feat] += 1
                all_static_features.add(feat)
            for feat in patient_recent_sets[i]:
                recent_counts[c_id][feat] += 1
                all_recent_features.add(feat)

        def _compute_top_diffs(counts_dict, feat_universe, c_id: int):
            opp_id = 1 - c_id
            c_size = cluster_sizes.get(c_id, 0)
            opp_size = cluster_sizes.get(opp_id, 0)
            diffs = []
            for feat in feat_universe:
                p_c = (counts_dict[c_id][feat] / c_size) if c_size > 0 else 0.0
                p_opp = (counts_dict[opp_id][feat] / opp_size) if opp_size > 0 else 0.0
                p_glob = (counts_dict[c_id][feat] + counts_dict[opp_id][feat]) / float(max(1, n_patients))
                if p_glob > 0.85:
                    continue
                diff = p_c - p_opp
                if diff >= 0.04 and (p_opp == 0 or (p_c / p_opp) >= 1.15):
                    diffs.append((feat, diff, diff * (p_c / (p_opp + 0.05))))
            return [
                {"feature": f"{ent} - {attr}", "value": val, "score": (diff if c_id == 0 else -diff)}
                for (ent, attr, val), diff, _ in sorted(diffs, key=lambda x: x[2], reverse=True)[:10]
            ]

        precomputed_drivers = {
            c_id: {
                "static": _compute_top_diffs(static_counts, all_static_features, c_id),
                "recent": _compute_top_diffs(recent_counts, all_recent_features, c_id),
            }
            for c_id in (0, 1) if c_id in unique_cids
        }

        t_outcomes = task_outcomes.get(task_key, {})
        task_event_info = t_outcomes.get("event_info", [(None, 0.0)] * n_patients)
        labels_by_h = t_outcomes.get("labels_by_h", {})

        horizons_data = {}
        for idx_h, h in enumerate(t_horizons):
            h_labels = labels_by_h.get(h, [-100] * n_patients)
            h_patients = []
            for i in range(n_patients):
                c_id = ui_cluster_labels[i]
                raw_lbl = int(h_labels[i])
                days_to_event, event_imminence = task_event_info[i]
                h_patients.append({
                    "id":              patient_ids[i],
                    "umap_x":          round(float(coords_2d[i, 0]), 3),
                    "umap_y":          round(float(coords_2d[i, 1]), 3),
                    "risk_score":      round(float(probs[i, idx_h]), 4),
                    "calibrated_risk": round(float(cal_probs[i, idx_h]), 4),
                    "cluster":         c_id,
                    "cluster_name":    cluster_names_lookup.get(c_id, "Medium risk"),
                    "cluster_color":   cluster_colors_lookup.get(c_id, "#FFA502"),
                    "true_label":      1 if raw_lbl == 1 else (0 if raw_lbl == 0 else -100),
                    "summary":         patient_summaries[i],
                    "days_to_event":   days_to_event,
                    "event_imminence": round(event_imminence, 4),
                })

            rv = probs[:, idx_h]
            risk_distribution = {
                "values": [round(float(x), 4) for x in rv],
                "mean":   round(float(rv.mean()), 4),
                "std":    round(float(rv.std()), 4),
                "percentiles": {
                    "p25": round(float(np.percentile(rv, 25)), 4),
                    "p50": round(float(np.percentile(rv, 50)), 4),
                    "p75": round(float(np.percentile(rv, 75)), 4),
                    "p90": round(float(np.percentile(rv, 90)), 4),
                },
            }

            cluster_profiles = {}
            for c_id in (0, 1):
                if c_id not in unique_cids:
                    continue
                # Strictly exclude right-censored (-100) patients when computing empirical event_rate!
                uncensored_labels = [
                    h_patients[i]["true_label"]
                    for i in range(n_patients)
                    if ui_cluster_labels[i] == c_id and h_patients[i]["true_label"] >= 0
                ]
                event_rate = float(np.mean(uncensored_labels)) if uncensored_labels else 0.0
                cluster_profiles[str(c_id)] = {
                    "name":         cluster_names_lookup.get(c_id, "Medium risk"),
                    "n_patients":   cluster_sizes.get(c_id, 0),
                    "event_rate":   round(event_rate, 4),
                    "color":        cluster_colors_lookup.get(c_id, "#FFA502"),
                    "top_features": precomputed_drivers[c_id],
                }

            horizons_data[str(h)] = {
                "patients": h_patients,
                "risk_distribution": risk_distribution,
                "cluster_profiles": cluster_profiles,
            }

        tasks_data[task_key] = {
            "horizons":      horizons_data,
            "logits_median": t_median.tolist(),
            "logits_std":    t_std.tolist(),
        }

    default_task_data = tasks_data.get(CONFIG.TASK) or next(iter(tasks_data.values()))
    _REAL_LOGITS_MEDIAN = np.array(default_task_data["logits_median"])
    _REAL_LOGITS_STD = np.array(default_task_data["logits_std"])

    bounds = global_umap_bounds or _REAL_GLOBAL_UMAP_BOUNDS or {
        "min_x": round(float(coords_2d[:, 0].min()), 3),
        "max_x": round(float(coords_2d[:, 0].max()), 3),
        "min_y": round(float(coords_2d[:, 1].min()), 3),
        "max_y": round(float(coords_2d[:, 1].max()), 3),
    }

    return {
        "fup":                fup_val,
        "global_umap_bounds": bounds,
        "horizons":           default_task_data["horizons"],
        "logits_median":      default_task_data["logits_median"],
        "logits_std":         default_task_data["logits_std"],
        "tasks":              tasks_data,
    }


def _get_cohort_cache_key_and_path(paths: dict) -> tuple[str, Path, dict]:
    model_dir = Path(paths["model_path"])
    safetensors_file = model_dir / "model.safetensors"
    model_mtime = round(safetensors_file.stat().st_mtime, 2) if safetensors_file.exists() else 0
    model_size = int(safetensors_file.stat().st_size) if safetensors_file.exists() else 0

    meta_hasher = hashlib.sha256()
    for meta_candidate in (model_dir / "config.json", model_dir.parent / "run_metadata.json"):
        if meta_candidate.exists():
            try:
                meta_hasher.update(meta_candidate.read_bytes())
            except OSError:
                pass

    try:
        norm_model_str = str(model_dir.resolve().relative_to(_ROOT.resolve()))
    except ValueError:
        norm_model_str = str(model_dir.resolve())

    cache_params = {
        "version": "v7_common_trunk_v39_calibrated",
        "model_path": norm_model_str,
        "model_mtime": model_mtime,
        "model_size": model_size,
        "run_metadata_hash": meta_hasher.hexdigest()[:16],
        "precomputed_fups": list(getattr(CONFIG, "PRECOMPUTED_FUPS", [])),
        "max_cohort_patients": int(getattr(CONFIG, "MAX_COHORT_PATIENTS", 500)),
        "umap_fit_sample_size": int(getattr(CONFIG, "UMAP_FIT_SAMPLE_SIZE", 1950)),
        "umap_n_neighbors": int(getattr(CONFIG, "UMAP_N_NEIGHBORS", 15)),
        "umap_min_dist": float(getattr(CONFIG, "UMAP_MIN_DIST", 0.1)),
        "umap_random_state": int(getattr(CONFIG, "UMAP_RANDOM_STATE", 42)),
        "clustering_method": getattr(CONFIG, "CLUSTERING_METHOD", "clusterer"),
        "clustering_umap_components": int(getattr(CONFIG, "CLUSTERING_UMAP_COMPONENTS", 2)),
        "hdbscan_min_cluster_size": int(getattr(CONFIG, "HDBSCAN_MIN_CLUSTER_SIZE", 15)),
        "hdbscan_min_samples": int(getattr(CONFIG, "HDBSCAN_MIN_SAMPLES", 5)),
        "tasks": [t["key"] for t in getattr(CONFIG, "AVAILABLE_TASKS", [])],
        "horizons": list(getattr(CONFIG, "AVAILABLE_HORIZONS", [])),
    }
    cache_hash = hashlib.sha256(json.dumps(cache_params, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    cache_dir = _ensure_private_dir(Path(getattr(CONFIG, "CACHE_DIR", Path(__file__).resolve().parent / ".cache")))
    return cache_hash, cache_dir / f"cohort_umap_{cache_hash}.pkl", cache_params


def _load_or_build_multi_fup_cache(paths: dict):
    global _COHORT_CACHE, _COHORT_DATA, _REAL_UMAP_REDUCER, _REAL_GLOBAL_UMAP_BOUNDS
    global _REAL_LOGITS_MEDIAN, _REAL_LOGITS_STD, _REAL_LOGITS_MEDIAN_BY_TASK, _REAL_LOGITS_STD_BY_TASK
    global _EMBEDDINGS_CACHE_BY_FUP, _REAL_PLATT_PARAMS_BY_COL
    import umap

    t0 = time.time()
    cache_hash, cache_file, cache_params = _get_cohort_cache_key_and_path(paths)

    if cache_file.exists():
        try:
            print(f"[AIIDKIT] Loading pre-computed startup cohort cache from {cache_file} ...")
            with open(cache_file, "rb") as f:
                cached = pickle.load(f)
            _COHORT_CACHE.update(cached["cohort_cache"])
            _EMBEDDINGS_CACHE_BY_FUP.update(cached.get("embeddings_by_fup", {}))
            _REAL_UMAP_REDUCER = cached["umap_reducer"]
            _REAL_GLOBAL_UMAP_BOUNDS = cached["global_umap_bounds"]
            _REAL_LOGITS_MEDIAN_BY_TASK.update(cached.get("logits_median_by_task", {}))
            _REAL_LOGITS_STD_BY_TASK.update(cached.get("logits_std_by_task", {}))
            _REAL_PLATT_PARAMS_BY_COL.update({
                int(k): tuple(v) for k, v in cached.get("platt_params_by_col", {}).items()
            })

            default_cohort = _COHORT_CACHE.get(CONFIG.DEFAULT_FUP) or next(iter(_COHORT_CACHE.values()))
            _COHORT_DATA = default_cohort
            _REAL_LOGITS_MEDIAN = np.array(default_cohort["logits_median"])
            _REAL_LOGITS_STD = np.array(default_cohort["logits_std"])
            print(
                f"[AIIDKIT] Loaded {len(_COHORT_CACHE)} FUP shaded cohorts + Platt calibrators + shared UMAP "
                f"in {time.time() - t0:.2f}s. Zero raw data access needed at runtime."
            )
            return
        except Exception as exc:
            print(f"[AIIDKIT] WARNING: Failed to load cohort cache ({exc}), recomputing at startup...")

    if paths.get("data_dir") is None:
        raise RuntimeError(f"Startup cache {cache_file.name} not found and DATA_DIR is unavailable.")

    data_dir = Path(paths["data_dir"])
    print("[AIIDKIT] Fitting validation L2-Platt calibration parameters per head...")
    _REAL_PLATT_PARAMS_BY_COL = _fit_startup_platt_calibration(data_dir)

    precomputed_fups = list(getattr(CONFIG, "PRECOMPUTED_FUPS", [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360]))
    shaded_by_fup = {}
    ephemeral_anon_id_map: dict[str, str] = {}

    print(
        f"[AIIDKIT] Startup cache miss (hash={cache_hash}). "
        f"Embedding cohort across {len(precomputed_fups)} FUP stages..."
    )
    for fup in precomputed_fups:
        fup_test_dir = data_dir / f"fup_{fup:04d}" / "test"
        if not fup_test_dir.exists():
            continue
        shaded_by_fup[fup] = _extract_fup_shaded_representations(
            fup_test_dir, fup=fup, anon_id_map=ephemeral_anon_id_map
        )
        print(f"  ✓ Embedded & shaded FUP {fup:4d}d ({shaded_by_fup[fup]['n_patients']} patients)")

    # Immediately discard ephemeral raw->anonymous ID mapping so zero real IDs survive in memory or disk!
    del ephemeral_anon_id_map

    if not shaded_by_fup:
        raise RuntimeError(f"No valid FUP test directories found under {data_dir}")

    fit_sample_size = int(getattr(CONFIG, "UMAP_FIT_SAMPLE_SIZE", 1950))
    per_fup_sample = max(50, fit_sample_size // len(shaded_by_fup))
    rng = np.random.default_rng(getattr(CONFIG, "UMAP_RANDOM_STATE", 42))
    pooled_for_fit = []
    for fup, shaded in sorted(shaded_by_fup.items()):
        embs = shaded["embeddings"]
        if len(embs) > per_fup_sample:
            idx = rng.choice(len(embs), size=per_fup_sample, replace=False)
            pooled_for_fit.append(embs[np.sort(idx)])
        else:
            pooled_for_fit.append(embs)
    pooled_fit_matrix = np.vstack(pooled_for_fit)

    print(f"[AIIDKIT] Fitting shared UMAP manifold on {len(pooled_fit_matrix)} pooled embeddings...")
    _REAL_UMAP_REDUCER = umap.UMAP(
        n_components=2,
        n_neighbors=getattr(CONFIG, "UMAP_N_NEIGHBORS", 15),
        min_dist=getattr(CONFIG, "UMAP_MIN_DIST", 0.1),
        random_state=getattr(CONFIG, "UMAP_RANDOM_STATE", 42),
    )
    _REAL_UMAP_REDUCER.fit(pooled_fit_matrix)

    coords_by_fup = {}
    all_x, all_y = [], []
    for fup, shaded in sorted(shaded_by_fup.items()):
        c2d = _REAL_UMAP_REDUCER.transform(shaded["embeddings"])
        coords_by_fup[fup] = c2d
        all_x.append(c2d[:, 0])
        all_y.append(c2d[:, 1])

    all_x_np = np.concatenate(all_x)
    all_y_np = np.concatenate(all_y)
    _REAL_GLOBAL_UMAP_BOUNDS = {
        "min_x": round(float(np.percentile(all_x_np, 0.5)), 3),
        "max_x": round(float(np.percentile(all_x_np, 99.5)), 3),
        "min_y": round(float(np.percentile(all_y_np, 0.5)), 3),
        "max_y": round(float(np.percentile(all_y_np, 99.5)), 3),
    }

    for fup, shaded in sorted(shaded_by_fup.items()):
        _COHORT_CACHE[fup] = _build_cohort_payload_from_shaded(shaded, coords_by_fup[fup], _REAL_GLOBAL_UMAP_BOUNDS)
        _EMBEDDINGS_CACHE_BY_FUP[fup] = {
            "patient_ids": shaded["patient_ids"],
            "embeddings": shaded["embeddings"].astype(np.float16),
            "logits": shaded["full_logits_all"].astype(np.float16),
            "coords_2d": coords_by_fup[fup].astype(np.float32),
        }

    del shaded_by_fup

    default_cohort = _COHORT_CACHE.get(CONFIG.DEFAULT_FUP) or next(iter(_COHORT_CACHE.values()))
    _COHORT_DATA = default_cohort
    _REAL_LOGITS_MEDIAN = np.array(default_cohort["logits_median"])
    _REAL_LOGITS_STD = np.array(default_cohort["logits_std"])

    try:
        _ensure_private_dir(cache_file.parent)
        tmp_file = cache_file.with_suffix(f".{os.getpid()}.pkl.tmp")
        with open(tmp_file, "wb") as f:
            pickle.dump(
                {
                    "cache_params": cache_params,
                    "vocab": _REAL_VOCAB,
                    "bin_intervals": _REAL_BIN_INT,
                    "platt_params_by_col": _REAL_PLATT_PARAMS_BY_COL,
                    "embeddings_by_fup": _EMBEDDINGS_CACHE_BY_FUP,
                    "cohort_cache": _COHORT_CACHE,
                    "umap_reducer": _REAL_UMAP_REDUCER,
                    "global_umap_bounds": _REAL_GLOBAL_UMAP_BOUNDS,
                    "logits_median_by_task": _REAL_LOGITS_MEDIAN_BY_TASK,
                    "logits_std_by_task": _REAL_LOGITS_STD_BY_TASK,
                },
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        _chmod_private_file(tmp_file)
        tmp_file.replace(cache_file)
        _chmod_private_file(cache_file)

        meta_file = cache_file.parent / "cache_metadata.json"
        meta_file.write_text(json.dumps({"active_cache_file": cache_file.name, **cache_params}, indent=2), encoding="utf-8")
        _chmod_private_file(meta_file)

        # Keep up to 3 most recent cache files so switching models does not force a rebuild
        existing_pkls = sorted(
            cache_file.parent.glob("cohort_umap_*.pkl"),
            key=lambda p: p.stat().st_mtime_ns,
            reverse=True,
        )
        for old_pkl in existing_pkls[3:]:
            try:
                old_pkl.unlink()
            except OSError:
                pass
        print(f"[AIIDKIT] Saved private (0600) startup cohort cache to {cache_file} ({time.time() - t0:.1f}s).")
    except Exception as exc:
        print(f"[AIIDKIT] WARNING: Failed to write cohort cache to disk: {exc}")


def _load_real_model(paths: dict):
    """Loads the trained Transformer checkpoint and auto-syncs tasks/horizons from `run_metadata.json`."""
    global _REAL_MODEL, _REAL_COLLATOR, _REAL_VOCAB, _REAL_BIN_INT
    global _REAL_TASK_COL_INDICES, _REAL_TASK_COLS_MAP, _REAL_TASK_HORIZONS_MAP
    import torch
    from src.model.patient_embedder import (
        PatientDataCollatorForClassification,
        PatientEmbeddingModelFactory,
    )

    model_dir = Path(paths["model_path"])
    cfg_dict = {}
    if (model_dir / "config.json").exists():
        cfg_dict = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))

    run_meta = paths.get("run_metadata")
    if run_meta is None:
        for cand in (model_dir / "run_metadata.json", model_dir.parent / "run_metadata.json"):
            if cand.exists():
                run_meta = json.loads(cand.read_text(encoding="utf-8"))
                break

    # Dynamically synchronize CONFIG tasks & horizons with the loaded checkpoint's metadata
    if isinstance(run_meta, dict):
        if isinstance(run_meta.get("tasks"), list) and run_meta["tasks"]:
            CONFIG.AVAILABLE_TASKS = build_available_tasks_list(run_meta["tasks"])
            if CONFIG.TASK not in run_meta["tasks"]:
                CONFIG.TASK = run_meta["tasks"][0]
        if isinstance(run_meta.get("horizons"), list) and run_meta["horizons"]:
            CONFIG.AVAILABLE_HORIZONS = sorted(int(h) for h in run_meta["horizons"])
            CONFIG.DEFAULT_HORIZON = (
                90 if 90 in CONFIG.AVAILABLE_HORIZONS
                else CONFIG.AVAILABLE_HORIZONS[min(2, len(CONFIG.AVAILABLE_HORIZONS) - 1)]
            )
            CONFIG.HORIZON_LABELS = build_horizon_labels_map(CONFIG.AVAILABLE_HORIZONS, style="long")
            CONFIG.HORIZON_SHORT_LABELS = build_horizon_labels_map(CONFIG.AVAILABLE_HORIZONS, style="short")

    vocab_path = paths.get("vocab_path")
    bin_int_path = paths.get("bin_intervals")
    if vocab_path is not None and bin_int_path is not None:
        with open(vocab_path, "rb") as f:
            _REAL_VOCAB = pickle.load(f)
        with open(bin_int_path, "rb") as f:
            _REAL_BIN_INT = pickle.load(f)
    else:
        _, cache_file, _ = _get_cohort_cache_key_and_path(paths)
        if not cache_file.exists():
            candidates = sorted(cache_file.parent.glob("cohort_umap_*.pkl")) if cache_file.parent.exists() else []
            if candidates:
                cache_file = candidates[-1]
        if not cache_file.exists():
            raise RuntimeError(f"Neither vocab.pkl/bin_intervals.pkl nor startup cache {cache_file} found.")
        with open(cache_file, "rb") as f:
            cached_meta = pickle.load(f)
        _REAL_VOCAB = cached_meta["vocab"]
        _REAL_BIN_INT = cached_meta["bin_intervals"]

    device = "cuda" if (paths["use_gpu"] and torch.cuda.is_available()) else "cpu"
    num_labels = len(cfg_dict.get("id2label", {})) or (
        len(run_meta["label_keys"]) if isinstance(run_meta, dict) and "label_keys" in run_meta else len(CONFIG.AVAILABLE_HORIZONS)
    )
    problem_type = cfg_dict.get("problem_type", "multi_label_classification")
    enforce_monotonicity = build_monotonic_groups(run_meta, cfg_dict.get("enforce_monotonicity", True))

    all_label_keys = (
        list(run_meta["label_keys"])
        if isinstance(run_meta, dict) and isinstance(run_meta.get("label_keys"), list) and len(run_meta["label_keys"]) == num_labels
        else [f"label_{CONFIG.TASK}_{h:04d}d" for h in CONFIG.AVAILABLE_HORIZONS]
    )
    use_num_emb = bool(run_meta.get("use_numerical_embedding", True)) if isinstance(run_meta, dict) else True

    _REAL_TASK_COLS_MAP = {}
    _REAL_TASK_HORIZONS_MAP = {}
    valid_tasks = []
    for t_info in CONFIG.AVAILABLE_TASKS:
        t_key = t_info["key"]
        t_hrzs = (
            sorted(int(h) for h in run_meta.get("task_horizons", {}).get(t_key, CONFIG.AVAILABLE_HORIZONS))
            if isinstance(run_meta, dict)
            else list(CONFIG.AVAILABLE_HORIZONS)
        )
        t_keys = [f"label_{t_key}_{h:04d}d" for h in t_hrzs]
        if all(k in all_label_keys for k in t_keys):
            _REAL_TASK_COLS_MAP[t_key] = [all_label_keys.index(k) for k in t_keys]
            _REAL_TASK_HORIZONS_MAP[t_key] = t_hrzs
            valid_tasks.append(t_info)

    if valid_tasks:
        CONFIG.AVAILABLE_TASKS = valid_tasks
        if CONFIG.TASK not in _REAL_TASK_COLS_MAP:
            CONFIG.TASK = valid_tasks[0]["key"]
        _REAL_TASK_COL_INDICES = _REAL_TASK_COLS_MAP[CONFIG.TASK]
    else:
        _REAL_TASK_COL_INDICES = list(range(len(CONFIG.AVAILABLE_HORIZONS)))
        _REAL_TASK_COLS_MAP = {CONFIG.TASK: _REAL_TASK_COL_INDICES}
        _REAL_TASK_HORIZONS_MAP = {CONFIG.TASK: list(CONFIG.AVAILABLE_HORIZONS)}

    emb_cfg = {
        "vocab_size":              len(_REAL_VOCAB),
        "eav_mappings":            CONFIG.EAV_MAPPINGS,
        "time_mapping":            CONFIG.TIME_MAPPING,
        "use_numerical_embedding": use_num_emb,
    }
    _REAL_MODEL = PatientEmbeddingModelFactory.from_pretrained(
        task="classification",
        pretrained_dir=str(model_dir),
        embedding_layer_config=emb_cfg,
        model_args={"num_labels": num_labels, "problem_type": problem_type},
        enforce_monotonicity=enforce_monotonicity,
    )
    _REAL_MODEL.eval()
    _REAL_MODEL.to(device)

    _REAL_COLLATOR = PatientDataCollatorForClassification(
        eav_mappings=CONFIG.EAV_MAPPINGS,
        time_mapping=CONFIG.TIME_MAPPING,
        use_numerical_embedding=use_num_emb,
        label_keys=all_label_keys,
        max_position_embeddings=getattr(_REAL_MODEL.config, "max_position_embeddings", 1024),
    )
    print(
        f"[AIIDKIT] Real model loaded from {model_dir} on {device} "
        f"(num_labels={num_labels}, tasks={len(CONFIG.AVAILABLE_TASKS)}, horizons={CONFIG.AVAILABLE_HORIZONS})"
    )

    try:
        _load_or_build_multi_fup_cache(paths)
    except Exception as exc:
        print(f"[AIIDKIT] WARNING: Could not load/build real cohort cache: {exc}")
        import traceback
        traceback.print_exc()


def _deterministic_seed(events: list[dict]) -> int:
    sorted_events = sorted(
        events,
        key=lambda e: (
            float(e.get("days_since_tpx", 0.0)),
            str(e.get("entity", "")),
            str(e.get("attribute", "")),
            str(e.get("value", "")),
        ),
    )
    event_str = "".join(
        f"{e.get('entity')}-{e.get('attribute')}-{e.get('value')}-{e.get('days_since_tpx')}"
        for e in sorted_events
    )
    return int(hashlib.md5(event_str.encode("utf-8")).hexdigest(), 16) & 0xFFFFFFFF


def _events_signature(events: list[dict]) -> str:
    sorted_events = sorted(
        events,
        key=lambda e: (
            float(e.get("days_since_tpx", 0.0)),
            str(e.get("entity", "")),
            str(e.get("attribute", "")),
            str(e.get("value", "")),
        ),
    )
    event_str = "|".join(
        f"{e.get('entity')}:{e.get('attribute')}:{e.get('value')}:{float(e.get('days_since_tpx', 0.0)):.2f}"
        for e in sorted_events
    )
    return hashlib.md5(event_str.encode("utf-8")).hexdigest()


def _truncate_events_for_fup(events: list[dict], fup: int | None) -> list[dict]:
    if fup is None or not events:
        return list(events)
    filtered = [e for e in events if float(e.get("days_since_tpx", 0.0)) <= float(fup)]
    if not filtered:
        min_day = min(float(e.get("days_since_tpx", 0.0)) for e in events)
        filtered = [e for e in events if float(e.get("days_since_tpx", 0.0)) <= min_day]
    return filtered


def _compute_patient_trajectory_real(full_events: list[dict], fup_steps: list[int]) -> tuple[list[dict], dict[int, np.ndarray]]:
    """
    Computes longitudinal UMAP trajectory + raw logits across all `fup_steps` in ONE batched
    forward pass + ONE batched UMAP transform. Returns `(trajectory, raw_logits_by_fup)`.
    """
    import torch
    from src.data.patient_dataset import BIN_LABELS, preprocess_batch

    if not fup_steps:
        return [], {}

    cache_key = (_events_signature(full_events), tuple(fup_steps))
    cached_val = _lru_get(_TRAJECTORY_CACHE, cache_key)
    if cached_val is not None:
        return cached_val

    device = next(_REAL_MODEL.parameters()).device
    model_dtype = next(_REAL_MODEL.parameters()).dtype
    device_type = "cuda" if "cuda" in str(device) else "cpu"

    samples = []
    truncated_lists = []
    for f_step in fup_steps:
        evs_f = _truncate_events_for_fup(full_events, f_step)
        truncated_lists.append(evs_f)
        batch_raw = {
            "entity":    [e["entity"] for e in evs_f],
            "attribute": [e["attribute"] for e in evs_f],
            "value":     [e["value"] for e in evs_f],
            "time":      [e.get("days_since_tpx", 0.0) for e in evs_f],
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
            "value_num":      np.array(processed["value_num"][0], dtype=np.float32),
            "days_since_tpx": np.array(processed["days_since_tpx"][0]),
            "split":          "test",
        }
        for k in _REAL_COLLATOR.label_keys:
            sample[k] = 0.0
        samples.append(sample)

    with _MODEL_LOCK:
        batch_collated = _REAL_COLLATOR.torch_call(samples)
        input_dict = {
            k: (v.to(device, dtype=model_dtype) if v.is_floating_point() else v.to(device))
            for k, v in batch_collated["input_dict"].items()
        }
        attn_mask = batch_collated["attention_mask"].to(device)
        with torch.no_grad():
            with torch.autocast(device_type=device_type, dtype=model_dtype):
                outputs = _REAL_MODEL(input_dict=input_dict, attention_mask=attn_mask, output_hidden_states=True)
                logits_batch = outputs.logits.detach().float().cpu().numpy()
                pooled_batch = _pool_and_project_embeddings(outputs, attn_mask).detach().float().cpu().numpy()

    if _REAL_UMAP_REDUCER is not None:
        try:
            coords_batch = _REAL_UMAP_REDUCER.transform(pooled_batch)
        except Exception as exc:
            print(f"[AIIDKIT] WARNING: Trajectory UMAP transform failed: {exc}")
            coords_batch = np.zeros((len(fup_steps), 2))
    else:
        coords_batch = np.zeros((len(fup_steps), 2))

    available_tasks = getattr(CONFIG, "AVAILABLE_TASKS", [{"key": CONFIG.TASK, "label": CONFIG.TASK}])
    global_horizons = list(getattr(CONFIG, "AVAILABLE_HORIZONS", [30, 60, 90, 180, 360, 720, 1800]))
    trajectory = []
    logits_by_fup: dict[int, np.ndarray] = {}

    for idx, f_step in enumerate(fup_steps):
        logits_by_fup[f_step] = logits_batch[idx]
        ux = round(float(coords_batch[idx, 0]), 3)
        uy = round(float(coords_batch[idx, 1]), 3)
        f_cohort = _COHORT_CACHE.get(f_step) or _COHORT_DATA or {}
        f_cohort_tasks = f_cohort.get("tasks", {})
        t_scores = {}
        for t_info in available_tasks:
            t_key = t_info["key"]
            cols = _REAL_TASK_COLS_MAP.get(t_key, _REAL_TASK_COL_INDICES)
            t_horizons = _REAL_TASK_HORIZONS_MAP.get(t_key, global_horizons)
            l_np = logits_batch[idx, cols]
            tc = f_cohort_tasks.get(t_key, f_cohort)
            c_med = np.array(tc.get("logits_median", _REAL_LOGITS_MEDIAN_BY_TASK.get(t_key, _REAL_LOGITS_MEDIAN)), dtype=float)
            c_std = np.clip(np.array(tc.get("logits_std", _REAL_LOGITS_STD_BY_TASK.get(t_key, _REAL_LOGITS_STD)), dtype=float), 1e-6, None)
            ev_sc = expit((l_np - c_med) / c_std)
            cal_pr = _apply_platt_vector(l_np, cols)
            t_scores[t_key] = {
                "risk_scores": {f"{h}d": round(float(np.clip(ev_sc[i], 0.01, 0.99)), 4) for i, h in enumerate(t_horizons)},
                "calibrated_risks": {f"{h}d": round(float(np.clip(cal_pr[i], 1e-4, 0.99)), 4) for i, h in enumerate(t_horizons)},
            }
        trajectory.append({
            "fup": f_step,
            "umap_x": ux,
            "umap_y": uy,
            "n_events": len(truncated_lists[idx]),
            "tasks": t_scores,
        })

    result_pair = (trajectory, logits_by_fup)
    _lru_put(_TRAJECTORY_CACHE, cache_key, result_pair, maxsize=128)
    return result_pair


def _compute_task_ig_attributions(events: list[dict], active_fup: int, task_key: str) -> dict[int, np.ndarray]:
    """
    Computes Value-Pathway Integrated Gradients in probability space `sigmoid(logits)`
    strictly for the `len(cols)` horizon heads of `task_key` (7 heads instead of 91 -> 13x faster!),
    preserving special tokens (`< 5`) and handling middle-truncation alignment.
    """
    import torch
    from src.data.patient_dataset import BIN_LABELS, preprocess_batch

    cols = _REAL_TASK_COLS_MAP.get(task_key, _REAL_TASK_COL_INDICES)
    ig_key = (_events_signature(events), int(active_fup), str(task_key))
    cached_ig = _lru_get(_IG_TASK_CACHE, ig_key)
    if cached_ig is not None:
        return cached_ig

    device = next(_REAL_MODEL.parameters()).device
    model_dtype = next(_REAL_MODEL.parameters()).dtype
    device_type = "cuda" if "cuda" in str(device) else "cpu"

    batch_raw = {
        "entity":    [e["entity"] for e in events],
        "attribute": [e["attribute"] for e in events],
        "value":     [e["value"] for e in events],
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
        "value_num":      np.array(processed["value_num"][0], dtype=np.float32),
        "days_since_tpx": np.array(processed["days_since_tpx"][0]),
        "split":          "test",
    }
    for k in _REAL_COLLATOR.label_keys:
        sample[k] = 0.0

    batch = _REAL_COLLATOR.torch_call([sample])
    input_dict = {
        k: (v.to(device, dtype=model_dtype) if v.is_floating_point() else v.to(device))
        for k, v in batch["input_dict"].items()
    }
    attn_mask = batch["attention_mask"].to(device)

    event_scores_by_col: dict[int, np.ndarray] = {}
    n_events = len(events)
    max_pos = int(getattr(_REAL_MODEL.config, "max_position_embeddings", 1024))

    try:
        with _MODEL_LOCK:
            n_steps = 32
            val = input_dict["value_id"]
            # Preserve all special tokens (< 5: PAD=0, MASK=1, BOS=2, EOS=3, SEP=4) in neutral baseline
            baseline_val = torch.where(val < 5, val, torch.zeros_like(val))
            baseline_val_num = (
                torch.full_like(input_dict["value_num"], float("nan"))
                if "value_num" in input_dict
                else None
            )

            with torch.no_grad():
                inp_val_emb = _REAL_MODEL.patient_embedder.encode_values(val, input_dict.get("value_num")).detach().float()
                base_val_emb = _REAL_MODEL.patient_embedder.encode_values(baseline_val, baseline_val_num).detach().float()
                diff_val_emb = (inp_val_emb - base_val_emb).squeeze(0)
                alphas = torch.linspace(1.0 / n_steps, 1.0, n_steps, device=device, dtype=torch.float32).view(n_steps, 1, 1)
                interp_val_emb = (base_val_emb + alphas * (inp_val_emb - base_val_emb)).to(dtype=model_dtype).detach()

            interp_val_emb.requires_grad_(True)
            rep_input_dict = {k: v.repeat(n_steps, 1) for k, v in input_dict.items()}
            rep_attn_mask = attn_mask.repeat(n_steps, 1)

            hook_handle = _REAL_MODEL.patient_embedder.value_embedding_hook.register_forward_hook(
                lambda module, inputs, output: interp_val_emb
            )
            try:
                with torch.enable_grad():
                    with torch.autocast(device_type=device_type, dtype=model_dtype):
                        outputs_ig = _REAL_MODEL(input_dict=rep_input_dict, attention_mask=rep_attn_mask)
                        probs_ig = torch.sigmoid(outputs_ig.logits.float())

                    for idx_c, col_idx in enumerate(cols):
                        retain = idx_c < (len(cols) - 1)
                        grads = torch.autograd.grad(
                            probs_ig[:, col_idx].sum(),
                            interp_val_emb,
                            retain_graph=retain,
                        )[0].float()
                        avg_grad = grads.mean(dim=0)
                        token_attr = (avg_grad * diff_val_emb).sum(dim=-1).detach().cpu().numpy()

                        # Align token positions back to `events` (accounting for BOS/EOS and optional middle truncation)
                        aligned = np.zeros(n_events, dtype=np.float64)
                        if n_events + 2 <= len(token_attr):
                            aligned[:] = token_attr[1 : 1 + n_events]
                        elif len(token_attr) == max_pos and n_events + 2 > max_pos:
                            # Collator keeps: [BOS] + head_events + [SEP] + tail_events + [EOS]
                            avail_event_slots = max_pos - 3
                            head_len = avail_event_slots // 2
                            tail_len = avail_event_slots - head_len
                            aligned[:head_len] = token_attr[1 : 1 + head_len]
                            aligned[n_events - tail_len :] = token_attr[2 + head_len : 2 + head_len + tail_len]
                        else:
                            copy_n = min(n_events, max(0, len(token_attr) - 2))
                            aligned[:copy_n] = token_attr[1 : 1 + copy_n]

                        event_scores_by_col[col_idx] = aligned
            finally:
                hook_handle.remove()
    except Exception as exc:
        print(f"[AIIDKIT] Integrated Gradients failed for {task_key}: {exc}")

    _lru_put(_IG_TASK_CACHE, ig_key, event_scores_by_col, maxsize=256)
    return event_scores_by_col


def _assemble_task_prediction_payload(
    *,
    t_key: str,
    t_label: str,
    full_logits_np: np.ndarray,
    umap_x: float,
    umap_y: float,
    events: list[dict],
    cohort: dict,
    active_fup: int,
    requested_horizon: int,
    attr_by_col: dict[int, np.ndarray] | None,
) -> dict:
    global_horizons = list(getattr(CONFIG, "AVAILABLE_HORIZONS", [30, 60, 90, 180, 360, 720, 1800]))
    cols = _REAL_TASK_COLS_MAP.get(t_key, _REAL_TASK_COL_INDICES)
    t_horizons = _REAL_TASK_HORIZONS_MAP.get(t_key, global_horizons)
    logits_np = full_logits_np[cols]

    t_cohort = cohort.get("tasks", {}).get(t_key, cohort)
    cohort_median = np.array(
        t_cohort.get("logits_median", _REAL_LOGITS_MEDIAN_BY_TASK.get(t_key, _REAL_LOGITS_MEDIAN)),
        dtype=float,
    )
    cohort_std = np.clip(
        np.array(t_cohort.get("logits_std", _REAL_LOGITS_STD_BY_TASK.get(t_key, _REAL_LOGITS_STD)), dtype=float),
        1e-6,
        None,
    )
    event_scores = expit((logits_np - cohort_median) / cohort_std)
    calibrated_probs = _apply_platt_vector(logits_np, cols)

    risk_scores = {f"{h}d": round(float(np.clip(event_scores[i], 0.01, 0.99)), 4) for i, h in enumerate(t_horizons)}
    calibrated_risks = {f"{h}d": round(float(np.clip(calibrated_probs[i], 1e-4, 0.99)), 4) for i, h in enumerate(t_horizons)}

    # Vectorize 2D nearest-neighbor lookup ONCE per task (UMAP coordinates are identical across horizons)
    first_h_pts = t_cohort["horizons"].get(str(t_horizons[0]), {}).get("patients", [])
    if first_h_pts:
        coords_arr = np.array([[p["umap_x"], p["umap_y"]] for p in first_h_pts], dtype=np.float64)
        sq_dists = np.sum((coords_arr - np.array([umap_x, umap_y])) ** 2, axis=1)
        top3_idx = np.argsort(sq_dists)[:3]
        nearest_idx = int(top3_idx[0])
    else:
        top3_idx = np.array([], dtype=int)
        nearest_idx = -1

    by_horizon = {}
    ig_computed = attr_by_col is not None
    for h_idx, h in enumerate(t_horizons):
        selected_risk = risk_scores[f"{h}d"]
        selected_calibrated_risk = calibrated_risks[f"{h}d"]
        active_cohort = t_cohort["horizons"][str(h)]
        cohort_pts = active_cohort.get("patients", [])

        if cohort_pts and nearest_idx >= 0 and nearest_idx < len(cohort_pts):
            nearest_pt = cohort_pts[nearest_idx]
            assigned_cluster_id = nearest_pt["cluster"]
            assigned_cluster_name = nearest_pt["cluster_name"]
            assigned_cluster_color = nearest_pt["cluster_color"]
            similar_patients = [
                {"id": cohort_pts[j]["id"], "risk_score": cohort_pts[j]["risk_score"], "summary": cohort_pts[j]["summary"]}
                for j in top3_idx if j < len(cohort_pts)
            ]
        else:
            assigned_cluster_id = 0 if selected_risk >= 0.50 else 1
            assigned_cluster_name = "High risk" if assigned_cluster_id == 0 else "Low risk"
            assigned_cluster_color = "#FF4757" if assigned_cluster_id == 0 else "#00C9A7"
            similar_patients = []

        cohort_risks = np.array([p["risk_score"] for p in cohort_pts]) if cohort_pts else np.array([0.5])
        percentile = int(np.round((cohort_risks < selected_risk).mean() * 100))

        col_idx = cols[h_idx]
        ev_scores = attr_by_col.get(col_idx) if attr_by_col else None
        attributions = []
        events_with_scores = []
        for i, e in enumerate(events):
            score = float(ev_scores[i]) if (ev_scores is not None and i < len(ev_scores)) else 0.0
            attributions.append({
                "feature": f"{e['entity']} - {e['attribute']}",
                "value": str(e["value"]),
                "score": round(score, 4),
            })
            events_with_scores.append({**e, "score": round(score, 4)})
        attributions = sorted(attributions, key=lambda x: abs(x["score"]), reverse=True)[:15]

        risk_txt = build_clinical_narrative(
            task_key=t_key,
            task_label=t_label,
            selected_horizon=h,
            horizons=t_horizons,
            risk_scores=risk_scores,
            calibrated_risks=calibrated_risks,
            cluster_name=assigned_cluster_name,
        )

        by_horizon[str(h)] = {
            "task":               t_key,
            "task_label":         t_label,
            "risk_scores":        risk_scores,
            "calibrated_risks":   calibrated_risks,
            "selected_horizon":   h,
            "risk_score":         selected_risk,
            "calibrated_risk":    selected_calibrated_risk,
            "risk_category":      assigned_cluster_name,
            "risk_color":         assigned_cluster_color,
            "percentile":         percentile,
            "cluster":            assigned_cluster_id,
            "cluster_name":       assigned_cluster_name,
            "cluster_color":      assigned_cluster_color,
            "umap_x":             umap_x,
            "umap_y":             umap_y,
            "days_to_event":      None,
            "event_imminence":    0.0,
            "attributions":       attributions,
            "events_with_scores": events_with_scores,
            "similar_patients":   similar_patients,
            "narrative":          risk_txt,
            "ig_computed":        ig_computed,
        }

    eff_h = str(requested_horizon if requested_horizon in t_horizons else t_horizons[0])
    return {
        **by_horizon[eff_h],
        "fup": active_fup,
        "ig_computed": ig_computed,
        "by_horizon": by_horizon,
    }


def _real_predict(
    events: list[dict],
    horizon: int,
    cohort: dict,
    selected_task: str | None = None,
    fup: int | None = None,
    trajectory_fups: list[int] | None = None,
) -> dict:
    full_events = sorted(list(events), key=lambda e: float(e.get("days_since_tpx", 0.0)))
    active_fup = fup if fup is not None else CONFIG.DEFAULT_FUP
    stage_events = _truncate_events_for_fup(full_events, active_fup)

    if trajectory_fups is None:
        trajectory_fups = [f for f in getattr(CONFIG, "AVAILABLE_FUPS", [0, 30, 60, 90]) if f <= active_fup]
    if active_fup not in trajectory_fups:
        trajectory_fups = sorted(set(trajectory_fups) | {active_fup})

    # 1. Single batched forward + UMAP pass across all trajectory FUPs (including active_fup!)
    trajectory, logits_by_fup = _compute_patient_trajectory_real(full_events, trajectory_fups)
    traj_by_fup = {pt["fup"]: pt for pt in trajectory}
    full_logits_np = logits_by_fup[active_fup]
    umap_x = traj_by_fup[active_fup]["umap_x"]
    umap_y = traj_by_fup[active_fup]["umap_y"]

    available_tasks = getattr(CONFIG, "AVAILABLE_TASKS", [{"key": CONFIG.TASK, "label": CONFIG.TASK}])
    valid_task_keys = {t["key"] for t in available_tasks}
    active_task = selected_task if (selected_task and selected_task in valid_task_keys) else CONFIG.TASK

    # 2. Compute Integrated Gradients strictly for `active_task` (7 heads instead of 91 -> 13x faster!),
    #    while attaching any previously cached IG attributions for other tasks.
    active_ig = _compute_task_ig_attributions(stage_events, active_fup, active_task)

    tasks_predictions = {}
    ev_sig = _events_signature(stage_events)
    for t_info in available_tasks:
        t_key = t_info["key"]
        attr_map = active_ig if t_key == active_task else _lru_get(_IG_TASK_CACHE, (ev_sig, int(active_fup), str(t_key)))
        tasks_predictions[t_key] = _assemble_task_prediction_payload(
            t_key=t_key,
            t_label=t_info["label"],
            full_logits_np=full_logits_np,
            umap_x=umap_x,
            umap_y=umap_y,
            events=stage_events,
            cohort=cohort,
            active_fup=active_fup,
            requested_horizon=horizon,
            attr_by_col=attr_map,
        )

    return {
        **tasks_predictions[active_task],
        "fup": active_fup,
        "trajectory": trajectory,
        "tasks": tasks_predictions,
    }


# ---------------------------------------------------------------------------
# App Initialisation & Vocabulary Enrichment
# ---------------------------------------------------------------------------

app = Flask(__name__)
CORS(app)

_PATHS = CONFIG.get_resolved_paths()


def _enrich_vocabulary_from_synthetic(vocab: dict) -> dict:
    synthetic_dir = Path(__file__).resolve().parent / "synthetic_patients"
    if not synthetic_dir.exists():
        return vocab

    enriched = {ent: {a: list(vs) for a, vs in attrs.items()} for ent, attrs in vocab.items()}
    for filename in ("low_risk.csv", "medium_risk.csv", "high_risk.csv"):
        file_path = synthetic_dir / filename
        if not file_path.exists():
            continue
        try:
            with open(file_path, mode="r", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    entity = (row.get("entity") or "").strip()
                    attribute = (row.get("attribute") or "").strip()
                    val = (row.get("value_binned") or row.get("value") or "").strip()
                    if entity and attribute and val:
                        enriched.setdefault(entity, {}).setdefault(attribute, [])
                        if val not in enriched[entity][attribute]:
                            enriched[entity][attribute].append(val)
        except Exception as exc:
            print(f"[AIIDKIT] Failed to read {filename} for vocabulary enrichment: {exc}")

    med_pairs = [
        ("Start (immunosuppression)", "Stop (immunosuppression)"),
        ("Start (induction)", "Stop (induction)"),
        ("Start (infection prophylaxis)", "Stop (infection prophylaxis)"),
        ("Start (other)", "Stop (other)"),
    ]
    if "Medication" in enriched:
        for a1, a2 in med_pairs:
            combined = sorted(set(enriched["Medication"].get(a1, [])) | set(enriched["Medication"].get(a2, [])))
            if combined:
                enriched["Medication"][a1] = list(combined)
                enriched["Medication"][a2] = list(combined)

    ordinals = ["Lowest", "Lower", "Low", "Middle", "High", "Higher", "Highest"]
    for entity in enriched:
        for attribute, vals in enriched[entity].items():
            try:
                if all(v.replace(".", "", 1).isdigit() or (v.startswith("-") and v[1:].replace(".", "", 1).isdigit()) for v in vals):
                    enriched[entity][attribute] = sorted(vals, key=float)
                    continue
            except Exception:
                pass
            if any(o in vals for o in ordinals):
                enriched[entity][attribute] = sorted(vals, key=lambda x: ordinals.index(x) if x in ordinals else 99)
            else:
                enriched[entity][attribute] = sorted(vals)

    return enriched


if __name__ != "__main__" or not CONFIG.DEBUG or os.environ.get("WERKZEUG_RUN_MAIN") == "true":
    print("[AIIDKIT] Dynamic vocabulary enrichment...")
    CONFIG.VOCABULARY = _enrich_vocabulary_from_synthetic(CONFIG.VOCABULARY)

    if not _PATHS["use_mock"]:
        try:
            _load_real_model(_PATHS)
            if _COHORT_CACHE:
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
    return jsonify({
        "task":                 CONFIG.TASK,
        "available_tasks":      getattr(CONFIG, "AVAILABLE_TASKS", [{"key": CONFIG.TASK, "label": CONFIG.TASK}]),
        "default_horizon":      CONFIG.DEFAULT_HORIZON,
        "available_horizons":   CONFIG.AVAILABLE_HORIZONS,
        "horizon_labels":       {str(k): v for k, v in getattr(CONFIG, "HORIZON_LABELS", {}).items()},
        "horizon_short_labels": {str(k): v for k, v in getattr(CONFIG, "HORIZON_SHORT_LABELS", {}).items()},
        "default_fup":          CONFIG.DEFAULT_FUP,
        "available_fups":       CONFIG.AVAILABLE_FUPS,
        "precomputed_fups":     getattr(CONFIG, "PRECOMPUTED_FUPS", []),
        "using_mock_data":      not _USING_REAL_MODEL,
        "vocabulary":           CONFIG.VOCABULARY,
        "risk_threshold":       CONFIG.RISK_THRESHOLD,
    })


def _ensure_cohort_for_fup(fup: int) -> dict:
    if fup in _COHORT_CACHE:
        return _COHORT_CACHE[fup]
    if _USING_REAL_MODEL and _COHORT_CACHE:
        nearest_fup = min(_COHORT_CACHE.keys(), key=lambda k: abs(int(k) - int(fup)))
        return _COHORT_CACHE[nearest_fup]
    with _CACHE_LOCK:
        if fup not in _COHORT_CACHE:
            _COHORT_CACHE[fup] = generate_cohort_data(fup=fup, seed=CONFIG.MOCK_RANDOM_SEED)
        return _COHORT_CACHE[fup]


@app.route("/api/cohort")
def api_cohort():
    fup_str = request.args.get("fup")
    try:
        fup = int(fup_str) if fup_str is not None else CONFIG.DEFAULT_FUP
    except ValueError:
        return jsonify({"error": "Invalid FUP value. Must be an integer."}), 400

    if fup not in CONFIG.AVAILABLE_FUPS:
        return jsonify({"error": f"FUP must be one of {CONFIG.AVAILABLE_FUPS}"}), 400

    try:
        return jsonify(_ensure_cohort_for_fup(fup))
    except Exception as exc:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Failed to load cohort: {exc}"}), 500


@app.route("/api/examples/<example_type>")
def api_example_patient(example_type: str):
    mapping = {
        "low": "low_risk.csv",
        "low_risk": "low_risk.csv",
        "mod": "medium_risk.csv",
        "medium": "medium_risk.csv",
        "medium_risk": "medium_risk.csv",
        "high": "high_risk.csv",
        "high_risk": "high_risk.csv",
    }
    filename = mapping.get(example_type.lower())
    if not filename:
        return jsonify({"error": "Invalid example type. Choose 'high' or 'low'."}), 400

    file_path = Path(__file__).resolve().parent / "synthetic_patients" / filename
    if not file_path.exists():
        return jsonify({"error": f"Example file {filename} not found."}), 404

    events = []
    try:
        with open(file_path, mode="r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                days_str = row.get("days_since_tpx")
                entity = row.get("entity")
                attribute = row.get("attribute")
                value = row.get("value_binned") if row.get("value_binned") is not None else row.get("value")
                if days_str is not None and entity and attribute and value is not None:
                    try:
                        days = float(days_str)
                        days = int(days) if days.is_integer() else days
                    except ValueError:
                        days = 0
                    events.append({
                        "entity": entity.strip(),
                        "attribute": attribute.strip(),
                        "value": value.strip(),
                        "days_since_tpx": days,
                    })
        events.sort(key=lambda e: float(e["days_since_tpx"]))
        return jsonify(events)
    except Exception as exc:
        return jsonify({"error": f"Failed to load example patient: {exc}"}), 500


def _get_or_compute_prediction(
    full_events: list[dict],
    horizon: int,
    fup: int,
    task: str,
    trajectory_fups: list[int],
) -> dict:
    sorted_full = sorted(list(full_events), key=lambda e: float(e.get("days_since_tpx", 0.0)))
    sig = _events_signature(sorted_full)
    cache_key = (sig, int(fup), tuple(trajectory_fups))
    cached = _lru_get(_PREDICTION_CACHE, cache_key)

    if cached is not None:
        t_pred = cached["tasks"].get(task)
        # Lazy IG computation if user switches to a task whose IG wasn't computed yet
        if _USING_REAL_MODEL and t_pred is not None and not t_pred.get("ig_computed", False):
            cohort = _ensure_cohort_for_fup(fup)
            stage_events = _truncate_events_for_fup(sorted_full, fup)
            _, logits_by_fup = _compute_patient_trajectory_real(sorted_full, trajectory_fups)
            ig_map = _compute_task_ig_attributions(stage_events, fup, task)
            t_info = next((t for t in CONFIG.AVAILABLE_TASKS if t["key"] == task), {"key": task, "label": task})
            updated_t_pred = _assemble_task_prediction_payload(
                t_key=task,
                t_label=t_info["label"],
                full_logits_np=logits_by_fup[fup],
                umap_x=t_pred["umap_x"],
                umap_y=t_pred["umap_y"],
                events=stage_events,
                cohort=cohort,
                active_fup=fup,
                requested_horizon=horizon,
                attr_by_col=ig_map,
            )
            with _CACHE_LOCK:
                cached["tasks"][task] = updated_t_pred
            t_pred = updated_t_pred

        t_pred = t_pred or cached["tasks"].get(CONFIG.TASK) or cached
        h_pred = t_pred.get("by_horizon", {}).get(str(horizon), t_pred)
        return {
            **h_pred,
            "fup": fup,
            "trajectory": cached["trajectory"],
            "tasks": cached["tasks"],
        }

    cohort = _ensure_cohort_for_fup(fup)
    if _USING_REAL_MODEL:
        res = _real_predict(sorted_full, horizon, cohort, selected_task=task, fup=fup, trajectory_fups=trajectory_fups)
    else:
        res = mock_predict(sorted_full, horizon, cohort, selected_task=task, fup=fup, trajectory_fups=trajectory_fups)

    _lru_put(_PREDICTION_CACHE, cache_key, res, maxsize=256)
    return res


@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Request body must be JSON"}), 400

    events = data.get("events", [])
    try:
        horizon = int(data.get("horizon", CONFIG.DEFAULT_HORIZON))
        fup = int(data.get("fup", CONFIG.DEFAULT_FUP))
        max_fup = int(data.get("max_fup", fup))
    except (ValueError, TypeError):
        return jsonify({"error": "horizon, fup, and max_fup must be integers"}), 400
    task = str(data.get("task", CONFIG.TASK))

    if not events:
        return jsonify({"error": "No events provided"}), 400
    if horizon not in CONFIG.AVAILABLE_HORIZONS:
        return jsonify({"error": f"Horizon must be one of {CONFIG.AVAILABLE_HORIZONS}"}), 400
    if fup not in CONFIG.AVAILABLE_FUPS:
        return jsonify({"error": f"FUP must be one of {CONFIG.AVAILABLE_FUPS}"}), 400

    required_keys = {"entity", "attribute", "value", "days_since_tpx"}
    sanitized_events = []
    for i, ev in enumerate(events):
        if not isinstance(ev, dict):
            return jsonify({"error": f"Event {i} must be an object"}), 400
        missing = required_keys - set(ev.keys())
        if missing:
            return jsonify({"error": f"Event {i} missing fields: {sorted(missing)}"}), 400
        try:
            day_val = float(ev["days_since_tpx"])
        except (ValueError, TypeError):
            return jsonify({"error": f"Event {i}: days_since_tpx must be numeric"}), 400
        sanitized_events.append({
            "entity": str(ev["entity"]).strip(),
            "attribute": str(ev["attribute"]).strip(),
            "value": str(ev["value"]).strip(),
            "days_since_tpx": day_val,
        })

    trajectory_fups = [f for f in CONFIG.AVAILABLE_FUPS if f <= max(fup, max_fup)]
    if fup not in trajectory_fups:
        trajectory_fups = sorted(set(trajectory_fups) | {fup})

    try:
        result = _get_or_compute_prediction(sanitized_events, horizon, fup, task, trajectory_fups)
        return jsonify(result)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Prediction failed: {exc}"}), 500


if __name__ == "__main__":
    app.run(
        host=CONFIG.HOST,
        port=CONFIG.PORT,
        debug=CONFIG.DEBUG,
        exclude_patterns=getattr(CONFIG, "EXCLUDE_PATTERNS", None),
    )
