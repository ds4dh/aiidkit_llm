"""
AIIDKIT Decision Support Tool — Multi-Domain Specialist Flask Backend
=====================================================================
Serves the single-page frontend and REST API endpoints across the 5-domain
specialist Transformer suite (`B_CAT_MT`: Bacteria, Viruses, Fungi, Graft Loss,
and Mortality) as well as single-model overrides and demo mode:
  GET  /                       Serve index.html
  GET  /api/health             Unauthenticated container liveness check
  GET  /api/config             Return vocabulary, specialist domains, tasks & horizons
  GET  /api/cohort             Return pre-computed UMAP cohort for (domain, fup)
  GET  /api/examples/<type>    Return synthetic reference patient sequence
  POST /api/predict            Score patient t-EAV sequence on active specialist + IG
"""

from __future__ import annotations

from collections import OrderedDict, defaultdict
import contextlib
import csv
import gc
import hashlib
import json
import os
from pathlib import Path
import pickle
import secrets
import sys
import threading
import time

from flask import Flask, Response, jsonify, render_template, request
from flask_cors import CORS
import numpy as np
from scipy.special import expit
from werkzeug.middleware.proxy_fix import ProxyFix

_WEB_DIR = Path(__file__).resolve().parent
_ROOT = _WEB_DIR.parent
for _p in (str(_WEB_DIR), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import config as CONFIG
from bundle_io import (
    BIN_LABELS,
    build_vocabulary_concept_bank,
    compute_bin_bounds,
    estimate_imminence_from_calibrated_probs,
    load_bin_intervals_json,
    load_cohort_embeddings_npz,
    load_platt_params_json,
    load_vocab_json,
    preprocess_events_batch,
    project_embeddings_to_logits,
    rank_embedding_subgroup_concepts,
)
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
_REAL_VOCAB: dict[str, int] | None = None
_REAL_BIN_INT: dict | None = None

# Per-domain metadata, UMAP reducers, Platt calibrators, and 13-FUP cohort payloads
_DOMAINS: OrderedDict[str, dict] = OrderedDict()
_TASK_TO_DOMAIN: dict[str, str] = {}
_TASK_HORIZONS: dict[str, list[int]] = {}

# LRU pool of loaded PyTorch (model, collator) pairs keyed by domain_key
_LOADED_MODELS_LRU: OrderedDict[str, tuple] = OrderedDict()

_MODEL_LOCK = threading.Lock()
_CACHE_LOCK = threading.Lock()
_PREDICTION_CACHE: OrderedDict = OrderedDict()
_TRAJECTORY_CACHE: OrderedDict = OrderedDict()
_IG_TASK_CACHE: OrderedDict = OrderedDict()
_MOCK_COHORT_CACHE: dict[int, dict] = {}
_USING_REAL_MODEL = False
_EMBEDDINGS_ONLY_MODE = False


def _lru_get(cache: OrderedDict, key):
    with _CACHE_LOCK:
        if key not in cache:
            return None
        cache.move_to_end(key)
        return cache[key]


def _lru_put(cache: OrderedDict, key, value, maxsize: int = 192):
    with _CACHE_LOCK:
        cache[key] = value
        cache.move_to_end(key)
        while len(cache) > maxsize:
            cache.popitem(last=False)


def _autocast_ctx(device_type: str, model_dtype):
    import torch

    if model_dtype in (torch.float16, torch.bfloat16):
        return torch.autocast(device_type=device_type, dtype=model_dtype)
    return contextlib.nullcontext()


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


def _apply_platt_vector(logits_vec: np.ndarray, cols: list[int], platt_map: dict[int, tuple[float, float]]) -> np.ndarray:
    out = np.empty(len(cols), dtype=np.float64)
    for i, col_idx in enumerate(cols):
        coef, intercept = platt_map.get(int(col_idx), (1.0, 0.0))
        out[i] = expit(coef * float(logits_vec[i]) + intercept)
    return out


def _apply_platt_matrix(logits_mat: np.ndarray, cols: list[int], platt_map: dict[int, tuple[float, float]]) -> np.ndarray:
    coefs = np.array([platt_map.get(int(c), (1.0, 0.0))[0] for c in cols], dtype=np.float64)
    intercepts = np.array([platt_map.get(int(c), (1.0, 0.0))[1] for c in cols], dtype=np.float64)
    return expit(logits_mat * coefs[None, :] + intercepts[None, :])


def _pool_model_embeddings(model, outputs, attn_mask):
    import torch

    last_hidden = outputs.hidden_states[-1]
    pooling_mode = getattr(getattr(model, "config", None), "classifier_pooling", "mean")
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


# ---------------------------------------------------------------------------
# Specialist Domain Model Pool & Cohort Assembly
# ---------------------------------------------------------------------------

def _register_domain_spec(d_key: str, d_spec: dict, use_gpu: bool) -> dict:
    """Parse checkpoint config & `run_metadata.json` for one specialist domain without loading weights yet."""
    model_dir = Path(d_spec["model_path"])
    cfg_dict = json.loads((model_dir / "config.json").read_text(encoding="utf-8")) if (model_dir / "config.json").exists() else {}
    run_meta = d_spec.get("run_metadata")
    if run_meta is None:
        for cand in (model_dir / "run_metadata.json", model_dir.parent / "run_metadata.json"):
            if cand.exists():
                run_meta = json.loads(cand.read_text(encoding="utf-8"))
                break
    run_meta = run_meta or {}

    raw_tasks = list(run_meta.get("tasks") or [CONFIG.TASK])
    raw_horizons = sorted(int(h) for h in (run_meta.get("horizons") or [30, 60, 90]))
    num_labels = len(cfg_dict.get("id2label", {})) or len(run_meta.get("label_keys", [])) or (len(raw_tasks) * len(raw_horizons))
    problem_type = cfg_dict.get("problem_type", "multi_label_classification")
    enforce_monotonicity = build_monotonic_groups(run_meta, cfg_dict.get("enforce_monotonicity", True))

    all_label_keys = (
        list(run_meta["label_keys"])
        if isinstance(run_meta.get("label_keys"), list) and len(run_meta["label_keys"]) == num_labels
        else [f"label_{t}_{h:04d}d" for t in raw_tasks for h in raw_horizons]
    )
    use_num_emb = bool(run_meta.get("use_numerical_embedding", True))

    task_cols_map: dict[str, list[int]] = {}
    task_horizons_map: dict[str, list[int]] = {}
    domain_tasks_meta = []
    for t_info in build_available_tasks_list(raw_tasks):
        t_key = t_info["key"]
        t_hrzs = sorted(int(h) for h in run_meta.get("task_horizons", {}).get(t_key, raw_horizons))
        t_lks = [f"label_{t_key}_{h:04d}d" for h in t_hrzs]
        if all(k in all_label_keys for k in t_lks):
            task_cols_map[t_key] = [all_label_keys.index(k) for k in t_lks]
            task_horizons_map[t_key] = t_hrzs
            domain_tasks_meta.append(t_info)

    default_h = 90 if 90 in raw_horizons else (360 if 360 in raw_horizons else raw_horizons[0])
    return {
        "domain": d_key,
        "label": d_spec.get("label", d_key.replace("_", " ").title()),
        "model_dir": model_dir,
        "use_gpu": use_gpu,
        "run_meta": run_meta,
        "num_labels": num_labels,
        "problem_type": problem_type,
        "enforce_monotonicity": enforce_monotonicity,
        "all_label_keys": all_label_keys,
        "use_num_emb": use_num_emb,
        "tasks_meta": domain_tasks_meta,
        "horizons": raw_horizons,
        "default_horizon": default_h,
        "task_cols_map": task_cols_map,
        "task_horizons_map": task_horizons_map,
        "platt_path": d_spec.get("platt_path"),
        "embeddings_path": d_spec.get("embeddings_path"),
        "domain_fingerprint": d_spec.get("domain_fingerprint"),
        "platt_params_by_col": {},
        "cohort_cache": {},
        "umap_reducer": None,
        "global_umap_bounds": None,
        "logits_median_by_task": {},
        "logits_std_by_task": {},
    }


def _get_domain_model_and_collator(d_key: str) -> tuple:
    """Thread-safe LRU loader for specialist `(model, collator)`. Must be called under `_MODEL_LOCK`."""
    import torch
    from src.model.patient_embedder import (
        PatientDataCollatorForClassification,
        PatientEmbeddingModelFactory,
    )

    if d_key in _LOADED_MODELS_LRU:
        _LOADED_MODELS_LRU.move_to_end(d_key)
        return _LOADED_MODELS_LRU[d_key]

    dom = _DOMAINS[d_key]
    max_in_ram = max(1, int(getattr(CONFIG, "MAX_MODELS_IN_RAM", 5)))
    while len(_LOADED_MODELS_LRU) >= max_in_ram:
        evicted_key, (ev_model, _) = _LOADED_MODELS_LRU.popitem(last=False)
        del ev_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"[AIIDKIT] Evicted LRU specialist '{evicted_key}' from RAM pool (max={max_in_ram}).")

    device = "cuda" if (dom["use_gpu"] and torch.cuda.is_available()) else "cpu"
    model_load_args = {"num_labels": dom["num_labels"], "problem_type": dom["problem_type"]}
    if device == "cpu":
        model_load_args["dtype"] = "float32"

    t0 = time.time()
    model = PatientEmbeddingModelFactory.from_pretrained(
        task="classification",
        pretrained_dir=str(dom["model_dir"]),
        embedding_layer_config={
            "vocab_size": len(_REAL_VOCAB),
            "eav_mappings": CONFIG.EAV_MAPPINGS,
            "time_mapping": CONFIG.TIME_MAPPING,
            "use_numerical_embedding": dom["use_num_emb"],
        },
        model_args=model_load_args,
        enforce_monotonicity=dom["enforce_monotonicity"],
    )
    model.eval()
    model.to(device)

    collator = PatientDataCollatorForClassification(
        eav_mappings=CONFIG.EAV_MAPPINGS,
        time_mapping=CONFIG.TIME_MAPPING,
        use_numerical_embedding=dom["use_num_emb"],
        label_keys=dom["all_label_keys"],
        max_position_embeddings=getattr(model.config, "max_position_embeddings", 1024),
    )
    _LOADED_MODELS_LRU[d_key] = (model, collator)
    print(f"[AIIDKIT] Loaded specialist '{d_key}' ({dom['label']}) on {device} in {time.time() - t0:.2f}s.")
    return model, collator


def _build_domain_fup_cohort_payload(dom: dict, shaded: dict, coords_2d: np.ndarray, global_bounds: dict) -> dict:
    import hdbscan
    from sklearn.cluster import KMeans
    import umap

    patient_ids = shaded["patient_ids"]
    full_logits_all = shaded["full_logits_all"]
    embeddings = shaded["embeddings"]
    n_patients = shaded["n_patients"]
    fup_val = shaded.get("fup")
    patient_summaries = shaded["patient_summaries"]
    task_outcomes = shaded["task_outcomes"]
    concept_bank = shaded.get("concept_bank")

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
            km = KMeans(n_clusters=2, n_init=10, random_state=42)
            cluster_labels = km.fit_predict(cluster_input)
            unique_labels = [0, 1]

    tasks_data = {}
    platt_map = dom["platt_params_by_col"]
    for t_info in dom["tasks_meta"]:
        task_key = t_info["key"]
        cols = dom["task_cols_map"][task_key]
        t_horizons = dom["task_horizons_map"][task_key]
        ref_h = dom["default_horizon"] if dom["default_horizon"] in t_horizons else t_horizons[min(2, len(t_horizons) - 1)]
        horizon_idx = t_horizons.index(ref_h)

        raw_logits_all = full_logits_all[:, cols]
        t_median = np.median(raw_logits_all, axis=0)
        t_std = np.std(raw_logits_all, axis=0)
        dom["logits_median_by_task"][task_key] = t_median
        dom["logits_std_by_task"][task_key] = t_std
        std_clamped = np.clip(t_std, a_min=1e-6, a_max=None)

        probs = expit((raw_logits_all - t_median[None, :]) / std_clamped[None, :])
        cal_probs = _apply_platt_matrix(raw_logits_all, cols, platt_map)

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
        for c_id in ui_cluster_labels:
            cluster_sizes[c_id] += 1

        precomputed_drivers = rank_embedding_subgroup_concepts(
            concept_bank=concept_bank,
            embeddings=embeddings,
            ui_cluster_labels=ui_cluster_labels,
            horizon_col_idx=cols[horizon_idx],
            top_k=8,
        )

        t_outcomes = task_outcomes.get(task_key, {})
        has_task_labels = bool(t_outcomes.get("labels_by_h"))
        task_event_info = t_outcomes.get("event_info", [(None, 0.0)] * n_patients)
        labels_by_h = t_outcomes.get("labels_by_h", {})
        pred_imminence = (
            estimate_imminence_from_calibrated_probs(cal_probs, t_horizons)
            if not has_task_labels
            else None
        )

        horizons_data = {}
        for idx_h, h in enumerate(t_horizons):
            h_labels = labels_by_h.get(h, [-100] * n_patients)
            h_patients = []
            for i in range(n_patients):
                c_id = ui_cluster_labels[i]
                raw_lbl = int(h_labels[i])
                days_to_event, event_imminence = task_event_info[i]
                if pred_imminence is not None:
                    event_imminence = float(pred_imminence[i])
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
                uncensored_labels = [
                    h_patients[i]["true_label"]
                    for i in range(n_patients)
                    if ui_cluster_labels[i] == c_id and h_patients[i]["true_label"] >= 0
                ]
                if uncensored_labels:
                    event_rate = float(np.mean(uncensored_labels))
                    rate_label = "Event Rate"
                else:
                    c_cal = [
                        h_patients[i]["calibrated_risk"]
                        for i in range(n_patients)
                        if ui_cluster_labels[i] == c_id
                    ]
                    event_rate = float(np.mean(c_cal)) if c_cal else 0.0
                    rate_label = "Mean Risk"
                cluster_profiles[str(c_id)] = {
                    "name":         cluster_names_lookup.get(c_id, "Medium risk"),
                    "n_patients":   cluster_sizes.get(c_id, 0),
                    "event_rate":   round(event_rate, 4),
                    "rate_label":   rate_label,
                    "color":        cluster_colors_lookup.get(c_id, "#FFA502"),
                    "top_features": precomputed_drivers.get(c_id, {"static": [], "recent": []}),
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

    default_task_data = next(iter(tasks_data.values()))
    return {
        "domain":             dom["domain"],
        "domain_label":       dom["label"],
        "fup":                fup_val,
        "global_umap_bounds": global_bounds,
        "horizons":           default_task_data["horizons"],
        "logits_median":      default_task_data["logits_median"],
        "logits_std":         default_task_data["logits_std"],
        "tasks":              tasks_data,
    }


def _load_or_build_domain_from_embeddings(
    dom: dict,
    embs_by_fup: dict[int, np.ndarray],
    labels_mat_by_fup: dict[int, np.ndarray],
    anon_ids_by_fup: dict[int, np.ndarray] | None = None,
):
    import umap

    d_key = dom["domain"]
    t0 = time.time()
    fp = dom.get("domain_fingerprint") or hashlib.sha256(
        f"{d_key}:{(dom['model_dir'] / 'model.safetensors').stat().st_size}".encode("utf-8")
    ).hexdigest()[:16]

    rt_cfg = {
        "version": "v11_multi_domain_consistent_traj",
        "domain": d_key,
        "domain_fingerprint": fp,
        "umap_fit_sample_size": int(getattr(CONFIG, "UMAP_FIT_SAMPLE_SIZE", 1950)),
        "umap_n_neighbors": int(getattr(CONFIG, "UMAP_N_NEIGHBORS", 15)),
        "umap_min_dist": float(getattr(CONFIG, "UMAP_MIN_DIST", 0.1)),
        "umap_random_state": int(getattr(CONFIG, "UMAP_RANDOM_STATE", 42)),
        "hdbscan_min_cluster_size": int(getattr(CONFIG, "HDBSCAN_MIN_CLUSTER_SIZE", 15)),
        "hdbscan_min_samples": int(getattr(CONFIG, "HDBSCAN_MIN_SAMPLES", 5)),
    }
    rt_hash = hashlib.sha256(json.dumps(rt_cfg, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    rt_dir = _ensure_private_dir(Path(getattr(CONFIG, "RUNTIME_CACHE_DIR", Path("/tmp/aiidkit_runtime_cache"))))
    rt_file = rt_dir / f"derived_domain_{d_key}_{rt_hash}.pkl"

    if rt_file.exists():
        try:
            with open(rt_file, "rb") as f:
                cached = pickle.load(f)
            dom["cohort_cache"] = cached["cohort_cache"]
            dom["umap_reducer"] = cached["umap_reducer"]
            dom["global_umap_bounds"] = cached["global_umap_bounds"]
            dom["logits_median_by_task"] = cached["logits_median_by_task"]
            dom["logits_std_by_task"] = cached["logits_std_by_task"]
            print(
                f"[AIIDKIT] Warm-loaded '{d_key}' ({dom['label']}) UMAP cohort cache "
                f"({len(dom['cohort_cache'])} FUPs) in {time.time() - t0:.2f}s."
            )
            return
        except Exception as exc:
            print(f"[AIIDKIT] Recomputing '{d_key}' UMAP cache ({exc})...")

    with _MODEL_LOCK:
        model, collator = _get_domain_model_and_collator(d_key)
        concept_bank = build_vocabulary_concept_bank(
            model=model,
            collator=collator,
            vocab=_REAL_VOCAB,
            bin_intervals=_REAL_BIN_INT,
            vocabulary_cfg=CONFIG.VOCABULARY,
            static_entities=STATIC_ENTITY_SET | {"Patient", "Donor"},
            time_mapping=CONFIG.TIME_MAPPING,
            eav_mappings=CONFIG.EAV_MAPPINGS,
            pool_fn=lambda out, am: _pool_model_embeddings(model, out, am),
        )
        logits_by_fup = {
            fup: project_embeddings_to_logits(model, embs)
            for fup, embs in sorted(embs_by_fup.items())
        }

    fit_sample_size = int(getattr(CONFIG, "UMAP_FIT_SAMPLE_SIZE", 1950))
    per_fup_sample = max(50, fit_sample_size // len(embs_by_fup))
    rng = np.random.default_rng(getattr(CONFIG, "UMAP_RANDOM_STATE", 42))
    pooled_for_fit = []
    for _, embs in sorted(embs_by_fup.items()):
        if len(embs) > per_fup_sample:
            idx = rng.choice(len(embs), size=per_fup_sample, replace=False)
            pooled_for_fit.append(embs[np.sort(idx)])
        else:
            pooled_for_fit.append(embs)
    pooled_fit_matrix = np.vstack(pooled_for_fit)

    print(f"[AIIDKIT] Fitting UMAP manifold for '{d_key}' on {len(pooled_fit_matrix)} embeddings...")
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=getattr(CONFIG, "UMAP_N_NEIGHBORS", 15),
        min_dist=getattr(CONFIG, "UMAP_MIN_DIST", 0.1),
        random_state=getattr(CONFIG, "UMAP_RANDOM_STATE", 42),
    )
    reducer.fit(pooled_fit_matrix)
    dom["umap_reducer"] = reducer

    fup_items = sorted(embs_by_fup.items())
    stacked_all = np.vstack([embs for _, embs in fup_items])
    stacked_coords = reducer.transform(stacked_all)
    coords_by_fup = {}
    offset = 0
    for fup, embs in fup_items:
        n_f = len(embs)
        coords_by_fup[fup] = stacked_coords[offset : offset + n_f]
        offset += n_f

    all_x_np = stacked_coords[:, 0]
    all_y_np = stacked_coords[:, 1]
    bounds = {
        "min_x": round(float(np.percentile(all_x_np, 0.5)), 3),
        "max_x": round(float(np.percentile(all_x_np, 99.5)), 3),
        "min_y": round(float(np.percentile(all_y_np, 0.5)), 3),
        "max_y": round(float(np.percentile(all_y_np, 99.5)), 3),
    }
    dom["global_umap_bounds"] = bounds

    for fup, embs in fup_items:
        n_pts = int(embs.shape[0])
        fup_task_outcomes = {}
        lbl_mat = labels_mat_by_fup.get(fup)
        if lbl_mat is not None and lbl_mat.shape[0] == n_pts:
            for t_info in dom["tasks_meta"]:
                t_key = t_info["key"]
                cols = dom["task_cols_map"][t_key]
                t_horizons = dom["task_horizons_map"][t_key]
                if len(cols) == len(t_horizons) and max(cols, default=-1) < lbl_mat.shape[1]:
                    labels_by_h = {
                        h: [int(v) for v in lbl_mat[:, cols[idx_h]]]
                        for idx_h, h in enumerate(t_horizons)
                    }
                    event_info = []
                    for row_i in range(n_pts):
                        prev_h = 0
                        ev_tuple = (None, 0.0)
                        for idx_h, h in enumerate(t_horizons):
                            if int(lbl_mat[row_i, cols[idx_h]]) == 1:
                                d_ev = int((prev_h + h) / 2)
                                ev_tuple = (d_ev, float(np.exp(-d_ev / 365.0)))
                                break
                            prev_h = h
                        event_info.append(ev_tuple)
                    fup_task_outcomes[t_key] = {"event_info": event_info, "labels_by_h": labels_by_h}

        anon_arr = (anon_ids_by_fup or {}).get(fup)
        if anon_arr is not None and len(anon_arr) == n_pts:
            p_ids = [f"REF_{int(aid):03d}" for aid in anon_arr]
        else:
            p_ids = [f"REF_{i + 1:03d}" for i in range(n_pts)]

        shaded_stub = {
            "fup": fup,
            "concept_bank": concept_bank,
            "patient_ids": p_ids,
            "full_logits_all": logits_by_fup[fup],
            "embeddings": embs,
            "n_patients": n_pts,
            "patient_summaries": ["Standard profile"] * n_pts,
            "task_outcomes": fup_task_outcomes,
        }
        dom["cohort_cache"][fup] = _build_domain_fup_cohort_payload(dom, shaded_stub, coords_by_fup[fup], bounds)

    try:
        tmp_rt = rt_file.with_suffix(f".{os.getpid()}.tmp")
        with open(tmp_rt, "wb") as f:
            pickle.dump(
                {
                    "cohort_cache": dom["cohort_cache"],
                    "umap_reducer": dom["umap_reducer"],
                    "global_umap_bounds": dom["global_umap_bounds"],
                    "logits_median_by_task": dom["logits_median_by_task"],
                    "logits_std_by_task": dom["logits_std_by_task"],
                },
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        _chmod_private_file(tmp_rt)
        tmp_rt.replace(rt_file)
    except OSError:
        pass

    print(
        f"[AIIDKIT] Built '{d_key}' ({dom['label']}) cohort manifold across "
        f"{len(dom['cohort_cache'])} FUPs in {time.time() - t0:.1f}s."
    )


def _extract_raw_domain_embeddings_and_platt(dom: dict, data_dir: Path) -> tuple[dict[int, tuple[float, float]], dict[int, np.ndarray], dict[int, np.ndarray]]:
    """Fallback when running on HPC directly from raw `DATA_DIR` without an exported `server_bundle/`."""
    import torch
    from datasets import disable_caching, load_from_disk
    from sklearn.linear_model import LogisticRegression

    disable_caching()
    d_key = dom["domain"]
    bin_bounds = compute_bin_bounds(_REAL_BIN_INT)
    label_keys = dom["all_label_keys"]
    num_labels = dom["num_labels"]

    with _MODEL_LOCK:
        model, collator = _get_domain_model_and_collator(d_key)
        device = next(model.parameters()).device
        model_dtype = next(model.parameters()).dtype
        device_type = "cuda" if "cuda" in str(device) else "cpu"

        def _score_split_dir(sp_dir: Path, split_name: str, max_rows: int, need_embs: bool):
            ds = load_from_disk(str(sp_dir))
            n_r = min(len(ds), max_rows)
            sub = ds.select(range(n_r), keep_in_memory=True)
            proc = preprocess_events_batch(
                batch=sub[:],
                vocab=_REAL_VOCAB,
                bin_intervals=_REAL_BIN_INT,
                time_mapping=CONFIG.TIME_MAPPING,
                eav_mappings=CONFIG.EAV_MAPPINGS,
                bin_labels=BIN_LABELS,
                bin_bounds=bin_bounds,
            )
            samples = []
            for i in range(n_r):
                s = {
                    "entity_id": np.array(proc["entity_id"][i]),
                    "attribute_id": np.array(proc["attribute_id"][i]),
                    "value_id": np.array(proc["value_id"][i]),
                    "value_num": np.array(proc["value_num"][i], dtype=np.float32),
                    "days_since_tpx": np.array(proc["days_since_tpx"][i]),
                    "split": split_name,
                }
                for lk in label_keys:
                    s[lk] = 0.0
                samples.append(s)

            lbl_mat = np.full((n_r, num_labels), -100, dtype=np.int8)
            for j, lk in enumerate(label_keys):
                if lk in sub.column_names:
                    raw = np.asarray(sub[lk], dtype=np.int64)
                    lbl_mat[:, j] = np.where(raw == -100, -100, np.where(raw == 1, 1, 0)).astype(np.int8)

            log_chunks, emb_chunks = [], []
            for idx in range(0, n_r, 48):
                b_coll = collator.torch_call(samples[idx : idx + 48])
                inp = {
                    k: (v.to(device, dtype=model_dtype) if v.is_floating_point() else v.to(device))
                    for k, v in b_coll["input_dict"].items()
                }
                am = b_coll["attention_mask"].to(device)
                with torch.no_grad():
                    with _autocast_ctx(device_type, model_dtype):
                        out = model(input_dict=inp, attention_mask=am, output_hidden_states=need_embs)
                        log_chunks.append(out.logits.float().cpu().numpy())
                        if need_embs:
                            emb_chunks.append(_pool_model_embeddings(model, out, am).cpu().numpy())
            del ds, sub, proc, samples
            return np.vstack(log_chunks), (np.vstack(emb_chunks) if need_embs else None), lbl_mat

        val_logs, val_lbls = [], []
        for vf in (0, 30, 60, 90, 120, 150, 180, 240, 300, 360):
            vd = data_dir / f"fup_{vf:04d}" / "validation"
            if vd.exists():
                vl, _, vlb = _score_split_dir(vd, "validation", 400, need_embs=False)
                val_logs.append(vl)
                val_lbls.append(vlb)

        platt_params: dict[int, tuple[float, float]] = {}
        if val_logs:
            all_vl = np.vstack(val_logs)
            all_vy = np.vstack(val_lbls)
            for j in range(num_labels):
                yc = all_vy[:, j]
                valid = (yc == 0) | (yc == 1)
                if valid.sum() < 10 or np.unique(yc[valid]).size < 2:
                    platt_params[j] = (1.0, 0.0)
                    continue
                yb = yc[valid].astype(np.int64)
                xc = np.clip(all_vl[valid, j], -15.0, 15.0).reshape(-1, 1)
                slope, intercept, fitted = 1.0, 0.0, False
                if int(np.sum(yb == 1)) >= 20:
                    try:
                        clf = LogisticRegression(C=1.0, solver="lbfgs", max_iter=500)
                        clf.fit(xc, yb)
                        if float(clf.coef_[0, 0]) > 0.05:
                            slope, intercept = float(clf.coef_[0, 0]), float(clf.intercept_[0])
                            fitted = True
                    except Exception:
                        pass
                if not fitted:
                    pb = float(np.clip(np.mean(yb), 1e-6, 1.0 - 1e-6))
                    slope, intercept = 1.0, float(np.log(pb / (1.0 - pb)) - np.mean(xc))
                platt_params[j] = (slope, intercept)

        embs_by_fup, labels_by_fup = {}, {}
        max_pts = int(getattr(CONFIG, "MAX_COHORT_PATIENTS", 500))
        for tf in getattr(CONFIG, "PRECOMPUTED_FUPS", [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360]):
            td = data_dir / f"fup_{tf:04d}" / "test"
            if td.exists():
                _, embs, tlb = _score_split_dir(td, "test", max_pts, need_embs=True)
                embs_by_fup[int(tf)] = embs.astype(np.float32)
                labels_by_fup[int(tf)] = tlb

    return platt_params, embs_by_fup, labels_by_fup


def _initialize_all_domains(paths: dict):
    global _REAL_VOCAB, _REAL_BIN_INT, _DOMAINS, _TASK_TO_DOMAIN, _TASK_HORIZONS
    global _USING_REAL_MODEL, _EMBEDDINGS_ONLY_MODE

    vocab_path = Path(paths["vocab_path"]) if paths.get("vocab_path") else None
    bin_path = Path(paths["bin_intervals"]) if paths.get("bin_intervals") else None
    if vocab_path is None or not vocab_path.exists() or bin_path is None or not bin_path.exists():
        raise FileNotFoundError(f"Vocabulary or bin_intervals missing: vocab={vocab_path}, bins={bin_path}")

    _REAL_VOCAB = load_vocab_json(vocab_path) if vocab_path.suffix.lower() == ".json" else pickle.loads(vocab_path.read_bytes())
    _REAL_BIN_INT = load_bin_intervals_json(bin_path) if bin_path.suffix.lower() == ".json" else pickle.loads(bin_path.read_bytes())

    is_bundle = paths.get("bundle_dir") is not None
    _EMBEDDINGS_ONLY_MODE = is_bundle
    _DOMAINS.clear()
    _TASK_TO_DOMAIN.clear()
    _TASK_HORIZONS.clear()

    ordered_tasks: list[dict] = []
    all_horizons_union: set[int] = set()
    all_fups_union: set[int] = set()

    for d_key, d_spec in paths["domains"].items():
        dom = _register_domain_spec(d_key, d_spec, use_gpu=bool(paths.get("use_gpu", True)))
        if is_bundle:
            dom["platt_params_by_col"] = load_platt_params_json(Path(dom["platt_path"]))
            embs_by_fup, labels_by_fup, anon_ids_by_fup = load_cohort_embeddings_npz(Path(dom["embeddings_path"]))
        else:
            platt_map, embs_by_fup, labels_by_fup = _extract_raw_domain_embeddings_and_platt(dom, Path(paths["data_dir"]))
            dom["platt_params_by_col"] = platt_map
            anon_ids_by_fup = {}

        _DOMAINS[d_key] = dom
        _load_or_build_domain_from_embeddings(dom, embs_by_fup, labels_by_fup, anon_ids_by_fup)
        all_fups_union.update(dom["cohort_cache"].keys())

        for t_info in dom["tasks_meta"]:
            t_key = t_info["key"]
            t_hrzs = dom["task_horizons_map"][t_key]
            _TASK_TO_DOMAIN[t_key] = d_key
            _TASK_HORIZONS[t_key] = list(t_hrzs)
            all_horizons_union.update(t_hrzs)
            ordered_tasks.append({
                **t_info,
                "domain": d_key,
                "domain_label": dom["label"],
                "horizons": list(t_hrzs),
            })

    if ordered_tasks:
        CONFIG.AVAILABLE_TASKS = ordered_tasks
        valid_t_keys = [t["key"] for t in ordered_tasks]
        if CONFIG.TASK not in valid_t_keys:
            CONFIG.TASK = valid_t_keys[0]
        CONFIG.AVAILABLE_HORIZONS = sorted(all_horizons_union)
        def_hrzs = _TASK_HORIZONS.get(CONFIG.TASK, CONFIG.AVAILABLE_HORIZONS)
        CONFIG.DEFAULT_HORIZON = 90 if 90 in def_hrzs else def_hrzs[0]
        CONFIG.HORIZON_LABELS = build_horizon_labels_map(CONFIG.AVAILABLE_HORIZONS, style="long")
        CONFIG.HORIZON_SHORT_LABELS = build_horizon_labels_map(CONFIG.AVAILABLE_HORIZONS, style="short")
    if all_fups_union:
        CONFIG.PRECOMPUTED_FUPS = sorted(int(f) for f in all_fups_union)
        CONFIG.AVAILABLE_FUPS = list(CONFIG.PRECOMPUTED_FUPS)

    # Pre-warm default domain model (and remaining domains if MAX_MODELS_IN_RAM >= len(_DOMAINS) on GPU)
    default_domain = _TASK_TO_DOMAIN.get(CONFIG.TASK, next(iter(_DOMAINS.keys())))
    with _MODEL_LOCK:
        _get_domain_model_and_collator(default_domain)
        if bool(paths.get("use_gpu", True)) and int(getattr(CONFIG, "MAX_MODELS_IN_RAM", 5)) >= len(_DOMAINS):
            import torch
            if torch.cuda.is_available():
                for dk in _DOMAINS.keys():
                    _get_domain_model_and_collator(dk)

    _USING_REAL_MODEL = bool(_DOMAINS)
    print(
        f"[AIIDKIT] Initialized {len(_DOMAINS)} specialist domain(s): "
        f"{[f'{k} ({len(v['tasks_meta'])}t)' for k, v in _DOMAINS.items()]} "
        f"| models_in_ram={len(_LOADED_MODELS_LRU)}/{getattr(CONFIG, 'MAX_MODELS_IN_RAM', 5)}"
    )


# ---------------------------------------------------------------------------
# Live Patient Trajectory & Value-Pathway Integrated Gradients (Domain-Routed)
# ---------------------------------------------------------------------------

def _compute_domain_trajectory_real(
    d_key: str,
    full_events: list[dict],
    fup_steps: list[int],
) -> tuple[list[dict], dict[int, np.ndarray]]:
    import torch

    if not fup_steps:
        return [], {}

    cache_key = (d_key, _events_signature(full_events), tuple(fup_steps))
    cached_val = _lru_get(_TRAJECTORY_CACHE, cache_key)
    if cached_val is not None:
        return cached_val

    dom = _DOMAINS[d_key]
    truncated_lists = []
    processed_rows = []
    bin_bounds = compute_bin_bounds(_REAL_BIN_INT)

    for f_step in fup_steps:
        evs_f = _truncate_events_for_fup(full_events, f_step)
        truncated_lists.append(evs_f)
        batch_raw = {
            "entity":    [e["entity"] for e in evs_f],
            "attribute": [e["attribute"] for e in evs_f],
            "value":     [e["value"] for e in evs_f],
            "time":      [e.get("days_since_tpx", 0.0) for e in evs_f],
        }
        processed_rows.append(
            preprocess_events_batch(
                {k: [v] for k, v in batch_raw.items()},
                vocab=_REAL_VOCAB,
                bin_intervals=_REAL_BIN_INT,
                bin_labels=BIN_LABELS,
                time_mapping=CONFIG.TIME_MAPPING,
                eav_mappings=CONFIG.EAV_MAPPINGS,
                bin_bounds=bin_bounds,
            )
        )

    with _MODEL_LOCK:
        model, collator = _get_domain_model_and_collator(d_key)
        device = next(model.parameters()).device
        model_dtype = next(model.parameters()).dtype
        device_type = "cuda" if "cuda" in str(device) else "cpu"

        samples = []
        for proc in processed_rows:
            sample = {
                "entity_id":      np.array(proc["entity_id"][0]),
                "attribute_id":   np.array(proc["attribute_id"][0]),
                "value_id":       np.array(proc["value_id"][0]),
                "value_num":      np.array(proc["value_num"][0], dtype=np.float32),
                "days_since_tpx": np.array(proc["days_since_tpx"][0]),
                "split":          "test",
            }
            for lk in collator.label_keys:
                sample[lk] = 0.0
            samples.append(sample)

        batch_collated = collator.torch_call(samples)
        input_dict = {
            k: (v.to(device, dtype=model_dtype) if v.is_floating_point() else v.to(device))
            for k, v in batch_collated["input_dict"].items()
        }
        attn_mask = batch_collated["attention_mask"].to(device)
        with torch.no_grad():
            with _autocast_ctx(device_type, model_dtype):
                outputs = model(input_dict=input_dict, attention_mask=attn_mask, output_hidden_states=True)
                logits_batch = outputs.logits.detach().float().cpu().numpy()
                pooled_batch = _pool_model_embeddings(model, outputs, attn_mask).detach().float().cpu().numpy()

    reducer = dom.get("umap_reducer")
    if reducer is not None:
        try:
            coords_batch = reducer.transform(pooled_batch)
        except Exception as exc:
            print(f"[AIIDKIT] WARNING: Trajectory UMAP transform failed for '{d_key}': {exc}")
            coords_batch = np.zeros((len(fup_steps), 2))
    else:
        coords_batch = np.zeros((len(fup_steps), 2))

    trajectory = []
    logits_by_fup: dict[int, np.ndarray] = {}
    platt_map = dom["platt_params_by_col"]

    for idx, f_step in enumerate(fup_steps):
        logits_by_fup[f_step] = logits_batch[idx]
        ux = round(float(coords_batch[idx, 0]), 3)
        uy = round(float(coords_batch[idx, 1]), 3)
        f_cohort = _ensure_cohort_for_domain_fup(d_key, f_step)
        f_cohort_tasks = f_cohort.get("tasks", {})
        t_scores = {}
        for t_info in dom["tasks_meta"]:
            t_key = t_info["key"]
            cols = dom["task_cols_map"][t_key]
            t_horizons = dom["task_horizons_map"][t_key]
            l_np = logits_batch[idx, cols]
            tc = f_cohort_tasks.get(t_key, f_cohort)
            c_med = np.array(tc.get("logits_median", dom["logits_median_by_task"].get(t_key, np.zeros(len(cols)))), dtype=float)
            c_std = np.clip(np.array(tc.get("logits_std", dom["logits_std_by_task"].get(t_key, np.ones(len(cols)))), dtype=float), 1e-6, None)
            ev_sc = expit((l_np - c_med) / c_std)
            cal_pr = _apply_platt_vector(l_np, cols, platt_map)
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
    _lru_put(_TRAJECTORY_CACHE, cache_key, result_pair, maxsize=256)
    return result_pair


def _compute_task_ig_attributions(d_key: str, events: list[dict], active_fup: int, task_key: str) -> dict[int, np.ndarray]:
    import torch

    dom = _DOMAINS[d_key]
    cols = dom["task_cols_map"][task_key]
    ig_key = (d_key, _events_signature(events), int(active_fup), str(task_key))
    cached_ig = _lru_get(_IG_TASK_CACHE, ig_key)
    if cached_ig is not None:
        return cached_ig

    batch_raw = {
        "entity":    [e["entity"] for e in events],
        "attribute": [e["attribute"] for e in events],
        "value":     [e["value"] for e in events],
        "time":      [e.get("days_since_tpx", 0.0) for e in events],
    }
    processed = preprocess_events_batch(
        {k: [v] for k, v in batch_raw.items()},
        vocab=_REAL_VOCAB,
        bin_intervals=_REAL_BIN_INT,
        bin_labels=BIN_LABELS,
        time_mapping=CONFIG.TIME_MAPPING,
        eav_mappings=CONFIG.EAV_MAPPINGS,
    )

    event_scores_by_col: dict[int, np.ndarray] = {}
    n_events = len(events)

    try:
        with _MODEL_LOCK:
            model, collator = _get_domain_model_and_collator(d_key)
            device = next(model.parameters()).device
            model_dtype = next(model.parameters()).dtype
            device_type = "cuda" if "cuda" in str(device) else "cpu"
            max_pos = int(getattr(model.config, "max_position_embeddings", 1024))
            cfg_ig_steps = getattr(CONFIG, "IG_STEPS", None)
            n_steps = int(cfg_ig_steps) if cfg_ig_steps else (32 if device_type == "cuda" else 16)

            sample = {
                "entity_id":      np.array(processed["entity_id"][0]),
                "attribute_id":   np.array(processed["attribute_id"][0]),
                "value_id":       np.array(processed["value_id"][0]),
                "value_num":      np.array(processed["value_num"][0], dtype=np.float32),
                "days_since_tpx": np.array(processed["days_since_tpx"][0]),
                "split":          "test",
            }
            for lk in collator.label_keys:
                sample[lk] = 0.0

            batch = collator.torch_call([sample])
            input_dict = {
                k: (v.to(device, dtype=model_dtype) if v.is_floating_point() else v.to(device))
                for k, v in batch["input_dict"].items()
            }
            attn_mask = batch["attention_mask"].to(device)

            val = input_dict["value_id"]
            baseline_val = torch.where(val < 5, val, torch.zeros_like(val))
            baseline_val_num = (
                torch.full_like(input_dict["value_num"], float("nan"))
                if "value_num" in input_dict
                else None
            )

            with torch.no_grad():
                inp_val_emb = model.patient_embedder.encode_values(val, input_dict.get("value_num")).detach().float()
                base_val_emb = model.patient_embedder.encode_values(baseline_val, baseline_val_num).detach().float()
                diff_val_emb = (inp_val_emb - base_val_emb).squeeze(0)
                alphas = torch.linspace(1.0 / n_steps, 1.0, n_steps, device=device, dtype=torch.float32).view(n_steps, 1, 1)
                interp_val_emb = (base_val_emb + alphas * (inp_val_emb - base_val_emb)).to(dtype=model_dtype).detach()

            interp_val_emb.requires_grad_(True)
            rep_input_dict = {k: v.repeat(n_steps, 1) for k, v in input_dict.items()}
            rep_attn_mask = attn_mask.repeat(n_steps, 1)

            hook_handle = model.patient_embedder.value_embedding_hook.register_forward_hook(
                lambda module, inputs, output: interp_val_emb
            )
            try:
                with torch.enable_grad():
                    with _autocast_ctx(device_type, model_dtype):
                        outputs_ig = model(input_dict=rep_input_dict, attention_mask=rep_attn_mask)
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

                        aligned = np.zeros(n_events, dtype=np.float64)
                        if n_events + 2 <= len(token_attr):
                            aligned[:] = token_attr[1 : 1 + n_events]
                        elif len(token_attr) == max_pos and n_events + 2 > max_pos:
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
        print(f"[AIIDKIT] Integrated Gradients failed for '{d_key}/{task_key}': {exc}")

    _lru_put(_IG_TASK_CACHE, ig_key, event_scores_by_col, maxsize=384)
    return event_scores_by_col


def _assemble_task_prediction_payload(
    *,
    dom: dict,
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
    cols = dom["task_cols_map"][t_key]
    t_horizons = dom["task_horizons_map"][t_key]
    logits_np = full_logits_np[cols]

    t_cohort = cohort.get("tasks", {}).get(t_key, cohort)
    cohort_median = np.array(
        t_cohort.get("logits_median", dom["logits_median_by_task"].get(t_key, np.zeros(len(cols)))),
        dtype=float,
    )
    cohort_std = np.clip(
        np.array(t_cohort.get("logits_std", dom["logits_std_by_task"].get(t_key, np.ones(len(cols)))), dtype=float),
        1e-6,
        None,
    )
    event_scores = expit((logits_np - cohort_median) / cohort_std)
    calibrated_probs = _apply_platt_vector(logits_np, cols, dom["platt_params_by_col"])

    risk_scores = {f"{h}d": round(float(np.clip(event_scores[i], 0.01, 0.99)), 4) for i, h in enumerate(t_horizons)}
    calibrated_risks = {f"{h}d": round(float(np.clip(calibrated_probs[i], 1e-4, 0.99)), 4) for i, h in enumerate(t_horizons)}

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

        if cohort_pts and 0 <= nearest_idx < len(cohort_pts):
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
            "domain":             dom["domain"],
            "domain_label":       dom["label"],
            "task":               t_key,
            "task_label":         t_label,
            "available_horizons": list(t_horizons),
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


def _real_predict_domain(
    d_key: str,
    events: list[dict],
    horizon: int,
    cohort: dict,
    selected_task: str,
    fup: int,
    trajectory_fups: list[int],
) -> dict:
    dom = _DOMAINS[d_key]
    full_events = sorted(list(events), key=lambda e: float(e.get("days_since_tpx", 0.0)))
    stage_events = _truncate_events_for_fup(full_events, fup)

    if fup not in trajectory_fups:
        trajectory_fups = sorted(set(trajectory_fups) | {fup})

    trajectory, logits_by_fup = _compute_domain_trajectory_real(d_key, full_events, trajectory_fups)
    traj_by_fup = {pt["fup"]: pt for pt in trajectory}
    full_logits_np = logits_by_fup[fup]
    umap_x = traj_by_fup[fup]["umap_x"]
    umap_y = traj_by_fup[fup]["umap_y"]

    valid_domain_tasks = [t["key"] for t in dom["tasks_meta"]]
    active_task = selected_task if selected_task in valid_domain_tasks else valid_domain_tasks[0]

    active_ig = _compute_task_ig_attributions(d_key, stage_events, fup, active_task)

    tasks_predictions = {}
    ev_sig = _events_signature(stage_events)
    for t_info in dom["tasks_meta"]:
        t_key = t_info["key"]
        attr_map = active_ig if t_key == active_task else _lru_get(_IG_TASK_CACHE, (d_key, ev_sig, int(fup), str(t_key)))
        tasks_predictions[t_key] = _assemble_task_prediction_payload(
            dom=dom,
            t_key=t_key,
            t_label=t_info["label"],
            full_logits_np=full_logits_np,
            umap_x=umap_x,
            umap_y=umap_y,
            events=stage_events,
            cohort=cohort,
            active_fup=fup,
            requested_horizon=horizon,
            attr_by_col=attr_map,
        )

    return {
        **tasks_predictions[active_task],
        "domain": d_key,
        "domain_label": dom["label"],
        "fup": fup,
        "trajectory": trajectory,
        "tasks": tasks_predictions,
    }


# ---------------------------------------------------------------------------
# App Initialisation & Vocabulary Enrichment
# ---------------------------------------------------------------------------

app = Flask(__name__)
if getattr(CONFIG, "TRUST_PROXY", True):
    app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)
CORS(app)

_PATHS = CONFIG.get_resolved_paths()


@app.before_request
def _enforce_basic_auth():
    expected_pass = getattr(CONFIG, "BASIC_AUTH_PASS", "")
    if not expected_pass:
        return None
    if request.path.rstrip("/") == "/api/health":
        return None

    expected_user = getattr(CONFIG, "BASIC_AUTH_USER", "aiidkit")
    realm = getattr(CONFIG, "BASIC_AUTH_REALM", "AIIDKIT Clinical Decision Support").replace('"', "")
    auth = request.authorization
    if (
        auth is None
        or auth.username is None
        or auth.password is None
        or not (
            secrets.compare_digest(auth.username.encode("utf-8"), expected_user.encode("utf-8"))
            and secrets.compare_digest(auth.password.encode("utf-8"), expected_pass.encode("utf-8"))
        )
    ):
        return Response(
            "Authentication required.\n",
            401,
            {"WWW-Authenticate": f'Basic realm="{realm}", charset="UTF-8"'},
        )
    return None


@app.after_request
def _add_security_headers(resp):
    resp.headers.setdefault("X-Content-Type-Options", "nosniff")
    resp.headers.setdefault("X-Frame-Options", "SAMEORIGIN")
    resp.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
    if request.path.startswith("/api/"):
        resp.headers["Cache-Control"] = "no-store"
    return resp


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
            _initialize_all_domains(_PATHS)
            if _USING_REAL_MODEL:
                print("[AIIDKIT] LIVE SPECIALIST MULTI-MODEL mode active.")
        except Exception as exc:
            _EMBEDDINGS_ONLY_MODE = False
            _USING_REAL_MODEL = False
            print(f"[AIIDKIT] WARNING: Could not initialize specialist models: {exc}")
            import traceback
            traceback.print_exc()
            print("[AIIDKIT] Falling back to DEMO MODE.")
    else:
        print("[AIIDKIT] DEMO MODE — set AIIDKIT_BUNDLE_DIR or DATA_DIR to use real specialist models.")

    if not _USING_REAL_MODEL:
        _TASK_TO_DOMAIN = dict(getattr(CONFIG, "TASK_TO_DOMAIN", {}))
        _TASK_HORIZONS = dict(getattr(CONFIG, "TASK_HORIZONS", {}))
        _MOCK_COHORT_CACHE[CONFIG.DEFAULT_FUP] = generate_cohort_data(fup=CONFIG.DEFAULT_FUP, seed=CONFIG.MOCK_RANDOM_SEED)
        print("[AIIDKIT] DEMO MODE active.")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

def _resolve_domain_key(domain_arg: str | None = None, task_arg: str | None = None) -> str:
    if not _USING_REAL_MODEL:
        return domain_arg or _TASK_TO_DOMAIN.get(task_arg or CONFIG.TASK, "bacteria")
    if domain_arg and domain_arg in _DOMAINS:
        return domain_arg
    if task_arg and task_arg in _TASK_TO_DOMAIN:
        return _TASK_TO_DOMAIN[task_arg]
    return _TASK_TO_DOMAIN.get(CONFIG.TASK, next(iter(_DOMAINS.keys())))


def _ensure_cohort_for_domain_fup(d_key: str, fup: int) -> dict:
    if _USING_REAL_MODEL:
        dom = _DOMAINS[d_key]
        cc = dom["cohort_cache"]
        if fup in cc:
            return cc[fup]
        nearest_fup = min(cc.keys(), key=lambda k: abs(int(k) - int(fup)))
        return cc[nearest_fup]
    with _CACHE_LOCK:
        if fup not in _MOCK_COHORT_CACHE:
            _MOCK_COHORT_CACHE[fup] = generate_cohort_data(fup=fup, seed=CONFIG.MOCK_RANDOM_SEED)
        return _MOCK_COHORT_CACHE[fup]


@app.route("/api/health")
def api_health():
    mode = "embeddings_only" if (_USING_REAL_MODEL and _EMBEDDINGS_ONLY_MODE) else ("live_model" if _USING_REAL_MODEL else "demo")
    first_dom = next(iter(_DOMAINS.values()), None)
    fups_loaded = len(first_dom["cohort_cache"]) if first_dom else len(_MOCK_COHORT_CACHE)
    return jsonify({
        "status": "ok",
        "mode": mode,
        "multi_domain": len(_DOMAINS) > 1,
        "domains_loaded": list(_DOMAINS.keys()),
        "models_in_ram": len(_LOADED_MODELS_LRU),
        "auth_enabled": bool(getattr(CONFIG, "BASIC_AUTH_PASS", "")),
        "fups_loaded": fups_loaded,
    })


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/config")
def api_config():
    domains_payload = (
        [
            {
                "domain": d["domain"],
                "label": d["label"],
                "tasks": [t["key"] for t in d["tasks_meta"]],
                "horizons": list(d["horizons"]),
                "default_horizon": d["default_horizon"],
            }
            for d in _DOMAINS.values()
        ]
        if _USING_REAL_MODEL
        else [
            {
                "domain": d["domain"],
                "label": d["label"],
                "tasks": list(d["tasks"]),
                "horizons": list(d["horizons"]),
                "default_horizon": 90 if 90 in d["horizons"] else d["horizons"][0],
            }
            for d in getattr(CONFIG, "SPECIALIST_DOMAIN_DEFS", [])
        ]
    )
    return jsonify({
        "task":                 CONFIG.TASK,
        "available_tasks":      getattr(CONFIG, "AVAILABLE_TASKS", [{"key": CONFIG.TASK, "label": CONFIG.TASK}]),
        "default_horizon":      CONFIG.DEFAULT_HORIZON,
        "available_horizons":   CONFIG.AVAILABLE_HORIZONS,
        "task_horizons":        _TASK_HORIZONS,
        "task_to_domain":       _TASK_TO_DOMAIN,
        "domains":              domains_payload,
        "multi_domain_mode":    len(domains_payload) > 1,
        "horizon_labels":       {str(k): v for k, v in getattr(CONFIG, "HORIZON_LABELS", {}).items()},
        "horizon_short_labels": {str(k): v for k, v in getattr(CONFIG, "HORIZON_SHORT_LABELS", {}).items()},
        "default_fup":          CONFIG.DEFAULT_FUP,
        "available_fups":       CONFIG.AVAILABLE_FUPS,
        "precomputed_fups":     getattr(CONFIG, "PRECOMPUTED_FUPS", []),
        "using_mock_data":      not _USING_REAL_MODEL,
        "embeddings_only_mode": _EMBEDDINGS_ONLY_MODE,
        "vocabulary":           CONFIG.VOCABULARY,
        "risk_threshold":       CONFIG.RISK_THRESHOLD,
    })


@app.route("/api/cohort")
def api_cohort():
    fup_str = request.args.get("fup")
    domain_arg = request.args.get("domain")
    task_arg = request.args.get("task")
    try:
        fup = int(fup_str) if fup_str is not None else CONFIG.DEFAULT_FUP
    except ValueError:
        return jsonify({"error": "Invalid FUP value. Must be an integer."}), 400

    if fup not in CONFIG.AVAILABLE_FUPS:
        return jsonify({"error": f"FUP must be one of {CONFIG.AVAILABLE_FUPS}"}), 400

    d_key = _resolve_domain_key(domain_arg=domain_arg, task_arg=task_arg)
    try:
        return jsonify(_ensure_cohort_for_domain_fup(d_key, fup))
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
    d_key = _resolve_domain_key(task_arg=task)
    sorted_full = sorted(list(full_events), key=lambda e: float(e.get("days_since_tpx", 0.0)))
    sig = _events_signature(sorted_full)
    cache_key = (d_key if _USING_REAL_MODEL else "mock", sig, int(fup), tuple(trajectory_fups))
    cached = _lru_get(_PREDICTION_CACHE, cache_key)

    if cached is not None:
        t_pred = cached["tasks"].get(task)
        if _USING_REAL_MODEL and t_pred is not None and not t_pred.get("ig_computed", False):
            dom = _DOMAINS[d_key]
            cohort = _ensure_cohort_for_domain_fup(d_key, fup)
            stage_events = _truncate_events_for_fup(sorted_full, fup)
            _, logits_by_fup = _compute_domain_trajectory_real(d_key, sorted_full, trajectory_fups)
            ig_map = _compute_task_ig_attributions(d_key, stage_events, fup, task)
            t_info = next((t for t in dom["tasks_meta"] if t["key"] == task), {"key": task, "label": task})
            updated_t_pred = _assemble_task_prediction_payload(
                dom=dom,
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

        t_pred = t_pred or next(iter(cached["tasks"].values()))
        t_hrzs = t_pred.get("available_horizons") or _TASK_HORIZONS.get(task, [horizon])
        eff_h = str(horizon if horizon in t_hrzs else t_hrzs[0])
        h_pred = t_pred.get("by_horizon", {}).get(eff_h, t_pred)
        return {
            **h_pred,
            "domain": cached.get("domain", d_key),
            "domain_label": cached.get("domain_label", ""),
            "fup": fup,
            "trajectory": cached["trajectory"],
            "tasks": cached["tasks"],
        }

    cohort = _ensure_cohort_for_domain_fup(d_key, fup)
    if _USING_REAL_MODEL:
        res = _real_predict_domain(d_key, sorted_full, horizon, cohort, selected_task=task, fup=fup, trajectory_fups=trajectory_fups)
    else:
        res = mock_predict(sorted_full, horizon, cohort, selected_task=task, fup=fup, trajectory_fups=trajectory_fups)
        res["domain"] = d_key
        res["domain_label"] = CONFIG.DOMAIN_LABELS.get(d_key, "Demo Model")

    _lru_put(_PREDICTION_CACHE, cache_key, res, maxsize=384)
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
    if fup not in CONFIG.AVAILABLE_FUPS:
        return jsonify({"error": f"FUP must be one of {CONFIG.AVAILABLE_FUPS}"}), 400

    # Snap horizon gracefully to task's valid specialist horizon set (e.g. 90d -> 360d when switching to Graft Loss)
    valid_task_horizons = _TASK_HORIZONS.get(task, CONFIG.AVAILABLE_HORIZONS)
    if horizon not in valid_task_horizons:
        horizon = 90 if 90 in valid_task_horizons else (360 if 360 in valid_task_horizons else valid_task_horizons[0])

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
