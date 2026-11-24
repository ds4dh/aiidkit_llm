"""
AIIDKIT Web Tool — Embeddings-Only Bundle I/O & Standalone Inference Helpers
============================================================================
Provides zero-pickle JSON/NPZ loading, pure-NumPy t-EAV preprocessing (mirroring
`src.data.patient_dataset.preprocess_batch` without requiring `datasets`, `pyarrow`,
`pandas`, or `matplotlib`), direct common-trunk embedding -> head logit projection,
and model-space subgroup concept prototyping (zero patient records required).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import numpy as np
from scipy.special import expit

BIN_LABELS: list[str] = ["Lowest", "Lower", "Low", "Middle", "High", "Higher", "Highest"]

# Sanitized keys allowed inside exported `run_metadata.json`
ALLOWED_RUN_METADATA_KEYS: tuple[str, ...] = (
    "tasks",
    "horizons",
    "task_horizons",
    "label_keys",
    "use_numerical_embedding",
    "enforce_monotonicity",
    "architecture_type",
)


def sanitize_run_metadata(run_meta: dict[str, Any] | None) -> dict[str, Any]:
    """Strip host paths, usernames, and training-only arguments from `run_metadata.json`."""
    if not isinstance(run_meta, dict):
        return {}
    clean: dict[str, Any] = {}
    for k in ALLOWED_RUN_METADATA_KEYS:
        if k in run_meta and run_meta[k] is not None:
            clean[k] = run_meta[k]
    return clean


def serialize_bin_intervals(bin_intervals: dict[str, Any]) -> dict[str, dict[str, list[float]]]:
    """Convert `dict[str, pd.IntervalIndex]` (or plain interval dict) into JSON-serializable bounds."""
    serialized: dict[str, dict[str, list[float]]] = {}
    for attr, iv in (bin_intervals or {}).items():
        if isinstance(iv, dict) and "left" in iv and "right" in iv:
            lefts = [float(x) for x in iv["left"]]
            rights = [float(x) for x in iv["right"]]
        else:
            lefts = [float(x) for x in np.asarray(iv.left, dtype=np.float64)]
            rights = [float(x) for x in np.asarray(iv.right, dtype=np.float64)]
        serialized[str(attr)] = {"left": lefts, "right": rights}
    return serialized


def load_vocab_json(path: Path) -> dict[str, int]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {str(k): int(v) for k, v in raw.items()}


def load_bin_intervals_json(path: Path) -> dict[str, dict[str, list[float]]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {
        str(attr): {
            "left": [float(x) for x in bounds["left"]],
            "right": [float(x) for x in bounds["right"]],
        }
        for attr, bounds in raw.items()
    }


def load_platt_params_json(path: Path) -> dict[int, tuple[float, float]]:
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {int(k): (float(v[0]), float(v[1])) for k, v in raw.items()}


def load_cohort_embeddings_npz(path: Path) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], dict[int, np.ndarray]]:
    """Load anonymized `(N, D)` trunk embeddings, `(N, L)` outcome labels, and `(N,)` anon slot IDs (`allow_pickle=False`)."""
    by_fup: dict[int, np.ndarray] = {}
    labels_by_fup: dict[int, np.ndarray] = {}
    anon_ids_by_fup: dict[int, np.ndarray] = {}
    with np.load(path, allow_pickle=False) as archive:
        for key in sorted(archive.files):
            if key.startswith("fup_"):
                fup = int(key[len("fup_") :])
                arr = np.asarray(archive[key], dtype=np.float32)
                if arr.ndim != 2:
                    raise ValueError(f"Expected 2D embedding array for {key}, got shape {arr.shape}")
                by_fup[fup] = arr
            elif key.startswith("labels_"):
                fup = int(key[len("labels_") :])
                l_arr = np.asarray(archive[key], dtype=np.int16)
                if l_arr.ndim == 2:
                    labels_by_fup[fup] = l_arr
            elif key.startswith("anon_ids_"):
                fup = int(key[len("anon_ids_") :])
                a_arr = np.asarray(archive[key], dtype=np.int32).reshape(-1)
                anon_ids_by_fup[fup] = a_arr
    return by_fup, labels_by_fup, anon_ids_by_fup


def normalize_numerical_value(
    f_val: float,
    lefts: np.ndarray,
    rights: np.ndarray,
    n_bins: int,
    median_width: float | None = None,
    right_skewed: bool = False,
    left_skewed: bool = False,
) -> tuple[int, float]:
    """Exact standalone equivalent of `src.data.patient_dataset.normalize_numerical_value`."""
    if not np.isfinite(f_val):
        return 0, np.nan
    if f_val <= lefts[0]:
        return 0, -1.0
    if f_val >= rights[-1]:
        return n_bins - 1, 1.0

    idx = int(np.clip(np.searchsorted(rights, f_val, side="left"), 0, n_bins - 1))
    width = float(rights[idx] - lefts[idx])
    if width > 1e-12:
        t = float(np.clip((f_val - float(lefts[idx])) / width, 0.0, 1.0))
        if median_width is not None and median_width > 1e-12:
            r = width / median_width
            if right_skewed and idx >= (n_bins // 2) and r > 1.5:
                frac = float(np.log1p((r - 1.0) * t) / np.log(r))
            elif left_skewed and idx <= (n_bins // 2) and r > 1.5:
                frac = float(1.0 - np.log1p((r - 1.0) * (1.0 - t)) / np.log(r))
            else:
                frac = t
        else:
            frac = t
    else:
        frac = 0.5
    cdf = (idx + frac) / float(n_bins)
    return idx, float(2.0 * cdf - 1.0)


def compute_bin_bounds(bin_intervals: dict[str, Any] | None) -> dict[str, tuple]:
    """Precompute quantile interval edges & skewness flags from JSON dicts or `pd.IntervalIndex`."""
    bounds: dict[str, tuple] = {}
    for attr, iv in (bin_intervals or {}).items():
        if isinstance(iv, dict) and "left" in iv and "right" in iv:
            lefts = np.asarray(iv["left"], dtype=np.float64)
            rights = np.asarray(iv["right"], dtype=np.float64)
        else:
            lefts = np.asarray(iv.left, dtype=np.float64)
            rights = np.asarray(iv.right, dtype=np.float64)
        n_iv = int(len(lefts))
        widths = np.maximum(rights - lefts, 0.0)
        pos_widths = widths[widths > 1e-12]
        med_w = float(np.median(pos_widths)) if len(pos_widths) > 0 else 1.0
        right_skewed = bool(len(widths) >= 3 and med_w > 1e-12 and (widths[-1] / med_w) > 2.0)
        left_skewed = bool(len(widths) >= 3 and med_w > 1e-12 and (widths[0] / med_w) > 2.0 and not right_skewed)
        bounds[attr] = (lefts, rights, n_iv, med_w, right_skewed, left_skewed)
    return bounds


def preprocess_events_batch(
    batch: dict[str, list],
    vocab: dict[str, int],
    bin_intervals: dict[str, Any] | None,
    time_mapping: dict[str, str],
    eav_mappings: dict[str, str],
    bin_labels: list[str] = BIN_LABELS,
    bin_bounds: dict[str, tuple] | None = None,
) -> dict[str, list]:
    """Standalone NumPy equivalent of `src.data.patient_dataset.preprocess_batch`."""
    unk_id = vocab["[UNK]"]
    labels_map = {i: label for i, label in enumerate(bin_labels)}
    if bin_bounds is None:
        bin_bounds = compute_bin_bounds(bin_intervals)

    time_out_key = list(time_mapping.keys())[0]
    time_in_key = list(time_mapping.values())[0]
    out_time = [
        [0.0 if (t is None or t != t) else float(t) for t in seq]
        for seq in batch[time_in_key]
    ]

    ent_in = eav_mappings["entity_id"]
    attr_in = eav_mappings["attribute_id"]
    val_in = "value" if "value" in batch else eav_mappings.get("value_id", "value_binned")
    out_ent_ids, out_attr_ids, out_val_ids, out_val_nums = [], [], [], []

    for entities, attributes, values in zip(batch[ent_in], batch[attr_in], batch[val_in]):
        out_ent_ids.append([vocab.get(e, unk_id) for e in entities])
        out_attr_ids.append([vocab.get(a, unk_id) for a in attributes])

        row_val_ids = []
        row_val_nums = []
        for v, attr in zip(values, attributes):
            token_str = str(v)
            norm_num = np.nan

            if attr in bin_bounds:
                lefts, rights, n_bins, med_w, r_skew, l_skew = bin_bounds[attr]
                try:
                    f_val = float(v)
                    idx, norm_num = normalize_numerical_value(
                        f_val,
                        lefts,
                        rights,
                        n_bins,
                        median_width=med_w,
                        right_skewed=r_skew,
                        left_skewed=l_skew,
                    )
                    if np.isfinite(norm_num):
                        token_str = labels_map[min(idx, len(bin_labels) - 1)]
                except (ValueError, TypeError, KeyError):
                    if token_str in bin_labels:
                        b_idx = bin_labels.index(token_str)
                        norm_num = float(2.0 * ((b_idx + 0.5) / float(len(bin_labels))) - 1.0)
            else:
                if token_str in bin_labels:
                    b_idx = bin_labels.index(token_str)
                    norm_num = float(2.0 * ((b_idx + 0.5) / float(len(bin_labels))) - 1.0)
                else:
                    try:
                        token_str = str(int(float(v)))
                    except ValueError:
                        pass

            row_val_ids.append(vocab.get(token_str, unk_id))
            row_val_nums.append(norm_num)

        out_val_ids.append(row_val_ids)
        out_val_nums.append(row_val_nums)

    return {
        time_out_key: out_time,
        "entity_id": out_ent_ids,
        "attribute_id": out_attr_ids,
        "value_id": out_val_ids,
        "value_num": out_val_nums,
    }


def apply_monotonic_groups_torch(raw_logits, enforce_monotonicity, problem_type: str | None):
    """Apply multi-horizon cumulative monotonic `softplus` parameterization to raw head logits."""
    import torch

    if not enforce_monotonicity or problem_type != "multi_label_classification" or raw_logits.shape[-1] <= 1:
        return raw_logits

    if isinstance(enforce_monotonicity, (list, tuple)):
        group_sizes = [int(g) for g in enforce_monotonicity]
        if sum(group_sizes) != raw_logits.shape[-1]:
            raise ValueError(
                f"enforce_monotonicity groups {group_sizes} do not cover {raw_logits.shape[-1]} output columns"
            )
    else:
        group_sizes = [raw_logits.shape[-1]]

    chunks, start = [], 0
    for size in group_sizes:
        group = raw_logits[:, start : start + size]
        start += size
        if size <= 1:
            chunks.append(group)
            continue
        monotonic_logits = [group[:, 0]]
        for i in range(1, size):
            monotonic_logits.append(monotonic_logits[-1] + torch.nn.functional.softplus(group[:, i]))
        chunks.append(torch.stack(monotonic_logits, dim=1))
    return torch.cat(chunks, dim=1)


def project_embeddings_to_logits(model, embeddings_np: np.ndarray) -> np.ndarray:
    """
    Project anonymized common-trunk embeddings `(N, D)` directly through the classification
    head (`head -> drop -> classifier` + monotonic softplus) to obtain exact `(N, num_labels)` logits.
    """
    import torch

    device = next(model.parameters()).device
    head_dtype = (
        next(model.head.parameters()).dtype
        if hasattr(model, "head")
        else next(model.parameters()).dtype
    )
    cls_dtype = next(model.classifier.parameters()).dtype

    z = torch.as_tensor(embeddings_np, device=device, dtype=head_dtype)
    with torch.no_grad():
        pooled_head = model.head(z) if hasattr(model, "head") else z
        if hasattr(model, "drop"):
            pooled_head = model.drop(pooled_head)
        raw_logits = model.classifier(pooled_head.to(dtype=cls_dtype))
        logits = apply_monotonic_groups_torch(
            raw_logits,
            getattr(model.config, "enforce_monotonicity", False),
            getattr(model.config, "problem_type", "multi_label_classification"),
        )
    return logits.detach().float().cpu().numpy()


def estimate_imminence_from_calibrated_probs(cal_probs_mat: np.ndarray, horizons: list[int]) -> np.ndarray:
    """
    Estimate smooth `[0, 1]` event imminence purely from multi-horizon calibrated cumulative
    probabilities `P(T <= h_k | z_i)` via discrete survival interval hazard weighting.
    """
    n = cal_probs_mat.shape[0]
    if n == 0 or len(horizons) == 0:
        return np.zeros(n, dtype=np.float64)

    p_cum = np.clip(np.maximum.accumulate(cal_probs_mat, axis=1), 0.0, 1.0)
    prev_p = np.zeros((n, 1), dtype=np.float64)
    inc_p = np.maximum(0.0, np.diff(np.hstack([prev_p, p_cum]), axis=1))

    h_arr = np.asarray(horizons, dtype=np.float64)
    prev_h = np.concatenate([[0.0], h_arr[:-1]])
    mid_days = 0.5 * (prev_h + h_arr)
    time_weights = np.exp(-mid_days / 365.0)

    # Combine conditional timing urgency with overall cumulative horizon risk
    raw_imminence = (inc_p * time_weights[None, :]).sum(axis=1)
    max_p = np.maximum(p_cum[:, -1], 1e-6)
    conditional_urgency = raw_imminence / max_p
    blended = np.sqrt(np.clip(raw_imminence, 0.0, 1.0) * np.clip(conditional_urgency, 0.0, 1.0))
    return np.clip(blended, 0.0, 1.0)


def build_vocabulary_concept_bank(
    *,
    model,
    collator,
    vocab: dict[str, int],
    bin_intervals: dict[str, Any] | None,
    vocabulary_cfg: dict[str, dict[str, list[str]]],
    static_entities: set[str],
    time_mapping: dict[str, str],
    eav_mappings: dict[str, str],
    pool_fn,
) -> dict[str, Any]:
    """
    Build a patient-free bank of clinical concept vectors `(entity, attribute, value)` in the
    model's common-trunk embedding space by probing single concepts from `CONFIG.VOCABULARY`
    relative to a neutral baseline sequence. Zero patient data is accessed.
    """
    import torch

    device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype
    bin_bounds = compute_bin_bounds(bin_intervals)

    # Select representative values per (entity, attribute) to keep concept bank compact & informative
    keep_ordinals = {"Lowest", "Lower", "Middle", "Higher", "Highest"}
    skip_attrs = {
        "Transplant event",
        "Transplant procedure",
        "Listing Organ Type",
        "Hospital discharge",
        "Waitlist removal",
        "Total Organ Count",
        "Transplant Order",
        "Adequacy",
        "Presentation",
    }

    concepts: list[dict[str, Any]] = []
    samples: list[dict[str, Any]] = []

    # Sample 0 & 1: neutral reference sequences at t=0 (static) and t=30 (recent)
    for ref_t in (0.0, 30.0):
        ref_sample = {
            "entity_id": np.array([0], dtype=np.int64),
            "attribute_id": np.array([0], dtype=np.int64),
            "value_id": np.array([0], dtype=np.int64),
            "value_num": np.array([np.nan], dtype=np.float32),
            "days_since_tpx": np.array([ref_t], dtype=np.float32),
            "split": "test",
        }
        for lk in collator.label_keys:
            ref_sample[lk] = 0.0
        samples.append(ref_sample)

    for ent, attrs in sorted(vocabulary_cfg.items()):
        is_static = ent in static_entities
        probe_time = 0.0 if is_static else 30.0
        for attr, vals in sorted(attrs.items()):
            if attr in skip_attrs or attr.startswith("Stop ("):
                continue
            cand_vals = [v for v in vals if (v in keep_ordinals or v not in BIN_LABELS) and v not in ("0", "False", "Missing")]
            for val in cand_vals[:6]:
                proc = preprocess_events_batch(
                    batch={
                        "entity": [[ent]],
                        "attribute": [[attr]],
                        "value": [[val]],
                        "time": [[probe_time]],
                    },
                    vocab=vocab,
                    bin_intervals=bin_intervals,
                    time_mapping=time_mapping,
                    eav_mappings=eav_mappings,
                    bin_bounds=bin_bounds,
                )
                sample = {
                    "entity_id": np.array(proc["entity_id"][0], dtype=np.int64),
                    "attribute_id": np.array(proc["attribute_id"][0], dtype=np.int64),
                    "value_id": np.array(proc["value_id"][0], dtype=np.int64),
                    "value_num": np.array(proc["value_num"][0], dtype=np.float32),
                    "days_since_tpx": np.array(proc["days_since_tpx"][0], dtype=np.float32),
                    "split": "test",
                }
                for lk in collator.label_keys:
                    sample[lk] = 0.0
                concepts.append({
                    "entity": ent,
                    "attribute": attr,
                    "value": str(val),
                    "feature": f"{ent} - {attr}",
                    "is_static": is_static,
                })
                samples.append(sample)

    all_embs = []
    all_logits = []
    with torch.no_grad():
        for idx in range(0, len(samples), 64):
            b_coll = collator.torch_call(samples[idx : idx + 64])
            inp = {
                k: (v.to(device, dtype=model_dtype) if v.is_floating_point() else v.to(device))
                for k, v in b_coll["input_dict"].items()
            }
            am = b_coll["attention_mask"].to(device)
            out = model(input_dict=inp, attention_mask=am, output_hidden_states=True)
            all_embs.append(pool_fn(out, am).detach().float().cpu().numpy())
            all_logits.append(out.logits.detach().float().cpu().numpy())

    embs_mat = np.vstack(all_embs)
    logits_mat = np.vstack(all_logits)
    ref_emb_static, ref_emb_recent = embs_mat[0], embs_mat[1]
    ref_log_static, ref_log_recent = logits_mat[0], logits_mat[1]

    concept_embs = embs_mat[2:]
    concept_logits = logits_mat[2:]
    is_static_mask = np.array([c["is_static"] for c in concepts], dtype=bool)

    delta_embs = np.where(is_static_mask[:, None], concept_embs - ref_emb_static[None, :], concept_embs - ref_emb_recent[None, :])
    delta_logits = np.where(is_static_mask[:, None], concept_logits - ref_log_static[None, :], concept_logits - ref_log_recent[None, :])
    norms = np.linalg.norm(delta_embs, axis=1, keepdims=True)
    unit_delta_embs = delta_embs / np.clip(norms, 1e-8, None)

    return {
        "concepts": concepts,
        "is_static": is_static_mask,
        "unit_delta_embs": unit_delta_embs,
        "delta_logits": delta_logits,
    }


def rank_embedding_subgroup_concepts(
    *,
    concept_bank: dict[str, Any] | None,
    embeddings: np.ndarray,
    ui_cluster_labels: list[int],
    horizon_col_idx: int,
    top_k: int = 8,
) -> dict[int, dict[str, list[dict[str, Any]]]]:
    """
    Rank clinical `Baseline` and `Recent` prototype concepts for High-risk (0) vs Low-risk (1)
    embedding clusters by combining embedding-space cluster-contrast cosine alignment with
    task-head directional logit shift. Zero patient clinical records are used.
    """
    empty = {0: {"static": [], "recent": []}, 1: {"static": [], "recent": []}}
    if not concept_bank or len(embeddings) == 0:
        return empty

    labels_arr = np.asarray(ui_cluster_labels, dtype=np.int64)
    mask_high = labels_arr == 0
    mask_low = labels_arr == 1
    if not mask_high.any() or not mask_low.any():
        return empty

    centroid_diff = embeddings[mask_high].mean(axis=0) - embeddings[mask_low].mean(axis=0)
    diff_norm = float(np.linalg.norm(centroid_diff))
    if diff_norm < 1e-8:
        return empty
    unit_contrast = centroid_diff / diff_norm

    # Cosine alignment of each concept's embedding shift with (mu_high - mu_low)
    cos_align = concept_bank["unit_delta_embs"] @ unit_contrast
    d_logit = concept_bank["delta_logits"][:, int(horizon_col_idx)]
    logit_scale = float(np.percentile(np.abs(d_logit), 90)) or 1.0
    norm_logit = np.tanh(d_logit / max(logit_scale, 1e-4))

    # Composite prototype relevance: positive -> High-risk concept, negative -> Low-risk/protective concept
    composite = 0.55 * cos_align + 0.45 * norm_logit
    concepts = concept_bank["concepts"]
    is_static = concept_bank["is_static"]

    def _select(subset_mask: np.ndarray, high_risk: bool) -> list[dict[str, Any]]:
        idxs = np.flatnonzero(subset_mask)
        if len(idxs) == 0:
            return []
        scores = composite[idxs] if high_risk else -composite[idxs]
        order = idxs[np.argsort(scores)[::-1]]
        picked: list[dict[str, Any]] = []
        seen_attrs: set[tuple[str, str]] = set()
        for idx in order:
            sc = float(composite[idx])
            if (high_risk and sc <= 0.02) or ((not high_risk) and sc >= -0.02):
                continue
            c = concepts[int(idx)]
            attr_key = (c["entity"], c["attribute"])
            if attr_key in seen_attrs:
                continue
            seen_attrs.add(attr_key)
            picked.append({
                "feature": c["feature"],
                "value": c["value"],
                "score": round(float(np.clip(sc, -0.99, 0.99)), 4),
            })
            if len(picked) >= top_k:
                break
        return picked

    return {
        0: {
            "static": _select(is_static, high_risk=True),
            "recent": _select(~is_static, high_risk=True),
        },
        1: {
            "static": _select(is_static, high_risk=False),
            "recent": _select(~is_static, high_risk=False),
        },
    }
