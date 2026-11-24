#!/usr/bin/env python3
"""
Export the De-Identified 5-Domain Specialist Server Bundle (`B_CAT_MT`)
=======================================================================
Exports all 5 domain-specialist multi-task Transformer models + their 5
de-identified common-trunk embedding archives into `web_tool/server_bundle/`
with ZERO raw patient data, ZERO patient IDs, and ZERO Python pickle files.

Domains exported by default (`B_CAT_MT` flagship architecture):
  1. `bacteria`   — 5 bacterial pathogen tasks @ [30d, 60d, 90d]
  2. `viruses`    — 4 viral pathogen tasks @ [30d, 60d, 90d]
  3. `fungi`      — 2 fungal pathogen tasks @ [30d, 60d, 90d]
  4. `graft_loss` — allograft failure @ [360d, 720d, 1800d] (1y, 2y, 5y)
  5. `death`      — patient mortality @ [360d, 720d, 1800d] (1y, 2y, 5y)

Bundle layout:
  server_bundle/
    bundle_manifest.json
    metadata/
      vocab.json
      bin_intervals.json
    domains/
      <domain>/
        model/ (config.json, model.safetensors, run_metadata.json)
        platt_params.json
        cohort_embeddings.npz     # CSPRNG-shuffled float16 (N, 768) + labels int8

Usage:
  apptainer exec --nv research-env.sif /opt/venv/bin/python web_tool/export_server_bundle.py
  apptainer exec research-env.sif /opt/venv/bin/python web_tool/export_server_bundle.py --verify-only
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import pickle
import secrets
import shutil
import sys
import time

import numpy as np

_WEB_DIR = Path(__file__).resolve().parent
_ROOT = _WEB_DIR.parent
for _p in (str(_WEB_DIR), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import config as CONFIG
from bundle_io import (
    BIN_LABELS,
    compute_bin_bounds,
    preprocess_events_batch,
    sanitize_run_metadata,
    serialize_bin_intervals,
)
from src.model.model_utils import build_monotonic_groups, resolve_best_checkpoint

FORBIDDEN_EXTENSIONS = {".pkl", ".pickle", ".pt", ".pth", ".bin", ".csv", ".tsv", ".parquet", ".arrow", ".feather"}
FORBIDDEN_JSON_SUBSTRINGS = ("/home/", "/srv/", "/scratch/", "patientid", "borneta")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_bundle_privacy(bundle_dir: Path) -> dict:
    """Audit exported single- or multi-domain bundle for privacy and pickle-free portability."""
    if not bundle_dir.exists():
        raise FileNotFoundError(f"Bundle directory does not exist: {bundle_dir}")

    manifest_path = bundle_dir / "bundle_manifest.json"
    if not manifest_path.exists():
        raise AssertionError(f"Missing {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    for req in (bundle_dir / "metadata" / "vocab.json", bundle_dir / "metadata" / "bin_intervals.json"):
        if not req.exists():
            raise AssertionError(f"Missing required metadata file: {req}")

    total_bytes = 0
    for fpath in bundle_dir.rglob("*"):
        if not fpath.is_file():
            continue
        total_bytes += fpath.stat().st_size
        ext = fpath.suffix.lower()
        if ext in FORBIDDEN_EXTENSIONS:
            raise AssertionError(f"Forbidden file type inside server bundle: {fpath}")
        if ext == ".json":
            low = fpath.read_text(encoding="utf-8").lower()
            for bad in FORBIDDEN_JSON_SUBSTRINGS:
                if bad in low:
                    raise AssertionError(f"Sensitive/host substring '{bad}' found inside {fpath}")

    domains_summary = {}
    domains_dict = manifest.get("domains") or {}
    if domains_dict:
        for d_key in sorted(domains_dict.keys()):
            d_dir = bundle_dir / "domains" / d_key
            for req in (
                d_dir / "model" / "model.safetensors",
                d_dir / "model" / "config.json",
                d_dir / "model" / "run_metadata.json",
                d_dir / "platt_params.json",
                d_dir / "cohort_embeddings.npz",
            ):
                if not req.exists():
                    raise AssertionError(f"Missing required domain file for '{d_key}': {req}")
            with np.load(d_dir / "cohort_embeddings.npz", allow_pickle=False) as npz:
                fup_keys = [k for k in npz.files if k.startswith("fup_")]
                lbl_keys = [k for k in npz.files if k.startswith("labels_")]
                if not fup_keys:
                    raise AssertionError(f"No fup_ arrays in {d_key}/cohort_embeddings.npz")
                for k in npz.files:
                    arr = npz[k]
                    if arr.ndim != 2 or not (np.issubdtype(arr.dtype, np.floating) or np.issubdtype(arr.dtype, np.integer)):
                        raise AssertionError(f"Invalid array {d_key}:{k} ({arr.shape}, {arr.dtype})")
                domains_summary[d_key] = {
                    "n_fups": len(fup_keys),
                    "has_labels": len(lbl_keys) == len(fup_keys),
                    "sample_shape": list(npz[sorted(fup_keys)[0]].shape),
                }
    else:
        with np.load(bundle_dir / "embeddings" / "cohort_embeddings.npz", allow_pickle=False) as npz:
            fup_keys = [k for k in npz.files if k.startswith("fup_")]
            domains_summary["default"] = {
                "n_fups": len(fup_keys),
                "sample_shape": list(npz[sorted(fup_keys)[0]].shape),
            }

    return {
        "bundle_dir": str(bundle_dir),
        "total_size_mb": round(total_bytes / (1024 * 1024), 2),
        "n_domains": len(domains_summary),
        "domains": domains_summary,
        "pickle_free": True,
        "raw_data_free": True,
    }


def _load_vocab_and_bins(data_dir: Path) -> tuple[dict[str, int], dict]:
    from src.data.patient_dataset import resolve_processed_cache_dir

    candidates = [
        resolve_processed_cache_dir(data_dir) / "pretraining_metadata",
        data_dir / "processed_cache" / "pretraining_metadata",
    ]
    for meta in candidates:
        if (meta / "vocab.pkl").exists() and (meta / "bin_intervals.pkl").exists():
            with open(meta / "vocab.pkl", "rb") as f:
                vocab = pickle.load(f)
            with open(meta / "bin_intervals.pkl", "rb") as f:
                bins = pickle.load(f)
            return vocab, bins

    cache_pkls = sorted((_WEB_DIR / ".cache").glob("cohort_umap_*.pkl"))
    if cache_pkls:
        with open(cache_pkls[-1], "rb") as f:
            c = pickle.load(f)
        return dict(c["vocab"]), c["bin_intervals"]

    raise FileNotFoundError(f"Could not find vocab.pkl / bin_intervals.pkl for {data_dir}")


def _pretokenize_fup_splits_once(
    data_dir: Path,
    vocab: dict[str, int],
    bin_intervals: dict,
    val_fups: list[int],
    test_fups: list[int],
    all_needed_label_keys: list[str],
    max_val_rows: int = 400,
    max_test_rows: int = 500,
) -> tuple[dict[int, dict], dict[int, dict]]:
    """
    Load and tokenize validation & test FUP splits ONCE in memory so sweeping all 5
    specialist models on GPU takes ~40s total instead of repeating Arrow I/O 5 times.
    """
    from datasets import disable_caching, load_from_disk

    disable_caching()
    bin_bounds = compute_bin_bounds(bin_intervals)

    # Ephemeral in-RAM map assigning each unique test patient a random CSPRNG pseudonym slot (1..M)
    # so anonymous REF_XXX dots track smoothly across FUP stages while destroying real patient IDs!
    raw_test_pids_seen: list[str] = []
    raw_test_pids_set: set[str] = set()

    def _load_split_map(fup_list: list[int], split_name: str, max_rows: int, collect_anon: bool = False) -> dict[int, dict]:
        out: dict[int, dict] = {}
        for fup in fup_list:
            sp_dir = data_dir / f"fup_{fup:04d}" / split_name
            if not sp_dir.exists():
                continue
            ds = load_from_disk(str(sp_dir))
            n_rows = min(len(ds), max_rows)
            if n_rows == 0:
                continue
            sub = ds.select(range(n_rows), keep_in_memory=True)
            raw_pids = None
            if collect_anon and "patientid" in sub.column_names:
                raw_pids = [str(x) for x in sub["patientid"]]
                for rp in raw_pids:
                    if rp not in raw_test_pids_set:
                        raw_test_pids_set.add(rp)
                        raw_test_pids_seen.append(rp)

            proc = preprocess_events_batch(
                batch=sub[:],
                vocab=vocab,
                bin_intervals=bin_intervals,
                time_mapping=CONFIG.TIME_MAPPING,
                eav_mappings=CONFIG.EAV_MAPPINGS,
                bin_labels=BIN_LABELS,
                bin_bounds=bin_bounds,
            )
            samples = []
            for i in range(n_rows):
                samples.append({
                    "entity_id":      np.array(proc["entity_id"][i]),
                    "attribute_id":   np.array(proc["attribute_id"][i]),
                    "value_id":       np.array(proc["value_id"][i]),
                    "value_num":      np.array(proc["value_num"][i], dtype=np.float32),
                    "days_since_tpx": np.array(proc["days_since_tpx"][i]),
                    "split":          split_name,
                })
            labels_by_col: dict[str, np.ndarray] = {}
            for lk in all_needed_label_keys:
                if lk in sub.column_names:
                    raw = np.asarray(sub[lk], dtype=np.int64)
                    labels_by_col[lk] = np.where(raw == -100, -100, np.where(raw == 1, 1, 0)).astype(np.int8)
            out[int(fup)] = {
                "samples": samples,
                "labels_by_col": labels_by_col,
                "n_rows": n_rows,
                "_raw_pids": raw_pids,
            }
            del ds, sub, proc
        return out

    print(f"[Export] Pre-tokenizing {len(val_fups)} validation + {len(test_fups)} test FUP slices once...")
    t0 = time.time()
    val_map = _load_split_map(val_fups, "validation", max_val_rows, collect_anon=False)
    test_map = _load_split_map(test_fups, "test", max_test_rows, collect_anon=True)

    # Replace _raw_pids with CSPRNG-shuffled integer slots (1..M) and immediately destroy real IDs
    slot_pool = list(range(1, len(raw_test_pids_seen) + 1))
    secrets.SystemRandom().shuffle(slot_pool)
    ephemeral_pid_to_slot = dict(zip(raw_test_pids_seen, slot_pool))
    for t_entry in test_map.values():
        rp_list = t_entry.pop("_raw_pids", None)
        if rp_list is not None:
            t_entry["anon_slots"] = np.array([ephemeral_pid_to_slot[rp] for rp in rp_list], dtype=np.int16)
        else:
            t_entry["anon_slots"] = np.arange(1, t_entry["n_rows"] + 1, dtype=np.int16)
    del raw_test_pids_seen, raw_test_pids_set, ephemeral_pid_to_slot

    print(f"[Export] Pre-tokenized all FUP slices in {time.time() - t0:.1f}s.")
    return val_map, test_map


def _export_single_domain_on_device(
    *,
    d_key: str,
    d_label: str,
    ckpt_dir: Path,
    clean_run_meta: dict,
    vocab: dict[str, int],
    val_map: dict[int, dict],
    test_map: dict[int, dict],
    domain_out_dir: Path,
    device: str,
) -> dict:
    import torch
    from sklearn.linear_model import LogisticRegression
    from src.model.patient_embedder import (
        PatientDataCollatorForClassification,
        PatientEmbeddingModelFactory,
    )

    t0 = time.time()
    cfg_dict = json.loads((ckpt_dir / "config.json").read_text(encoding="utf-8"))
    num_labels = len(cfg_dict.get("id2label", {})) or len(clean_run_meta.get("label_keys", []))
    problem_type = cfg_dict.get("problem_type", "multi_label_classification")
    enforce_monotonicity = build_monotonic_groups(clean_run_meta, cfg_dict.get("enforce_monotonicity", True))
    label_keys = list(clean_run_meta.get("label_keys") or [])
    if len(label_keys) != num_labels:
        tasks = clean_run_meta.get("tasks", [])
        hrzs = clean_run_meta.get("horizons", [])
        label_keys = [f"label_{t}_{int(h):04d}d" for t in tasks for h in hrzs]
        clean_run_meta["label_keys"] = label_keys

    use_num_emb = bool(clean_run_meta.get("use_numerical_embedding", True))
    model_args = {"num_labels": num_labels, "problem_type": problem_type}
    if device == "cpu":
        model_args["dtype"] = "float32"

    model = PatientEmbeddingModelFactory.from_pretrained(
        task="classification",
        pretrained_dir=str(ckpt_dir),
        embedding_layer_config={
            "vocab_size": len(vocab),
            "eav_mappings": CONFIG.EAV_MAPPINGS,
            "time_mapping": CONFIG.TIME_MAPPING,
            "use_numerical_embedding": use_num_emb,
        },
        model_args=model_args,
        enforce_monotonicity=enforce_monotonicity,
    )
    model.eval()
    model.to(device)
    model_dtype = next(model.parameters()).dtype
    use_amp = model_dtype in (torch.float16, torch.bfloat16)

    collator = PatientDataCollatorForClassification(
        eav_mappings=CONFIG.EAV_MAPPINGS,
        time_mapping=CONFIG.TIME_MAPPING,
        use_numerical_embedding=use_num_emb,
        label_keys=label_keys,
        max_position_embeddings=getattr(model.config, "max_position_embeddings", 1024),
    )
    pooling_mode = getattr(model.config, "classifier_pooling", "mean")

    def _forward_batches(samples_list: list[dict], need_embs: bool) -> tuple[np.ndarray, np.ndarray | None]:
        prepared = [{**s, **{lk: 0.0 for lk in label_keys}} for s in samples_list]
        logits_chunks = []
        emb_chunks = []
        bs = 64 if device == "cuda" else 32
        for idx in range(0, len(prepared), bs):
            b_coll = collator.torch_call(prepared[idx : idx + bs])
            inp = {
                k: (v.to(device, dtype=model_dtype) if v.is_floating_point() else v.to(device))
                for k, v in b_coll["input_dict"].items()
            }
            am = b_coll["attention_mask"].to(device)
            with torch.no_grad():
                if use_amp:
                    with torch.autocast(device_type=("cuda" if "cuda" in device else "cpu"), dtype=model_dtype):
                        out = model(input_dict=inp, attention_mask=am, output_hidden_states=need_embs)
                else:
                    out = model(input_dict=inp, attention_mask=am, output_hidden_states=need_embs)
                logits_chunks.append(out.logits.float().cpu().numpy())
                if need_embs:
                    lh = out.hidden_states[-1]
                    if pooling_mode == "last":
                        sl = am.sum(dim=1).clamp(min=1) - 1
                        bi = torch.arange(lh.size(0), device=lh.device)
                        pooled = lh[bi, sl]
                    else:
                        m = am.unsqueeze(-1)
                        pooled = (lh * m).sum(dim=1) / m.sum(dim=1).clamp(min=1e-9)
                    emb_chunks.append(pooled.float().cpu().numpy())
        return np.vstack(logits_chunks), (np.vstack(emb_chunks) if need_embs else None)

    # 1. Fit L2-Platt scaling per head on validation FUPs
    val_logits_list, val_labels_list = [], []
    for fup, v_entry in sorted(val_map.items()):
        v_log, _ = _forward_batches(v_entry["samples"], need_embs=False)
        n_r = v_entry["n_rows"]
        v_lbl = np.full((n_r, num_labels), -100, dtype=np.int8)
        for j, lk in enumerate(label_keys):
            if lk in v_entry["labels_by_col"]:
                v_lbl[:, j] = v_entry["labels_by_col"][lk]
        val_logits_list.append(v_log)
        val_labels_list.append(v_lbl)

    platt_params: dict[int, tuple[float, float]] = {}
    if val_logits_list:
        all_v_log = np.vstack(val_logits_list)
        all_v_lbl = np.vstack(val_labels_list)
        for j in range(num_labels):
            y_col = all_v_lbl[:, j]
            valid = (y_col == 0) | (y_col == 1)
            if valid.sum() < 10 or np.unique(y_col[valid]).size < 2:
                platt_params[j] = (1.0, 0.0)
                continue
            y_bin = y_col[valid].astype(np.int64)
            x_col = np.clip(all_v_log[valid, j], -15.0, 15.0).reshape(-1, 1)
            slope, intercept, fitted = 1.0, 0.0, False
            if int(np.sum(y_bin == 1)) >= 20:
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
            platt_params[j] = (slope, intercept)

    # 2. Extract common-trunk embeddings + horizon label matrices across all 13 test FUPs (CSPRNG-permuted!)
    npz_arrays = {}
    patients_per_fup = {}
    for fup, t_entry in sorted(test_map.items()):
        _, embs = _forward_batches(t_entry["samples"], need_embs=True)
        n_r = t_entry["n_rows"]
        t_lbl = np.full((n_r, num_labels), -100, dtype=np.int8)
        for j, lk in enumerate(label_keys):
            if lk in t_entry["labels_by_col"]:
                t_lbl[:, j] = t_entry["labels_by_col"][lk]

        rng = np.random.default_rng(secrets.randbits(64))
        perm = rng.permutation(n_r)
        npz_arrays[f"fup_{int(fup):04d}"] = np.ascontiguousarray(embs[perm], dtype=np.float16)
        npz_arrays[f"labels_{int(fup):04d}"] = np.ascontiguousarray(t_lbl[perm], dtype=np.int8)
        npz_arrays[f"anon_ids_{int(fup):04d}"] = np.ascontiguousarray(t_entry["anon_slots"][perm].reshape(-1, 1), dtype=np.int16)
        patients_per_fup[str(fup)] = int(n_r)

    # 3. Write model weights, sanitized run_metadata.json, platt_params.json, and cohort_embeddings.npz
    (domain_out_dir / "model").mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ckpt_dir / "model.safetensors", domain_out_dir / "model" / "model.safetensors")
    shutil.copyfile(ckpt_dir / "config.json", domain_out_dir / "model" / "config.json")
    (domain_out_dir / "model" / "run_metadata.json").write_text(
        json.dumps(clean_run_meta, indent=2, sort_keys=True), encoding="utf-8"
    )
    (domain_out_dir / "platt_params.json").write_text(
        json.dumps({str(k): [round(v[0], 6), round(v[1], 6)] for k, v in sorted(platt_params.items())}, indent=2),
        encoding="utf-8",
    )
    npz_path = domain_out_dir / "cohort_embeddings.npz"
    np.savez_compressed(npz_path, **npz_arrays)

    model_sha = _sha256_file(domain_out_dir / "model" / "model.safetensors")[:20]
    emb_sha = _sha256_file(npz_path)[:20]
    d_fp = hashlib.sha256(f"{d_key}:{model_sha}:{emb_sha}".encode("utf-8")).hexdigest()[:16]

    del model, collator
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    print(
        f"  ✓ Exported domain '{d_key}' ({d_label}): {num_labels} heads, "
        f"{len(patients_per_fup)} FUPs in {time.time() - t0:.1f}s"
    )
    return {
        "domain": d_key,
        "label": d_label,
        "domain_fingerprint": d_fp,
        "tasks": clean_run_meta.get("tasks", []),
        "horizons": clean_run_meta.get("horizons", []),
        "num_labels": num_labels,
        "patients_per_fup": patients_per_fup,
    }


def export_bundle(
    output_dir: Path,
    finetuning_root: Path | None = None,
    single_model: Path | None = None,
    data_dir: Path | None = None,
) -> dict:
    import torch

    eff_data_dir = Path(data_dir) if data_dir else CONFIG.DATA_DIR
    if eff_data_dir is None or not eff_data_dir.exists():
        raise FileNotFoundError(f"DATA_DIR not found: {eff_data_dir}")

    if single_model is not None:
        ckpt = resolve_best_checkpoint(single_model)
        if ckpt is None or not (ckpt / "model.safetensors").exists():
            raise FileNotFoundError(f"No model.safetensors found in {single_model}")
        rm = CONFIG._discover_run_metadata(ckpt) or {}
        domains_to_export = {
            "default": {"domain": "default", "label": "Active Model", "model_path": ckpt, "run_metadata": rm}
        }
    else:
        domains_to_export = CONFIG.discover_specialist_checkpoints(finetuning_root or CONFIG.FINETUNING_ROOT)
        if not domains_to_export:
            raise FileNotFoundError(f"No specialist checkpoints discovered under {finetuning_root or CONFIG.FINETUNING_ROOT}")

    vocab, bin_intervals = _load_vocab_and_bins(eff_data_dir)

    all_label_keys_set: set[str] = set()
    cleaned_metas: dict[str, dict] = {}
    for d_key, d_spec in domains_to_export.items():
        cm = sanitize_run_metadata(d_spec.get("run_metadata"))
        lks = list(cm.get("label_keys") or [])
        if not lks:
            lks = [f"label_{t}_{int(h):04d}d" for t in cm.get("tasks", []) for h in cm.get("horizons", [])]
            cm["label_keys"] = lks
        cleaned_metas[d_key] = cm
        all_label_keys_set.update(lks)

    val_fups = [0, 30, 60, 90, 120, 150, 180, 240, 300, 360]
    test_fups = list(CONFIG.PRECOMPUTED_FUPS)
    val_map, test_map = _pretokenize_fup_splits_once(
        eff_data_dir, vocab, bin_intervals, val_fups, test_fups, sorted(all_label_keys_set)
    )

    if output_dir.exists():
        shutil.rmtree(output_dir)
    (output_dir / "metadata").mkdir(parents=True, exist_ok=True)
    (output_dir / "domains").mkdir(parents=True, exist_ok=True)

    (output_dir / "metadata" / "vocab.json").write_text(
        json.dumps({str(k): int(v) for k, v in sorted(vocab.items(), key=lambda x: x[1])}, indent=2),
        encoding="utf-8",
    )
    (output_dir / "metadata" / "bin_intervals.json").write_text(
        json.dumps(serialize_bin_intervals(bin_intervals), indent=2, sort_keys=True),
        encoding="utf-8",
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Export] Exporting {len(domains_to_export)} specialist domain(s) on {device}...")

    manifest_domains = {}
    fp_parts = []
    for d_key, d_spec in domains_to_export.items():
        d_info = _export_single_domain_on_device(
            d_key=d_key,
            d_label=d_spec["label"],
            ckpt_dir=Path(d_spec["model_path"]),
            clean_run_meta=cleaned_metas[d_key],
            vocab=vocab,
            val_map=val_map,
            test_map=test_map,
            domain_out_dir=output_dir / "domains" / d_key,
            device=device,
        )
        manifest_domains[d_key] = d_info
        fp_parts.append(d_info["domain_fingerprint"])

    bundle_fp = hashlib.sha256(":".join(fp_parts).encode("utf-8")).hexdigest()[:20]
    manifest = {
        "bundle_format_version": "2.0.0",
        "privacy_mode": "multi_domain_embeddings_only",
        "bundle_fingerprint": bundle_fp,
        "fups": [int(f) for f in sorted(test_map.keys())],
        "domains": manifest_domains,
    }
    (output_dir / "bundle_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    audit = verify_bundle_privacy(output_dir)
    print(f"[Export] Multi-domain bundle verified & ready at {output_dir}: {json.dumps(audit)}")
    return audit


def main():
    parser = argparse.ArgumentParser(description="Export de-identified 5-domain specialist server bundle for Docker.")
    parser.add_argument("--output", "-o", type=Path, default=_WEB_DIR / "server_bundle", help="Output bundle directory")
    parser.add_argument("--finetuning-root", type=Path, default=None, help="Root finetuning directory containing specialist subfolders")
    parser.add_argument("--model", "-m", type=Path, default=None, help="Optional single-model checkpoint override")
    parser.add_argument("--data-dir", "-d", type=Path, default=None, help="Raw t-EAV split directory")
    parser.add_argument("--verify-only", action="store_true", help="Only run privacy & structure audit on --output")
    args = parser.parse_args()

    if args.verify_only:
        res = verify_bundle_privacy(args.output)
        print(f"[Verify OK] {json.dumps(res, indent=2)}")
    else:
        export_bundle(args.output, finetuning_root=args.finetuning_root, single_model=args.model, data_dir=args.data_dir)


if __name__ == "__main__":
    main()
