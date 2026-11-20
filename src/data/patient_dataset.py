import os
import shutil
import pickle
import hashlib
import json
import tempfile
import numpy as np
import pandas as pd
import pyarrow as pa
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Any
from pathlib import Path
from filelock import FileLock
from tqdm import tqdm
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datasets import Dataset, DatasetDict, load_from_disk, concatenate_datasets, disable_caching

# Globally disable HuggingFace Datasets' implicit on-disk Arrow transform caching so
# `.select()`, `.flatten_indices()`, and `.filter()` NEVER write `cache-*.arrow` files
# inside read-only raw dataset folders (`/home/shares/ds4dh/.../fup_XXXX/`).
disable_caching()

# Worker count for the (CPU-bound) preprocessing map passes. Respect Slurm cgroup CPU affinity.
_AVAIL_CPUS = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 4)
SAFE_NUM_PROCS = int(os.environ.get(
    "AIIDKIT_MAP_NUM_PROCS",
    max(1, min(14, _AVAIL_CPUS - 1)),
))
# Worker threads for the parallel raw follow-up folder loading (arrow IO releases
# the GIL, so threads parallelize the reads/copies well without pickling datasets).
LOAD_NUM_THREADS = int(os.environ.get("AIIDKIT_LOAD_NUM_THREADS", 8))
MAP_BATCH_SIZE = 256
BIN_LABELS = ["Lowest", "Lower", "Low", "Middle", "High", "Higher", "Highest"]


def _resolve_local_tmp_dir() -> str | None:
    """Return fast node-local scratch directory (/scratch/$USER/$SLURM_JOB_ID or /tmp) for temporary .map() shards."""
    home_prefix = str(Path.home().resolve())
    candidates = [
        os.environ.get("AIIDKIT_SCRATCH_DIR", "").strip(),
        f"/scratch/{os.environ.get('USER', '')}/{os.environ.get('SLURM_JOB_ID', '')}"
        if os.environ.get("USER") and os.environ.get("SLURM_JOB_ID")
        else "",
        os.environ.get("TMPDIR", "").strip(),
        "/tmp",
    ]
    for c in candidates:
        if not c:
            continue
        p = Path(c)
        try:
            if p.exists() and os.access(p, os.W_OK) and not str(p.resolve()).startswith(home_prefix):
                return str(p)
        except OSError:
            continue
    return None


def resolve_processed_cache_dir(root_path: Path | str) -> Path:
    """
    Resolve the external `processed_cache` directory for a split's raw t-EAV `root_path`
    (e.g. `/home/shares/ds4dh/.../v3.9/teav/random_split`), keeping raw shared data 100% clean.

    Resolution priority:
      1. `AIIDKIT_CACHE_DIR` or `AIIDKIT_PROCESSED_CACHE_DIR` environment variable.
      2. Ephemeral `/tmp` or `/scratch` datasets stay co-located (`root_path / "processed_cache"`) so temp tests self-clean.
      3. Baobab shared high-capacity scratch: `/srv/beegfs/scratch/users/<u>/<USER>/aiidkit_cache/<dataset_tag>/<split>/processed_cache`.
      4. User XDG cache fallback: `$XDG_CACHE_HOME/aiidkit/<dataset_tag>/<split>/processed_cache` (or `~/.cache/aiidkit/...`).
    """
    rp = Path(root_path)
    try:
        resolved = rp.resolve()
    except OSError:
        resolved = rp

    resolved_str = str(resolved)
    # Ephemeral test directories in /tmp or node-local /scratch keep co-located cache for automatic cleanup
    if resolved_str.startswith("/tmp/") or resolved_str.startswith("/scratch/"):
        return rp / "processed_cache"

    parts = [p for p in resolved.parts if p not in ("/", "")]
    split_name = parts[-1] if parts else "default_split"
    parent_parts = parts[-3:-1] if len(parts) >= 3 else parts[:-1]
    dataset_label = "_".join(parent_parts) if parent_parts else "dataset"
    safe_label = "".join(c if c.isalnum() or c in ("-", "_", ".") else "_" for c in dataset_label)
    parent_key = str(resolved.parent) if resolved.parent != resolved else resolved_str
    path_hash = hashlib.sha1(parent_key.encode("utf-8")).hexdigest()[:8]
    ns_rel = Path(f"{safe_label}_{path_hash}") / split_name / "processed_cache"

    def _ensure_private_root(base_root: Path) -> Path:
        """Create cache root with strict 0700 owner-only permissions for patient-data privacy."""
        try:
            base_root.mkdir(parents=True, exist_ok=True, mode=0o700)
            os.chmod(base_root, 0o700)
        except OSError:
            pass
        return base_root / ns_rel

    for env_key in ("AIIDKIT_CACHE_DIR", "AIIDKIT_PROCESSED_CACHE_DIR"):
        env_val = os.environ.get(env_key, "").strip()
        if env_val:
            return _ensure_private_root(Path(env_val))

    user = os.environ.get("USER", "").strip() or Path.home().name
    if user:
        beegfs_user_scratch = Path(f"/srv/beegfs/scratch/users/{user[0]}/{user}")
        try:
            if beegfs_user_scratch.exists() and os.access(beegfs_user_scratch, os.W_OK):
                return _ensure_private_root(beegfs_user_scratch / "aiidkit_cache")
        except OSError:
            pass

    xdg_cache = os.environ.get("XDG_CACHE_HOME", "").strip()
    fallback_base = Path(xdg_cache) if xdg_cache else (Path.home() / ".cache")
    return _ensure_private_root(fallback_base / "aiidkit")


def _fup_key(fup):
    if fup is None:
        return None
    if isinstance(fup, list):
        return list(fup)
    return [fup]


def _make_processed_cache_path(
    cache_base_dir: Path,
    fup_train,
    fup_valid,
    fup_test,
    time_mapping,
    eav_mappings,
    is_pretraining: bool,
):
    """
    Build a task-agnostic processed cache path per (split, FUP grid, EAV/time mapping).

    Fine-tuning tasks and Optuna trials share ONE unified tokenized FUP cache (`ft_<hash>`)
    containing all rows and `label_*` columns; task-specific `drop_unlabeled` and
    `target_undersampling_ratio` slicing runs in <0.1s in memory via PyArrow `Table.take`.
    """
    cache_config = {
        "mode": "pretraining" if is_pretraining else "finetuning_shared_fups",
        "fup_train": _fup_key(fup_train),
        "fup_valid": _fup_key(fup_valid),
        "fup_test": _fup_key(fup_test),
        "time_mapping": time_mapping,
        "eav_mappings": eav_mappings,
        "bin_labels": BIN_LABELS,
        "value_num_schema": "quantile_cdf_skewlog_v3_shared_fups",
    }

    cache_json = json.dumps(cache_config, sort_keys=True, separators=(",", ":"))
    cache_hash = hashlib.sha1(cache_json.encode("utf-8")).hexdigest()[:16]

    prefix = "pt" if is_pretraining else "ft"
    cache_dir = cache_base_dir / "datasets" / f"{prefix}_{cache_hash}"
    meta_path = cache_dir / "config.json"

    return cache_dir, cache_json, meta_path


def _undersample_local_indices(
    label_matrix_slice: np.ndarray,
    target_ratio: float,
    seed: int = 1234,
    bal_col_slice: np.ndarray | None = None,
) -> np.ndarray:
    """Exact NumPy index equivalent of `undersample_dataset` on a single FUP slice."""
    rng = np.random.default_rng(seed)
    if bal_col_slice is not None:
        pos_indices = np.where(bal_col_slice == 1)[0]
        neg_indices = np.where(bal_col_slice == 0)[0]
        unlabeled_indices = np.where((bal_col_slice != 0) & (bal_col_slice != 1))[0]
    else:
        any_positive_mask = (np.max(label_matrix_slice, axis=1) == 1)
        fully_unlabeled = (label_matrix_slice == -100).all(axis=1)
        pos_indices = np.where(any_positive_mask & ~fully_unlabeled)[0]
        neg_indices = np.where(~any_positive_mask & ~fully_unlabeled)[0]
        unlabeled_indices = np.where(fully_unlabeled)[0]

    n_pos = len(pos_indices)
    n_neg = len(neg_indices)
    if n_pos < n_neg:
        majority_indices = neg_indices
        minority_indices = pos_indices
        keep_n = int(n_pos * target_ratio)
    else:
        majority_indices = pos_indices
        minority_indices = neg_indices
        keep_n = int(n_neg * target_ratio)

    if len(majority_indices) <= keep_n:
        return np.arange(len(label_matrix_slice), dtype=np.int64)

    selected_majority = rng.choice(majority_indices, size=keep_n, replace=False)
    final_indices = np.concatenate([selected_majority, minority_indices, unlabeled_indices])
    rng.shuffle(final_indices)
    return final_indices.astype(np.int64, copy=False)


def _scan_numeric_fups(root_path: Path) -> list[int]:
    """Discover all numeric `fup_XXXX` landmark folders under `root_path` in ascending order."""
    if not root_path.exists():
        return []
    fups = []
    for p in root_path.iterdir():
        if p.is_dir() and p.name.startswith("fup_"):
            suf = p.name[4:]
            if suf.isdigit():
                fups.append(int(suf))
    return sorted(fups)


def _apply_post_cache_filtering(
    dataset: DatasetDict,
    fup_by_split: dict[str, Any],
    label_keys: list[str] | None,
    target_undersampling_ratio: float | None,
    undersampling_balancing_label: str | None,
    drop_unlabeled: bool,
) -> DatasetDict:
    """Apply fast in-memory per-FUP selection, `drop_unlabeled`, and `undersample` onto the shared cached DatasetDict."""
    out_splits = {}
    for sp_name, ds in dataset.items():
        if len(ds) == 0:
            out_splits[sp_name] = ds
            continue

        has_all_labels = bool(label_keys and all(k in ds.column_names for k in label_keys))
        do_drop = bool(drop_unlabeled and has_all_labels)
        do_undersample = bool(
            sp_name == "train"
            and target_undersampling_ratio is not None
            and has_all_labels
            and (undersampling_balancing_label is None or undersampling_balancing_label in ds.column_names)
        )

        fup_col = np.asarray(ds.data.column("fup"))
        raw_fups = fup_by_split.get(sp_name)
        fups_list = raw_fups if isinstance(raw_fups, list) else [raw_fups]
        wanted_fups = [int(f) if f is not None else -1 for f in fups_list]

        # Fast path: no row filtering needed and caller requested the exact cached FUP set in order
        if not (do_drop or do_undersample):
            uniq_present = sorted(int(u) for u in np.unique(fup_col))
            if wanted_fups == uniq_present:
                out_splits[sp_name] = ds
                continue

        label_matrix = (
            np.stack([np.asarray(ds.data.column(k)) for k in label_keys], axis=1)
            if (do_drop or do_undersample)
            else None
        )
        bal_col = (
            np.asarray(ds.data.column(undersampling_balancing_label))
            if (do_undersample and undersampling_balancing_label is not None)
            else None
        )

        # Stable mergesort groups global row indices by `fup` in O(N log N) while preserving intra-FUP order
        order = np.argsort(fup_col, kind="mergesort")
        sorted_fups = fup_col[order]
        uniq_fups, split_pts = np.unique(sorted_fups, return_index=True)
        fup_to_idx = {int(u): grp for u, grp in zip(uniq_fups, np.split(order, split_pts[1:]))}

        kept_chunks = []
        for fup_val in wanted_fups:
            idx = fup_to_idx.get(fup_val)
            if idx is None or len(idx) == 0:
                continue
            if do_drop and label_matrix is not None:
                keep = np.where((label_matrix[idx] != -100).any(axis=1))[0]
                idx = idx[keep]
                if len(idx) == 0:
                    continue
            if do_undersample and label_matrix is not None:
                sub_idx = _undersample_local_indices(
                    label_matrix[idx],
                    target_ratio=float(target_undersampling_ratio),
                    seed=1234,
                    bal_col_slice=(bal_col[idx] if bal_col is not None else None),
                )
                idx = idx[sub_idx]
            kept_chunks.append(idx)

        final_idx = np.concatenate(kept_chunks) if kept_chunks else np.empty(0, dtype=np.int64)
        if len(final_idx) == len(ds) and np.array_equal(final_idx, np.arange(len(ds), dtype=np.int64)):
            out_splits[sp_name] = ds
        else:
            out_splits[sp_name] = Dataset(ds.data.table.take(pa.array(final_idx, type=pa.int64())))

    return DatasetDict(out_splits)


def load_hf_data_and_metadata(
    data_dir: str,
    sanity_check_output_dir: str | None = None,
    fup_train: int | list[int] | None = None,
    fup_valid: int | list[int] | None = None,
    fup_test: int | list[int] | None = None,
    target_undersampling_ratio: float | None = None,
    label_keys: str | list[str] | None = None,
    time_mapping: dict[str, str] = {"days_since_tpx": "time"},
    eav_mappings: dict[str, str] = {
        "entity_id": "entity",
        "attribute_id": "attribute",
        "value_id": "value_binned",
    },
    undersampling_balancing_label: str | None = None,
    drop_unlabeled: bool = False,
) -> tuple[DatasetDict, dict[str, pd.IntervalIndex], dict[str, int]]:
    """
    Build data and metadata, with persistent shared processed-dataset caching.
    """
    # Safety checks
    if isinstance(label_keys, str):
        label_keys = [label_keys]
    if target_undersampling_ratio is not None and not label_keys:
        raise ValueError("Cannot perform undersampling without valid 'label_keys'.")

    # Initialization: store processed caches on shared scratch (/srv/beegfs/scratch/...), never inside raw data_dir
    root_path = Path(data_dir)
    cache_base_dir = resolve_processed_cache_dir(root_path)
    metadata_path = cache_base_dir / "pretraining_metadata"
    legacy_metadata_path = root_path / "processed_cache" / "pretraining_metadata"
    is_pretraining = (fup_train is None)
    vocab = None
    bin_intervals = None

    # For fine-tuning / downstream analysis, always key the shared `ft_*` cache over all
    # numeric FUP folders on disk so training, Optuna, interpretability (`fup=[360]`), and
    # stratification (`fup=[90]`) all share the exact same prewarmed cache and slice FUPs in memory.
    all_disk_fups = None if is_pretraining else (_scan_numeric_fups(root_path) or fup_train)
    cache_fup_tr = None if is_pretraining else all_disk_fups
    cache_fup_va = None if is_pretraining else all_disk_fups
    cache_fup_te = None if is_pretraining else all_disk_fups

    # Build shared processed-cache paths (independent of label_keys, FUP subset, and undersampling ratio)
    processed_cache_path, cache_json, cache_meta_path = _make_processed_cache_path(
        cache_base_dir=cache_base_dir,
        fup_train=cache_fup_tr,
        fup_valid=cache_fup_va,
        fup_test=cache_fup_te,
        time_mapping=time_mapping,
        eav_mappings=eav_mappings,
        is_pretraining=is_pretraining,
    )

    # Synchronize concurrent jobs targeting the same processed_cache_path so only
    # one job loads raw data and runs .map(), while any concurrent job waits and
    # then immediately loads the completed cache without colliding on Arrow temp files.
    processed_cache_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = processed_cache_path.parent / f"{processed_cache_path.name}.lock"

    dataset = None
    with FileLock(str(lock_path)):
        has_valid_meta = (
            ((metadata_path / "vocab.pkl").exists() and (metadata_path / "bin_intervals.pkl").exists())
            or ((legacy_metadata_path / "vocab.pkl").exists() and (legacy_metadata_path / "bin_intervals.pkl").exists())
        )
        # Require atomic completion marker `config.json` (`cache_meta_path`); clean up interrupted directories
        cache_hit = processed_cache_path.exists() and cache_meta_path.exists()
        if processed_cache_path.exists() and not cache_meta_path.exists():
            print(f"[WARN] Removing incomplete cache directory (missing config.json): {processed_cache_path}")
            shutil.rmtree(processed_cache_path, ignore_errors=True)
        need_raw_data = (not cache_hit) or (is_pretraining and not has_valid_meta)

        if need_raw_data:
            print(f"Loading raw data from disk ({root_path}) -> cache root: {cache_base_dir}")
            dataset = DatasetDict({
                "train": _load_and_tag(root_path, cache_fup_tr, "train"),
                "validation": _load_and_tag(root_path, cache_fup_va, "validation"),
                "test": _load_and_tag(root_path, cache_fup_te, "test"),
            })
        else:
            print(f"Processed dataset cache hit: {processed_cache_path} (raw loading skipped)")

        # Metadata: load or compute under an exclusive per-split lock so concurrent cold-cache jobs never race
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_lock_path = metadata_path.parent / "pretraining_metadata.lock"
        with FileLock(str(metadata_lock_path)):
            if (metadata_path / "vocab.pkl").exists() and (metadata_path / "bin_intervals.pkl").exists():
                active_meta_path = metadata_path
            elif (legacy_metadata_path / "vocab.pkl").exists() and (legacy_metadata_path / "bin_intervals.pkl").exists():
                active_meta_path = legacy_metadata_path
            else:
                active_meta_path = None

            if active_meta_path is not None:
                print(f"Loading pretraining metadata from {active_meta_path}")
                with open(active_meta_path / "vocab.pkl", "rb") as f:
                    vocab = pickle.load(f)
                with open(active_meta_path / "bin_intervals.pkl", "rb") as f:
                    bin_intervals = pickle.load(f)
            else:
                if is_pretraining and dataset is not None:
                    print("Pretraining mode. Computing statistics and vocabulary strictly on train split...")
                    scanned_data = dataset["train"]
                else:
                    print(f"Pretraining metadata not found at {metadata_path}; computing once from {root_path / 'fup_None'} train split...")
                    scanned_data = _load_and_tag(root_path, None, "train")
                bin_intervals, vocab = scan_and_compute_metadata(scanned_data)

                print(f"Saving new pre-training metadata to: {metadata_path}")
                metadata_path.mkdir(parents=True, exist_ok=True)
                with open(metadata_path / "bin_intervals.pkl", "wb") as f:
                    pickle.dump(bin_intervals, f)
                with open(metadata_path / "vocab.pkl", "wb") as f:
                    pickle.dump(vocab, f)

        # Load processed dataset from cache, or build and save a slim ID-only dataset
        if cache_hit:
            print(f"Loading processed dataset from cache: {processed_cache_path}")
            dataset = load_from_disk(str(processed_cache_path))
        else:
            print(f"Processed cache miss. Building slim shared dataset at: {processed_cache_path}")
            time_out_key = list(time_mapping.keys())[0]
            time_in_key = list(time_mapping.values())[0]
            protected_out_cols = {time_out_key, *eav_mappings.keys(), "value_num"}
            precomputed_bounds = _compute_bin_bounds(bin_intervals)
            local_tmp_base = _resolve_local_tmp_dir()
            with tempfile.TemporaryDirectory(
                prefix=f"hf_map_{processed_cache_path.name}_{os.getpid()}_",
                dir=local_tmp_base,
            ) as tmp_map_dir:
                mapped_splits = {}
                for split, ds in dataset.items():
                    raw_drop_cols = [
                        c for c in ("entity", "attribute", "value", "value_binned", time_in_key)
                        if c in ds.column_names and c not in protected_out_cols
                    ]
                    mapped_splits[split] = ds.map(
                        preprocess_batch,
                        batched=True,
                        batch_size=MAP_BATCH_SIZE,
                        remove_columns=raw_drop_cols,
                        desc=f"Preprocessing {split} set",
                        num_proc=(SAFE_NUM_PROCS if (os.name != "nt" and len(ds) >= 2000 and SAFE_NUM_PROCS > 1) else None),
                        load_from_cache_file=False,
                        keep_in_memory=False,
                        cache_file_name=str(Path(tmp_map_dir) / f"{split}_cache.arrow"),
                        fn_kwargs={
                            "vocab": vocab,
                            "bin_intervals": bin_intervals,
                            "bin_labels": BIN_LABELS,
                            "time_mapping": time_mapping,
                            "eav_mappings": eav_mappings,
                            "bin_bounds": precomputed_bounds,
                        },
                    )
                dataset = DatasetDict(mapped_splits)

                dataset.save_to_disk(str(processed_cache_path))
                cache_meta_path.write_text(cache_json)
                print(f"Saved slim processed dataset cache to: {processed_cache_path}")
                print(f"Saved cache metadata to: {cache_meta_path}")
            # Reload from final disk path so memory-mapped files point to processed_cache_path, not tmp_map_dir
            dataset = load_from_disk(str(processed_cache_path))

    # Apply task-specific per-FUP drop_unlabeled & undersampling in memory (~0.05s via PyArrow Table.take)
    if not is_pretraining:
        dataset = _apply_post_cache_filtering(
            dataset=dataset,
            fup_by_split={"train": fup_train, "validation": fup_valid, "test": fup_test},
            label_keys=label_keys,
            target_undersampling_ratio=target_undersampling_ratio,
            undersampling_balancing_label=undersampling_balancing_label,
            drop_unlabeled=drop_unlabeled,
        )

    # Sanity check, if required
    if sanity_check_output_dir is not None:
        plot_category_distributions(
            dataset,
            vocab=vocab,
            output_dir=sanity_check_output_dir,
            entity_key="entity_id",
            attribute_key="attribute_id",
            value_key="value_id",
        )

    # Final formatting and filtering
    all_cols = dataset["train"].column_names
    if label_keys:
        label_cols = [c for c in label_keys if c in all_cols]
    else:
        label_cols = [c for c in all_cols if c.startswith("label_")]
    time_key = list(time_mapping.keys())[0]
    teav_cols = list(eav_mappings.keys()) + ["value_num", time_key]
    cols_to_keep = [*teav_cols, *label_cols, "patientid", "fup"]
    dataset.set_format(type="numpy", columns=cols_to_keep)

    return dataset, bin_intervals, vocab


def normalize_numerical_value(
    f_val: float,
    lefts: np.ndarray,
    rights: np.ndarray,
    n_bins: int,
    median_width: float | None = None,
    right_skewed: bool = False,
    left_skewed: bool = False,
) -> tuple[int, float]:
    """
    Map a raw numerical float `f_val` using the empirical quantile intervals `(lefts, rights)`
    into:
      1. `bin_idx` in `[0, n_bins - 1]` (discrete quantile bin index)
      2. `u` in `[-1.0, +1.0]`, a continuous empirical CDF value `2 * F(x) - 1`.
         When an attribute's quantile intervals exhibit right or left skewness
         (tail bin width > 2x median bin width), intra-bin interpolation automatically
         switches from linear to logarithmic compression (`log(1 + (r - 1)*t) / log(r)`)
         so stretched tail bins are smoothly resolved without compressing values.
    """
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


def _compute_bin_bounds(bin_intervals: dict | None) -> dict[str, tuple]:
    """Precompute quantile interval edges & skewness flags once per attribute."""
    bounds = {}
    for attr, iv in (bin_intervals or {}).items():
        lefts = np.asarray(iv.left, dtype=np.float64)
        rights = np.asarray(iv.right, dtype=np.float64)
        widths = np.maximum(rights - lefts, 0.0)
        pos_widths = widths[widths > 1e-12]
        med_w = float(np.median(pos_widths)) if len(pos_widths) > 0 else 1.0
        right_skewed = bool(len(widths) >= 3 and med_w > 1e-12 and (widths[-1] / med_w) > 2.0)
        left_skewed = bool(len(widths) >= 3 and med_w > 1e-12 and (widths[0] / med_w) > 2.0 and not right_skewed)
        bounds[attr] = (lefts, rights, len(iv), med_w, right_skewed, left_skewed)
    return bounds


def preprocess_batch(batch, vocab, bin_intervals, time_mapping, eav_mappings, bin_labels, bin_bounds=None):
    """
    Module-level function to clean time, bin values, compute quantile-normalized
    continuous numerical values (`value_num`), and map tokens to IDs in a single pass.
    """
    unk_id = vocab["[UNK]"]
    labels_map = {i: label for i, label in enumerate(bin_labels)}
    if bin_bounds is None:
        bin_bounds = _compute_bin_bounds(bin_intervals)
    
    # Process time
    time_out_key = list(time_mapping.keys())[0]
    time_in_key = list(time_mapping.values())[0]    
    out_time = []
    for seq in batch[time_in_key]:  # handle time cleaning (with NaN -> 0.0)
        out_time.append([0.0 if (t is None or t != t) else t for t in seq])

    # Process (entity, attribute, value) entries by batch
    ent_in = eav_mappings["entity_id"]      # usually "entity"
    attr_in = eav_mappings["attribute_id"]  # usually "attribute"
    val_in = "value" if "value" in batch else eav_mappings.get("value_id", "value_binned")
    out_ent_ids, out_attr_ids, out_val_ids, out_val_nums = [], [], [], []
    for entities, attributes, values in zip(batch[ent_in], batch[attr_in], batch[val_in]):
        
        # Map entity and attribute directly
        out_ent_ids.append([vocab.get(e, unk_id) for e in entities])
        out_attr_ids.append([vocab.get(a, unk_id) for a in attributes])
        
        # Map values (both discrete token ID and quantile-normalized float for numerical attributes)
        row_val_ids = []
        row_val_nums = []
        for v, attr in zip(values, attributes):
            token_str = str(v)
            norm_num = np.nan
            
            # Numerical attribute (empirical quantile CDF normalization + bin token)
            if attr in bin_bounds:
                lefts, rights, n_bins, med_w, r_skew, l_skew = bin_bounds[attr]
                try:
                    f_val = float(v)
                    idx, norm_num = normalize_numerical_value(
                        f_val, lefts, rights, n_bins,
                        median_width=med_w, right_skewed=r_skew, left_skewed=l_skew,
                    )
                    if np.isfinite(norm_num):
                        token_str = labels_map[min(idx, len(bin_labels) - 1)]
                except (ValueError, TypeError, KeyError):
                    if token_str in bin_labels:
                        b_idx = bin_labels.index(token_str)
                        norm_num = float(2.0 * ((b_idx + 0.5) / float(len(bin_labels))) - 1.0)
            
            # Categorical attribute (clean formatting, or pre-binned label from UI/CSV)
            else:
                if token_str in bin_labels:
                    b_idx = bin_labels.index(token_str)
                    norm_num = float(2.0 * ((b_idx + 0.5) / float(len(bin_labels))) - 1.0)
                else:
                    try:
                        token_str = str(int(float(v)))  # "1.0" -> "1" for categorical integers
                    except ValueError:
                        pass
            
            # Map the resulting token string to an ID
            row_val_ids.append(vocab.get(token_str, unk_id))
            row_val_nums.append(norm_num)
            
        out_val_ids.append(row_val_ids)
        out_val_nums.append(row_val_nums)

    # Return dictionary matching the model's expected input format
    return {
        time_out_key: out_time,
        "entity_id": out_ent_ids,
        "attribute_id": out_attr_ids,
        "value_id": out_val_ids,
        "value_num": out_val_nums,
    }


def scan_and_compute_metadata(dataset, attribute_key="attribute", value_key="value"):
    """
    Read-only scan of the training dataset to calculate quantile bin intervals and vocabulary.
    Resilient to isolated unparseable strings in otherwise continuous numerical attributes.
    """
    values_by_attr = defaultdict(list)
    non_num_by_attr = defaultdict(list)
    unique_tokens = set()

    if hasattr(dataset, "column_names") and value_key not in dataset.column_names and "value_binned" in dataset.column_names:
        value_key = "value_binned"

    for sample in tqdm(dataset, desc="Compute dataset metadata"):
        unique_tokens.update(sample["entity"])
        unique_tokens.update(sample["attribute"])

        for attr, val in zip(sample[attribute_key], sample[value_key]):
            try:
                f_val = float(val)
                if np.isfinite(f_val):
                    values_by_attr[attr].append(f_val)
            except (ValueError, TypeError):
                s_val = str(val)
                non_num_by_attr[attr].append(s_val)
                unique_tokens.add(s_val)

    rng = np.random.default_rng(1234)
    bin_intervals = {}
    all_attrs = set(values_by_attr.keys()) | set(non_num_by_attr.keys())
    for attr in all_attrs:
        values = values_by_attr.get(attr, [])
        n_num = len(values)
        n_cat = len(non_num_by_attr.get(attr, []))
        is_numerical = (n_num > 0) and ((n_num / float(n_num + n_cat)) >= 0.95) and (len(set(values)) > 10)
        if is_numerical:
            try:
                values_arr = np.asarray(values, dtype=np.float64)
                noise = rng.uniform(-1e-6, 1e-6, size=values_arr.shape)
                binned = pd.qcut(x=values_arr + noise, q=len(BIN_LABELS), duplicates="drop")
            except ValueError:
                binned = pd.cut(x=values, bins=len(BIN_LABELS))
            bin_intervals[attr] = binned.categories
        else:
            for v in set(values):
                if np.isfinite(v):
                    unique_tokens.add(str(int(v)) if v == int(v) else str(v))

    # Build vocabulary (always include special tokens and bin labels)
    base_vocab = {"[PAD]": 0, "[MASK]": 1, "[BOS]": 2, "[EOS]": 3, "[UNK]": 4}
    vocab = dict(base_vocab)
    unique_tokens.update(BIN_LABELS)  # e.g. "Highest", "Low", ...
    start_idx = len(vocab)
    for idx, term in enumerate(sorted(list(unique_tokens))):
        vocab[term] = idx + start_idx

    return bin_intervals, vocab


def _load_and_tag(
    root_path: Path,
    fup_input: int | list[int] | None,
    split: str,
    num_threads: int | None = None,
):
    """Load requested follow-up folders in parallel (order preserved), tag `fup`, and concatenate."""
    fups = fup_input if isinstance(fup_input, list) else [fup_input]

    def _load_one(fup):
        suffix = f"{fup:04d}" if fup is not None else "None"
        folder_path = root_path / f"fup_{suffix}"
        if not folder_path.exists():
            raise FileNotFoundError(f"Dataset not found: {folder_path}")
        ds = load_from_disk(str(folder_path))[split]
        fup_val = fup if fup is not None else -1
        return ds.add_column("fup", [fup_val] * len(ds))

    for fup in fups:
        suffix = f"{fup:04d}" if fup is not None else "None"
        if not (root_path / f"fup_{suffix}").exists():
            raise FileNotFoundError(f"Dataset not found: {root_path / f'fup_{suffix}'}")

    workers = min(max(1, num_threads or LOAD_NUM_THREADS), max(1, len(fups)))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        datasets = list(tqdm(
            executor.map(_load_one, fups),
            total=len(fups),
            desc=f"Loading {split} data from listed follow-up periods",
        ))

    return concatenate_datasets(datasets) if len(datasets) > 0 else Dataset.from_dict({})


def plot_category_distributions(
    dataset: DatasetDict,
    vocab: dict[str, int],
    output_dir: str,
    entity_key: str = "entity_id",
    attribute_key: str = "attribute_id",
    value_key: str = "value_id",
    max_samples: int = 50000, 
    max_categories_to_plot: int = 50
):
    """
    Plots distributions for all categorical attributes (binned or natively categorical).
    - Decodes Integer IDs back to Strings.
    - Organizes plots into subfolders based on Entity.
    - Vectorized for speed.
    """
    print(f"Running sanity check: plotting distributions to {output_dir}")
    
    # 1. Create a Reverse Vocab to decode IDs -> Strings
    id2token = {v: k for k, v in vocab.items()}
    
    # 2. Load data efficiently into Pandas
    # We use the training set and subsample if necessary to save RAM
    ds = dataset["train"]
    if len(ds) > max_samples:
        print(f"Subsampling {max_samples} patients for plotting efficiency...")
        ds = ds.select(range(max_samples), keep_in_memory=True)
        
    # Load only necessary columns
    # This comes as lists of lists (sequences), e.g. [[1, 1, ...], [2, 2, ...]]
    df = ds.select_columns([entity_key, attribute_key, value_key]).to_pandas()
    
    # 3. Explode: Flatten sequences into individual events (Long format)
    print("Flattening sequences for aggregation...")
    df = df.explode([entity_key, attribute_key, value_key])
    
    # Remove padding (assuming 0 is PAD in your vocab)
    df = df[df[entity_key] != 0]

    # 4. Decode: Convert Integers back to Strings
    print("Decoding tokens...")
    # Use map. fillna handles cases where an ID might not be in vocab (shouldn't happen)
    df["entity"] = df[entity_key].map(id2token).fillna("Unknown_Entity")
    df["attribute"] = df[attribute_key].map(id2token).fillna("Unknown_Attribute")
    df["value"] = df[value_key].map(id2token).fillna("Unknown_Value")

    # 5. Group and Plot
    # We group by (Entity, Attribute) to separate the charts
    grouped = df.groupby(["entity", "attribute"])
    
    # Helper to enforce specific order for binned values (Lowest -> Highest)
    bin_order_map = {label: i for i, label in enumerate(BIN_LABELS)}
    
    for (ent_name, attr_name), group_df in tqdm(grouped, desc="Generating plots"):
        
        # Count occurrences of each value
        val_counts = group_df["value"].value_counts()
        
        # --- CARDINALITY CHECK ---
        # If an attribute has > 50 unique values (like Transplant Age), 
        # it is likely continuous/numerical. We skip it to keep plots readable.
        if len(val_counts) > max_categories_to_plot:
            continue
        if len(val_counts) < 2:
            continue # Skip constant values

        # --- SORTING LOGIC ---
        unique_vals = val_counts.index.tolist()
        
        # If the values look like our Bin Labels (Low, High...), sort by logical order
        if any(v in bin_order_map for v in unique_vals):
            # Filter to keep only valid bin labels and sort them
            valid_bins = [v for v in unique_vals if v in bin_order_map]
            order = sorted(valid_bins, key=lambda x: bin_order_map[x])
        else:
            # Otherwise, sort by frequency (most frequent first)
            order = val_counts.index.tolist()

        # --- SAVE PLOT ---
        # 1. Create Subfolder for the Entity
        safe_ent = "".join(c if c.isalnum() or c in (' ', '-', '_') else "_" for c in str(ent_name)).strip()
        safe_attr = "".join(c if c.isalnum() or c in (' ', '-', '_') else "_" for c in str(attr_name)).strip()
        
        entity_dir = os.path.join(output_dir, safe_ent)
        os.makedirs(entity_dir, exist_ok=True)
        
        # 2. Plot
        plt.figure(figsize=(10, 6))
        sns.barplot(
            x=val_counts.reindex(order).index, 
            y=val_counts.reindex(order).values, 
            palette="viridis",
            order=order
        )
        
        plt.title(f"{ent_name}: {attr_name}")
        plt.xlabel("Value")
        plt.ylabel("Count")
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        plt.tight_layout()
        
        plt.savefig(os.path.join(entity_dir, f"{safe_attr}.png"))
        plt.close()

    print(f"Sanity check complete. Plots saved to {output_dir}")
