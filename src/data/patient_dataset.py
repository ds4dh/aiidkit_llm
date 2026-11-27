"""
Dynamic t-EAV Dataset Loader for AIIDKIT (v4.0).

Loads single-copy portable dynamic t-EAV stores (``manifest.json``,
``cohort.parquet``, ``sequences.npz``, ``labels.npz``) directly without
pre-sliced ``fup_XXXX`` folders or processed-cache prewarming:
  - Pretraining (``fup_train=None``): full trajectories up to temporal cutoff.
  - Fine-tuning train: continuous daily ``random_landmark`` (with per-epoch
    resampling) or ``fixed_grid``.
  - Validation & test: deterministic ``fixed_grid`` over evaluation milestones.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.data.dynamic_store import (
    BIN_LABELS,
    LandmarkSummaryPreprocessor,
    PortableDynamicTEAVDataset,
    _normalize_numerical_value as normalize_numerical_value,
    get_dynamic_classic_ml_store,
    get_dynamic_teav_store,
    is_dynamic_teav_dir,
)


def resolve_processed_cache_dir(root_path: Path | str) -> Path:
    """Kept for ``web_tool`` metadata lookup compatibility (dynamic stores need no cache)."""
    return Path(root_path)


def load_hf_data_and_metadata(
    data_dir: str | Path,
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
    undersampling_mode: str = "global",
    ipw_weight_cap: float = 10.0,
    patient_weight_alpha: float = 0.0,
    drop_unlabeled: bool = False,
    landmark_mode: str = "fixed_grid",
    random_landmark_stride_days: int = 30,
    random_landmark_max_fup: int = 3600,
    random_landmark_seed: int = 1234,
    max_length: int = 1022,
    use_global_salience_truncation: bool = True,
    separate_static: bool = False,
    classic_ml_data_dir: str | Path | None = None,
    landmark_preprocessor: LandmarkSummaryPreprocessor | None = None,
) -> tuple[dict[str, PortableDynamicTEAVDataset], dict[str, pd.IntervalIndex], dict[str, int]]:
    """Load train/validation/test dynamic t-EAV datasets, bin intervals, and vocabulary."""
    if isinstance(label_keys, str):
        label_keys = [label_keys]
    if target_undersampling_ratio is not None and not label_keys:
        raise ValueError("Cannot perform undersampling without valid 'label_keys'.")

    root_path = Path(data_dir)
    if not is_dynamic_teav_dir(root_path):
        raise FileNotFoundError(
            f"Expected a v4.0 dynamic t-EAV store (manifest.json + sequences.npz) at: {root_path}"
        )

    store = get_dynamic_teav_store(root_path)
    time_out_key = list(time_mapping.keys())[0] if time_mapping else "days_since_tpx"

    if fup_train is None:
        print(f"[Dynamic t-EAV] Pretraining store loaded from {root_path} (vocab={len(store.vocab)}, separate_static={separate_static})")
        datasets = {
            sp: PortableDynamicTEAVDataset(
                store=store,
                split=sp,
                mode="pretraining",
                time_out_key=time_out_key,
                max_length=max_length,
                use_global_salience_truncation=use_global_salience_truncation,
                separate_static=separate_static,
            )
            for sp in ("train", "validation", "test")
        }
        return datasets, store.bin_intervals_compat, store.vocab

    def _as_fup_list(val: int | list[int] | None) -> list[int]:
        if val is None:
            return list(store.default_eval_fups)
        return list(val) if isinstance(val, list) else [int(val)]

    tr_mode = "random_landmark" if landmark_mode == "random_landmark" else "fixed_grid"
    cml_store = None
    lm_prep = landmark_preprocessor
    if classic_ml_data_dir is not None:
        cml_store = get_dynamic_classic_ml_store(classic_ml_data_dir)
        if lm_prep is None:
            lm_prep = LandmarkSummaryPreprocessor(cml_store.semi_static_columns)
    print(
        f"[Dynamic t-EAV] Loaded {root_path} "
        f"(train={tr_mode}, stride={random_landmark_stride_days}d, vocab={len(store.vocab)}, "
        f"separate_static={separate_static}, landmark_summary={cml_store is not None})"
    )
    train_ds = PortableDynamicTEAVDataset(
        store=store,
        split="train",
        mode=tr_mode,
        fixed_fups=_as_fup_list(fup_train),
        label_keys=label_keys,
        time_out_key=time_out_key,
        max_length=max_length,
        use_global_salience_truncation=use_global_salience_truncation,
        max_fup_cap=random_landmark_max_fup,
        stride_days=random_landmark_stride_days,
        drop_unlabeled=drop_unlabeled,
        target_undersampling_ratio=target_undersampling_ratio,
        undersampling_balancing_label=undersampling_balancing_label,
        undersampling_mode=undersampling_mode,
        ipw_weight_cap=ipw_weight_cap,
        patient_weight_alpha=patient_weight_alpha,
        seed=random_landmark_seed,
        separate_static=separate_static,
        classic_ml_store=cml_store,
        landmark_preprocessor=lm_prep,
    )
    datasets = {
        "train": train_ds,
        "validation": PortableDynamicTEAVDataset(
            store=store,
            split="validation",
            mode="fixed_grid",
            fixed_fups=_as_fup_list(fup_valid),
            label_keys=label_keys,
            time_out_key=time_out_key,
            max_length=max_length,
            use_global_salience_truncation=use_global_salience_truncation,
            drop_unlabeled=drop_unlabeled,
            separate_static=separate_static,
            classic_ml_store=cml_store,
            landmark_preprocessor=lm_prep,
        ),
        "test": PortableDynamicTEAVDataset(
            store=store,
            split="test",
            mode="fixed_grid",
            fixed_fups=_as_fup_list(fup_test),
            label_keys=label_keys,
            time_out_key=time_out_key,
            max_length=max_length,
            use_global_salience_truncation=use_global_salience_truncation,
            drop_unlabeled=drop_unlabeled,
            separate_static=separate_static,
            classic_ml_store=cml_store,
            landmark_preprocessor=lm_prep,
        ),
    }
    return datasets, store.bin_intervals_compat, store.vocab
