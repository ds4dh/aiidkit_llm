"""Prewarm processed caches for all split strategies, from the fresh v3.9 data.

Builds, per split (temporal/random/center):
  1. pt_*  cache  — MLM pretraining load (fup_train=None -> fup_None folder)
  2. ft_*  caches — every prediction task of the binary classification config,
                    with train_data_augment="all" (train = all fup folders,
                    valid/test = all fup folders), same params as train_*.py
                    so training runs hit the cache instead of rebuilding.

Run: /opt/venv/bin/python scripts/prewarm_caches.py
"""
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.patient_dataset import load_hf_data_and_metadata
from scripts.script_utils import scan_all_fups

SPLITS = ["temporal_split"]  # edit as needed
ROOT = Path(__file__).resolve().parents[1]


def load_cfg(name):
    with open(ROOT / "configs" / name) as f:
        return yaml.safe_load(f)


def prewarm_split(split, disc):
    disc_dir = Path(disc["data_dir"]) / split
    coll_d = disc["data_collator"]

    print(f"\n{'='*70}\n>>> SPLIT: {split}\n{'='*70}", flush=True)
    all_fups = scan_all_fups(disc_dir)
    print(f"[{split}] {len(all_fups)} fup folders: {all_fups[:3]}...{all_fups[-1:]}", flush=True)

    # 1) MLM pretraining cache (reads fup_None; shared by all tasks)
    print(f">>> [{split}] MLM pretrain cache", flush=True)
    load_hf_data_and_metadata(
        data_dir=str(disc_dir),
        fup_train=None, fup_valid=None,
        time_mapping=coll_d["time_mapping"],
        eav_mappings=coll_d["eav_mappings"],
    )

    # 2) Binary / Multi-task fine-tune caches (train_data_augment=all)
    for task_key, specs in disc["prediction_tasks"].items():
        sub_tasks = specs.get("tasks", [task_key]) if isinstance(specs, dict) else [task_key]
        for horizons in specs["horizons"]:
            hs = horizons if isinstance(horizons, list) else [horizons]
            label_keys = [f"label_{t}_{h:04d}d" for t in sub_tasks for h in hs]
            print(f">>> [{split}] binary ft cache: {task_key} ({len(label_keys)} heads)", flush=True)
            load_hf_data_and_metadata(
                data_dir=str(disc_dir),
                fup_train=all_fups, fup_valid=all_fups, fup_test=all_fups,
                label_keys=label_keys,
                target_undersampling_ratio=disc.get("target_undersampling_ratio"),
                drop_unlabeled=disc.get("drop_unlabeled", False),
                time_mapping=coll_d["time_mapping"],
                eav_mappings=coll_d["eav_mappings"],
            )


def main():
    disc = load_cfg("discriminative_training.yaml")
    for split in SPLITS:
        prewarm_split(split, disc)
    print("\nPREWARM COMPLETE for all splits.", flush=True)


if __name__ == "__main__":
    main()
