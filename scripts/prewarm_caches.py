"""Prewarm processed caches for all split strategies, from the fresh v3.9 data.

Builds, per split (temporal/random/center):
  1. pt_*  cache  — MLM pretraining load (fup_train=None -> fup_None folder)
  2. ft_*  caches — every prediction task of the survival and binary configs,
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

# Skip cache builds whose label contains any of these substrings (e.g. virus
# endpoints not currently of interest). Task lists come from the configs.
SKIP_SUBSTRINGS = ["virus"]


def load_cfg(name):
    with open(ROOT / "configs" / name) as f:
        return yaml.safe_load(f)


def prewarm_split(split, cfgs):
    surv, disc = cfgs["surv"], cfgs["disc"]
    surv_dir = Path(surv["data_dir"]) / split
    disc_dir = Path(disc["data_dir"]) / split
    coll_s, coll_d = surv["data_collator"], disc["data_collator"]

    print(f"\n{'='*70}\n>>> SPLIT: {split}\n{'='*70}", flush=True)
    all_fups = scan_all_fups(surv_dir)
    print(f"[{split}] {len(all_fups)} fup folders: {all_fups[:3]}...{all_fups[-1:]}", flush=True)

    # 1) MLM pretraining cache (reads fup_None; shared by all tasks)
    print(f">>> [{split}] MLM pretrain cache", flush=True)
    load_hf_data_and_metadata(
        data_dir=str(surv_dir),
        fup_train=None, fup_valid=None,
        time_mapping=coll_s["time_mapping"],
        eav_mappings=coll_s["eav_mappings"],
    )

    # 2) Survival fine-tune caches (train_data_augment=all)
    for task_key in surv["prediction_tasks"]:
        if any(s in task_key for s in SKIP_SUBSTRINGS):
            print(f"[{split}] SKIP survival {task_key}", flush=True)
            continue
        from src.data.survival_labels import resolve_survival_label_keys
        from src.data.survival_labels import resolve_survival_label_keys
        time_key, status_key, _ = resolve_survival_label_keys(task_key)
        print(f">>> [{split}] survival ft cache: {task_key} ({time_key}, {status_key})", flush=True)
        load_hf_data_and_metadata(
            data_dir=str(surv_dir),
            fup_train=all_fups, fup_valid=all_fups, fup_test=all_fups,
            label_keys=[time_key, status_key],
            target_undersampling_ratio=surv.get("target_undersampling_ratio"),
            undersampling_balancing_label=status_key,
            drop_unlabeled=surv.get("drop_unlabeled", False),
            time_mapping=coll_s["time_mapping"],
            eav_mappings=coll_s["eav_mappings"],
        )

    # 3) Binary fine-tune caches (train_data_augment=all)
    for task_key, specs in disc["prediction_tasks"].items():
        if any(s in task_key for s in SKIP_SUBSTRINGS):
            print(f"[{split}] SKIP binary {task_key}", flush=True)
            continue
        for horizons in specs["horizons"]:
            hs = horizons if isinstance(horizons, list) else [horizons]
            label_keys = [f"label_{task_key}_{h:04d}d" for h in hs]
            print(f">>> [{split}] binary ft cache: {task_key} {label_keys}", flush=True)
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
    cfgs = {"surv": load_cfg("survival_training.yaml"),
            "disc": load_cfg("discriminative_training.yaml")}
    for split in SPLITS:
        prewarm_split(split, cfgs)
    print("\nPREWARM COMPLETE for all splits.", flush=True)


if __name__ == "__main__":
    main()
