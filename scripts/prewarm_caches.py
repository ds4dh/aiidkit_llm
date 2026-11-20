"""Prewarm slim processed dataset caches onto Baobab shared scratch (/srv/beegfs/scratch/...).

Builds, per split (random/temporal/center), with strict 0700 owner-only permissions:
  1. pt_*  cache — MLM pretraining load (fup_train=None -> fup_None folder) + vocab/bin_intervals
  2. ft_*  cache — Unified shared fine-tuning FUP cache across all 121 FUPs (shared across all tasks, horizons, and undersampling ratios)

Usage:
  apptainer exec --bind "$HOME:$HOME,/home/shares/ds4dh:/home/shares/ds4dh,/srv/beegfs/scratch:/srv/beegfs/scratch" \
      research-env.sif /opt/venv/bin/python scripts/prewarm_caches.py
  apptainer exec ... /opt/venv/bin/python scripts/prewarm_caches.py --splits random_split
"""
import argparse
import sys
from pathlib import Path
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.patient_dataset import load_hf_data_and_metadata, resolve_processed_cache_dir
from scripts.script_utils import scan_all_fups

ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description="Prewarm slim t-EAV processed caches onto external scratch.")
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="configs/discriminative_training.yaml",
        help="Path to training YAML config.",
    )
    parser.add_argument(
        "--splits",
        "-s",
        nargs="+",
        default=["random_split", "temporal_split", "center_split"],
        help="Splits to prewarm (default: random_split temporal_split center_split).",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Optional override for t-EAV root directory (defaults to `data_dir` in YAML config).",
    )
    return parser.parse_args()


def prewarm_split(split: str, disc: dict, data_dir_override: str | None = None):
    base_data_dir = Path(data_dir_override) if data_dir_override else Path(disc["data_dir"])
    disc_dir = base_data_dir / split
    coll_d = disc["data_collator"]
    cache_root = resolve_processed_cache_dir(disc_dir)

    print(f"\n{'=' * 70}\n>>> SPLIT: {split} | Cache root: {cache_root}\n{'=' * 70}", flush=True)
    all_fups = scan_all_fups(disc_dir)
    print(f"[{split}] {len(all_fups)} fup folders: {all_fups[:3]}...{all_fups[-1:]}", flush=True)

    # 1) MLM pretraining cache (reads fup_None; builds vocab.pkl & bin_intervals.pkl)
    print(f">>> [{split}] MLM pretrain cache + pretraining_metadata", flush=True)
    load_hf_data_and_metadata(
        data_dir=str(disc_dir),
        fup_train=None,
        fup_valid=None,
        time_mapping=coll_d["time_mapping"],
        eav_mappings=coll_d["eav_mappings"],
    )

    # 2) Unified fine-tuning FUP cache (shared across all tasks, horizons, and undersampling ratios)
    print(f">>> [{split}] Unified shared fine-tuning FUP cache ({len(all_fups)} FUPs)", flush=True)
    load_hf_data_and_metadata(
        data_dir=str(disc_dir),
        fup_train=all_fups,
        fup_valid=all_fups,
        fup_test=all_fups,
        label_keys=None,
        target_undersampling_ratio=None,
        drop_unlabeled=False,
        time_mapping=coll_d["time_mapping"],
        eav_mappings=coll_d["eav_mappings"],
    )


def main():
    args = parse_args()
    cfg_path = Path(args.config) if Path(args.config).is_absolute() else (ROOT / args.config)
    with open(cfg_path, "r") as f:
        disc = yaml.safe_load(f)
    for split in args.splits:
        prewarm_split(split, disc, data_dir_override=args.data_dir)
    print("\nPREWARM COMPLETE for requested splits.", flush=True)


if __name__ == "__main__":
    main()
