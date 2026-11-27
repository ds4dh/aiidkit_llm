"""Verify that v4.0 dynamic t-EAV stores exist (no cache prewarming needed)."""

import argparse
import sys
from pathlib import Path
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.dynamic_store import is_dynamic_teav_dir
from scripts.script_utils import scan_all_fups

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate v4.0 dynamic t-EAV stores (zero prewarming required).")
    parser.add_argument("--config", "-c", type=str, default="configs/discriminative_training.yaml")
    parser.add_argument("--splits", "-s", nargs="+", default=["random_split", "temporal_split", "center_split"])
    parser.add_argument("--data-dir", type=str, default=None)
    args = parser.parse_args()

    cfg_path = Path(args.config) if Path(args.config).is_absolute() else (ROOT / args.config)
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    base = Path(args.data_dir) if args.data_dir else Path(cfg["data_dir"])
    for split in args.splits:
        sp_dir = base / split
        if not is_dynamic_teav_dir(sp_dir):
            sys.exit(f"[ERROR] Missing v4.0 dynamic t-EAV store at: {sp_dir}")
        fups = scan_all_fups(sp_dir)
        print(f"[OK] {split}: dynamic t-EAV store ready ({len(fups)} eval FUPs, zero prewarming needed)")


if __name__ == "__main__":
    main()
