#!/usr/bin/env python3
"""Generate Markdown summary tables across splits for Unified/Pan/Category Transformer and Classic ML models."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
import re
from datetime import datetime
import numpy as np

from src.tasks import UNIFIED_HORIZONS, UNIFIED_TASKS, get_task_meta

SPLITS_DEFAULT = ["random_split", "temporal_split", "center_split"]
JSON_CACHE: dict[Path, dict] = {}


def _load_json(path: Path) -> dict:
    if path not in JSON_CACHE:
        JSON_CACHE[path] = json.loads(path.read_text()) if path.exists() else {}
    return JSON_CACHE[path]


def _category_folder(task: str) -> str:
    group = get_task_meta(task)["group"]
    return {
        "Bacterial Infections": "multitask_bacteria",
        "Viral Infections": "multitask_viruses",
        "Fungal Infections": "multitask_fungi",
    }.get(group, task)


def _find_matching_results_json(parent_dir: Path, horizon: int, eval_split: str, require_fut_all: bool = True) -> dict:
    if not parent_dir.exists():
        return {}
    fname = f"{eval_split}_results.json"
    tag = f"{horizon:04d}"
    for d in sorted(parent_dir.glob("hrz(*)*"), key=lambda p: p.stat().st_mtime_ns, reverse=True):
        if not d.is_dir() or not (d / fname).exists():
            continue
        if require_fut_all and "_fut(" in d.name and "_fut(all)_" not in d.name:
            continue
        m = re.match(r"hrz\(([0-9\-]+)\)", d.name)
        if m and tag in m.group(1).split("-"):
            return _load_json(d / fname)
    return {}


def get_tf_horizon_metrics(results_dir: Path, family_dir: str, split: str, run_id: str, mt_folder: str, task: str, horizons: list[int], eval_split: str, metric: str) -> dict[int, float]:
    clean_run_id = re.sub(r"_(raw|bin)$", "", str(run_id))
    parent = results_dir / family_dir / split / f"{clean_run_id}_raw" / "finetuning" / mt_folder
    out: dict[int, float] = {}
    for h in horizons:
        d = _find_matching_results_json(parent, h, eval_split, require_fut_all=True)
        v = d.get(f"{eval_split}_all_{metric}_label_{task}_{h:04d}d")
        if v is not None and not np.isnan(v) and v >= 0:
            out[h] = float(v)
    return out


def get_cml_horizon_metrics(results_dir: Path, model: str, split: str, task: str, horizons: list[int], eval_split: str, metric: str) -> dict[int, float]:
    for m_dir in (model, f"{model}_raw"):
        parent = results_dir / "classic_ml" / split / m_dir / task
        if not parent.exists():
            continue
        out: dict[int, float] = {}
        for h in horizons:
            d = _find_matching_results_json(parent, h, eval_split, require_fut_all=False)
            v = d.get(f"{eval_split}_all_{metric}_label_{task}_{h:04d}d")
            if v is not None and not np.isnan(v) and v >= 0:
                out[h] = float(v)
        if out:
            return out
    return {}


def get_tf_multi_folder_horizon_metrics(
    results_dir: Path, family_dir: str, split: str, run_id: str, mt_folders: list[str], task: str, horizons: list[int], eval_split: str, metric: str
) -> dict[int, float]:
    out: dict[int, float] = {}
    for mt_folder in mt_folders:
        out.update(get_tf_horizon_metrics(results_dir, family_dir, split, run_id, mt_folder, task, horizons, eval_split, metric))
    return out


MODEL_COLUMNS = [
    ("B_CAT_MT",   lambda r, s, run_id, t, hs, es, m: get_tf_horizon_metrics(r, "transformer", s, run_id, _category_folder(t), t, hs, es, m)),
    ("LR",         lambda r, s, run_id, t, hs, es, m: get_cml_horizon_metrics(r, "logistic_regression", s, t, hs, es, m)),
    ("RF",         lambda r, s, run_id, t, hs, es, m: get_cml_horizon_metrics(r, "random_forest", s, t, hs, es, m)),
    ("XGB",        lambda r, s, run_id, t, hs, es, m: get_cml_horizon_metrics(r, "xgboost", s, t, hs, es, m)),
]


def compute_split_matrix(
    results_dir: Path,
    split: str,
    run_id: str,
    tasks: list[str],
    horizons: list[int],
    eval_split: str,
    metric: str,
    align_horizons: bool = True,
):
    rows = []
    for t in tasks:
        per_model = [fn(results_dir, split, run_id, t, horizons, eval_split, metric) for _, fn in MODEL_COLUMNS]
        non_empty_sets = [set(d.keys()) for d in per_model if d]
        common_hs = set.intersection(*non_empty_sets) if (align_horizons and len(non_empty_sets) >= 2) else None
        row_vals: list[float | None] = []
        for d in per_model:
            if not d:
                row_vals.append(None)
            elif common_hs:
                row_vals.append(float(np.mean([d[h] for h in sorted(common_hs)])))
            else:
                row_vals.append(float(np.mean(list(d.values()))))
        rows.append(row_vals)
    return rows


def format_markdown_table(tasks: list[str], rows: list[list[float | None]]) -> str:
    active_cols = [
        idx for idx in range(len(MODEL_COLUMNS))
        if any(r[idx] is not None for r in rows)
    ]
    if not active_cols:
        return "_No results found on disk for this slice._"

    headers = ["Task", *[f"`{MODEL_COLUMNS[i][0]}`" for i in active_cols]]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| :--- | " + " | ".join([":---:"] * len(active_cols)) + " |",
    ]
    for t, full_vals in zip(tasks, rows):
        vals = [full_vals[i] for i in active_cols]
        valid = [v for v in vals if v is not None]
        best = max(valid) if valid else None
        cells = []
        for v in vals:
            if v is None:
                cells.append("N/A")
            elif best is not None and abs(v - best) < 1e-6:
                cells.append(f"**{v:.4f}**")
            else:
                cells.append(f"{v:.4f}")
        lines.append(f"| `{t}` | " + " | ".join(cells) + " |")

    means = []
    for col_idx in active_cols:
        col_vals = [r[col_idx] for r in rows if r[col_idx] is not None]
        means.append(float(np.mean(col_vals)) if col_vals else None)
    valid_m = [m for m in means if m is not None]
    best_m = max(valid_m) if valid_m else None
    m_cells = [
        "N/A" if m is None else (f"**{m:.4f}**" if best_m is not None and abs(m - best_m) < 1e-6 else f"{m:.4f}")
        for m in means
    ]
    lines.append(f"| **Mean ({len(tasks)} Tasks)** | " + " | ".join(m_cells) + " |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Generate Markdown summary report across models, tasks, and splits.")
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--run-id", type=str, default="e00-a15-v60")
    parser.add_argument("--splits", nargs="+", default=SPLITS_DEFAULT)
    parser.add_argument("--tasks", nargs="+", default=UNIFIED_TASKS)
    parser.add_argument("--horizons", type=int, nargs="+", default=UNIFIED_HORIZONS)
    parser.add_argument(
        "--unaligned-horizons",
        action="store_true",
        help="Average each model over all available horizons instead of aligning to shared horizons per task.",
    )
    parser.add_argument("--output", type=Path, default=Path("results/summary_report.md"))
    args = parser.parse_args()

    clean_run_id = re.sub(r"_(raw|bin)$", "", str(args.run_id))
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    hrz_str = ", ".join(f"{h}d" for h in args.horizons)
    align_note = "Unaligned per-model average" if args.unaligned_horizons else "Aligned to shared horizons per task"
    md = [
        "# AIIDKIT Multi-Split Benchmark Summary Report",
        "",
        f"**Generated:** {now_str}  ",
        f"**Run ID:** `{clean_run_id}_raw` | **Horizons:** `{hrz_str}` ({align_note}) | **Tasks:** `{len(args.tasks)}`",
        "",
    ]

    for idx, split in enumerate(args.splits, start=1):
        md.append("---")
        md.append(f"## {idx}. Split: `{split}`")
        for eval_split, eval_title in [("validation", "Validation Set"), ("test", "Held-Out Test Set")]:
            for metric, m_title in [("roc_auc", "AUROC"), ("pr_auc", "PR-AUC")]:
                rows = compute_split_matrix(
                    args.results_dir, split, args.run_id, args.tasks, args.horizons,
                    eval_split, metric, align_horizons=not args.unaligned_horizons,
                )
                md.append(f"\n### `{split}` — {eval_title} {m_title}")
                md.append(format_markdown_table(args.tasks, rows))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(md) + "\n")
    print(f"[SUCCESS] Wrote {args.output}")


if __name__ == "__main__":
    main()
