"""Head-to-head comparison of a proposed classification model against
contender baselines (e.g., t-EAV-transformer vs classic ML models).

Inputs: Use '{split}_predictions.csv' files written by the training
scripts (e.g., results/transformer/... and results/classic_ml/...).
Note: Field 'prob' in the csv files must mean P(event by horizon H).

Usage:
    python scripts/analysis_comparison.py \
        --proposed-model tf_raw_mt \
        --contender-models tf_raw_st logistic_regression random_forest xgboost

Models are declared in MODEL_REGISTRY (path_template / display / full / calibrated)
Unknown model names fall back to the classic_ml layout with a generated abbreviation.
Outputs land under results/analysis/comparison/.../<PROPOSED>-vs-<CONTENDERS>/.
Derived caches are invalidated from the input CSV fingerprints.
"""
import sys
import io
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import argparse
import functools
import hashlib
import json
import re
import yaml
from concurrent.futures import ThreadPoolExecutor

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from scripts.script_utils import (
    build_patient_cluster_bootstrap_counts,
    calibrate_array_pair,
    calibrate_dataframe_pair,
    clustered_mcnemar_pvalue,
    paired_bootstrap_pr_auc_pvalue,
    vectorized_bootstrap_ap_and_roc,
    vectorized_bootstrap_counts,
)


# ================================================================================
# Command-line arguments
# ================================================================================

parser = argparse.ArgumentParser(description="Evaluation pipeline")
parser.add_argument(
    "--threshold_mode",
    choices=["global", "window_specific"],
    default="window_specific",
    help="Whether to use a single global threshold for all clinical windows or compute a separate threshold per window.",
)
parser.add_argument(
    "--target_recall",
    type=int,
    default=80,
    help="Target recall percentage (0-100) for threshold selection.",
)
parser.add_argument(
    "--workers",
    type=int,
    default=14,
    help="Number of threads used for bootstrap resampling (does not affect results).",
)
parser.add_argument(
    "--config", "-c",
    default=None,
    help="YAML training config used to discover active tasks when --tasks is omitted.",
)
parser.add_argument(
    "--tasks", "-t",
    nargs="+",
    default=None,
    # default from config when --tasks is omitted
    help="Outcome tasks to analyze (one comparison report per task). When the proposed "
         "model's predictions CSV carries a 'task' column (multi-task runs), this "
         "selects the corresponding head per task. E.g. --tasks infection_bacteria "
         "infection_grp_cmv",
)
parser.add_argument(
    "--proposed-model",
    default="tf_raw_mt",
    help="Name of the proposed model (registry key: tf_raw_mt, tf_raw_st, tf_bin_mt, tf_bin_st). See MODEL_REGISTRY.",
)
parser.add_argument(
    "--contender-models",
    nargs="+",
    default=["logistic_regression", "random_forest", "xgboost"],
    help="List of contender/baseline model names (registry keys) to compare against the proposed model.",
)
parser.add_argument(
    "--run-id",
    default="e00-a15-v60",
    help="Pre-training run id used to locate transformer-family result directories.",
)
parser.add_argument(
    "--results-dir",
    default="results",
    help="Directory containing all results.",
)
parser.add_argument(
    "--splits",
    nargs="+",
    default=["random_split", "temporal_split", "center_split"],
    help="Data split strategies to include in the comparison.",
)
parser.add_argument(
    "--horizons",
    type=int,
    nargs="+",
    default=None,
    help="Optional explicit prediction horizons in days (e.g. --horizons 30 60 90 180 360 720 1800). "
         "Defaults to [30, 60, 90] for infections and [360, 720, 1800] for death/graft_loss.",
)
args = parser.parse_args()

THRESHOLD_MODE = args.threshold_mode
TARGET_RECALL_INPUT = args.target_recall
TARGET_RECALL = TARGET_RECALL_INPUT / 100.0
THRESHOLD_TUNING_SET = "validation"
MAX_WORKERS = max(1, args.workers)
RESULTS_DIR = Path(args.results_dir)


# ================================================================================
# Model registry for classification model comparison.
# Each entry four fields (identical for all models):
#   path_template : the FULL path (relative to RESULTS_DIR = --results-dir,
#                   default "results/")
#                   of the folder that DIRECTLY contains
#                   'validation_predictions.csv' and 'test_predictions.csv'.
#                   There is NO auto-discovery: the path you write here is
#                   exactly the folder that is read.
#                   Optional placeholders filled in by the script:
#                     {split}        random_split | temporal_split | center_split
#                     {run_id}       value of --run-id (pretraining run id)
#                     {task}         infection_bacteria | infection_virus
#                     {model}        the registry key itself
#                     {horizon:04d}  zero-padded horizon (0030, 0060, 0090),
#                                    for models storing one folder per horizon
#                   Literal path segments (like the hrz(...)/fuv(...) folder
#                   names below) are simply fixed directory names produced by
#                   the training scripts; keep or drop them as your output
#                   tree requires.
#                   If a template is horizon-independent, one CSV may hold
#                   several horizons: rows are filtered on its 'horizon'
#                   column. Required CSV columns: patient_id, fup, horizon,
#                   label, prob.
#   display       : short label (plot legend + output subfolder name).
#   full          : full model name (used in report text).
#   calibrated    : True  -> the script calibrates the probabilities in memory
#                            before computing metrics (Platt scaling: an
#                            L2-regularized logistic fit on validation
#                            logit-space probabilities, applied to the test
#                            set);
#                   False -> probabilities are used as-is.
# ================================================================================

_DEFAULT_FUV_TAG = "0000-0030-0060-0090-0120-0150-0180-0240-0300-0360-0540-0720-0900-1080-1440-1800-2700-3600"

MODEL_REGISTRY = {
    "tf_raw_st": dict(
        path_template=f"transformer/{{split}}/{{run_id}}_raw/finetuning/{{task}}/hrz(0030-0060-0090)_fut(all)_fuv({_DEFAULT_FUV_TAG})",
        display="TFRS",
        folder_tag="tf_raw_st",
        full="t-EAV-Transformer (single-task, raw values)",
        calibrated=True,
        val_tag="raw",
        multitask=False,
        results_subdir="transformer",
    ),
    "tf_raw_mt": dict(
        path_template=f"transformer/{{split}}/{{run_id}}_raw/finetuning/multitask_bacteria/hrz(0030-0060-0090)_fut(all)_fuv({_DEFAULT_FUV_TAG})",
        display="TFRM",
        folder_tag="tf_raw_mt",
        full="t-EAV-Transformer",
        calibrated=True,
        val_tag="raw",
        multitask=True,
        multitask_scope="domain",
        results_subdir="transformer",
    ),
    "tf_raw_unified": dict(
        path_template=f"transformer/{{split}}/{{run_id}}_raw/finetuning/multitask_unified/hrz(0030-0060-0090-0180-0360-0720-1800)_fut(all)_fuv({_DEFAULT_FUV_TAG})",
        display="TFRU",
        folder_tag="tf_raw_unified",
        full="t-EAV-Transformer (Unified 13t×7h)",
        calibrated=True,
        val_tag="raw",
        multitask=True,
        multitask_scope="unified",
        results_subdir="transformer",
    ),
    "tf_raw_bv": dict(
        path_template=f"transformer/{{split}}/{{run_id}}_raw/finetuning/multitask_bv/hrz(0030-0060-0090)_fut(all)_fuv({_DEFAULT_FUV_TAG})",
        display="TFBV",
        folder_tag="tf_raw_bv",
        full="t-EAV-Transformer (Acute B+V 9t)",
        calibrated=True,
        val_tag="raw",
        multitask=True,
        multitask_scope="multitask_bv",
        results_subdir="transformer",
    ),
    "tf_raw_inf": dict(
        path_template=f"transformer/{{split}}/{{run_id}}_raw/finetuning/multitask_infections/hrz(0030-0060-0090)_fut(all)_fuv({_DEFAULT_FUV_TAG})",
        display="TFINF",
        folder_tag="tf_raw_inf",
        full="t-EAV-Transformer (Acute Infections 11t)",
        calibrated=True,
        val_tag="raw",
        multitask=True,
        multitask_scope="multitask_infections",
        results_subdir="transformer",
    ),
    "tf_raw_pan13": dict(
        path_template=f"transformer/{{split}}/{{run_id}}_raw/finetuning/multitask_pan_13/hrz(0030-0060-0090-0360-0720-1800)_fut(all)_fuv({_DEFAULT_FUV_TAG})",
        display="TFP13",
        folder_tag="tf_raw_pan13",
        full="t-EAV-Transformer (Pan-13 Matched Horizons)",
        calibrated=True,
        val_tag="raw",
        multitask=True,
        multitask_scope="multitask_pan_13",
        results_subdir="transformer",
    ),
    "tf_bin_st": dict(
        path_template=f"transformer/{{split}}/{{run_id}}_bin/finetuning/{{task}}/hrz(0030-0060-0090)_fut(all)_fuv({_DEFAULT_FUV_TAG})",
        display="TFBS",
        folder_tag="tf_bin_st",
        full="t-EAV-Transformer (single-task, binned values)",
        calibrated=True,
        val_tag="bin",
        multitask=False,
        results_subdir="transformer",
    ),
    "tf_bin_mt": dict(
        path_template=f"transformer/{{split}}/{{run_id}}_bin/finetuning/multitask_bacteria/hrz(0030-0060-0090)_fut(all)_fuv({_DEFAULT_FUV_TAG})",
        display="TFBM",
        folder_tag="tf_bin_mt",
        full="t-EAV-Transformer (multi-task, binned values)",
        calibrated=True,
        val_tag="bin",
        multitask=True,
        multitask_scope="domain",
        results_subdir="transformer",
    ),
    "tf_bin_unified": dict(
        path_template=f"transformer/{{split}}/{{run_id}}_bin/finetuning/multitask_unified/hrz(0030-0060-0090-0180-0360-0720-1800)_fut(all)_fuv({_DEFAULT_FUV_TAG})",
        display="TFBU",
        folder_tag="tf_bin_unified",
        full="t-EAV-Transformer (unified multi-task, binned values)",
        calibrated=True,
        val_tag="bin",
        multitask=True,
        multitask_scope="unified",
        results_subdir="transformer",
    ),
    "tfg_raw_st": dict(
        path_template=f"generative_transformer/{{split}}/{{run_id}}_raw/finetuning/{{task}}/hrz(0030-0060-0090)_fut(all)_fuv({_DEFAULT_FUV_TAG})",
        display="TFGRS",
        folder_tag="tfg_raw_st",
        full="Generative t-EAV-Transformer (single-task, raw values)",
        calibrated=True,
        val_tag="raw",
        multitask=False,
        results_subdir="generative_transformer",
    ),
    "tfg_raw_mt": dict(
        path_template=f"generative_transformer/{{split}}/{{run_id}}_raw/finetuning/multitask_bacteria/hrz(0030-0060-0090)_fut(all)_fuv({_DEFAULT_FUV_TAG})",
        display="TFGRM",
        folder_tag="tfg_raw_mt",
        full="Generative t-EAV-Transformer (multi-task, raw values)",
        calibrated=True,
        val_tag="raw",
        multitask=True,
        multitask_scope="domain",
        results_subdir="generative_transformer",
    ),
    "tfg_raw_unified": dict(
        path_template=f"generative_transformer/{{split}}/{{run_id}}_raw/finetuning/multitask_unified/hrz(0030-0060-0090-0180-0360-0720-1800)_fut(all)_fuv({_DEFAULT_FUV_TAG})",
        display="TFGRU",
        folder_tag="tfg_raw_unified",
        full="Generative t-EAV-Transformer (unified multi-task, raw values)",
        calibrated=True,
        val_tag="raw",
        multitask=True,
        multitask_scope="unified",
        results_subdir="generative_transformer",
    ),
    "tfg_bin_st": dict(
        path_template=f"generative_transformer/{{split}}/{{run_id}}_bin/finetuning/{{task}}/hrz(0030-0060-0090)_fut(all)_fuv({_DEFAULT_FUV_TAG})",
        display="TFGBS",
        folder_tag="tfg_bin_st",
        full="Generative t-EAV-Transformer (single-task, binned values)",
        calibrated=True,
        val_tag="bin",
        multitask=False,
        results_subdir="generative_transformer",
    ),
    "tfg_bin_mt": dict(
        path_template=f"generative_transformer/{{split}}/{{run_id}}_bin/finetuning/multitask_bacteria/hrz(0030-0060-0090)_fut(all)_fuv({_DEFAULT_FUV_TAG})",
        display="TFGBM",
        folder_tag="tfg_bin_mt",
        full="Generative t-EAV-Transformer (multi-task, binned values)",
        calibrated=True,
        val_tag="bin",
        multitask=True,
        multitask_scope="domain",
        results_subdir="generative_transformer",
    ),
    "tfg_bin_unified": dict(
        path_template=f"generative_transformer/{{split}}/{{run_id}}_bin/finetuning/multitask_unified/hrz(0030-0060-0090-0180-0360-0720-1800)_fut(all)_fuv({_DEFAULT_FUV_TAG})",
        display="TFGBU",
        folder_tag="tfg_bin_unified",
        full="Generative t-EAV-Transformer (unified multi-task, binned values)",
        calibrated=True,
        val_tag="bin",
        multitask=True,
        multitask_scope="unified",
        results_subdir="generative_transformer",
    ),
    "survival_transformer": dict(
        path_template=f"survival_transformer/{{split}}/{{run_id}}_raw/finetuning/{{task}}/hrz(0030-0060-0090-0180-0360-0720-1800)_fut(all)_fuv({_DEFAULT_FUV_TAG})",
        display="SURV",
        folder_tag="surv",
        full="t-EAV-Transformer (Cox)",
        calibrated=False,
        val_tag="raw",
        multitask=False,
        results_subdir="survival_transformer",
    ),
    "logistic_regression": dict(
        path_template=f"classic_ml/{{split}}/logistic_regression/{{task}}/hrz({{horizon:04d}})_fut(all)_fuv({_DEFAULT_FUV_TAG})",
        display="LR",
        folder_tag="lr",
        full="Logistic regression",
        calibrated=True,
    ),
    "random_forest": dict(
        path_template=f"classic_ml/{{split}}/random_forest/{{task}}/hrz({{horizon:04d}})_fut(all)_fuv({_DEFAULT_FUV_TAG})",
        display="RF",
        folder_tag="rf",
        full="Random forest",
        calibrated=True,
    ),
    "xgboost": dict(
        path_template=f"classic_ml/{{split}}/xgboost/{{task}}/hrz({{horizon:04d}})_fut(all)_fuv({_DEFAULT_FUV_TAG})",
        display="XGB",
        folder_tag="xgb",
        full="XGBoost",
        calibrated=True,
    ),
}


_MODEL_ALIASES = {
    "tf_raw": "tf_raw_st",
    "tf_bin": "tf_bin_st",
    "tfg_raw": "tfg_raw_st",
    "tfg_bin": "tfg_bin_st",
    "lr_raw": "logistic_regression",
    "rf_raw": "random_forest",
    "xgb_raw": "xgboost",
}


def _registry_entry(name: str) -> dict:
    """Return registry metadata for a model, with a graceful fallback for unknown names."""
    canon = _MODEL_ALIASES.get(name, name)
    if canon in MODEL_REGISTRY:
        return dict(MODEL_REGISTRY[canon])
    disp = "".join(w[0].upper() for w in re.split(r"[_\-\s]+", name) if w) or name[:3].upper()
    print(f"[WARN] Model '{name}' not in MODEL_REGISTRY; assuming the default "
          f"classic_ml layout (classic_ml/<split>/{name}/<task>/hrz(<horizon>)_"
          f"fut(all)_fuv(...)) and display name '{disp}'. If that is not where "
          f"your CSVs are, add an explicit entry to MODEL_REGISTRY.")
    return dict(
        path_template=(f"classic_ml/{{split}}/{name}/{{task}}/"
                       "hrz({horizon:04d})_fut(all)_fuv(0000-0030-0060-0090-0180-0360)"),
        display=disp,
        folder_tag=name,
        full=name.replace("_", " ").capitalize(),
        calibrated=True,
    )


# ================================================================================
# Model parameters and derived paths
# ================================================================================

PROPOSED_MODEL = args.proposed_model
CONTENDER_MODELS = list(args.contender_models)
if PROPOSED_MODEL in CONTENDER_MODELS:
    raise ValueError(f"--proposed-model '{PROPOSED_MODEL}' must not also appear in --contender-models.")
if not CONTENDER_MODELS:
    raise ValueError("--contender-models must contain at least one model.")

MODEL_ENTRIES = {m: _registry_entry(m) for m in [PROPOSED_MODEL, *CONTENDER_MODELS]}
MODEL_RUN_ID = re.sub(r"_(raw|bin)$", "", args.run_id.strip())
USE_CALIBRATED_PROBS = {m: e["calibrated"] for m, e in MODEL_ENTRIES.items()}
THRESHOLD_SUBFOLDER = f"rec-{TARGET_RECALL_INPUT}"
CALIB_SUBFOLDER = f"calib-{int(USE_CALIBRATED_PROBS[PROPOSED_MODEL])}"
ANALYSIS_DIR = RESULTS_DIR / "analysis" / "comparison" / "lt-1-3_vlt-3-10" / THRESHOLD_SUBFOLDER
MODEL_SET_TAG = (
    f"{MODEL_ENTRIES[PROPOSED_MODEL].get('folder_tag', PROPOSED_MODEL)}-vs-"
    + "-".join(MODEL_ENTRIES[m].get("folder_tag", m) for m in CONTENDER_MODELS)
)
ANALYSIS_SUBDIR = ANALYSIS_DIR / THRESHOLD_MODE / CALIB_SUBFOLDER / MODEL_SET_TAG
CACHE_DIR = ANALYSIS_DIR / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)



# ================================================================================
# Model comparison constants and utilities
# ================================================================================

ALL_MODELS = [PROPOSED_MODEL, *CONTENDER_MODELS]
PROPOSED_DISP = MODEL_ENTRIES[PROPOSED_MODEL]["display"]
def _resolve_tasks_from_config(config_arg: str | None) -> list[str]:
    """Extract individual task keys from a training YAML config (expanding multi-task entries)."""
    candidates = (
        [Path(config_arg)] if config_arg else [
            # Default training config (Option 1 multi-task)
            Path("configs/discriminative_training.yaml"),
            Path("configs/discriminative_classic_ml.yaml"),
        ]
    )
    resolved: list[str] = []
    for cfg_path in candidates:
        if not cfg_path.exists():
            continue
        with open(cfg_path, "r") as f:
            cfg = yaml.safe_load(f) or {}
        for t_key, t_specs in (cfg.get("prediction_tasks") or {}).items():
            sub_tasks = t_specs.get("tasks", [t_key]) if isinstance(t_specs, dict) else [t_key]
            for st in sub_tasks:
                if st not in resolved:
                    resolved.append(st)
        if config_arg and resolved:
            break
    return resolved or ["infection_bacteria", "infection_virus"]

TASKS = list(args.tasks) if args.tasks else _resolve_tasks_from_config(args.config)
SPLIT_TYPES = list(args.splits)
TARGET_HORIZONS = [int(h) for h in args.horizons] if args.horizons else [30, 60, 90]
LONG_TERM_ENDPOINT_HORIZONS = {
    "death": [int(h) for h in args.horizons] if args.horizons else [360, 720, 1800],
    "graft_loss": [int(h) for h in args.horizons] if args.horizons else [360, 720, 1800],
}


def _horizons_for_task(task: str) -> list[int]:
    if args.horizons:
        return [int(h) for h in args.horizons]
    return list(LONG_TERM_ENDPOINT_HORIZONS.get(task, TARGET_HORIZONS))

N_BOOTSTRAP = 2000

CLINICAL_WINDOWS = {
    "Perioperative\n(0-30 days)": (0, 30),
    "Opportunistic\n(31-180 days)": (31, 180),
    "Maintenance\n(181-360 days)": (181, 360),
    "Long-term\n(361-1080 days)": (361, 1080),
    "Very long-term\n(1081-3600 days)": (1081, 3600),
}
CONCISE_WINDOW_LABELS = ["POP", "OPT", "MTN", "LT", "VLT"]

MODEL_DISPLAY_MAP = {m: e["display"] for m, e in MODEL_ENTRIES.items()}
MODEL_FULLNAME_MAP = {m: e["full"] for m, e in MODEL_ENTRIES.items()}
BASELINE_FULL_NAME_MAP = {MODEL_DISPLAY_MAP[m]: m for m in CONTENDER_MODELS}
ALL_ABBREVS = {MODEL_DISPLAY_MAP[m] for m in ALL_MODELS}

MODEL_PLOT_ORDER = [*CONTENDER_MODELS, PROPOSED_MODEL]  # contenders first, proposed model last (highlighted in red).
CONTENDER_PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b", "#e377c2", "#17becf", "#bcbd22"]
PROPOSED_COLOR = "#d62728"
COLORS = {
    **{m: CONTENDER_PALETTE[i % len(CONTENDER_PALETTE)] for i, m in enumerate(CONTENDER_MODELS)},
    PROPOSED_MODEL: PROPOSED_COLOR,
}

METRIC_MAPPING = [
    ("ROC-AUC", "↑"), ("PR-AUC", "↑"), ("ECE", "↓"),
    ("Sensitivity", "↑"), ("Precision", "↑"), ("Specificity", "↑"),
]
MAIN_METRIC_MAPPING = [
    ("ROC-AUC", "↑"), ("PR-AUC", "↑"), ("ECE", "↓"),
    ("Sensitivity", "↑"), ("Specificity", "↑"),
]


# ================================================================================
# Cache logic (some computations are long)
# ================================================================================

CACHE_SCHEMA_VERSION = 7

def _predictions_csv(model: str, split: str, task: str, dataset_split: str, horizon: int) -> Path:
    """Resolve the exact CSV path for one model/split/task/horizon."""
    rel = MODEL_ENTRIES[model]["path_template"].format(
        split=split, run_id=MODEL_RUN_ID, task=task, model=model, horizon=horizon,
    )
    cand = RESULTS_DIR / rel / f"{dataset_split}_predictions.csv"
    if not cand.exists() and (task.startswith("infection_grp_") or task.startswith("grp_")):
        alt_task = task.replace("infection_grp_", "grp_", 1) if task.startswith("infection_grp_") else f"infection_{task}"
        alt_rel = MODEL_ENTRIES[model]["path_template"].format(
            split=split, run_id=MODEL_RUN_ID, task=alt_task, model=model, horizon=horizon,
        )
        alt_cand = RESULTS_DIR / alt_rel / f"{dataset_split}_predictions.csv"
        if alt_cand.exists():
            return alt_cand
    entry = MODEL_ENTRIES[model]
    is_tf_single = ("val_tag" in entry and not entry.get("multitask", False))
    is_tf_multi = ("val_tag" in entry and entry.get("multitask", False))
    if is_tf_single and cand.exists():
        return cand
    if is_tf_single or is_tf_multi:
        val_tag = entry["val_tag"]
        results_subdir = entry.get("results_subdir", "transformer")
        mt_scope = entry.get("multitask_scope", "domain")
        valid_cands = []
        if is_tf_single and cand.exists():
            valid_cands.append(cand)
        ft_root = RESULTS_DIR / results_subdir / split / f"{MODEL_RUN_ID}_{val_tag}" / "finetuning"
        if ft_root.exists():
            for meta_file in sorted(ft_root.glob("*/*/run_metadata.json")):
                if "_fut(" in meta_file.parent.name and "_fut(all)_" not in meta_file.parent.name:
                    continue
                try:
                    meta = json.loads(meta_file.read_text())
                    tasks_in_run = meta.get("tasks", [])
                    run_task_key = meta.get("task_key", meta_file.parent.parent.name)
                    is_standalone_category = (
                        is_tf_multi and mt_scope == "domain" and task in ("death", "graft_loss") and tasks_in_run == [task]
                    )
                    if is_tf_multi and len(tasks_in_run) <= 1 and not is_standalone_category:
                        continue
                    if is_tf_single and len(tasks_in_run) > 1:
                        continue
                    if task not in tasks_in_run:
                        continue
                    if meta.get("head_pairs"):
                        if [task, int(horizon)] not in [[str(t), int(h)] for t, h in meta["head_pairs"]]:
                            continue
                    elif meta.get("horizons"):
                        if int(horizon) not in [int(h) for h in meta["horizons"]]:
                            continue
                    mt_cand = meta_file.parent / f"{dataset_split}_predictions.csv"
                    if not mt_cand.exists():
                        continue
                    is_unified = (run_task_key == "multitask_unified")
                    is_other_pan = (
                        run_task_key in (
                            "multitask_all", "multitask_pan_clinical", "multitask_infections",
                            "multitask_longterm", "multitask_acute_13", "multitask_longterm_13",
                            "multitask_infections_11", "multitask_outcomes_2",
                        )
                        or len(tasks_in_run) > 6
                    )
                    if not is_tf_multi:
                        valid_cands.append(mt_cand)
                    elif mt_scope == "unified" and is_unified:
                        valid_cands.append(mt_cand)
                    elif mt_scope == run_task_key:
                        valid_cands.append(mt_cand)
                    elif mt_scope == "domain" and not is_unified and not is_other_pan:
                        valid_cands.append(mt_cand)
                except Exception:
                    continue
        if valid_cands:
            return max(valid_cands, key=lambda p: p.stat().st_mtime_ns)
        if cand.exists() and not (cand.parent / "run_metadata.json").exists():
            return cand
    if not cand.exists() and "_fuv(" in str(cand.parent):
        glob_pattern = re.sub(r"_fuv\([^)]*\)", "_fuv(*)", cand.parent.name)
        fuv_cands = [
            d / f"{dataset_split}_predictions.csv"
            for d in cand.parent.parent.glob(glob_pattern)
            if (d / f"{dataset_split}_predictions.csv").exists()
        ]
        if fuv_cands:
            return max(fuv_cands, key=lambda p: p.stat().st_mtime_ns)
    return cand


def compute_param_hash(task: str, split: str, horizon) -> str:
    """Per-(task, split, horizon) cache hash — invariant to CLI `--tasks` or `--splits` subsets."""
    hs_list = _horizons_for_task(task) if horizon == "combined" else [int(horizon)]
    paths = []
    for model in ALL_MODELS:
        for h in hs_list:
            for ds in ("validation", "test"):
                p = _predictions_csv(model, split, task, ds, h)
                if p.exists():
                    paths.append(p)
    paths = sorted(set(paths))
    csv_sig = "|".join(f"{p}:{p.stat().st_mtime_ns}:{p.stat().st_size}" for p in paths)
    sig_obj = {
        "schema": CACHE_SCHEMA_VERSION,
        "threshold_mode": THRESHOLD_MODE,
        "target_recall": TARGET_RECALL,
        "tuning_set": THRESHOLD_TUNING_SET,
        "calibration": USE_CALIBRATED_PROBS,
        "n_bootstrap": N_BOOTSTRAP,
        "windows": CLINICAL_WINDOWS,
        "models": ALL_MODELS,
        "proposed_model": PROPOSED_MODEL,
        "contender_models": CONTENDER_MODELS,
        "horizons": hs_list,
        "run_id": MODEL_RUN_ID,
        "csv_sig": hashlib.sha256(csv_sig.encode()).hexdigest()[:12],
    }
    return hashlib.sha256(json.dumps(sig_obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:12]


# ================================================================================
# Utility and path functions
# ================================================================================

def horizon_name(h):
    """Return a formatted string representation for the prediction horizon."""
    return f"horizon_{h:04d}d" if isinstance(h, int) else "horizon_combined"


def cache_path(task, split, hs, phash):
    """Generate path for numpy cache files."""
    return CACHE_DIR / f"plot_cache_v{CACHE_SCHEMA_VERSION}_{task}_{split}_{hs}_{phash}.npz"


def winners_path(task, split, hs, phash):
    """Generate path for JSON files storing statistical test winners."""
    return CACHE_DIR / f"winners_v{CACHE_SCHEMA_VERSION}_{task}_{split}_{hs}_{phash}.json"


def raw_cache(kind, *x):
    """Generate path for intermediate parquet data caches."""
    return CACHE_DIR / (kind + "_" + "_".join(map(str, x)) + ".parquet")


def ece(y, p, n_bins=10):
    """Calculate Expected Calibration Error (ECE)."""
    if len(y) == 0:
        return np.nan
    bins = np.linspace(0, 1, n_bins + 1)
    result = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (p >= lo) & ((p < hi) if hi < 1 else (p <= hi))
        if mask.any():
            result += mask.mean() * abs(p[mask].mean() - y[mask].mean())
    return float(result)


# ================================================================================
# Data loading functions
# ================================================================================

@functools.lru_cache(maxsize=6)
def _read_predictions_csv_cached(csv_path_str: str, mtime_ns: int, size: int) -> pd.DataFrame:
    """LRU-cached CSV reader so multi-hundred-MB multi-task prediction CSVs are parsed once."""
    return pd.read_csv(csv_path_str)


def _load_predictions_csv_slice(csv_path: Path, horizon: int, task: str | None = None) -> pd.DataFrame:
    """
    Read a tidy '{dataset_split}_predictions.csv' written by the training
    scripts and return the classic analysis schema:
        patient_id (= patient id), time_step (= fup), horizon, y_true, y_prob

    Multi-task prediction CSVs mix several outcome tasks in one file (extra
    'task' column): keep only the rows of the task under analysis.
    """
    st = csv_path.stat()
    df = _read_predictions_csv_cached(str(csv_path), int(st.st_mtime_ns), int(st.st_size))
    if "task" in df.columns and task is not None:
        df = df[df["task"] == task]
    df = df[df["horizon"] == horizon]
    if df.empty:
        return pd.DataFrame()
    return pd.DataFrame({
        "patient_id": df["patient_id"].astype(str),
        "time_step": df["fup"].astype(int),
        "horizon": df["horizon"].astype(int),
        "y_true": df["label"].astype(int),
        "y_prob": df["prob"].astype(float),
    }).reset_index(drop=True)


def load_model_predictions(model, task, split, dataset_split, horizon):
    """Load a model's predictions from the tidy per-run '{dataset_split}_predictions.csv'.

    The only raw input this analysis uses: no raw datasets, no checkpoints.
    """
    if horizon == "combined":
        x = [load_model_predictions(model, task, split, dataset_split, h) for h in _horizons_for_task(task)]
        return (
            pd.concat([d for d in x if not d.empty], ignore_index=True)
            if any(not d.empty for d in x) else pd.DataFrame()
        )

    src = _predictions_csv(model, split, task, dataset_split, horizon)
    if not src.exists():
        return pd.DataFrame()

    # Fingerprint the source CSV (mtime + size) in the cache key so that
    # regenerated CSVs or a different --run-id never serve stale parquet.
    st = src.stat()
    cp = raw_cache("raw", model, task, split, dataset_split, horizon,
                   f"{int(st.st_mtime_ns)}_{st.st_size}")
    if cp.exists():
        try:
            return pd.read_parquet(cp)
        except Exception:
            pass

    df = _load_predictions_csv_slice(src, horizon, task=task)
    if not df.empty:
        tmp_cp = cp.with_suffix(f".{os.getpid()}.tmp.parquet")
        df.to_parquet(tmp_cp, index=False)
        os.replace(tmp_cp, cp)
    return df


# ================================================================================
# Evaluation and statistical metrics
# ================================================================================

def _filter_clinical_window(df: pd.DataFrame, lo: int, hi: int, task: str = "") -> pd.DataFrame:
    """Filter prediction frames strictly by the post-transplant landmark visit `lo <= time_step <= hi`.

    Ensures identical landmark visit cohorts across all prediction horizons (30d, 60d, 90d
    or 360d, 720d, 1800d) and across all tasks without dropping boundary follow-up visits.
    """
    if df.empty:
        return df
    return df[(df.time_step >= lo) & (df.time_step <= hi)]


def threshold(df, lo, hi, label, task=""):
    """Compute classification threshold based on target recall on `df` within `[lo, hi]`."""
    d = _filter_clinical_window(df, lo, hi, task=task)
    if d.empty or d.y_true.nunique() != 2:
        if df is not None and not df.empty and df.y_true.nunique() == 2:
            d = df
        else:
            return 0.5

    _, recall, thresholds = precision_recall_curve(d.y_true, d.y_prob)
    idx = np.where(recall[:-1] >= TARGET_RECALL)[0]
    if not len(idx):
        return float(thresholds[0]) if len(thresholds) else 0.5
    return float(thresholds[idx[-1]])


def metric(y, p, t, kind):
    """Evaluate specific performance metrics given labels, probabilities, and threshold."""
    if kind == "ROC-AUC":
        return roc_auc_score(y, p) if np.unique(y).size == 2 else 0.5
    if kind == "PR-AUC":
        return average_precision_score(y, p) if np.unique(y).size == 2 else 0.0
    if kind == "ECE":
        return ece(y, p)

    b = p >= t
    tp = np.sum(b & (y == 1))
    fp = np.sum(b & (y == 0))
    tn = np.sum(~b & (y == 0))
    fn = np.sum(~b & (y == 1))

    if kind == "Sensitivity":
        return tp / (tp + fn) if tp + fn else 0.0
    if kind == "Precision":
        return tp / (tp + fp) if tp + fp else 0.0
    if kind == "Specificity":
        return tn / (tn + fp) if tn + fp else 0.0
    raise ValueError(kind)


def _compute_slice_points_and_boots(df: pd.DataFrame, model: str, t: float, valid_counts: np.ndarray):
    """Return `(points_dict, boot_arrays_dict)` for all 6 metrics on a single-horizon window slice."""
    y = df["y_true"].to_numpy()
    p = df[f"y_prob_{model}"].to_numpy()
    n_samples = len(y)
    kinds = [k for k, _ in METRIC_MAPPING]
    points = {k: float(metric(y, p, t, k)) for k in kinds}
    if valid_counts.shape[0] == 0:
        return points, {k: np.empty(0, dtype=np.float64) for k in kinds}

    ap_boot, roc_boot = vectorized_bootstrap_ap_and_roc(
        y_true=y, y_prob=p, counts=valid_counts, valid_boot_mask=None, compute_roc=True
    )

    n_bins = 10
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    binids = np.clip(np.digitize(p, bins) - 1, 0, n_bins - 1)
    diff_mat = np.zeros((n_samples, n_bins), dtype=np.float64)
    diff_mat[np.arange(n_samples), binids] = y.astype(np.float64) - p.astype(np.float64)
    boot_totals = np.maximum(1.0, np.sum(valid_counts, axis=1))
    ece_boot = np.sum(np.abs(valid_counts @ diff_mat), axis=1) / boot_totals

    b = p >= t
    tp_b = valid_counts @ (b & (y == 1)).astype(np.float64)
    fp_b = valid_counts @ (b & (y == 0)).astype(np.float64)
    tn_b = valid_counts @ (~b & (y == 0)).astype(np.float64)
    fn_b = valid_counts @ (~b & (y == 1)).astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        sens_boot = np.where(tp_b + fn_b > 0, tp_b / (tp_b + fn_b), 0.0)
        prec_boot = np.where(tp_b + fp_b > 0, tp_b / (tp_b + fp_b), 0.0)
        spec_boot = np.where(tn_b + fp_b > 0, tn_b / (tn_b + fp_b), 0.0)

    return points, {
        "ROC-AUC": roc_boot,
        "PR-AUC": ap_boot,
        "ECE": ece_boot,
        "Sensitivity": sens_boot,
        "Precision": prec_boot,
        "Specificity": spec_boot,
    }


def _two_sided_paired_boot_pvalue(boot_prop: np.ndarray, boot_cont: np.ndarray) -> float:
    """Symmetric two-sided paired cluster-bootstrap p-value with finite-sample +1 correction."""
    if len(boot_prop) == 0 or len(boot_prop) != len(boot_cont):
        return np.nan
    diffs = boot_prop - boot_cont
    if np.all(np.abs(diffs) < 1e-15):
        return 1.0
    n_b = float(len(diffs))
    p_le = (float(np.sum(diffs <= 0.0)) + 1.0) / (n_b + 1.0)
    p_ge = (float(np.sum(diffs >= 0.0)) + 1.0) / (n_b + 1.0)
    return float(min(1.0, 2.0 * min(p_le, p_ge)))


def aligned(dfs):
    """Align patient cohorts across all model dataframes (proposed model is the anchor)."""
    keys = ["patient_id", "time_step", "horizon", "y_true"]
    common = dfs[PROPOSED_MODEL][keys].drop_duplicates()
    for m in CONTENDER_MODELS:
        common = common.merge(dfs[m][keys].drop_duplicates(), on=keys, how="inner")

    if common.empty:
        raise RuntimeError("No common samples across all models")
    return {
        m: dfs[m].merge(common, on=keys, how="inner").sort_values(keys).reset_index(drop=True)
        for m in ALL_MODELS
    }


# ================================================================================
# Report structuring (ROC-AUC primary test + PR-AUC + Clustered McNemar)
# ================================================================================

def build_report_layout_and_mapping():
    """Build the row order and display-name mapping for the detailed CSV report."""
    metric_sort_order = ["Total evaluation frames"]
    row_cleaning_map = {"Total evaluation frames": "Total evaluation frames"}

    for m in CONTENDER_MODELS:
        disp = MODEL_DISPLAY_MAP[m]
        key_head = f"{PROPOSED_DISP} vs {disp}"
        metric_sort_order.append(key_head)
        row_cleaning_map[key_head] = key_head

        sub_metrics = [
            (f"  ROC-AUC winner ({disp})", "  Statistical winner (ROC-AUC)"),
            (f"  Threshold-free ROC-AUC p-value ({disp})", f"  ROC-AUC cluster-bootstrap p-value ({PROPOSED_DISP} vs {disp})"),
            (f"  Threshold-free ROC-AUC q-value ({disp})", f"  ROC-AUC cluster-bootstrap FDR q-value ({PROPOSED_DISP} vs {disp})"),
            (f"  PR-AUC winner ({disp})", "  Statistical winner (PR-AUC)"),
            (f"  Threshold-free PR-AUC p-value ({disp})", f"  PR-AUC cluster-bootstrap p-value ({PROPOSED_DISP} vs {disp})"),
            (f"  Threshold-free PR-AUC q-value ({disp})", f"  PR-AUC cluster-bootstrap FDR q-value ({PROPOSED_DISP} vs {disp})"),
            (f"  McNemar winner ({disp})", "  Statistical winner (McNemar)"),
            (f"  McNemar p-value ({disp})", "  Clustered McNemar p-value"),
            (f"  McNemar q-value ({disp})", "  Clustered McNemar FDR q-value"),
            (f"  Discordant pairs ({PROPOSED_DISP} correct) ({disp})", f"  Discordant pairs ({PROPOSED_DISP} correct)"),
            (f"  Discordant pairs ({disp} correct) ({disp})", f"  Discordant pairs ({disp} correct)"),
        ]
        for internal_key, clean_name in sub_metrics:
            metric_sort_order.append(internal_key)
            row_cleaning_map[internal_key] = clean_name

    for metric_prefix, arrow in METRIC_MAPPING:
        head_lbl = f"{metric_prefix} ({arrow})"
        metric_sort_order.append(head_lbl)
        row_cleaning_map[head_lbl] = head_lbl
        for model in ALL_MODELS:
            disp = MODEL_DISPLAY_MAP[model]
            metric_sort_order.append(f"  {head_lbl} ({disp})")
            row_cleaning_map[f"  {head_lbl} ({disp})"] = f"  {disp}"

    return metric_sort_order, row_cleaning_map


def fdr_and_reports(results, outdir, split, hs):
    """Apply Benjamini-Hochberg FDR separately within each statistical test family
    (`ROC-AUC`, `PR-AUC`, `McNemar`) and export detailed + main summary CSV reports.
    """
    test_fields = (
        ("ROC-AUC", "  Threshold-free ROC-AUC p-value ({disp})"),
        ("PR-AUC", "  Threshold-free PR-AUC p-value ({disp})"),
        ("McNemar", "  McNemar p-value ({disp})"),
    )
    q_map = {}
    for test_key, field_tmpl in test_fields:
        fam_entries = []
        for window_name, record in results.items():
            for m in CONTENDER_MODELS:
                disp = MODEL_DISPLAY_MAP[m]
                try:
                    p_val = float(record.get(field_tmpl.format(disp=disp), "NaN"))
                    if not np.isnan(p_val):
                        fam_entries.append({"window": window_name, "model": disp, "p_val": p_val})
                except ValueError:
                    pass
        if fam_entries:
            p_arr = np.asarray([e["p_val"] for e in fam_entries], dtype=np.float64)
            n_p = len(p_arr)
            order = np.argsort(p_arr)
            ranked = p_arr[order] * n_p / np.arange(1, n_p + 1, dtype=np.float64)
            ranked_monotone = np.minimum.accumulate(ranked[::-1])[::-1]
            q_vals = np.empty_like(p_arr)
            q_vals[order] = np.clip(ranked_monotone, 0.0, 1.0)
            for e, q in zip(fam_entries, q_vals):
                q_map[(e["window"], e["model"], test_key)] = float(q)

    def format_p_or_q(val):
        if np.isnan(val):
            return "NaN"
        return "< 0.001" if val < 0.001 else f"= {val:.3f}"

    def format_sci(val):
        return "NaN" if np.isnan(val) else f"{val:.3e}"

    main_table_data = {}
    for window_name, record in results.items():
        main_win_record = {"N frames": record["Total evaluation frames"]}

        for metric_prefix, arrow in MAIN_METRIC_MAPPING:
            head_lbl = f"{metric_prefix} ({arrow})"
            main_win_record[head_lbl] = ""
            for model in MODEL_PLOT_ORDER:
                disp = MODEL_DISPLAY_MAP[model]
                main_win_record[f"  {disp}_{head_lbl}"] = record.get(f"  {head_lbl} ({disp})", "")

        main_win_record["ROC-AUC bootstrap statistical winner and p-value"] = ""
        main_win_record["PR-AUC bootstrap statistical winner and p-value"] = ""
        main_win_record["McNemar statistical winner and p-value"] = ""

        for m in CONTENDER_MODELS:
            disp = MODEL_DISPLAY_MAP[m]

            for test_key, w_field, q_field in (
                ("ROC-AUC", f"  ROC-AUC winner ({disp})", f"  Threshold-free ROC-AUC q-value ({disp})"),
                ("PR-AUC", f"  PR-AUC winner ({disp})", f"  Threshold-free PR-AUC q-value ({disp})"),
                ("McNemar", f"  McNemar winner ({disp})", f"  McNemar q-value ({disp})"),
            ):
                q_val = q_map.get((window_name, disp, test_key), np.nan)
                raw_winner = record.get(w_field, "Tie")
                fin_winner = "Tie" if (np.isnan(q_val) or q_val >= 0.05) else raw_winner
                record[w_field] = fin_winner
                record[q_field] = format_sci(q_val)
                main_win_record[f"  {PROPOSED_DISP} vs {disp} ({test_key})"] = f"{fin_winner} (q {format_p_or_q(q_val)})"

        main_table_data[window_name] = main_win_record

    metric_sort_order, row_cleaning_map = build_report_layout_and_mapping()
    df_detailed = pd.DataFrame(results).reindex(metric_sort_order)
    df_detailed.index = df_detailed.index.map(row_cleaning_map)
    df_detailed.index.name = "Evaluation metric"
    df_detailed = df_detailed.fillna("")
    df_detailed.to_csv(outdir / f"{split}_{hs}_head_to_head_report.csv")

    main_rows_order = ["N frames"]
    for metric_prefix, arrow in MAIN_METRIC_MAPPING:
        head_lbl = f"{metric_prefix} ({arrow})"
        main_rows_order.append(head_lbl)
        for model in MODEL_PLOT_ORDER:
            main_rows_order.append(f"  {MODEL_DISPLAY_MAP[model]}_{head_lbl}")
    for hdr, tkey in (
        ("ROC-AUC bootstrap statistical winner and p-value", "ROC-AUC"),
        ("PR-AUC bootstrap statistical winner and p-value", "PR-AUC"),
        ("McNemar statistical winner and p-value", "McNemar"),
    ):
        main_rows_order.append(hdr)
        for m in CONTENDER_MODELS:
            main_rows_order.append(f"  {PROPOSED_DISP} vs {MODEL_DISPLAY_MAP[m]} ({tkey})")

    df_main = pd.DataFrame(main_table_data).reindex(main_rows_order)
    df_main.index.name = "Evaluation metric"
    df_main = df_main.fillna("")
    df_main.to_csv(outdir / f"{split}_{hs}_main_summary_report.csv")

    winners = {}
    for window_name in results:
        winners[window_name] = {}
        for m in CONTENDER_MODELS:
            disp = MODEL_DISPLAY_MAP[m]
            winners[window_name][disp] = {
                tkey: {
                    "winner": results[window_name].get(f"  {tkey} winner ({disp})", "Tie"),
                    "q_value": q_map.get((window_name, disp, tkey)),
                }
                for tkey in ("ROC-AUC", "PR-AUC", "McNemar")
            }

    return df_main, df_detailed, winners


# ================================================================================
# Main processing pipeline (per-horizon calibration + macro-averaged combined)
# ================================================================================

def _load_calibrated_horizon_pair(task: str, split: str, h: int):
    """Load, align, and globally Platt-calibrate `(val, test)` for one single horizon `h`."""
    val_h = {m: load_model_predictions(m, task, split, "validation", h) for m in ALL_MODELS}
    test_h = {m: load_model_predictions(m, task, split, "test", h) for m in ALL_MODELS}
    if any(d.empty for d in val_h.values()) or any(d.empty for d in test_h.values()):
        return None, None
    val_h = aligned(val_h)
    test_h = aligned(test_h)
    for m in ALL_MODELS:
        if USE_CALIBRATED_PROBS[m]:
            val_h[m], test_h[m] = calibrate_dataframe_pair(df_val=val_h[m], df_test=test_h[m], prob_col="y_prob")
    return val_h, test_h


def process(task, split, horizon, outdir):
    """Run model evaluation pipeline for a specific task, split, and horizon."""
    hs = horizon_name(horizon)
    phash = compute_param_hash(task, split, horizon)
    cp = cache_path(task, split, hs, phash)
    wp = winners_path(task, split, hs, phash)
    main = outdir / f"{split}_{hs}_main_summary_report.csv"
    detailed = outdir / f"{split}_{hs}_head_to_head_report.csv"

    if cp.exists() and main.exists() and detailed.exists():
        try:
            with np.load(cp, allow_pickle=False) as z:
                if (
                    int(z["__schema_version__"]) != CACHE_SCHEMA_VERSION
                    or str(z["__param_hash__"]) != phash
                ):
                    raise RuntimeError("schema or parameter hash mismatch")
                payload = {k: z[k] for k in z.files}
            cache = {}
            for w in CLINICAL_WINDOWS:
                if payload.get(f"{w}_is_empty", False):
                    continue
                if f"{w}_time_step" in payload:
                    req = [
                        f"{w}_time_step", f"{w}_horizon", f"{w}_y_true", f"{w}_thresholds_json",
                        *[f"{w}_y_prob_{m}" for m in ALL_MODELS],
                    ]
                    missing = [k for k in req if k not in payload]
                    if missing:
                        raise KeyError(f"{w}: missing {missing}")
                    df = pd.DataFrame({
                        "time_step": payload[f"{w}_time_step"],
                        "horizon": payload[f"{w}_horizon"],
                        "y_true": payload[f"{w}_y_true"],
                        **{f"y_prob_{m}": payload[f"{w}_y_prob_{m}"] for m in ALL_MODELS},
                    })
                    cache[w] = (df, json.loads(str(payload[f"{w}_thresholds_json"])))
            print(f"[CACHE HIT] split={split}, horizon={hs}, hash={phash}")
            return cache, pd.read_csv(main, index_col="Evaluation metric")
        except Exception as exc:
            print(f"[CACHE INVALID] split={split}, horizon={hs}: {type(exc).__name__}: {exc}")
    else:
        print(f"[CACHE MISS] split={split}, horizon={hs}, hash={phash}")

    hs_list = _horizons_for_task(task) if horizon == "combined" else [int(horizon)]
    per_h_data: dict[int, tuple[dict, dict, dict]] = {}
    for h in hs_list:
        vh, th = _load_calibrated_horizon_pair(task, split, h)
        if vh is None or th is None:
            continue
        tune_h = vh if THRESHOLD_TUNING_SET == "validation" else th
        glob_h = (
            {m: threshold(tune_h[m], 0, 9999, f"{m}@{h}d", task=task) for m in ALL_MODELS}
            if THRESHOLD_MODE == "global" else {}
        )
        per_h_data[h] = (vh, th, glob_h)

    if not per_h_data:
        print(f"[SKIP] missing predictions for {split}/{hs}")
        return None, None

    results = {}
    cache = {}
    payload = {}
    kinds = [k for k, _ in METRIC_MAPPING]

    for w, (lo, hi) in CLINICAL_WINDOWS.items():
        h_slices: dict[int, tuple[pd.DataFrame, dict[str, float]]] = {}
        for h, (vh, th, glob_h) in per_h_data.items():
            sub_h = (
                _filter_clinical_window(th[PROPOSED_MODEL], lo, hi, task=task)[
                    ["patient_id", "time_step", "horizon", "y_true", "y_prob"]
                ].rename(columns={"y_prob": f"y_prob_{PROPOSED_MODEL}"})
            )
            for m in CONTENDER_MODELS:
                x = (
                    _filter_clinical_window(th[m], lo, hi, task=task)[
                        ["patient_id", "time_step", "horizon", "y_prob"]
                    ].rename(columns={"y_prob": f"y_prob_{m}"})
                )
                sub_h = sub_h.merge(x, on=["patient_id", "time_step", "horizon"], how="inner")
            if sub_h.empty or sub_h.y_true.nunique() != 2:
                continue
            tune_h = vh if THRESHOLD_TUNING_SET == "validation" else th
            ts_h = {
                m: (glob_h[m] if THRESHOLD_MODE == "global" else threshold(tune_h[m], lo, hi, f"{m}@{h}d", task=task))
                for m in ALL_MODELS
            }
            h_slices[h] = (sub_h, ts_h)

        if not h_slices:
            print(f"[WINDOW SKIP] {split}/{hs}/{w}: empty or single-class test set")
            payload[f"{w}_is_empty"] = np.array(True)
            continue

        # Build ONE shared patient-cluster bootstrap over unique patients in this window
        all_win_pids = np.unique(np.concatenate([s["patient_id"].astype(str).to_numpy() for s, _ in h_slices.values()]))
        pid_to_cluster = {pid: idx for idx, pid in enumerate(all_win_pids)}
        n_clusters = len(all_win_pids)
        rng = np.random.default_rng(42)
        if n_clusters > 1:
            cluster_boot = rng.integers(0, n_clusters, size=(N_BOOTSTRAP, n_clusters))
            cluster_counts = vectorized_bootstrap_counts(cluster_boot, n_clusters)
        else:
            cluster_counts = None

        counts_by_h = {}
        joint_valid_mask = np.ones(N_BOOTSTRAP, dtype=bool)
        for h, (sub_h, _) in h_slices.items():
            y_h = sub_h["y_true"].to_numpy(dtype=np.float64)
            if cluster_counts is not None:
                inv_h = np.array([pid_to_cluster[p] for p in sub_h["patient_id"].astype(str)], dtype=np.int64)
                c_h = cluster_counts[:, inv_h]
            else:
                c_h = vectorized_bootstrap_counts(rng.integers(0, len(y_h), size=(N_BOOTSTRAP, len(y_h))), len(y_h))
            pos_b = c_h @ y_h
            tot_b = np.sum(c_h, axis=1)
            counts_by_h[h] = c_h
            joint_valid_mask &= (pos_b > 0) & (pos_b < tot_b)

        # Compute per-horizon point estimates & bootstrap arrays, then equal-weight macro-average across horizons
        model_points: dict[str, dict[str, float]] = {}
        model_boots: dict[str, dict[str, np.ndarray]] = {}
        for m in ALL_MODELS:
            pts_per_h = []
            boots_per_h = []
            for h, (sub_h, ts_h) in h_slices.items():
                pts_h, b_h = _compute_slice_points_and_boots(sub_h, m, ts_h[m], counts_by_h[h][joint_valid_mask])
                pts_per_h.append(pts_h)
                boots_per_h.append(b_h)
            model_points[m] = {k: float(np.mean([p[k] for p in pts_per_h])) for k in kinds}
            model_boots[m] = {
                k: (np.mean(np.stack([b[k] for b in boots_per_h], axis=0), axis=0) if boots_per_h[0][k].size else np.empty(0))
                for k in kinds
            }

        # Stack per-horizon slices (each carrying its own per-horizon threshold for McNemar & curve plotting)
        stacked_parts = []
        ts_by_horizon_json = {}
        for h, (sub_h, ts_h) in sorted(h_slices.items()):
            part = sub_h.copy()
            for m in ALL_MODELS:
                part[f"y_pred_{m}"] = (part[f"y_prob_{m}"].to_numpy() >= ts_h[m])
            stacked_parts.append(part)
            ts_by_horizon_json[str(h)] = {m: float(ts_h[m]) for m in ALL_MODELS}
        sub = pd.concat(stacked_parts, ignore_index=True)

        y_stack = sub["y_true"].to_numpy()
        pids_stack = sub["patient_id"].astype(str).to_numpy()
        rec = {"Total evaluation frames": len(sub)}

        for m in CONTENDER_MODELS:
            disp = MODEL_DISPLAY_MAP[m]
            # 1) Primary threshold-free ROC-AUC paired patient-cluster bootstrap test
            p_roc = _two_sided_paired_boot_pvalue(model_boots[PROPOSED_MODEL]["ROC-AUC"], model_boots[m]["ROC-AUC"])
            tf_roc, b_roc = model_points[PROPOSED_MODEL]["ROC-AUC"], model_points[m]["ROC-AUC"]
            roc_winner = "NaN" if pd.isna(p_roc) else ("Tie" if (p_roc >= 0.05 or tf_roc == b_roc) else (PROPOSED_DISP if tf_roc > b_roc else disp))

            # 2) Threshold-free PR-AUC paired patient-cluster bootstrap test
            p_pr = _two_sided_paired_boot_pvalue(model_boots[PROPOSED_MODEL]["PR-AUC"], model_boots[m]["PR-AUC"])
            tf_pr, b_pr = model_points[PROPOSED_MODEL]["PR-AUC"], model_points[m]["PR-AUC"]
            pr_winner = "NaN" if pd.isna(p_pr) else ("Tie" if (p_pr >= 0.05 or tf_pr == b_pr) else (PROPOSED_DISP if tf_pr > b_pr else disp))

            # 3) Threshold-based Clustered McNemar test (evaluated at per-horizon 80%-recall thresholds)
            tfc = sub[f"y_pred_{PROPOSED_MODEL}"].to_numpy() == y_stack
            bc = sub[f"y_pred_{m}"].to_numpy() == y_stack
            n10 = int(np.sum(tfc & ~bc))
            n01 = int(np.sum(~tfc & bc))
            try:
                p_m = float(clustered_mcnemar_pvalue(tfc, bc, patient_ids=pids_stack))
                mcn_winner = "Tie" if (p_m >= 0.05 or n10 == n01) else (PROPOSED_DISP if n10 > n01 else disp)
            except Exception:
                p_m = np.nan
                mcn_winner = "NaN"

            rec[f"{PROPOSED_DISP} vs {disp}"] = ""
            rec[f"  Threshold-free ROC-AUC p-value ({disp})"] = f"{p_roc:.3e}" if not pd.isna(p_roc) else "NaN"
            rec[f"  ROC-AUC winner ({disp})"] = roc_winner
            rec[f"  Threshold-free PR-AUC p-value ({disp})"] = f"{p_pr:.3e}" if not pd.isna(p_pr) else "NaN"
            rec[f"  PR-AUC winner ({disp})"] = pr_winner
            rec[f"  Discordant pairs ({PROPOSED_DISP} correct) ({disp})"] = n10
            rec[f"  Discordant pairs ({disp} correct) ({disp})"] = n01
            rec[f"  McNemar p-value ({disp})"] = f"{p_m:.3e}" if not pd.isna(p_m) else "NaN"
            rec[f"  McNemar winner ({disp})"] = mcn_winner

        for metric_prefix, arrow in METRIC_MAPPING:
            head_lbl = f"{metric_prefix} ({arrow})"
            raw_estimates = {}
            string_outputs = {}
            rec[head_lbl] = ""
            for m in ALL_MODELS:
                pe = model_points[m][metric_prefix]
                b_vec = model_boots[m][metric_prefix]
                if b_vec.size > 0:
                    low, high = (float(x) for x in np.percentile(b_vec, [2.5, 97.5]))
                    text = f"{pe:.2f} ({low:.2f}-{high:.2f})"
                else:
                    low, high, text = pe, pe, f"{pe:.2f} (NaN-NaN)"
                raw_estimates[m] = pe
                string_outputs[m] = text
                payload[f"{w}_{m}_{metric_prefix}_pe"] = pe
                payload[f"{w}_{m}_{metric_prefix}_low"] = low
                payload[f"{w}_{m}_{metric_prefix}_high"] = high

            best_model = min(raw_estimates, key=raw_estimates.get) if metric_prefix == "ECE" else max(raw_estimates, key=raw_estimates.get)
            for m in ALL_MODELS:
                disp = MODEL_DISPLAY_MAP[m]
                text = string_outputs[m]
                if m == best_model:
                    text = f"**{text}**"
                rec[f"  {head_lbl} ({disp})"] = text

        sub_store = sub[["time_step", "horizon", "y_true", *[f"y_prob_{m}" for m in ALL_MODELS]]].copy()
        results[w] = rec
        cache[w] = (sub_store, ts_by_horizon_json)
        payload.update({
            f"{w}_is_empty": np.array(False),
            f"{w}_time_step": sub_store.time_step.values,
            f"{w}_horizon": sub_store.horizon.values,
            f"{w}_y_true": sub_store.y_true.values,
            f"{w}_thresholds_json": np.array(json.dumps(ts_by_horizon_json, sort_keys=True)),
        })
        payload.update({f"{w}_y_prob_{m}": sub_store[f"y_prob_{m}"].values for m in ALL_MODELS})

    if not results:
        return None, None

    df_main, df_detailed, winners = fdr_and_reports(results, outdir, split, hs)

    for w_name in CLINICAL_WINDOWS:
        for m in CONTENDER_MODELS:
            disp = MODEL_DISPLAY_MAP[m]
            for tkey in ("ROC-AUC", "PR-AUC", "McNemar"):
                payload[f"{w_name}_{disp}_{tkey}_winner"] = winners.get(w_name, {}).get(disp, {}).get(tkey, {}).get("winner", "Tie")

    payload["__schema_version__"] = np.array(CACHE_SCHEMA_VERSION)
    payload["__param_hash__"] = np.array(phash)
    np.savez_compressed(cp, **payload)
    wp.write_text(json.dumps(winners, indent=2, sort_keys=True, default=str))
    print(f"[CACHE SAVED] {cp.name}")

    with pd.option_context("display.max_colwidth", None, "display.max_rows", None):
        print(f"\n>>> MAIN SUMMARY REPORT: SPLIT [{split.upper()}] | HORIZON: {hs.upper()} <<<")
        print(df_main.to_markdown())

    return cache, df_main


# ================================================================================
# Visualization functions
# ================================================================================

def _nice_pr_axis(max_val: float) -> tuple[tuple[float, float], np.ndarray]:
    """Choose an unclipped PR-AUC y-limit and clean tick step across both rare and common endpoints."""
    for cap, step in ((0.08, 0.02), (0.15, 0.05), (0.25, 0.05), (0.35, 0.10), (0.50, 0.10), (0.70, 0.10), (1.02, 0.20)):
        if max_val <= cap * 0.96:
            top_tick = 1.0 if cap > 1.0 else cap
            return (0.0, cap), np.round(np.arange(0.0, top_tick + step * 0.1, step), 2)
    return (0.0, 1.02), np.arange(0.0, 1.05, 0.2)


def render_bars(task, horizon, outdir, hs, csv_summaries_in_memory=None):
    """Render grouped bar charts (rows 1-4) + threshold-free statistical test matrix rows (ROC-AUC & PR-AUC)."""
    metrics_to_plot = [("ROC-AUC", "↑"), ("PR-AUC", "↑"), ("Sensitivity", "↑"), ("Specificity", "↑")]

    csv_summary_by_split = csv_summaries_in_memory or {}
    npz_data_by_split = {}
    for split in SPLIT_TYPES:
        phash = compute_param_hash(task, split, horizon)
        cp = cache_path(task, split, hs, phash)
        if cp.exists():
            with np.load(cp, allow_pickle=True) as z:
                npz_data_by_split[split] = {k: z[k] for k in z.files}
        if split not in csv_summary_by_split:
            main_csv_path = outdir / f"{split}_{hs}_main_summary_report.csv"
            if main_csv_path.exists():
                df_csv = pd.read_csv(main_csv_path, index_col="Evaluation metric")
                csv_summary_by_split[split] = df_csv.fillna("")

    active_splits = [s for s in SPLIT_TYPES if s in npz_data_by_split or s in csv_summary_by_split] or list(SPLIT_TYPES)
    n_windows = len(CLINICAL_WINDOWS)
    n_splits = len(active_splits)
    n_metrics = len(metrics_to_plot)
    total_rows = n_metrics + 2
    row_height_ratios = [1.0, 1.0, 1.0, 1.0, 0.8, 0.8]

    window_keys = list(CLINICAL_WINDOWS.keys())

    # Compute dynamic, unclipped ROC-AUC floor and PR-AUC ceiling across all splits/windows
    roc_lows, pr_highs = [], []
    for s in active_splits:
        cd = npz_data_by_split.get(s, {})
        for w in window_keys:
            for m in MODEL_PLOT_ORDER:
                if f"{w}_{m}_ROC-AUC_low" in cd:
                    roc_lows.append(float(cd[f"{w}_{m}_ROC-AUC_low"]))
                if f"{w}_{m}_PR-AUC_high" in cd:
                    pr_highs.append(float(cd[f"{w}_{m}_PR-AUC_high"]))
    roc_floor = min(0.4, max(0.0, np.floor(min(roc_lows) * 10.0) / 10.0)) if roc_lows else 0.4
    pr_ylim, pr_yticks = _nice_pr_axis(max(pr_highs) if pr_highs else 0.30)

    metric_ylim_map = {
        "ROC-AUC": (roc_floor, 1.02),
        "PR-AUC": pr_ylim,
        "Sensitivity": (0.0, 1.02),
        "Specificity": (0.0, 1.02),
    }
    metric_yticks_map = {
        "ROC-AUC": np.round(np.arange(roc_floor, 1.05, 0.1), 2),
        "PR-AUC": pr_yticks,
        "Sensitivity": np.arange(0.0, 1.1, 0.2),
        "Specificity": np.arange(0.0, 1.1, 0.2),
    }

    fig, axes = plt.subplots(
        total_rows, n_splits, figsize=(8.5 * n_splits, 25),
        gridspec_kw={"height_ratios": row_height_ratios}, squeeze=False,
    )

    for m_idx, (metric_prefix, arrow) in enumerate(metrics_to_plot):
        metric_label = f"{metric_prefix} ({arrow})"
        ylim = metric_ylim_map[metric_prefix]
        yticks = metric_yticks_map[metric_prefix]

        for s_idx, split in enumerate(active_splits):
            ax = axes[m_idx, s_idx]
            ax.set_axisbelow(True)
            ax.minorticks_off()

            split_title = split.replace("_", " ").capitalize()
            x_group_centers = np.arange(n_windows)
            n_models = len(MODEL_PLOT_ORDER)
            bar_width = min(0.20, 0.8 / max(n_models, 1))
            offsets = (np.arange(n_models) - (n_models - 1) / 2.0) * bar_width

            cache_dict = npz_data_by_split.get(split, {})

            if metric_prefix == "ROC-AUC":
                for w_i, _ in enumerate(window_keys):
                    lbl = "Random level" if (m_idx == 0 and s_idx == 0 and w_i == 0) else ""
                    ax.hlines(
                        y=0.50, xmin=w_i - 0.42, xmax=w_i + 0.42,
                        colors="black", linestyles="--", linewidth=1.8, zorder=3,
                        label=lbl,
                    )
            elif metric_prefix == "PR-AUC":
                for w_i, window_name in enumerate(window_keys):
                    y_key = f"{window_name}_y_true"
                    h_key = f"{window_name}_horizon"
                    if y_key in cache_dict:
                        y_arr = cache_dict[y_key]
                        if len(y_arr) > 0:
                            if h_key in cache_dict and len(np.unique(cache_dict[h_key])) > 1:
                                h_arr = cache_dict[h_key]
                                prev = float(np.mean([np.mean(y_arr[h_arr == u] == 1) for u in np.unique(h_arr)]))
                            else:
                                prev = float(np.mean(y_arr == 1))
                            ax.hlines(
                                y=prev, xmin=w_i - 0.42, xmax=w_i + 0.42,
                                colors="black", linestyles="--", linewidth=1.8, zorder=3,
                            )

            for m_model_idx, m in enumerate(MODEL_PLOT_ORDER):
                c_hex = COLORS[m]
                pes, yerr_low, yerr_high = [], [], []

                for window_name in window_keys:
                    pe_key = f"{window_name}_{m}_{metric_prefix}_pe"
                    low_key = f"{window_name}_{m}_{metric_prefix}_low"
                    high_key = f"{window_name}_{m}_{metric_prefix}_high"
                    if pe_key in cache_dict:
                        pe, low, high = float(cache_dict[pe_key]), float(cache_dict[low_key]), float(cache_dict[high_key])
                    else:
                        pe, low, high = np.nan, np.nan, np.nan

                    if np.isnan(pe):
                        pes.append(np.nan); yerr_low.append(0.0); yerr_high.append(0.0)
                    else:
                        pes.append(pe)
                        yerr_low.append(max(0.0, pe - low))
                        yerr_high.append(max(0.0, high - pe))

                ax.bar(
                    x_group_centers + offsets[m_model_idx], pes, width=bar_width,
                    yerr=[yerr_low, yerr_high], capsize=4,
                    error_kw={"elinewidth": 1.8, "capthick": 1.5, "zorder": 3},
                    color=c_hex, edgecolor="black", linewidth=1.0, alpha=0.85, zorder=2,
                    label=MODEL_FULLNAME_MAP[m] if (m_idx == 0 and s_idx == 0) else "",
                )

            ax.set_xticks(x_group_centers)
            ax.set_xticklabels(CONCISE_WINDOW_LABELS, fontsize=20, fontweight="bold")
            ax.set_ylim(ylim)
            ax.set_yticks(yticks)
            ax.grid(True, which="major", linestyle=":", color="#eeeeee", linewidth=1.0, alpha=0.9, axis="y", zorder=0)
            ax.grid(False, which="minor")

            if m_idx == 0:
                ax.set_title(split_title, fontsize=26, fontweight="bold", pad=18)
            if s_idx == 0:
                ax.set_ylabel(metric_label, fontsize=26, fontweight="bold", labelpad=22)
                ax.tick_params(axis="y", labelsize=20, length=6, width=1.0, left=True, labelleft=True, which="both")
                ax.tick_params(axis="x", length=6, width=1.0)
            else:
                ax.tick_params(axis="y", left=False, labelleft=False, which="both")
                ax.tick_params(axis="x", length=6, width=1.0)

    test_rows_info = [("ROC-AUC test", "ROC-AUC"), ("PR-AUC test", "PR-AUC")]
    sub_baselines = [MODEL_DISPLAY_MAP[m] for m in CONTENDER_MODELS]
    y_positions = [0.0] if len(sub_baselines) <= 1 else list(np.linspace(0.65, -0.65, len(sub_baselines)))

    for test_idx, (test_row_label, test_type_key) in enumerate(test_rows_info):
        ax_row = n_metrics + test_idx
        for s_idx, split in enumerate(active_splits):
            ax = axes[ax_row, s_idx]
            ax.set_axisbelow(True)
            ax.minorticks_off()
            ax.set_xlim([-0.5, n_windows - 0.5])
            ax.set_ylim([-1.1, 1.1])
            ax.set_yticks(y_positions)
            ax.set_yticklabels([""] * len(y_positions))

            if s_idx == 0:
                ax.set_ylabel(test_row_label, fontsize=26, fontweight="bold", labelpad=22)
                ax.tick_params(axis="y", length=6, width=1.0, left=True, labelleft=False, which="both")
                ax.tick_params(axis="x", length=6, width=1.0)
                for b_idx, base_disp in enumerate(sub_baselines):
                    ax.text(
                        -0.03, y_positions[b_idx], f"vs\n{base_disp}",
                        transform=ax.get_yaxis_transform(), ha="right", va="center",
                        fontsize=20, color="#222222",
                    )
            else:
                ax.tick_params(axis="y", left=False, labelleft=False, which="both")
                ax.tick_params(axis="x", length=6, width=1.0)

            ax.set_xticks(np.arange(n_windows))
            ax.set_xticklabels(CONCISE_WINDOW_LABELS, fontsize=20, fontweight="bold")
            for y_pos in y_positions:
                ax.axhline(y_pos, color="#eeeeee", linestyle=":", linewidth=1.0, zorder=0)

            df_summary = csv_summary_by_split.get(split, pd.DataFrame())
            cache_dict = npz_data_by_split.get(split, {})

            for w_idx, window_name in enumerate(window_keys):
                for b_idx, base_disp in enumerate(sub_baselines):
                    winner_token = None
                    row_key_unique = f"  {PROPOSED_DISP} vs {base_disp} ({test_type_key})"
                    col_match = None
                    if not df_summary.empty:
                        w_prefix = window_name.split("\n")[0].strip()
                        for col in df_summary.columns:
                            if col.strip().startswith(w_prefix):
                                col_match = col
                                break
                    if not df_summary.empty and row_key_unique in df_summary.index and col_match:
                        cell_val = str(df_summary.loc[row_key_unique, col_match]).strip()
                        if "(" in cell_val:
                            winner_token = cell_val.split("(")[0].strip()
                        elif cell_val in ALL_ABBREVS | {"Tie"}:
                            winner_token = cell_val
                    if not winner_token:
                        cache_key = f"{window_name}_{base_disp}_{test_type_key}_winner"
                        winner_token = str(cache_dict.get(cache_key, "Tie"))

                    if winner_token == PROPOSED_DISP:
                        face_c, edge_c, alpha_v = "#d62728", "#8b0000", 0.85
                    elif winner_token in sub_baselines:
                        full_name = BASELINE_FULL_NAME_MAP.get(winner_token, winner_token)
                        face_c, edge_c, alpha_v = COLORS.get(full_name, "#f0f0f0"), "#333333", 0.85
                    else:
                        face_c, edge_c, alpha_v = "#f0f0f0", "#cccccc", 0.95

                    ax.add_patch(plt.Rectangle(
                        (w_idx - 0.38, y_positions[b_idx] - 0.25), 0.76, 0.50,
                        facecolor=face_c, edgecolor=edge_c, linewidth=1.0, alpha=alpha_v, zorder=2,
                    ))
                    ax.text(w_idx, y_positions[b_idx], winner_token, ha="center", va="center",
                            fontsize=16, fontweight="bold", color="#000000", zorder=3)

            for k in range(n_windows - 1):
                ax.axvline(k + 0.5, color="#dddddd", linestyle="--", linewidth=1.0, zorder=1)

    fig.align_ylabels(axes[:, 0])
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.014),
            ncol=len(handles), fontsize=22, frameon=True, framealpha=0.9, borderpad=0.25,
        )

    acronym_explanation = "Clinical phases:  POP = Perioperative  |  OPT = Opportunistic  |  MTN = Maintenance  |  LT = Long-term  |  VLT = Very long-term"
    fig.text(
        0.5, 0.973, acronym_explanation, ha="center", va="center", fontsize=22, color="black",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#f8f8f8", edgecolor="#cccccc", alpha=0.9),
    )

    plt.subplots_adjust(top=0.935, hspace=0.25, wspace=0.06)
    fig.savefig(outdir / f"matrix_bar_performance_comparison_{hs}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def render_curves(cache, outdir, hs):
    """Render ROC, PR, and Decision Curve Analysis (DCA) grids (macro-averaged across horizons when combined)."""
    active_splits = [s for s in SPLIT_TYPES if s in cache and cache[s]] or list(SPLIT_TYPES)
    n_windows = len(CLINICAL_WINDOWS)
    n_splits = len(active_splits)

    fig_roc, axes_roc = plt.subplots(n_windows, n_splits, figsize=(7.5 * n_splits, 4.5 * n_windows), squeeze=False)
    fig_pr, axes_pr = plt.subplots(n_windows, n_splits, figsize=(7.5 * n_splits, 4.5 * n_windows), squeeze=False)
    fig_dca, axes_dca = plt.subplots(n_windows, n_splits, figsize=(7.5 * n_splits, 4.5 * n_windows), squeeze=False)

    fpr_grid = np.linspace(0.0, 1.0, 201)
    rec_grid = np.linspace(0.0, 1.0, 201)

    for w_idx, window_name in enumerate(CLINICAL_WINDOWS):
        prevalences = []
        for split in active_splits:
            if split in cache and window_name in cache[split]:
                sub_df, _ = cache[split][window_name]
                if len(sub_df) > 0:
                    uniq_h = np.unique(sub_df.horizon.values)
                    p_w = float(np.mean([np.mean(sub_df.y_true.values[sub_df.horizon.values == u] == 1) for u in uniq_h]))
                    prevalences.append(p_w)

        max_prev = max(prevalences) if prevalences else 0.01
        target_top = max(max_prev * 1.15, 0.005)
        if target_top <= 0.012:
            step = 0.003
        elif target_top <= 0.03:
            step = 0.01
        elif target_top <= 0.08:
            step = 0.02
        elif target_top <= 0.20:
            step = 0.05
        else:
            step = 0.10
        upper_ylim = float(np.ceil(target_top / step) * step)
        dca_yticks = np.round(np.arange(0.0, upper_ylim + step * 0.1, step), 4)
        lower_ylim = -0.15 * upper_ylim
        dca_min_t = max(0.001, round(0.1 * max_prev, 4))
        dca_max_t = min(0.50, max(0.10, round(4.0 * max_prev, 2)))
        dca_thresh = np.linspace(dca_min_t, dca_max_t, 60)
        dca_fmt = "%.3f" if step < 0.01 else "%.2f"

        for s_idx, split in enumerate(active_splits):
            ax_roc, ax_pr, ax_dca = axes_roc[w_idx, s_idx], axes_pr[w_idx, s_idx], axes_dca[w_idx, s_idx]
            split_title = split.replace("_", " ").capitalize()

            for ax in (ax_roc, ax_pr, ax_dca):
                ax.set_axisbelow(True)
                ax.minorticks_off()

            if split not in cache or window_name not in cache[split]:
                for ax, lbl in [(ax_roc, "Sensitivity"), (ax_pr, "Precision"), (ax_dca, "Net benefit")]:
                    ax.text(0.5, 0.5, "Not applicable for this horizon", fontsize=20, color="darkred",
                            ha="center", va="center", weight="bold")
                    ax.set_xlim([0.0, 1.0]); ax.set_ylim([0.0, 1.0]); ax.grid(False)
                    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
                    if w_idx == 0:
                        ax.set_title(split_title, fontsize=26, fontweight="bold", pad=18)
                    if s_idx == 0:
                        ax.set_ylabel(f"{window_name}\n\n{lbl}", fontsize=26, fontweight="bold", labelpad=16)
                continue

            sub_df, _ = cache[split][window_name]
            uniq_hs = [u for u in sorted(np.unique(sub_df.horizon.values)) if np.unique(sub_df.y_true.values[sub_df.horizon.values == u]).size == 2]
            is_multi_h = len(uniq_hs) > 1
            prevalence = float(np.mean([np.mean(sub_df.y_true.values[sub_df.horizon.values == u] == 1) for u in uniq_hs])) if uniq_hs else 0.0

            # --- ROC Curves ---
            for m in MODEL_PLOT_ORDER:
                c_hex = COLORS[m]
                label = MODEL_FULLNAME_MAP[m] if (w_idx == 0 and s_idx == 0) else ""
                if is_multi_h:
                    tprs = []
                    for u in uniq_hs:
                        msk = sub_df.horizon.values == u
                        fpr_u, tpr_u, _ = roc_curve(sub_df.y_true.values[msk], sub_df[f"y_prob_{m}"].values[msk])
                        tpr_interp = np.interp(fpr_grid, fpr_u, tpr_u)
                        tpr_interp[0] = 0.0
                        tprs.append(tpr_interp)
                    ax_roc.plot(fpr_grid, np.mean(tprs, axis=0), label=label, color=c_hex, lw=3.5, zorder=2)
                elif uniq_hs:
                    msk = sub_df.horizon.values == uniq_hs[0]
                    fpr, tpr, _ = roc_curve(sub_df.y_true.values[msk], sub_df[f"y_prob_{m}"].values[msk])
                    ax_roc.plot(fpr, tpr, label=label, color=c_hex, lw=3.5, zorder=2)
            ax_roc.plot(
                [0, 1], [0, 1], linestyle="--", color="gray", alpha=0.5, lw=2.0,
                label="Random" if (w_idx == 0 and s_idx == 0) else "", zorder=1
            )
            ax_roc.set_xlim([0.0, 1.0])
            ax_roc.set_ylim([0.0, 1.05])
            ax_roc.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
            ax_roc.grid(True, which="major", linestyle=":", color="#eeeeee", linewidth=1.0, alpha=0.9, axis="y", zorder=0)
            ax_roc.grid(False, which="minor")

            if w_idx == 0:
                ax_roc.set_title(split_title, fontsize=26, fontweight="bold", pad=18)
            if s_idx == 0:
                ax_roc.set_ylabel(f"{window_name}\n\nSensitivity", fontsize=26, fontweight="bold", labelpad=16)
                ax_roc.tick_params(axis="y", labelsize=20, length=6, width=1.0, left=True, labelleft=True, which="both")
                ax_roc.tick_params(axis="x", labelsize=20, length=6, width=1.0)
            else:
                ax_roc.tick_params(axis="y", left=False, labelleft=False, which="both")
                ax_roc.tick_params(axis="x", labelsize=20, length=6, width=1.0)
            if w_idx == n_windows - 1:
                ax_roc.set_xlabel("1.0 - Specificity", fontsize=24, fontweight="bold", labelpad=14)

            # --- PR Curves ---
            for m in MODEL_PLOT_ORDER:
                c_hex = COLORS[m]
                label = MODEL_FULLNAME_MAP[m] if (w_idx == 0 and s_idx == 0) else ""
                if is_multi_h:
                    precs = []
                    for u in uniq_hs:
                        msk = sub_df.horizon.values == u
                        p_u, r_u, _ = precision_recall_curve(sub_df.y_true.values[msk], sub_df[f"y_prob_{m}"].values[msk])
                        # Step-exact right-continuous precision envelope on ascending recall grid
                        r_asc, p_asc = r_u[::-1], np.maximum.accumulate(p_u)[::-1]
                        idx_step = np.clip(np.searchsorted(r_asc, rec_grid, side="left"), 0, len(p_asc) - 1)
                        precs.append(p_asc[idx_step])
                    ax_pr.plot(rec_grid, np.mean(precs, axis=0), label=label, color=c_hex, lw=3.5, zorder=2)
                elif uniq_hs:
                    msk = sub_df.horizon.values == uniq_hs[0]
                    prec_arr, rec_arr, _ = precision_recall_curve(sub_df.y_true.values[msk], sub_df[f"y_prob_{m}"].values[msk])
                    ax_pr.plot(rec_arr, prec_arr, label=label, color=c_hex, lw=3.5, drawstyle="steps-pre", zorder=2)
            ax_pr.axhline(
                prevalence, linestyle="--", color="gray", alpha=0.6, lw=2.0,
                label="Random" if (w_idx == 0 and s_idx == 0) else "", zorder=1
            )
            ax_pr.set_xlim([0.0, 1.0])
            ax_pr.set_ylim([0.0, 1.05])
            ax_pr.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
            ax_pr.grid(True, which="major", linestyle=":", color="#eeeeee", linewidth=1.0, alpha=0.9, axis="y", zorder=0)
            ax_pr.grid(False, which="minor")

            if w_idx == 0:
                ax_pr.set_title(split_title, fontsize=26, fontweight="bold", pad=18)
            if s_idx == 0:
                ax_pr.set_ylabel(f"{window_name}\n\nPrecision", fontsize=26, fontweight="bold", labelpad=16)
                ax_pr.tick_params(axis="y", labelsize=20, length=6, width=1.0, left=True, labelleft=True, which="both")
                ax_pr.tick_params(axis="x", labelsize=20, length=6, width=1.0)
            else:
                ax_pr.tick_params(axis="y", left=False, labelleft=False, which="both")
                ax_pr.tick_params(axis="x", labelsize=20, length=6, width=1.0)
            if w_idx == n_windows - 1:
                ax_pr.set_xlabel("Recall", fontsize=24, fontweight="bold", labelpad=14)

            # --- DCA Curves ---
            ax_dca.plot(dca_thresh, np.zeros_like(dca_thresh), color="#1a1a1a", linestyle="--", lw=3.5,
                        label="Treat none" if (w_idx == 0 and s_idx == 0) else "", zorder=1)
            treat_all_nb = np.mean(
                [
                    (np.mean(sub_df.y_true.values[sub_df.horizon.values == u] == 1))
                    - (1.0 - np.mean(sub_df.y_true.values[sub_df.horizon.values == u] == 1)) * (dca_thresh / (1.0 - dca_thresh))
                    for u in uniq_hs
                ],
                axis=0,
            ) if uniq_hs else np.zeros_like(dca_thresh)
            ax_dca.plot(dca_thresh, treat_all_nb, color="#a0a0a0", linestyle="--", lw=3.5,
                        label="Treat all" if (w_idx == 0 and s_idx == 0) else "", zorder=1)

            for m in MODEL_PLOT_ORDER:
                c_hex = COLORS[m]
                label = MODEL_FULLNAME_MAP[m] if (w_idx == 0 and s_idx == 0) else ""
                nbs_h = []
                for u in uniq_hs:
                    msk = sub_df.horizon.values == u
                    y_u = sub_df.y_true.values[msk]
                    p_u = sub_df[f"y_prob_{m}"].values[msk]
                    n_u = float(len(y_u))
                    pred_mat = p_u[:, None] >= dca_thresh[None, :]
                    tp_vec = np.sum(pred_mat & (y_u[:, None] == 1), axis=0) / n_u
                    fp_vec = np.sum(pred_mat & (y_u[:, None] == 0), axis=0) / n_u
                    nbs_h.append(tp_vec - fp_vec * (dca_thresh / (1.0 - dca_thresh)))
                if nbs_h:
                    ax_dca.plot(dca_thresh, np.mean(nbs_h, axis=0), label=label, color=c_hex, lw=3.5, zorder=2)

            ax_dca.set_xlim([0.0, dca_max_t])
            ax_dca.set_ylim([lower_ylim, upper_ylim])
            ax_dca.set_yticks(dca_yticks)
            ax_dca.yaxis.set_major_formatter(ticker.FormatStrFormatter(dca_fmt))
            ax_dca.grid(True, which="major", linestyle=":", color="#eeeeee", linewidth=1.0, alpha=0.9, axis="y", zorder=0)
            ax_dca.grid(False, which="minor")

            if w_idx == 0:
                ax_dca.set_title(split_title, fontsize=26, fontweight="bold", pad=18)
            if s_idx == 0:
                ax_dca.set_ylabel(f"{window_name}\n\nNet benefit", fontsize=26, fontweight="bold", labelpad=16)
                ax_dca.tick_params(axis="y", labelsize=20, length=6, width=1.0, left=True, labelleft=True, which="both")
                ax_dca.tick_params(axis="x", labelsize=20, length=6, width=1.0)
            else:
                ax_dca.tick_params(axis="y", left=False, labelleft=False, which="both")
                ax_dca.tick_params(axis="x", labelsize=20, length=6, width=1.0)
            if w_idx == n_windows - 1:
                ax_dca.set_xlabel("Threshold probability", fontsize=24, fontweight="bold", labelpad=14)

    for fig_obj, ax_grid in [(fig_roc, axes_roc), (fig_pr, axes_pr), (fig_dca, axes_dca)]:
        handles, labels = ax_grid[0, 0].get_legend_handles_labels()
        if handles:
            bbox_x = 0.5 if fig_obj is fig_dca else 0.544
            fig_obj.legend(
                handles, labels, loc="upper center", bbox_to_anchor=(bbox_x, 1.04),
                ncol=len(handles), fontsize=24, frameon=True, framealpha=0.9,
            )

    fig_roc.align_ylabels(axes_roc[:, 0])
    fig_pr.align_ylabels(axes_pr[:, 0])
    fig_dca.align_ylabels(axes_dca[:, 0])

    fig_roc.tight_layout()
    fig_roc.savefig(outdir / f"matrix_roc_comparison_curves_{hs}.png", dpi=200, bbox_inches="tight")
    plt.close(fig_roc)

    fig_pr.tight_layout()
    fig_pr.savefig(outdir / f"matrix_pr_comparison_curves_{hs}.png", dpi=200, bbox_inches="tight")
    plt.close(fig_pr)

    fig_dca.tight_layout()
    fig_dca.savefig(outdir / f"matrix_dca_comparison_curves_{hs}.png", dpi=200, bbox_inches="tight")
    plt.close(fig_dca)
    
    
# ================================================================================
# Pipeline orchestration
# ================================================================================

def pipeline(task, horizon):
    """Execute evaluation and visualization pipeline for a given task and horizon."""
    hs = horizon_name(horizon)
    outdir = ANALYSIS_SUBDIR / task / hs
    outdir.mkdir(parents=True, exist_ok=True)
    cache = {}
    csv_summaries = {}
    prevalence_records = []

    for split in SPLIT_TYPES:
        wc, df_main = process(task, split, horizon, outdir)
        if wc is not None:
            cache[split] = wc
            csv_summaries[split] = df_main

            # Extract test set prevalence for each clinical window (period)
            for window_name, (sub_df, _) in wc.items():
                y_true = sub_df["y_true"].to_numpy()
                uniq_hs = sorted(int(u) for u in np.unique(sub_df["horizon"])) if "horizon" in sub_df.columns else []
                if len(uniq_hs) > 1:
                    prevs_h = [
                        float(np.mean(y_true[sub_df["horizon"].to_numpy() == u] == 1))
                        for u in uniq_hs
                        if np.sum(sub_df["horizon"].to_numpy() == u) > 0
                    ]
                    prev = float(np.mean(prevs_h)) if prevs_h else np.nan
                    n_total = len(y_true)
                    n_pos = int(np.sum(y_true == 1))
                else:
                    n_total = len(y_true)
                    n_pos = int(np.sum(y_true == 1))
                    prev = float(n_pos / n_total) if n_total > 0 else np.nan

                # Format period to single line for cleaner display
                clean_period = window_name.replace("\n", " ")

                prevalence_records.append({
                    "Task": task,
                    "Horizon": hs,
                    "Split": split,
                    "Period": clean_period,
                    "N_Pos": n_pos,
                    "N_Total": n_total,
                    "Prevalence": prev,
                    "Prevalence (%)": f"{prev * 100:.2f}%" if not np.isnan(prev) else "NaN",
                })

    if cache:
        render_curves(cache, outdir, hs)
        render_bars(task, horizon, outdir, hs, csv_summaries)

    return prevalence_records


if __name__ == "__main__":

    # Run the evaluation and visualization pipeline for all tasks and horizons
    all_prevalences = []
    for task in TASKS:
        for horizon in ["combined", *_horizons_for_task(task)]:
            records = pipeline(task, horizon)
            if records:
                all_prevalences.extend(records)

    # Print summary at the very end
    if all_prevalences:
        prev_df = pd.DataFrame(all_prevalences)
        print("\n" + "=" * 80)
        print(">>> TEST SET PREVALENCES (DASHED BASELINE IN PR-AUC PLOTS) <<<")
        print("=" * 80)
        with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 1000):
            print(prev_df.to_string(index=False))
        print("=" * 80 + "\n")
