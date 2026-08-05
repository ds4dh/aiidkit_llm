import argparse
import hashlib
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from datasets import load_from_disk
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests

from scripts.script_utils import (
    calibrate_array_pair,
    calibrate_dataframe_pair,
    paired_bootstrap_pr_auc_pvalue,
)


# ==========================================
# CLI Argument Parsing & Global Configurations
# ==========================================

parser = argparse.ArgumentParser(description="Evaluation pipeline")
parser.add_argument(
    "--threshold_mode",
    choices=["global", "window_specific"],
    default="window_specific",
)
parser.add_argument(
    "--target_recall",
    type=int,
    default=80,
)
parser.add_argument(
    "--workers",
    type=int,
    default=14,
    help="Number of threads used for bootstrap resampling (does not affect results).",
)
args = parser.parse_args()

THRESHOLD_MODE = args.threshold_mode
TARGET_RECALL_INPUT = args.target_recall
TARGET_RECALL = TARGET_RECALL_INPUT / 100.0
THRESHOLD_TUNING_SET = "validation"
MAX_WORKERS = max(1, args.workers)

USE_CALIBRATED_PROBS = {
    "Transformer": True,
    "logistic_regression": True,
    "random_forest": True,
    "xgboost": True,
}

THRESHOLD_SUBFOLDER = f"rec-{TARGET_RECALL_INPUT}"
CALIB_SUBFOLDER = f"calib-{int(USE_CALIBRATED_PROBS['Transformer'])}"

BASE_DATA_PATH = Path(
    "/home/shares/ds4dh/aiidkit_project/data_new/processed/v3.6_old/teav"
)
RESULTS_DIR = Path("results_final")
ANALYSIS_DIR = RESULTS_DIR / "analysis" / "comparison" / "lt-1-3_vlt-3-10" / THRESHOLD_SUBFOLDER
ANALYSIS_SUBDIR = ANALYSIS_DIR / THRESHOLD_MODE / CALIB_SUBFOLDER
CACHE_DIR = ANALYSIS_DIR / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

CLASSIC_MODELS = ["logistic_regression", "random_forest", "xgboost"]
ALL_MODELS = ["Transformer", *CLASSIC_MODELS]
TASKS = ["infection_bacteria", "infection_virus"]
SPLIT_TYPES = ["random_split", "temporal_split", "center_split"]
TARGET_HORIZONS = [30, 60, 90]
N_BOOTSTRAP = 1000

CLINICAL_WINDOWS = {
    "Perioperative\n(0-30 days)": (0, 30),
    "Opportunistic\n(31-180 days)": (31, 180),
    "Maintenance\n(181-360 days)": (181, 360),
    "Long-term\n(361-1080 days)": (361, 1080),
    "Very long-term\n(1081-3600 days)": (1081, 3600),
}

CONCISE_WINDOW_LABELS = ["POP", "OPT", "MTN", "LT", "VLT"]

MODEL_DISPLAY_MAP = {
    "Transformer": "TF",
    "logistic_regression": "LR",
    "random_forest": "RF",
    "xgboost": "XGB",
}
MODEL_FULLNAME_MAP = {
    "Transformer": "t-EAV-Transformer",
    "logistic_regression": "Logistic regression",
    "random_forest": "Random forest",
    "xgboost": "XGBoost",
}
BASELINE_FULL_NAME_MAP = {"LR": "logistic_regression", "RF": "random_forest", "XGB": "xgboost"}

MODEL_PLOT_ORDER = ["logistic_regression", "random_forest", "xgboost", "Transformer"]
COLORS_BY_PLOT_ORDER = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
COLORS = {m: c for m, c in zip(MODEL_PLOT_ORDER, COLORS_BY_PLOT_ORDER)}

METRIC_MAPPING = [
    ("ROC-AUC", "↑"), ("PR-AUC", "↑"), ("ECE", "↓"),
    ("Sensitivity", "↑"), ("Precision", "↑"), ("Specificity", "↑"),
]
MAIN_METRIC_MAPPING = [
    ("ROC-AUC", "↑"), ("PR-AUC", "↑"), ("ECE", "↓"),
    ("Sensitivity", "↑"), ("Specificity", "↑"),
]

CACHE_SCHEMA_VERSION = 3
PARAM_SIGNATURE = {
    "schema": CACHE_SCHEMA_VERSION,
    "threshold_mode": THRESHOLD_MODE,
    "target_recall": TARGET_RECALL,
    "tuning_set": THRESHOLD_TUNING_SET,
    "calibration": USE_CALIBRATED_PROBS,
    "n_bootstrap": N_BOOTSTRAP,
    "windows": CLINICAL_WINDOWS,
    "models": ALL_MODELS,
}
PARAM_HASH = hashlib.sha256(
    json.dumps(PARAM_SIGNATURE, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()[:12]


# ==========================================
# Utility & Path Functions
# ==========================================

def horizon_name(h):
    """Return a formatted string representation for the prediction horizon."""
    return f"horizon_{h:04d}d" if isinstance(h, int) else "horizon_combined"


def cache_path(task, split, hs):
    """Generate path for numpy cache files."""
    return CACHE_DIR / f"plot_cache_v{CACHE_SCHEMA_VERSION}_{task}_{split}_{hs}_{PARAM_HASH}.npz"


def winners_path(task, split, hs):
    """Generate path for JSON files storing statistical test winners (kept for backward compatibility)."""
    return CACHE_DIR / f"winners_v{CACHE_SCHEMA_VERSION}_{task}_{split}_{hs}_{PARAM_HASH}.json"


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


# ==========================================
# Data Loading Functions
# ==========================================

def load_transformer(task, split, dataset_split, horizon):
    """Load and process Transformer prediction probabilities and true labels."""
    if horizon == "combined":
        x = [load_transformer(task, split, dataset_split, h) for h in TARGET_HORIZONS]
        return (
            pd.concat([d for d in x if not d.empty], ignore_index=True)
            if any(not d.empty for d in x) else pd.DataFrame()
        )

    cp = raw_cache("raw_tf", task, split, dataset_split, horizon)
    if cp.exists():
        return pd.read_parquet(cp)

    fname = "validation_probs.npz" if dataset_split == "validation" else "test_probs.npz"
    root = RESULTS_DIR / "transformer" / split / "e00-a15-v60" / "finetuning" / task
    if not root.exists():
        return pd.DataFrame()

    found = None
    for d in root.iterdir():
        m = re.search(r"hrz\(([\d-]+)\)", d.name)
        if d.is_dir() and m and horizon in [int(v) for v in m.group(1).split("-")]:
            hs = [int(v) for v in m.group(1).split("-")]
            found = (d, hs.index(horizon), hs)
            break

    if found is None or not (found[0] / fname).exists():
        return pd.DataFrame()

    d, target_idx, hs = found
    z = np.load(d / fname, allow_pickle=True)
    pref = "validation_" if dataset_split == "validation" else "test_"
    days = sorted(int(m.group(1)) for k in z.files if (m := re.match(rf"^{pref}fup_(\d+)_labels$", k)))

    records = []
    for day in days:
        labels = z[f"{pref}fup_{day:04d}_labels"]
        probs = z[f"{pref}fup_{day:04d}_probs"]
        y = labels if labels.ndim == 1 else labels[:, target_idx]
        p = probs if probs.ndim == 1 else probs[:, target_idx]

        dsdir = BASE_DATA_PATH / split / f"fup_{day:04d}"
        if not dsdir.exists():
            dsdir = BASE_DATA_PATH / split / f"fup_{day:04d}d"
        if not dsdir.exists():
            continue

        ds = load_from_disk(str(dsdir))[dataset_split]
        keys = ds["patientkey"]
        cols = [f"label_{task}_{h:04d}d" for h in hs if f"label_{task}_{h:04d}d" in ds.column_names]
        keep = (
            (np.stack([ds[c] for c in cols], 1) != -100).any(1)
            if cols else np.ones(len(keys), bool)
        )
        valid = np.where(keep)[0]

        for i in range(min(len(valid), len(y), len(p))):
            if int(y[i]) != -100:
                records.append(
                    dict(
                        patientkey=keys[valid[i]], time_step=day, horizon=horizon,
                        y_true=int(y[i]), y_prob=float(p[i]),
                    )
                )

    df = pd.DataFrame(records)
    if not df.empty:
        df.to_parquet(cp, index=False)
    return df


def load_classic(model, task, split, dataset_split, horizon):
    """Load and process classic ML model prediction probabilities and labels."""
    if horizon == "combined":
        x = [load_classic(model, task, split, dataset_split, h) for h in TARGET_HORIZONS]
        return (
            pd.concat([d for d in x if not d.empty], ignore_index=True)
            if any(not d.empty for d in x) else pd.DataFrame()
        )

    cp = raw_cache("raw_ml", model, task, split, dataset_split, horizon)
    if cp.exists():
        return pd.read_parquet(cp)

    fname = "val_predictions.npz" if dataset_split == "validation" else "test_predictions.npz"
    root = RESULTS_DIR / "classic_ml" / split / model / task
    if not root.exists():
        return pd.DataFrame()

    d = next(
        (x for x in root.iterdir() if x.is_dir() and f"hrz({horizon:04d})" in x.name),
        None,
    )
    if d is None or not (d / fname).exists():
        return pd.DataFrame()

    z = np.load(d / fname, allow_pickle=True)
    pref = "validation_" if dataset_split == "validation" else "test_"
    days = sorted(int(m.group(1)) for k in z.files if (m := re.match(rf"^{pref}fup_(\d+)_labels$", k)))

    records = []
    for day in days:
        y = z[f"{pref}fup_{day:04d}_labels"].ravel()
        p = z[f"{pref}fup_{day:04d}_probs"].ravel()

        dsdir = BASE_DATA_PATH / split / f"fup_{day:04d}"
        if not dsdir.exists():
            dsdir = BASE_DATA_PATH / split / f"fup_{day:04d}d"
        if not dsdir.exists():
            continue

        ds = load_from_disk(str(dsdir))[dataset_split]
        keys = ds["patientkey"]
        col = f"label_{task}_{horizon:04d}d"
        valid = np.where(
            np.asarray(ds[col]) != -100 if col in ds.column_names else np.ones(len(keys), bool)
        )[0]

        for i in range(min(len(valid), len(y), len(p))):
            if int(y[i]) != -100:
                records.append(dict(
                    patientkey=keys[valid[i]], time_step=day, horizon=horizon,
                    y_true=int(y[i]), y_prob=float(p[i]),
                ))

    df = pd.DataFrame(records)
    if not df.empty:
        df.to_parquet(cp, index=False)
    return df


# ==========================================
# Evaluation & Statistical Metrics
# ==========================================

def threshold(df, lo, hi, label):
    """Compute classification threshold based on target recall."""
    d = df[(df.time_step >= lo) & ((df.time_step + df.horizon) <= hi)]
    if d.empty or d.y_true.nunique() != 2:
        raise RuntimeError(f"Threshold failure for {label}: no valid two-class validation data in {lo}-{hi} days")

    _, recall, thresholds = precision_recall_curve(d.y_true, d.y_prob)
    idx = np.where(recall[:-1] >= TARGET_RECALL)[0]
    if not len(idx):
        raise RuntimeError(f"Threshold failure for {label}: target recall {TARGET_RECALL:.0%} cannot be reached")
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


def _bootstrap_chunk(y, p, t, kind, index_chunk):
    """Compute the metric for a chunk of pre-generated bootstrap index arrays."""
    values = []
    for ix in index_chunk:
        yt = y[ix]
        if len(np.unique(yt)) < 2:
            continue
        values.append(metric(yt, p[ix], t, kind))
    return values


def bootstrap_ci(df, model, t, kind):
    """Compute bootstrap confidence intervals for a given metric."""
    y = df["y_true"].to_numpy()
    p = df[f"y_prob_{model}"].to_numpy()
    point = metric(y, p, t, kind)

    rng = np.random.default_rng(42)
    all_indices = rng.integers(0, len(y), size=(N_BOOTSTRAP, len(y)))
    chunks = np.array_split(all_indices, MAX_WORKERS) if MAX_WORKERS > 1 else [all_indices]

    values = []
    if MAX_WORKERS > 1 and len(chunks) > 1:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            for chunk_result in executor.map(lambda c: _bootstrap_chunk(y, p, t, kind, c), chunks):
                values.extend(chunk_result)
    else:
        values = _bootstrap_chunk(y, p, t, kind, all_indices)

    if not values:
        return point, point, point, f"{point:.2f} (NaN-NaN)"

    low, high = np.percentile(values, [2.5, 97.5])
    return point, float(low), float(high), f"{point:.2f} ({low:.2f}-{high:.2f})"


def aligned(dfs):
    """Align patient cohorts across all model dataframes."""
    keys = ["patientkey", "time_step", "horizon", "y_true"]
    common = dfs["Transformer"][keys].drop_duplicates()
    for m in CLASSIC_MODELS:
        common = common.merge(dfs[m][keys].drop_duplicates(), on=keys, how="inner")

    if common.empty:
        raise RuntimeError("No common samples across all models")
    return {
        m: dfs[m].merge(common, on=keys, how="inner").sort_values(keys).reset_index(drop=True)
        for m in ALL_MODELS
    }


# ==========================================
# Report Structuring (restored old layout)
# ==========================================

def build_report_layout_and_mapping():
    """Build the row order and display-name mapping for the detailed CSV report."""
    metric_sort_order = ["Total evaluation frames"]
    row_cleaning_map = {"Total evaluation frames": "Total evaluation frames"}

    for m in CLASSIC_MODELS:
        disp = MODEL_DISPLAY_MAP[m]
        key_head = f"TF vs {disp}"
        metric_sort_order.append(key_head)
        row_cleaning_map[key_head] = key_head

        sub_metrics = [
            (f"  PR-AUC winner ({disp})", "  Statistical winner (PR-AUC)"),
            (f"  Threshold-free PR-AUC p-value ({disp})", f"  PR-AUC bootstrap p-value (TF > {disp})"),
            (f"  Threshold-free PR-AUC q-value ({disp})", f"  PR-AUC bootstrap FDR q-value (TF > {disp})"),
            (f"  McNemar winner ({disp})", "  Statistical winner (McNemar)"),
            (f"  McNemar p-value ({disp})", "  McNemar p-value"),
            (f"  McNemar q-value ({disp})", "  McNemar FDR q-value"),
            (f"  Discordant pairs (TF correct) ({disp})", "  Discordant pairs (TF correct)"),
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
    """Apply FDR correction and export the detailed + main summary CSV reports
    with the original layout (arrows, bold-winners, q-values, discordant pairs).
    """
    p_entries = []
    for window_name, record in results.items():
        for m in CLASSIC_MODELS:
            disp = MODEL_DISPLAY_MAP[m]
            for test_key, field in (("PR-AUC", f"  Threshold-free PR-AUC p-value ({disp})"),
                                     ("McNemar", f"  McNemar p-value ({disp})")):
                try:
                    p_val = float(record.get(field, "NaN"))
                    if not np.isnan(p_val):
                        p_entries.append({"window": window_name, "model": disp, "test": test_key, "p_val": p_val})
                except ValueError:
                    pass

    q_map = {}
    if p_entries:
        _, q_vals, _, _ = multipletests([e["p_val"] for e in p_entries], alpha=0.05, method="fdr_bh")
        for e, q in zip(p_entries, q_vals):
            q_map[(e["window"], e["model"], e["test"])] = q

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

        main_win_record["PR-AUC bootstrap statistical winner and p-value"] = ""
        main_win_record["McNemar statistical winner and p-value"] = ""

        for m in CLASSIC_MODELS:
            disp = MODEL_DISPLAY_MAP[m]

            q_pr = q_map.get((window_name, disp, "PR-AUC"), np.nan)
            raw_pr_winner = record.get(f"  PR-AUC winner ({disp})", "Tie")
            pr_winner = "Tie" if (not np.isnan(q_pr) and q_pr >= 0.05) else raw_pr_winner
            record[f"  PR-AUC winner ({disp})"] = pr_winner
            record[f"  Threshold-free PR-AUC q-value ({disp})"] = format_sci(q_pr)
            main_win_record[f"  TF vs {disp} (PR-AUC)"] = f"{pr_winner} (q {format_p_or_q(q_pr)})"

            q_mcn = q_map.get((window_name, disp, "McNemar"), np.nan)
            raw_mcn_winner = record.get(f"  McNemar winner ({disp})", "Tie")
            mcn_winner = "Tie" if (not np.isnan(q_mcn) and q_mcn >= 0.05) else raw_mcn_winner
            record[f"  McNemar winner ({disp})"] = mcn_winner
            record[f"  McNemar q-value ({disp})"] = format_sci(q_mcn)
            main_win_record[f"  TF vs {disp} (McNemar)"] = f"{mcn_winner} (q {format_p_or_q(q_mcn)})"

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
    main_rows_order.append("PR-AUC bootstrap statistical winner and p-value")
    for m in CLASSIC_MODELS:
        main_rows_order.append(f"  TF vs {MODEL_DISPLAY_MAP[m]} (PR-AUC)")
    main_rows_order.append("McNemar statistical winner and p-value")
    for m in CLASSIC_MODELS:
        main_rows_order.append(f"  TF vs {MODEL_DISPLAY_MAP[m]} (McNemar)")

    df_main = pd.DataFrame(main_table_data).reindex(main_rows_order)
    df_main.index.name = "Evaluation metric"
    df_main = df_main.fillna("")
    df_main.to_csv(outdir / f"{split}_{hs}_main_summary_report.csv")

    winners = {}
    for window_name in results:
        winners[window_name] = {}
        for m in CLASSIC_MODELS:
            disp = MODEL_DISPLAY_MAP[m]
            winners[window_name][disp] = {
                "PR-AUC": {"winner": results[window_name].get(f"  PR-AUC winner ({disp})", "Tie"),
                           "q_value": q_map.get((window_name, disp, "PR-AUC"))},
                "McNemar": {"winner": results[window_name].get(f"  McNemar winner ({disp})", "Tie"),
                            "q_value": q_map.get((window_name, disp, "McNemar"))},
            }

    return df_main, df_detailed, winners


# ==========================================
# Main Processing Pipeline
# ==========================================

def process(task, split, horizon, outdir):
    """Run model evaluation pipeline for a specific task, split, and horizon."""
    hs = horizon_name(horizon)
    cp = cache_path(task, split, hs)
    wp = winners_path(task, split, hs)
    main = outdir / f"{split}_{hs}_main_summary_report.csv"
    detailed = outdir / f"{split}_{hs}_head_to_head_report.csv"

    if cp.exists() and main.exists() and detailed.exists():
        try:
            with np.load(cp, allow_pickle=False) as z:
                if (
                    int(z["__schema_version__"]) != CACHE_SCHEMA_VERSION
                    or str(z["__param_hash__"]) != PARAM_HASH
                ):
                    raise RuntimeError("schema or parameter hash mismatch")
                payload = {k: z[k] for k in z.files}
            cache = {}
            for w in CLINICAL_WINDOWS:
                # Check whether the window was skipped when building the cache
                if payload.get(f"{w}_is_empty", False):
                    continue

                # Check window keys dynamically only if samples existed
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
            print(f"[CACHE HIT] split={split}, horizon={hs}, hash={PARAM_HASH}")
            return cache, pd.read_csv(main, index_col="Evaluation metric")
        except Exception as exc:
            print(f"[CACHE INVALID] split={split}, horizon={hs}: {type(exc).__name__}: {exc}")
    else:
        print(f"[CACHE MISS] split={split}, horizon={hs}, hash={PARAM_HASH}")

    val = {"Transformer": load_transformer(task, split, "validation", horizon)}
    test = {"Transformer": load_transformer(task, split, "test", horizon)}
    for m in CLASSIC_MODELS:
        val[m] = load_classic(m, task, split, "validation", horizon)
        test[m] = load_classic(m, task, split, "test", horizon)

    if any(d.empty for d in val.values()) or any(d.empty for d in test.values()):
        print(f"[SKIP] missing predictions for {split}/{hs}")
        return None, None

    val = aligned(val)
    test = aligned(test)

    if THRESHOLD_MODE == "global":
        for m in ALL_MODELS:
            if USE_CALIBRATED_PROBS[m]:
                val[m], test[m] = calibrate_dataframe_pair(df_val=val[m], df_test=test[m], prob_col="y_prob")

    tuning = val if THRESHOLD_TUNING_SET == "validation" else test
    glob = (
        {m: threshold(tuning[m], 0, 9999, m) for m in ALL_MODELS}
        if THRESHOLD_MODE == "global" else {}
    )

    results = {}
    cache = {}
    payload = {}

    for w, (lo, hi) in CLINICAL_WINDOWS.items():
        base = test["Transformer"]
        sub = (
            base[(base.time_step >= lo) & ((base.time_step + base.horizon) <= hi)][
                ["patientkey", "time_step", "horizon", "y_true", "y_prob"]
            ].rename(columns={"y_prob": "y_prob_Transformer"})
        )
        for m in CLASSIC_MODELS:
            x = (
                test[m][(test[m].time_step >= lo) & ((test[m].time_step + test[m].horizon) <= hi)][
                    ["patientkey", "time_step", "horizon", "y_prob"]
                ].rename(columns={"y_prob": f"y_prob_{m}"})
            )
            sub = sub.merge(x, on=["patientkey", "time_step", "horizon"], how="inner")

        if sub.empty or sub.y_true.nunique() != 2:
            print(f"[WINDOW SKIP] {split}/{hs}/{w}: empty or single-class test set")
            payload[f"{w}_is_empty"] = np.array(True)
            continue

        ts = {}
        for m in ALL_MODELS:
            v = tuning[m][(tuning[m].time_step >= lo) & ((tuning[m].time_step + tuning[m].horizon) <= hi)].copy()
            if THRESHOLD_MODE == "window_specific" and USE_CALIBRATED_PROBS[m]:
                cv, ct = calibrate_array_pair(
                    y_val_true=v.y_true.values, y_val_prob=v.y_prob.values,
                    y_test_prob=sub[f"y_prob_{m}"].values,
                )
                v["y_prob"] = cv
                sub[f"y_prob_{m}"] = ct
            ts[m] = glob[m] if THRESHOLD_MODE == "global" else threshold(v, lo, hi, m)

        y = sub.y_true.values
        rec = {"Total evaluation frames": len(sub)}
        pred = {m: (sub[f"y_prob_{m}"].values >= ts[m]) for m in ALL_MODELS}

        for m in CLASSIC_MODELS:
            disp = MODEL_DISPLAY_MAP[m]
            tfc = pred["Transformer"] == y
            bc = pred[m] == y
            n10 = int(np.sum(tfc & ~bc))
            n01 = int(np.sum(~tfc & bc))
            try:
                p_m = float(mcnemar([[np.sum(tfc & bc), n10], [n01, np.sum(~tfc & ~bc)]], exact=True).pvalue)
                mcn_winner = "Tie" if p_m > 0.05 else ("TF" if n10 > n01 else disp)
            except Exception:
                p_m = np.nan
                mcn_winner = "NaN"

            p_pr = float(paired_bootstrap_pr_auc_pvalue(
                y_true=y, p_tf=sub.y_prob_Transformer.values, p_base=sub[f"y_prob_{m}"].values,
                n_bootstraps=N_BOOTSTRAP,
            ))
            tfpr = average_precision_score(y, sub.y_prob_Transformer) if np.unique(y).size == 2 else 0.0
            bpr = average_precision_score(y, sub[f"y_prob_{m}"]) if np.unique(y).size == 2 else 0.0
            pr_winner = "NaN" if pd.isna(p_pr) else ("Tie" if p_pr > 0.05 else ("TF" if tfpr > bpr else disp))

            rec[f"TF vs {disp}"] = ""
            rec[f"  Threshold-free PR-AUC p-value ({disp})"] = f"{p_pr:.3e}" if not pd.isna(p_pr) else "NaN"
            rec[f"  PR-AUC winner ({disp})"] = pr_winner
            rec[f"  Discordant pairs (TF correct) ({disp})"] = n10
            rec[f"  Discordant pairs ({disp} correct) ({disp})"] = n01
            rec[f"  McNemar p-value ({disp})"] = f"{p_m:.3e}" if not pd.isna(p_m) else "NaN"
            rec[f"  McNemar winner ({disp})"] = mcn_winner

        for metric_prefix, arrow in METRIC_MAPPING:
            head_lbl = f"{metric_prefix} ({arrow})"
            raw_estimates = {}
            string_outputs = {}
            rec[head_lbl] = ""
            for m in ALL_MODELS:
                pe, low, high, text = bootstrap_ci(sub, m, ts[m], metric_prefix)
                raw_estimates[m] = pe
                string_outputs[m] = text
                payload[f"{w}_{m}_{metric_prefix}_pe"] = pe
                payload[f"{w}_{m}_{metric_prefix}_low"] = low
                payload[f"{w}_{m}_{metric_prefix}_high"] = high

            if metric_prefix == "ECE":
                best_model = min(raw_estimates, key=raw_estimates.get)
            else:
                best_model = max(raw_estimates, key=raw_estimates.get)
            for m in ALL_MODELS:
                disp = MODEL_DISPLAY_MAP[m]
                text = string_outputs[m]
                if m == best_model:
                    text = f"**{text}**"
                rec[f"  {head_lbl} ({disp})"] = text

        results[w] = rec
        cache[w] = (sub, ts)
        payload.update({
            f"{w}_is_empty": np.array(False),
            f"{w}_time_step": sub.time_step.values,
            f"{w}_horizon": sub.horizon.values,
            f"{w}_y_true": sub.y_true.values,
            f"{w}_thresholds_json": np.array(json.dumps(ts, sort_keys=True)),
        })
        payload.update({f"{w}_y_prob_{m}": sub[f"y_prob_{m}"].values for m in ALL_MODELS})

    if not results:
        return None, None

    df_main, df_detailed, winners = fdr_and_reports(results, outdir, split, hs)

    # Store FDR-corrected winner tokens directly in the cache payload for plotting.
    for w_name in CLINICAL_WINDOWS:
        for m in CLASSIC_MODELS:
            disp = MODEL_DISPLAY_MAP[m]
            payload[f"{w_name}_{disp}_PR-AUC_winner"] = winners.get(w_name, {}).get(disp, {}).get("PR-AUC", {}).get("winner", "Tie")
            payload[f"{w_name}_{disp}_McNemar_winner"] = winners.get(w_name, {}).get(disp, {}).get("McNemar", {}).get("winner", "Tie")

    payload["__schema_version__"] = np.array(CACHE_SCHEMA_VERSION)
    payload["__param_hash__"] = np.array(PARAM_HASH)
    np.savez_compressed(cp, **payload)
    wp.write_text(json.dumps(winners, indent=2, sort_keys=True, default=str))
    print(f"[CACHE SAVED] {cp.name}")

    with pd.option_context("display.max_colwidth", None, "display.max_rows", None):
        print(f"\n>>> MAIN SUMMARY REPORT: SPLIT [{split.upper()}] | HORIZON: {hs.upper()} <<<")
        print(df_main.to_markdown())

    return cache, df_main


# ==========================================
# Visualization Functions
# ==========================================

def render_bars(task, horizon, outdir, hs, csv_summaries_in_memory=None):
    """Render grouped bar charts (rows 1-4) + statistical test matrix rows (5-6),
    strictly locking y-tick locations across all subplots in every row.
    """
    metrics_to_plot = [("ROC-AUC", "↑"), ("PR-AUC", "↑"), ("Sensitivity", "↑"), ("Specificity", "↑")]
    
    metric_yticks_map = {
        "ROC-AUC": np.arange(0.5, 0.95, 0.1),
        "PR-AUC": np.arange(0.0, 0.45, 0.1),
        "Sensitivity": np.arange(0.0, 1.1, 0.2),
        "Specificity": np.arange(0.0, 1.1, 0.2),
    }
    metric_ylim_map = {"ROC-AUC": (0.5, 0.9), "PR-AUC": (0.0, 0.4), "Sensitivity": (0.0, 1.0), "Specificity": (0.0, 1.0)}

    n_windows = len(CLINICAL_WINDOWS)
    n_splits = len(SPLIT_TYPES)
    n_metrics = len(metrics_to_plot)
    total_rows = n_metrics + 2
    row_height_ratios = [1.0, 1.0, 1.0, 1.0, 0.8, 0.8]

    fig, axes = plt.subplots(
        total_rows, n_splits, figsize=(8.5 * n_splits, 25),
        gridspec_kw={"height_ratios": row_height_ratios}, squeeze=False,
    )

    csv_summary_by_split = csv_summaries_in_memory or {}
    npz_data_by_split = {}
    for split in SPLIT_TYPES:
        cp = cache_path(task, split, hs)
        if cp.exists():
            with np.load(cp, allow_pickle=True) as z:
                npz_data_by_split[split] = {k: z[k] for k in z.files}
        if split not in csv_summary_by_split:
            main_csv_path = outdir / f"{split}_{hs}_main_summary_report.csv"
            if main_csv_path.exists():
                df_csv = pd.read_csv(main_csv_path, index_col="Evaluation metric")
                csv_summary_by_split[split] = df_csv.fillna("")

    window_keys = list(CLINICAL_WINDOWS.keys())

    for m_idx, (metric_prefix, arrow) in enumerate(metrics_to_plot):
        metric_label = f"{metric_prefix} ({arrow})"
        ylim = metric_ylim_map.get(metric_prefix, (0.0, 1.0))
        yticks = metric_yticks_map[metric_prefix]

        for s_idx, split in enumerate(SPLIT_TYPES):
            ax = axes[m_idx, s_idx]
            ax.set_axisbelow(True)
            ax.minorticks_off()

            split_title = split.replace("_", " ").capitalize()
            x_group_centers = np.arange(n_windows)
            n_models = len(MODEL_PLOT_ORDER)
            bar_width = 0.20
            offsets = (np.arange(n_models) - (n_models - 1) / 2.0) * bar_width

            for m_model_idx, m in enumerate(MODEL_PLOT_ORDER):
                c_hex = COLORS[m]
                pes, yerr_low, yerr_high = [], [], []
                cache_dict = npz_data_by_split.get(split, {})

                for window_name in window_keys:
                    pe_key = f"{window_name}_{m}_{metric_prefix}_pe"
                    low_key = f"{window_name}_{m}_{metric_prefix}_low"
                    high_key = f"{window_name}_{m}_{metric_prefix}_high"
                    if pe_key in cache_dict:
                        pe, low, high = float(cache_dict[pe_key]), float(cache_dict[low_key]), float(cache_dict[high_key])
                    else:
                        pe, low, high = np.nan, np.nan, np.nan

                    if np.isnan(pe):
                        pes.append(0); yerr_low.append(0); yerr_high.append(0)
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

    test_rows_info = [("PR-AUC test", "PR-AUC"), ("McNemar test", "McNemar")]
    sub_baselines = ["LR", "RF", "XGB"]
    y_positions = [0.65, 0.0, -0.65]

    for test_idx, (test_row_label, test_type_key) in enumerate(test_rows_info):
        ax_row = n_metrics + test_idx
        for s_idx, split in enumerate(SPLIT_TYPES):
            ax = axes[ax_row, s_idx]
            ax.set_axisbelow(True)
            ax.minorticks_off()
            ax.set_xlim([-0.5, n_windows - 0.5])
            ax.set_ylim([-1.1, 1.1])
            
            ax.set_yticks(y_positions)
            ax.set_yticklabels(["", "", ""])
            
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
                    row_key_unique = f"  TF vs {base_disp} ({test_type_key})"
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
                        elif cell_val in {"TF", "LR", "RF", "XGB", "Tie"}:
                            winner_token = cell_val
                    if not winner_token:
                        cache_key = f"{window_name}_{base_disp}_{test_type_key}_winner"
                        winner_token = str(cache_dict.get(cache_key, "Tie"))

                    if winner_token == "TF":
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
            ncol=len(MODEL_PLOT_ORDER), fontsize=22, frameon=True, framealpha=0.9, borderpad=0.25,
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
    """Render ROC, PR, and Decision Curve Analysis (DCA) grids with synchronized y-ticks across all columns."""
    n_windows = len(CLINICAL_WINDOWS)
    n_splits = len(SPLIT_TYPES)

    fig_roc, axes_roc = plt.subplots(n_windows, n_splits, figsize=(7.5 * n_splits, 4.5 * n_windows), squeeze=False)
    fig_pr, axes_pr = plt.subplots(n_windows, n_splits, figsize=(7.5 * n_splits, 4.5 * n_windows), squeeze=False)
    fig_dca, axes_dca = plt.subplots(n_windows, n_splits, figsize=(7.5 * n_splits, 4.5 * n_windows), squeeze=False)

    for w_idx, window_name in enumerate(CLINICAL_WINDOWS):
        max_dca_y_limit = 0.02
        for split in SPLIT_TYPES:
            if split in cache and window_name in cache[split]:
                sub_df, _ = cache[split][window_name]
                prevalence = np.sum(sub_df.y_true.values == 1) / len(sub_df)
                max_dca_y_limit = max(max_dca_y_limit, prevalence * 1.05)

        dca_yticks = np.linspace(0.0, max_dca_y_limit, 5)

        for s_idx, split in enumerate(SPLIT_TYPES):
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
            y_true = sub_df.y_true.values

            # --- ROC Curves ---
            for m in MODEL_PLOT_ORDER:
                p_arr = sub_df[f"y_prob_{m}"].values
                c_hex = COLORS[m]
                label = MODEL_FULLNAME_MAP[m] if (w_idx == 0 and s_idx == 0) else ""
                fpr, tpr, _ = roc_curve(y_true, p_arr)
                ax_roc.plot(fpr, tpr, label=label, color=c_hex, lw=3.5, zorder=2)
            ax_roc.plot([0, 1], [0, 1], linestyle="--", color="gray", alpha=0.5, zorder=1)
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
                p_arr = sub_df[f"y_prob_{m}"].values
                c_hex = COLORS[m]
                label = MODEL_FULLNAME_MAP[m] if (w_idx == 0 and s_idx == 0) else ""
                prec_arr, rec_arr, _ = precision_recall_curve(y_true, p_arr)
                ax_pr.plot(rec_arr, prec_arr, label=label, color=c_hex, lw=3.5, zorder=2)
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
            dca_thresh = np.linspace(0.01, 0.50, 50)
            prevalence = np.sum(y_true == 1) / len(y_true)
            ax_dca.plot(dca_thresh, np.zeros_like(dca_thresh), color="#1a1a1a", linestyle="--", lw=3.5,
                        label="Treat none" if (w_idx == 0 and s_idx == 0) else "", zorder=1)
            ax_dca.plot(dca_thresh, prevalence - (1.0 - prevalence) * (dca_thresh / (1.0 - dca_thresh)),
                        color="#a0a0a0", linestyle="--", lw=3.5, label="Treat all" if (w_idx == 0 and s_idx == 0) else "", zorder=1)
            for m in MODEL_PLOT_ORDER:
                p_arr = sub_df[f"y_prob_{m}"].values
                c_hex = COLORS[m]
                label = MODEL_FULLNAME_MAP[m] if (w_idx == 0 and s_idx == 0) else ""
                net_benefit = [
                    (np.sum((p_arr >= t) & (y_true == 1)) / len(y_true))
                    - (np.sum((p_arr >= t) & (y_true == 0)) / len(y_true)) * (t / (1.0 - t))
                    for t in dca_thresh
                ]
                ax_dca.plot(dca_thresh, net_benefit, label=label, color=c_hex, lw=3.5, zorder=2)

            ax_dca.set_xlim([0.0, 0.5])
            ax_dca.set_ylim([-0.005 if max_dca_y_limit < 0.1 else -0.03, max_dca_y_limit])
            
            ax_dca.set_yticks(dca_yticks)
            ax_dca.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
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
            fig_obj.legend(
                handles, labels, loc="upper center", bbox_to_anchor=(0.544, 1.04),
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


# ==========================================
# Pipeline Orchestration
# ==========================================

def pipeline(task, horizon):
    """Execute evaluation and visualization pipeline for a given task and horizon."""
    hs = horizon_name(horizon)
    outdir = ANALYSIS_SUBDIR / task / hs
    outdir.mkdir(parents=True, exist_ok=True)
    cache = {}
    csv_summaries = {}

    for split in SPLIT_TYPES:
        wc, df_main = process(task, split, horizon, outdir)
        if wc is not None:
            cache[split] = wc
            csv_summaries[split] = df_main

    if cache:
        render_curves(cache, outdir, hs)
        render_bars(task, horizon, outdir, hs, csv_summaries)


if __name__ == "__main__":
    for task in TASKS:
        for horizon in ["combined", *TARGET_HORIZONS]:
            pipeline(task, horizon)