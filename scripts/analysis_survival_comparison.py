"""
Head-to-head comparison of model families on the SAME survival ground truth.

Families compared (all aligned per patient_id and per fup observation point):
    1. Classic ML baselines        (results/classic_ml/...)
    2. Binary transformer          (results/transformer/...)
    3. Survival transformer (Cox)  (results/survival_transformer/...)

The survival ground truth (time, status) comes from the raw t-EAV datasets
(label_{outcome}_time / label_{outcome}_status, derived on-the-fly for the
bacteria/virus/fungi categories). To compare models whose native outputs are
different (P(event by H) vs Cox log hazard ratio), we evaluate shared metrics:

    - AUROC @ horizon H : binary target derived from (time, status) by
                          binarization at H (censored-before-H rows masked out);
                          survival risks are 1 - S(H|x) from a Breslow baseline.
    - AP     @ horizon H: same binary target.
    - IPCW Brier @ H     : inverse-probability-of-censoring-weighted Brier score.
    - C-index (stratified by fup) : using each model's risk at the largest horizon.
    - IBS                    : IPCW integrated Brier score over the horizon grid.

Additionally, paired stratified bootstrap p-values are computed for AUROC/AP
differences between a reference model (default: the survival transformer) and
every other model, per task and horizon.

Usage:
    python scripts/analysis_survival_comparison.py -c configs/survival_training.yaml
(the analysis parameters live in the 'comparison:' section of that config)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import re
import yaml
import numpy as np
import pandas as pd
from collections import defaultdict
from datasets import load_from_disk
from sklearn.metrics import roc_auc_score, average_precision_score

from src.data.survival_labels import (
    resolve_survival_label_keys, derive_category_survival_labels,
    survival_keep_mask, binarize_survival_labels,
)
from src.evaluation.survival_metrics import (
    concordance_index, ipcw_brier_scores, integrated_brier_score,
)
from src.tasks import normalize_task_key


# ---------------------------------------------------------------------------
# Ground truth (survival labels from the raw t-EAV datasets)
# ---------------------------------------------------------------------------

def load_survival_truth(
    teav_root: Path,
    split_type: str,
    hf_split: str,
    fup: int,
    task_key: str,
    cache: dict,
) -> pd.DataFrame:
    """
    Returns DataFrame(patient_id, time, status) for one fup snapshot and outcome,
    with invalid/zero-information survival rows removed.
    """
    ds = cache.get(fup)
    if ds is None:
        ds = load_from_disk(str(teav_root / split_type / f"fup_{fup:04d}"))[hf_split]
        cache[fup] = ds

    time_key, status_key, _ = resolve_survival_label_keys(task_key)
    if status_key not in ds.column_names:
        ds = derive_category_survival_labels(ds, task_key)

    patientid = np.asarray(ds["patientid"])
    times = np.asarray(ds[time_key], dtype=np.float64)
    statuses = np.asarray(ds[status_key], dtype=np.int64)
    keep = survival_keep_mask(times, statuses)
    return pd.DataFrame({
        "patient_id": patientid[keep].astype(str),
        "time": times[keep],
        "status": statuses[keep],
    })


# ---------------------------------------------------------------------------
# Per-family prediction loaders -> DataFrame(patient_id, risk) per (fup, horizon)
# All families now read the SAME tidy per-run table written by the training
# scripts: {dataset_split}_predictions.csv with columns
#     patient_id, fup, horizon, label, prob [, prob_cal, time, status, risk,
#                                            pred_median_time]
# ---------------------------------------------------------------------------

CSV_CACHE: dict = {}


def _read_predictions_csv(run_dir: Path, dataset_split: str) -> pd.DataFrame:
    """Loads (and caches) one run's tidy predictions table for a dataset split."""
    key = (str(run_dir), dataset_split)
    df = CSV_CACHE.get(key)
    if df is None:
        path = run_dir / f"{dataset_split}_predictions.csv"
        df = pd.read_csv(path) if path.exists() else pd.DataFrame()
        CSV_CACHE[key] = df
    return df


def _find_run_dir(task_dir: Path, horizon: int, csv_name: str) -> tuple[Path, list[int]]:
    """Finds a run directory (hrz(...)_fut(...)_fuv(...)) containing the horizon and CSV."""
    if not task_dir.exists():
        raise FileNotFoundError(f"Missing task directory: {task_dir}")
    candidates = []
    for run_dir in sorted(task_dir.iterdir()):
        if not run_dir.is_dir() or not (run_dir / csv_name).exists():
            continue
        match = re.search(r"hrz\(([\d-]+)\)", run_dir.name)
        if not match:
            continue
        horizons = [int(h) for h in match.group(1).split("-")]
        if horizon in horizons:
            candidates.append((run_dir, horizons))
    if not candidates:
        raise FileNotFoundError(f"No run with horizon {horizon} and {csv_name} inside {task_dir}")
    return candidates[0]


def _load_predictions(
    task_dir: Path,
    horizon: int,
    fup: int,
    dataset_split: str,
    risk_col: str = "prob",
    task_filter: str | None = None,
) -> pd.DataFrame:
    """
    Extracts one (horizon, fup) slice of a run's tidy predictions table as
    DataFrame(patient_id, risk). Returns empty when the run has no rows for
    this follow-up point.
    """
    csv_name = f"{dataset_split}_predictions.csv"
    if not task_dir.exists():
        raise FileNotFoundError(f"Missing task directory: {task_dir}")
    run_dir, _ = _find_run_dir(task_dir, horizon, csv_name)
    df = _read_predictions_csv(run_dir, dataset_split)
    if df.empty:
        return pd.DataFrame()
    sub = df[(df["horizon"] == horizon) & (df["fup"] == fup)]
    if task_filter is not None and "task" in sub.columns:
        sub = sub[sub["task"].isin([task_filter, normalize_task_key(task_filter)])]
    if sub.empty:
        return pd.DataFrame()
    return pd.DataFrame({
        "patient_id": sub["patient_id"].astype(str).to_numpy(),
        "risk": sub[risk_col].to_numpy(dtype=np.float64),
    })


def load_binary_transformer_predictions(
    results_root: Path,
    split_type: str,
    run_id: str,
    task_key: str,
    horizon: int,
    fup: int,
    dataset_split: str,
) -> pd.DataFrame:
    """Loads P(event by horizon) from single-task or multi-task binary transformer runs."""
    norm_task = normalize_task_key(task_key)
    clean_run_id = run_id if run_id.endswith(("_raw", "_bin")) else f"{run_id}_raw"
    ft_base = results_root / split_type / clean_run_id / "finetuning"
    folder_candidates = [
        norm_task,
        task_key,
        "multitask_unified",
        "multitask_all",
        "multitask_bacteria",
        "multitask_viruses",
        "multitask_fungi",
    ]
    for folder in folder_candidates:
        cand_dir = ft_base / folder
        if not cand_dir.exists():
            continue
        try:
            df = _load_predictions(cand_dir, horizon, fup, dataset_split, task_filter=norm_task)
            if not df.empty:
                return df
        except FileNotFoundError:
            continue
    raise FileNotFoundError(f"No binary transformer predictions for ({task_key}, h={horizon}) under {ft_base}")


def load_classic_ml_predictions(
    results_root: Path,
    split_type: str,
    model_type: str,
    task_key: str,
    horizon: int,
    fup: int,
    dataset_split: str,
) -> pd.DataFrame:
    """Loads P(event by horizon) from a classic ML run (supports both grp_* and infection_grp_*)."""
    norm_task = normalize_task_key(task_key)
    for t_cand in (norm_task, task_key):
        cand_dir = results_root / split_type / model_type / t_cand
        if cand_dir.exists():
            return _load_predictions(cand_dir, horizon, fup, dataset_split)
    raise FileNotFoundError(f"Missing classic_ml directory for ({model_type}, {task_key})")


def load_survival_transformer_predictions(
    results_root: Path,
    split_type: str,
    run_id: str,
    task_key: str,
    horizon: int,
    fup: int,
    dataset_split: str,
) -> pd.DataFrame:
    """Loads P(event by horizon) = 1 - S(horizon|x) from the Cox survival run (tidy CSV)."""
    clean_run_id = run_id if run_id.endswith(("_raw", "_bin")) else f"{run_id}_raw"
    norm_task = normalize_task_key(task_key)
    for subdir in ("finetuning_survival", "finetuning"):
        for t_cand in (task_key, norm_task):
            cand_dir = results_root / split_type / clean_run_id / subdir / t_cand
            if cand_dir.exists():
                return _load_predictions(cand_dir, horizon, fup, dataset_split)
    raise FileNotFoundError(f"Missing survival predictions for {task_key} under {results_root}")


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def horizon_metrics(df: pd.DataFrame, horizon: int, risk_col: str = "risk") -> dict:
    """
    AUROC / AP / IPCW-Brier on the pooled (fup-stratified rows) subset of a model.
    df columns: patient_id, fup, time, status, and the risk column (one row per patient/fup).
    """
    y = binarize_survival_labels(df["time"].to_numpy(), df["status"].to_numpy(), horizon)
    risk = df[risk_col].to_numpy()
    valid = y != -100
    out = {"n": int(valid.sum())}
    if valid.sum() == 0:
        return {**out, "auroc": np.nan, "ap": np.nan, "ipcw_brier": np.nan}
    y_v, r_v = y[valid], risk[valid]
    out["event_rate"] = float(np.mean(y_v))
    out["auroc"] = float(roc_auc_score(y_v, r_v)) if len(np.unique(y_v)) > 1 else np.nan
    out["ap"] = float(average_precision_score(y_v, r_v)) if len(np.unique(y_v)) > 1 else np.nan

    # IPCW Brier with per-fup censoring weights (patients may start at different fups)
    briers, weights = [], []
    horizons = [horizon]
    for fup, grp in df[valid].groupby("fup"):
        s_pred = np.clip(1.0 - grp[risk_col].to_numpy(), 0.0, 1.0)[:, None]
        b = ipcw_brier_scores(
            grp["time"].to_numpy(), grp["status"].to_numpy(), s_pred, horizons,
        )
        briers.append(b[0])
        weights.append(len(grp))
    if briers:
        out["ipcw_brier"] = float(np.average(briers, weights=weights))
    else:
        out["ipcw_brier"] = np.nan
    return out


def survival_level_metrics(df: pd.DataFrame, grid: list[int]) -> dict:
    """
    C-index (stratified by fup, using the risk at the largest grid horizon) and
    IPCW Integrated Brier Score over the horizon grid. df must contain one row
    per (patient_id, fup) with columns risk_{h} for every grid horizon.
    """
    risk_max_h = grid[-1]
    cindex = np.nan
    ibs = np.nan
    if f"risk_{risk_max_h}" in df.columns:
        sub = df.dropna(subset=[f"risk_{risk_max_h}"])
        if len(sub) > 0:
            cindex = concordance_index(
                sub["time"].to_numpy(), sub["status"].to_numpy(),
                sub[f"risk_{risk_max_h}"].to_numpy(), strata=sub["fup"].to_numpy(),
            )
    grid_cols = [f"risk_{h}" for h in grid]
    if all(c in df.columns for c in grid_cols):
        sub = df.dropna(subset=grid_cols)
        briers_per_fup, weights = [], []
        for fup, grp in sub.groupby("fup"):
            if grp["status"].nunique() == 0:
                continue
            surv = np.clip(1.0 - grp[grid_cols].to_numpy(), 0.0, 1.0)
            briers_per_fup.append(ipcw_brier_scores(
                grp["time"].to_numpy(), grp["status"].to_numpy(), surv, grid,
            ))
            weights.append(len(grp))
        if briers_per_fup:
            pooled = np.average(np.stack(briers_per_fup), axis=0, weights=weights)
            ibs = integrated_brier_score(pooled, grid)
    return {"cindex": float(cindex) if not np.isnan(cindex) else np.nan, "ibs": ibs}


def stratified_paired_bootstrap(
    df: pd.DataFrame,
    col_a: str,
    col_b: str,
    y_col: str,
    n_bootstraps: int = 1000,
    seed: int = 42,
) -> dict:
    """
    Patient-cluster paired bootstrap testing two-sided differences between model A and model B
    on AUROC and AP across correlated longitudinal snapshots per `patient_id`.
    """
    from scripts.script_utils import build_patient_cluster_bootstrap_counts, vectorized_bootstrap_ap_and_roc

    y = df[y_col].to_numpy(dtype=np.float64)
    valid = (y == 0) | (y == 1)
    if np.sum(valid) < 5 or len(np.unique(y[valid])) < 2:
        return {"auroc_delta": np.nan, "auroc_p": np.nan, "ap_delta": np.nan, "ap_p": np.nan}

    y_v = y[valid]
    pa = df[col_a].to_numpy(dtype=np.float64)[valid]
    pb = df[col_b].to_numpy(dtype=np.float64)[valid]
    pids = df["patient_id"].to_numpy()[valid] if "patient_id" in df.columns else None

    d_auc = float(roc_auc_score(y_v, pa) - roc_auc_score(y_v, pb))
    d_ap = float(average_precision_score(y_v, pa) - average_precision_score(y_v, pb))

    counts = build_patient_cluster_bootstrap_counts(pids, len(y_v), n_bootstraps, seed=seed)
    pos_counts = counts @ y_v
    tot_counts = np.sum(counts, axis=1)
    valid_boot = (pos_counts > 0) & (pos_counts < tot_counts)
    if not np.any(valid_boot):
        return {"auroc_delta": d_auc, "auroc_p": np.nan, "ap_delta": d_ap, "ap_p": np.nan}

    ap_a, roc_a = vectorized_bootstrap_ap_and_roc(y_v, pa, counts, valid_boot_mask=valid_boot, compute_roc=True)
    ap_b, roc_b = vectorized_bootstrap_ap_and_roc(y_v, pb, counts, valid_boot_mask=valid_boot, compute_roc=True)

    def _two_sided_p(diffs: np.ndarray) -> float:
        if len(diffs) == 0:
            return np.nan
        n_b = len(diffs)
        p_le = (np.sum(diffs <= 0.0) + 1.0) / (n_b + 1.0)
        p_ge = (np.sum(diffs >= 0.0) + 1.0) / (n_b + 1.0)
        return float(min(1.0, 2.0 * min(p_le, p_ge)))

    return {
        "auroc_delta": d_auc,
        "auroc_p": _two_sided_p(roc_a - roc_b),
        "ap_delta": d_ap,
        "ap_p": _two_sided_p(ap_a - ap_b),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Compare classic ML vs binary vs survival models.")
    parser.add_argument("--config", "-c", type=str, default="configs/survival_training.yaml")
    parser.add_argument("--overrides", "-o", action="append", default=[], help="Overrides config (JSON string or key=value).")
    return parser.parse_args()


def main():
    from src.utils import apply_config_overrides
    args = parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    cfg = apply_config_overrides(cfg, args.overrides)

    # Analysis parameters live in the "comparison:" section of the shared
    # training config; top-level keys are used as fallback for standalone configs.
    comp = cfg.get("comparison", cfg)
    split_type = cfg["data_split_type"]
    dataset_split = comp.get("dataset_split", "test")
    # Survival ground truth (time/status): same raw t-EAV datasets as training.
    teav_root = Path(comp.get("teav_root", cfg["data_dir"]))
    out_dir = Path(comp.get("output_dir", "results/analysis/survival_comparison")) / split_type / dataset_split
    out_dir.mkdir(parents=True, exist_ok=True)

    reference = comp.get("reference_model", "transformer_survival")
    n_boot = int(cfg.get("n_bootstrap", comp.get("n_bootstrap", 1000)))
    run_id = comp.get("run_id", "e00-a15-v60")

    # Registry of model loaders: name -> callable(task, horizon, fup) -> df(patient_id, risk)
    model_loaders = {}
    if comp.get("binary_transformer", {}).get("enabled", True):
        bt = comp.get("binary_transformer", {})
        model_loaders["transformer_binary"] = lambda task, h, fup, _bt=bt: \
            load_binary_transformer_predictions(
                Path(_bt.get("results_dir", "results/transformer")), split_type,
                _bt.get("run_id", run_id), task, h, fup,
                dataset_split,
            )
    if comp.get("survival_transformer", {}).get("enabled", True):
        st = comp.get("survival_transformer", {})
        model_loaders["transformer_survival"] = lambda task, h, fup, _st=st: \
            load_survival_transformer_predictions(
                Path(_st.get("results_dir", "results/survival_transformer")), split_type,
                _st.get("run_id", run_id), task, h, fup, dataset_split,
            )
    for model_type in comp.get("classic_ml_models", []):
        cm = comp.get("classic_ml", {})
        model_loaders[f"classic_{model_type}"] = (
            lambda task, h, fup, _mt=model_type, _cm=cm: load_classic_ml_predictions(
                Path(_cm.get("results_dir", "results/classic_ml")), split_type, _mt, task, h, fup, dataset_split,
            )
        )

    horizon_rows, survival_rows, bootstrap_rows = [], [], []

    tasks = comp.get("prediction_tasks") or cfg["prediction_tasks"]
    for task_key, task_specs in tasks.items():
        fups = task_specs["fups"]
        grid = sorted(int(h) for h in task_specs["horizons"])
        print(f"\n===== Task: {task_key} | fups={fups} | horizons={grid} =====")

        # 1) Ground truth per fup
        truths = {}
        for fup in fups:
            try:
                tdf = load_survival_truth(teav_root,
                                         split_type, dataset_split, fup, task_key, TRUTH_CACHE)
            except FileNotFoundError:
                print(f"  [skip] missing raw fup_{fup:04d} dataset")
                continue
            if len(tdf):
                tdf["fup"] = fup
                truths[fup] = tdf
        if not truths:
            print("  [skip] no ground truth available")
            continue

        # 2) Predictions per model: wide frame with risk_{h} columns, joined on truth rows
        model_frames = {}
        for model_name, loader in model_loaders.items():
            frames = []
            for fup in truths:
                for h in grid:
                    try:
                        pred = loader(task_key, h, fup)
                    except FileNotFoundError:
                        continue  # model has no run for this horizon
                    if pred.empty:
                        continue
                    pred = pred.assign(fup=fup, horizon=h)
                    frames.append(pred)
            if not frames:
                print(f"  [skip] no predictions found for model '{model_name}'")
                continue
            long_df = pd.concat(frames, ignore_index=True)
            wide = long_df.pivot_table(
                index=["fup", "patient_id"], columns="horizon", values="risk", aggfunc="first",
            ).rename(columns={h: f"risk_{h}" for h in long_df["horizon"].unique()})
            # Join to the ground-truth rows (left join keeps all truth patients)
            joined = pd.concat(truths.values(), ignore_index=True).merge(
                wide.reset_index(), on=["fup", "patient_id"], how="inner",
            )
            model_frames[model_name] = joined
            print(f"  [{model_name}] joined rows: {len(joined)}")

        if not model_frames:
            continue

        # 3) Shared horizon-wise metrics
        for model_name, frame in model_frames.items():
            for h in grid:
                col = f"risk_{h}"
                if col not in frame.columns:
                    continue
                sub = frame.dropna(subset=[col])
                m = horizon_metrics(sub, h, risk_col=col)
                horizon_rows.append({"task": task_key, "model": model_name, "horizon": h, **m})

        # 4) Survival-level metrics (C-index stratified by fup + IBS)
        for model_name, frame in model_frames.items():
            m = survival_level_metrics(frame, grid)
            survival_rows.append({"task": task_key, "model": model_name, **m})

        # 5) Paired stratified bootstrap vs the reference model
        if reference in model_frames:
            ref = model_frames[reference]
            for model_name, frame in model_frames.items():
                if model_name == reference:
                    continue
                for h in grid:
                    if f"risk_{h}" not in ref.columns or f"risk_{h}" not in frame.columns:
                        continue
                    a = ref[["fup", "patient_id", f"risk_{h}"]].rename(columns={f"risk_{h}": "risk_a"})
                    b = frame[["fup", "patient_id", f"risk_{h}"]].rename(columns={f"risk_{h}": "risk_b"})
                    pair_h = a.merge(b, on=["fup", "patient_id"]).merge(
                        ref[["fup", "patient_id", "time", "status"]], on=["fup", "patient_id"])
                    pair_h = pair_h.dropna(subset=["risk_a", "risk_b"])
                    if pair_h.empty:
                        continue
                    pair_h = pair_h.assign(
                        y=binarize_survival_labels(
                            pair_h["time"].to_numpy(), pair_h["status"].to_numpy(), h)
                    )
                    pair_h = pair_h[pair_h["y"] != -100]
                    stats = stratified_paired_bootstrap(pair_h, "risk_a", "risk_b", "y", n_bootstraps=n_boot)
                    bootstrap_rows.append({
                        "task": task_key, "horizon": h,
                        "model_a": reference, "model_b": model_name, **stats,
                    })

    # ------------------------------------------------------------------ outputs
    df_h = pd.DataFrame(horizon_rows)
    df_s = pd.DataFrame(survival_rows)
    df_b = pd.DataFrame(bootstrap_rows)
    df_h.to_csv(out_dir / "metrics_by_horizon.csv", index=False)
    df_s.to_csv(out_dir / "metrics_survival_level.csv", index=False)
    df_b.to_csv(out_dir / "bootstrap_pvalues.csv", index=False)

    print("\n================= SHARED METRICS BY HORIZON =================")
    if not df_h.empty:
        print(df_h.pivot_table(index=["task", "model"], columns="horizon",
                               values=["auroc", "ap"]).round(3).to_string())
        print("\n--- IPCW Brier by horizon ---")
        print(df_h.pivot_table(index=["task", "model"], columns="horizon",
                               values="ipcw_brier").round(4).to_string())
    print("\n============ SURVIVAL-LEVEL METRICS (C-index / IBS) ============")
    if not df_s.empty:
        print(df_s.round(4).to_string(index=False))
    print("\n============ PAIRED BOOTSTRAP (reference vs others) ============")
    if not df_b.empty:
        print(df_b.round(4).to_string(index=False))
    print(f"\nSaved outputs to: {out_dir}")


# Cache for raw HF datasets keyed by fup (shared between truth and alignment)
TRUTH_CACHE: dict = {}


if __name__ == "__main__":
    main()
