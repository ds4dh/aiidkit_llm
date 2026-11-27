import re
import json
import src.utils  # activates UTF-8 log tee when AIIDKIT_LOG_FILE is set
import numpy as np
import pandas as pd
from pathlib import Path
from datasets import Dataset
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

from src.model.model_utils import resolve_best_checkpoint
from src.tasks import DEFAULT_VALIDATION_FUPS, get_task_meta, normalize_task_key


def get_value_embedding_tag(cfg: dict) -> str:
    """Return short embedding notation (e.g. 'raw', 'bin', 'raw_stcond', 'bin_stcond')."""
    emb_cfg = cfg.get("model", {}).get("embedding_layer_config", {})
    use_num = emb_cfg.get("use_numerical_embedding", cfg.get("data_collator", {}).get("use_numerical_embedding", True))
    use_st = bool(emb_cfg.get("use_static_conditioning", False))
    base = "raw" if bool(use_num) else "bin"
    return f"{base}_stcond" if use_st else base


def format_pretrain_run_id(cfg: dict, masking_rules: dict | None = None) -> str:
    """Build the pre-training subfolder identifier, e.g. 'e00-a15-v60_raw' or 'e00-a15-v60_raw_stcond'."""
    val_tag = get_value_embedding_tag(cfg)
    if masking_rules is None and cfg.get("run_id"):
        return resolve_run_id_with_val_tag(str(cfg["run_id"]), val_tag)
    rules = masking_rules if masking_rules is not None else cfg["data_collator"]["mlm_masking_rules"]
    mask_tag = "-".join([f"{k[0]}{int(round(v * 100)):02d}" for k, v in rules.items()])
    return f"{mask_tag}_{val_tag}"


def resolve_run_id_with_val_tag(run_id: str, val_tag: str) -> str:
    """Ensure a pretraining run_id ends with the expected value/static embedding suffix."""
    base = re.sub(r"_(raw|bin)(_stcond)?$", "", run_id.strip())
    return f"{base}_{val_tag}" if base else f"e00-a15-v60_{val_tag}"


def scan_all_fups(data_dir: Path) -> list[int]:
    """Read evaluation follow-up milestones from a v4.0 dynamic store's ``manifest.json``."""
    manifest_p = Path(data_dir) / "manifest.json"
    if not manifest_p.exists():
        return []
    meta = json.loads(manifest_p.read_text(encoding="utf-8"))
    return sorted(int(f) for f in meta.get("default_eval_fups", []))


def prepare_dataset_fup_dict(dataset: Dataset, fup_list: list[int]):
    """
    Creates a dictionary of datasets for different follow-up periods.
    """
    out_dict = {"all": dataset}
    fup_array = np.array(dataset["fup"])
    for fup in fup_list:
        indices = np.where(fup_array == fup)[0]
        if len(indices) > 0:
            subset = dataset.select(indices, keep_in_memory=True)  # in-memory dataset view
            out_dict[f"fup_{fup:04d}"] = subset
            
    return out_dict


def save_predictions_table(
    all_predictions: dict,
    split_label: str,
    horizons: list[int],
    output_dir: Path,
    csv_name: str | None = None,
    task_keys: list[str] | None = None,
    head_pairs: list[tuple[str, int]] | None = None,
) -> pd.DataFrame | None:
    """
    Flatten the per-FUP prediction arrays saved by the final test pass into a
    tidy (long-format) table with one row per (patient, fup, [task,] horizon):

        patient_id, fup, horizon, label, prob [, prob_cal, time, status,
                                               risk, pred_median_time]
        (multi-task runs add a 'task' column: one row per patient, fup, task,
         horizon, with the label/prob of that task's head)

    - Reads keys of the form '{split_label}_fup_{XXXX}_labels', '..._probs'
      and (optionally) '..._patientid', '..._probs_cal', '..._times',
      '..._statuses', '..._risks', '..._median_times' from the dict built by
      the test passes of train_classification.py / train_survival.py /
      train_classic_ml.py.
    - 'task_keys': for multi-task runs, the ordered list of tasks owning the
      matrix columns (task-major: [t1_h1..t1_hK, t2_h1..t2_hK, ...]). When
      given, every matrix column becomes a (task, horizon) row and a 'task'
      column is written. When None (classic single-task runs) the columns are
      interpreted as the horizon list, as before.
    - 'label' is 0/1 (binary tasks) or the horizon-binarized survival label
      (1 = event within horizon, 0 = known event-free at horizon). Rows whose
      label is unknown at that horizon (-100, i.e. censored before horizon)
      are skipped.
    - 'prob' is the model's predicted probability of the positive case
      (sigmoid output for binary tasks; 1 - S(horizon) for survival).
    - Survival runs additionally expose the raw ground truth and model output
      per row: 'time'/'status' = observed (censored) time to next event and
      event indicator from the fup observation point; 'risk' = Cox log hazard
      ratio; 'pred_median_time' = model-predicted median time to the next
      event (NaN when no median is reached within the observed baseline).
    - Writes a plain CSV (one file per split, per training run).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fup_re = re.compile(rf"^{re.escape(split_label)}_fup_(\d+)_labels$")

    # Optional per-row (horizon-independent) arrays: npz key suffix -> CSV column
    extras_map = {"times": "time", "statuses": "status", "risks": "risk",
                  "median_times": "pred_median_time"}

    frames = []
    for key in sorted(all_predictions.keys()):
        match = fup_re.match(key)
        if not match:
            continue
        fup_str, fup_val = match.group(1), int(match.group(1))
        prefix = f"{split_label}_fup_{fup_str}"
        if f"{prefix}_probs" not in all_predictions:
            continue

        labels = np.asarray(all_predictions[key])
        probs = np.asarray(all_predictions[f"{prefix}_probs"])
        if labels.ndim == 1:
            labels = labels[:, None]
        if probs.ndim == 1:
            probs = probs[:, None]
        n_rows = labels.shape[0]

        patient_ids = all_predictions.get(f"{prefix}_patientid", None)
        if patient_ids is not None:
            patient_ids = np.asarray(patient_ids)
            if len(patient_ids) != n_rows:
                print(f"    [WARN] {prefix}: patient ids ({len(patient_ids)}) "
                      f"misaligned with scored rows ({n_rows}); using row indices")
                patient_ids = None
        if patient_ids is None:
            patient_ids = np.arange(n_rows)

        probs_cal = all_predictions.get(f"{prefix}_probs_cal", None)
        extras = {}
        for suffix, col in extras_map.items():
            arr = all_predictions.get(f"{prefix}_{suffix}", None)
            if arr is not None and len(np.asarray(arr)) == n_rows:
                extras[col] = np.asarray(arr)

        # Map each matrix column to its (task, horizon) pair. Multi-task runs
        # with heterogeneous horizons pass explicit `head_pairs` from run_metadata.json.
        n_h = len(horizons)
        if head_pairs:
            if labels.shape[1] != len(head_pairs):
                print(f"    [WARN] {prefix}: {labels.shape[1]} label columns vs "
                      f"{len(head_pairs)} head_pairs expected; slicing to min")
            col_map = [(str(t), int(h)) for t, h in head_pairs[:labels.shape[1]]]
        elif task_keys:
            if labels.shape[1] != len(task_keys) * n_h:
                print(f"    [WARN] {prefix}: {labels.shape[1]} label columns but "
                      f"{len(task_keys)} tasks x {n_h} horizons expected; skipping extras")
            col_map = [(task_keys[c // n_h], horizons[c % n_h])
                       for c in range(min(labels.shape[1], len(task_keys) * n_h))]
        else:
            col_map = [(None, horizons[c]) for c in range(min(labels.shape[1], n_h))]

        for c, (task_name, horizon_val) in enumerate(col_map):
            lab_col = labels[:, c]
            prob_col = probs[:, c] if probs.shape[1] > c else probs[:, -1]
            valid = lab_col != -100  # drop rows with unknown label at this horizon
            if valid.sum() == 0:
                continue
            n_valid = int(valid.sum())
            row = {
                "patient_id": patient_ids[valid],
                "fup": np.full(n_valid, fup_val, dtype=np.int32),
                "horizon": np.full(n_valid, int(horizon_val), dtype=np.int32),
                "label": lab_col[valid].astype(np.int8),
                "prob": prob_col[valid].astype(np.float32),
            }
            if task_name is not None:
                row["task"] = np.full(n_valid, task_name, dtype=object)
            if probs_cal is not None:
                probs_cal = np.asarray(probs_cal)
                cal_col = (probs_cal[:, c] if probs_cal.ndim > 1 and probs_cal.shape[1] > c
                           else probs_cal.reshape(-1))
                row["prob_cal"] = np.asarray(cal_col)[valid].astype(np.float32)
            for col, arr in extras.items():
                if arr.ndim == 2 and task_keys and arr.shape[1] == len(task_keys):
                    row[col] = arr[valid, c // n_h]
                elif arr.ndim == 1:
                    row[col] = arr[valid]
            frames.append(pd.DataFrame(row))

    if not frames:
        print(f"    [WARN] No per-FUP prediction arrays found for split '{split_label}'; "
              f"tidy table not written.")
        return None

    table = pd.concat(frames, ignore_index=True)
    try:  # patient ids are numeric strings in this dataset -> store as int
        table["patient_id"] = table["patient_id"].astype(np.int64)
    except (TypeError, ValueError):
        pass  # keep original dtype if not parseable as int
    sort_cols = [c for c in ["patient_id", "fup", "task", "horizon"] if c in table.columns]
    table = table.sort_values(sort_cols).reset_index(drop=True)

    if csv_name:
        table.to_csv(output_dir / csv_name, index=False, float_format="%.6f")
        print(f"    [SUCCESS] Saved tidy predictions CSV: {output_dir / csv_name} "
              f"({len(table)} rows, columns: {', '.join(table.columns)})")
    return table


def _resolve_best_checkpoint_in_run_dir(run_dir: Path) -> Path | None:
    """Return the best checkpoint inside `run_dir` (honoring `trainer_state.json` if present)."""
    return resolve_best_checkpoint(run_dir)


def resolve_base_dir_for_ckpt(base_root: Path, split: str, run_id: str, val_tag: str) -> tuple[Path, str]:
    """Resolve `<base_root>/<split>/<run_id>_<val_tag>`, falling back cleanly if needed."""
    resolved_id = resolve_run_id_with_val_tag(run_id, val_tag) if run_id else ""
    candidates = ([resolved_id, run_id] if run_id else []) + [f"e00-a15-v60_{val_tag}", "e00-a15-v60"]
    seen = set()
    for cand_id in candidates:
        if not cand_id or cand_id in seen:
            continue
        seen.add(cand_id)
        cand_dir = Path(base_root) / split / cand_id
        if cand_dir.exists():
            return cand_dir, cand_id
    fallback_id = resolved_id or f"e00-a15-v60_{val_tag}"
    return Path(base_root) / split / fallback_id, fallback_id


def find_best_checkpoint(base_dir: Path, task_key: str, horizon: int) -> Path:
    """Locates the best single-task model checkpoint inside structured execution logs."""
    task_dir = base_dir / "finetuning" / task_key
    if not task_dir.exists():
        raise FileNotFoundError(f"Task directory missing: {task_dir}")
    
    h_str = f"{horizon:04d}"
    pattern = re.compile(rf"hrz\(([^)]*\b{h_str}\b[^)]*)\)")
    all_cands = [p for p in task_dir.iterdir() if p.is_dir() and pattern.search(p.name)]
    fut_all_cands = [p for p in all_cands if "_fut(" not in p.name or "_fut(all)_" in p.name or "_fut(random)_" in p.name]
    candidates = fut_all_cands or all_cands
    if not candidates:
        raise FileNotFoundError(f"No run found matching horizon configuration {h_str} inside {task_dir}")
    
    resolved = []
    for run_dir in candidates:
        ckpt = _resolve_best_checkpoint_in_run_dir(run_dir)
        if ckpt is not None:
            resolved.append(ckpt)
    if not resolved:
        raise FileNotFoundError(f"No valid checkpoints discovered inside structural path: {task_dir}")
    
    return max(resolved, key=lambda p: p.stat().st_mtime_ns)


def extract_horizons_from_path(checkpoint_path: Path) -> list[int]:
    """Extracts targeted validation prediction horizons from a checkpoint parent directory name."""
    run_dir = checkpoint_path
    while "hrz(" not in run_dir.name and run_dir.parent != run_dir:
        run_dir = run_dir.parent
    
    match = re.search(r"hrz\(([\d-]+)\)", run_dir.name)
    if not match:
        raise ValueError(f"Could not parse horizon keys from path layout string: {run_dir.name}")
    
    return [int(h) for h in match.group(1).split("-")] 


def load_run_metadata(run_dir: Path) -> dict | None:
    """Read the run_metadata.json written by train_classification.py, if present.

    Contains {"task_key", "tasks", "horizons", "label_keys"} describing the head
    layout of the run (one classifier head per (task, horizon) pair, task-major).
    """
    meta_path = Path(run_dir) / "run_metadata.json"
    if meta_path.exists():
        return json.loads(meta_path.read_text())
    return None


def find_best_checkpoint_multitask(
    base_dir: Path, task_key: str, horizon: int | None, scope: str = "any"
) -> tuple[Path, dict]:
    """Locate a checkpoint of a MULTI-TASK run that contains heads for `task_key`.

    `scope`:
      - "domain": category-grouped runs (`multitask_bacteria`, `multitask_viruses`, `multitask_fungi`, + standalone `death`/`graft_loss`)
      - "unified": `multitask_unified`
      - "pan": `multitask_all`
      - "any": match any multi-task run containing `task_key`
    """
    base_dir = Path(base_dir)
    ft_root = base_dir / "finetuning"
    if not ft_root.exists():
        raise FileNotFoundError(f"Task directory missing: {ft_root}")

    h_str = f"{horizon:04d}" if horizon is not None else None
    pattern = re.compile(rf"hrz\(([^)]*\b{h_str}\b[^)]*)\)") if h_str is not None else re.compile(r"hrz\([^)]+\)")
    matches: list[tuple[Path, dict]] = []
    for run_dir in sorted(p for p in ft_root.iterdir() if p.is_dir()):
        if scope == "pan" and run_dir.name != "multitask_all":
            continue
        if scope == "unified" and run_dir.name != "multitask_unified":
            continue
        if scope == "domain" and run_dir.name in (
            "multitask_all", "multitask_unified", "multitask_pan_clinical",
            "multitask_infections", "multitask_longterm",
            "multitask_acute_13", "multitask_longterm_13",
            "multitask_infections_11", "multitask_outcomes_2",
        ):
            continue
        hrz_dirs = sorted(p for p in run_dir.iterdir() if p.is_dir() and pattern.search(p.name))
        fut_all_hrz = [p for p in hrz_dirs if "_fut(" not in p.name or "_fut(all)_" in p.name or "_fut(random)_" in p.name]
        for hrz_dir in (fut_all_hrz or hrz_dirs):
            meta = load_run_metadata(hrz_dir)
            tasks_in_run = meta.get("tasks", []) if meta else []
            if scope == "domain" and task_key in ("death", "graft_loss") and tasks_in_run == [task_key]:
                pass
            elif len(tasks_in_run) <= 1 or task_key not in tasks_in_run:
                continue
            if horizon is not None and meta and meta.get("head_pairs"):
                if [task_key, int(horizon)] not in [[str(t), int(h)] for t, h in meta["head_pairs"]]:
                    continue
            ckpt = _resolve_best_checkpoint_in_run_dir(hrz_dir)
            if ckpt is not None:
                matches.append((ckpt, meta))
    if matches:
        return max(matches, key=lambda item: item[0].stat().st_mtime_ns)
    raise FileNotFoundError(
        f"No multi-task run (scope='{scope}') with heads for task '{task_key}' (horizon {h_str or 'any'}) under {ft_root}"
    )


def vectorized_bootstrap_counts(boot_indices: np.ndarray, n_samples: int) -> np.ndarray:
    """Convert a (B, N) integer bootstrap index array into a (B, N) float64 sample-multiplicity matrix."""
    n_boot = boot_indices.shape[0]
    flat = np.arange(n_boot, dtype=np.int64)[:, None] * n_samples + boot_indices
    return np.bincount(flat.ravel(), minlength=n_boot * n_samples).reshape(n_boot, n_samples).astype(np.float64)


def vectorized_bootstrap_ap_and_roc(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    counts: np.ndarray,
    valid_boot_mask: np.ndarray | None = None,
    compute_roc: bool = False,
    chunk_size: int = 250,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Compute exact bootstrap Average Precision (PR-AUC) and optionally ROC-AUC
    by sorting `y_prob` once and accumulating bootstrap sample multiplicities.

    Matches `sklearn.metrics.average_precision_score(y_true[idx], y_prob[idx])`
    and `sklearn.metrics.roc_auc_score(y_true[idx], y_prob[idx])` to float64
    machine precision (~1e-16).
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_prob = np.asarray(y_prob)
    if valid_boot_mask is not None:
        counts = counts[valid_boot_mask]
    n_boot = counts.shape[0]
    if n_boot == 0:
        return np.array([], dtype=np.float64), (np.array([], dtype=np.float64) if compute_roc else None)

    desc = np.argsort(y_prob, kind="mergesort")[::-1]
    p_s = y_prob[desc]
    y_s = y_true[desc]
    distinct_mask = np.r_[np.diff(p_s) != 0, True]
    grp_starts = np.r_[0, np.flatnonzero(distinct_mask[:-1]) + 1]

    ap_out = np.empty(n_boot, dtype=np.float64)
    roc_out = np.empty(n_boot, dtype=np.float64) if compute_roc else None

    for start in range(0, n_boot, chunk_size):
        end = min(start + chunk_size, n_boot)
        W = counts[start:end][:, desc]
        W_pos = np.add.reduceat(W * y_s[None, :], grp_starts, axis=1)
        W_tot = np.add.reduceat(W, grp_starts, axis=1)

        tps = np.cumsum(W_pos, axis=1)
        tot = np.cumsum(W_tot, axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            prec = np.where(tot > 0, tps / tot, 0.0)
            ap_out[start:end] = np.sum(prec * W_pos, axis=1) / tps[:, -1]
            if compute_roc:
                fps = tot - tps
                dfp = W_tot - W_pos
                prev_tps = tps - W_pos
                auc_area = np.sum(dfp * (tps + prev_tps) * 0.5, axis=1)
                roc_out[start:end] = auc_area / (tps[:, -1] * fps[:, -1])

    return ap_out, roc_out


def build_patient_cluster_bootstrap_counts(
    patient_ids: np.ndarray | None,
    n_samples: int,
    n_bootstraps: int,
    seed: int = 42,
) -> np.ndarray:
    """Draw (B, N) sample-multiplicity matrix by resampling unique patient clusters with replacement.

    Properly preserves longitudinal correlation across repeated follow-up visits (FUPs)
    and across multiple prediction horizons of the same patient. Falls back to i.i.d.
    row resampling only if `patient_ids` is None.
    """
    rng = np.random.default_rng(seed)
    if patient_ids is None or len(patient_ids) != n_samples:
        boot_indices = rng.integers(0, n_samples, size=(n_bootstraps, n_samples))
        return vectorized_bootstrap_counts(boot_indices, n_samples)
    _, inv_idx = np.unique(np.asarray(patient_ids, dtype=str), return_inverse=True)
    n_clusters = int(inv_idx.max()) + 1 if n_samples > 0 else 0
    if n_clusters <= 1:
        boot_indices = rng.integers(0, n_samples, size=(n_bootstraps, n_samples))
        return vectorized_bootstrap_counts(boot_indices, n_samples)
    cluster_boot = rng.integers(0, n_clusters, size=(n_bootstraps, n_clusters))
    cluster_counts = vectorized_bootstrap_counts(cluster_boot, n_clusters)
    return cluster_counts[:, inv_idx]


def clustered_mcnemar_pvalue(
    correct_a: np.ndarray,
    correct_b: np.ndarray,
    patient_ids: np.ndarray | None = None,
) -> float:
    """Two-sided matched-pair test accounting for within-patient clustering (Durkalski et al., Stat Med 2003).

    When each patient contributes at most 1 observation, uses the exact binomial McNemar test.
    When patients contribute repeated longitudinal / multi-horizon frames, aggregates within-patient
    discordant differences d_g = n_{10,g} - n_{01,g} and evaluates Z^2 = (sum_g d_g)^2 / sum_g d_g^2 ~ chi2_1.
    """
    from scipy.stats import binomtest, chi2

    ca = np.asarray(correct_a, dtype=bool)
    cb = np.asarray(correct_b, dtype=bool)
    n10_vec = (ca & ~cb).astype(np.int64)
    n01_vec = (~ca & cb).astype(np.int64)
    n10 = int(n10_vec.sum())
    n01 = int(n01_vec.sum())
    if n10 + n01 == 0:
        return 1.0

    if patient_ids is not None and len(patient_ids) == len(ca):
        _, inv_idx = np.unique(np.asarray(patient_ids, dtype=str), return_inverse=True)
        n_clusters = int(inv_idx.max()) + 1
        if n_clusters < len(ca):
            d_g = np.bincount(inv_idx, weights=(n10_vec - n01_vec), minlength=n_clusters)
            denom = float(np.sum(d_g * d_g))
            if denom <= 0.0:
                return 1.0
            num = float(np.sum(d_g)) ** 2
            return float(chi2.sf(num / denom, df=1))

    return float(binomtest(min(n10, n01), n10 + n01, 0.5).pvalue)


def paired_bootstrap_pr_auc_pvalue(
    y_true: np.ndarray,
    p_tf: np.ndarray,
    p_base: np.ndarray,
    n_bootstraps: int = 1000,
    seed: int = 42,
    max_workers: int = 4,
    patient_ids: np.ndarray | None = None,
    two_sided: bool = True,
) -> float:
    """
    Computes a paired patient-cluster bootstrap p-value comparing PR-AUC between
    two models on the exact same test cohort. Defaults to a symmetric two-sided test
    consistent with McNemar winner assignment.
    """
    if len(np.unique(y_true)) < 2:
        return np.nan

    n_samples = len(y_true)
    counts = build_patient_cluster_bootstrap_counts(
        patient_ids=patient_ids, n_samples=n_samples, n_bootstraps=n_bootstraps, seed=seed,
    )

    y_f = np.asarray(y_true, dtype=np.float64)
    pos_counts = counts @ y_f
    tot_counts = np.sum(counts, axis=1)
    valid_boot = (pos_counts > 0) & (pos_counts < tot_counts)
    if not np.any(valid_boot):
        return np.nan

    ap_tf, _ = vectorized_bootstrap_ap_and_roc(y_f, p_tf, counts, valid_boot_mask=valid_boot, compute_roc=False)
    ap_base, _ = vectorized_bootstrap_ap_and_roc(y_f, p_base, counts, valid_boot_mask=valid_boot, compute_roc=False)
    diffs = ap_tf - ap_base

    if two_sided:
        n_b = len(diffs)
        p_le = (np.sum(diffs <= 0) + 1.0) / (n_b + 1.0)
        p_ge = (np.sum(diffs >= 0) + 1.0) / (n_b + 1.0)
        return float(min(1.0, 2.0 * min(p_le, p_ge)))
    return float(np.mean(diffs <= 0))


def calibrate_array_pair(
    y_val_true: np.ndarray, 
    y_val_prob: np.ndarray, 
    y_test_prob: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fits monotone Platt scaling on valid binary (`0`/`1`) validation outputs and transforms
    both validation and test probabilities:
      - Masks out `-100` censored entries before fitting.
      - Requires `n_pos >= 20` and a strictly positive fitted slope (`coef > 0.05`) for
        2-parameter Platt scaling; otherwise falls back to intercept-only prior calibration
        (`1.0 * logit(p) + b`), aligning mean predicted risk to validation prevalence while
        strictly preserving prediction rank order (`AUROC` and `PR-AUC` 100% invariant).
    """
    from scipy.special import expit

    y_val_arr = np.asarray(y_val_true)
    valid = (y_val_arr == 0) | (y_val_arr == 1)
    if not np.any(valid):
        return y_val_prob, y_test_prob
    y_bin = y_val_arr[valid].astype(np.int64)
    if len(np.unique(y_bin)) < 2:
        return y_val_prob, y_test_prob

    val_all_logits = logit(np.clip(np.asarray(y_val_prob, dtype=np.float64), 1e-7, 1.0 - 1e-7))
    test_all_logits = logit(np.clip(np.asarray(y_test_prob, dtype=np.float64), 1e-7, 1.0 - 1e-7))
    val_fit_logits = val_all_logits[valid]

    n_pos = int(np.sum(y_bin == 1))
    slope, intercept = 1.0, 0.0
    fitted = False
    if n_pos >= 20:
        calibrator = LogisticRegression(C=1.0, solver="lbfgs", random_state=42)
        calibrator.fit(val_fit_logits.reshape(-1, 1), y_bin)
        if hasattr(calibrator, "coef_") and float(calibrator.coef_[0, 0]) > 0.05:
            slope = float(calibrator.coef_[0, 0])
            intercept = float(calibrator.intercept_[0])
            fitted = True
    if not fitted:
        p_bar = float(np.clip(np.mean(y_bin), 1e-6, 1.0 - 1e-6))
        slope = 1.0
        intercept = float(logit(p_bar) - np.mean(val_fit_logits))

    cal_val = expit(slope * val_all_logits + intercept)
    cal_test = expit(slope * test_all_logits + intercept)
    return cal_val, cal_test


def calibrate_dataframe_pair(
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    prob_col: str = "y_prob",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convenience wrapper around calibrate_array_pair for pandas DataFrames.
    """
    if df_val.empty or df_test.empty:
        return df_val, df_test

    df_val_out = df_val.copy()
    df_test_out = df_test.copy()

    cal_val, cal_test = calibrate_array_pair(
        y_val_true=df_val["y_true"].values,
        y_val_prob=df_val[prob_col].values,
        y_test_prob=df_test[prob_col].values,
    )

    df_val_out[prob_col] = cal_val
    df_test_out[prob_col] = cal_test

    return df_val_out, df_test_out


# =====================================================================
# 5. UNIFIED MODEL REGISTRY & PREDICTION PATH RESOLUTION
# =====================================================================

MODEL_REGISTRY: dict[str, dict[str, str]] = {
    # Single-task Discriminative Transformer
    "tf_raw":               {"name": "Transformer (Raw)",            "type": "transformer",     "val": "raw"},
    "tf_raw_st":            {"name": "Transformer (Raw)",            "type": "transformer",     "val": "raw"},
    "tf_bin":               {"name": "Transformer (Bin)",            "type": "transformer",     "val": "bin"},
    "tf_bin_st":            {"name": "Transformer (Bin)",            "type": "transformer",     "val": "bin"},
    # Category-grouped Multi-task Discriminative Transformer (Canonical B_CAT_MT)
    "tf_raw_mt":            {"name": "Transformer Category-MT (Raw)","type": "transformer_mt",  "val": "raw", "mt_mode": "category"},
    "tf_bin_mt":            {"name": "Transformer Category-MT (Bin)","type": "transformer_mt",  "val": "bin", "mt_mode": "category"},
    # Unified & Timescale-Separated Multi-task Discriminative Transformer Variants
    "tf_raw_unified":       {"name": "Transformer Unified-MT (Raw)", "type": "transformer_mt",  "val": "raw", "mt_mode": "unified"},
    "tf_bin_unified":       {"name": "Transformer Unified-MT (Bin)", "type": "transformer_mt",  "val": "bin", "mt_mode": "unified"},
    "tf_raw_bv":            {"name": "Transformer Acute B+V (Raw)",  "type": "transformer_mt",  "val": "raw", "mt_mode": "multitask_bv"},
    "tf_raw_inf":           {"name": "Transformer Acute-11 (Raw)",   "type": "transformer_mt",  "val": "raw", "mt_mode": "multitask_infections"},
    "tf_raw_pan13":         {"name": "Transformer Pan-13 (Raw)",     "type": "transformer_mt",  "val": "raw", "mt_mode": "multitask_pan_13"},
    # Generative Causal Transformer (single-task + multi-task variants)
    "tfg_raw":              {"name": "Gen-Transformer (Raw)",        "type": "transformer_gen", "val": "raw"},
    "tfg_raw_st":           {"name": "Gen-Transformer (Raw)",        "type": "transformer_gen", "val": "raw"},
    "tfg_bin":              {"name": "Gen-Transformer (Bin)",        "type": "transformer_gen", "val": "bin"},
    "tfg_bin_st":           {"name": "Gen-Transformer (Bin)",        "type": "transformer_gen", "val": "bin"},
    "tfg_raw_mt":           {"name": "Gen-Transformer Cat-MT (Raw)", "type": "transformer_gen", "val": "raw", "mt_mode": "category"},
    "tfg_bin_mt":           {"name": "Gen-Transformer Cat-MT (Bin)", "type": "transformer_gen", "val": "bin", "mt_mode": "category"},
    "tfg_raw_unified":      {"name": "Gen-Transformer Unified (Raw)","type": "transformer_gen", "val": "raw", "mt_mode": "unified"},
    "tfg_bin_unified":      {"name": "Gen-Transformer Unified (Bin)","type": "transformer_gen", "val": "bin", "mt_mode": "unified"},
    # Classic ML Baselines
    "xgboost":              {"name": "XGBoost",                      "type": "classic_ml",      "algo": "xgboost"},
    "xgb_raw":              {"name": "XGBoost",                      "type": "classic_ml",      "algo": "xgboost"},
    "random_forest":        {"name": "Random Forest",                "type": "classic_ml",      "algo": "random_forest"},
    "rf_raw":               {"name": "Random Forest",                "type": "classic_ml",      "algo": "random_forest"},
    "logistic_regression":  {"name": "Logistic Regression",          "type": "classic_ml",      "algo": "logistic_regression"},
    "lr_raw":               {"name": "Logistic Regression",          "type": "classic_ml",      "algo": "logistic_regression"},
    # Optuna-tuned Best Model
    "opt_raw":              {"name": "Optuna Best (Raw)",            "type": "optuna",          "val": "raw"},
}

TRANSFORMER_MODEL_SPECS: dict[str, dict[str, str]] = {
    k: v for k, v in MODEL_REGISTRY.items()
    if v["type"] in ("transformer", "transformer_mt", "transformer_gen", "optuna")
}


def resolve_registry_entry(model_key: str) -> dict[str, str]:
    """Resolves static or dynamic ``opt_*`` keys against :data:`MODEL_REGISTRY`."""
    if model_key in MODEL_REGISTRY:
        return MODEL_REGISTRY[model_key]
    if model_key.startswith("opt_"):
        opt_suffix = model_key[len("opt_"):]
        val = "bin" if opt_suffix.endswith("bin") else "raw"
        return {
            "name": f"Optuna ({opt_suffix})",
            "type": "optuna",
            "val": val,
            "opt_task_override": None if opt_suffix in ("raw", "bin") else opt_suffix,
        }
    raise KeyError(f"Model key '{model_key}' not recognized in MODEL_REGISTRY.")


def _category_mt_folder_for_task(task_key: str) -> str:
    """Map a canonical task key to its category-grouped multi-task folder name."""
    meta = get_task_meta(task_key)
    grp = meta.get("group", "")
    if grp == "Bacterial Infections":
        return "multitask_bacteria"
    if grp == "Viral Infections":
        return "multitask_viruses"
    if grp == "Fungal Infections":
        return "multitask_fungi"
    return normalize_task_key(task_key)


def resolve_predictions_csv(
    *,
    results_dir: Path,
    model_key: str,
    split_type: str,
    task: str,
    horizon: int | None,
    split_subset: str = "test",
    tf_pt_config: str = "e00-a15-v60",
    tfg_pt_config: str = "e00-a15-v60",
) -> Path | None:
    """Locates ``{split_subset}_predictions.csv`` for any model in :data:`MODEL_REGISTRY`."""
    info = resolve_registry_entry(model_key)
    m_type = info["type"]
    m_val = info.get("val", "raw")
    csv_filename = f"{split_subset}_predictions.csv"
    norm_task = task[len("infection_"):] if task.startswith("infection_") else task
    full_task = normalize_task_key(task)
    if full_task not in ("death", "graft_loss", "infection") and not full_task.startswith("infection_"):
        full_task = f"infection_{full_task}"

    def _match_horizon_dir(parent_dir: Path) -> Path | None:
        if not parent_dir.exists():
            return None
        all_cands = sorted(
            [d for d in parent_dir.glob("hrz(*)*") if d.is_dir()],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        fut_all = [d for d in all_cands if "_fut(" not in d.name or "_fut(all)_" in d.name or "_fut(random)_" in d.name]
        candidates = fut_all or all_cands
        if horizon is None:
            for d in candidates:
                f = d / csv_filename
                if f.exists():
                    return f
            return None
        tag = f"{horizon:04d}"
        for d in candidates:
            m = re.match(r"hrz\(([0-9\-]+)\)", d.name)
            if m and tag in m.group(1).split("-") and (d / csv_filename).exists():
                return d / csv_filename
        return None

    if m_type == "classic_ml":
        algo = info["algo"]
        for algo_dir in (algo, f"{algo}_raw", f"{algo}_bin"):
            for t_cand in (full_task, norm_task, task):
                found = _match_horizon_dir(results_dir / "classic_ml" / split_type / algo_dir / t_cand)
                if found is not None:
                    return found
        return None

    if m_type == "optuna":
        opt_override = info.get("opt_task_override")
        candidates = (
            [opt_override]
            if opt_override
            else [_category_mt_folder_for_task(full_task), "multitask_unified", full_task, norm_task]
        )
        for cand in candidates:
            f = results_dir / "optuna" / split_type / cand / "best_model" / csv_filename
            if f.exists():
                return f
        return None

    is_gen = (m_type == "transformer_gen")
    root_sub = "generative_transformer" if is_gen else "transformer"
    pt_cfg = tfg_pt_config if is_gen else tf_pt_config
    clean_pt = re.sub(r"_(raw|bin)$", "", pt_cfg)
    ft_root = results_dir / root_sub / split_type / f"{clean_pt}_{m_val}" / "finetuning"

    mt_mode = info.get("mt_mode")
    if mt_mode == "unified":
        folder_candidates = ["multitask_unified"]
    elif mt_mode in ("multitask_bv", "multitask_infections", "multitask_pan_13"):
        folder_candidates = [mt_mode]
    elif mt_mode == "category":
        folder_candidates = [_category_mt_folder_for_task(full_task), full_task, norm_task]
    else:
        folder_candidates = [full_task, norm_task]

    for folder in folder_candidates:
        found = _match_horizon_dir(ft_root / folder)
        if found is not None:
            return found
    return None
