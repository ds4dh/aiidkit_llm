"""
Survival label adapter.

The preprocessing repository (aiidkit_mhmmdrz) computes, for every fup snapshot,
two columns per endpoint outcome in `PatientTrajectory.compute_survival_labels()`:

    label_{outcome}_time    -> time-to-event (days since the fup observation end)
                               or time-to-censoring, whichever happens first
    label_{outcome}_status  -> 1 if the event was observed, 0 if right-censored

Available endpoints directly present in the raw datasets:
    infection, death, graft_loss, grp_{pathogen_group}

Category-level endpoints (bacteria / virus / fungi) are stored in the raw
datasets as of preprocessing v3.9 (compute_survival_labels adds
'infection_{cat}' endpoints), so for v3.9+ data the pooling fallback below is
DORMANT: every call site checks for the stored columns first, and this module
only pools when they are genuinely absent (it then prints a WARN).

Do NOT treat the fallback as equivalent to the stored columns: measured on
v3.9 temporal_split (fup 0000/0030/0360, train+test) it disagrees with the
stored category labels on ~0.1-1.5% of rows and recovers ~3% fewer events,
because the preprocessing repo also counts bacterial/viral/fungal episodes that
carry no pathogen-group assignment. Pooling rule (for reference):
    - event time   = min(group event times) over groups with status == 1
    - censored time= min(group censoring times) if no group event exists

Column naming (verified against the v3.9 raw datasets):
    survival, per endpoint : label_{outcome}_status / label_{outcome}_time
    survival, per group    : label_grp_{group}_status / label_grp_{group}_time
    binary, per horizon    : label_{outcome}_{H:04d}d  and
                             label_infection_grp_{group}_{H:04d}d
    -> a group task key is 'grp_{group}' for survival but 'infection_grp_{group}'
       for the binary pipeline (there is no label_grp_{group}_{H}d column).

Group ids stored in v3.9 (21):
    Bacteria: anaerobic_bacteria, clostridium, enterobacterales, gn_nonfermenters,
              gram_positive, other_bacteria, other_gn_bacteria, other_opportunistic
    Fungi   : aspergillus, candida, other_fungi, pneumocystis
    Virus   : bkpyv, cmv, enteric_viruses, hepatitis_viruses, herpes_viruses,
              influenza, other_resp_viruses, other_viruses, sars_cov_2

This module also provides helpers to binarize survival labels at a fixed
horizon H (to compare with the binary classification models) and to build
row keep-masks (rows with unknown or zero-information survival labels).
"""

import numpy as np
from datasets import Dataset

# Special mask value used everywhere in the pipeline for "unknown" labels
MASK = -100

# Prefix mapping from the preprocessing repo (src/constants.py, GROUP_TO_CATEGORY).
# NOTE: v3.9+ raw datasets store category-level survival columns directly; this
# mapping is only used by the legacy fallback derivation below (never for v3.9 data).
_BACTERIA_PREFIXES = [
    "enterobacterales", "gn_nonfermenters", "gram_positive",
    "clostridium", "other_gn", "anaerobic", "other_opp", "other_bac",
]
_FUNGI_PREFIXES = ["aspergillus", "candida", "other_fungi", "pneumocystis"]

# Task keys (same naming as the binary configs) -> survival category name
CATEGORY_TASKS = {
    "infection_bacteria": "Bacteria",
    "infection_virus": "Virus",
    "infection_fungi": "Fungi",
}


def group_to_category(group_id: str) -> str:
    """Maps a pathogen group id (e.g. 'cmv', 'enterobacterales') to a category."""
    if any(group_id.startswith(p) for p in _BACTERIA_PREFIXES):
        return "Bacteria"
    if any(group_id.startswith(p) for p in _FUNGI_PREFIXES):
        return "Fungi"
    return "Virus"


def get_group_column_names(dataset_columns, category: str | None = None) -> list[str]:
    """
    Returns all group-level survival status columns (label_grp_*_status) available
    in a dataset, optionally restricted to a single category (e.g. 'Virus').
    """
    cols = [c for c in dataset_columns if c.startswith("label_grp_") and c.endswith("_status")]
    if category is None:
        return cols
    out = []
    for col in cols:
        group_id = col[len("label_grp_"): -len("_status")]
        if group_to_category(group_id) == category.lower() or group_to_category(group_id) == category:
            out.append(col)
    return out


def resolve_survival_label_keys(task_key: str) -> tuple[str, str, bool]:
    """
    Resolves a prediction task key into survival label column names.

    Accepts both survival group keys ('grp_enterobacterales') and canonical
    binary group keys ('infection_grp_enterobacterales'), mapping both to the
    stored v3.9 `label_grp_{group}_{time,status}` columns.
    """
    category = CATEGORY_TASKS.get(task_key)
    if category is not None:
        time_key = f"label_{task_key}_time"
        status_key = f"label_{task_key}_status"
        # Category endpoint: stored columns win, pooling is a pre-v3.9 fallback only
        return time_key, status_key, True
    raw_key = task_key.removeprefix("infection_") if task_key.startswith("infection_grp_") else task_key
    return f"label_{raw_key}_time", f"label_{raw_key}_status", False


def derive_category_survival_labels(
    dataset: Dataset,
    task_key: str,
) -> Dataset:
    """
    Adds `label_{task_key}_time` / `label_{task_key}_status` columns to a dataset
    by pooling pathogen-group survival labels of one category.

    Pooling logic (an approximation of what the preprocessing repo computes; see
    the module docstring for the measured difference w.r.t. the stored v3.9 labels):
        - any group event  -> status 1, time = min event time across event groups
        - no group event   -> status 0, time = min censored time across groups
        - all groups unknown (-100) -> status/time = -100
    """
    _, status_key, is_derived = resolve_survival_label_keys(task_key)
    if not is_derived or status_key in dataset.column_names:
        return dataset  # nothing to derive (endpoint already stored)

    category = CATEGORY_TASKS[task_key]
    group_status_cols = get_group_column_names(dataset.column_names, category)
    group_time_cols = [c[: -len("_status")] + "_time" for c in group_status_cols]
    if not group_status_cols:
        raise ValueError(
            f"Cannot derive survival labels for '{task_key}': no group columns "
            f"of category '{category}' were found in the dataset."
        )
    print(
        f"[WARN] legacy fallback: pooled {len(group_status_cols)} '{category}' "
        f"group labels into {status_key} because this dataset has no stored "
        f"{status_key} column (pre-v3.9 data?). Regenerate with preprocessing "
        f"v3.9+ to train on the authoritative labels instead."
    )

    # Stack group labels: [n_rows, n_groups]
    times = np.stack([np.asarray(dataset[c], dtype=np.int64) for c in group_time_cols], axis=1)
    statuses = np.stack([np.asarray(dataset[c], dtype=np.int64) for c in group_status_cols], axis=1)

    out_time = np.full(times.shape[0], MASK, dtype=np.int64)
    out_status = np.full(times.shape[0], MASK, dtype=np.int64)

    # Rows with at least one observed group event -> earliest group event
    event_mask = statuses == 1
    has_event = event_mask.any(axis=1)
    if has_event.any():
        event_times = np.where(event_mask, times, np.iinfo(np.int64).max)
        out_time[has_event] = event_times[has_event].min(axis=1)
        out_status[has_event] = 1

    # Rows with no group event but at least one known censored group -> min censor time
    no_event = ~has_event
    censor_known = no_event & (statuses == 0).any(axis=1)
    if censor_known.any():
        censor_times = np.where(statuses == 0, times, np.iinfo(np.int64).max)
        out_time[censor_known] = censor_times[censor_known].min(axis=1)
        out_status[censor_known] = 0

    dataset = dataset.add_column(resolve_survival_label_keys(task_key)[0], out_time.tolist())
    dataset = dataset.add_column(status_key, out_status.tolist())
    return dataset


def survival_keep_mask(times: np.ndarray, statuses: np.ndarray) -> np.ndarray:
    """
    Rows usable for survival training/evaluation:
        - labels must be known (not -100)
        - rows censored at t0 (status 0, duration 0) carry no information
    """
    times = np.asarray(times)
    statuses = np.asarray(statuses)
    known = (times != MASK) & (statuses != MASK) & (times >= 0) & ((statuses == 0) | (statuses == 1))
    informative = (statuses == 1) | (times > 0)
    return known & informative


def binarize_survival_labels(
    times: np.ndarray,
    statuses: np.ndarray,
    horizon: int,
) -> np.ndarray:
    """
    Converts (time, status) survival labels into a binary 'event within
    horizon days' target comparable with the binary classification labels:

        1    : observed event before or at the horizon
        0    : known event-free at the horizon (censored after it, or event after it)
        -100 : right-censored before or at the horizon (unknown outcome)
    """
    times = np.asarray(times, dtype=np.float64)
    statuses = np.asarray(statuses, dtype=np.float64)
    labels = np.full(times.shape, MASK, dtype=np.int64)
    labels[(times > horizon)] = 0
    labels[(statuses == 1) & (times <= horizon)] = 1
    # Propagate unknown labels (invalid/negative times or statuses)
    labels[(times < 0) | (statuses < 0)] = MASK
    return labels
