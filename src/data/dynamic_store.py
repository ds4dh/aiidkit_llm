"""
Standalone Portable Dynamic Landmark Stores for AIIDKIT (v4.0).

Loads single-copy, dependency-free v4.0 artifacts produced by ``stcs_fastehr``
(standard JSON + Parquet + compressed NumPy ``.npz`` — zero pickle coupling):
  1. ``PortableDynamicTEAVStore`` + ``PortableDynamicTEAVDataset``:
       - ``mode='pretraining'``: full trajectories up to temporal cutoff.
       - ``mode='fixed_grid'``: deterministic evaluation over FUP milestones.
       - ``mode='random_landmark'``: follow-up-proportional continuous daily
         landmarks with deterministic per-epoch resampling (``set_epoch``),
         global-salience middle truncation, and vectorized undersampling /
         survival Horvitz-Thompson IPW.
  2. ``PortableDynamicClassicMLStore``:
       - Materializes 916-column v4.0 Classic ML DataFrames in RAM for either
         ``mode='random_landmark'`` (training) or ``mode='fixed_grid'`` (val/test).
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from torch.utils.data import Dataset as TorchDataset

KEY_STRIDE = np.int64(1_000_000)
DAY_OFFSET = np.int64(500_000)
INF_DAY = np.int64(10**9)
NO_LIMIT_SENTINEL = np.int64(10**9)
BIN_LABELS = ["Lowest", "Lower", "Low", "Middle", "High", "Higher", "Highest"]
BINARY_PREFIXES = (
    "Med_", "MedCat_", "Inf_", "Rej_", "RejTreat_", "Comorb_", "OrgEv_",
    "Hosp_inpatient_", "Immu_AntiHLA_", "Immu_cm_", "Immu_t_", "Immu_b_", "SubUse_",
)

_TEAV_CACHE: Dict[str, "PortableDynamicTEAVStore"] = {}
_CML_CACHE: Dict[str, "PortableDynamicClassicMLStore"] = {}


def is_dynamic_teav_dir(path: str | Path) -> bool:
    p = Path(path)
    return p.is_dir() and (p / "manifest.json").exists() and (p / "sequences.npz").exists()


def is_dynamic_classic_ml_dir(path: str | Path) -> bool:
    p = Path(path)
    return p.is_dir() and (p / "manifest.json").exists() and (p / "indexes.npz").exists()


def get_dynamic_teav_store(path: str | Path) -> "PortableDynamicTEAVStore":
    key = str(Path(path).resolve())
    if key not in _TEAV_CACHE:
        _TEAV_CACHE[key] = PortableDynamicTEAVStore.load(path)
    return _TEAV_CACHE[key]


def get_dynamic_classic_ml_store(path: str | Path) -> "PortableDynamicClassicMLStore":
    key = str(Path(path).resolve())
    if key not in _CML_CACHE:
        _CML_CACHE[key] = PortableDynamicClassicMLStore.load(path)
    return _CML_CACHE[key]


# =====================================================================
# Shared Vectorized Primitives (NumPy binary search & prefix sums)
# =====================================================================

def _is_binary_col(col: str) -> bool:
    return (not col.endswith("_count")) and (not col.startswith("Recency_")) and any(col.startswith(p) for p in BINARY_PREFIXES)


def _has_event_in_window(keys: np.ndarray, p_idx: np.ndarray, start_d: np.ndarray, end_d: np.ndarray) -> np.ndarray:
    n = len(p_idx)
    if len(keys) == 0 or n == 0:
        return np.zeros(n, dtype=bool)
    lo = p_idx.astype(np.int64) * KEY_STRIDE + (start_d.astype(np.int64) + DAY_OFFSET)
    hi = p_idx.astype(np.int64) * KEY_STRIDE + (end_d.astype(np.int64) + DAY_OFFSET)
    pos = np.searchsorted(keys, lo, side="left")
    in_b = pos < len(keys)
    return (start_d <= end_d) & in_b & (keys[np.where(in_b, pos, 0)] <= hi)


def _first_event_after(keys: np.ndarray, p_idx: np.ndarray, obs_end: np.ndarray) -> np.ndarray:
    n = len(p_idx)
    if len(keys) == 0 or n == 0:
        return np.full(n, INF_DAY, dtype=np.int64)
    lo = p_idx.astype(np.int64) * KEY_STRIDE + (obs_end.astype(np.int64) + 1 + DAY_OFFSET)
    hi = p_idx.astype(np.int64) * KEY_STRIDE + (KEY_STRIDE - 1)
    pos = np.searchsorted(keys, lo, side="left")
    in_b = pos < len(keys)
    safe = np.where(in_b, pos, 0)
    return np.where(in_b & (keys[safe] <= hi), (keys[safe] % KEY_STRIDE) - DAY_OFFSET, INF_DAY)


def _count_events_in_window(keys: np.ndarray, p_idx: np.ndarray, start_incl: np.ndarray, end_incl: np.ndarray) -> np.ndarray:
    n = len(p_idx)
    if len(keys) == 0 or n == 0:
        return np.zeros(n, dtype=np.int32)
    lo = p_idx.astype(np.int64) * KEY_STRIDE + (start_incl.astype(np.int64) + DAY_OFFSET)
    hi = p_idx.astype(np.int64) * KEY_STRIDE + (end_incl.astype(np.int64) + DAY_OFFSET)
    cnt = np.searchsorted(keys, hi, side="right") - np.searchsorted(keys, lo, side="left")
    return np.where(start_incl <= end_incl, np.maximum(0, cnt), 0).astype(np.int32)


def _days_since_last_event(keys: np.ndarray, p_idx: np.ndarray, obs_end: np.ndarray) -> np.ndarray:
    n = len(p_idx)
    out = np.full(n, np.nan, dtype=np.float64)
    if len(keys) == 0 or n == 0:
        return out
    lo_pat = p_idx.astype(np.int64) * KEY_STRIDE
    hi = p_idx.astype(np.int64) * KEY_STRIDE + (obs_end.astype(np.int64) + DAY_OFFSET)
    pos = np.searchsorted(keys, hi, side="right") - 1
    valid = pos >= 0
    safe = np.where(valid, pos, 0)
    matched = valid & (keys[safe] >= lo_pat)
    out[matched] = (obs_end[matched] - ((keys[safe] % KEY_STRIDE) - DAY_OFFSET)[matched]).astype(np.float64)
    return out


def _compute_labels_vectorized(
    p_idx: np.ndarray,
    obs_end: np.ndarray,
    split: str,
    *,
    strategy: str,
    has_limit: bool,
    censor_days_all: np.ndarray,
    death_days_all: np.ndarray,
    gl_days_all: np.ndarray,
    days_to_limit_all: np.ndarray,
    horizons: List[int],
    pgroups: List[str],
    grp_to_cat: Dict[str, str],
    sig_inf_all: np.ndarray,
    gl_keys: np.ndarray,
    death_keys: np.ndarray,
    sig_inf_cat: Dict[str, np.ndarray],
    sig_inf_unk: Dict[str, np.ndarray],
    sig_inf_grp_bin: Dict[str, np.ndarray],
    sig_inf_grp_surv: Dict[str, np.ndarray],
    label_keys: Optional[Sequence[str]] = None,
) -> Dict[str, np.ndarray]:
    """Shared O(log K) binary + survival label engine with automatic temporal cutoff freeze."""
    p_idx, obs_end = np.asarray(p_idx, np.int64), np.asarray(obs_end, np.int64)
    wanted = set(label_keys) if label_keys is not None else None
    req_h = [h for h in horizons if wanted is None or any(k.endswith(f"_{h:04d}d") for k in wanted)]

    censor_days = censor_days_all[p_idx]
    death_days = death_days_all[p_idx]
    gl_days = gl_days_all[p_idx]
    eff_censor_death = np.minimum(censor_days, death_days)

    out: Dict[str, np.ndarray] = {}
    _fmt = lambda s: f"label_{s.lower().replace(' ', '_')}"

    for h in req_h:
        start, end = obs_end + 1, obs_end + h
        is_cens, eff_end = censor_days < end, np.minimum(end, censor_days)
        _res = lambda ev: np.where(ev, 1, np.where(is_cens, -100, 0)).astype(np.int64)

        for cname, karr in (
            (_fmt(f"Infection {h:04d}D"), sig_inf_all),
            (_fmt(f"Graft loss {h:04d}D"), gl_keys),
            (_fmt(f"Death {h:04d}D"), death_keys),
        ):
            if wanted is None or cname in wanted:
                out[cname] = _res(_has_event_in_window(karr, p_idx, start, eff_end))

        for cat in ("Bacteria", "Virus", "Fungi"):
            cname = _fmt(f"Infection {cat} {h:04d}D")
            if wanted is None or cname in wanted:
                out[cname] = _res(_has_event_in_window(sig_inf_cat[cat], p_idx, start, eff_end))

        need_grps = [g for g in pgroups if wanted is None or _fmt(f"Infection grp {g} {h:04d}D") in wanted]
        if need_grps:
            has_unk = {c: _has_event_in_window(sig_inf_unk[c], p_idx, start, eff_end) for c in ("Bacteria", "Virus", "Fungi")}
            for g in need_grps:
                has_mem = _has_event_in_window(sig_inf_grp_bin[g], p_idx, start, eff_end)
                out[_fmt(f"Infection grp {g} {h:04d}D")] = np.where(
                    has_mem, 1, np.where(is_cens | has_unk[grp_to_cat[g]], -100, 0)
                ).astype(np.int64)

    if wanted is None or any(k.endswith(("_time", "_status")) for k in wanted):
        def _add_ep(key: str, target_fn, eff_censor: np.ndarray) -> None:
            lbl = key.replace("_", " ").capitalize()
            t_col, s_col = _fmt(f"{lbl} time"), _fmt(f"{lbl} status")
            if wanted is not None and t_col not in wanted and s_col not in wanted:
                return
            t_days = target_fn()
            is_ev = (t_days < INF_DAY) & (t_days <= eff_censor)
            out[t_col] = np.maximum(0, np.where(is_ev, t_days, eff_censor) - obs_end).astype(np.int64)
            out[s_col] = is_ev.astype(np.int64)

        _add_ep("death", lambda: death_days, censor_days)
        _add_ep("graft_loss", lambda: gl_days, eff_censor_death)
        _add_ep("infection", lambda: _first_event_after(sig_inf_all, p_idx, obs_end), eff_censor_death)
        for g in pgroups:
            _add_ep(f"grp_{g}", lambda gid=g: _first_event_after(sig_inf_grp_surv[gid], p_idx, obs_end), eff_censor_death)
        for cat in ("Bacteria", "Virus", "Fungi"):
            _add_ep(f"infection_{cat.lower()}", lambda c=cat: _first_event_after(sig_inf_cat[c], p_idx, obs_end), eff_censor_death)

    if strategy == "temporal" and has_limit and split in ("train", "validation"):
        d2l = days_to_limit_all[p_idx]
        wall = np.maximum(0, d2l - obs_end).astype(np.int64)
        for h in req_h:
            leaks = (obs_end + h) > d2l
            if leaks.any():
                suf = f"_{h:04d}d"
                for k in out:
                    if k.endswith(suf):
                        out[k] = np.where(leaks, -100, out[k])
        for t_col in [c for c in out if c.endswith("_time")]:
            leaks = (obs_end + out[t_col]) > d2l
            if leaks.any():
                out[t_col] = np.where(leaks, wall, out[t_col])
                s_col = t_col[:-5] + "_status"
                if s_col in out:
                    out[s_col] = np.where(leaks, 0, out[s_col])
    return out


def _normalize_numerical_value(
    f_val: float,
    lefts: np.ndarray,
    rights: np.ndarray,
    n_bins: int,
    median_width: float | None = None,
    right_skewed: bool = False,
    left_skewed: bool = False,
) -> Tuple[int, float]:
    """Map raw float to (quantile_bin_idx, skew-aware empirical CDF value in [-1, 1])."""
    if not np.isfinite(f_val):
        return 0, float("nan")
    if f_val <= lefts[0]:
        return 0, -1.0
    if f_val >= rights[-1]:
        return n_bins - 1, 1.0

    idx = int(np.clip(np.searchsorted(rights, f_val, side="left"), 0, n_bins - 1))
    width = float(rights[idx] - lefts[idx])
    if width > 1e-12:
        t = float(np.clip((f_val - float(lefts[idx])) / width, 0.0, 1.0))
        r = width / median_width if (median_width is not None and median_width > 1e-12) else 1.0
        if right_skewed and idx >= (n_bins // 2) and r > 1.5:
            frac = float(np.log1p((r - 1.0) * t) / np.log(r))
        elif left_skewed and idx <= (n_bins // 2) and r > 1.5:
            frac = float(1.0 - np.log1p((r - 1.0) * (1.0 - t)) / np.log(r))
        else:
            frac = t
    else:
        frac = 0.5
    return idx, float(2.0 * ((idx + frac) / float(n_bins)) - 1.0)


def _tokenize_value_single(
    attr: str, val_str: str, vocab: Dict[str, int], bin_bounds: Dict[str, Tuple], unk_id: int
) -> Tuple[int, float]:
    tok_str, norm_u = str(val_str), float("nan")
    if attr in bin_bounds:
        try:
            bidx, norm_u = _normalize_numerical_value(float(val_str), *bin_bounds[attr])
            if np.isfinite(norm_u):
                tok_str = BIN_LABELS[min(bidx, len(BIN_LABELS) - 1)]
        except (ValueError, TypeError):
            pass
    else:
        try:
            tok_str = str(int(float(val_str)))
        except (ValueError, TypeError):
            pass
    return int(vocab.get(tok_str, unk_id)), float(norm_u)


class _NumericSeriesView:
    __slots__ = ("keys", "days", "vals", "cumsum", "cumsum_sq")

    def __init__(self, keys: np.ndarray, days: np.ndarray, vals: np.ndarray, cumsum: np.ndarray, cumsum_sq: np.ndarray):
        self.keys, self.days, self.vals, self.cumsum, self.cumsum_sq = keys, days, vals, cumsum, cumsum_sq

    def days_since_last(self, p_idx: np.ndarray, obs_end: np.ndarray) -> np.ndarray:
        return _days_since_last_event(self.keys, p_idx, obs_end)

    def query(self, p_idx: np.ndarray, start_excl: np.ndarray, end_incl: np.ndarray, stats: Tuple[str, ...] = ("mean",)) -> Dict[str, np.ndarray]:
        n = len(p_idx)
        res = {s: np.full(n, np.nan, dtype=np.float64) for s in stats}
        if len(self.keys) == 0 or n == 0:
            return res
        lo_k = p_idx.astype(np.int64) * KEY_STRIDE + (start_excl.astype(np.int64) + 1 + DAY_OFFSET)
        hi_k = p_idx.astype(np.int64) * KEY_STRIDE + (end_incl.astype(np.int64) + DAY_OFFSET)
        pos_lo, pos_hi = np.searchsorted(self.keys, lo_k, side="left"), np.searchsorted(self.keys, hi_k, side="right")
        counts = pos_hi - pos_lo
        nonempty = (start_excl < end_incl) & (counts > 0)
        if not nonempty.any():
            return res

        nz = np.flatnonzero(nonempty)
        lo_nz, hi_nz, cnt_nz = pos_lo[nz], pos_hi[nz], counts[nz].astype(np.float64)
        if "mean" in stats:
            res["mean"][nz] = (self.cumsum[hi_nz] - self.cumsum[lo_nz]) / cnt_nz
        if "last" in stats:
            res["last"][nz] = self.vals[hi_nz - 1]
        if "delta" in stats:
            ge2 = counts[nz] >= 2
            if ge2.any():
                nz2 = nz[ge2]
                res["delta"][nz2] = self.vals[pos_hi[nz2] - 1] - self.vals[pos_lo[nz2]]
        if "std" in stats:
            ge2 = counts[nz] >= 2
            if ge2.any():
                nz2 = nz[ge2]
                a2, b2, c2 = pos_lo[nz2], pos_hi[nz2], counts[nz2].astype(np.float64)
                s1, s2 = self.cumsum[b2] - self.cumsum[a2], self.cumsum_sq[b2] - self.cumsum_sq[a2]
                res["std"][nz2] = np.sqrt(np.maximum(0.0, (s2 - (s1 * s1) / c2) / (c2 - 1.0)))
        if "min" in stats:
            arr_min = res["min"]
            for i_out, a, b in zip(nz, lo_nz, hi_nz):
                arr_min[i_out] = self.vals[a:b].min()
        if "max" in stats:
            arr_max = res["max"]
            for i_out, a, b in zip(nz, lo_nz, hi_nz):
                arr_max[i_out] = self.vals[a:b].max()
        if "slope" in stats:
            ge2 = counts[nz] >= 2
            if ge2.any():
                arr_slope = res["slope"]
                for i_out, a, b in zip(nz[ge2], lo_nz[ge2], hi_nz[ge2]):
                    x = self.days[a:b]
                    dx = x - x.mean()
                    denom = float(np.dot(dx, dx))
                    if denom > 1e-9:
                        y = self.vals[a:b]
                        arr_slope[i_out] = (float(np.dot(dx, y - y.mean())) / denom) * 30.0
        return res


class _IntervalView:
    __slots__ = ("keys", "p_idx", "prefix_max_stop")

    def __init__(self, keys: np.ndarray, p_idx: np.ndarray, prefix_max_stop: np.ndarray):
        self.keys, self.p_idx, self.prefix_max_stop = keys, p_idx, prefix_max_stop

    def active_in_window(self, p_idx: np.ndarray, w_start: np.ndarray, eff_w_end: np.ndarray) -> np.ndarray:
        n = len(p_idx)
        if len(self.keys) == 0 or n == 0:
            return np.zeros(n, dtype=np.int8)
        hi_k = p_idx.astype(np.int64) * KEY_STRIDE + (eff_w_end.astype(np.int64) + DAY_OFFSET)
        pos = np.searchsorted(self.keys, hi_k, side="right") - 1
        valid = pos >= 0
        safe = np.where(valid, pos, 0)
        return (valid & (self.p_idx[safe] == p_idx) & (self.prefix_max_stop[safe] > w_start)).astype(np.int8)


# =====================================================================
# 1. Portable Dynamic t-EAV Store & PyTorch Dataset
# =====================================================================

class PortableDynamicTEAVStore:
    """Reader for ``data/processed/v4.0/teav/<strat>_split/``."""

    def __init__(self, manifest: Dict[str, Any], cohort_df: pd.DataFrame, seq_arrays: Dict[str, np.ndarray], lbl_arrays: Dict[str, np.ndarray]):
        self.manifest, self.cohort_df, self.seq_arrays, self.lbl_arrays = manifest, cohort_df, seq_arrays, lbl_arrays
        self.strategy = str(manifest["strategy"])
        self.has_limit = manifest.get("limit_date") is not None
        self.vocab = {str(k): int(v) for k, v in manifest["vocab"].items()}
        self.unk_id = int(self.vocab["[UNK]"])
        self.bin_bounds = {
            a: (np.asarray(d["lefts"], np.float64), np.asarray(d["rights"], np.float64), int(d["n_bins"]), float(d["median_width"]), bool(d["right_skewed"]), bool(d["left_skewed"]))
            for a, d in manifest["bin_bounds"].items()
        }
        self.bin_intervals_compat = {a: pd.IntervalIndex.from_arrays(d["lefts"], d["rights"]) for a, d in manifest["bin_bounds"].items()}
        self.horizons = [int(h) for h in manifest["horizons"]]
        self.default_eval_fups = [int(f) for f in manifest["default_eval_fups"]]
        self.derived_age_order = int(manifest.get("derived_age_order", 13_000_000))
        self.active_med_order = int(manifest.get("active_med_order", 15_000_000))
        self.salient_entity_ids = {self.vocab[e] for e in manifest.get("salient_entities", []) if e in self.vocab}

        self.patientkeys = cohort_df["patientkey"].astype(str).to_numpy()
        self.patids = cohort_df["patid"].astype(str).to_numpy()
        self.splits_arr = cohort_df["split_group"].astype(str).to_numpy()
        self.censor_days = cohort_df["censor_day"].to_numpy(np.int64)
        self.death_days = cohort_df["death_day"].to_numpy(np.int64)
        self.graft_loss_days = cohort_df["graft_loss_day"].to_numpy(np.int64)
        self.yob_arr = cohort_df["yob"].to_numpy(np.float64)
        self.tpx_year_arr = cohort_df["tpx_year"].to_numpy(np.float64)
        self.hosp_start_days = cohort_df["hosp_start_day"].to_numpy(np.float64)
        self.hosp_stop_days = cohort_df["hosp_stop_day"].to_numpy(np.float64)
        self.max_allowed_fup_arr = cohort_df["max_allowed_fup"].to_numpy(np.int64)
        self.days_to_limit = cohort_df["days_to_limit"].to_numpy(np.int64)

        self._id_tpx_info = self.vocab.get("Transplant info", self.unk_id)
        self._id_hosp_dur = self.vocab.get("Hosp. stay duration so far (days)", self.unk_id)
        self._id_pat_vitals = self.vocab.get("Patient vitals", self.unk_id)
        self._id_tpx_age = self.vocab.get("Transplant age (years)", self.unk_id)
        self._id_pat_age = self.vocab.get("Patient age (years)", self.unk_id)
        self._id_active_med = self.vocab.get("Active medication", self.unk_id)

        self._lbl_cat = {c: lbl_arrays[f"sig_inf_cat__{c}"] for c in ("Bacteria", "Virus", "Fungi")}
        self._lbl_unk = {c: lbl_arrays[f"sig_inf_unk__{c}"] for c in ("Bacteria", "Virus", "Fungi")}
        self._lbl_grp_bin = {g: lbl_arrays[f"sig_inf_grp_bin__{g}"] for g in manifest["pathogen_groups"]}
        self._lbl_grp_surv = {g: lbl_arrays[f"sig_inf_grp_surv__{g}"] for g in manifest["pathogen_groups"]}

    @classmethod
    def load(cls, store_dir: str | Path) -> "PortableDynamicTEAVStore":
        p = Path(store_dir)
        manifest = json.loads((p / "manifest.json").read_text(encoding="utf-8"))
        cohort_df = pd.read_parquet(p / "cohort.parquet")
        with np.load(p / "sequences.npz", allow_pickle=False) as zs:
            seq_arrays = {k: zs[k] for k in zs.files}
        with np.load(p / "labels.npz", allow_pickle=False) as zl:
            lbl_arrays = {k: zl[k] for k in zl.files}
        return cls(manifest, cohort_df, seq_arrays, lbl_arrays)

    def build_tokenized_sequence(
        self, pidx: int, max_time_inclusive: Optional[int], derived_day: int,
        strict_upper_bound: Optional[int] = None, max_length: int = 1022,
        use_global_salience_truncation: bool = True, time_out_key: str = "days_since_tpx",
        separate_static: bool = False,
    ) -> Dict[str, np.ndarray]:
        sa = self.seq_arrays
        s_lo, s_hi = int(sa["static_offsets"][pidx]), int(sa["static_offsets"][pidx + 1])
        t_lo, t_hi = int(sa["timed_offsets"][pidx]), int(sa["timed_offsets"][pidx + 1])
        st_e, st_a, st_v, st_vn = sa["static_ent_id"][s_lo:s_hi], sa["static_attr_id"][s_lo:s_hi], sa["static_val_id"][s_lo:s_hi], sa["static_val_num"][s_lo:s_hi]

        hosp_dur_row: Optional[Tuple[int, int, int, int, int, float]] = None
        if derived_day >= 0 and not np.isnan(self.hosp_start_days[pidx]):
            hs, he = self.hosp_start_days[pidx], self.hosp_stop_days[pidx]
            eff_end = float(derived_day) if (np.isnan(he) or he > derived_day) else he
            vid, vnum = _tokenize_value_single("Hosp. stay duration so far (days)", str(0 if eff_end < hs else int(eff_end - hs)), self.vocab, self.bin_bounds, self.unk_id)
            if separate_static:
                hosp_dur_row = (1, self.derived_age_order + 2, self._id_tpx_info, self._id_hosp_dur, int(vid), float(vnum))
            else:
                st_e = np.append(st_e, np.int32(self._id_tpx_info))
                st_a = np.append(st_a, np.int32(self._id_hosp_dur))
                st_v = np.append(st_v, np.int32(vid))
                st_vn = np.append(st_vn, np.float32(vnum))
        st_t_arr = np.zeros(len(st_e), dtype=np.float32)

        tm_time = sa["timed_time"][t_lo:t_hi]
        cut = int(np.searchsorted(tm_time, max_time_inclusive, side="right")) if max_time_inclusive is not None else len(tm_time)
        if strict_upper_bound is not None:
            cut = min(cut, int(np.searchsorted(tm_time, strict_upper_bound - 1, side="right")))

        tm_time_c = tm_time[:cut]
        tm_rank_c, tm_ord_c = sa["timed_rank"][t_lo : t_lo + cut], sa["timed_ord"][t_lo : t_lo + cut]
        tm_e_c, tm_a_c = sa["timed_ent_id"][t_lo : t_lo + cut], sa["timed_attr_id"][t_lo : t_lo + cut]
        tm_v_c, tm_vn_c = sa["timed_val_id"][t_lo : t_lo + cut], sa["timed_val_num"][t_lo : t_lo + cut]

        inj_by_day: Dict[int, List[Tuple[int, int, int, int, int, float]]] = {}
        if derived_day >= 0 and not np.isnan(self.yob_arr[pidx]) and (strict_upper_bound is None or derived_day < strict_upper_bound):
            base_age, years = self.tpx_year_arr[pidx] - self.yob_arr[pidx], derived_day / 365.25
            v1, vn1 = _tokenize_value_single("Transplant age (years)", f"{years:.2f}", self.vocab, self.bin_bounds, self.unk_id)
            v2, vn2 = _tokenize_value_single("Patient age (years)", f"{base_age + years:.2f}", self.vocab, self.bin_bounds, self.unk_id)
            inj_by_day.setdefault(derived_day, []).extend([
                (1, self.derived_age_order, self._id_pat_vitals, self._id_tpx_age, v1, vn1),
                (1, self.derived_age_order + 1, self._id_pat_vitals, self._id_pat_age, v2, vn2),
            ])
        if hosp_dur_row is not None and (strict_upper_bound is None or derived_day < strict_upper_bound):
            inj_by_day.setdefault(derived_day, []).append(hosp_dur_row)

        m_lo, m_hi = int(sa["med_offsets"][pidx]), int(sa["med_offsets"][pidx + 1])
        if m_hi > m_lo:
            m_sd, m_ed, m_at, m_vl = sa["med_start"][m_lo:m_hi], sa["med_stop"][m_lo:m_hi], sa["med_attr_id"][m_lo:m_hi], sa["med_val_id"][m_lo:m_hi]
            for d_snap in ([derived_day] if max_time_inclusive is not None else sorted({0, derived_day})):
                if d_snap >= 0 and (strict_upper_bound is None or d_snap < strict_upper_bound):
                    act = (m_sd <= d_snap) & (m_ed > d_snap)
                    if act.any():
                        pairs = sorted({(int(a), int(v)) for a, v in zip(m_at[act], m_vl[act])})
                        inj_by_day.setdefault(d_snap, []).extend(
                            (3, self.active_med_order + j, self._id_active_med, a_id, v_id, float("nan"))
                            for j, (a_id, v_id) in enumerate(pairs)
                        )

        if separate_static:
            t_parts, e_parts, a_parts, v_parts, vn_parts = [], [], [], [], []
        else:
            t_parts, e_parts, a_parts, v_parts, vn_parts = [st_t_arr], [st_e], [st_a], [st_v], [st_vn]
        cursor = 0
        for d_inj in sorted(inj_by_day):
            lo = max(cursor, int(np.searchsorted(tm_time_c, d_inj, side="left")))
            hi = max(lo, int(np.searchsorted(tm_time_c, d_inj, side="right")))
            if lo > cursor:
                t_parts.append(tm_time_c[cursor:lo].astype(np.float32))
                e_parts.append(tm_e_c[cursor:lo]); a_parts.append(tm_a_c[cursor:lo])
                v_parts.append(tm_v_c[cursor:lo]); vn_parts.append(tm_vn_c[cursor:lo])
            rows = sorted(
                [(int(tm_rank_c[k]), int(tm_ord_c[k]), int(tm_e_c[k]), int(tm_a_c[k]), int(tm_v_c[k]), float(tm_vn_c[k])) for k in range(lo, hi)]
                + inj_by_day[d_inj],
                key=lambda r: (r[0], r[2], r[1]),
            )
            m_r = len(rows)
            t_parts.append(np.full(m_r, float(d_inj), dtype=np.float32))
            e_parts.append(np.fromiter((r[2] for r in rows), np.int32, count=m_r))
            a_parts.append(np.fromiter((r[3] for r in rows), np.int32, count=m_r))
            v_parts.append(np.fromiter((r[4] for r in rows), np.int32, count=m_r))
            vn_parts.append(np.fromiter((r[5] for r in rows), np.float32, count=m_r))
            cursor = hi
        if cursor < cut:
            t_parts.append(tm_time_c[cursor:cut].astype(np.float32))
            e_parts.append(tm_e_c[cursor:cut]); a_parts.append(tm_a_c[cursor:cut])
            v_parts.append(tm_v_c[cursor:cut]); vn_parts.append(tm_vn_c[cursor:cut])

        if t_parts:
            t_out = np.concatenate(t_parts)
            e_out, a_out, v_out = np.concatenate(e_parts).astype(np.int64), np.concatenate(a_parts).astype(np.int64), np.concatenate(v_parts).astype(np.int64)
            vn_out = np.concatenate(vn_parts).astype(np.float32)
        else:
            t_out = np.empty(0, dtype=np.float32)
            e_out, a_out, v_out = np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
            vn_out = np.empty(0, dtype=np.float32)

        if use_global_salience_truncation and max_length > 0 and len(t_out) > max_length:
            n_tot = len(t_out)
            head_cap = 32 if separate_static else 128
            head_keep, salient_budget = min(head_cap, max(1, max_length // 4)), 192
            mid_end = n_tot - max(1, max_length - head_keep - salient_budget)
            mid_sal = [i for i in range(head_keep, mid_end) if int(e_out[i]) in self.salient_entity_ids][-salient_budget:]
            tail_start = n_tot - (max_length - head_keep - len(mid_sal))
            mid_sal = [i for i in mid_sal if i < tail_start]
            keep = np.concatenate([np.arange(head_keep, dtype=np.int64), np.asarray(mid_sal, dtype=np.int64), np.arange(tail_start, n_tot, dtype=np.int64)])
            t_out, e_out, a_out, v_out, vn_out = t_out[keep], e_out[keep], a_out[keep], v_out[keep], vn_out[keep]

        out = {time_out_key: t_out, "entity_id": e_out, "attribute_id": a_out, "value_id": v_out, "value_num": vn_out}
        if separate_static:
            out.update({
                "static_entity_id": st_e.astype(np.int64),
                "static_attribute_id": st_a.astype(np.int64),
                "static_value_id": st_v.astype(np.int64),
                "static_value_num": st_vn.astype(np.float32),
            })
        return out

    def compute_labels_batch(self, p_idx: np.ndarray, obs_end: np.ndarray, split: str, label_keys: Optional[Sequence[str]] = None) -> Dict[str, np.ndarray]:
        return _compute_labels_vectorized(
            p_idx, obs_end, split,
            strategy=self.strategy, has_limit=self.has_limit,
            censor_days_all=self.censor_days, death_days_all=self.death_days,
            gl_days_all=self.graft_loss_days, days_to_limit_all=self.days_to_limit,
            horizons=self.horizons, pgroups=self.manifest["pathogen_groups"], grp_to_cat=self.manifest["group_to_category"],
            sig_inf_all=self.lbl_arrays["sig_inf_all"], gl_keys=self.lbl_arrays["gl_event_keys"], death_keys=self.lbl_arrays["death_event_keys"],
            sig_inf_cat=self._lbl_cat, sig_inf_unk=self._lbl_unk,
            sig_inf_grp_bin=self._lbl_grp_bin, sig_inf_grp_surv=self._lbl_grp_surv,
            label_keys=label_keys,
        )


class _ColumnTableProxy:
    """Minimal ``ds.data.column(name)`` proxy matching HuggingFace ``Dataset``."""

    def __init__(self, ds: "PortableDynamicTEAVDataset"):
        self._ds = ds

    def column(self, name: str) -> np.ndarray:
        return self._ds._get_column_array(name)


class PortableDynamicTEAVDataset(TorchDataset):
    """Pickle-safe PyTorch Dataset with HuggingFace-compatible column/select helpers."""

    def __init__(
        self,
        store: PortableDynamicTEAVStore,
        split: str = "train",
        mode: str = "fixed_grid",
        fixed_fups: Optional[Sequence[int]] = None,
        label_keys: Optional[Sequence[str]] = None,
        time_out_key: str = "days_since_tpx",
        max_length: int = 1022,
        use_global_salience_truncation: bool = True,
        max_fup_cap: int = 3600,
        stride_days: int = 30,
        drop_unlabeled: bool = False,
        target_undersampling_ratio: Optional[float] = None,
        undersampling_balancing_label: Optional[str] = None,
        undersampling_mode: str = "global",
        ipw_weight_cap: float = 10.0,
        patient_weight_alpha: float = 0.0,
        seed: int = 1234,
        separate_static: bool = False,
        classic_ml_store: Optional["PortableDynamicClassicMLStore"] = None,
        landmark_preprocessor: Optional["LandmarkSummaryPreprocessor"] = None,
    ):
        assert mode in ("pretraining", "fixed_grid", "random_landmark")
        assert undersampling_mode in ("stratified_ipw", "global", "stratified"), f"Unknown undersampling_mode='{undersampling_mode}'"
        self.store, self.split, self.mode = store, split, mode
        self.fixed_fups = [int(f) for f in fixed_fups] if fixed_fups is not None else list(store.default_eval_fups)
        self.label_keys = list(label_keys) if label_keys is not None else None
        self.time_out_key, self.max_length = time_out_key, int(max_length)
        self.use_global_salience_truncation = bool(use_global_salience_truncation)
        self.max_fup_cap, self.stride_days = int(max_fup_cap), max(1, int(stride_days))
        self.drop_unlabeled = bool(drop_unlabeled)
        self.target_undersampling_ratio = target_undersampling_ratio
        self.undersampling_balancing_label = undersampling_balancing_label
        self.undersampling_mode = str(undersampling_mode)
        self.ipw_weight_cap = float(ipw_weight_cap)
        self.patient_weight_alpha = float(patient_weight_alpha)
        self.seed = int(seed)
        self.separate_static = bool(separate_static)
        self.classic_ml_store = classic_ml_store
        self.landmark_preprocessor = landmark_preprocessor
        self._landmark_summary_mat: Optional[np.ndarray] = None

        self._split_pidx = np.flatnonzero(store.splits_arr == split)
        self._pidx_arr = np.empty(0, dtype=np.int64)
        self._fup_arr = np.empty(0, dtype=np.int64)
        self._labels: Dict[str, np.ndarray] = {}
        self._extra_cols: Dict[str, np.ndarray] = {}
        self._fixed_len: Optional[int] = None
        self._survival_resampling_cfg: Optional[Tuple[List[Tuple[str, str]], float, Optional[float]]] = None
        self.epoch = 0
        self.data = _ColumnTableProxy(self)
        self._build_epoch_schedule(epoch=0)

    def configure_survival_resampling(self, survival_pairs: List[Tuple[str, str]], max_horizon: float, us_ratio: Optional[float]) -> None:
        self._survival_resampling_cfg = (
            [(str(t), str(s)) for t, s in survival_pairs],
            float(max_horizon),
            (float(us_ratio) if us_ratio is not None else None),
        )

    def _apply_survival_postprocess(self, lbls: Dict[str, np.ndarray], fups: np.ndarray, epoch: int) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        assert self._survival_resampling_cfg is not None
        survival_pairs, max_horizon, us_ratio = self._survival_resampling_cfg
        any_keep = np.zeros(len(fups), dtype=bool)
        for t_col, s_col in survival_pairs:
            t_arr, s_arr = lbls[t_col], lbls[s_col]
            any_keep |= (t_arr != -100) & (s_arr != -100) & (t_arr >= 0) & ((s_arr == 0) | (s_arr == 1)) & ((s_arr == 1) | (t_arr > 0))
        keep_idx = np.flatnonzero(any_keep)
        if len(keep_idx) == 0 or us_ratio is None or us_ratio <= 0.0:
            return keep_idx, None

        tr_t = np.stack([lbls[t][keep_idx] for t, _ in survival_pairs], axis=1)
        tr_s = np.stack([lbls[s][keep_idx] for _, s in survival_pairs], axis=1)
        is_acute = ((tr_s == 1) & (tr_t >= 0) & (tr_t <= max_horizon)).any(axis=1)
        bins = fups[keep_idx] // 30
        rng = np.random.default_rng(self.seed + 7919 * epoch + 31)
        order = np.argsort(bins, kind="mergesort")
        _, split_pts = np.unique(bins[order], return_index=True)
        kept_idx_l, kept_w_l = [], []
        for grp in np.split(order, split_pts[1:]):
            pos_g, rest_g = grp[is_acute[grp]], grp[~is_acute[grp]]
            keep_rest_n = int(round(max(1, len(pos_g)) * us_ratio))
            if len(rest_g) > keep_rest_n > 0:
                chosen_rest, w_rest = rng.choice(rest_g, size=keep_rest_n, replace=False), float(len(rest_g)) / float(keep_rest_n)
            else:
                chosen_rest, w_rest = rest_g, 1.0
            merged_g = np.concatenate([pos_g, chosen_rest])
            merged_w = np.concatenate([np.ones(len(pos_g), np.float32), np.full(len(chosen_rest), w_rest, np.float32)])
            perm = rng.permutation(len(merged_g))
            kept_idx_l.append(keep_idx[merged_g[perm]])
            kept_w_l.append(merged_w[perm])
        return np.concatenate(kept_idx_l), np.concatenate(kept_w_l)

    def _build_epoch_schedule(self, epoch: int) -> None:
        self.epoch = int(epoch)
        split_pidx = self._split_pidx
        if self.mode == "pretraining":
            self._pidx_arr, self._fup_arr, self._labels = split_pidx.astype(np.int64), np.full(len(split_pidx), -1, np.int64), {}
            self._extra_cols.pop("ipw_weight", None)
            return

        if self.mode == "fixed_grid":
            p_chunks, o_chunks = [], []
            max_f = self.store.max_allowed_fup_arr[split_pidx]
            for fup in self.fixed_fups:
                ok = max_f >= int(fup)
                if ok.any():
                    chosen = split_pidx[ok]
                    p_chunks.append(chosen); o_chunks.append(np.full(len(chosen), int(fup), np.int64))
            p_arr = np.concatenate(p_chunks) if p_chunks else np.empty(0, np.int64)
            o_arr = np.concatenate(o_chunks) if o_chunks else np.empty(0, np.int64)
        else:
            rng = np.random.default_rng(self.seed + 1009 * epoch)
            max_d = np.minimum(self.store.max_allowed_fup_arr[split_pidx], self.max_fup_cap)
            valid = max_d >= 0
            vp, vd = split_pidx[valid], max_d[valid]
            counts = 1 + (vd // self.stride_days)
            p_arr = np.repeat(vp, counts)
            o_arr = rng.integers(0, np.repeat(vd, counts) + 1, dtype=np.int64) if len(p_arr) > 0 else np.empty(0, np.int64)

        lbls = self.store.compute_labels_batch(p_arr, o_arr, split=self.split, label_keys=self.label_keys) if len(p_arr) > 0 else {}
        ipw_weights: Optional[np.ndarray] = None

        if len(p_arr) > 0 and self._survival_resampling_cfg is not None:
            keep, ipw_weights = self._apply_survival_postprocess(lbls, o_arr, epoch)
            if self.mode == "random_landmark" and self._fixed_len is not None and len(keep) > 0:
                rng_l = np.random.default_rng(self.seed + 31337 * epoch)
                sel = rng_l.choice(len(keep), size=self._fixed_len, replace=(len(keep) < self._fixed_len)) if len(keep) != self._fixed_len else np.arange(len(keep))
                keep = keep[sel]
                if ipw_weights is not None:
                    ipw_weights = ipw_weights[sel]
            p_arr, o_arr, lbls = p_arr[keep], o_arr[keep], {k: v[keep] for k, v in lbls.items()}
        elif len(p_arr) > 0 and self.label_keys and all(k in lbls for k in self.label_keys):
            lmat = np.stack([lbls[k] for k in self.label_keys], axis=1)
            # Pre-undersampling frame count per patient (covariate-only, independent of Y)
            if self.mode == "random_landmark":
                pat_max_d = np.minimum(self.store.max_allowed_fup_arr[p_arr], self.max_fup_cap)
                pat_total_counts = np.maximum(1, 1 + (pat_max_d // self.stride_days)).astype(np.float64)
            else:
                uniq_p, inv_p, cnt_p = np.unique(p_arr, return_inverse=True, return_counts=True)
                pat_total_counts = cnt_p[inv_p].astype(np.float64)

            keep = np.arange(len(p_arr), dtype=np.int64)
            if self.drop_unlabeled:
                keep = keep[(lmat != -100).any(axis=1)]
            keep_w = np.ones(len(keep), dtype=np.float64)
            has_sample_weights = False

            if self.split == "train" and self.target_undersampling_ratio is not None and len(keep) > 0:
                rng_u = np.random.default_rng(self.seed + 7919 * epoch + 17)
                bal_col = lbls.get(self.undersampling_balancing_label) if self.undersampling_balancing_label else None

                def _split_group(g_idx: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
                    if bal_col is not None:
                        bc = bal_col[g_idx]
                        return g_idx[bc == 1], g_idx[bc == 0], g_idx[(bc != 0) & (bc != 1)]
                    gm = lmat[g_idx]
                    any_pos, unlab = (gm.max(axis=1) == 1), (gm == -100).all(axis=1)
                    return g_idx[any_pos & ~unlab], g_idx[~any_pos & ~unlab], g_idx[unlab]

                if self.undersampling_mode == "global":
                    pos_i, neg_i, unl_i = _split_group(keep)
                    maj, mino = (neg_i, pos_i) if len(pos_i) < len(neg_i) else (pos_i, neg_i)
                    cap = int(round(max(1, len(mino)) * float(self.target_undersampling_ratio)))
                    if len(maj) > cap:
                        maj = rng_u.choice(maj, size=cap, replace=False)
                    merged = np.concatenate([maj, mino, unl_i])
                    merged_w = np.ones(len(merged), dtype=np.float64)
                    perm = rng_u.permutation(len(merged))
                    keep, keep_w = merged[perm], merged_w[perm]
                else:
                    bins = o_arr[keep] // 30
                    order = np.argsort(bins, kind="mergesort")
                    _, split_pts = np.unique(bins[order], return_index=True)
                    use_ipw = self.undersampling_mode == "stratified_ipw"
                    kept_chunks, kept_w_chunks = [], []
                    for grp in np.split(order, split_pts[1:]):
                        g_idx = keep[grp]
                        pos_i, neg_i, unl_i = _split_group(g_idx)
                        maj, mino = (neg_i, pos_i) if len(pos_i) < len(neg_i) else (pos_i, neg_i)
                        cap = int(round(max(1, len(mino)) * float(self.target_undersampling_ratio)))
                        if len(maj) > cap:
                            w_maj = min(self.ipw_weight_cap, float(len(maj)) / float(cap)) if use_ipw else 1.0
                            maj = rng_u.choice(maj, size=cap, replace=False)
                            if use_ipw:
                                has_sample_weights = True
                        else:
                            w_maj = 1.0
                        merged = np.concatenate([maj, mino, unl_i])
                        merged_w = np.concatenate([
                            np.full(len(maj), w_maj, dtype=np.float64),
                            np.ones(len(mino) + len(unl_i), dtype=np.float64),
                        ])
                        perm = rng_u.permutation(len(merged))
                        kept_chunks.append(merged[perm])
                        kept_w_chunks.append(merged_w[perm])
                    keep = np.concatenate(kept_chunks) if kept_chunks else np.empty(0, np.int64)
                    keep_w = np.concatenate(kept_w_chunks) if kept_w_chunks else np.empty(0, np.float64)

            if self.split == "train" and self.patient_weight_alpha > 0.0 and len(keep) > 0:
                w_pat = np.power(pat_total_counts[keep], -float(self.patient_weight_alpha))
                keep_w = keep_w * w_pat
                has_sample_weights = True

            if self.mode == "random_landmark" and self._fixed_len is not None and len(keep) > 0 and len(keep) != self._fixed_len:
                rng_l = np.random.default_rng(self.seed + 31337 * epoch)
                sel = rng_l.choice(len(keep), size=self._fixed_len, replace=(len(keep) < self._fixed_len))
                keep = keep[sel]
                keep_w = keep_w[sel]

            if self.split == "train" and has_sample_weights and len(keep) > 0:
                ipw_weights = (keep_w / max(float(keep_w.mean()), 1e-8)).astype(np.float32)

            p_arr, o_arr, lbls = p_arr[keep], o_arr[keep], {k: v[keep] for k, v in lbls.items()}

        if self.mode == "random_landmark" and self._fixed_len is None:
            self._fixed_len = len(p_arr)
        self._pidx_arr, self._fup_arr, self._labels = p_arr, o_arr, lbls
        if ipw_weights is not None:
            self._extra_cols["ipw_weight"] = ipw_weights
        else:
            self._extra_cols.pop("ipw_weight", None)
        if "split" in self._extra_cols:
            self._extra_cols["split"] = np.full(len(p_arr), self.split, dtype=object)

        if self.classic_ml_store is not None and self.landmark_preprocessor is not None and len(p_arr) > 0:
            pkeys = self.store.patientkeys[p_arr]
            raw_semi = self.classic_ml_store.extract_semi_static_matrix(pkeys, o_arr)
            if self.split == "train" and not self.landmark_preprocessor.is_fitted:
                self.landmark_preprocessor.fit(raw_semi)
            self._landmark_summary_mat = self.landmark_preprocessor.transform(raw_semi)
        else:
            self._landmark_summary_mat = None

    def set_epoch(self, epoch: int) -> None:
        if self.mode == "random_landmark":
            self._build_epoch_schedule(epoch)

    @property
    def column_names(self) -> List[str]:
        cols = ["entity_id", "attribute_id", "value_id", "value_num", self.time_out_key]
        if self.separate_static:
            cols.extend(["static_entity_id", "static_attribute_id", "static_value_id", "static_value_num"])
        if self._landmark_summary_mat is not None:
            cols.append("landmark_summary")
        cols.extend(["patientid", "patientkey", "fup", *self._labels.keys()])
        return cols + [k for k in self._extra_cols if k not in cols]

    def _get_column_array(self, name: str) -> np.ndarray:
        if name == "fup":
            return self._fup_arr
        if name == "patientid":
            return self.store.patids[self._pidx_arr]
        if name == "patientkey":
            return self.store.patientkeys[self._pidx_arr]
        if name == "split":
            return self._extra_cols.get("split", np.full(len(self._pidx_arr), self.split, dtype=object))
        if name in self._labels:
            return self._labels[name]
        if name in self._extra_cols:
            return self._extra_cols[name]
        raise KeyError(f"Column '{name}' not available on PortableDynamicTEAVDataset.")

    def select(self, indices: Sequence[int] | np.ndarray, keep_in_memory: bool = True) -> "PortableDynamicTEAVDataset":
        idx = np.asarray(indices, dtype=np.int64)
        clone = copy.copy(self)
        clone.data = _ColumnTableProxy(clone)
        clone._pidx_arr, clone._fup_arr = self._pidx_arr[idx], self._fup_arr[idx]
        clone._labels = {k: v[idx] for k, v in self._labels.items()}
        clone._extra_cols = {k: v[idx] for k, v in self._extra_cols.items()}
        if self._landmark_summary_mat is not None:
            clone._landmark_summary_mat = self._landmark_summary_mat[idx]
        clone._fixed_len = len(clone._pidx_arr)
        return clone

    def flatten_indices(self, keep_in_memory: bool = True) -> "PortableDynamicTEAVDataset":
        return self

    def add_column(self, name: str, values: Sequence[Any] | np.ndarray) -> "PortableDynamicTEAVDataset":
        clone = copy.copy(self)
        clone.data = _ColumnTableProxy(clone)
        clone._extra_cols = {**self._extra_cols, name: np.asarray(values)}
        return clone

    def set_format(self, type: str = "numpy", columns: Optional[Sequence[str]] = None) -> None:
        return None

    def __len__(self) -> int:
        return len(self._pidx_arr)

    def __getitem__(self, index: int | str) -> Any:
        if isinstance(index, str):
            return self._get_column_array(index)
        pidx, fup = int(self._pidx_arr[index]), int(self._fup_arr[index])
        if self.mode == "pretraining":
            d2l = int(self.store.days_to_limit[pidx])
            sample = self.store.build_tokenized_sequence(
                pidx, max_time_inclusive=None, derived_day=int(self.store.censor_days[pidx]),
                strict_upper_bound=(d2l if d2l < int(NO_LIMIT_SENTINEL // 2) else None),
                max_length=self.max_length, use_global_salience_truncation=self.use_global_salience_truncation,
                time_out_key=self.time_out_key, separate_static=self.separate_static,
            )
        else:
            sample = self.store.build_tokenized_sequence(
                pidx, max_time_inclusive=fup, derived_day=fup, strict_upper_bound=None,
                max_length=self.max_length, use_global_salience_truncation=self.use_global_salience_truncation,
                time_out_key=self.time_out_key, separate_static=self.separate_static,
            )
        sample.update({"fup": fup, "patientid": str(self.store.patids[pidx]), "patientkey": str(self.store.patientkeys[pidx]), "split": self.split})
        if self._landmark_summary_mat is not None:
            sample["landmark_summary"] = self._landmark_summary_mat[index]
        for k, arr in self._labels.items():
            sample[k] = int(arr[index])
        for k, arr in self._extra_cols.items():
            val = arr[index]
            sample[k] = float(val) if isinstance(val, (np.floating, float)) else (val.item() if hasattr(val, "item") else val)
        return sample


def make_dynamic_epoch_callback(train_dataset: Any):
    """HuggingFace ``TrainerCallback`` that resamples continuous random landmarks each epoch."""
    from transformers import TrainerCallback

    class DynamicEpochResampleCallback(TrainerCallback):
        def __init__(self, ds):
            self._ds, self._last_epoch = ds, 0

        def on_epoch_begin(self, args, state, control, **kwargs):
            ep = int(round(state.epoch or 0.0))
            if ep > self._last_epoch and hasattr(self._ds, "set_epoch"):
                self._ds.set_epoch(ep)
                self._last_epoch = ep

    return DynamicEpochResampleCallback(train_dataset)


# =====================================================================
# 2. Portable Dynamic Classic ML Store
# =====================================================================

_DEFAULT_CLASSIC_ML_FEATURE_CONFIG: Dict[str, Any] = {
    "windows": {
        "include_cumulative": True,
        "post_transplant_days": [30],
        "pre_observation_days": [7, 30, 90, 365],
    },
    "numeric_series": {
        "rich_windows": ["All", "Last90d"],
        "rich_stats": ["mean", "last", "min", "max", "std", "delta", "slope"],
        "default_stats": ["mean"],
        "patient_index_stats": ["mean", "min", "last", "delta", "slope"],
        "virology_stat": "max",
        "banff_stat": "max",
        "dsa_stat": "max",
    },
    "events": {
        "include_hospitalization": True,
        "include_medications": True,
        "include_immunology": True,
        "count_windows": ["All", "Last30d", "Last90d", "Last365d"],
        "infection_count_categories": ["Bacteria", "Virus", "Fungi"],
        "rejection_treatment_count_windows": ["All", "Last90d", "Last365d"],
        "substance_use_windows": ["All", "Last30d", "Last90d", "Last365d"],
    },
    "recency": {
        "enabled": True,
        "events": {
            "Infection_Any": ["misc", "inf_all"],
            "Infection_Bacteria": ["inf_count", "Bacteria"],
            "Infection_Virus": ["inf_count", "Virus"],
            "Infection_Fungi": ["inf_count", "Fungi"],
            "Rejection_BiopsyProven": ["rej_count", "Biops proven rj"],
            "RejectionTreatment_Any": ["misc", "rej_treat_all"],
        },
        "measurements": {
            "Creatinine": ["lab", "Creatinine (µmol/l)"],
            "eGFR": ["patient_index", "eGFR (mL/min/1.73m²)"],
            "BKV": ["virus", "BKV (copies/ml)"],
        },
        "include_hospital_discharge": True,
    },
}


class PortableDynamicClassicMLStore:
    """Reader & on-the-fly materializer for ``data/processed/v4.0/classic_ml/<strat>_split/``."""

    def __init__(self, manifest: Dict[str, Any], cohort_df: pd.DataFrame, static_df: pd.DataFrame, arrays: Dict[str, np.ndarray]):
        self.manifest, self.cohort_df, self.static_df = manifest, cohort_df, static_df
        self.strategy = str(manifest["strategy"])
        self.has_limit = manifest.get("limit_date") is not None
        self.output_columns = [c for c in manifest["master_columns"] if c != "split_group"]
        self.horizons = [int(h) for h in manifest["horizons"]]
        self.default_eval_fups = [int(f) for f in manifest["default_eval_fups"]]
        self.feature_config = manifest.get("feature_config") or _DEFAULT_CLASSIC_ML_FEATURE_CONFIG
        win_cfg = self.feature_config.get("windows", {})
        self.aggr_post_tpx_days = [int(d) for d in manifest.get("aggr_post_tpx_days", win_cfg.get("post_transplant_days", [30]))]
        self.aggr_pre_obs_days = [int(d) for d in manifest.get("aggr_pre_obs_days", win_cfg.get("pre_observation_days", [7, 30, 90, 365]))]
        self.include_cumulative = bool(manifest.get("include_cumulative_window", win_cfg.get("include_cumulative", True)))

        self.patientkeys = cohort_df["patientkey"].astype(str).to_numpy()
        self._pkey_to_idx: Dict[str, int] = {k: i for i, k in enumerate(self.patientkeys)}
        self._static_col_set = set(static_df.columns)
        self.semi_static_columns: List[str] = [
            c for c in self.output_columns
            if c not in ("patientid", "patientkey", "obs_end_days")
            and not c.startswith("label_")
            and c not in self._static_col_set
        ]
        self.patids = cohort_df["patid"].astype(str).to_numpy()
        self.splits = cohort_df["split_group"].astype(str).to_numpy()
        self.censor_days = cohort_df["censor_day"].to_numpy(np.int64)
        self.death_days = cohort_df["death_day"].to_numpy(np.int64)
        self.graft_loss_days = cohort_df["graft_loss_day"].to_numpy(np.int64)
        self.yob_arr = cohort_df["yob"].to_numpy(np.float64)
        self.tpx_year_arr = cohort_df["tpx_year"].to_numpy(np.float64)
        self.hosp_start_days = cohort_df["hosp_start_day"].to_numpy(np.float64)
        self.hosp_stop_days = cohort_df["hosp_stop_day"].to_numpy(np.float64)
        self.max_allowed_fup = cohort_df["max_allowed_fup"].to_numpy(np.int64)
        self.days_to_limit = cohort_df["days_to_limit"].to_numpy(np.int64)

        self._num_groups = {
            g: {
                name: _NumericSeriesView(
                    arrays[f"num_{g}_{i}_keys"], arrays[f"num_{g}_{i}_days"], arrays[f"num_{g}_{i}_vals"],
                    arrays[f"num_{g}_{i}_cumsum"], arrays[f"num_{g}_{i}_cumsum_sq"],
                )
                for i, name in enumerate(names)
            }
            for g, names in manifest["numeric_series_keys"].items()
        }
        self._interval_groups = {
            g: {name: _IntervalView(arrays[f"iv_{g}_{i}_keys"], arrays[f"iv_{g}_{i}_p_idx"], arrays[f"iv_{g}_{i}_pmax"]) for i, name in enumerate(names)}
            for g, names in manifest["interval_keys"].items()
        }
        self._event_groups = {
            g: {name: arrays[f"ev_{g}_{i}"] for i, name in enumerate(names)}
            for g, names in manifest["event_keys"].items()
        }

    @classmethod
    def load(cls, store_dir: str | Path) -> "PortableDynamicClassicMLStore":
        p = Path(store_dir)
        manifest = json.loads((p / "manifest.json").read_text(encoding="utf-8"))
        cohort_df = pd.read_parquet(p / "cohort.parquet")
        static_df = pd.read_parquet(p / "static.parquet")
        with np.load(p / "indexes.npz", allow_pickle=False) as z:
            arrays = {k: z[k] for k in z.files}
        return cls(manifest, cohort_df, static_df, arrays)

    def sample_landmarks(
        self, split: str, mode: str = "fixed_grid", fups: Optional[Sequence[int]] = None,
        stride_days: int = 30, max_fup_cap: int = 3600, seed: int = 1234,
    ) -> Tuple[np.ndarray, np.ndarray]:
        split_pidx = np.flatnonzero(self.splits == split)
        if len(split_pidx) == 0:
            return np.empty(0, np.int64), np.empty(0, np.int64)
        if mode == "fixed_grid":
            p_chunks, o_chunks = [], []
            max_f = self.max_allowed_fup[split_pidx]
            for fup in (list(fups) if fups is not None else self.default_eval_fups):
                ok = max_f >= int(fup)
                if ok.any():
                    chosen = split_pidx[ok]
                    p_chunks.append(chosen); o_chunks.append(np.full(len(chosen), int(fup), np.int64))
            return (np.concatenate(p_chunks), np.concatenate(o_chunks)) if p_chunks else (np.empty(0, np.int64), np.empty(0, np.int64))
        if mode == "random_landmark":
            rng = np.random.default_rng(seed)
            max_d = np.minimum(self.max_allowed_fup[split_pidx], int(max_fup_cap))
            valid = max_d >= 0
            vp, vd = split_pidx[valid], max_d[valid]
            if len(vp) == 0:
                return np.empty(0, np.int64), np.empty(0, np.int64)
            counts = 1 + (vd // max(1, int(stride_days)))
            return np.repeat(vp, counts), rng.integers(0, np.repeat(vd, counts) + 1, dtype=np.int64)
        raise ValueError(f"Unsupported mode='{mode}'.")

    def _extract_dynamic_feature_dict(self, p_idx: np.ndarray, obs_end: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute all causal time-varying semi-static landmark summary features for ``(p_idx, obs_end)``."""
        n = len(p_idx)
        windows: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        if self.include_cumulative:
            windows["All"] = (np.full(n, -100_000, np.int64), obs_end)
        for d in self.aggr_post_tpx_days:
            windows[f"First{d}d"] = (np.zeros(n, np.int64), np.minimum(np.full(n, int(d), np.int64), obs_end))
        for d in self.aggr_pre_obs_days:
            windows[f"Last{d}d"] = (obs_end - int(d), obs_end)

        cols: Dict[str, Any] = {}
        yob, tpx_yr = self.yob_arr[p_idx], self.tpx_year_arr[p_idx]
        has_yob = ~np.isnan(yob)
        cols["Vital_Transplant age (days)_Current"] = np.where(has_yob, obs_end.astype(np.float64), np.nan)
        cols["Vital_Patient age (years)_Current"] = np.where(has_yob, (tpx_yr - yob) + (obs_end / 365.25), np.nan)

        numeric_cfg = self.feature_config.get("numeric_series", _DEFAULT_CLASSIC_ML_FEATURE_CONFIG["numeric_series"])
        events_cfg = self.feature_config.get("events", _DEFAULT_CLASSIC_ML_FEATURE_CONFIG["events"])
        recency_cfg = self.feature_config.get("recency", _DEFAULT_CLASSIC_ML_FEATURE_CONFIG["recency"])

        hs, he = self.hosp_start_days[p_idx], self.hosp_stop_days[p_idx]
        has_hs = ~np.isnan(hs)
        if events_cfg.get("include_hospitalization", True):
            s_day, e_day = np.where(has_hs, hs, 0.0), np.where(has_hs, np.where(~np.isnan(he), np.maximum(he, hs), hs), 0.0)
            for wname, (w0, eff_w1) in windows.items():
                ov = np.where(has_hs, np.maximum(0.0, np.minimum(e_day, eff_w1) - np.maximum(s_day, w0)), 0.0).astype(np.int64)
                cols[f"Hosp_days_{wname}"], cols[f"Hosp_inpatient_{wname}"] = ov, (ov > 0).astype(np.int8)

        rich_wins = set(numeric_cfg.get("rich_windows", ())) & set(windows)
        rich_stats = tuple(numeric_cfg.get("rich_stats", ()))
        default_stats = tuple(numeric_cfg.get("default_stats", ()))
        for prefix, gname in (("Lab", "lab"), ("Vital", "vital"), ("OrganLab", "organ_lab")):
            for tname, sidx in self._num_groups[gname].items():
                for wname, (w0, eff_w1) in windows.items():
                    stats = rich_stats if wname in rich_wins else default_stats
                    if not stats:
                        continue
                    for sname, arr in sidx.query(p_idx, w0, eff_w1, stats=stats).items():
                        cols[f"{prefix}_{tname}_{wname}_{sname}"] = arr

        patient_index_stats = tuple(numeric_cfg.get("patient_index_stats", ()))
        if patient_index_stats:
            for tname, sidx in self._num_groups["patient_index"].items():
                for wname, (w0, eff_w1) in windows.items():
                    for sname, arr in sidx.query(p_idx, w0, eff_w1, stats=patient_index_stats).items():
                        cols[f"Idx_{tname}_{wname}_{sname}"] = arr

        virology_stat = numeric_cfg.get("virology_stat")
        if virology_stat:
            for vname, sidx in self._num_groups["virus"].items():
                for wname, (w0, eff_w1) in windows.items():
                    cols[f"Vir_{vname}_{wname}_{virology_stat}_log"] = sidx.query(p_idx, w0, eff_w1, stats=(virology_stat,))[virology_stat]
        banff_stat = numeric_cfg.get("banff_stat")
        if banff_stat:
            for bcol, sidx in self._num_groups["banff"].items():
                for wname, (w0, eff_w1) in windows.items():
                    cols[f"Banff_{bcol}_{wname}_{banff_stat}"] = sidx.query(p_idx, w0, eff_w1, stats=(banff_stat,))[banff_stat]

        if events_cfg.get("include_medications", True):
            for prefix, gname in (("Med", "med"), ("MedCat", "medcat")):
                for name, iidx in self._interval_groups[gname].items():
                    for wname, (w0, eff_w1) in windows.items():
                        cols[f"{prefix}_{name}_{wname}"] = iidx.active_in_window(p_idx, w0, eff_w1)

        for prefix, gname in (("Inf", "inf"), ("Rej", "rej"), ("OrgEv", "orgev"), ("Comorb", "comorb")):
            for cname, keys in self._event_groups[gname].items():
                for wname, (w0, eff_w1) in windows.items():
                    cols[f"{prefix}_{cname}_{wname}"] = _has_event_in_window(keys, p_idx, w0 + 1, eff_w1).astype(np.int8)

        misc_ev = self._event_groups["misc"]
        rej_treat_count_wins = set(events_cfg.get("rejection_treatment_count_windows", ()))
        for gname, keys in self._event_groups["rej_treat"].items():
            for wname, (w0, eff_w1) in windows.items():
                cnt = _count_events_in_window(keys, p_idx, w0 + 1, eff_w1)
                cols[f"RejTreat_{gname}_{wname}"] = (cnt > 0).astype(np.int8)
                if wname in rej_treat_count_wins:
                    cols[f"RejTreat_{gname}_{wname}_count"] = cnt

        if len(misc_ev["rej_treat_all"]) > 0:
            for wname, (w0, eff_w1) in windows.items():
                cnt = _count_events_in_window(misc_ev["rej_treat_all"], p_idx, w0 + 1, eff_w1)
                cols[f"RejTreat_Any_{wname}"], cols[f"RejTreat_Any_{wname}_count"] = (cnt > 0).astype(np.int8), cnt

        subuse_wins = [w for w in events_cfg.get("substance_use_windows", ()) if w in windows]
        for sname, keys in self._event_groups["subuse"].items():
            for wname in subuse_wins:
                w0, eff_w1 = windows[wname]
                cols[f"SubUse_{sname}_{wname}"] = _has_event_in_window(keys, p_idx, w0 + 1, eff_w1).astype(np.int8)

        count_wins = [w for w in events_cfg.get("count_windows", ()) if w in windows]
        if len(misc_ev["inf_all"]) > 0:
            for wname in count_wins:
                w0, eff_w1 = windows[wname]
                cols[f"Inf_Any_{wname}_count"] = _count_events_in_window(misc_ev["inf_all"], p_idx, w0 + 1, eff_w1)
        for prefix, gname in (("Inf", "inf_count"), ("Rej", "rej_count")):
            for cname, keys in self._event_groups[gname].items():
                for wname in count_wins:
                    w0, eff_w1 = windows[wname]
                    cols[f"{prefix}_{cname}_{wname}_count"] = _count_events_in_window(keys, p_idx, w0 + 1, eff_w1)

        if recency_cfg.get("enabled", True):
            for rname, (gname, kname) in recency_cfg.get("events", {}).items():
                rkeys = self._event_groups.get(gname, {}).get(kname, np.empty(0, np.int64))
                cols[f"Recency_days_since_{rname}"] = _days_since_last_event(rkeys, p_idx, obs_end)

            for lname, (gname, sname) in recency_cfg.get("measurements", {}).items():
                sidx = self._num_groups.get(gname, {}).get(sname)
                if sidx is not None:
                    cols[f"Recency_days_since_{lname}"] = sidx.days_since_last(p_idx, obs_end)

            if recency_cfg.get("include_hospital_discharge", True):
                cols["Recency_days_since_HospitalDischarge"] = np.where(has_hs & (~np.isnan(he)) & (he <= obs_end), obs_end - he, np.nan)

        if events_cfg.get("include_immunology", True):
            dsa_stat = numeric_cfg.get("dsa_stat")
            dsa_idx = self._num_groups["dsa"]["_dsa"]
            for wname, (w0, eff_w1) in windows.items():
                cols[f"Immu_AntiHLA_{wname}"] = _has_event_in_window(misc_ev["hla_pos"], p_idx, w0 + 1, eff_w1).astype(np.int8)
                if dsa_stat:
                    dsa_col = f"Immu_MaxDSA_{wname}_log" if dsa_stat == "max" else f"Immu_DSA_{wname}_{dsa_stat}_log"
                    cols[dsa_col] = dsa_idx.query(p_idx, w0, eff_w1, stats=(dsa_stat,))[dsa_stat]
            if len(misc_ev["cm_any"]) > 0:
                neg_inf = np.full(n, -100_000, np.int64)
                for cname, pos_keys in self._event_groups["cm_pos"].items():
                    cols[f"Immu_{cname}"] = _has_event_in_window(pos_keys, p_idx, neg_inf, obs_end).astype(np.int8)
        return cols

    def extract_semi_static_matrix(self, patient_keys: np.ndarray, landmark_days: np.ndarray) -> np.ndarray:
        """Vectorized extraction of all semi-static ML landmark summaries aligned to ``(patient_keys, landmark_days)``."""
        n = len(patient_keys)
        m = len(self.semi_static_columns)
        if n == 0 or m == 0:
            return np.empty((n, m), dtype=np.float32)
        p_idx = np.fromiter((self._pkey_to_idx[str(k)] for k in patient_keys), dtype=np.int64, count=n)
        obs_end = np.asarray(landmark_days, dtype=np.int64)
        cols = self._extract_dynamic_feature_dict(p_idx, obs_end)
        out = np.empty((n, m), dtype=np.float32)
        for j, col in enumerate(self.semi_static_columns):
            if col in cols:
                out[:, j] = np.asarray(cols[col], dtype=np.float32)
            elif _is_binary_col(col) or col.endswith("_count"):
                out[:, j] = 0.0
            else:
                out[:, j] = np.nan
        return out

    def materialize_split(
        self, split: str, mode: str = "fixed_grid", fups: Optional[Sequence[int]] = None,
        stride_days: int = 30, max_fup_cap: int = 3600, seed: int = 1234,
    ) -> pd.DataFrame:
        p_idx, obs_end = self.sample_landmarks(split, mode, fups, stride_days, max_fup_cap, seed)
        n = len(p_idx)
        if n == 0:
            df = pd.DataFrame(columns=self.output_columns)
            df["__fup__"] = np.empty(0, dtype=np.int64)
            return df

        cols: Dict[str, Any] = {"patientid": self.patids[p_idx], "patientkey": self.patientkeys[p_idx], "obs_end_days": obs_end}
        st_slice = self.static_df.iloc[p_idx]
        for c in st_slice.columns:
            cols[c] = st_slice[c].to_numpy()
        cols.update(self._extract_dynamic_feature_dict(p_idx, obs_end))

        lbl_misc = self._event_groups["lbl_misc"]
        cols.update(_compute_labels_vectorized(
            p_idx, obs_end, split,
            strategy=self.strategy, has_limit=self.has_limit,
            censor_days_all=self.censor_days, death_days_all=self.death_days,
            gl_days_all=self.graft_loss_days, days_to_limit_all=self.days_to_limit,
            horizons=self.horizons, pgroups=self.manifest["pathogen_groups"], grp_to_cat=self.manifest["group_to_category"],
            sig_inf_all=lbl_misc["sig_inf_all"], gl_keys=lbl_misc["gl"], death_keys=lbl_misc["death"],
            sig_inf_cat=self._event_groups["lbl_cat"], sig_inf_unk=self._event_groups["lbl_unk"],
            sig_inf_grp_bin=self._event_groups["lbl_grp_bin"], sig_inf_grp_surv=self._event_groups["lbl_grp_surv"],
        ))

        ordered = {
            col: (
                cols[col] if col in cols
                else np.zeros(n, np.int8) if _is_binary_col(col)
                else np.zeros(n, np.int32) if col.endswith("_count")
                else np.full(n, -100, np.int64) if col.startswith("label_")
                else np.full(n, np.nan, np.float64)
            )
            for col in self.output_columns
        }
        df = pd.DataFrame(ordered)
        df["__fup__"] = obs_end.astype(np.int64)
        return df


def _classify_semi_static_domain(col: str) -> str:
    """Group semi-static landmark summary columns into 5 coherent clinical domains."""
    if col.startswith(("Lab_", "Vital_", "OrganLab_", "Idx_")) or col in (
        "Recency_days_since_Creatinine",
        "Recency_days_since_eGFR",
    ):
        return "labs_vitals"
    if col.startswith(("Inf_", "Vir_", "Recency_days_since_Infection_", "Recency_days_since_BKV")):
        return "infections_virology"
    if col.startswith(("Rej_", "RejTreat_", "Immu_", "Banff_", "Recency_days_since_Rejection")):
        return "rejection_immunology"
    if col.startswith(("Med_", "MedCat_")):
        return "medications"
    return "hospital_comorb_lifestyle"


class LandmarkSummaryPreprocessor:
    """
    Train-only domain-grouped preprocessor for semi-static Classic ML landmark summaries.

    Groups the ~386 time-varying summary columns into 5 clinical domains, standardizes
    continuous/count/recency columns strictly on training landmarks (clipped to [-5, 5]),
    appends explicit binary missingness indicators for columns with NaNs in training,
    drops zero-variance columns, and outputs one contiguous float32 matrix ordered by group
    so ``LandmarkSummaryEncoder`` can split into domain tokens with zero copy.
    """

    DOMAIN_ORDER = (
        "labs_vitals",
        "infections_virology",
        "rejection_immunology",
        "medications",
        "hospital_comorb_lifestyle",
    )

    def __init__(self, column_names: Sequence[str]):
        self.column_names: List[str] = list(column_names)
        self.is_fitted: bool = False
        self.group_names: List[str] = []
        self.group_dims: List[int] = []
        self._group_specs: List[Dict[str, Any]] = []

    def fit(self, X_train: np.ndarray) -> "LandmarkSummaryPreprocessor":
        X = np.asarray(X_train, dtype=np.float32)
        grouped_indices: Dict[str, List[int]] = {d: [] for d in self.DOMAIN_ORDER}
        for idx, col in enumerate(self.column_names):
            grouped_indices[_classify_semi_static_domain(col)].append(idx)

        self.group_names = []
        self.group_dims = []
        self._group_specs = []
        for domain in self.DOMAIN_ORDER:
            raw_idxs = grouped_indices[domain]
            if not raw_idxs:
                continue
            num_idxs, bin_idxs = [], []
            medians, stds, miss_sub_idxs = [], [], []
            for j in raw_idxs:
                col_vals = X[:, j]
                finite = np.isfinite(col_vals)
                if not finite.any():
                    continue
                valid = col_vals[finite]
                if float(np.nanstd(valid)) < 1e-8 and finite.all():
                    continue
                if _is_binary_col(self.column_names[j]):
                    bin_idxs.append(j)
                else:
                    sub_pos = len(num_idxs)
                    num_idxs.append(j)
                    med = float(np.median(valid))
                    sd = float(np.std(valid))
                    medians.append(med)
                    stds.append(sd if sd > 1e-6 else 1.0)
                    if not finite.all():
                        miss_sub_idxs.append(sub_pos)

            dim = len(num_idxs) + len(miss_sub_idxs) + len(bin_idxs)
            if dim == 0:
                continue
            self.group_names.append(domain)
            self.group_dims.append(dim)
            self._group_specs.append({
                "domain": domain,
                "num_idxs": np.asarray(num_idxs, dtype=np.int64),
                "medians": np.asarray(medians, dtype=np.float32),
                "stds": np.asarray(stds, dtype=np.float32),
                "miss_sub_idxs": np.asarray(miss_sub_idxs, dtype=np.int64),
                "bin_idxs": np.asarray(bin_idxs, dtype=np.int64),
            })

        self.is_fitted = True
        return self

    def transform(self, X_in: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("LandmarkSummaryPreprocessor must be fitted on training split before transform().")
        X = np.asarray(X_in, dtype=np.float32)
        n = X.shape[0]
        parts: List[np.ndarray] = []
        for spec in self._group_specs:
            num_idxs = spec["num_idxs"]
            bin_idxs = spec["bin_idxs"]
            if len(num_idxs) > 0:
                sub = X[:, num_idxs]
                is_miss = ~np.isfinite(sub)
                filled = np.where(is_miss, spec["medians"][None, :], sub)
                scaled = np.clip((filled - spec["medians"][None, :]) / spec["stds"][None, :], -5.0, 5.0).astype(np.float32)
                parts.append(scaled)
                if len(spec["miss_sub_idxs"]) > 0:
                    parts.append(is_miss[:, spec["miss_sub_idxs"]].astype(np.float32))
            if len(bin_idxs) > 0:
                bsub = X[:, bin_idxs]
                parts.append(np.where(np.isfinite(bsub), bsub, 0.0).astype(np.float32))
        return np.concatenate(parts, axis=1) if parts else np.zeros((n, 0), dtype=np.float32)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "column_names": self.column_names,
            "group_names": self.group_names,
            "group_dims": self.group_dims,
            "group_specs": [
                {
                    "domain": s["domain"],
                    "num_idxs": s["num_idxs"].tolist(),
                    "medians": s["medians"].tolist(),
                    "stds": s["stds"].tolist(),
                    "miss_sub_idxs": s["miss_sub_idxs"].tolist(),
                    "bin_idxs": s["bin_idxs"].tolist(),
                }
                for s in self._group_specs
            ],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LandmarkSummaryPreprocessor":
        obj = cls(data["column_names"])
        obj.group_names = list(data["group_names"])
        obj.group_dims = [int(d) for d in data["group_dims"]]
        obj._group_specs = [
            {
                "domain": s["domain"],
                "num_idxs": np.asarray(s["num_idxs"], dtype=np.int64),
                "medians": np.asarray(s["medians"], dtype=np.float32),
                "stds": np.asarray(s["stds"], dtype=np.float32),
                "miss_sub_idxs": np.asarray(s["miss_sub_idxs"], dtype=np.int64),
                "bin_idxs": np.asarray(s["bin_idxs"], dtype=np.int64),
            }
            for s in data["group_specs"]
        ]
        obj.is_fitted = True
        return obj
