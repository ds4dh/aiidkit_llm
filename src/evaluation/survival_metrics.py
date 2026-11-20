"""
Shared survival-analysis metrics, usable for BOTH model families:

    - Survival transformers (Cox): risk = log hazard ratio, with a Breslow
      baseline survival function estimated on a reference cohort.
    - Binary classifiers (transformer multi-horizon or classic ML): risk at a
      fixed horizon H; survival function built as a right-continuous step
      function S(t) = 1 - P(event by t) over the evaluated horizons.

All functions operate on NumPy arrays:
    times    : observed follow-up time per patient (days from fup observation end)
    statuses : 1 = event observed, 0 = right-censored
    risks    : higher risk = worse prognosis
"""

import numpy as np

MASK = -100.0


def _concordance_counts(times, statuses, risks):
    """
    Exact Harrell concordance counting for one stratum.
    Returns (concordant, discordant); tied risk scores count as 0.5 concordant.
    Comparable pairs: pairs where the shorter time is an observed event.
    """
    n = len(times)
    if n < 2:
        return 0.0, 0.0
    conc, disc = 0.0, 0.0
    # Chunk over rows to bound memory for large strata
    chunk = max(1, int(4_000_000 // max(n, 1)))
    for start in range(0, n, chunk):
        idx = np.arange(start, min(start + chunk, n))
        # i = focus index (candidate shorter time), j = compared index
        comp = (times[idx, None] < times[None, :]) & (statuses[idx, None] == 1)
        if not comp.any():
            continue
        r_diff = risks[idx, None] - risks[None, :]
        conc += float((comp & (r_diff > 0)).sum()) + 0.5 * float((comp & (r_diff == 0)).sum())
        disc += float((comp & (r_diff < 0)).sum())
    return conc, disc


# ---------------------------------------------------------------------------
# Concordance (Harrell's C-index)
# ---------------------------------------------------------------------------

def concordance_index(
    times: np.ndarray,
    statuses: np.ndarray,
    risks: np.ndarray,
    strata: np.ndarray | None = None,
) -> float:
    """
    Harrell's C-index. When `strata` is given (e.g. the fup observation point of
    each sample), concordant/discordant pairs are only counted WITHIN strata and
    aggregated across strata (matching sets), which avoids comparing patients
    whose time-zero is different.
    """
    times = np.asarray(times, dtype=np.float64)
    statuses = np.asarray(statuses).astype(bool)
    risks = np.asarray(risks, dtype=np.float64)

    if strata is None:
        strata = np.zeros(len(times), dtype=int)
    strata = np.asarray(strata)

    total_conc, total_disc = 0.0, 0.0
    for s in np.unique(strata):
        m = strata == s
        if m.sum() < 2 or statuses[m].sum() == 0:
            continue
        conc, discord = _concordance_counts(times[m], statuses[m].astype(int), risks[m])
        total_conc += conc
        total_disc += discord
    denom = total_conc + total_disc
    return float(total_conc / denom) if denom > 0 else float("nan")


# ---------------------------------------------------------------------------
# Breslow baseline survival (to turn Cox risk scores into S(t) curves)
# ---------------------------------------------------------------------------

def breslow_baseline(
    times: np.ndarray,
    statuses: np.ndarray,
    risks: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Estimates the Breslow baseline cumulative hazard H0(t) from a reference
    cohort with Cox scores r(x):   S(t | x) = exp(-H0(t) * exp(r(x))).

    Returns:
        (event_times, cumulative_baseline_hazard) step function arrays.
    """
    times = np.asarray(times, dtype=np.float64)
    statuses = np.asarray(statuses).astype(int)
    risks = np.exp(np.clip(np.asarray(risks, dtype=np.float64), -30, 30))

    order = np.argsort(times)
    t, e, r = times[order], statuses[order], risks[order]

    # Risk-set sums with ties included: reverse cumulative sum over tied blocks
    rev_csum = np.cumsum(r[::-1])[::-1]  # rev_csum[i] = sum of r[j], j >= i
    unique_t, first_idx = np.unique(t, return_index=True)
    # For ascending-sorted arrays, the risk set of time u is {t_j >= u}, whose
    # sum equals rev_csum at the FIRST index where t == u
    risk_at_unique = rev_csum[first_idx]

    # Breslow increments at unique times using O(N) np.add.reduceat over sorted tied blocks
    d_per_unique = np.add.reduceat(e.astype(np.float64), first_idx) if len(unique_t) > 0 else np.empty(0, dtype=np.float64)
    event_counts = np.where(d_per_unique > 0, d_per_unique / np.maximum(risk_at_unique, 1e-12), 0.0)

    keep = event_counts > 0
    return unique_t[keep], np.cumsum(event_counts[keep])


def survival_matrix_from_baseline(
    event_times: np.ndarray,
    baseline_hazard: np.ndarray,
    risks: np.ndarray,
    horizons: list[int] | np.ndarray,
) -> np.ndarray:
    """
    Evaluates S(t_i | x_j) on a grid of horizons for a cohort of risk scores.
    Step function: H0(u) = hazard of the largest event time <= u (0 before first).
    Returns array [n_patients, n_horizons].
    """
    risks = np.asarray(risks, dtype=np.float64)
    horizons = np.asarray(horizons, dtype=np.float64)
    if len(event_times) == 0:
        return np.ones((len(risks), len(horizons)))
    h0 = np.searchsorted(event_times, horizons, side="right") - 1
    h0_vals = np.where(h0 >= 0, baseline_hazard[np.clip(h0, 0, None)], 0.0)
    h0_vals = np.where(horizons < event_times[0], 0.0, h0_vals)
    return np.exp(-np.outer(np.exp(np.clip(risks, -30, 30)), h0_vals))


def survival_matrix_from_binary_horizons(
    risks: np.ndarray,
    horizons: list[int] | np.ndarray,
) -> np.ndarray:
    """
    Builds a step survival function for a horizon-based binary model, given a
    matrix of predicted event probabilities P(event by h_k) over increasing
    horizons: S(h_k) = 1 - p_k. Columns must be ordered by increasing horizon.
    """
    probs = np.clip(np.asarray(risks, dtype=np.float64), 0.0, 1.0)
    return 1.0 - probs


# ---------------------------------------------------------------------------
# Kaplan-Meier censoring distribution + IPCW Brier score
# ---------------------------------------------------------------------------

def kaplan_meier(
    times: np.ndarray,
    events: np.ndarray,
    eval_times: np.ndarray,
) -> np.ndarray:
    """
    Right-continuous Kaplan-Meier estimate of P(T > u) evaluated at eval_times.
    (Pass events = 1 - statuses to estimate the CENSORING survival G(u).)
    """
    times = np.asarray(times, dtype=np.float64)
    events = np.asarray(events).astype(int)
    eval_times = np.asarray(eval_times, dtype=np.float64)

    order = np.argsort(times)
    t, e = times[order], events[order]
    n = len(t)

    unique_t, first_idx = np.unique(t, return_index=True)
    at_risk = n - first_idx  # number at risk just before unique_t
    d = np.add.reduceat(e.astype(np.float64), first_idx) if len(unique_t) > 0 else np.empty(0, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        surv_steps = np.cumprod(1.0 - np.where(at_risk > 0, d / np.maximum(at_risk, 1), 0.0))

    idx = np.searchsorted(unique_t, eval_times, side="right") - 1
    km = np.where(idx >= 0, surv_steps[np.clip(idx, 0, None)], 1.0)
    return km


def ipcw_brier_scores(
    times: np.ndarray,
    statuses: np.ndarray,
    survival_matrix: np.ndarray,
    horizons: list[int] | np.ndarray,
    eps: float = 1e-3,
) -> np.ndarray:
    """
    IPCW (Uno-style) Brier score at each horizon, for a predicted survival
    matrix S(h_k | x_i) of shape [n_patients, n_horizons].

    BS(t) = mean_i [ 1{T_i<=t, d_i=1} S(t|x)^2 / G(T_i-)
                   + 1{T_i>t}          (1 - S(t|x))^2 / G(t)  ]
    where G is the Kaplan-Meier estimate of the censoring survival.
    Returns [n_horizons] array.
    """
    times = np.asarray(times, dtype=np.float64)
    statuses = np.asarray(statuses).astype(int)
    horizons = np.asarray(horizons, dtype=np.float64)
    surv = np.clip(np.asarray(survival_matrix, dtype=np.float64), 0.0, 1.0)

    # G(u): censoring survival; G(T_i-) evaluated at times just below T_i
    g_at_horizons = kaplan_meier(times, 1 - statuses, horizons)
    g_before_events = kaplan_meier(times, 1 - statuses, times - 0.5)
    g_at_horizons = np.maximum(g_at_horizons, eps)
    g_before_events = np.maximum(g_before_events, eps)

    briers = np.zeros(len(horizons), dtype=np.float64)
    n = len(times)
    for k, t in enumerate(horizons):
        s_k = surv[:, k]
        term1_mask = (times <= t) & (statuses == 1)
        term2_mask = times > t
        # Graf's IPCW Brier: outcome is I(T > t); events before t -> outcome 0 -> residual S(t)^2,
        # subjects still at risk past t -> outcome 1 -> residual (1 - S(t))^2
        term1 = ((s_k[term1_mask]) ** 2 / g_before_events[term1_mask]).sum()
        term2 = (((1.0 - s_k[term2_mask])) ** 2 / g_at_horizons[k]).sum()
        briers[k] = (term1 + term2) / n
    return briers


def integrated_brier_score(
    briers: np.ndarray | list[float],
    horizons: list[int] | np.ndarray,
) -> float:
    """Trapezoidal integration of the IPCW Brier score over the horizon range."""
    briers = np.asarray(briers, dtype=np.float64)
    horizons = np.asarray(horizons, dtype=np.float64)
    if len(briers) < 2:
        return float(briers[0]) if len(briers) == 1 else float("nan")
    order = np.argsort(horizons)
    _trapz = getattr(np, "trapezoid", None) or np.trapz
    return float(_trapz(briers[order], horizons[order]) / (horizons[order][-1] - horizons[order][0]))
