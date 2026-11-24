"""
Survival analysis utilities: neural heads, empirical hazard priors, and proper
scoring losses for time-to-event fine-tuning of the patient t-EAV Transformer.

Supports two survival formulations:
  1. `surv_ehr` (default, inspired by SurvivEHR [Gadd et al., 2026], DeSurv
     [Danks & Yau, 2022], and MOTOR [Steinberg et al., 2023]):
       - Multi-scale sequence pooling (lifelong mean + 30-day acute exponential
         recency pooling) + sinusoidal landmark & recency time embeddings.
       - Task x Time-Interval conditioned non-proportional neural hazard head
         anchored on empirical `(task, clinical_window, interval)` log-baseline
         hazards computed strictly from the training split.
       - Exact Horvitz-Thompson importance-weighted continuous-time piecewise
         exponential survival NLL + cumulative-incidence proper scoring rule,
         yielding closed-form calibrated CIFs `F_k(t | x) = 1 - exp(-H_k(t | x))`,
         RMST risk scores, and continuous median survival times.
  2. `cox` (legacy baseline):
       - Single scalar log-hazard ratio per task trained with Breslow Cox
         partial likelihood (`CoxNLLLoss`).
"""

from __future__ import annotations

import inspect
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

MASK = -100.0

CLINICAL_WINDOW_BOUNDS: tuple[tuple[int, int], ...] = (
    (0, 30),
    (31, 180),
    (181, 360),
    (361, 1080),
    (1081, 100000),
)


def fup_to_window_index_np(fups: np.ndarray) -> np.ndarray:
    """Map landmark `fup` days to clinical window indices `0..4`."""
    fups_arr = np.asarray(fups, dtype=np.float64)
    out = np.full(fups_arr.shape, len(CLINICAL_WINDOW_BOUNDS) - 1, dtype=np.int64)
    for idx, (lo, hi) in enumerate(CLINICAL_WINDOW_BOUNDS):
        mask = (fups_arr >= lo) & (fups_arr <= hi)
        out[mask] = idx
    return out


def build_survival_time_grid(horizons: list[int], max_time: float | None = None) -> list[float]:
    """
    Build a clinically aligned piecewise-continuous time grid `0 = tau_0 < ... < tau_M`
    ensuring every reporting horizon in `horizons` lands exactly on a knot.
    """
    hz = sorted(set(int(h) for h in horizons if int(h) > 0))
    max_h = max(hz) if hz else 90
    if max_h <= 90:
        cap = float(max_time) if max_time is not None else 360.0
        base_knots = [0.0, 10.0, 20.0, 30.0, 45.0, 60.0, 90.0, 180.0, 360.0]
    else:
        cap = float(max_time) if max_time is not None else float(max(max_h, 1800))
        base_knots = [0.0, 30.0, 90.0, 180.0, 360.0, 540.0, 720.0, 1080.0, 1800.0]
    knots = sorted(set([0.0, *[float(h) for h in hz], *[k for k in base_knots if k <= cap], cap]))
    return knots


def compute_empirical_log_hazard_priors(
    times_2d: np.ndarray,
    statuses_2d: np.ndarray,
    fups_1d: np.ndarray,
    time_grid: list[float] | np.ndarray,
    laplace_events: float = 0.5,
    laplace_days: float = 100.0,
) -> np.ndarray:
    """
    Compute empirical daily log-hazard priors `log_lambda0[k, w, m]` of shape
    `(num_heads, num_windows, num_intervals)` strictly from the FULL training split.

    Shrinks sparse `(task, window, interval)` cells smoothly toward the task's
    global interval hazard rate so rare pathogen heads start calibrated at Step 0.
    """
    grid = np.asarray(time_grid, dtype=np.float64)
    lefts, rights = grid[:-1], grid[1:]
    widths = rights - lefts
    num_heads = times_2d.shape[1]
    num_wins = len(CLINICAL_WINDOW_BOUNDS)
    num_int = len(widths)
    win_ids = fup_to_window_index_np(fups_1d)

    log_priors = np.full((num_heads, num_wins, num_int), -9.0, dtype=np.float32)
    for k in range(num_heads):
        t_k = np.asarray(times_2d[:, k], dtype=np.float64)
        s_k = np.asarray(statuses_2d[:, k], dtype=np.int64)
        valid = (t_k != MASK) & (s_k != MASK) & (t_k >= 0)
        if not np.any(valid):
            continue

        t_v, s_v, w_v = t_k[valid], s_k[valid], win_ids[valid]
        t_eff = np.where((t_v == 0.0) & (s_v == 1), 0.5, t_v)
        exp_global = np.clip(t_eff[:, None] - lefts[None, :], 0.0, widths[None, :]).sum(axis=0)
        ev_global = np.zeros(num_int, dtype=np.float64)
        for m in range(num_int):
            in_bin = (t_v >= lefts[m]) & (t_v <= rights[m]) if m == 0 else ((t_v > lefts[m]) & (t_v <= rights[m]))
            ev_global[m] = float(np.sum((s_v == 1) & in_bin))
        rate_global = (ev_global + laplace_events) / np.maximum(exp_global + laplace_days, 1.0)

        for w_idx in range(num_wins):
            mw = w_v == w_idx
            if not np.any(mw):
                log_priors[k, w_idx, :] = np.log(np.clip(rate_global, 1e-7, 1.0)).astype(np.float32)
                continue
            t_w, s_w = t_v[mw], s_v[mw]
            t_w_eff = np.where((t_w == 0.0) & (s_w == 1), 0.5, t_w)
            exp_w = np.clip(t_w_eff[:, None] - lefts[None, :], 0.0, widths[None, :]).sum(axis=0)
            ev_w = np.zeros(num_int, dtype=np.float64)
            for m in range(num_int):
                in_bin = (t_w >= lefts[m]) & (t_w <= rights[m]) if m == 0 else ((t_w > lefts[m]) & (t_w <= rights[m]))
                ev_w[m] = float(np.sum((s_w == 1) & in_bin))
            # Empirical-Bayes pseudo-exposure blending toward global task interval rate
            prior_exp = 500.0
            rate_w = (ev_w + prior_exp * rate_global) / np.maximum(exp_w + prior_exp, 1.0)
            log_priors[k, w_idx, :] = np.log(np.clip(rate_w, 1e-7, 1.0)).astype(np.float32)

    return log_priors


class SurvivEHRHead(nn.Module):
    """
    SurvivEHR / MOTOR-inspired Landmark-Conditioned Non-Proportional Neural Hazard Head.

    Combines:
      1. Lifelong history mean pooling (`h_mean`) + 30-day acute exponential recency
         pooling (`h_acute`) over token representations.
      2. Sinusoidal + logarithmic encoding of landmark follow-up time (`fup`) and
         elapsed time since the most recent clinical event (`recency_gap`).
      3. Direct + low-rank Task x Time-Interval conditioned residual log-hazards
         added to a learnable `(K, W, M)` empirical log-baseline hazard table.

    Outputs float32 log daily hazards `eta` of shape `[B, K * M]`.
    """

    def __init__(
        self,
        hidden_size: int,
        num_tasks: int,
        time_grid: list[float],
        empirical_log_hazards: np.ndarray | list | None = None,
        dropout_prob: float = 0.1,
        sub_dim: int = 64,
        acute_tau_days: float = 30.0,
    ):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.num_tasks = int(num_tasks)
        self.time_grid = [float(x) for x in time_grid]
        self.num_intervals = len(self.time_grid) - 1
        self.num_windows = len(CLINICAL_WINDOW_BOUNDS)
        self.sub_dim = int(sub_dim)
        self.acute_tau_days = float(acute_tau_days)

        # Fixed buffers for window boundaries and sinusoidal frequencies
        win_lo = torch.tensor([b[0] for b in CLINICAL_WINDOW_BOUNDS], dtype=torch.float32)
        win_hi = torch.tensor([b[1] for b in CLINICAL_WINDOW_BOUNDS], dtype=torch.float32)
        self.register_buffer("win_lo", win_lo, persistent=False)
        self.register_buffer("win_hi", win_hi, persistent=False)
        freqs = torch.exp(torch.linspace(0.0, math.log(3600.0), 6, dtype=torch.float32))
        self.register_buffer("sin_freqs", freqs, persistent=False)

        # Time feature dim: 12 (sin/cos fup) + 1 (log fup) + 1 (log recency) + 5 (window one-hot) = 19 -> 32D
        self.time_proj = nn.Sequential(
            nn.Linear(19, 32),
            nn.SiLU(),
        )
        self.ctx_fusion = nn.Sequential(
            nn.Linear(2 * self.hidden_size + 32, self.hidden_size),
            nn.GELU(),
            nn.LayerNorm(self.hidden_size),
            nn.Dropout(float(dropout_prob)),
        )

        # Direct projection + Task x Time-Interval conditioned interaction (SurvivEHR Fig. 1A / MOTOR)
        self.direct_head = nn.Linear(self.hidden_size, self.num_tasks * self.num_intervals)
        nn.init.normal_(self.direct_head.weight, mean=0.0, std=0.005)
        nn.init.zeros_(self.direct_head.bias)

        self.ctx_sub = nn.Linear(self.hidden_size, self.sub_dim)
        self.task_emb = nn.Parameter(torch.randn(self.num_tasks, self.sub_dim) * 0.02)

        # Initialize interval embeddings smoothly from log-midpoint time + small random symmetry breaker
        mids = [0.5 * (self.time_grid[m] + self.time_grid[m + 1]) for m in range(self.num_intervals)]
        int_init = torch.zeros(self.num_intervals, self.sub_dim, dtype=torch.float32)
        for m, mid_t in enumerate(mids):
            for d in range(0, min(16, self.sub_dim), 2):
                div = 1000.0 ** (d / 16.0)
                int_init[m, d] = math.sin(mid_t / div) * 0.05
                if d + 1 < self.sub_dim:
                    int_init[m, d + 1] = math.cos(mid_t / div) * 0.05
        self.interval_emb = nn.Parameter(int_init)
        self.shared_out_scale = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))

        if empirical_log_hazards is not None:
            init_prior = torch.as_tensor(empirical_log_hazards, dtype=torch.float32)
            if init_prior.shape != (self.num_tasks, self.num_windows, self.num_intervals):
                init_prior = torch.full((self.num_tasks, self.num_windows, self.num_intervals), -8.5, dtype=torch.float32)
        else:
            init_prior = torch.full((self.num_tasks, self.num_windows, self.num_intervals), -8.5, dtype=torch.float32)
        self.log_base_hazard = nn.Parameter(init_prior)

    def _window_indices(self, fup: torch.Tensor) -> torch.Tensor:
        in_win = (fup[:, None] >= self.win_lo[None, :]) & (fup[:, None] <= self.win_hi[None, :])
        has_any = in_win.any(dim=1)
        idx = in_win.float().argmax(dim=1)
        return torch.where(has_any, idx, torch.full_like(idx, self.num_windows - 1))

    def _encode_landmark_time(self, fup: torch.Tensor, recency_gap: torch.Tensor, win_idx: torch.Tensor) -> torch.Tensor:
        scaled = fup[:, None].clamp(min=0.0) / self.sin_freqs[None, :]
        sin_feat = torch.sin(scaled)
        cos_feat = torch.cos(scaled)
        log_fup = torch.log1p(fup.clamp(min=0.0) / 30.0)[:, None]
        log_rec = torch.log1p(recency_gap.clamp(min=0.0) / 7.0)[:, None]
        win_oh = F.one_hot(win_idx, num_classes=self.num_windows).float()
        raw = torch.cat([sin_feat, cos_feat, log_fup, log_rec, win_oh], dim=-1)
        return self.time_proj(raw)

    def forward(
        self,
        last_hidden: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        token_times: torch.Tensor | None = None,
        landmark_fup: torch.Tensor | None = None,
    ) -> torch.Tensor:
        dev_type = last_hidden.device.type if last_hidden.device.type in ("cuda", "cpu") else "cpu"
        with torch.amp.autocast(device_type=dev_type, enabled=False):
            h = last_hidden.float()
            bsz, seq_len, _ = h.shape
            if attention_mask is not None:
                mask = attention_mask.float()
            else:
                mask = torch.ones((bsz, seq_len), device=h.device, dtype=torch.float32)

            mask_sum = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
            h_mean = (h * mask.unsqueeze(-1)).sum(dim=1) / mask_sum

            # Determine landmark clock `fup` and acute 30-day recency-weighted token pooling
            if token_times is not None:
                t_tok = token_times.float()
                # Exclude BOS/EOS zero-padded timestamps on interior positions by masking pad tokens
                neg_inf = torch.full_like(t_tok, -1e9)
                t_max_tok = torch.where(mask > 0.5, t_tok, neg_inf).max(dim=1).values.clamp(min=0.0)
                fup_vec = landmark_fup.float().to(h.device) if landmark_fup is not None else t_max_tok
                ref_time = torch.maximum(fup_vec, t_max_tok)
                recency_gap = (fup_vec - t_max_tok).clamp(min=0.0)

                dt = (ref_time[:, None] - t_tok).clamp(min=0.0)
                # Mask out padding tokens; give recent clinical events exponential weight exp(-dt / 30d)
                attn_logits = torch.where(mask > 0.5, -dt / self.acute_tau_days, torch.full_like(dt, -1e9))
                acute_weights = torch.softmax(attn_logits, dim=1)
                h_acute = (h * acute_weights.unsqueeze(-1)).sum(dim=1)
            else:
                fup_vec = (
                    landmark_fup.float().to(h.device)
                    if landmark_fup is not None
                    else torch.zeros(bsz, device=h.device, dtype=torch.float32)
                )
                recency_gap = torch.zeros(bsz, device=h.device, dtype=torch.float32)
                last_idx = (mask.sum(dim=1).long().clamp(min=1) - 1)
                h_acute = h[torch.arange(bsz, device=h.device), last_idx]

            win_idx = self._window_indices(fup_vec)
            t_emb = self._encode_landmark_time(fup_vec, recency_gap, win_idx)
            ctx = self.ctx_fusion(torch.cat([h_mean, h_acute, t_emb], dim=-1))

            delta_direct = self.direct_head(ctx).view(bsz, self.num_tasks, self.num_intervals)
            ctx_s = self.ctx_sub(ctx)[:, None, :] + self.task_emb[None, :, :]
            delta_shared = (
                torch.einsum("bkd,md->bkm", F.silu(ctx_s), self.interval_emb)
                * (self.shared_out_scale / math.sqrt(float(self.sub_dim)))
            )
            delta_eta = (delta_direct + delta_shared).clamp(min=-8.0, max=8.0)

            base_eta = self.log_base_hazard[:, win_idx, :].permute(1, 0, 2)  # [B, K, M]
            eta = base_eta + delta_eta
            return eta.reshape(bsz, self.num_tasks * self.num_intervals)


class SurvivEHRPiecewiseLoss(nn.Module):
    """
    Exact Horvitz-Thompson Importance-Weighted Piecewise-Continuous Survival Loss.

    Combines:
      1. Continuous-time Piecewise-Exponential survival NLL across `time_grid`
         (exact event-day density + exact right-censoring exposure).
      2. Censoring-aware Cumulative-Incidence NLL at `horizons` (`1 - exp(-H_k(h))`),
         reinforcing calibrated horizon probabilities from the same hazard curve.
      3. Centered temporal smoothness regularization across adjacent intervals so
         patient-specific log-hazard ratios vary smoothly without penalizing
         population baseline hazard shifts.
    """

    def __init__(
        self,
        time_grid: list[float],
        horizons: list[int],
        tail_weight: float = 0.25,
        cif_weight: float = 0.5,
        smooth_weight: float = 0.005,
    ):
        super().__init__()
        self.time_grid = [float(x) for x in time_grid]
        self.horizons = sorted(int(h) for h in horizons)
        self.num_intervals = len(self.time_grid) - 1
        self.tail_weight = float(tail_weight)
        self.cif_weight = float(cif_weight)
        self.smooth_weight = float(smooth_weight)

        grid_t = torch.tensor(self.time_grid, dtype=torch.float32)
        lefts = grid_t[:-1]
        rights = grid_t[1:]
        widths = rights - lefts
        max_eval_h = float(max(self.horizons)) if self.horizons else 90.0
        int_weights = torch.where(rights <= max_eval_h + 1e-6, torch.ones_like(widths), torch.full_like(widths, self.tail_weight))
        # Horizon exposure matrix [J, M]: exact overlap of [0, h_j] with each interval m
        hz_t = torch.tensor([float(h) for h in self.horizons], dtype=torch.float32)
        hz_overlap = torch.clamp(hz_t[:, None] - lefts[None, :], min=0.0)
        hz_overlap = torch.minimum(hz_overlap, widths[None, :])

        self.register_buffer("lefts", lefts, persistent=False)
        self.register_buffer("rights", rights, persistent=False)
        self.register_buffer("widths", widths, persistent=False)
        self.register_buffer("int_weights", int_weights, persistent=False)
        self.register_buffer("hz_t", hz_t, persistent=False)
        self.register_buffer("hz_overlap", hz_overlap, persistent=False)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        eta_flat = inputs.float()
        targets = targets.float()
        if targets.ndim == 1:
            targets = targets[:, None]
        bsz = eta_flat.shape[0]
        num_int = self.num_intervals
        if eta_flat.shape[-1] % num_int != 0:
            raise ValueError(f"SurvivEHRPiecewiseLoss: logits dim {eta_flat.shape[-1]} not divisible by M={num_int}")
        num_heads = eta_flat.shape[-1] // num_int
        eta = eta_flat.view(bsz, num_heads, num_int).clamp(min=-18.0, max=2.0)
        lam = torch.exp(eta)

        if targets.shape[-1] == 2 * num_heads + 1:
            ipw_all = targets[:, 2 * num_heads].clamp(min=0.1, max=200.0)
        elif targets.shape[-1] == 2 * num_heads:
            ipw_all = torch.ones(bsz, device=eta.device, dtype=torch.float32)
        else:
            raise ValueError(
                f"SurvivEHRPiecewiseLoss expected {2 * num_heads} or {2 * num_heads + 1} target cols, got {targets.shape[-1]}"
            )

        lefts = self.lefts.to(eta.device)
        rights = self.rights.to(eta.device)
        widths = self.widths.to(eta.device)
        int_w = self.int_weights.to(eta.device)
        hz_t = self.hz_t.to(eta.device)
        hz_ov = self.hz_overlap.to(eta.device)

        head_losses = []
        for k in range(num_heads):
            times = targets[:, 2 * k]
            statuses = targets[:, 2 * k + 1]
            valid = (times != MASK) & (statuses != MASK) & (times >= 0.0)
            if not bool(valid.any()):
                continue

            t_v = times[valid]
            s_v = statuses[valid]
            w_v = ipw_all[valid]
            eta_v = eta[valid, k, :]  # [N_v, M]
            lam_v = lam[valid, k, :]  # [N_v, M]

            # Exact continuous exposure days per interval [N_v, M]
            t_eff = torch.where((t_v == 0.0) & (s_v == 1.0), torch.full_like(t_v, 0.5), t_v)
            exposure = torch.clamp(t_eff[:, None] - lefts[None, :], min=0.0)
            exposure = torch.minimum(exposure, widths[None, :])

            in_first = (t_v >= lefts[0]) & (t_v <= rights[0])
            in_rest = (t_v[:, None] > lefts[None, 1:]) & (t_v[:, None] <= rights[None, 1:])
            in_bin = torch.cat([in_first[:, None], in_rest], dim=1)
            d_ind = ((s_v == 1.0)[:, None] & in_bin).float()

            # 1) Continuous-time Piecewise-Exponential NLL (per-population IPW mean)
            pc_sample = ((lam_v * exposure - d_ind * eta_v) * int_w[None, :]).sum(dim=1)
            w_denom = w_v.sum().clamp(min=1.0)
            loss_pc = (w_v * pc_sample).sum() / w_denom

            # 2) Cumulative-incidence proper scoring rule at reporting horizons h_j
            cum_h = lam_v @ hz_ov.t()  # [N_v, J]
            is_pos_h = (s_v[:, None] == 1.0) & (t_v[:, None] <= hz_t[None, :])
            is_neg_h = t_v[:, None] > hz_t[None, :]
            known_h = is_pos_h | is_neg_h
            # Numerically exact -log(1 - exp(-H)) for positives and +H for event-free controls
            pos_nll = -torch.log((-torch.expm1(-cum_h.clamp(min=1e-7))).clamp(min=1e-12))
            cif_mat = torch.where(is_pos_h, pos_nll, torch.where(is_neg_h, cum_h, torch.zeros_like(cum_h)))
            w_mat = w_v[:, None] * known_h.float()
            loss_cif = (w_mat * cif_mat).sum() / w_mat.sum().clamp(min=1.0)

            head_losses.append((1.0 - self.cif_weight) * loss_pc + self.cif_weight * loss_cif)

        if not head_losses:
            return 0.0 * eta_flat.sum()

        total_loss = torch.stack(head_losses).mean()
        if self.smooth_weight > 0.0 and num_int > 1 and bsz > 1:
            centered = eta - eta.mean(dim=0, keepdim=True)
            smooth = ((centered[:, :, 1:] - centered[:, :, :-1]) ** 2).mean()
            total_loss = total_loss + self.smooth_weight * smooth
        return total_loss


def evaluate_pc_hazard_curves(
    logits_2d: np.ndarray,
    num_heads: int,
    time_grid: list[float] | np.ndarray,
    horizons: list[int] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Analytically evaluate closed-form survival curves from `SurvivEHRHead` logits:
      Returns:
        - `surv_at_horizons`: `[N, K, H]` exact survival `S_k(h | x) = exp(-H_k(h | x))`
        - `probs_at_horizons`: `[N, K, H]` exact cumulative incidence `1 - S_k(h | x)`
        - `rmst_risk_scores`: `[N, K]` `-RMST_k(max_h | x)` (higher = higher risk, SurvivEHR Eq. 20)
        - `median_times`: `[N, K]` exact piecewise-interpolated median time `S_k(t) = 0.5` (`NaN` if > tau_M)
    """
    grid = np.asarray(time_grid, dtype=np.float64)
    lefts, rights = grid[:-1], grid[1:]
    widths = rights - lefts
    num_int = len(widths)
    hz = np.asarray(horizons, dtype=np.float64)
    n_rows = logits_2d.shape[0]

    eta = np.clip(np.asarray(logits_2d, dtype=np.float64).reshape(n_rows, num_heads, num_int), -18.0, 2.0)
    lam = np.exp(eta)  # [N, K, M]

    # Cumulative hazard at reporting horizons: overlap [H, M]
    hz_ov = np.minimum(np.clip(hz[:, None] - lefts[None, :], 0.0, None), widths[None, :])  # [H, M]
    cum_h_hz = np.einsum("nkm,hm->nkh", lam, hz_ov)
    surv_hz = np.exp(-cum_h_hz)
    probs_hz = np.clip(1.0 - surv_hz, 0.0, 1.0)

    # Exact closed-form Restricted Mean Survival Time up to max(horizons)
    max_h = float(np.max(hz)) if len(hz) > 0 else float(rights[-1])
    rmst_ov = np.minimum(np.clip(max_h - lefts, 0.0, None), widths)  # [M]
    inc_h = lam * widths[None, None, :]  # [N, K, M]
    cum_h_knots = np.concatenate([np.zeros((n_rows, num_heads, 1)), np.cumsum(inc_h, axis=2)], axis=2)
    s_start = np.exp(-cum_h_knots[:, :, :-1])  # [N, K, M]
    lam_dt = lam * rmst_ov[None, None, :]
    # Stable (1 - exp(-x)) / x * dt
    ratio = np.where(lam_dt > 1e-6, -np.expm1(-lam_dt) / np.maximum(lam, 1e-12), rmst_ov[None, None, :])
    rmst = np.sum(s_start * ratio, axis=2)  # [N, K]
    rmst_risks = -rmst

    # Exact piecewise-interpolated median survival time (H_k(t) = ln 2)
    ln2 = math.log(2.0)
    total_h = cum_h_knots[:, :, -1]
    median_times = np.full((n_rows, num_heads), np.nan, dtype=np.float64)
    reached = total_h >= ln2
    if np.any(reached):
        # First interval m where cum_h_knots[:, :, m+1] >= ln2
        crossed = cum_h_knots[:, :, 1:] >= ln2
        m_star = np.argmax(crossed, axis=2)  # [N, K]
        h_prev = np.take_along_axis(cum_h_knots[:, :, :-1], m_star[:, :, None], axis=2)[..., 0]
        lam_star = np.take_along_axis(lam, m_star[:, :, None], axis=2)[..., 0]
        t_left = lefts[m_star]
        t_med = t_left + np.maximum(0.0, ln2 - h_prev) / np.maximum(lam_star, 1e-12)
        median_times = np.where(reached, t_med, np.nan)

    return surv_hz, probs_hz, rmst_risks, median_times


class CoxNLLLoss(nn.Module):
    """
    Cox proportional hazards negative log partial likelihood with Breslow
    tie handling. Supports both single-task `[B, 1]` and multi-task `[B, K]`.
    """

    MASK = -100.0

    def __init__(self, reduction: str = "events", max_risk: float = 30.0):
        super().__init__()
        if reduction not in ("events", "samples", "sum"):
            raise ValueError("reduction must be one of 'events', 'samples', 'sum'")
        self.reduction = reduction
        self.max_risk = max_risk

    def _single_head_loss(
        self,
        risk: torch.Tensor,
        times: torch.Tensor,
        statuses: torch.Tensor,
    ) -> tuple[torch.Tensor, bool]:
        valid = (times != self.MASK) & (statuses != self.MASK) & (times >= 0)
        risk, times, statuses = risk[valid], times[valid], statuses[valid]

        n_events = int((statuses == 1).sum().item())
        if n_events == 0:
            return 0.0 * risk.sum(), False

        risk = (risk - risk.max()).clamp(min=-self.max_risk)
        order = torch.argsort(times, descending=False)
        risk_s = risk[order]
        times_s = times[order]
        events_s = statuses[order]

        exp_risk = torch.exp(risk_s)
        rev_csum = torch.cumsum(exp_risk.flip(0), dim=0).flip(0)

        _, inverse, counts = torch.unique(times_s, return_inverse=True, return_counts=True)
        block_start_idx = torch.cumsum(counts, dim=0) - counts
        block_start_rev_csum = rev_csum[block_start_idx]
        risk_set_sum = block_start_rev_csum[inverse]

        nll = events_s * (torch.log(risk_set_sum.clamp(min=1e-12)) - risk_s)
        if self.reduction == "sum":
            return nll.sum(), True
        if self.reduction == "samples":
            return nll.sum() / max(len(nll), 1), True
        return nll.sum() / max(n_events, 1), True

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()
        if targets.ndim == 1:
            targets = targets[:, None]
        if inputs.ndim == 1 or inputs.shape[-1] == 1:
            loss_val, _ = self._single_head_loss(inputs.reshape(-1).float(), targets[:, 0], targets[:, 1])
            return loss_val

        num_heads = inputs.shape[-1]
        if targets.shape[-1] < 2 * num_heads:
            raise ValueError(
                f"Multi-task CoxNLLLoss expected targets shape [B, >={2 * num_heads}] for {num_heads} heads, "
                f"got {tuple(targets.shape)}."
            )
        active_losses = []
        for k in range(num_heads):
            head_loss, has_events = self._single_head_loss(
                inputs[:, k].float(), targets[:, 2 * k], targets[:, 2 * k + 1]
            )
            if has_events:
                active_losses.append(head_loss)
        if not active_losses:
            return 0.0 * inputs.float().sum()
        return torch.stack(active_losses).mean()


def make_survival_loss_func(loss_name: str, loss_args: dict | None = None):
    """
    Factory creating a `compute_loss_func` for the HuggingFace Trainer.

    Supported names:
      - "surv_ehr" / "pc_hazard" / "neural_cif": SurvivEHRPiecewiseLoss
      - "cox": CoxNLLLoss (Breslow partial likelihood)
    """
    if loss_args is None:
        loss_args = {}
    loss_name = loss_name.lower()

    if loss_name in ("surv_ehr", "pc_hazard", "neural_cif"):
        loss_cls = SurvivEHRPiecewiseLoss
    elif loss_name == "cox":
        loss_cls = CoxNLLLoss
    else:
        raise ValueError(f"Survival loss name '{loss_name}' not recognized.")

    sig = inspect.signature(loss_cls)
    filtered_args = {k: v for k, v in loss_args.items() if k in sig.parameters}
    loss_fct = loss_cls(**filtered_args)

    def compute_loss(outputs, labels, num_items_in_batch=None):
        if isinstance(outputs, dict):
            logits = outputs["logits"]
        else:
            logits = outputs[0]
        if labels.dtype != torch.float:
            labels = labels.float()
        return loss_fct(logits, labels)

    return compute_loss
