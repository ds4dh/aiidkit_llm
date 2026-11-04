"""
Survival analysis utilities: losses used to fine-tune the patient transformer
with a time-to-event (survival) objective instead of binary horizon labels.

Fine-tuning setup (mirrors the binary classification pipeline):
    - backbone: same patched PatientEmbeddingModel (AutoModelForSequenceClassification)
    - head: single logit per patient -> log hazard ratio  r(x)
    - loss: Cox proportional hazards negative log partial likelihood (Breslow)
    - labels: float tensor of shape [batch, 2] with columns (time, status)
"""

import inspect
import torch
import torch.nn as nn


class CoxNLLLoss(nn.Module):
    """
    Cox proportional hazards negative log partial likelihood with Breslow
    tie handling. Operates on a batch of risk scores (log hazard ratios).

    Args:
        inputs  : logits [B, 1] or [B]              -> risk score r(x_i)
        targets : labels [B, 2] (time, status)      -> may contain -100 (masked)

    Notes:
        - Handles right-censoring: only events (status == 1) contribute a term.
        - Ties in event times are handled with the Breslow approximation.
        - Numerically stabilized by shifting risk scores (partial likelihood
          is invariant to a constant shift).
    """

    MASK = -100.0

    def __init__(self, reduction: str = "events", max_risk: float = 30.0):
        super().__init__()
        if reduction not in ("events", "samples", "sum"):
            raise ValueError("reduction must be one of 'events', 'samples', 'sum'")
        self.reduction = reduction
        self.max_risk = max_risk

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Flatten logits, cast to float32 for numerical stability
        risk = inputs.reshape(-1).float()
        targets = targets.float()
        if targets.ndim == 1:
            targets = targets[:, None]  # defensive: shouldn't happen
        times = targets[:, 0]
        statuses = targets[:, 1]

        # Valid rows: known time and status
        valid = (times != self.MASK) & (statuses != self.MASK) & (times >= 0)
        risk, times, statuses = risk[valid], times[valid], statuses[valid]

        # Keep the autograd graph alive when the batch has no events (censored batches)
        n_events = int((statuses == 1).sum().item())
        if n_events == 0:
            return 0.0 * risk.sum()

        # Constant shift invariance -> stabilize exp()
        risk = (risk - risk.max()).clamp(min=-self.max_risk)

        # Sort ascending by time; risk set R(t) = {j : t_j >= t}
        order = torch.argsort(times, descending=False)
        risk_s = risk[order]
        times_s = times[order]
        events_s = statuses[order]

        exp_risk = torch.exp(risk_s)
        # Risk set R(t) = {j : t_j >= t} -> reverse cumulative sum (t_s ascending)
        rev_csum = torch.cumsum(exp_risk.flip(0), dim=0).flip(0)

        # Breslow: risk set sums must include all tied-time samples, so take the
        # reverse cumulative sum at the START of each tied-time block, then scatter back.
        unique_times, inverse, counts = torch.unique(
            times_s, return_inverse=True, return_counts=True
        )
        block_start_idx = torch.cumsum(counts, dim=0) - counts
        block_start_rev_csum = rev_csum[block_start_idx]
        risk_set_sum = block_start_rev_csum[inverse]

        # Breslow negative log partial likelihood (per event)
        nll = events_s * (torch.log(risk_set_sum.clamp(min=1e-12)) - risk_s)

        if self.reduction == "sum":
            return nll.sum()
        if self.reduction == "samples":
            return nll.sum() / max(len(nll), 1)
        return nll.sum() / max(n_events, 1)  # mean over events


def make_survival_loss_func(loss_name: str, loss_args: dict | None = None):
    """
    Factory creating a `compute_loss_func` for the HuggingFace Trainer,
    with the same signature as `make_loss_func` in model_utils.py.

    Supported names:
        - "cox" : Cox NLL partial likelihood (Breslow ties)
    """
    if loss_args is None:
        loss_args = {}
    loss_name = loss_name.lower()

    if loss_name == "cox":
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
