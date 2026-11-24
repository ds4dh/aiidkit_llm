import json
import math
import inspect
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any
try:
    from datasets import DatasetDict
except ImportError:
    DatasetDict = Any


# -------------------------------------------------------------------
# Positional and time embedding layers
# -------------------------------------------------------------------

class TimeEmbedding(nn.Module):
    """
    Create a sinusoidal time embedding for a tensor of integers.
    """
    def __init__(
        self,
        embedding_dim: int,
        time_scale: int=10000,
    ):
        super().__init__()
        if embedding_dim % 2 != 0:
            raise ValueError("Embedding dimension must be even")
        self.half_dim = embedding_dim // 2
        self.ratio = math.log(time_scale) / self.half_dim
        self.register_buffer("inv_freq", torch.exp(-self.ratio * torch.arange(self.half_dim, dtype=torch.float32)), persistent=False)

    def forward(self, times: torch.Tensor) -> torch.Tensor:
        """
        Add time embedding to the input tensor, with dropout

        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
            times: Tensor, shape [batch_size, seq_len]
        """
        inv_freq = self.inv_freq.to(device=times.device)
        times_scaled = times.unsqueeze(-1) * inv_freq
        time_embeddings = torch.cat([torch.sin(times_scaled), torch.cos(times_scaled)], dim=-1)
        
        return time_embeddings


class NumericalValueEmbedding(nn.Module):
    """
    Continuous numerical value embedder for quantile-normalized scalars in [-1, 1].
    Combines direct linear magnitude with multi-scale sinusoidal (Fourier) features
    followed by a 2-layer SiLU MLP and LayerNorm.
    """
    def __init__(
        self,
        embedding_dim: int,
        n_frequencies: int = 64,
        max_freq: float = 16.0,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.n_frequencies = n_frequencies
        freqs = torch.exp(
            torch.linspace(
                math.log(0.5 * math.pi),
                math.log(max_freq * math.pi),
                n_frequencies,
                dtype=torch.float32,
            )
        )
        self.register_buffer("freqs", freqs, persistent=False)
        in_dim = 1 + 2 * n_frequencies
        self.proj = nn.Sequential(
            nn.Linear(in_dim, embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            values: Tensor of shape [batch_size, seq_len] with normalized values in [-1, 1].
        Returns:
            Tensor of shape [batch_size, seq_len, embedding_dim].
        """
        v = values.float().unsqueeze(-1)
        freqs = self.freqs.to(device=values.device, dtype=torch.float32)
        angles = v * freqs
        fourier = torch.cat([v, torch.sin(angles), torch.cos(angles)], dim=-1)
        target_dtype = self.proj[0].weight.dtype
        return self.proj(fourier.to(dtype=target_dtype))


# -------------------------------------------------------------------
# Shared Checkpoint & Multi-Task Monotonicity Helpers
# -------------------------------------------------------------------

def resolve_best_checkpoint(run_dir: str | Path) -> Path | None:
    """Resolve the best HF Trainer checkpoint inside `run_dir` via `trainer_state.json`.

    Checks `trainer_state.json['best_model_checkpoint']` in the latest `checkpoint-*`
    folder (remapping paths if the run folder was moved across mounts), falls back to
    the highest-numbered `checkpoint-*`, or `run_dir` itself if `model.safetensors` /
    `config.json` sits directly inside `run_dir`.
    """
    rd = Path(run_dir)
    if not rd.exists():
        return None
    if rd.name.startswith("checkpoint-") and (rd / "config.json").exists():
        return rd

    ckpts = sorted(
        (p for p in rd.glob("checkpoint-*") if p.is_dir()),
        key=lambda p: int(p.name.split("-")[-1]) if p.name.split("-")[-1].isdigit() else -1,
    )
    if ckpts:
        state_file = ckpts[-1] / "trainer_state.json"
        if state_file.exists():
            try:
                best_raw = json.loads(state_file.read_text(encoding="utf-8")).get("best_model_checkpoint")
                if best_raw:
                    best_p = Path(best_raw)
                    if best_p.exists():
                        return best_p
                    remapped = rd / best_p.name
                    if remapped.exists():
                        return remapped
            except Exception:
                pass
        return ckpts[-1]

    if (rd / "config.json").exists():
        return rd
    return None


def build_monotonic_groups(
    run_meta: dict | None,
    fallback_monotonic: bool | list[int] = False,
) -> bool | list[int]:
    """Compute per-task horizon group sizes `[n_h_t1, n_h_t2, ...]` for `enforce_monotonicity`.

    Handles both uniform `horizons` and heterogeneous `task_horizons` in `run_metadata.json`,
    and automatically disables monotonicity when `fallback_monotonic` is falsy or every task has <= 1 horizon.
    """
    if isinstance(fallback_monotonic, (list, tuple)):
        return [int(g) for g in fallback_monotonic]
    if not fallback_monotonic:
        return False
    if not run_meta:
        return bool(fallback_monotonic)

    tasks = run_meta.get("tasks") or []
    default_hs = run_meta.get("horizons") or []
    task_hs = run_meta.get("task_horizons") or {}
    if len(tasks) > 1:
        groups = []
        for t in tasks:
            hs = task_hs.get(t, default_hs)
            groups.append(len(hs) if isinstance(hs, list) else 1)
        return groups if any(g > 1 for g in groups) else False
    if len(tasks) == 1:
        hs = task_hs.get(tasks[0], default_hs)
        n_h = len(hs) if isinstance(hs, list) else 1
        return True if n_h > 1 else False
    return bool(fallback_monotonic)


# -------------------------------------------------------------------
# Gradient clipping: observability + non-finite gradient guard
# -------------------------------------------------------------------

class GradientClippingGuard:
    """
    Mixin for HF's Trainer that makes gradient clipping observable and safe.

    Context: the Trainer logs the *pre-clip* total gradient norm and clips to
    `args.max_grad_norm` (1.0 by default) afterwards, so logs alone cannot tell
    a run where clipping never fires apart from one where every step is
    rescaled by ~100x. On top of that, a single bf16 overflow (total norm = inf
    or nan) is *not* fixed by clipping: the clip coefficient itself becomes
    nan/inf and the poisoned gradient lands in the Adam moments.

    This mixin adds, with no change to the optimisation math:
      * `clip_*` metrics on every logging step: how often the cap fired
        (`clip_frac`), how much of the gradient survived it (`clip_ratio`, 1.0
        when never clipped) and the pre-clip norm distribution seen in the
        window (`grad_norm_med`, `grad_norm_max`).
      * a guard that drops the whole gradient (p.grad = None, so AdamW leaves
        weights *and* moments untouched for that step instead of applying stale
        momentum) whenever the total norm is non-finite, counted as
        `grad_skipped`. DeepSpeed/FSDP report no norm and are left untouched.
    """

    _GC_WINDOW = 1024  # optimizer steps kept for the median/max statistics

    def _gc_state(self) -> dict:
        # Lazily initialised so the mixin never has to touch __init__/MRO.
        if not hasattr(self, "_gc"):
            self._gc = {
                "norms": deque(maxlen=self._GC_WINDOW),
                "steps": 0, "clipped": 0, "skipped": 0, "ratio_sum": 0.0,
            }
        return self._gc

    def clip_gradient(self, *args, **kwargs):
        total_norm = super().clip_gradient(*args, **kwargs)
        if total_norm is None:  # DeepSpeed clips on its own side
            return total_norm

        model = args[0] if args else kwargs.get("model")
        max_norm = args[1] if len(args) > 1 else kwargs.get("max_norm", kwargs.get("norm"))
        gc = self._gc_state()
        gc["steps"] += 1
        norm = float(total_norm)

        if not math.isfinite(norm):
            # Clipping cannot rescue this step: drop it instead of updating.
            if model is not None:
                for param in model.parameters():
                    if param.grad is not None:
                        param.grad = None
            gc["skipped"] += 1
            if getattr(self, "_grad_norm", None) is not None:
                self._grad_norm = torch.zeros_like(self._grad_norm)  # keep device/dtype
            print(f"[grad-guard] non-finite gradient at step {getattr(self.state, 'global_step', '?')}: update skipped")
            return torch.zeros_like(total_norm) if torch.is_tensor(total_norm) else 0.0

        gc["norms"].append(norm)
        if max_norm and max_norm > 0 and norm > max_norm:
            gc["clipped"] += 1
            gc["ratio_sum"] += max_norm / norm
        else:
            gc["ratio_sum"] += 1.0
        return total_norm

    def log(self, logs, *args, **kwargs):
        gc = self._gc_state()
        if gc["steps"]:
            norms = list(gc["norms"])
            logs = {
                **logs,
                "clip_frac": round(gc["clipped"] / gc["steps"], 3),
                "clip_ratio": round(gc["ratio_sum"] / gc["steps"], 4),
                "grad_norm_med": round(float(np.median(norms)), 2) if norms else 0.0,
                "grad_norm_max": round(max(norms), 2) if norms else 0.0,
                "grad_skipped": gc["skipped"],
            }
            gc.update(steps=0, clipped=0, skipped=0, ratio_sum=0.0)
            gc["norms"].clear()
        return super().log(logs, *args, **kwargs)


# -------------------------------------------------------------------
# Loss Functions (refactored for multi-label / BCE with -100 masking)
# -------------------------------------------------------------------

class FocalLoss(nn.Module):
    """
    Binary Focal Loss for Multi-Label Classification.
    Handles -100 in targets by masking them out.
    """
    def __init__(
        self,
        gamma: float = 2.0,
        alpha: float | list | torch.Tensor = None,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        
        # In multi-label, alpha acts as a positive class weight (pos_weight)
        if isinstance(alpha, (float, int)):
            self.register_buffer("alpha", torch.tensor([alpha]))
        elif isinstance(alpha, (list, tuple)):
            self.register_buffer("alpha", torch.tensor(alpha))
        else:
            self.register_buffer("alpha", alpha)

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        inputs: Logits [Batch, Num_Labels]
        targets: Binary Targets [Batch, Num_Labels] (float), can contain -100
        """
        # Create mask for valid labels
        valid_mask = (targets != -100).float()

        # Create safe targets for calculation (replace -100 with 0 to avoid NaN in BCE)
        safe_targets = targets.clamp(min=0, max=1)

        # Compute binary cross entropy (element-wise)
        bce_loss = F.binary_cross_entropy_with_logits(inputs, safe_targets, reduction="none")
        
        # Compute probabilities (pt) for the correct class
        # - if target=1, pt = sigmoid(input)
        # - if target=0, pt = 1 - sigmoid(input)
        p = torch.sigmoid(inputs)
        pt = p * safe_targets + (1 - p) * (1 - safe_targets)
        
        # Focal term: (1 - pt)^gamma
        focal_term = (1 - pt) ** self.gamma
        loss = focal_term * bce_loss

        # Apply positive-class alpha weighting without mutating buffer in-place
        if self.alpha is not None:
            alpha = self.alpha.to(device=inputs.device, dtype=loss.dtype)
            loss = loss * torch.where(safe_targets > 0.5, alpha, torch.ones_like(loss))
        
        # Apply mask to zero out contributions from invalid labels
        loss = loss * valid_mask

        # Reduction
        if self.reduction == "mean":
            return loss.sum() / valid_mask.sum().clamp(min=1e-6)
        elif self.reduction == "sum":
            return loss.sum()
        
        return loss


class WeightedCELoss(nn.Module):
    """
    Binary Cross Entropy with specific positive weights for imbalance.
    Handles -100 in targets by masking them out.
    """
    def __init__(
        self,
        class_weights: list | torch.Tensor = None,
        reduction: str = "mean",
    ):
        super().__init__()
        if class_weights is not None:
            if isinstance(class_weights, list):
                class_weights = torch.tensor(class_weights)
            self.register_buffer("pos_weight", class_weights)
        else:
            self.pos_weight = None
            
        self.reduction = reduction

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        # Create mask for valid labels
        valid_mask = (targets != -100).float()

        # Create safe targets for calculation
        safe_targets = targets.clamp(min=0, max=1)

        weight = self.pos_weight
        if weight is not None and weight.device != inputs.device:
            self.pos_weight = weight.to(inputs.device)
            weight = self.pos_weight
        
        # Compute element-wise loss with reduction='none'
        loss = F.binary_cross_entropy_with_logits(
            inputs, 
            safe_targets, 
            pos_weight=weight, 
            reduction="none"
        )

        # Apply mask
        loss = loss * valid_mask

        # Reduction
        if self.reduction == "mean":
            return loss.sum() / valid_mask.sum().clamp(min=1e-6)
        elif self.reduction == "sum":
            return loss.sum()
        
        return loss


class Poly1FocalLoss(nn.Module):
    """
    Poly1 Focal Loss adapted for Multi-Label (Binary) Classification.
    Loss = FL(pt) + epsilon * (1 - pt)
    Handles -100 in targets by masking them out.
    """
    def __init__(
        self,
        epsilon: float = 1.0,
        gamma: float = 2.0,
        alpha: float | list | torch.Tensor = None,
        reduction: str = "mean",
    ):
        super().__init__()
        self.epsilon = epsilon
        self.gamma = gamma
        self.reduction = reduction

        if isinstance(alpha, (float, int)):
            self.register_buffer("alpha", torch.tensor([alpha]))
        elif isinstance(alpha, (list, tuple)):
            self.register_buffer("alpha", torch.tensor(alpha))
        else:
            self.register_buffer("alpha", alpha)

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        # Create mask for valid labels
        valid_mask = (targets != -100).float()

        # Create safe targets for calculation
        safe_targets = targets.clamp(min=0, max=1)

        # Base binary focal loss
        bce_loss = F.binary_cross_entropy_with_logits(inputs, safe_targets, reduction="none")
        p = torch.sigmoid(inputs)
        pt = p * safe_targets + (1 - p) * (1 - safe_targets)
        focal_term = (1 - pt) ** self.gamma
        fl_loss = focal_term * bce_loss
        
        # Add poly1 term
        poly1_loss = self.epsilon * (1 - pt)
        loss = fl_loss + poly1_loss

        # Positive-class alpha weighting without mutating buffer in-place
        if self.alpha is not None:
            alpha = self.alpha.to(device=inputs.device, dtype=loss.dtype)
            loss = loss * torch.where(safe_targets > 0.5, alpha, torch.ones_like(loss))

        # Apply mask
        loss = loss * valid_mask

        # Reduction
        if self.reduction == "mean":
            return loss.sum() / valid_mask.sum().clamp(min=1e-6)
        elif self.reduction == "sum":
            return loss.sum()
        
        return loss


# -------------------------------------------------------------------
# Factory
# -------------------------------------------------------------------


def compute_loss_args(
    dataset: DatasetDict,
    label_keys: list[str],
    max_weight: float = 10.0,
) -> dict[str, Any]:
    """ 
    Compute robust class weights for multi-label loss.
    """
    pos_counts = []
    neg_counts = []

    # Count positives and negatives for each label
    for key in label_keys:
        labels = np.asarray(dataset.data.column(key)) if hasattr(dataset, "data") else np.array(dataset[key])
        labels = labels[labels != -100]  # filter valid (-100) if not done
        n_pos = int(np.sum(labels == 1))
        n_neg = int(np.sum(labels == 0))
        pos_counts.append(n_pos)
        neg_counts.append(n_neg)

    print("Label statistics (pos/neg):")
    for k, p, n in zip(label_keys, pos_counts, neg_counts):
        print(f"  {k}: {p}/{n} (ratio: {n/max(p, 1):.1f})")

    # Compute weights (with capping to prevent gradient explosions)
    # - standard BCE uses neg/pos (linear).
    # - focal loss / poly1 works best with sqrt(neg/pos)
    raw_ratios = [n / max(p, 1) for n, p in zip(neg_counts, pos_counts)]
    dampened_weights = [np.sqrt(r) for r in raw_ratios]
    dampened_weights = [min(w, max_weight) for w in dampened_weights]
    raw_ratios = [min(r, max_weight ** 2) for r in raw_ratios]

    return {
        # Used by focal loss / poly1 (dampened to avoid exploding gradients)
        "alpha": torch.tensor(dampened_weights, dtype=torch.float),
        
        # Used by WeightedBCE
        "class_weights": torch.tensor(raw_ratios, dtype=torch.float),
        
        "gamma": 2.0, 
        "epsilon": 1.0,
    }


def make_loss_func(loss_name: str, loss_args: dict = None):
    """
    Factory to create a compute_loss function for HF Trainer (Multi-Label).
    Automatically filters loss_args to match the chosen loss class signature.
    """
    if loss_args is None: loss_args = {}
    loss_name = loss_name.lower()

    # Select the class (but don't instantiate it yet)
    if loss_name == "ce":
        # loss_cls = nn.BCEWithLogitsLoss
        loss_cls = WeightedCELoss
        loss_args["class_weights"] = None  # equivalent to classic CE loss
    elif loss_name == "dampened_ce":
        loss_cls = WeightedCELoss
        loss_args["class_weights"] = loss_args.get("alpha")
    elif loss_name == "weighted_ce":
        loss_cls = WeightedCELoss
    elif loss_name == "focal":
        loss_cls = FocalLoss
    elif loss_name in ["poly1", "poly"]:
        loss_cls = Poly1FocalLoss
    else:
        raise ValueError(f"Loss name '{loss_name}' not recognized.")

    # Instantiate loss function with clean arguments
    sig = inspect.signature(loss_cls)
    filtered_args = {k: v for k, v in loss_args.items() if k in sig.parameters}
    loss_fct = loss_cls(**filtered_args)

    # Define the wrapper function required by HF Trainer
    def compute_loss(outputs, labels, num_items_in_batch=None):
        if isinstance(outputs, dict):
            logits = outputs["logits"]
        else:
            logits = outputs[0]
            
        # Ensure labels are float for binary cross-entropy
        if labels.dtype != torch.float:
            labels = labels.float()
            
        # The loss classes handle masking of -100 internally
        return loss_fct(logits, labels)

    return compute_loss