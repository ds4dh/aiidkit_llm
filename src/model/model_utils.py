import math
import inspect
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any
from datasets import DatasetDict
from transformers import TrainerState, TrainerControl, TrainingArguments
from transformers.trainer_callback import EarlyStoppingCallback


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


class PositionalEncoding(nn.Module):
    """
    Encode token position using sines and cosines of different frequencies
    """
    def __init__(
        self,
        embedding_dim: int,
        dropout: float=0.1,
        max_len: int=1000,  # in AIIDKIT data, longest sequence has 843 events
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2) * (-math.log(10000.0) / embedding_dim))
        
        pe = torch.zeros(max_len, 1, embedding_dim)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer("pe", pe.transpose(0, 1))  # shape [1, max_len, embedding_dim]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to the input tensor, with dropout

        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# -------------------------------------------------------------------
# Potentially useful custom callbacks
# -------------------------------------------------------------------

def preprocess_logits_for_metrics(logits, labels):
    """
    Align logits and labels for metric computation (useful for causal language models)
    """
    # The first element of the tuple is the prediction scores
    if isinstance(logits, tuple): logits = logits[0]

    # For causal LM, last logit not needed for prediction and first label not predicted
    return logits[:, :-1, :].argmax(dim=-1), labels[:, 1:]


class EarlyStoppingCallbackWithWarmup(EarlyStoppingCallback):
    """
    An EarlyStoppingCallback that disables early stopping for some warm-up steps
    """
    def __init__(self, warmup_steps: int, early_stopping_patience: int = 1, early_stopping_threshold: float = 0.0):
        # Initialize the parent class with its parameters
        super().__init__(early_stopping_patience, early_stopping_threshold)
        
        # Store the new warmup_steps parameter
        self.warmup_steps = warmup_steps

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics,
        **kwargs,
    ):
        """
        Overrides the on_evaluate method to check for the warmup condition.
        """
        if state.global_step < self.warmup_steps:
            self.early_stopping_patience_counter = 0
            return

        # If the warm-up phase is over, execute the original early stopping logic
        super().on_evaluate(args, state, control, metrics, **kwargs)


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

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
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

        # Apply alpha (simple scalar weighting)
        if self.alpha is not None:
             if self.alpha.device != inputs.device:
                 self.alpha = self.alpha.to(inputs.device)
             loss = loss * self.alpha
        
        # Apply mask to zero out contributions from invalid labels
        loss = loss * valid_mask

        # Reduction
        if self.reduction == "mean":
            # Sum loss and divide by the number of valid elements
            return loss.sum() / valid_mask.sum().clamp(min=1.0)
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

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
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
            return loss.sum() / valid_mask.sum().clamp(min=1.0)
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

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
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

        # Alpha weighting
        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            loss = loss * self.alpha

        # Apply mask
        loss = loss * valid_mask

        # Reduction
        if self.reduction == "mean":
            return loss.sum() / valid_mask.sum().clamp(min=1.0)
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
        labels = np.array(dataset[key])
        labels = labels[labels != -100]  # filter valid (-100) if not done
        n_pos = np.sum(labels == 1)
        n_neg = np.sum(labels == 0)
        pos_counts.append(n_pos)
        neg_counts.append(n_neg)

    print(f"Label statistics (pos/neg):")
    for k, p, n in zip(label_keys, pos_counts, neg_counts):
        print(f"  {k}: {p}/{n} (ratio: {n/max(p,1):.1f})")

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
            
        # The loss classes now handle masking of -100 internally
        return loss_fct(logits, labels)

    return compute_loss