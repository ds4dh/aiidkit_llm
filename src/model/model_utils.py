import math
import inspect
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any
from datasets import DatasetDict
from transformers import TrainerState, TrainerControl, TrainingArguments
from transformers.trainer_callback import EarlyStoppingCallback


# -------------------------------------------------------------------
#  Positional and time embedding layers
# -------------------------------------------------------------------

class TimeEmbedding(nn.Module):
    """
    Create a sinusoidal time embedding for a tensor of integers.
    """
    def __init__(
        self,
        embedding_dim: int,
        dropout: float=0.1,
        time_scale: int=10000,
    ):
        super().__init__()
        if embedding_dim % 2 != 0:
            raise ValueError("Embedding dimension must be even")
        self.dropout = nn.Dropout(p=dropout)
        self.half_dim = embedding_dim // 2
        self.ratio = math.log(time_scale) / self.half_dim

    def forward(self, x: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
        """
        Add time embedding to the input tensor, with dropout

        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
            times: Tensor, shape [batch_size, seq_len]
        """
        freq_indices = torch.arange(self.half_dim, device=times.device)
        times_scaled = times.unsqueeze(-1) * torch.exp(-self.ratio * freq_indices)
        time_embeddings = torch.cat([torch.sin(times_scaled), torch.cos(times_scaled)], dim=-1)
        
        return self.dropout(x + time_embeddings)


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
#  Potentially useful custom callbacks
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
#  Loss Functions (refactored for multi-label / BCE)
# -------------------------------------------------------------------

class FocalLoss(nn.Module):
    """
    Binary Focal Loss for Multi-Label Classification.
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
        targets: Binary Targets [Batch, Num_Labels] (float)
        """
        # Compute binary cross entropy (element-wise)
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        
        # Compute probabilities (pt) for the correct class
        # - if target=1, pt = sigmoid(input)
        # - if target=0, pt = 1 - sigmoid(input)
        p = torch.sigmoid(inputs)
        pt = p * targets + (1 - p) * (1 - targets)
        
        # Focal term: (1 - pt)^gamma
        focal_term = (1 - pt) ** self.gamma
        loss = focal_term * bce_loss

        # Apply alpha (simple scalar weighting)
        if self.alpha is not None:
             if self.alpha.device != inputs.device:
                 self.alpha = self.alpha.to(inputs.device)
             loss = loss * self.alpha
             
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class WeightedCELoss(nn.Module):
    """
    Binary Cross Entropy with specific positive weights for imbalance.
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
        weight = self.pos_weight
        if weight is not None and weight.device != inputs.device:
            self.pos_weight = weight.to(inputs.device)
            weight = self.pos_weight
        return F.binary_cross_entropy_with_logits(
            inputs, 
            targets, 
            pos_weight=weight, 
            reduction=self.reduction
        )


class Poly1FocalLoss(nn.Module):
    """
    Poly1 Focal Loss adapted for Multi-Label (Binary) Classification.
    Loss = FL(pt) + epsilon * (1 - pt)
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
        # Base binary focal loss
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        p = torch.sigmoid(inputs)
        pt = p * targets + (1 - p) * (1 - targets)
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

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


# -------------------------------------------------------------------
#  Factory
# -------------------------------------------------------------------


def compute_loss_args(dataset: DatasetDict, label_keys: list[str]) -> dict[str, Any]:
    """ 
    Compute robust class weights for Multi-Label Poly1 Focal Loss.
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

    # Compute weights
    # - standard BCE uses neg/pos (linear). 
    # - focal loss / poly1 works best with sqrt(neg/pos).
    raw_ratios = [n / max(p, 1) for n, p in zip(neg_counts, pos_counts)]
    dampened_weights = [np.sqrt(r) for r in raw_ratios]

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
        loss_cls = nn.BCEWithLogitsLoss
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
            
        return loss_fct(logits, labels)

    return compute_loss