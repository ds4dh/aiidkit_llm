import wandb
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import TrainerState, TrainerControl, TrainingArguments
from transformers.trainer_callback import TrainerCallback, EarlyStoppingCallback


def preprocess_logits_for_metrics(logits, labels):
    """
    Align logits and labels for metric computation (useful for causal language models)
    """
    # The first element of the tuple is the prediction scores
    if isinstance(logits, tuple): logits = logits[0]

    # For causal LM, last logit not needed for prediction and first label not predicted
    return logits[:, :-1, :].argmax(dim=-1), labels[:, 1:]


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


class FocalLoss(nn.Module):
    """
    Focal Loss for Dense Object Detection (He et al.)
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

        # Prepare alpha
        if isinstance(alpha, float):
            self.register_buffer("alpha", torch.tensor([1 - alpha, alpha]))
        elif isinstance(alpha, list):
            self.register_buffer("alpha", torch.tensor(alpha))
        else:
            self.register_buffer("alpha", alpha) # Assumed Tensor or None

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # standard cross entropy (no reduction yet)
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")

        pt = torch.exp(-ce_loss)
        focal_term = (1 - pt) ** self.gamma
        loss = focal_term * ce_loss

        if self.alpha is not None:
            # Alpha is registered buffer, so it matches device automatically
            alpha_t = self.alpha.gather(0, targets)
            loss = alpha_t * loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class WeightedCELoss(nn.Module):
    """
    Standard Cross Entropy but with explicit class weights for imbalance.
    Args:
        class_weights (list | Tensor): Weights for each class (e.g., [1.0, 10.0])
    """
    def __init__(self, class_weights: list | torch.Tensor = None, reduction: str = "mean"):
        super().__init__()
        if class_weights is not None:
            if isinstance(class_weights, list):
                class_weights = torch.tensor(class_weights)
            # Register as buffer to handle device placement automatically
            self.register_buffer("weight", class_weights)
        else:
            self.weight = None
        
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(inputs, targets, weight=self.weight, reduction=self.reduction)


class DiceLoss(nn.Module):
    """
    Dice Loss for checking overlap. Excellent for strong imbalance.
    Calculates 1 - DiceCoefficient.
    """
    def __init__(self, smooth: float = 1e-6, square_denominator: bool = False, reduction: str = "mean"):
        super().__init__()
        self.smooth = smooth
        self.square_denominator = square_denominator
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        inputs: Logits [B, C]
        targets: Labels [B]
        """
        # Get probabilities
        probs = F.softmax(inputs, dim=1)
        
        # One-hot encode targets
        num_classes = inputs.shape[1]
        targets_one_hot = F.one_hot(targets, num_classes=num_classes)
        
        # Intersection: sum(probs * targets)
        intersection = (probs * targets_one_hot).sum(dim=1)
        
        if self.square_denominator:
            # Soft dice approach (squares in denominator)
            cardinality = (probs ** 2).sum(dim=1) + (targets_one_hot ** 2).sum(dim=1)
        else:
            # Standard dice
            cardinality = probs.sum(dim=1) + targets_one_hot.sum(dim=1)

        dice_score = (2. * intersection + self.smooth) / (cardinality + self.smooth)

        # We want to minimize loss, so loss = 1 - dice
        loss = 1.0 - dice_score

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class Poly1Loss(nn.Module):
    """
    Poly-1 Loss: A polynomial expansion of Cross Entropy.
    L = CE + epsilon * (1 - pt)
    Provides a heavier tail than Focal Loss, often more stable.
    """
    def __init__(self, epsilon: float = 1.0, class_weights: list | torch.Tensor = None, reduction: str = "mean"):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction
        
        if class_weights is not None:
            if isinstance(class_weights, list):
                class_weights = torch.tensor(class_weights)
            self.register_buffer("weight", class_weights)
        else:
            self.weight = None

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Cross-entropy term
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction="none")
        
        # Poly-1 term: epsilon * (1 - pt), pt being probability of the correct class
        probs = F.softmax(inputs, dim=1)
        pt = probs.gather(1, targets.view(-1, 1)).squeeze()
        
        poly_term = self.epsilon * (1 - pt)
        
        loss = ce_loss + poly_term
        
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


def make_loss_func(loss_name: str, loss_args: dict = None):
    """
    Factory to create a compute_loss function for HF Trainer.
    
    Args:
        loss_name: 'focal', 'weighted_ce', 'dice', 'poly1'
        loss_args: Dictionary of arguments for the specific loss class.
    """
    if loss_args is None:
        loss_args = {}

    # Instantiate the correct Loss Module
    if loss_name.lower() == "focal":
        loss_fct = FocalLoss(**loss_args)
    elif loss_name.lower() in ["weighted_ce", "weighted"]:
        loss_fct = WeightedCELoss(**loss_args)
    elif loss_name.lower() == "dice":
        loss_fct = DiceLoss(**loss_args)
    elif loss_name.lower() in ["poly1", "poly"]:
        loss_fct = Poly1Loss(**loss_args)
    else:
        raise ValueError(f"Loss name '{loss_name}' not recognized.")

    # Define the inner function compatible with Hugging Face Trainer
    def compute_loss(outputs, labels, num_items_in_batch=None):
        # Handle HF ModelOutput object or tuple
        if isinstance(outputs, dict):
            logits = outputs["logits"]
        else:
            # Fallback for tuple outputs
            logits = outputs[0]
        
        return loss_fct(logits, labels)

    return compute_loss