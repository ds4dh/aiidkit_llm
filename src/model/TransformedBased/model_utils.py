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


class FocalLoss(nn.Module):
    """
    Implemented with Gemini from the paper 'Focal Loss for Dense Object Detection'

    It is a dynamically scaled cross-entropy loss, where the scaling factor
    decays to zero as confidence in the correct class increases. This helps
    focus training on hard, misclassified examples.

    Args:
        gamma (float): The focusing parameter. A higher value gives more weight
            to hard-to-classify examples. Default: 2.0
        alpha (Optional[torch.Tensor]): A manual rescaling weight given to each
            class. If given, it has to be a Tensor of size C (number of classes)
        reduction (str): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'. Default: 'mean'
    """
    def __init__(self, gamma=2.0, alpha=None, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs (torch.Tensor): The model's raw logits, shape (N, C)
            targets (torch.Tensor): The ground truth labels shape (N,)
        """
        # Calculate the cross-entropy loss, but without reduction
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")

        # Calculate the Focal Loss
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt)**self.gamma * ce_loss

        # Apply alpha weighting if provided
        if self.alpha is not None:
            if self.alpha.device != focal_loss.device:
                self.alpha = self.alpha.to(focal_loss.device)

            # Gather the alpha values corresponding to the targets
            alpha_t = self.alpha.gather(0, targets)
            focal_loss = alpha_t * focal_loss

        # Apply the specified reduction
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


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


class SupervisedTaskWeightSchedulerCallback(TrainerCallback):
    """
    A callback to handle annealing a weight during training.
    """
    def __init__(self, start_steps, end_steps, start_value, end_value, strategy="linear"):
        self.start_steps = start_steps
        self.end_steps = end_steps
        self.start_value = start_value
        self.end_value = end_value
        self.strategy = strategy
        self.current_weight = start_value

    def on_step_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """
        Called at the beginning of a training step.
        """
        model = kwargs.get("model")
        if model is None:
            return

        current_step = state.global_step

        # Before annealing starts (during warmup)
        if current_step < self.start_steps:
            self.current_weight = self.start_value

        # After annealing is complete
        elif current_step >= self.end_steps:
            self.current_weight = self.end_value

        # During the annealing phase
        else:
            annealing_duration = self.end_steps - self.start_steps
            progress = (current_step - self.start_steps) / annealing_duration

            if self.strategy == "linear":
                self.current_weight = self.start_value + progress * (self.end_value - self.start_value)
            
            elif self.strategy == "cosine":
                # Check if we are increasing or decreasing the weight
                if self.start_value < self.end_value:
                    # This creates the "sine up" phase shape. The factor goes from 0 to 1.
                    # It's based on the formula: 0.5 * (1 - cos(pi * progress))
                    cosine_factor = 0.5 * (1.0 - math.cos(math.pi * progress))
                else:
                    # This is the standard "cosine down" decay. The factor goes from 1 to 0.
                    # It's based on the formula: 0.5 * (1 + cos(pi * progress))
                    # We subtract from 1 to make the factor go from 0 to 1 for consistent interpolation.
                    cosine_factor_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
                    cosine_factor = 1.0 - cosine_factor_decay

                # Apply the factor using standard linear interpolation
                self.current_weight = self.start_value + cosine_factor * (self.end_value - self.start_value)

            else:
                # Default to linear if strategy is unknown
                self.current_weight = self.start_value + progress * (self.end_value - self.start_value)

        # Update the model's attribute directly
        if hasattr(model, "supervised_task_weight"):
            model.supervised_task_weight = self.current_weight

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: dict,
        **kwargs,
    ):
        """
        Adds the current supervised task weight to the logs
        """
        if state.is_local_process_zero:
            logs["supervised_task_weight"] = self.current_weight


class WandbPlottingCallback(TrainerCallback):
    """
    Callback acting as a "mailbox" for plots generated during evaluation, logging
    plots to wandb at the correct step
    """
    def __init__(self):
        super().__init__()
        self.plots_to_log = {}  # temporary storage to log at the right time

    def on_log(
        self, 
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: dict[str, float],
        **kwargs,
    ):
        # We only log plots during evaluation, i.e., when an evaluation key is in
        # the `logs` dict, "eval_loss" being a reliable key for this purpose
        if "eval_loss" in logs and self.plots_to_log:

            # Check if there are any plots in our temporary storage
            if self.plots_to_log:

                # Log the plots with the correct global step
                wandb.log(self.plots_to_log, step=state.global_step)

                # Clear the storage to prevent re-logging
                self.plots_to_log = {}
