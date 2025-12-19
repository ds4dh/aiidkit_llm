import wandb
import numpy as np
import torch
from transformers.trainer_utils import EvalPrediction

import optuna
from optuna.samplers import TPESampler
optuna.logging.set_verbosity(optuna.logging.WARNING)
import umap
import hdbscan
import cupy as cp
import cuml
import gc
import io
from contextlib import redirect_stdout

import plotly.graph_objects as go
from plotly.colors import qualitative
from scipy.special import softmax
from sklearn.metrics import (
    silhouette_score, adjusted_mutual_info_score,
    roc_auc_score, average_precision_score, precision_recall_curve,
    accuracy_score, balanced_accuracy_score,
    precision_score, recall_score, f1_score,
)

import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)
import src.constants as constants
csts = constants.ConstantsNamespace()


class UMAP_HDBSCAN_Clusterer:
    """ Class for clustering UMAP-reduced embeddings with HDBSCAN
    """
    def __init__(self, n_optuna_trials: int=25) -> None:
        self.n_optuna_trials = n_optuna_trials

    def perform_analysis(
        self,
        embeddings: np.ndarray,
        max_samples: int=2500,
    ) -> tuple[dict[str, float], go.Figure | None]:
        """
        High-level wrapper to subsample, fit, predict, score, and plot.
        Returns:
            metrics: dict containing 'silhouette_score'
            fig: Plotly figure (or None on error/noise)
        """
        # Subsample if dataset is too large
        if len(embeddings) > max_samples:
            indices = np.random.choice(len(embeddings), max_samples, replace=False)
            curr_embeddings = embeddings[indices]
        else:
            curr_embeddings = embeddings

        try:
            # Fit and predict
            reduced_data, labels_cluster = self.fit_predict(
                curr_embeddings, 
            )

            # Compute float metrics (only silhouette for now)
            metrics = {}
            unique_labels = np.unique(labels_cluster)
            n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
            if n_clusters > 1:
                sil_score = silhouette_score(reduced_data, labels_cluster)
                metrics["silhouette_score"] = float(sil_score)
            else:
                metrics["silhouette_score"] = -1.0

            # Generate cluster plot
            fig = self.plot(embeddings_2d=reduced_data, labels=labels_cluster)

            return metrics, fig

        except Exception as e:
            print(f"Error during clustering analysis: {e}")
            return {"silhouette_score": -1.0}, None

    def fit(
        self,
        pooled_embeddings: np.ndarray,
    ) -> dict[str, int|float|str]:
        """
        Find the best set of hyper-parameters for clustering UMAP-reduced embeddings
        """
        # Default parameters for a set of 100-1000-d embeddings with 1-10 clusters
        num_samples = len(pooled_embeddings)
        best_params = {
            "n_components": 15,
            "min_cluster_size": max(2, int(num_samples / 100)),
            "min_samples": max(2, int(num_samples / 100)),
        }

        # Parameters identified using
        if self.n_optuna_trials is not None and self.n_optuna_trials > 0:
            study = optuna.create_study(direction="maximize", sampler=TPESampler())
            objective_fn = lambda trial: self.cluster_objective(trial, pooled_embeddings)
            study.optimize(
                func=objective_fn,
                n_trials=self.n_optuna_trials,
                show_progress_bar=True,
            )
            best_params = study.best_params

        return best_params

    def cluster_objective(
        self,
        trial: optuna.Trial,
        embeddings: np.ndarray,
    ) -> float:
        """
        Objective function for clustering UMAP-reduced embeddings with HDBSCAN
        """
        params = {

            # General parameters
            "compute_clusters": True,
            "use_cuml": True,

            # UMAP parameters
            "n_components": trial.suggest_int("n_components", 2, 20),
            "n_neighbors": trial.suggest_int("n_neighbors", 5, 50),
            "min_dist": trial.suggest_float("min_dist", 0.0, 0.5),

            # HDBSCAN parameters
            "min_cluster_size": trial.suggest_int("min_cluster_size", 5, 60),
            "min_samples": trial.suggest_int("min_samples", 2, 25),

        }

        # Pass all params; the next function will sort them out
        try:
            reduced_embeddings, cluster_labels = self.predict(embeddings, **params)
        
        except Exception:
            print(f"Out of memory with these parameters: {params}. Skipping trial")
            return -1.0

        # Compute silhouette score
        no_cluster_correction = (1 if -1 in np.unique(cluster_labels) else 0)
        num_clusters = len(np.unique(cluster_labels)) - no_cluster_correction
        if num_clusters > 1:
            silhouette_score_ = silhouette_score(reduced_embeddings, cluster_labels)
        else:
            silhouette_score_ = -1.0

        return silhouette_score_

    def predict(
        self,
        embeddings: np.ndarray,
        compute_clusters: bool=True,
        use_cuml: bool=False,
        **kwargs,
    ) -> tuple[np.ndarray, np.ndarray|None]:
        """
        Reduce a set of embeddings with UMAP and cluster them with HDBSCAN
        """
        # Filter kwargs to create dictionaries for each function
        UMAP_KEYS = {"n_components", "n_neighbors", "min_dist"}
        HDBSCAN_KEYS = {"min_cluster_size", "min_samples"}
        umap_args = {k: v for k, v in kwargs.items() if k in UMAP_KEYS}
        hdbscan_args = {k: v for k, v in kwargs.items() if k in HDBSCAN_KEYS}

        # Using CUML for GPU acceleration
        if use_cuml:
            return self.reduce_and_cluster_gpu(
                embeddings,
                compute_clusters=compute_clusters,
                umap_args=umap_args,
                hdbscan_args=hdbscan_args,
            )

        # Fall back to CPU if GPU is unavailable
        else:
            return self.reduce_and_cluster_cpu(
                embeddings,
                compute_clusters=compute_clusters,
                umap_args=umap_args,
                hdbscan_args=hdbscan_args,
            )

    @staticmethod
    def reduce_and_cluster_gpu(
        embeddings: np.ndarray,
        compute_clusters=True,
        umap_args: dict[str, int|float|str]={},
        hdbscan_args: dict[str, int|float|str]={},
    ):
        """
        Reduce embeddings with UMAP and cluster them with HDBSCAN on GPU
        """
        # Compute dimensionality-reduced embeddings on GPU
        embeddings_gpu = cp.asarray(embeddings, dtype=cp.float16)
        reducer = cuml.UMAP(**umap_args, verbose=False)
        with redirect_stdout(io.StringIO()):  # suppressing an info message
            reduced_embeddings_gpu = reducer.fit_transform(embeddings_gpu)

        # Move result back to CPU and return, if no clustering is needed
        if not compute_clusters:
            return cp.asnumpy(reduced_embeddings_gpu), None

        # Compute clusters on GPU
        clusterer = cuml.cluster.hdbscan.HDBSCAN(**hdbscan_args, verbose=False)
        cluster_labels_gpu = clusterer.fit_predict(reduced_embeddings_gpu)

        # Move results back to CPU
        reduced_embeddings = cp.asnumpy(reduced_embeddings_gpu).astype(np.float32)
        cluster_labels = cp.asnumpy(cluster_labels_gpu).astype(np.float32)

        # Free GPU memory used by CUML
        del embeddings_gpu, reduced_embeddings_gpu, cluster_labels_gpu
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()
        
        return reduced_embeddings, cluster_labels

    @staticmethod
    def reduce_and_cluster_cpu(
        embeddings: np.ndarray,
        compute_clusters=True,
        umap_args: dict[str, int|float|str]={},
        hdbscan_args: dict[str, int|float|str]={},
    ):
        """
        Reduce embeddings with UMAP and cluster them with HDBSCAN on CPU
        """
        # Compute dimensionality-reduced embeddings on CPU
        reducer = umap.UMAP(**umap_args)
        reduced_embeddings = reducer.fit_transform(embeddings)

        # Only reduce embeddings if clustering is not needed
        if not compute_clusters:
            return reduced_embeddings, None

        # Compute clusters on CPU
        clusterer = hdbscan.HDBSCAN(**hdbscan_args)
        cluster_labels = clusterer.fit_predict(reduced_embeddings)

        return reduced_embeddings, cluster_labels

    def fit_predict(
        self,
        pooled_embeddings: np.ndarray,
        **kwargs,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Fit the clusterer (i.e., identify best hyper-parameters) and predict clusters
        """
        best_params = self.fit(pooled_embeddings)
        reduced_embeddings, cluster_labels = self.predict(pooled_embeddings, **best_params)

        return reduced_embeddings, cluster_labels

    @staticmethod
    def plot(
        embeddings_2d: np.ndarray,
        labels: np.ndarray[int|str],
        noise_label: int=-1,
    ) -> go.Figure:
        """
        Interactive 2D scatter plot of embeddings colored by cluster labels
        """
        # Check label type, to avoid any bug in the label comparison code
        if not isinstance(labels, np.ndarray):
            raise TypeError("Labels should be given in an np.ndarray format")

        # Create a color mapping for each cluster in a plotly-compatible RGB string format
        unique_labels = set(labels)
        label_keys = sorted([l for l in unique_labels if l != noise_label])
        palette = qualitative.Plotly
        color_map = {noise_label: "#000000"}  # black
        for i, key in enumerate(label_keys):
            color_map[key] = palette[i % len(palette)]

        # Plot samples with corresponding label colors
        fig = go.Figure()
        for k in sorted(list(unique_labels), key=lambda x: (x != -1, x)):  # noise last
            class_member_mask = (labels == k)
            xy = embeddings_2d[class_member_mask]
            fig.add_trace(go.Scatter(
                x=xy[:, 0], y=xy[:, 1], mode="markers", name=str(k),
                marker=dict(
                    color=color_map[k], line=dict(color="black", width=1),
                    size=3 if k == -1 else 6, opacity=0.3 if k == noise_label else 0.8,
                )
            ))

        # Polish figure
        fig.update_layout(
            xaxis_title="UMAP-1", yaxis_title="UMAP-2", legend_title="Clusters",
            yaxis_scaleanchor="x", width=1080, height=1080,
            margin=dict(l=40, r=40, b=40, t=80),
        )

        return fig


class DiscriminativeEmbeddingEvaluatorForMaskedLanguageModelling:
    def __init__(
        self,
        do_clustering: bool=True,
        max_clustered_samples: int=2500,
        n_optuna_trials: int=25,
        **kwargs,
    ):
        """
        Args:
            do_clustering: whether to run UMAP/HDBSCAN on the pooled embeddings
            n_optuna_trials: 0 to skip hyperparam search, > 0 to optimize clustering
            max_clustered_samples: maximum number of samples to cluster
        """
        self.do_clustering = do_clustering
        self.max_clustered_samples = max_clustered_samples
        self.n_optuna_trials = n_optuna_trials
        if do_clustering:
            self.clusterer_module = UMAP_HDBSCAN_Clusterer(n_optuna_trials=n_optuna_trials)

    def __call__(self, eval_preds: EvalPrediction) -> dict[str, float]:
        """
        The trainer will call this function at evaluation time, on CPU
        eval_preds.predictions contains the tuple returned by preprocess_logits_for_metrics
        """
        # Retrieve model outputs
        (pred_ids, embeddings), labels = eval_preds.predictions, eval_preds.label_ids
        metrics = {}

        # Compute MLM accuracy
        preds_flat, labels_flat = pred_ids.flatten(), labels.flatten()
        mask = labels_flat != -100  # filtering out ignored tokens
        metrics["mlm_accuracy"] = accuracy_score(labels_flat[mask], preds_flat[mask])
        
        # Compute clustering metrics using your custom module
        if self.do_clustering and embeddings is not None:
            cluster_metrics, cluster_fig = self.clusterer_module.perform_analysis(
                embeddings, 
                max_samples=self.max_clustered_samples
            )

            # Update metrics and plot clusters
            metrics.update(cluster_metrics)
            if cluster_fig is not None and wandb.run is not None:
                wandb.log({"eval_plots/clusters": cluster_fig}, commit=False)

        return metrics


class BaseEmbeddingEvaluatorForClassification:
    """
    Base class containing shared logic for metrics, clustering, and logging.
    Subclasses must implement `prepare_predictions`.
    """
    def __init__(
        self,
        do_clustering: bool = False,
        n_optuna_trials: int = 0,
        max_clustered_samples: int = 2500,
        pos_label: int = 1,
    ):
        self.do_clustering = do_clustering
        self.max_clustered_samples = max_clustered_samples
        self.pos_label = pos_label
        if do_clustering:
            self.clusterer_module = UMAP_HDBSCAN_Clusterer(n_optuna_trials=n_optuna_trials)

    def prepare_predictions(self, eval_preds: EvalPrediction) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """
        Abstract method to unpack predictions.
        Must return: (logits, labels, embeddings)
        """
        raise NotImplementedError("Subclasses must implement prepare_predictions")

    def __call__(self, eval_preds: EvalPrediction) -> dict[str, float]:
        """
        Main entry point called by the Trainer
        """
        # Standardize Input using the subclass implementation
        logits, labels, embeddings = self.prepare_predictions(eval_preds)

        # Compute Probabilities
        probs = softmax(logits, axis=-1)
        if logits.shape[1] == 2:
            probs_pos = probs[:, self.pos_label]
        else:
            # Fallback for multiclass: take max prob
            probs_pos = probs.max(axis=1)

        # Probability-based metrics
        metrics = {}
        metrics["ece"] = self._expected_calibration_error(labels, probs)
        try:
            metrics["roc_auc"] = roc_auc_score(labels, probs_pos)
            metrics["pr_auc"] = average_precision_score(labels, probs_pos)
        except ValueError:
            # Fail if we have only one class in the batch
            metrics["roc_auc"] = -1.0
            metrics["pr_auc"] = -1.0

        # Threshold-based metrics
        preds_optimal = self._get_preds_from_best_threshold(labels, probs_pos)
        avg_method = "binary" if logits.shape[1] == 2 else "macro"
        metrics["acc"] = accuracy_score(labels, preds_optimal)
        metrics["bal_acc"] = balanced_accuracy_score(labels, preds_optimal)
        metrics["precision"] = precision_score(labels, preds_optimal, average=avg_method, zero_division=0)
        metrics["recall"] = recall_score(labels, preds_optimal, average=avg_method, zero_division=0)
        metrics["f1"] = f1_score(labels, preds_optimal, average=avg_method, zero_division=0)

        # WandB Logging
        if wandb.run is not None:
            self._log_plots(labels, probs, preds_optimal, prefix="eval_plots")

        # Clustering Analysis
        if self.do_clustering and embeddings is not None:
            cluster_metrics, cluster_fig = self.clusterer_module.perform_analysis(
                embeddings, 
                max_samples=self.max_clustered_samples
            )
            metrics.update(cluster_metrics)
            if cluster_fig is not None and wandb.run is not None:
                wandb.log({"eval_plots/clusters": cluster_fig}, commit=False)

        return metrics

    def _log_plots(self, labels, probs, preds, prefix="eval_plots"):
        try:
            # Confusion Matrix
            cm = wandb.plot.confusion_matrix(probs=None, y_true=labels, preds=preds)
            wandb.log({f"{prefix}/conf_mat": cm}, commit=False)

            # Binary specific curves
            if probs.shape[1] == 2:
                wandb.log({f"{prefix}/pr_curve": wandb.plot.pr_curve(labels, probs)}, commit=False)
                wandb.log({f"{prefix}/roc_curve": wandb.plot.roc_curve(labels, probs)}, commit=False)
        except Exception as e:
            print(f"WandB logging error: {e}")

    @staticmethod
    def _get_preds_from_best_threshold(y_true, y_score):
        """
        Infer predictions using F1-score to determine the optimal threshold
        """
        if len(np.unique(y_true)) < 2:
            return np.round(y_score)

        precisions, recalls, thresholds = precision_recall_curve(y_true, y_score)
        denominator = precisions + recalls
        denominator[denominator == 0] = 1e-9 
        f1_scores = (2 * precisions * recalls) / denominator
        f1_scores = f1_scores[:-1]

        if len(f1_scores) == 0:
            return np.zeros_like(y_true)

        best_threshold = thresholds[np.argmax(f1_scores)]

        return (y_score >= best_threshold).astype(int)

    @staticmethod
    def _expected_calibration_error(y_true, y_probs, n_bins=10):
        n_classes = y_probs.shape[1]
        if n_classes > 2:
            confidences = np.max(y_probs, axis=1)
            is_correct = (np.argmax(y_probs, axis=1) == y_true)
        else:
            confidences = y_probs[:, 1]
            is_correct = (y_true == 1)
        
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(confidences, bin_boundaries[1:-1])
        ece = 0.0
        for i in range(n_bins):
            in_bin = (bin_indices == i)
            n_in_bin = np.sum(in_bin)
            if n_in_bin > 0:
                acc_in_bin = np.mean(is_correct[in_bin])
                conf_in_bin = np.mean(confidences[in_bin])
                ece += np.abs(acc_in_bin - conf_in_bin) * n_in_bin
        return ece / len(y_true)


class DiscriminativeEmbeddingEvaluatorForClassification(BaseEmbeddingEvaluatorForClassification):
    """
    Evaluator for standard BERT-like sequence classification models.
    Expects predictions to be formatted by `preprocess_logits_for_metrics_discriminative`.
    """
    def prepare_predictions(self, eval_preds: EvalPrediction):
        # Unpack predictions
        if isinstance(eval_preds.predictions, tuple):
            logits, embeddings = eval_preds.predictions
        else:
            logits = eval_preds.predictions
            embeddings = None

        labels = eval_preds.label_ids
        return logits, labels, embeddings


class GenerativeEmbeddingEvaluatorForClassification(BaseEmbeddingEvaluatorForClassification):
    """
    Evaluator for Generative LLMs (CausalLM) performing classification.
    Expects predictions to be formatted by `preprocess_logits_for_metrics_generative`.
    """
    def __init__(self, tokenizer, positive_token="1", negative_token="0", **kwargs):
        # Enforce clustering to False (at least for now)
        kwargs["do_clustering"] = False
        super().__init__(**kwargs)
        self.tokenizer = tokenizer
        self.pos_token_id = tokenizer.encode(positive_token, add_special_tokens=False)[-1]
        self.neg_token_id = tokenizer.encode(negative_token, add_special_tokens=False)[-1]

    def prepare_predictions(self, eval_preds: EvalPrediction):
        # Unpack predictions (from preprocess function output)
        logits, embeddings = eval_preds.predictions, None  # no clustering (for now)

        # Reconstruct labels (first valid token in the sequence, using argmax)
        label_ids = eval_preds.label_ids
        answer_indices = (label_ids != -100).argmax(axis=1)

        # Extract the actual token IDs at those positions
        actual_token_ids = label_ids[np.arange(len(label_ids)), answer_indices]

        # Convert to binary 0/1 (1 if it matches pos_token_id, else 0)
        labels = (actual_token_ids == self.pos_token_id).astype(int)

        return logits, labels, embeddings


def preprocess_logits_for_mlm_metrics(logits, labels):
    """
    Compute model predictions and pool last hidden states as sequence embeddings
    """
    # The model returns (logits, hidden_states)
    if isinstance(logits, tuple):
        real_logits = logits[0]  # [batch, seq, vocab]
        hidden_states = logits[1] # layer tuple (usually includes only the last one)
        last_hidden = hidden_states[-1] # [batch, seq, hidden]
    else:
        real_logits = logits
        last_hidden = None

    # Compute masked token prediction from the model logits
    pred_ids = torch.argmax(real_logits, dim=-1)
    
    # Pool embeddings by averaging all non-pad token embeddings
    if last_hidden is not None:
        mask = (labels != -100).unsqueeze(-1).to(last_hidden.dtype)
        sum_embeddings = (last_hidden * mask).sum(dim=1)
        sum_mask = mask.sum(dim=1).clamp(min=1e-9)  # avoid division by zero
        pooled_embeddings = sum_embeddings / sum_mask

    return pred_ids, pooled_embeddings


def make_preprocess_logits_for_metrics_generative(
    tokenizer,
    pos_token: str="1",
    neg_token: str="0",
):
    """
    Creates a pre-processing function to extract only the logits
    for the '0' and '1' tokens and the last hidden state for embeddings.
    """
    pos_id = tokenizer.encode(pos_token, add_special_tokens=False)[-1]
    neg_id = tokenizer.encode(neg_token, add_special_tokens=False)[-1]

    def preprocess_logits_for_metrics_generative(logits, labels):
        if isinstance(logits, tuple):
            logits = logits[0]

        # Find index of the first valid label (the answer token)
        # labels format: [-100 (prompt), ..., answer, eos, -100 (padding)]
        # argmax on a boolean tensor returns the index of the first True
        answer_indices = (labels != -100).long().argmax(dim=-1)

        # The logit used to predict token at answer_indices is located at answer_indices - 1
        # Clamped to 0 to avoid -1 indexing errors, though -1 should not happen in valid data
        logit_indices = (answer_indices - 1).clamp(min=0)
        
        # Extract logits at the answer position
        batch_range = torch.arange(logits.shape[0], device=logits.device)
        final_logits = logits[batch_range, logit_indices][:, [neg_id, pos_id]]

        return final_logits

    return preprocess_logits_for_metrics_generative
