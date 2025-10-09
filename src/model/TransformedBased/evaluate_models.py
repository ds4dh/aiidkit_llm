import numpy as np
import torch
from torch.utils.data import DataLoader
from datasets import Dataset
from transformers.trainer_utils import EvalPrediction
from sentence_transformers import SentenceTransformer
from src.model.patient_sequence_embedder import PatientDataCollatorForSequenceEmbedding
from src.model.model_utils import WandbPlottingCallback

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

import wandb
import plotly.graph_objects as go
from plotly.colors import qualitative
from scipy.special import softmax
from sklearn.metrics import (
    silhouette_score, adjusted_mutual_info_score, ConfusionMatrixDisplay,
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
            "min_cluster_size": int(num_samples / 100),
            "min_samples": int(num_samples / 100),
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
        best_params = self.fit(pooled_embeddings, optuna_trials=kwargs.get("optuna_trials", 25))
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


class CustomEmbeddingEvaluator:
    """
    Class used by HuggingFace's trainer to compute embedding metrics
    """
    def __init__(
        self,
        eval_dataset: Dataset,
        embedding_mode: str,
        do_clustering_analysis: bool=True,
        optuna_trials_for_clustering: int|None=None,
        eval_label_key: str|None=None,
        eval_batch_size: int|None=None,  # used for sequence embedding evaluation
        eval_data_collator: PatientDataCollatorForSequenceEmbedding|None=None,
        wandb_plotting_callback: WandbPlottingCallback|None=None,
        *args, **kwargs,
    ):
        if embedding_mode not in ["token", "sequence"]:
            raise ValueError("Embedding mode must be either 'token' or 'sequence'")

        self.embedding_mode = embedding_mode
        self.do_clustering_analysis = do_clustering_analysis
        if do_clustering_analysis:
            self.clusterer = UMAP_HDBSCAN_Clusterer(optuna_trials_for_clustering)
        self.wandb_plotting_callback = wandb_plotting_callback

        # Prepare evaluation data
        self.eval_dataset = eval_dataset
        self.eval_batch_size = eval_batch_size
        self.eval_data_collator = eval_data_collator
        if eval_label_key is not None:
            self.true_labels = np.array(self.eval_dataset[eval_label_key])
        else:
            self.true_labels = None

    def __call__(self, *args, **kwargs) -> dict[str, float]:
        """
        Evaluate embeddings with different cluster-related metrics
        """
        # Initialize metric and plot dictionaries (logged differently!)
        metric_dict = {}
        plot_dict = {}

        # Extract embeddings from the model output
        if self.embedding_mode == "token":
            # The huggingface trainer passes a single EvalPrediction object
            eval_preds: EvalPrediction = args[0]  # no other argument
            eval_preds: dict[str, torch.Tensor] = self._format_eval_prediction(eval_preds)
            pooled_embeddings = eval_preds["pooled_embeddings"]
            mlm_accuracy = self._compute_mlm_accuracy_for_token_model(eval_preds)
            metric_dict.update({"mlm_accuracy": mlm_accuracy})
        else:
            # The sentence-transformer trainer passes the model as the first argument
            model: SentenceTransformer = args[0]
            eval_path, epoch_float, global_step = args[1:]  # in case useful
            pooled_embeddings = self._get_pooled_embeddings_for_sequence_model(model)

        # Compute cluster-related metrics and plots
        if self.do_clustering_analysis:
            best_params = self.clusterer.fit(pooled_embeddings)
            reduced_embeddings, cluster_labels = self.clusterer.predict(pooled_embeddings, **best_params)
            silhouette_score_ = silhouette_score(reduced_embeddings, cluster_labels)
            metric_dict.update({"silhouette_score": silhouette_score_})

            # Create a 2-dimensional visualization
            vis_params = best_params.copy()
            vis_params["n_components"] = 2
            reduced_embeddings_2d, _ = self.clusterer.predict(pooled_embeddings, compute_clusters=False, **vis_params)
            cluster_plot = self.clusterer.plot(reduced_embeddings_2d, cluster_labels)
            plot_dict.update({"cluster_plot": cluster_plot})

        # Compute infection-label-related metrics and plots
        if self.true_labels is not None:

            # Compute adjusted mutual information score if clusters were computed
            if self.do_clustering_analysis:
                ami_score = adjusted_mutual_info_score(self.true_labels, cluster_labels)
                truth_plot = self.clusterer.plot(reduced_embeddings_2d, self.true_labels, noise_label=0)
                plot_dict.update({"truth_plot": truth_plot})
                metric_dict.update({"ami_score": ami_score})

            # Compute classification metrics using a regularized classifier
            cm_plot_dict, classification_metric_dict = self._evaluate_supervised_classifier(eval_preds)
            plot_dict.update(cm_plot_dict)
            metric_dict.update(classification_metric_dict)

        # Compute a survival metric (using higher is better convention)
        survival_score = 0.0
        if "mlm_accuracy" in metric_dict:
            survival_score += metric_dict["mlm_accuracy"]
        if "supervised_task_f1_macro" in classification_metric_dict:
            survival_score += classification_metric_dict["supervised_task_f1_macro"]
        metric_dict.update({"survival_score": survival_score})

        # Send the plots to the wandb plotting callback
        if plot_dict:
            if self.wandb_plotting_callback is not None:
                self.wandb_plotting_callback.plots_to_log.update(plot_dict)

        return metric_dict

    @staticmethod
    def _format_eval_prediction(
        eval_prediction: EvalPrediction,
    ) -> dict[str, torch.Tensor]:
        """ Extract information from EvalPrediction object with names
        """
        # Extract last hidden state from the model output
        model_output, task_labels = eval_prediction
        mlm_loss, mlm_logits, cutoff_days, \
        supervised_task_loss, supervised_task_logits, \
        hidden_states, pooled_embeddings = model_output

        mlm_predictions = mlm_logits.argmax(axis=-1)
        last_hidden_state = hidden_states[-1]

        # Extract MLM task labels, to know where the padded tokens are
        if isinstance(task_labels, tuple):
            mlm_labels = task_labels[0]
            supervised_task_labels = task_labels[1:]
        else:
            mlm_labels = task_labels
            supervised_task_labels = ()

        # Huggingface's convention is to have label == -100 for padding tokens 
        no_pad_mask = (mlm_labels != -100)

        return {
            "last_hidden_state": last_hidden_state,
            "no_pad_mask": no_pad_mask,
            "mlm_logits": mlm_logits,
            "mlm_predictions": mlm_predictions,
            "mlm_labels": mlm_labels,
            "mlm_loss": mlm_loss,
            "cutoff_days": cutoff_days,
            "supervised_task_loss": supervised_task_loss,
            "supervised_task_logits": supervised_task_logits,
            "supervised_task_labels": supervised_task_labels,
            "pooled_embeddings": pooled_embeddings,
        }

    def _compute_mlm_accuracy_for_token_model(
        self,
        eval_predictions: dict[str, torch.Tensor],
    ) -> float:
        """
        Compute raw accuracy on MLM task (labels already aligned with the model
        output, thanks to the data collator)
        """
        # Flatten the predictions and labels to make them comparable
        predictions_flat = eval_predictions["mlm_predictions"].flatten()
        labels_flat = eval_predictions["mlm_labels"].flatten()
        no_pad_mask_flat = eval_predictions["no_pad_mask"].flatten()

        # Filter out ignored labels
        predictions_valid = predictions_flat[no_pad_mask_flat]
        labels_valid = labels_flat[no_pad_mask_flat]

        return np.mean(predictions_valid == labels_valid)  # i.e., accuracy

    def _get_pooled_embeddings_for_sequence_model(
        self,
        model: SentenceTransformer,
    ) -> np.ndarray:
        """
        Get final sequence embeddings from a SentenceTransformer model
        """
        # Use the same data collator used during training
        dataloader = DataLoader(
            self.eval_dataset,
            batch_size=self.eval_batch_size,
            collate_fn=self.eval_data_collator,
            shuffle=False,  # important if any label comparison later
        )

        # Perform inference with the sentence-transformer model
        model.eval()
        all_embeddings = []
        with torch.no_grad():
            input_layer_device = next(model.parameters()).device
            for batch in dataloader:

                # The collator might add non-model-input keys (e.g., labels)
                model_input = {
                    k: v.to(input_layer_device) for k, v in batch.items()
                    if isinstance(v, torch.Tensor)
                }

                # Pass the processed batch to the model
                outputs = model(model_input)
                embeddings = outputs["sentence_embedding"].cpu().numpy()  # (batch_size, embed_dim)
                all_embeddings.append(embeddings)

        return np.concatenate(all_embeddings, axis=0)  # (num_eval_samples, embed_dim)

    # def _evaluate_supervised_classifier(
    #     self,
    #     eval_predictions: dict[str, np.ndarray],
    #     cutoff_days: str = None,
    # ) -> tuple[go.Figure, dict[str, float]]:
    #     """
    #     Compute classification metrics from the model's supervised task logits
    #     """
    #     y_logits = eval_predictions["supervised_task_logits"]
    #     y_true = eval_predictions["supervised_task_labels"][0]  # it is a tuple!
    #     y_probs = softmax(y_logits, axis=1)
    #     n_classes = y_logits.shape[1]

    #     # Binary classification case
    #     if n_classes == 2:
    #         y_score = y_probs[:, 1]  # score = probability of positive class
    #         y_pred = self._get_preds_from_best_threshold(y_true, y_score)
    #         auroc_score = roc_auc_score(y_true, y_score)
    #         auprc_score = average_precision_score(y_true, y_score)
    #         sensitivity = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    #         specificity = recall_score(y_true, y_pred, pos_label=0, zero_division=0)

    #     # Multi-class classification
    #     else:
    #         y_pred = np.argmax(y_probs, axis=1)  # prediction = class with highest probability
    #         auroc_score = roc_auc_score(y_true, y_probs, multi_class="ovr", average="macro")
    #         auprc_score = None
    #         sensitivity = None
    #         specificity = None

    #     # Compute confusion matrix plot (to be logged later)
    #     cm_plot = wandb.plot.confusion_matrix(y_true=y_true, preds=y_pred)
        
    #     # Add threshold-related metrics
    #     metrics = {
    #         "supervised_task_accuracy": accuracy_score(y_true, y_pred),
    #         "supervised_task_balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
    #         "supervised_task_precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
    #         "supervised_task_recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
    #         "supervised_task_f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
    #         "supervised_task_ece": self._expected_calibration_error(y_true, y_probs),
    #         "supervised_task_roc_auc": auroc_score,
    #         "supervised_task_pr_auc": auprc_score if auprc_score is not None else "N/A",
    #         "supervised_task_sensitivity": sensitivity if sensitivity is not None else "N/A",
    #         "supervised_task_specificity": specificity if specificity is not None else "N/A",
    #     }

    #     return cm_plot, metrics

    def _compute_classification_metrics(
        self,
        y_true: np.ndarray,
        y_logits: np.ndarray,
    ) -> tuple[go.Figure, dict[str, float]]:
        """
        Core logic to compute classification metrics from labels and logits.
        
        Args:
            y_true: Ground truth labels.
            y_logits: Raw logit outputs from the model.

        Returns:
            A tuple containing the confusion matrix plot and a dictionary of metrics.
        """
        # Ensure there are samples to evaluate
        if y_true.size == 0:
            return None, {}

        y_probs = softmax(y_logits, axis=1)
        n_classes = y_logits.shape[1]

        # Binary classification case
        if n_classes == 2:
            y_score = y_probs[:, 1]  # score = probability of positive class
            y_pred = self._get_preds_from_best_threshold(y_true, y_score)
            auroc_score = roc_auc_score(y_true, y_score)
            auprc_score = average_precision_score(y_true, y_score)
            sensitivity = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
            specificity = recall_score(y_true, y_pred, pos_label=0, zero_division=0)

        # Multi-class classification
        else:
            y_pred = np.argmax(y_probs, axis=1)  # prediction = class with highest probability
            auroc_score = roc_auc_score(y_true, y_probs, multi_class="ovr", average="macro")
            auprc_score = None
            sensitivity = None
            specificity = None

        # Compute confusion matrix plot (to be logged later)
        cm_plot = wandb.plot.confusion_matrix(y_true=y_true, preds=y_pred)

        # Add threshold-related metrics
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
            "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
            "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
            "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
            "ece": self._expected_calibration_error(y_true, y_probs),
            "roc_auc": auroc_score,
            "pr_auc": auprc_score if auprc_score is not None else "N/A",
            "sensitivity": sensitivity if sensitivity is not None else "N/A",
            "specificity": specificity if specificity is not None else "N/A",
        }

        return cm_plot, metrics

    def _evaluate_supervised_classifier(
        self,
        eval_predictions: dict[str, np.ndarray],
    ) -> tuple[go.Figure, dict[str, float]]:
        """
        Compute classification metrics for the overall supervised task.
        """
        # Extract evaluation data
        y_logits = eval_predictions["supervised_task_logits"]
        y_true = eval_predictions["supervised_task_labels"][0]  # it is a tuple!

        # Evaluate all predictions at once
        cm_plot, metrics = self._compute_classification_metrics(y_true, y_logits)
        metric_dict = {f"sup_{k}": v for k, v in metrics.items()}
        plot_dict = {"cm_plot": cm_plot}

        # Evaluate predictions stratifying per cutoff day
        cutoff_days = eval_predictions.get("cutoff_days", None)
        if cutoff_days is not None:
            unique_cutoffs = np.unique(cutoff_days)        
            for cutoff in unique_cutoffs:

                # Filter labels and logits
                indices = (cutoff_days == cutoff)
                y_true_cutoff = y_true[indices]
                y_logits_cutoff = y_logits[indices]

                # Compute metrics for this subset of data
                cm_plot_cutoff, metrics_cutoff = self._compute_classification_metrics(
                    y_true_cutoff, y_logits_cutoff
                )

                # Add a prefix for the cutoff day to each metric key for logging
                plot_dict[f"cm_plot_cutoff_{int(cutoff)}"] = cm_plot_cutoff
                for key, value in metrics_cutoff.items():
                    metric_dict[f"sup_{key}_cut_{int(cutoff)}"] = value

        return plot_dict, metric_dict

    def _evaluate_supervised_classifier_per_cutoff_day(
        self,
        eval_predictions: dict[str, np.ndarray],
    ) -> tuple[dict[str, go.Figure], dict[str, float]]:
        """
        Compute classification metrics for each cutoff day separately.
        """
        y_logits = eval_predictions["supervised_task_logits"]
        y_true = eval_predictions["supervised_task_labels"][0]
        cutoff_days = eval_predictions["cutoff_days"][0]

        

    @staticmethod
    def _get_preds_from_best_threshold(
        y_true: np.ndarray,
        y_score: np.ndarray,
    ) -> np.ndarray:
        """
        Find the optimal threshold for a binary classifier based on the f1-score
        of the positive class
        """
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_score)
        f1_scores = (2 * precisions * recalls) / (precisions + recalls + 1e-9)
        f1_scores = f1_scores[:-1]
        best_threshold = thresholds[np.argmax(f1_scores)]

        return (y_score >= best_threshold).astype(int)

    @staticmethod
    def _expected_calibration_error(y_true, y_probs, n_bins=10):
        """
        Computes the Expected Calibration Error (ECE) for a model.

        Args:
            y_true (np.ndarray): True labels, shape (n_samples,).
            y_probs (np.ndarray): Predicted probabilities, shape (n_samples, n_classes).
            n_bins (int): Number of bins to use for calibration.

        Returns:
            float: The ECE score.
        """
        # For multi-class, we need the confidence (max prob) and the predictions
        n_classes = y_probs.shape[1]
        if n_classes > 2:
            confidences = np.max(y_probs, axis=1)
            predictions = np.argmax(y_probs, axis=1)
            is_correct = (predictions == y_true)

        # For binary, we use the probability of the positive class
        else:
            confidences = y_probs[:, 1]
            is_correct = (y_true == 1)

        # Bin confidences and calculate accuracy and average confidence per bin
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(confidences, bin_boundaries[1:-1])
        ece = 0.0
        for i in range(n_bins):
            in_bin = (bin_indices == i)
            n_in_bin = np.sum(in_bin)
            if n_in_bin > 0:
                acc_in_bin = np.mean(is_correct[in_bin])  # bin accuracy
                conf_in_bin = np.mean(confidences[in_bin])  # bin confidence
                ece += np.abs(acc_in_bin - conf_in_bin) * n_in_bin  # update ECE

        return ece / len(y_true)
