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
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    silhouette_score, adjusted_mutual_info_score, roc_curve,
    roc_auc_score, average_precision_score, precision_recall_curve,
    brier_score_loss, accuracy_score, balanced_accuracy_score,
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
        
        # Compute clustering metrics using custom clustering module
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
        self.current_prefix = "eval"  # to updated during each _log_plot call
        self.iso_reg = None  # used for calibration plots
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
        # Retrieve model outputs and compute probabilities
        logits, labels, embeddings = self.prepare_predictions(eval_preds)
        probs = softmax(logits, axis=-1)
        probs_pos = probs[:, self.pos_label] if logits.shape[1] == 2 else probs.max(axis=1)
        probs_cal = self._apply_calibration_strategy(labels, probs_pos)

        # Calibration metrics (Brier score, expected calibration error)
        metrics = {}
        metrics["brier_raw"] = brier_score_loss(labels, probs_pos, pos_label=self.pos_label)
        metrics["brier_cal"] = brier_score_loss(labels, probs_cal, pos_label=self.pos_label)
        metrics["ece_raw"] = self._expected_calibration_error(labels, probs_pos)
        metrics["ece_cal"] = self._expected_calibration_error(labels, probs_cal)

        # Probability-based metrics
        try:
            metrics["roc_auc"] = roc_auc_score(labels, probs_pos)
            metrics["pr_auc"] = average_precision_score(labels, probs_pos)
        except ValueError:
            # Fail if we have only one class in the batch
            metrics["roc_auc"] = -1.0
            metrics["pr_auc"] = -1.0

        # Threshold-based metrics
        t = 0.1  # for optimal-f1 threshold, use "best_f1", else use float between 0.0 and 1.0
        threshold, preds = self._get_preds_from_threshold(labels, probs_pos, t)
        avg_method = "binary" if logits.shape[1] == 2 else "macro"
        if isinstance(t, float): t = int(t * 100)
        metrics[f"acc_t{t}"] = accuracy_score(labels, preds)
        metrics[f"bal_acc_t{t}"] = balanced_accuracy_score(labels, preds)
        metrics[f"precision_t{t}"] = precision_score(labels, preds, average=avg_method, zero_division=0)
        metrics[f"recall_t{t}"] = recall_score(labels, preds, average=avg_method, zero_division=0)
        metrics[f"f1_t{t}"] = f1_score(labels, preds, average=avg_method, zero_division=0)
        metrics[f"net_benefit_t{t}"] = self._calculate_net_benefit(labels, probs_pos, threshold=threshold)
        
        # Plot logging, directly to WandB
        if wandb.run is not None and probs.shape[1] == 2:
            self._log_plots(labels, preds, probs_pos, probs_cal)

        # Clustering analysis  # for now, skipping
        if 0 and self.do_clustering and embeddings is not None:
            cluster_metrics, cluster_fig = self.clusterer_module.perform_analysis(
                embeddings, 
                max_samples=self.max_clustered_samples
            )
            metrics.update(cluster_metrics)
            if cluster_fig is not None and wandb.run is not None:
                wandb.log({"eval_plots/clusters": cluster_fig}, commit=False)

        return metrics

    @staticmethod
    def _get_preds_from_threshold(
        y_true: np.ndarray,
        y_score: np.ndarray,
        threshold: float | str = "best_f1",
    ) -> tuple[float, np.ndarray]:
        """
        Infer predictions with a given threshold or computing the f1-optimal threshold
        """
        if len(np.unique(y_true)) < 2:
            return threshold if isinstance(threshold, float) else 0.5, np.round(y_score)

        if threshold == "best_f1":
            precisions, recalls, thresholds = precision_recall_curve(y_true, y_score)
            denominator = precisions + recalls
            denominator[denominator == 0] = 1e-9 
            f1_scores = (2 * precisions * recalls) / denominator
            f1_scores = f1_scores[:-1]

            if len(f1_scores) == 0:
                return np.zeros_like(y_true)

            threshold = thresholds[np.argmax(f1_scores)]
        
        preds = (y_score >= threshold).astype(int)
        return threshold, preds
    
    @staticmethod
    def _expected_calibration_error(y_true, y_prob_pos, n_bins=10):
        if y_prob_pos.ndim > 1: y_prob_pos = y_prob_pos[:, 1]
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(y_prob_pos, bin_boundaries[1:-1])
        
        ece = 0.0
        for i in range(n_bins):
            in_bin = (bin_indices == i)
            n_in_bin = np.sum(in_bin)
            if n_in_bin > 0:
                acc_in_bin = np.mean(y_true[in_bin] == 1) # Assumes binary 0/1
                conf_in_bin = np.mean(y_prob_pos[in_bin])
                ece += np.abs(acc_in_bin - conf_in_bin) * n_in_bin
                
        return ece / len(y_true)
    
    @staticmethod
    def _calculate_net_benefit(y_true, y_prob, threshold):
        """
        Calculates net benefit at a specific clinical decision threshold.
        Formula: (TP/N) - (FP/N) * (t / (1-t)) 
        """
        if threshold >= 1.0 - 1e-9:  # avoid division by zero
            return 0.0

        preds = (y_prob >= threshold).astype(int)
        tp = np.sum((preds == 1) & (y_true == 1))
        fp = np.sum((preds == 1) & (y_true == 0))
        n = len(y_true)

        # Weight represents the ratio of harm (false positive) to benefit (true positive)
        weight = threshold / (1 - threshold)
        nb = (tp / n) - (fp / n) * weight

        return nb
    
    def _log_plots(self, labels, preds, probs_pos, probs_cal):
        """Generates and logs plots directly to WandB with the correct prefix."""
        try:
            plots = {}
            prefix = self.current_prefix
            
            # Standard metric plots
            plots[f"{prefix}/conf_mat"] = wandb.plot.confusion_matrix(probs=None, y_true=labels, preds=preds)
            plots[f"{prefix}/decision_curve"] = self._plot_decision_curve(labels, probs_pos)
            plots[f"{prefix}/risk_distribution"] = self._plot_risk_distribution(labels, probs_pos)
            plots[f"{prefix}/roc_curve"] = self._plot_roc_curve(labels, probs_pos)

            # Calibration plots (using raw and calibrated probabilities)
            plots[f"{prefix}/calibration_inf_raw"] = self._plot_calibration(labels, probs_pos, False, False)
            plots[f"{prefix}/calibration_inf_cal"] = self._plot_calibration(labels, probs_cal, False, True)

            # Log without committing (let HF's trainer commit at the end of the step)
            wandb.log(plots, commit=False)

        except Exception as e:
            print(f"Plot logging error: {e}")

    def _apply_calibration_strategy(
        self,
        probs_pos: np.ndarray,
        labels: np.ndarray,
    ) -> np.ndarray:
        """
        Stateful calibration logic:
        - If prefix is NOT 'test' (e.g., 'eval', 'val'): fit a new calibrator.
        - If prefix IS 'test': transform using the existing calibrator.
        """
        # Validation phase (fit)
        if "test" not in self.current_prefix:
            self.iso_reg = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
            try:
                self.iso_reg.fit(probs_pos, labels)
                return self.iso_reg.transform(probs_pos)
            except Exception as e:
                print(f"Calibration fitting failed (data too small?): {e}")
                self.iso_reg = None
                return probs_pos
            
        # Test phase (transform)
        else:
            if self.iso_reg is None:
                print(
                    f"Warning: evaluating '{self.current_prefix}' but no "
                    "calibration was fit! (Did you run validation first?)"
                )
                return probs_pos
            try:
                return self.iso_reg.transform(probs_pos)
            except Exception as e:
                print(f"Calibration transform failed: {e}")
                return probs_pos

    def _plot_calibration(self, y_true, y_prob_infection, is_surv=False, is_corr=False, n_bins=10):
        """
        Flexible calibration plotter
        - is_surv=True: plots survival probability vs observed survival fraction
        - is_surv=False: plots infection probability vs observed infection fraction
        """
        # Determine labels and probabilities based on mode
        if is_surv:
            # Target: Survival (0 -> 1, 1 -> 0)
            target_y = 1 - y_true
            target_prob = 1 - y_prob_infection
            title_mode = "Survival (non-infection)"
            axis_label = "Probability of survival"
            line_color = "royalblue"
        else:
            # Target: Infection
            target_y = y_true
            target_prob = y_prob_infection
            title_mode = "Infection event"
            axis_label = "Probability of infection"
            line_color = "firebrick"

        # Compute Calibration Curve
        prob_true, prob_pred = calibration_curve(target_y, target_prob, n_bins=n_bins)
        
        # Initialize figure
        title_suffix = "(with isotonic correction)" if is_corr else "(raw model output)"
        symbol = "diamond" if is_corr else "circle"
        fig = go.Figure()
        
        # Diagonal "perfect" line
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode='lines', name='Ideal',
            line=dict(dash='dash', color='gray'), showlegend=False
        ))
        
        # The Model Curve
        fig.add_trace(go.Scatter(
            x=prob_pred, y=prob_true, mode='lines+markers', 
            name='Model',
            marker=dict(color=line_color, symbol=symbol, size=8),
            line=dict(color=line_color, width=2)
        ))

        # Polish figure
        fig.update_layout(
            title=f"Reliability: {title_mode} <br><sup>{title_suffix}</sup>",
            xaxis_title=f"Predicted {axis_label}",
            yaxis_title="Observed fraction",
            width=600, height=600,
            xaxis=dict(range=[0, 1]),
            yaxis=dict(range=[0, 1]),
            margin=dict(l=40, r=40, t=60, b=40),
        )
        
        return fig

    def _plot_decision_curve(self, y_true, y_prob):
        """Decision curve analysis plotting model net benefit vs threshold."""
        # Calculate model net benefit for each threshold using the helper
        thresholds = np.linspace(0.01, 0.99, 100)
        net_benefits = [self._calculate_net_benefit(y_true, y_prob, t) for t in thresholds]
        
        # Calculate "treat all" strategy (reference as only positive predictions)
        n = len(y_true)
        prevalence = np.sum(y_true) / n
        net_benefit_all = prevalence - (1 - prevalence) * (thresholds / (1 - thresholds))
        
        # The "treat none" strategy is always 0
        net_benefit_none = np.zeros_like(thresholds)

        # Generate decision curve plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=thresholds, y=net_benefits,
            mode='lines', name='Model',
        ))
        fig.add_trace(go.Scatter(
            x=thresholds, y=net_benefit_all,
            mode='lines', name='Treat All', line=dict(dash='dot', color='gray'),
        ))
        fig.add_trace(go.Scatter(
            x=thresholds, y=net_benefit_none,
            mode='lines', name='Treat None', line=dict(color='black'),
        ))
        fig.update_layout(
            title="Decision curve analysis",
            xaxis_title="Threshold probability",
            yaxis_title="Net benefit",
            width=600, height=600,
            yaxis=dict(range=[-0.1, prevalence + 0.1]),
        )
        return fig

    def _plot_risk_distribution(self, y_true, y_prob):
        """Violin/box plot of predicted probabilities for events vs non-events."""
        fig = go.Figure()
        fig.add_trace(go.Violin(
            x=y_true[y_true==0], y=y_prob[y_true==0],
            name="Class 0 (no event)", side='positive', line_color='blue',
        ))
        fig.add_trace(go.Violin(
            x=y_true[y_true==1], y=y_prob[y_true==1],
            name="Class 1 (event)", side='negative', line_color='red',
        ))
        fig.update_layout(
            title="Risk distribution by outcome",
            yaxis_title="Predicted probability",
            xaxis_title="Outcome",
            width=600, height=600,
            xaxis=dict(range=[-0.5, 1.5]),
            yaxis=dict(range=[0.0, 1.0]),
        )
        return fig
    
    def _plot_roc_curve(self, y_true, y_score):
        """
        Custom ROC Curve using Plotly (replacing wandb.plot.roc_curve)
        """
        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc_score = roc_auc_score(y_true, y_score)

        fig = go.Figure()
        
        # Diagonal reference line
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode='lines', name='Chance',
            line=dict(dash='dash', color='gray'), showlegend=False
        ))

        # Model ROC
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr, mode='lines', 
            name=f'Model (AUC = {auc_score:.3f})',
            line=dict(color='darkorange', width=2)
        ))

        fig.update_layout(
            title="ROC Curve",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            width=600, height=600,
            xaxis=dict(range=[0, 1], constrain='domain'),
            yaxis=dict(range=[0, 1], scaleanchor="x", scaleratio=1),
            margin=dict(l=40, r=40, t=60, b=40),
            legend=dict(x=0.6, y=0.1)
        )
        return fig


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
