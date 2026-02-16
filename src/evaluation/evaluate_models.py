import os
import wandb
import numpy as np
import torch
import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=SyntaxWarning, module="hdbscan")

from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    silhouette_score, adjusted_mutual_info_score, roc_curve,
    roc_auc_score, average_precision_score, precision_recall_curve,
    brier_score_loss, accuracy_score, balanced_accuracy_score,
    precision_score, recall_score, f1_score,
)
from transformers.trainer_utils import EvalPrediction
from scipy.stats import hmean, gmean
from scipy.special import expit

import optuna
from optuna.samplers import TPESampler
optuna.logging.set_verbosity(optuna.logging.WARNING)
import umap
import hdbscan
import gc
import io
from contextlib import redirect_stdout
import plotly.graph_objects as go
from plotly.colors import qualitative

from functools import lru_cache
@lru_cache(maxsize=1)
def _get_gpu_backend():
    """
    Return (cp, cuml, error_str). Cached so we try import once per process.
    """
    if os.environ.get("AIIDKIT_DISABLE_GPU", "").lower() in {"1", "true", "yes"}:
        return None, None, "Disabled by AIIDKIT_DISABLE_GPU"
    try:
        print("Importing cupy and cuml...")
        import cupy as cp
        import cuml
        return cp, cuml, ""
    except Exception as e:
        return None, None, f"{type(e).__name__}: {e}"


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
        compute_clusters: bool = True,
        use_cuml: bool = False,
        **kwargs
    ):
        """
        Reduce a set of embeddings with UMAP and cluster them with HDBSCAN
        """
        UMAP_KEYS = {"n_components", "n_neighbors", "min_dist"}
        HDBSCAN_KEYS = {"min_cluster_size", "min_samples"}
        umap_args = {k: v for k, v in kwargs.items() if k in UMAP_KEYS}
        hdbscan_args = {k: v for k, v in kwargs.items() if k in HDBSCAN_KEYS}

        if use_cuml:
            cp, cuml, err = _get_gpu_backend()
            if cp is None:
                # Optional: only print once, because _get_gpu_backend is cached.
                print(f"GPU mode disabled (cuml/cupy unavailable): {err}")
            else:
                try:
                    return self.reduce_and_cluster_gpu(
                        embeddings,
                        compute_clusters=compute_clusters,
                        umap_args=umap_args,
                        hdbscan_args=hdbscan_args,
                        cp=cp,
                        cuml=cuml,
                    )
                except Exception as e:
                    print(f"GPU clustering failed, falling back to CPU: {type(e).__name__}: {e}")

        return self.reduce_and_cluster_cpu(
            embeddings,
            compute_clusters=compute_clusters,
            umap_args=umap_args,
            hdbscan_args=hdbscan_args,
        )

    @staticmethod
    def reduce_and_cluster_gpu(
        embeddings: np.ndarray,
        compute_clusters: bool = True,
        umap_args: dict | None = None,
        hdbscan_args: dict | None = None,
        *, cp, cuml,
    ):
        """
        Reduce embeddings with UMAP and cluster them with HDBSCAN on GPU
        """
        umap_args = umap_args or {}
        hdbscan_args = hdbscan_args or {}
        
        embeddings_gpu = cp.asarray(embeddings, dtype=cp.float32)
        reducer = cuml.UMAP(**umap_args, verbose=False)
        with redirect_stdout(io.StringIO()):
            reduced_gpu = reducer.fit_transform(embeddings_gpu)
            
        if not compute_clusters:
            return cp.asnumpy(reduced_gpu).astype(np.float32), None

        clusterer = cuml.cluster.hdbscan.HDBSCAN(**hdbscan_args, verbose=False)
        labels_gpu = clusterer.fit_predict(reduced_gpu)
        reduced = cp.asnumpy(reduced_gpu).astype(np.float32)
        labels = cp.asnumpy(labels_gpu).astype(np.int32)  # keep labels integral

        del embeddings_gpu, reduced_gpu, labels_gpu
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()

        return reduced, labels


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
    def __init__(
        self,
        do_clustering: bool = False,
        n_optuna_trials: int = 0,
        max_clustered_samples: int = 2500,
        label_names: list[str] = None,
        early_stopping_metric: str = "roc_auc",
    ):
        self.do_clustering = do_clustering
        self.max_clustered_samples = max_clustered_samples
        self.label_names = label_names
        self.early_stopping_metric = early_stopping_metric
        self.current_prefix = "eval"
        self.calibrators = None 
        if do_clustering:
            self.clusterer_module = UMAP_HDBSCAN_Clusterer(n_optuna_trials=n_optuna_trials)

    def prepare_predictions(self, eval_preds: EvalPrediction) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        raise NotImplementedError

    def __call__(self, eval_preds: EvalPrediction) -> dict[str, float]:
        """
        This is called by Huggingface's Trainer at evaluation time.
        """
        # Extract logits, labels, embeddings, and output probabilities
        logits, labels, embeddings = self.prepare_predictions(eval_preds)
        if labels.ndim == 1:  # standardize shapes to [batch_size, num_labels]
            labels = labels[:, None]
            logits = logits[:, None]
        num_labels = labels.shape[1]
        probs = expit(logits)  # sigmoid is valid for both binary (1 col) and multi-label
        probs_cal = self._apply_calibration_strategy(labels, probs)

        # Prepare metrics for logging by Huggingface's Trainer
        metrics = {}
        valid_early_stopping_metric = []
        names = self.label_names if self.label_names else [f"lbl{i}" for i in range(num_labels)]
        for i, name in enumerate(names[:num_labels]):
            
            # Gather data (with or without an invalid label)
            y_true_all = labels[:, i]
            y_prob_all = probs[:, i]
            y_cal_all  = probs_cal[:, i]
            
            # Filter out invalid labels ("-100" means invalid)
            valid_mask = y_true_all != -100
            if valid_mask.sum() == 0: continue
            y_true = y_true_all[valid_mask]
            y_prob = y_prob_all[valid_mask]
            y_cal  = y_cal_all[valid_mask]
            
            # Compute metrics
            if len(np.unique(y_true)) < 2: continue   # skip constant labels
            sub_metrics = self._compute_single_label_metrics(y_true, y_prob, y_cal)
            metrics.update({f"{k}_{name}": v for k, v in sub_metrics.items()})
            
            # Record valid early stopping metrics
            if sub_metrics[self.early_stopping_metric] != -1.0:
                valid_early_stopping_metric.append(sub_metrics[self.early_stopping_metric])

        # Compute macro-average early stopping metric
        if valid_early_stopping_metric:
            valid_early_stopping_metric = np.array(valid_early_stopping_metric) + 1e-6
            # metrics["early_stopping_metric"] = np.mean(valid_early_stopping_metric)
            # metrics["early_stopping_metric"] = hmean(valid_early_stopping_metric)
            metrics["early_stopping_metric"] = gmean(valid_early_stopping_metric)
        else:
            metrics["early_stopping_metric"] = 0.0  # fallback if nothing valid

        # Log plots directly to WandB
        if wandb.run is not None:
            if num_labels == 1:  # detailed clinical plots (risk, calibration, decision curves)
                y_true = labels[:, 0]
                y_prob = probs[:, 0]
                y_cal = probs_cal[:, 0]
                _, preds = self._get_preds_from_threshold(y_true, y_prob, threshold=0.1)
                self._log_plots(y_true, preds, y_prob, y_cal)
            else:  # summary plots only
                self._log_multilabel_plots(labels, probs, probs_cal)

        return metrics

    def _apply_calibration_strategy(self, labels, probs):
        """Calibrate model output probabilities given outcome probabilities."""
        is_train = "test" not in self.current_prefix
        num_labels = labels.shape[1]

        # Init list if needed
        if is_train and self.calibrators is None:
            self.calibrators = [
                IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1) 
                for _ in range(num_labels)
            ]
        
        if self.calibrators is None:
            return probs

        probs_cal = np.zeros_like(probs)
        for i in range(num_labels):
            try:
                if is_train:
                    self.calibrators[i].fit(probs[:, i], labels[:, i])
                probs_cal[:, i] = self.calibrators[i].transform(probs[:, i])
            except Exception:
                probs_cal[:, i] = probs[:, i]  # fallback
        
        return np.clip(probs_cal, 0.0, 1.0)

    def _compute_single_label_metrics(self, y_true, y_prob, y_cal):
        """
        Calculates metrics given a fixed clinical threshold.
        """
        metrics = {}
        
        # Probability-based metrics
        metrics["brier"] = brier_score_loss(y_true, y_cal)
        metrics["ece"] = self._expected_calibration_error(y_true, y_cal)
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
            metrics["pr_auc"] = average_precision_score(y_true, y_prob)
        except ValueError:
            metrics["roc_auc"] = -1.0
            metrics["pr_auc"] = -1.0

        # Threshold-based metrics at multiple thresholds
        thresholds = [i / 10 for i in range(1, 10)] + ["best_f1"]
        for t in thresholds:
            
            # Determine threshold (if best_f1) and associated predictions
            t, preds = self._get_preds_from_threshold(y_true, y_prob, t)
            t_str = (f"t{int(t * 100)}" if isinstance(t, float) else t)
            
            # Log metrics
            metrics[f"acc_{t_str}"] = accuracy_score(y_true, preds)
            metrics[f"bal_acc_{t_str}"] = balanced_accuracy_score(y_true, preds)
            metrics[f"precision_{t_str}"] = precision_score(y_true, preds, zero_division=0)
            metrics[f"recall_{t_str}"] = recall_score(y_true, preds, zero_division=0)
            metrics[f"f1_{t_str}"] = f1_score(y_true, preds, zero_division=0)
            metrics[f"nb_{t_str}"] = self._calculate_net_benefit(y_true, y_prob, t)

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
        """
        Generates and logs detailed clinical plots for a SINGLE label (Binary case).
        Logs directly to WandB with the correct prefix.
        """
        try:
            plots = {}
            prefix = self.current_prefix
            
            # Probability-based plots
            plots[f"{prefix}/conf_mat"] = wandb.plot.confusion_matrix(probs=None, y_true=labels, preds=preds)
            plots[f"{prefix}/roc_curve"] = self._plot_roc_curve(labels, probs_pos)
            
            # Clinically relevant plots
            plots[f"{prefix}/decision_curve"] = self._plot_decision_curve(labels, probs_pos)
            plots[f"{prefix}/risk_distribution"] = self._plot_risk_distribution(labels, probs_pos)

            # Calibration plots
            # We plot both to see if Isotonic Regression actually helped
            plots[f"{prefix}/calibration_inf_raw"] = self._plot_calibration(labels, probs_pos, is_corr=False)
            plots[f"{prefix}/calibration_inf_cal"] = self._plot_calibration(labels, probs_cal, is_corr=True)

            # Log without committing (HuggingFace's Trainer commits at end of step)
            wandb.log(plots, commit=False)

        except Exception as e:
            print(f"Single-label plot logging error: {e}")

    def _log_multilabel_plots(self, labels, probs, probs_cal):
        """
        Generates summary plots for MULTIPLE labels.
        Plots all labels on the SAME chart for easy comparison.
        """
        try:
            prefix = self.current_prefix
            if self.label_names:
                label_names = self.label_names
            else:
                label_names = [f"L{i}" for i in range(labels.shape[1])]
            
            # Combined ROC Curve
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1], mode='lines', 
                line=dict(dash='dash', color='gray'), showlegend=False
            ))
            
            for i, name in enumerate(label_names):
                # Skip if label is constant (cannot compute ROC)
                if len(np.unique(labels[:, i])) < 2: continue
                
                fpr, tpr, _ = roc_curve(labels[:, i], probs[:, i])
                auc = roc_auc_score(labels[:, i], probs[:, i])
                fig_roc.add_trace(go.Scatter(
                    x=fpr, y=tpr, mode='lines', 
                    name=f"{name} (AUC={auc:.2f})", opacity=0.8
                ))
            
            fig_roc.update_layout(
                title=f"Multi-Label ROC ({prefix})", 
                xaxis_title="False Positive Rate", 
                yaxis_title="True Positive Rate", 
                width=700, height=500
            )
            
            # Combined calibration curve (reliability diagram)
            fig_cal = go.Figure()
            fig_cal.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1], mode='lines', 
                line=dict(dash='dash', color='gray'), showlegend=False
            ))
            
            for i, name in enumerate(label_names):
                if len(np.unique(labels[:, i])) < 2: continue

                # Compute calibration curve (observed vs predicted)
                prob_true, prob_pred = calibration_curve(labels[:, i], probs_cal[:, i], n_bins=10)
                fig_cal.add_trace(go.Scatter(
                    x=prob_pred, y=prob_true, mode='lines+markers', 
                    name=f"{name}", opacity=0.8
                ))
                
            fig_cal.update_layout(
                title=f"Multi-Label Calibration ({prefix})", 
                xaxis_title="Predicted Probability", 
                yaxis_title="Observed Fraction", 
                width=700, height=500
            )

            wandb.log({
                f"{prefix}/roc_combined": fig_roc,
                f"{prefix}/cal_combined": fig_cal
            }, commit=False)

        except Exception as e:
            print(f"Multi-label plot logging error: {e}")

    def _plot_roc_curve(self, y_true, y_score):
        """Standard ROC Curve using Plotly"""
        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc_score = roc_auc_score(y_true, y_score)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode='lines',
            line=dict(dash='dash', color='gray'), showlegend=False,
        ))
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr, mode='lines', 
            name=f'AUC = {auc_score:.3f}', line=dict(color='darkorange', width=2)
        ))
        fig.update_layout(
            title="ROC Curve", xaxis_title="FPR", yaxis_title="TPR", 
            width=600, height=600, xaxis=dict(range=[0, 1]), yaxis=dict(range=[0, 1])
        )
        return fig

    def _plot_calibration(self, y_true, y_prob, is_corr=False, n_bins=10):
        """Reliability Diagram"""
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)
        title_suffix = "(Calibrated)" if is_corr else "(Raw)"
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode='lines',
            line=dict(dash='dash', color='gray'), showlegend=False,
        ))
        fig.add_trace(go.Scatter(
            x=prob_pred, y=prob_true, mode='lines+markers', 
            name='Model', marker=dict(color='firebrick')
        ))
        fig.update_layout(
            title=f"Reliability {title_suffix}", 
            xaxis_title="Predicted", yaxis_title="Observed", 
            width=600, height=600, xaxis=dict(range=[0, 1]), yaxis=dict(range=[0, 1])
        )
        return fig

    def _plot_risk_distribution(self, y_true, y_prob):
        """Violin plot of probabilities split by outcome"""
        fig = go.Figure()
        fig.add_trace(go.Violin(
            x=y_true[y_true==0], y=y_prob[y_true==0], 
            name="Negatives", side='positive', line_color='blue'
        ))
        fig.add_trace(go.Violin(
            x=y_true[y_true==1], y=y_prob[y_true==1], 
            name="Positives", side='negative', line_color='red'
        ))
        fig.update_layout(
            title="Risk Distribution", yaxis_title="Predicted Probability", 
            width=600, height=600, yaxis=dict(range=[0, 1])
        )
        return fig

    def _plot_decision_curve(self, y_true, y_prob):
        """Decision curve analysis (DCA), showing net nenefit vs threshold"""
        thresholds = np.linspace(0.01, 0.99, 100)
        net_benefits = [self._calculate_net_benefit(y_true, y_prob, t) for t in thresholds]
        prev = np.sum(y_true) / len(y_true)
        nb_all = prev - (1 - prev) * (thresholds / (1 - thresholds))
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=thresholds, y=net_benefits, mode='lines', name='Model',
        ))
        fig.add_trace(go.Scatter(
            x=thresholds, y=nb_all, mode='lines',
            name='Treat All', line=dict(dash='dot', color='gray'),
        ))
        fig.add_trace(go.Scatter(
            x=thresholds, y=np.zeros_like(thresholds), mode='lines',
            name='Treat None', line=dict(color='black'),
        ))
        fig.update_layout(
            title="Decision Curve Analysis", 
            xaxis_title="Threshold", yaxis_title="Net Benefit", 
            width=600, height=600, yaxis=dict(range=[-0.05, prev + 0.1])
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
