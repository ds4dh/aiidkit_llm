import os
import wandb
import numpy as np
import torch
import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=SyntaxWarning, module="hdbscan")

from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, silhouette_score, roc_curve,
    roc_auc_score, average_precision_score, precision_recall_curve,
    brier_score_loss,
)
from transformers.trainer_utils import EvalPrediction
from scipy.stats import hmean, gmean
from scipy.special import expit, logit

import optuna
from optuna.samplers import TPESampler
optuna.logging.set_verbosity(optuna.logging.WARNING)
try:
    import umap
except ImportError:
    umap = None

try:
    import hdbscan
except ImportError:
    try:
        from sklearn import cluster as hdbscan
    except ImportError:
        hdbscan = None

import gc
import io
from tqdm import tqdm
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


_GPU_WARNING_PRINTED = False
_CPU_WARNING_PRINTED = False

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
            "min_cluster_size": max(2, int(num_samples / 10)),
            "min_samples": max(1, int(num_samples / 20)),
            "n_neighbors": 15,
            "min_dist": 0.1,
        }

        # Run hyperparameter optimization if Optuna is available
        def objective(trial):
            n_components = trial.suggest_int("n_components", 2, min(30, max(2, num_samples - 2)))
            min_cluster_size = trial.suggest_int("min_cluster_size", 2, max(3, int(num_samples / 5)))
            min_samples = trial.suggest_int("min_samples", 1, max(2, int(num_samples / 10)))
            n_neighbors = trial.suggest_int("n_neighbors", 5, max(6, int(num_samples / 2)))
            min_dist = trial.suggest_float("min_dist", 0.0, 0.99)

            umap_args = {
                "n_components": n_components,
                "n_neighbors": n_neighbors,
                "min_dist": min_dist,
            }
            hdbscan_args = {
                "min_cluster_size": min_cluster_size,
                "min_samples": min_samples,
            }

            try:
                reduced_data, labels = self.predict(
                    pooled_embeddings,
                    compute_clusters=True,
                    **umap_args,
                    **hdbscan_args,
                )
                unique_labels = np.unique(labels)
                n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
                if n_clusters > 1:
                    score = silhouette_score(reduced_data, labels)
                    return float(score)
                return -1.0
            except Exception:
                return -1.0

        study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=42))
        effective_trials = min(self.n_optuna_trials, 2) if (_get_gpu_backend()[0] is None or num_samples < 50) else self.n_optuna_trials
        study.optimize(objective, n_trials=effective_trials, show_progress_bar=True)

        if study.best_trial and study.best_value > -1.0:
            best_params.update(study.best_params)

        return best_params

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
        global _GPU_WARNING_PRINTED
        UMAP_KEYS = {"n_components", "n_neighbors", "min_dist"}
        HDBSCAN_KEYS = {"min_cluster_size", "min_samples"}
        umap_args = {k: v for k, v in kwargs.items() if k in UMAP_KEYS}
        hdbscan_args = {k: v for k, v in kwargs.items() if k in HDBSCAN_KEYS}

        if use_cuml:
            cp, cuml, err = _get_gpu_backend()
            if cp is None:
                if not _GPU_WARNING_PRINTED:
                    print(f"[INFO] RAPIDS GPU acceleration unavailable ({err}). Using CPU backend for UMAP/HDBSCAN.")
                    _GPU_WARNING_PRINTED = True
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
        global _CPU_WARNING_PRINTED
        if umap is None or hdbscan is None:
            if not _CPU_WARNING_PRINTED:
                missing = []
                if umap is None:
                    missing.append("umap-learn")
                if hdbscan is None:
                    missing.append("hdbscan")
                print(f"[WARNING] Clustering skipped: {', '.join(missing)} is missing. Install with: uv pip install {' '.join(missing)}")
                _CPU_WARNING_PRINTED = True
            return np.zeros((len(embeddings), 2)), np.full(len(embeddings), -1)

        # Compute dimensionality-reduced embeddings on CPU
        effective_umap_args = dict(umap_args)
        effective_umap_args.setdefault("random_state", 42)
        n_samples = len(embeddings)
        if n_samples <= effective_umap_args.get("n_neighbors", 15):
            effective_umap_args["n_neighbors"] = max(2, n_samples - 1)
        reducer = umap.UMAP(**effective_umap_args)
        reduced_embeddings = reducer.fit_transform(embeddings)

        # Only reduce embeddings if clustering is not needed
        if not compute_clusters:
            return reduced_embeddings, None

        # Compute clusters on CPU
        effective_hdbscan_args = dict(hdbscan_args)
        if n_samples < effective_hdbscan_args.get("min_cluster_size", 15) * 2:
            mcs = max(2, n_samples // 3)
            effective_hdbscan_args["min_cluster_size"] = mcs
            effective_hdbscan_args["min_samples"] = min(effective_hdbscan_args.get("min_samples", 5), max(1, mcs - 1))
        clusterer = hdbscan.HDBSCAN(**effective_hdbscan_args)
        cluster_labels = clusterer.fit_predict(reduced_embeddings)
        if len(set(cluster_labels) - {-1}) < 2 and n_samples >= 4:
            from sklearn.cluster import KMeans
            cluster_labels = KMeans(n_clusters=2, random_state=42, n_init="auto").fit_predict(reduced_embeddings)

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

        # Compute MLM accuracy (overall + split by discrete tokens vs continuous quantile-bins)
        preds_flat, labels_flat = pred_ids.flatten(), labels.flatten()
        mask = labels_flat != -100  # filtering out ignored tokens
        metrics["mlm_accuracy"] = float(accuracy_score(labels_flat[mask], preds_flat[mask])) if mask.any() else 0.0
        disc_mask = labels_flat >= 0
        if disc_mask.any():
            metrics["mlm_discrete_accuracy"] = float(accuracy_score(labels_flat[disc_mask], preds_flat[disc_mask]))
        cont_mask = (labels_flat < 0) & (labels_flat != -100)
        if cont_mask.any():
            metrics["mlm_continuous_accuracy"] = float(accuracy_score(labels_flat[cont_mask], preds_flat[cont_mask]))
        
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


def _grouped_cummax(probs: np.ndarray, group_sizes: list[int] | None) -> np.ndarray:
    """np.maximum.accumulate along the column axis, within contiguous column groups.

    group_sizes=None -> one global group (classic single-task horizon ordering).
    Otherwise each contiguous block of columns (one task's horizon slice in a
    multi-task run) is made monotonic independently.
    """
    if not group_sizes or sum(group_sizes) != probs.shape[1]:
        return np.maximum.accumulate(probs, axis=1)
    out = np.empty_like(probs)
    start = 0
    for size in group_sizes:
        out[:, start:start + size] = np.maximum.accumulate(probs[:, start:start + size], axis=1)
        start += size
    return out


class BaseEmbeddingEvaluatorForClassification:
    def __init__(
        self,
        do_clustering: bool = False,
        n_optuna_trials: int = 0,
        max_clustered_samples: int = 2500,
        label_names: list[str] = None,
        early_stopping_metric: str = "roc_auc",
        enforce_monotonicity: bool = False,
        monotonic_groups: list[int] | None = None,
        early_stopping_subset: list[str] | None = None,
        early_stopping_mean: str = "harmonic",
        n_bootstrap_eval: int = 100,
    ):
        self.do_clustering = do_clustering
        self.max_clustered_samples = max_clustered_samples
        self.label_names = label_names
        self.early_stopping_metric = early_stopping_metric
        self.enforce_monotonicity = enforce_monotonicity
        # Contiguous column-group sizes for monotonicity (multi-task runs pass
        # one group per task; None = all columns form a single horizon chain).
        self.monotonic_groups = monotonic_groups
        # Heads contributing to the early-stopping/checkpoint-selection score.
        # None = every valid head votes (default, classic single-task runs).
        # Multi-task runs can restrict the vote to reliable task keys (all heads
        # are still COMPUTED and reported as metrics regardless).
        self.early_stopping_subset = early_stopping_subset
        # Aggregation across the voting heads: 'harmonic' (default, punishes the
        # weakest head), 'mean' (arithmetic) or 'geometric'.
        self.early_stopping_mean = early_stopping_mean
        self.n_bootstrap_eval = int(n_bootstrap_eval or 0)
        self.current_prefix = "eval"
        self.calibrators = None
        self.saved_labels_and_probs = None
        if do_clustering:
            self.clusterer_module = UMAP_HDBSCAN_Clusterer(n_optuna_trials=n_optuna_trials)

    def prepare_predictions(self, eval_preds: EvalPrediction) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        raise NotImplementedError

    def _in_early_stopping_subset(self, label_name: str) -> bool:
        """Whether the head named `label_name` votes on the early-stopping metric.

        Label names follow the 'label_{task}_{H:04d}d' convention; a task must
        match the FULL task segment (so task 'infection' does not accidentally
        match heads of 'infection_bacteria'). Unnamed fallback columns
        ('lbl0') always vote.
        """
        if not self.early_stopping_subset:
            return True
        if not label_name.startswith("label_"):
            return True
        for t in self.early_stopping_subset:
            prefix = f"label_{t}_"
            rest = label_name[len(prefix):]
            if label_name.startswith(prefix) and len(rest) == 5 \
                    and rest[:4].isdigit() and rest[4] == "d":
                return True
        return False

    def _bootstrap_early_stopping_score(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        y_cal: np.ndarray,
        seed: int = 1234,
    ) -> tuple[float, float, float]:
        """
        Compute `self.n_bootstrap_eval` paired stratified bootstrap replicates of
        `self.early_stopping_metric` using a fixed RNG seed across evaluation steps.
        Returns (conservative_score = mean - 0.5 * std, boot_mean, boot_std) so
        checkpoint selection favors high performance with low bootstrap variance.
        """
        pos_idx = np.where(y_true == 1)[0]
        neg_idx = np.where(y_true == 0)[0]
        n_pos, n_neg = len(pos_idx), len(neg_idx)
        if n_pos == 0 or n_neg == 0:
            return 0.0, 0.0, 0.0

        rng = np.random.default_rng(seed)
        metric_name = self.early_stopping_metric
        boot_vals = np.empty(self.n_bootstrap_eval, dtype=np.float64)

        if metric_name not in ("pr_auc", "brier"):
            # Fast exact Mann-Whitney U (with tie-averaging) stratified bootstrap for ROC-AUC (~15x faster on 90+ heads)
            pos_scores = np.asarray(y_prob[pos_idx], dtype=np.float64)
            neg_scores = np.asarray(y_prob[neg_idx], dtype=np.float64)
            inv_n_neg = 1.0 / float(n_neg)
            for b in range(self.n_bootstrap_eval):
                p_b = pos_scores[rng.integers(0, n_pos, size=n_pos)]
                n_b = np.sort(neg_scores[rng.integers(0, n_neg, size=n_neg)])
                left = np.searchsorted(n_b, p_b, side="left")
                right = np.searchsorted(n_b, p_b, side="right")
                boot_vals[b] = float(np.mean(0.5 * (left + right))) * inv_n_neg
        else:
            y_boot = np.concatenate([np.ones(n_pos, dtype=y_true.dtype), np.zeros(n_neg, dtype=y_true.dtype)])
            for b in range(self.n_bootstrap_eval):
                b_pos = pos_idx[rng.integers(0, n_pos, size=n_pos)]
                b_neg = neg_idx[rng.integers(0, n_neg, size=n_neg)]
                b_idx = np.concatenate([b_pos, b_neg])
                if metric_name == "pr_auc":
                    boot_vals[b] = average_precision_score(y_boot, y_prob[b_idx])
                else:
                    # Invert Brier into (1 - Brier) so higher-is-better holds across all aggregators
                    boot_vals[b] = 1.0 - brier_score_loss(y_boot, y_cal[b_idx])

        boot_mean = float(np.mean(boot_vals))
        boot_std = float(np.std(boot_vals))
        # Variance-penalized bootstrap score (rewards stability across resampled cohorts)
        boot_score = float(max(0.0, boot_mean - 0.5 * boot_std))
        return boot_score, boot_mean, boot_std

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
        if self.enforce_monotonicity and num_labels > 1:
            probs_cal = _grouped_cummax(probs_cal, self.monotonic_groups)
            
        # Expose arrays for saving (useful for the final test)
        self.last_logits = logits
        self.saved_labels_and_probs = {
            "labels": labels,
            "probs": probs,
            "probs_cal": probs_cal,
            # "embeddings": embeddings,  # could be used for cluster analysis
        }

        # Prepare metrics for logging by Huggingface's Trainer
        metrics = {}
        valid_early_stopping_metric = []
        valid_early_stopping_raw_means = []
        valid_early_stopping_stds = []
        valid_early_stopping_names = []
        valid_early_stopping_pos_counts = []
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
            n_pos_i = int(np.sum(y_true == 1))
            metrics[f"n_pos_{name}"] = n_pos_i
            
            # Record valid early stopping metrics (optionally restricted to the
            # heads of the early_stopping_subset tasks; every head is still
            # reported as a metric regardless of the vote)
            if sub_metrics[self.early_stopping_metric] != -1.0 \
                    and self._in_early_stopping_subset(name):
                if self.n_bootstrap_eval > 0 and self.current_prefix.startswith("eval"):
                    boot_score, boot_mean, boot_std = self._bootstrap_early_stopping_score(
                        y_true, y_prob, y_cal, seed=1234 + i
                    )
                    metrics[f"{self.early_stopping_metric}_boot_mean_{name}"] = boot_mean
                    metrics[f"{self.early_stopping_metric}_boot_std_{name}"] = boot_std
                    valid_early_stopping_metric.append(boot_score)
                    valid_early_stopping_raw_means.append(boot_mean)
                    valid_early_stopping_stds.append(boot_std)
                else:
                    pt_val = float(sub_metrics[self.early_stopping_metric])
                    approx_se = 0.5 / np.sqrt(max(1.0, float(n_pos_i)))
                    valid_early_stopping_metric.append(pt_val)
                    valid_early_stopping_raw_means.append(pt_val)
                    valid_early_stopping_stds.append(float(approx_se))
                valid_early_stopping_names.append(name)
                valid_early_stopping_pos_counts.append(n_pos_i)

        # Aggregate the early-stopping score across the voting heads
        if valid_early_stopping_metric:
            vals = np.array(valid_early_stopping_metric) + 1e-6
            kind = (self.early_stopping_mean or "harmonic").lower()
            if kind in ("random_effects", "precision_capped"):
                # Hierarchical DerSimonian-Laird random-effects pooling:
                # 1) Within each task, weight horizons by w_h = 1 / (tau^2 + se_h^2) so noisy
                #    low-count horizons self-dampen without letting huge horizons monopolize.
                # 2) Across tasks, combine task scores with w_t = 1 / (tau^2 + se_t^2) so every
                #    well-powered outcome shares nearly equal influence while high-variance tasks shrink.
                tau2 = 0.03 ** 2
                task_buckets: dict[str, list[tuple[float, float]]] = {}
                for nm, m_val, se_val in zip(valid_early_stopping_names, valid_early_stopping_raw_means, valid_early_stopping_stds):
                    # Strip trailing `_XXXXd` horizon tag to group heads by clinical outcome task
                    t_base = nm[:-6] if (len(nm) > 6 and nm[-1] == "d" and nm[-5:-1].isdigit() and nm[-6] == "_") else nm
                    task_buckets.setdefault(t_base, []).append((float(m_val), float(se_val)))
                task_scores = []
                task_weights = []
                for items in task_buckets.values():
                    ms = np.array([x[0] for x in items], dtype=np.float64)
                    ses = np.array([x[1] for x in items], dtype=np.float64)
                    ws = 1.0 / (tau2 + ses ** 2)
                    w_sum = float(np.sum(ws))
                    t_mean = float(np.sum(ws * ms) / w_sum)
                    t_var = float(np.sum((ws ** 2) * (ses ** 2)) / (w_sum ** 2))
                    task_scores.append(t_mean)
                    task_weights.append(1.0 / (tau2 + t_var))
                metrics["early_stopping_metric"] = float(
                    np.average(np.array(task_scores, dtype=np.float64), weights=np.array(task_weights, dtype=np.float64))
                )
            elif kind in ("mean", "arithmetic"):
                metrics["early_stopping_metric"] = float(np.mean(vals))
            elif kind in ("log_support", "support_weighted"):
                weights = np.log1p(np.maximum(1, np.array(valid_early_stopping_pos_counts, dtype=np.float64)))
                metrics["early_stopping_metric"] = float(np.average(vals, weights=weights))
            elif kind == "geometric":
                metrics["early_stopping_metric"] = float(gmean(vals))
            else:  # 'harmonic' (default): most conservative, punishes the worst head
                metrics["early_stopping_metric"] = float(hmean(vals))
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
        """
        Calibrate model output probabilities using monotone Platt scaling:
        - Fits on validation (`eval*`, `validation`, `validation_all`) only; resets calibrators
          freshly on each fit pass so stale checkpoint calibrators are never reused.
        - Guards against finite-sample slope inversion (`coef <= 0.05` or `n_pos < 20`) by
          falling back to intercept-only prior calibration (`1.0 * logit(p) + b`), strictly
          preserving within-head rank order (AUROC & PR-AUC invariant).
        """
        is_train = self.current_prefix.startswith("eval") or self.current_prefix in ("validation", "validation_all")
        num_labels = labels.shape[1]

        if is_train:
            self.calibrators = [None] * num_labels
        elif self.calibrators is None:
            print(f"[WARN] No fitted calibrators found for prefix '{self.current_prefix}'; returning raw probabilities.")
            return probs

        probs_cal = np.zeros_like(probs)
        for i in range(num_labels):
            valid_mask = (labels[:, i] == 0) | (labels[:, i] == 1)
            if valid_mask.sum() == 0:
                probs_cal[:, i] = probs[:, i]
                continue

            valid_labels = labels[valid_mask, i].astype(np.int64)
            safe_probs = np.clip(probs[valid_mask, i], 1e-7, 1.0 - 1.0e-7)
            logits_i = logit(safe_probs)

            try:
                if is_train and len(np.unique(valid_labels)) > 1:
                    n_pos = int(np.sum(valid_labels == 1))
                    slope, intercept = 1.0, 0.0
                    fitted = False
                    if n_pos >= 20:
                        lr = LogisticRegression(C=1.0, solver="lbfgs", random_state=42)
                        lr.fit(logits_i.reshape(-1, 1), valid_labels)
                        if hasattr(lr, "coef_") and float(lr.coef_[0, 0]) > 0.05:
                            slope = float(lr.coef_[0, 0])
                            intercept = float(lr.intercept_[0])
                            fitted = True
                    if not fitted:
                        # Calibration-in-the-large (intercept-only offset; strictly order-preserving)
                        p_bar = float(np.clip(np.mean(valid_labels), 1e-6, 1.0 - 1e-6))
                        slope = 1.0
                        intercept = float(logit(p_bar) - np.mean(logits_i))
                    self.calibrators[i] = (slope, intercept)

                cal_params = self.calibrators[i] if i < len(self.calibrators) else None
                if cal_params is not None:
                    slope, intercept = cal_params
                    probs_cal[valid_mask, i] = expit(slope * logits_i + intercept)
                else:
                    probs_cal[valid_mask, i] = probs[valid_mask, i]
            except Exception:
                probs_cal[valid_mask, i] = probs[valid_mask, i]

            probs_cal[~valid_mask, i] = probs[~valid_mask, i]

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

        # Vectorized threshold-based metrics across fixed grid [0.1..0.9] + F1-optimal threshold ("best_f1")
        y_t = np.asarray(y_true, dtype=np.int64)
        y_p = np.asarray(y_prob, dtype=np.float64)
        n_total = len(y_t)
        pos_mask = (y_t == 1)
        n_pos = float(np.sum(pos_mask))
        n_neg = float(n_total - n_pos)

        fixed_ts = np.arange(1, 10, dtype=np.float64) * 0.1
        best_f1_t, _ = self._get_preds_from_threshold(y_t, y_p, "best_f1")
        all_ts = np.append(fixed_ts, float(best_f1_t))
        t_names = [f"t{int(round(t * 100))}" for t in fixed_ts] + ["best_f1"]

        pred_mat = y_p[:, None] >= all_ts[None, :]  # [N, 10]
        tp = np.sum(pred_mat & pos_mask[:, None], axis=0).astype(np.float64)
        fp = np.sum(pred_mat & (~pos_mask)[:, None], axis=0).astype(np.float64)
        tn = n_neg - fp

        acc_arr = (tp + tn) / max(float(n_total), 1.0)
        rec_arr = np.where(n_pos > 0, tp / np.maximum(n_pos, 1.0), 0.0)
        spec_arr = np.where(n_neg > 0, tn / np.maximum(n_neg, 1.0), 0.0)
        bal_acc_arr = 0.5 * (rec_arr + spec_arr)
        pred_pos = tp + fp
        prec_arr = np.where(pred_pos > 0, tp / np.maximum(pred_pos, 1.0), 0.0)
        denom_f1 = prec_arr + rec_arr
        f1_arr = np.where(denom_f1 > 0, 2.0 * prec_arr * rec_arr / np.maximum(denom_f1, 1e-12), 0.0)
        safe_ts = np.clip(all_ts, 1e-6, 1.0 - 1e-6)
        nb_arr = (tp / max(float(n_total), 1.0)) - (fp / max(float(n_total), 1.0)) * (safe_ts / (1.0 - safe_ts))

        for j, t_str in enumerate(t_names):
            metrics[f"acc_{t_str}"] = float(acc_arr[j])
            metrics[f"bal_acc_{t_str}"] = float(bal_acc_arr[j])
            metrics[f"precision_{t_str}"] = float(prec_arr[j])
            metrics[f"recall_{t_str}"] = float(rec_arr[j])
            metrics[f"f1_{t_str}"] = float(f1_arr[j])
            metrics[f"nb_{t_str}"] = float(nb_arr[j])

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
                return 0.5, np.zeros_like(y_true)

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
            
            # Combined calibration curve (reliability diagram)
            fig_cal = go.Figure()
            fig_cal.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1], mode='lines', 
                line=dict(dash='dash', color='gray'), showlegend=False
            ))
            for i, name in enumerate(label_names):
                
                # Masking logic for metrics computation
                valid_mask = labels[:, i] != -100
                if valid_mask.sum() == 0: continue
                valid_labels = labels[valid_mask, i]
                valid_probs = probs[valid_mask, i]
                valid_probs_cal = probs_cal[valid_mask, i]
                if len(np.unique(valid_labels)) < 2: continue
                
                # ROC Curve
                fpr, tpr, _ = roc_curve(valid_labels, valid_probs)
                auc = roc_auc_score(valid_labels, valid_probs)
                fig_roc.add_trace(go.Scatter(
                    x=fpr, y=tpr, mode='lines', 
                    name=f"{name} (AUC={auc:.2f})", opacity=0.8
                ))
            
                # Calibration curve
                prob_true, prob_pred = calibration_curve(valid_labels, valid_probs_cal, n_bins=10)
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


class DiscriminativeEmbeddingEvaluatorForSurvival(DiscriminativeEmbeddingEvaluatorForClassification):
    """
    Evaluator for the survival fine-tuning task (Cox proportional hazards).

    The model outputs a single log-hazard-ratio per patient. This evaluator:
        - computes Harrell's C-index on (time, status) labels,
        - estimates a Breslow baseline survival on the evaluation cohort to
          convert risk scores into predicted survival probabilities S(h|x)
          at the configured reporting horizons,
        - binarizes survival labels at those horizons to report AUROC / AP /
          IPCW-Brier, directly comparable with the binary classifiers,
        - exposes (labels=[B,K], probs=[B,K]) arrays so the final test pass
          saves the same JSON/NPZ structure as the classification pipeline.
    """

    def __init__(
        self,
        horizons: list[int],
        early_stopping_metric: str = "cindex",
        **kwargs,
    ):
        kwargs["do_clustering"] = False  # keep evaluation lightweight
        super().__init__(early_stopping_metric=early_stopping_metric, **kwargs)
        self.horizons = sorted(int(h) for h in horizons)
        self.current_prefix = "eval"
        self.saved_labels_and_probs = None
        # When True, scoring passes reuse the Breslow baseline previously
        # fitted on the matching VALIDATION segment instead of refitting on
        # the scored cohort (used for the final test pass so that absolute
        # probabilities never depend on test-set outcomes).
        self.freeze_baseline_for_test = False
        self.fitted_baselines: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def _baseline_for_cohort(
        self,
        times: np.ndarray,
        statuses: np.ndarray,
        risk: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (event_times, cumulative_baseline_hazard) for this scoring pass.

        Baselines are stratified per scoring cohort (one pass per FUP segment),
        so each landmark keeps its own event clock (early FUPs have their own
        high early hazard, late FUPs theirs). For the final test pass
        (freeze_baseline_for_test=True) the baseline fitted on the matching
        validation segment is reused: the probability scale then depends only
        on validation data, exactly like the downstream Platt calibration and
        the recall-target thresholds.
        """
        from src.evaluation.survival_metrics import breslow_baseline

        if self.freeze_baseline_for_test:
            frozen = self._lookup_frozen_baseline(self.current_prefix)
            if frozen is not None:
                return frozen
        event_times, baseline_hazard = breslow_baseline(times, statuses, risk)
        self.fitted_baselines[self.current_prefix] = (event_times, baseline_hazard)
        return event_times, baseline_hazard

    def _lookup_frozen_baseline(self, prefix: str):
        """Find the validation baseline for a test scoring pass.

        Prefixes look like 'test_fup_0030' / 'validation_fup_0030' / 'test_all'.
        Preference: exact validation counterpart -> nearest FUP segment
        (handles test-only FUPs) -> the validation 'all' mega pass -> None
        (falls back to fitting on the scored cohort).
        """
        import re

        candidates = {k: v for k, v in self.fitted_baselines.items() if k.startswith("validation")}
        if not candidates:
            return None
        counterpart = prefix.replace("test", "validation", 1)
        if counterpart in candidates:
            return candidates[counterpart]
        m = re.search(r"fup_(\d+)", prefix)
        if m:
            target = int(m.group(1))
            scored = sorted(
                (abs(int(mm.group(1)) - target), k)
                for k in candidates
                if (mm := re.search(r"fup_(\d+)", k))
            )
            if scored:
                return candidates[scored[0][1]]
        return candidates.get("validation_all")

    def __call__(self, eval_preds: EvalPrediction) -> dict[str, float]:
        from src.data.survival_labels import binarize_survival_labels
        from src.evaluation.survival_metrics import (
            breslow_baseline, survival_matrix_from_baseline,
            ipcw_brier_scores, concordance_index,
        )

        # Unpack model outputs: risk = log hazard ratio
        logits, labels, embeddings = self.prepare_predictions(eval_preds)
        self.last_logits = logits
        self.last_raw_labels = labels
        risk = logits.reshape(logits.shape[0], -1)[:, 0].astype(np.float64)
        if labels.ndim == 1 or labels.shape[1] < 2:
            raise ValueError("Survival labels must contain (time, status) columns")
        times = labels[:, 0].astype(np.float64)
        statuses = labels[:, 1].astype(np.int64)
        valid = (times != -100) & (statuses != -100)
        times, statuses, risk = times[valid], statuses[valid], risk[valid]

        metrics: dict[str, float] = {}
        if len(times) == 0 or np.unique(statuses).size == 0:
            metrics["cindex"] = 0.0
            metrics["early_stopping_metric"] = 0.0
            return metrics

        # 1) Concordance (discrimination, horizon-free)
        cindex = concordance_index(times, statuses, risk)
        metrics["cindex"] = 0.0 if np.isnan(cindex) else float(cindex)

        # 2) Baseline survival -> S(h|x) at reporting horizons (per-FUP
        #    stratified; frozen to the validation-fitted baseline on test)
        event_times, baseline_hazard = self._baseline_for_cohort(times, statuses, risk)
        surv = survival_matrix_from_baseline(event_times, baseline_hazard, risk, self.horizons)
        risk_probs = np.clip(1.0 - surv, 0.0, 1.0)  # P(event within horizon h)

        # 2b) Predicted median time to event per patient: smallest t with
        #     S(t|x) <= 0.5, i.e. H0(t) >= ln(2)/exp(r); NaN if never reached
        #     within the observed baseline follow-up window.
        if len(event_times) > 0 and len(baseline_hazard) > 0:
            target_h0 = np.log(2.0) / np.exp(np.clip(risk, -30, 30))
            med_idx = np.searchsorted(baseline_hazard, target_h0, side="left")
            median_times = np.where(
                med_idx < len(event_times),
                event_times[np.clip(med_idx, 0, len(event_times) - 1)],
                np.nan,
            )
        else:
            median_times = np.full(len(risk), np.nan)

        # 3) Horizon-wise binary metrics (comparable with binary classifiers)
        binary_labels = np.stack(
            [binarize_survival_labels(times, statuses, h) for h in self.horizons], axis=1
        )
        briers = ipcw_brier_scores(times, statuses, surv, self.horizons)
        for k, h in enumerate(self.horizons):
            y_true = binary_labels[:, k]
            y_prob = risk_probs[:, k]
            mask = y_true != -100
            h_str = f"{h:04d}d"
            if mask.sum() > 0 and len(np.unique(y_true[mask])) >= 2:
                metrics[f"roc_auc_{h_str}"] = float(roc_auc_score(y_true[mask], y_prob[mask]))
                metrics[f"pr_auc_{h_str}"] = float(average_precision_score(y_true[mask], y_prob[mask]))
            else:
                metrics[f"roc_auc_{h_str}"] = -1.0
                metrics[f"pr_auc_{h_str}"] = -1.0
            metrics[f"ipcw_brier_{h_str}"] = float(briers[k])

        # 4) Early-stopping metric
        if self.early_stopping_metric == "cindex":
            metrics["early_stopping_metric"] = metrics["cindex"]
        else:
            vals = [
                metrics[f"{self.early_stopping_metric}_{h:04d}d"]
                for h in self.horizons
                if metrics.get(f"{self.early_stopping_metric}_{h:04d}d", -1.0) > 0.0
            ]
            metrics["early_stopping_metric"] = float(np.mean(vals)) if vals else 0.0

        # 5) Expose arrays for the final test pass (the tidy predictions CSV
        #    keeps: horizon-binarized labels, predicted event probabilities,
        #    and the raw survival ground truth / model outputs per row)
        self.saved_labels_and_probs = {
            "labels": binary_labels,
            "probs": risk_probs,
            "times": times,
            "statuses": statuses,
            "risks": risk,
            "median_times": median_times,
            "horizons": np.asarray(self.horizons),
        }

        # 6) WandB plots: KM curves by risk tertile
        if wandb.run is not None:
            try:
                wandb.log(
                    {f"{self.current_prefix}/km_risk_tertiles": self._plot_km_tertiles(times, statuses, risk)},
                    commit=False,
                )
            except Exception as e:
                print(f"Survival KM plot logging error: {e}")

        return metrics

    @staticmethod
    def _plot_km_tertiles(times, statuses, risk):
        """Kaplan-Meier survival curves stratified by predicted-risk tertiles."""
        from lifelines import KaplanMeierFitter
        q33, q66 = np.quantile(risk, [1 / 3, 2 / 3])
        strata = np.select([risk <= q33, risk <= q66], ["Low risk", "Mid risk"], default="High risk")
        fig = go.Figure()
        for name in ["Low risk", "Mid risk", "High risk"]:
            m = strata == name
            if m.sum() == 0:
                continue
            kmf = KaplanMeierFitter().fit(times[m], statuses[m], label=f"{name} (n={int(m.sum())})")
            sf = kmf.survival_function_
            fig.add_trace(go.Scatter(
                x=sf.index, y=sf.iloc[:, 0], mode="steps",
                name=name, line=dict(shape="hv"),
            ))
        fig.update_layout(
            title="Survival by predicted-risk tertile",
            xaxis_title="Days since observation point", yaxis_title="Survival probability",
            width=700, height=500,
        )
        return fig


class ModelInterpreter:
    """Extracts embeddings and logits from the model for a given dataloader."""
    def __init__(self, model, device=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = model
        self.model.eval()
        self.model.to(device)
        self.device = device

    def get_embeddings_and_predictions(self, dataloader, extract_logits=True, use_classifier_head=False):
        """
        Extract sequence representations from the common Transformer trunk (`outputs.hidden_states[-1]`)
        so pre-trained and fine-tuned embeddings live in the exact same architectural layer across tasks.

        Args:
            extract_logits: Set to False for pre-trained models (MLM) where sequence lengths
                            vary by batch, causing np.vstack to crash.
            use_classifier_head: Keep False (default) to extract the shared Transformer trunk
                                 representation prior to any task-specific projection head.
        """
        embeddings_list, logits_list, labels_list = [], [], []
        device_type = "cuda" if ("cuda" in str(self.device) and torch.cuda.is_available()) else "cpu"
        try:
            param_dtype = next(self.model.parameters()).dtype
        except StopIteration:
            param_dtype = torch.float32
        target_dtype = param_dtype if (device_type == "cuda" and param_dtype in (torch.float16, torch.bfloat16)) else (
            torch.bfloat16 if device_type == "cuda" else torch.float32
        )
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Extracting embeddings"):
                input_dict = {k: v.to(self.device) for k, v in batch["input_dict"].items()}
                attn_mask = batch["attention_mask"].to(self.device)

                with torch.autocast(device_type=device_type, dtype=target_dtype):
                    outputs = self.model(
                        input_dict=input_dict,
                        attention_mask=attn_mask,
                        output_hidden_states=True,
                    )
                    last_hidden = outputs.hidden_states[-1]
                    pooling_mode = getattr(self.model.config, "classifier_pooling", "mean")
                    if pooling_mode == "last":
                        seq_lens = attn_mask.sum(dim=1).clamp(min=1) - 1
                        batch_idx = torch.arange(last_hidden.size(0), device=last_hidden.device)
                        pooled = last_hidden[batch_idx, seq_lens]
                    else:
                        mask = attn_mask.unsqueeze(-1)
                        sum_embeddings = (last_hidden * mask).sum(dim=1)
                        sum_mask = mask.sum(dim=1).clamp(min=1e-9)
                        pooled = sum_embeddings / sum_mask

                    if use_classifier_head and hasattr(self.model, "classifier") and hasattr(self.model, "head"):
                        head_dtype = next(self.model.head.parameters()).dtype
                        pooled = self.model.head(pooled.to(dtype=head_dtype))
                
                embeddings_list.append(pooled.float().cpu().numpy())
                labels_list.append(batch["labels"].cpu().numpy())
                
                if extract_logits:
                    logits_list.append(outputs.logits.float().cpu().numpy())
        
        res = {
            "embeddings": np.vstack(embeddings_list),
            "labels": np.vstack(labels_list),
        }
        
        if extract_logits:
            res["logits"] = np.vstack(logits_list)
            
        return res


def preprocess_logits_for_mlm_metrics(logits, labels):
    """
    Compute model predictions and pool last hidden states as sequence embeddings
    """
    # The model returns (logits, hidden_states) where hidden_states may be (pred_num, last_hidden) or (last_hidden,)
    pred_num = None
    if isinstance(logits, tuple):
        real_logits = logits[0]  # [batch, seq, vocab]
        hidden_states = logits[1] # layer tuple
        last_hidden = hidden_states[-1] # [batch, seq, hidden]
        if len(hidden_states) >= 2 and hidden_states[0].ndim == 2:
            pred_num = hidden_states[0]  # [batch, seq] continuous predictions in [-1, 1]
    else:
        real_logits = logits
        last_hidden = None

    # Compute masked token prediction from the discrete vocabulary logits
    pred_ids = torch.argmax(real_logits, dim=-1)

    # For continuous masked positions (encoded as labels in {-1..-7}), convert continuous prediction to -1 - bin_idx
    if pred_num is not None and labels is not None:
        cont_mask = (labels < 0) & (labels != -100)
        if cont_mask.any():
            pred_bins = -1 - torch.clamp(torch.floor((pred_num.float() + 1.0) * 0.5 * 7.0).long(), 0, 6)
            pred_ids = torch.where(cont_mask, pred_bins, pred_ids)
    
    # Pool embeddings by averaging all masked token embeddings
    pooled_embeddings = None
    if last_hidden is not None:
        mask = (labels != -100).unsqueeze(-1).to(last_hidden.dtype)
        sum_embeddings = (last_hidden * mask).sum(dim=1)
        sum_mask = mask.sum(dim=1).clamp(min=1e-9)  # avoid division by zero
        pooled_embeddings = sum_embeddings / sum_mask

    return pred_ids, pooled_embeddings
