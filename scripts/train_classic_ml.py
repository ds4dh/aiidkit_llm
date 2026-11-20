import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os
import argparse
import yaml
import json
import numpy as np
import pandas as pd
import optuna
from concurrent.futures import ThreadPoolExecutor
from tqdm.auto import tqdm
from scipy.special import logit
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectPercentile, f_classif, VarianceThreshold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from transformers.trainer_utils import EvalPrediction
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from src.utils import apply_config_overrides
from src.evaluation.plot_results import plot_task_results
from src.evaluation.evaluate_models import (
    DiscriminativeEmbeddingEvaluatorForClassification as CustomEvaluator
)
from scripts.script_utils import save_predictions_table, extract_horizons_from_path, scan_all_fups
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Minimum training positives required to attempt a classic-ML run for a given
# (task, horizon); below this the fitted model would be degenerate (see
# train_model_run) and rare groups are skipped with a warning instead.
MIN_TRAIN_POSITIVES = 5

# Per-process cache keyed on (str(data_root), tuple(fups), filename) so 121 FUP
# parquet tables and feature-type inference are computed ONCE per split rather
# than repeated across all (model, task, horizon) combinations.
_SPLIT_FRAME_CACHE: dict[tuple[str, tuple[int, ...], str], tuple[pd.DataFrame, list[str]]] = {}
_FEATURE_TYPES_CACHE: dict[tuple[str, tuple[str, ...]], tuple[list[int], list[int], list[int]]] = {}


# =================
# RUNNING FUNCTIONS
# =================

def parse_args():
    parser = argparse.ArgumentParser(description="Train baseline ML models.")
    parser.add_argument("--config", "-c", type=str, default="configs/discriminative_classic_ml.yaml")
    parser.add_argument("--fixed_hp", "-f", action="store_true", help="Skip Optuna hyperparameter search.")
    parser.add_argument("--overrides", "-o", action="append", default=[], help="Overrides config (JSON string or key=value).")
    parser.add_argument("--tasks", "-t", nargs="+", default=[], help="Comma- or space-separated prediction_tasks keys to run.")
    parser.add_argument("--models", "-m", nargs="+", default=[], help="Comma- or space-separated model_types to run.")
    return parser.parse_args()


def format_fup_string(fups: list[int] | str) -> str:
    """Matches the folder naming convention of the transformer script."""
    if isinstance(fups, str): return fups
    if not fups: return "none"
    return "-".join(f"{i:04d}" for i in sorted(fups))


def main():
    # Load configuration
    args = parse_args()
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    cfg = apply_config_overrides(cfg, args.overrides)
    if args.tasks:
        wanted = [t.strip() for arg in (args.tasks if isinstance(args.tasks, list) else [args.tasks]) for t in arg.split(",") if t.strip()]
        cfg["prediction_tasks"] = {t: cfg["prediction_tasks"][t] for t in wanted if t in cfg["prediction_tasks"]}
    if args.models:
        wanted_m = [m.strip() for arg in (args.models if isinstance(args.models, list) else [args.models]) for m in arg.split(",") if m.strip()]
        cfg["model_types"] = [m for m in wanted_m if m in cfg["models"]]

    # Run all models
    model_types = cfg['model_types']
    for model_type in model_types:
        
        # Check model
        if model_type not in cfg['models']:
            raise ValueError(f"Model '{model_type}' not found in 'models' config section.")
        model_config = cfg['models'][model_type]

        # Check data
        data_root = Path(cfg['data_dir']) / cfg['data_split_type']
        print(f"Data root for split '{cfg['data_split_type']}': {data_root}")

        # Iterate over tasks
        train_data_augment = cfg.get("train_data_augment", "all")
        for task_key, task_specs in cfg["prediction_tasks"].items():
            
            valid_fups = task_specs["fups"]

            # Determine training configuration (train vs valid follow-up periods)
            if train_data_augment == "none":
                # One run per follow-up period
                run_configs = [([f], [f]) for f in valid_fups]
            else:
                # Single run with aggregated data
                if train_data_augment == "valid":
                    train_fups = valid_fups
                elif train_data_augment == "all":
                    train_fups = scan_all_fups(data_root)
                else:
                    raise ValueError(f"Unknown train_data_augment: {train_data_augment}")
                
                run_configs = [(train_fups, valid_fups)]

            # Train a model for all combinations of horizons and follow-up periods
            for horizon in task_specs["horizons"]:
                for train_fups_list, valid_fups_list in run_configs:
                    
                    # Train model
                    t_str = "All" if train_data_augment == "all" else str(train_fups_list)
                    print(f"\nProcessing task: {task_key} | Model: {model_type}")
                    print(f"Horizon: {horizon}d | Train FUPs: {t_str}")
                    train_model_run(
                        cfg=cfg, data_root=data_root,
                        task_key=task_key,
                        train_fups=train_fups_list,
                        valid_fups=valid_fups_list,
                        test_fups=train_fups_list,  # same fups as for training set!
                        horizon=horizon,
                        train_data_augment=train_data_augment,
                        use_optuna=(not args.fixed_hp),
                        model_type=model_type,
                        model_config=model_config,
                    )

            # Plotting results
            print(f"\nGenerating plots for task: {task_key}...")
            result_dir = Path(cfg['result_dir']) / cfg['data_split_type']
            finetuning_dir = result_dir / model_type
            plot_task_results(
                task_key=task_key,
                task_specs=task_specs,
                finetuning_dir=finetuning_dir,
                train_data_augment=train_data_augment,
            )


def _load_cached_split_frame(data_root: Path, fups: list[int], filename: str, base_ignore_cols: set[str]) -> tuple[pd.DataFrame | None, list[str]]:
    """Loads all FUP parquets once per (data_root, fups, filename) without split-dependent encoding."""
    cache_key = (str(data_root.resolve()), tuple(int(f) for f in fups), filename)
    if cache_key in _SPLIT_FRAME_CACHE:
        return _SPLIT_FRAME_CACHE[cache_key]

    def _read_one(fup):
        fup_dir = data_root / f"fup_{fup:04d}d"
        if not fup_dir.exists():
            fup_dir = data_root / f"fup_{fup:04d}"
        file_path = fup_dir / filename
        if not file_path.exists():
            return None
        df_chunk = pd.read_parquet(file_path)
        if len(df_chunk) > 0:
            df_chunk = df_chunk.copy()
            df_chunk["__fup__"] = int(fup)
        return df_chunk

    workers = min(8, max(1, len(fups)))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        dfs = [d for d in ex.map(_read_one, fups) if d is not None]

    if not dfs:
        return None, []
    full_df = pd.concat(dfs, ignore_index=True)
    feats = [c for c in full_df.columns if c not in base_ignore_cols and c != "__fup__" and not c.startswith("label_")]
    _SPLIT_FRAME_CACHE[cache_key] = (full_df, feats)
    return full_df, feats


def load_combined_data(
    data_root, fups, filename, label_key, ignore_cols, enforced_features=None, return_fups=False,
):
    """
    Loads and concatenates data from multiple FUP directories (served from in-memory split cache).
    """
    base_ignore = {c for c in ignore_cols if not c.startswith("label_")}
    full_df, detected_feats = _load_cached_split_frame(Path(data_root), list(fups), filename, base_ignore)
    if full_df is None or full_df.empty:
        return (None, None, None, None, None) if return_fups else (None, None, None, None)

    if label_key in full_df.columns:
        valid_mask = full_df[label_key].to_numpy() != -100
        sub_df = full_df.loc[valid_mask]
    else:
        sub_df = full_df

    if sub_df.empty:
        return (None, None, None, None, None) if return_fups else (None, None, None, None)

    feats = detected_feats if enforced_features is None else enforced_features
    X_df = sub_df[feats] if enforced_features is None else sub_df.reindex(columns=feats)
    X = X_df.to_numpy()
    y = sub_df[label_key].to_numpy() if label_key in sub_df.columns else None
    patient_ids = sub_df["patientid"].to_numpy() if "patientid" in sub_df.columns else None
    if return_fups:
        fup_arr = sub_df["__fup__"].to_numpy() if "__fup__" in sub_df.columns else None
        return X, y, feats, patient_ids, fup_arr

    return X, y, feats, patient_ids


def train_model_run(
    cfg: dict,
    data_root: Path,
    task_key: str,
    train_fups: list[int],
    valid_fups: list[int],
    test_fups: list[int],
    horizon: int,
    train_data_augment: str,
    use_optuna: bool,
    model_type: str,
    model_config: dict,
):
    """
    Trains one model using aggregated training data, evaluates on stratified test sets.
    """
    label_key = f"label_{task_key}_{horizon:04d}d"
    ignore_cols = set(cfg['ignore_columns'] + [label_key])

    # Load training data
    print(f"Loading training data from {len(train_fups)} folders...")
    X_train, y_train, features, _ = load_combined_data(
        data_root, train_fups, "train.parquet",
        label_key, ignore_cols, enforced_features=None,
    )
    if X_train is None or len(X_train) == 0:
        print("Training set empty. Skipping.")
        return

    # Load validation data
    X_val, y_val, _, _ = load_combined_data(
        data_root, valid_fups, "validation.parquet",
        label_key, ignore_cols, enforced_features=features,
    )
    if X_val is None:
        print("Validation set empty. Skipping.")
        return

    # Rare (task, horizon) combinations (e.g. fungal groups at 30 days) can
    # leave too few events to tune or fit: the pipeline then degenerates, emits
    # nan probabilities and fails EVERY Optuna trial ('The value nan is not
    # acceptable'), crashing the run with 'No trials are completed yet'.
    # Skip such combinations gracefully instead.
    n_pos_train = int(np.sum(y_train == 1))
    n_pos_val = int(np.sum(y_val == 1))
    if n_pos_train < MIN_TRAIN_POSITIVES or n_pos_val < 1:
        print(f"[SKIP] {task_key} @ {horizon}d: too few events to fit a baseline "
              f"({n_pos_train} positives in train, {n_pos_val} in validation).")
        return

    # Setup output directory
    hrz_str = f"{horizon:04d}"
    if train_data_augment == "none":
        fut_str = format_fup_string(train_fups)
    else:
        fut_str = train_data_augment
    fuv_str = format_fup_string(valid_fups)
    task_subdir = f"hrz({hrz_str})_fut({fut_str})_fuv({fuv_str})"
    result_dir = Path(cfg['result_dir']) / cfg['data_split_type']
    output_dir = result_dir / model_type / task_key / task_subdir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Train
    target_us_ratio = None
    inv_tusr = cfg.get('target_undersampling_ratio')
    if inv_tusr is not None: 
        target_us_ratio = 1.0 / inv_tusr

    if use_optuna:
        trainer = OptunaTrainer(
            model_type=model_type,
            optuna_config=model_config.get('optuna_params', {}),
            n_trials=model_config.get('n_optuna_trials', 50),
            output_dir=output_dir,
            target_ratio=target_us_ratio,
        )
        trainer.optimize_and_train(X_train, y_train, X_val, y_val)
    else:
        base_params = {k: v for k, v in model_config.get('optuna_params', {}).items() if not isinstance(v, list)}
        final_params = {**base_params, **model_config.get('fixed_params', {})}
        trainer = BaselineTrainer(
            model_type=model_type,
            model_params=final_params,
            output_dir=output_dir,
            target_ratio=target_us_ratio,
        )
        trainer.train(X_train, y_train)
    
    # Initialize custom clinical evaluation pipelines 
    evaluator = CustomEvaluator(
        do_clustering=False, 
        label_names=[label_key], 
        early_stopping_metric="roc_auc"
    )
    
    # Automatically scan data split root directories for all available follow-up tracking milestones
    all_possible_fups = scan_all_fups(data_root)
    
    # Launch final dynamic head-to-head testing run
    test_classic_ml_model(
        trainer=trainer,
        data_root=data_root,
        all_possible_fups=all_possible_fups,
        label_key=label_key,
        ignore_cols=ignore_cols,
        features=features,
        evaluator=evaluator,
        output_dir=output_dir
    )
    

# ================
# TRAINING CLASSES
# ================

def _evaluate_pipeline_predictions(pipeline, X, y, evaluator, prefix="eval"):
    """Shared probability-to-logit adapter for BaselineTrainer and OptunaTrainer."""
    if pipeline is None:
        raise RuntimeError("Trainer must be trained before evaluation!")
    probs = pipeline.predict_proba(X)[:, 1]
    logits = logit(np.clip(probs, 1e-6, 1.0 - 1e-6))
    if logits.ndim == 1:
        logits = logits[:, None]
    if y.ndim == 1:
        y = y[:, None]
    evaluator.current_prefix = prefix
    metrics = evaluator(EvalPrediction(predictions=logits, label_ids=y))
    return {f"{prefix}_{k}": v for k, v in metrics.items()}


class BaselineTrainer:
    def __init__(self, model_type, model_params, output_dir, target_ratio):
        self.output_dir = Path(output_dir)
        self.model_type = model_type
        self.model_params = model_params
        self.pipeline = None 
        self.target_ratio = target_ratio

    def train(self, X_train, y_train):
        print(f"Training {self.model_type}...")
        self.pipeline = build_model_pipeline(
            X_train, y_train, self.model_type, self.model_params, self.target_ratio,
        )
        self.pipeline.fit(X_train, y_train)

    def evaluate(self, X, y, evaluator, prefix="eval"):
        return _evaluate_pipeline_predictions(self.pipeline, X, y, evaluator, prefix=prefix)


class OptunaTrainer:
    def __init__(self, model_type, optuna_config, n_trials, output_dir, target_ratio):
        self.model_type = model_type
        self.optuna_config = optuna_config
        self.n_trials = n_trials
        self.output_dir = output_dir
        self.pipeline = None
        self.best_params = None
        self.target_ratio = target_ratio

    def optimize_and_train(self, X_train, y_train, X_val, y_val):
        """
        Runs seeded Optuna optimization to find best params (precomputing preprocessing
        and ANOVA feature ranking once), then retrains on full training data.
        """
        base_pipe = build_model_pipeline(X_train, y_train, self.model_type, self._merge_fixed_params({}), self.target_ratio)
        prep_only = Pipeline([base_pipe.steps[0], base_pipe.steps[1]])
        X_train_prep = prep_only.fit_transform(X_train, y_train)
        X_val_prep = prep_only.transform(X_val)

        f_scores, _ = f_classif(X_train_prep, y_train)
        f_scores = np.nan_to_num(f_scores, nan=-np.inf)
        feat_rank_desc = np.argsort(f_scores)[::-1]
        n_prep_feats = X_train_prep.shape[1]

        def objective(trial):
            params = dict(self._suggest_params(trial))
            pct = int(params.pop("feature_fraction", 100))
            k_keep = max(1, int(np.ceil(n_prep_feats * (pct / 100.0))))
            keep_cols = feat_rank_desc[:k_keep]
            X_tr_sel = X_train_prep[:, keep_cols]
            X_va_sel = X_val_prep[:, keep_cols]

            pipeline = build_model_pipeline(
                X_tr_sel, y_train, self.model_type, params, self.target_ratio,
                include_preprocessor=False, include_selector=False,
            )
            try:
                pipeline.fit(X_tr_sel, y_train)
                if hasattr(pipeline, "predict_proba"):
                    probs = pipeline.predict_proba(X_va_sel)[:, 1]
                    if np.isnan(probs).any():
                        return 0.5
                    score = roc_auc_score(y_val, probs)
                    return 0.5 if np.isnan(score) else float(score)
                return 0.5
            except Exception:
                return 0.5

        print(f"Starting Optuna optimization ({self.n_trials} trials)...")
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=1234))
        with tqdm(total=self.n_trials, colour='green', desc="Tuning") as pbar:
            study.optimize(
                objective, 
                n_trials=self.n_trials, 
                callbacks=[lambda study, trial: pbar.update(1)]
            )
        
        self.best_params = study.best_trial.params
        print(f"Best params found: {self.best_params} (AUC: {study.best_value:.4f})")
        
        # Re-merge fixed params that weren't optimized
        final_params = self._merge_fixed_params(self.best_params)

        # Train final model on Train data (or Train+Val if you prefer)
        print("Retraining final model on training set...")
        self.pipeline = build_model_pipeline(
            X_train, y_train, self.model_type, final_params, self.target_ratio,
        )
        self.pipeline.fit(X_train, y_train)

        # Save best params for reference
        serializable_params = {}
        for k, v in final_params.items():
            if isinstance(v, (np.integer, int)): serializable_params[k] = int(v)
            elif isinstance(v, (np.floating, float)): serializable_params[k] = float(v)
            else: serializable_params[k] = v
        with open(self.output_dir / "best_params.json", "w") as f:
            json.dump(serializable_params, f, indent=4)

    def evaluate(self, X, y, evaluator, prefix="eval"):
        return _evaluate_pipeline_predictions(self.pipeline, X, y, evaluator, prefix=prefix)

    def _suggest_params(self, trial):
        """Parses the YAML config to suggest params via Optuna."""
        params = {}
        for key, config_val in self.optuna_config.items():
            # If it's a list like ["int", 10, 100], treat as search space
            if isinstance(config_val, list) and len(config_val) >= 3:
                dtype = config_val[0]
                low = config_val[1]
                high = config_val[2]
                if dtype == "int":
                    params[key] = trial.suggest_int(key, low, high)
                elif dtype == "float":
                    # Check for log scale option if provided (e.g. ["float", 0.01, 1.0, "log"])
                    log = True if (len(config_val) > 3 and config_val[3] == "log") else False
                    params[key] = trial.suggest_float(key, low, high, log=log)
                elif dtype == "categorical":
                    params[key] = trial.suggest_categorical(key, config_val[1:])
            else:
                # Fixed value
                # We don't add it here, we add it in _merge_fixed_params
                pass
        
        # Merge fixed params now to allow pipeline construction inside objective
        return self._merge_fixed_params(params)

    def _merge_fixed_params(self, suggested_params):
        """Combines optimized params with fixed params from config."""
        final = suggested_params.copy()
        for key, val in self.optuna_config.items():
            # If it's not a search definition list but a fixed value
            if not (isinstance(val, list) and len(val) >= 3 and val[0] in ["int", "float", "categorical"]):
                final[key] = val
        return final


# ================
# HELPER FUNCTIONS
# ================

def infer_feature_types(X, cat_threshold=20):
    """
    Heuristic to categorize columns into Binary, Categorical, and Numerical
    based on the training data.
    """
    binary_cols = []
    categorical_cols = []
    numerical_cols = []
    
    # Analyze a sample to speed up (optional, but X is usually small enough here)
    n_samples, n_features = X.shape
    
    for i in range(n_features):
        col = X[:, i]
        # Drop NaNs for unique count
        valid_values = col[~pd.isna(col)]
        unique_vals = np.unique(valid_values)
        n_unique = len(unique_vals)
        
        if n_unique <= 2:
            binary_cols.append(i)
        elif n_unique <= cat_threshold:
            # Heuristic: Small integer range = categorical
            # (Check if values are essentially integers)
            try:
                is_int_like = np.all(np.mod(valid_values.astype(float), 1) == 0)
            except (TypeError, ValueError):
                is_int_like = True
            if is_int_like:
                 categorical_cols.append(i)
            else:
                 numerical_cols.append(i)
        else:
            numerical_cols.append(i)
            
    return binary_cols, categorical_cols, numerical_cols


def build_model_pipeline(
    X_train, y_train, model_type, model_params, target_ratio=None,
    include_preprocessor=True, include_selector=True,
):
    """
    Constructs a ColumnTransformer pipeline fitted strictly on training data.
    """
    if include_preprocessor:
        cache_sig = (id(X_train), X_train.shape)
        if cache_sig not in _FEATURE_TYPES_CACHE:
            _FEATURE_TYPES_CACHE[cache_sig] = infer_feature_types(X_train)
        bin_idx, cat_idx, num_idx = _FEATURE_TYPES_CACHE[cache_sig]
    else:
        bin_idx, cat_idx, num_idx = [], [], []

    # Numerical: median impute -> standard scale (robust to outliers/skew)
    num_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Categorical: most frequent impute -> one-hot (handle new categories by ignoring)
    cat_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(drop='if_binary', handle_unknown='ignore', sparse_output=False))
    ])

    # Binary: most frequent impute -> passthrough (already 0/1)
    bin_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value=0))
    ])

    # Combine into preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, num_idx),
            ('cat', cat_transformer, cat_idx),
            ('bin', bin_transformer, bin_idx)
        ],
        verbose_feature_names_out=False
    )

    # Remove constant features (0 variance)
    dropper = VarianceThreshold(threshold=0)

    # Feature selector (keeps top N-% of features based on ANOVA F-value
    model_params = dict(model_params)
    if model_params.get("n_jobs") == -1:
        model_params["n_jobs"] = (
            len(os.sched_getaffinity(0))
            if hasattr(os, "sched_getaffinity")
            else int(os.environ.get("SLURM_CPUS_PER_TASK", 16))
        )
    percentile = model_params.pop('feature_fraction', 100)
    selector = SelectPercentile(score_func=f_classif, percentile=percentile)

    # Attach model
    if model_type == 'logistic_regression':
        clf = LogisticRegression(**model_params)
    elif model_type == 'random_forest':
        clf = RandomForestClassifier(**model_params)
    elif model_type == 'xgboost':
        clf = XGBClassifier(**model_params)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Return full pipeline
    # return Pipeline([
    #     ('preprocessor', preprocessor),
    #     ('dropper', dropper),
    #     ('selector', selector),
    #     ('clf', clf)
    # ])
    
    # Initialize pipeline step list
    steps = []
    if include_preprocessor:
        steps.extend([
            ('preprocessor', preprocessor),
            ('dropper', dropper),
        ])
    if include_selector:
        steps.append(('selector', selector))
    
    # Add under-sampler if ratio was provided and data needs to be under-sampled
    if target_ratio is not None:
        n_pos, n_neg = np.sum(y_train == 1), np.sum(y_train == 0)
        if n_neg > 0:
            current_ratio = n_pos / n_neg
            if current_ratio < target_ratio:
                # print(f"   [Balancing] Current: {current_ratio:.3f} < Target: {target_ratio}")
                sampler = RandomUnderSampler(sampling_strategy=target_ratio, random_state=42)
                steps.append(('sampler', sampler))
    
    # Add model
    steps.append(('clf', clf))

    return ImbPipeline(steps)


def test_classic_ml_model(
    trainer,  # BaselineTrainer or OptunaTrainer instance
    data_root: Path,
    all_possible_fups: list[int],
    label_key: str,
    ignore_cols: set,
    features: list,
    evaluator: CustomEvaluator,
    output_dir: Path
):
    """
    Symmetrical scoring pipeline for baseline ML models. Sequentially extracts 
    every frame on disk and logs matching validation/test matrix diagnostics.
    """
    from transformers.trainer_utils import EvalPrediction
    from scipy.special import logit
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Define execution matrices for validation and testing splits
    split_configs = [
        {"split_label": "validation", "filename": "validation.parquet", "json_name": "validation_results.json"},
        {"split_label": "test",       "filename": "test.parquet",       "json_name": "test_results.json"}
    ]
    
    for config in split_configs:
        # Check if the target source files actually exist on disk before launching the pass
        file_check_sample = data_root / f"fup_{all_possible_fups[0]:04d}d" / config["filename"]
        if not file_check_sample.exists():
            file_check_sample = data_root / f"fup_{all_possible_fups[0]:04d}" / config["filename"]
        if not file_check_sample.exists():
            print(f"    [Skip] Partition data missing for split: [{config['split_label']}]")
            continue
            
        print(f"\n>>> Running classic ML final evaluation pass over all FUPs for: [{config['split_label'].upper()}]")
        
        final_metrics = {}
        all_predictions = {}
        
        # Compute the global aggregated matrix cut across ALL discovered FUP folders in ONE pass
        X_all, y_all, _, pid_all, fup_all = load_combined_data(
            data_root, all_possible_fups, config["filename"], label_key, ignore_cols,
            enforced_features=features, return_fups=True,
        )
        
        if X_all is not None and len(X_all) > 0:
            prefix = f"{config['split_label']}_all"
            metrics_all = trainer.evaluate(X_all, y_all, evaluator, prefix=prefix)
            final_metrics.update(metrics_all)
            mega_logits = np.asarray(evaluator.last_logits)
            mega_labels = np.asarray(evaluator.saved_labels_and_probs["labels"])
            evaluator.saved_labels_and_probs = None
                
            # Slice per-FUP predictions in memory without re-reading or re-transforming 121 parquets
            for fup in all_possible_fups:
                idx = np.where(fup_all == fup)[0]
                if len(idx) == 0:
                    continue
                prefix = f"{config['split_label']}_fup_{fup:04d}"
                evaluator.current_prefix = prefix
                raw_fup_metrics = evaluator(
                    EvalPrediction(predictions=mega_logits[idx], label_ids=mega_labels[idx])
                )
                final_metrics.update({f"{prefix}_{k}": v for k, v in raw_fup_metrics.items()})
                for key, array in evaluator.saved_labels_and_probs.items():
                    all_predictions[f"{prefix}_{key}"] = array
                if pid_all is not None:
                    all_predictions[f"{prefix}_patientid"] = np.asarray(pid_all[idx])
                    
        # Explicitly flush out evaluator memory storage arrays between passes
        evaluator.saved_labels_and_probs = None
        
        # Write metrics to disk (predictions go to the tidy CSV below)
        with open(output_dir / config["json_name"], "w") as f:
            json.dump(final_metrics, f, indent=4)
        print(f"    [SUCCESS] Saved {config['split_label']} classic ML stats to: {output_dir}")

        # Tidy long-format table: patient_id, fup, horizon, label, prob, prob_cal
        save_predictions_table(
            all_predictions=all_predictions,
            split_label=config["split_label"],
            horizons=extract_horizons_from_path(output_dir),
            output_dir=output_dir,
            csv_name=f"{config['split_label']}_predictions.csv",
        )
            
        print(f"    [SUCCESS] Saved {config['split_label']} classic ML arrays and stats to: {output_dir}")
        
        
if __name__ == "__main__":
    main()
