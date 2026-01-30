import argparse
import yaml
import json
import numpy as np
import pandas as pd
import optuna
from pathlib import Path
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
from src.evaluation.plot_results import plot_task_results
from src.evaluation.evaluate_models import (
    DiscriminativeEmbeddingEvaluatorForClassification as CustomEvaluator
)
optuna.logging.set_verbosity(optuna.logging.WARNING)


# =================
# RUNNING FUNCTIONS
# =================

def parse_args():
    parser = argparse.ArgumentParser(description="Train baseline ML models.")
    parser.add_argument("--config", "-c", type=str, default="configs/discriminative_classic_ml.yaml")
    return parser.parse_args()


def scan_all_fups(data_root: Path) -> list[int]:
    """Find all available follow-up folders (fup_XXXXd) in the data directory."""
    fups = []
    if not data_root.exists(): return []
    for path in data_root.iterdir():
        if path.is_dir() and path.name.startswith("fup_") and path.name.endswith("d"):
            try:
                # Extract integer from "fup_0090d" -> 90
                val = int(path.name.split("_")[-1].replace("d", ""))
                fups.append(val)
            except ValueError:
                continue
    return sorted(fups)


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

    # Check model
    model_type = cfg['selected_model_type']
    if model_type not in cfg['models']:
        raise ValueError(f"Model '{model_type}' not found in 'models' config section.")
    model_config = cfg['models'][model_type]

    # Check data
    data_root = Path(cfg['data_path'])
    print(f"Data Root: {data_root}")

    # Iterate over tasks
    train_data_augment = cfg.get("train_data_augment", "none")
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

        # Iterate over horizons
        for horizon in task_specs["horizons"]:
            
            # Execute runs
            for train_fups_list, valid_fups_list in run_configs:
                
                # Naming for display
                t_str = "All" if train_data_augment == "all" else str(train_fups_list)
                print(f"\nProcessing task: {task_key} | horizon: {horizon}d | train FUPs: {t_str}")

                train_model_run(
                    cfg, data_root,
                    task_key, train_fups_list, valid_fups_list, horizon,
                    model_type, model_config, train_data_augment
                )

        # Plotting results
        print(f"\nGenerating plots for task: {task_key}...")
        finetuning_dir = Path(cfg['results_dir']) / model_type
        plot_task_results(
            task_key=task_key,
            task_specs=task_specs,
            finetuning_dir=finetuning_dir,
            train_data_augment=train_data_augment,
        )


def load_combined_data(
    data_root, fups, filename, label_key, ignore_cols, enforced_features=None,
):
    """
    Loads and concatenates data from multiple FUP directories.
    
    Args:
        enforced_features (list):
        - If provided, the output X will be reindexed to match features exactly
        - If None (training time), features are detected dynamically
    """
    dfs = []    
    for fup in fups:
        fup_dir = data_root / f"fup_{fup:04d}d"
        file_path = fup_dir / filename
        if file_path.exists():
            df_chunk = pd.read_parquet(file_path)
            if label_key in df_chunk.columns:
                df_chunk = df_chunk[df_chunk[label_key] != -100]
            dfs.append(df_chunk)

    if not dfs: return None, None, None
    full_df = pd.concat(dfs, ignore_index=True)

    # Feature selection logic
    if enforced_features is None:
        # Dynamic detection (training phase)
        feats = [c for c in full_df.columns if c not in ignore_cols and not c.startswith("label_")]
        X_df = full_df[feats]
    else:
        # Strict enforcement (validation/test phase)
        feats = enforced_features
        # reindex ensures X has exactly 'feats' columns in the right order.
        # Missing columns in 'full_df' become NaNs (which SimpleImputer will handle).
        X_df = full_df.reindex(columns=feats)

    X = X_df.values
    y = full_df[label_key].values if label_key in full_df.columns else None
    
    return X, y, feats


def train_model_run(
    cfg, data_root, task_key, train_fups, valid_fups, horizon,
    model_type, model_config, train_data_augment,
):
    """
    Trains one model using aggregated training data, evaluates on stratified test sets.
    """
    label_key = f"label_{task_key}_{horizon:04d}d"
    ignore_cols = set(cfg['ignore_columns'] + [label_key])

    # Load training data
    print(f"Loading training data from {len(train_fups)} folders...")
    X_train, y_train, features = load_combined_data(
        data_root, train_fups, "train.parquet",
        label_key, ignore_cols, enforced_features=None,
    )
    if X_train is None or len(X_train) == 0:
        print("Training set empty. Skipping.")
        return

    # Load validation data
    X_val, y_val, _ = load_combined_data(
        data_root, valid_fups, "validation.parquet",
        label_key, ignore_cols, enforced_features=features,
    )
    if X_val is None:
        print("Validation set empty. Skipping.")
        return

    # Setup output directory    
    # Logic: if "none", we list the specific FUP (e.g., 0090). If "valid"/"all", we use the string "valid"/"all" 
    hrz_str = f"{horizon:04d}"
    if train_data_augment == "none":
        fut_str = format_fup_string(train_fups) # e.g. "0090"
    else:
        fut_str = train_data_augment # e.g. "valid" or "all"
    fuv_str = format_fup_string(valid_fups)
    task_subdir = f"hrz({hrz_str})_fut({fut_str})_fuv({fuv_str})"
    output_dir = Path(cfg['results_dir']) / model_type / task_key / task_subdir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Train
    trainer = OptunaTrainer(
        model_type=model_type,
        optuna_config=model_config.get('optuna_params', {}),
        n_trials=model_config.get('n_optuna_trials', 10),
        output_dir=output_dir,
        target_ratio=cfg.get('target_undersampling_ratio', None),
    )
    trainer.optimize_and_train(X_train, y_train, X_val, y_val)

    # Evaluate
    evaluator = CustomEvaluator(
        do_clustering=False, 
        label_names=[label_key], 
        early_stopping_metric="roc_auc"
    )

    # Calibration (on aggregated validation set)
    print("Fitting calibration...")
    evaluator.current_prefix = "val_all"
    _ = trainer.evaluate(X_val, y_val, evaluator, prefix="val_all")
    
    # Test all follow-up periods
    final_metrics = {}
    X_test_all, y_test_all, _ = load_combined_data(
        data_root, valid_fups, "test.parquet",
        label_key, ignore_cols, enforced_features=features,
    )
    if X_test_all is not None:
        metrics_all = trainer.evaluate(
            X_test_all, y_test_all, evaluator, prefix="test_all",
        )
        final_metrics.update(metrics_all)

    # Test per follow-up period (stratified)
    for fup in valid_fups:
        X_test_fup, y_test_fup, _ = load_combined_data(
            data_root, [fup], "test.parquet",
            label_key, ignore_cols, enforced_features=features,
        )
        if X_test_fup is not None and len(X_test_fup) > 0:
            metrics_fup = trainer.evaluate(
                X_test_fup, y_test_fup, evaluator, prefix=f"test_fup_{fup:04d}"
            )
            final_metrics.update(metrics_fup)

    # Save results
    json_path = output_dir / "test_results.json"
    with open(json_path, "w") as f:
        json.dump(final_metrics, f, indent=4)
    print(f"Results saved to {json_path}")


# ================
# TRAINING CLASSES
# ================

class BaselineTrainer:
    def __init__(self, model_type, model_params, output_dir, target_ratio):
        self.output_dir = Path(output_dir)
        self.model_type = model_type
        self.model_params = model_params
        self.pipeline = None 
        self.target_ratio = target_ratio

    def train(self, X_train, y_train):
        print(f"Training {self.model_type}...")
        
        # Build a data transform + model pipeline based on X_train statistics
        self.pipeline = build_model_pipeline(
            X_train, y_train, self.model_type, self.model_params, self.target_ratio,
        )
        
        # Fit model with nicer data
        self.pipeline.fit(X_train, y_train)

    def evaluate(self, X, y, evaluator, prefix="eval"):
        if self.pipeline is None:
            raise RuntimeError("Trainer must be trained before evaluation!")

        if hasattr(self.pipeline._final_estimator, "predict_proba"):
            probs = self.pipeline.predict_proba(X)[:, 1]
        else:
            raise AttributeError("Model must support predict_proba")

        # Invert Sigmoid: Logit(p) = log(p/(1-p))
        probs_clipped = np.clip(probs, 1e-6, 1 - 1e-6)
        logits = logit(probs_clipped)

        if logits.ndim == 1: logits = logits[:, None]
        if y.ndim == 1: y = y[:, None]
        
        eval_preds = EvalPrediction(predictions=logits, label_ids=y)
        evaluator.current_prefix = prefix
        
        metrics = evaluator(eval_preds)
        prefixed_metrics = {f"{prefix}_{k}": v for k, v in metrics.items()}
        
        return prefixed_metrics


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
        Runs Optuna optimization to find best params, then retrains on full data.
        """
        # Define objective function
        def objective(trial):
            # Sample parameters
            params = self._suggest_params(trial)
            
            # Build temporary pipeline
            pipeline = build_model_pipeline(
                X_train, y_train, self.model_type, params, self.target_ratio,
            )
            
            # Fit on Train, Score on Val (Simple Hold-out optimization)
            # For more robustness, replace this with cross_val_score on X_train
            pipeline.fit(X_train, y_train)
            if hasattr(pipeline, "predict_proba"):
                probs = pipeline.predict_proba(X_val)[:, 1]
                # Using ROC-AUC as objective
                try:
                    score = roc_auc_score(y_val, probs)
                except ValueError:
                    score = 0.5
            else:
                score = 0.5
            return score

        # Run hyper-parameter optimization
        print(f"Starting Optuna optimization ({self.n_trials} trials)...")
        study = optuna.create_study(direction="maximize")
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
        with open(self.output_dir / "best_params.json", "w") as f:
            json.dump(final_params, f, indent=4)

    def evaluate(self, X, y, evaluator, prefix="eval"):
        if self.pipeline is None: raise RuntimeError("Not trained")
        
        probs = self.pipeline.predict_proba(X)[:, 1]
        probs_clipped = np.clip(probs, 1e-6, 1 - 1e-6)
        logits = logit(probs_clipped)
        
        if logits.ndim == 1: logits = logits[:, None]
        if y.ndim == 1: y = y[:, None]
        
        eval_preds = EvalPrediction(predictions=logits, label_ids=y)
        evaluator.current_prefix = prefix
        
        metrics = evaluator(eval_preds)
        return {f"{prefix}_{k}": v for k, v in metrics.items()}

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
            if np.all(np.mod(valid_values, 1) == 0):
                 categorical_cols.append(i)
            else:
                 numerical_cols.append(i)
        else:
            numerical_cols.append(i)
            
    return binary_cols, categorical_cols, numerical_cols


def build_model_pipeline(X_train, y_train, model_type, model_params, target_ratio=None):
    """
    Constructs a ColumnTransformer pipeline that treats features differently.
    """
    # Infer types
    bin_idx, cat_idx, num_idx = infer_feature_types(X_train)
    # print(f"   [Auto-Preprocessing] Detected: {len(bin_idx)} Binary, {len(cat_idx)} Categorical, {len(num_idx)} Numerical features.")

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
        ('imputer', SimpleImputer(strategy='most_frequent'))
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
    steps = [
        ('preprocessor', preprocessor),
        ('dropper', dropper),
        ('selector', selector),
    ]
    
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


if __name__ == "__main__":
    main()
