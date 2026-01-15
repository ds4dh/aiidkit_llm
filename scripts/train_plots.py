import os
import argparse
import yaml
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Plotting configuration (visualization specifics kept in code)
INVERSE_FUP_PLOT_ORDER = True  # for now, I messed up, will remove later
TRAIN_DATA_FILTERED = False  # matches the default "False" in finetuning loop
TARGET_METRICS = ["eval_roc_auc", "eval_pr_auc"]
Y_AXIS_DICT = {
    "eval_roc_auc": {"label": "ROC AUC", "lim": (0.5, 1.0)},
    "eval_pr_auc": {"label": "PR AUC", "lim": (0.0, 1.0)},
    "eval_recall": {"label": "Recall", "lim": (0.0, 1.0)},
    "eval_precision": {"label": "Precision", "lim": (0.0, 1.0)},
    "eval_f1": {"label": "F1 score", "lim": (0.0, 1.0)},
    "eval_acc": {"label": "Accuracy", "lim": (0.0, 1.0)},
    "eval_bal_acc": {"label": "Balanced accuracy", "lim": (0.5, 1.0)},
}
BAR_WIDTH = 0.25
COLORS = ["tab:blue", "tab:green", "tab:orange"]

# CLI initialization
parser = argparse.ArgumentParser(description="Plot fine-tuning results.")
parser.add_argument("--config", "-c", type=str, default="configs/discriminative_training.yaml")
cli_args = parser.parse_args()
CLI_CFG: dict = {}


def plot_task_performance_results():
    """
    Generate plots for each classification task based on fine-tuning results
    """
    # Reconstruct the results directory path dynamically
    mlm_rules = CLI_CFG["data_collator"]["mlm_masking_rules"]
    run_id = "-".join([f"{k[0]}{int(v * 100):02d}" for k, v in mlm_rules.items()])
    base_result_dir = Path(CLI_CFG["result_dir"]) / run_id / "finetuning"
    print(f"Loading results from: {base_result_dir}")

    # Iterate and plot
    for task_key, task_specs in CLI_CFG["prediction_tasks"].items():
        print(f"Plotting results for task: {task_key}")
        plot_one_task(task_key, task_specs, base_result_dir)


def plot_one_task(task_key: str, task_specs: dict[str, dict], base_dir: Path):
    """
    Plot results for a given task and multiple evaluation metrics vertically
    """
    # Task configuration
    task_result_dir = base_dir / task_key
    horizons = task_specs["horizons"]
    all_fups = task_specs["fups"]
    
    # Figure setup
    num_metrics = len(TARGET_METRICS)
    fig_width = BAR_WIDTH * 2 * (1 + (1 + len(all_fups)) * len(horizons))
    fig_height = 4 * num_metrics
    _, axes = plt.subplots(num_metrics, 1, figsize=(fig_width, fig_height), dpi=300, sharex=True)
    axes = np.atleast_1d(axes)
    x = np.arange(len(all_fups))

    # Iterate over each metric
    for idx, metric_name in enumerate(TARGET_METRICS):
        ax = axes[idx]
        y_config = Y_AXIS_DICT[metric_name]

        # Extract data for this specific metric
        data = {h: [] for h in horizons}
        for h in horizons:
            plotted_fups = all_fups[::-1] if INVERSE_FUP_PLOT_ORDER else all_fups
            for f in plotted_fups:  # PUT BACK TO ALL_FUPS AFTER MESS UP IS CORRECTED!
                train_fups = f if TRAIN_DATA_FILTERED else all_fups
                fmt_fn = lambda x: "-".join(f"{i:04d}" for i in ([x] if isinstance(x, int) else x or []))                
                run_id = f"hrz({fmt_fn(h)})_fut({fmt_fn(train_fups)})_fuv({fmt_fn(f)})"
                run_path = task_result_dir / run_id
                val = get_best_metric_from_run(run_path, metric_name)
                data[h].append(val)

        # Plot bars
        for i, horizon in enumerate(horizons):
            offset = (i - 1) * BAR_WIDTH 
            color = COLORS[i % len(COLORS)]
            rects = ax.bar(
                x + offset, data[horizon], BAR_WIDTH, 
                label=f"{horizon} days", edgecolor='black',
                color=color, linewidth=0.5
            )
            ax.bar_label(rects, fmt='%.2f', padding=2, fontsize=9)

        # Formatting
        y_label = f"{task_key.replace('_', ' ').capitalize()} - {y_config['label']}"
        ax.set_ylabel(y_label, fontsize=12, labelpad=10)
        ax.set_ylim(*y_config['lim'])
        ax.grid(axis='y', linestyle='--', alpha=0.7, zorder=0)
        if idx == 0:  # legend only on top plot
            ax.legend(title="Prediction horizon", title_fontsize=11, fontsize=11, loc='upper left')
        if idx == num_metrics - 1:  # X-axis label only on bottom plot
            ax.set_xlabel('Follow-up period (days)', fontsize=12, labelpad=10)
            ax.set_xticks(x)
            ax.set_xticklabels(all_fups, fontsize=11)

    # Save figure
    os.makedirs(task_result_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(task_result_dir / "combined_results.png")
    plt.close()


def get_best_metric_from_run(run_dir: Path, metric_name: str) -> float:
    """
    Robustly extracts a metric from trainer_state.json.
    
    1. Checks for 'checkpoint-*' subdirs (Training Mode).
    2. If none, looks in run_dir directly (Eval Mode).
    3. If 'best_global_step' is null, gets the latest log entry with the metric.
    """
    if not run_dir.exists():
        print(f"Warning: Directory not found {run_dir}")
        return 0.0

    # Train-and-eval mode (look inside the last checkpoint for the most complete history)
    checkpoints = sorted(run_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split('-')[-1]))
    if checkpoints:
        json_path = checkpoints[-1] / "trainer_state.json"

    # Eval-only mode (look in the root run directory)
    elif (run_dir / "trainer_state.json").exists():
        json_path = run_dir / "trainer_state.json"

    else:
        print(f"Warning: No trainer_state.json found in {run_dir}")
        return 0.0

    # Extract metric data
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)

        best_step = data.get("best_global_step")
        log_history = data.get("log_history", [])

        # If a specific 'best_step' (typical for train-evaluate mode)
        if best_step is not None:
            best_log = next(
                (log for log in log_history if log.get("step") == best_step), 
                None
            )
            if best_log and metric_name in best_log:
                return best_log[metric_name]

        # If 'best_step' is null (eval-only mode) or metric wasn't in best_step
        # We simply look for the *last* log entry that contains our target metric.
        if log_history:
            # Iterate backwards to find the most recent occurrence of the metric
            for log in reversed(log_history):
                if metric_name in log:
                    return log[metric_name]
        
        print(f"Warning: Metric '{metric_name}' not found in {json_path}")
        return 0.0

    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error reading JSON for {run_dir}: {e}")
        return 0.0


if __name__ == "__main__":
    with open(cli_args.config, 'r') as f:
        CLI_CFG = yaml.safe_load(f)
    plot_task_performance_results()