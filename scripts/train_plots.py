import os
import argparse
import yaml
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Plotting configuration
INVERSE_FUP_PLOT_ORDER = False 
TRAIN_DATA_FILTERED = False 
TARGET_METRICS = ["roc_auc", "pr_auc"] # Suffixes only
Y_AXIS_DICT = {
    "roc_auc": {"label": "ROC AUC", "lim": (0.5, 1.0)},
    "pr_auc": {"label": "PR AUC", "lim": (0.0, 1.0)},
    "recall": {"label": "Recall", "lim": (0.0, 1.0)},
    "precision": {"label": "Precision", "lim": (0.0, 1.0)},
    "f1": {"label": "F1 score", "lim": (0.0, 1.0)},
    "acc": {"label": "Accuracy", "lim": (0.0, 1.0)},
    "bal_acc": {"label": "Balanced accuracy", "lim": (0.5, 1.0)},
}
BAR_WIDTH = 0.25
COLORS = ["tab:blue", "tab:green", "tab:orange"]

parser = argparse.ArgumentParser(description="Plot fine-tuning results.")
parser.add_argument("--config", "-c", type=str, default="configs/discriminative_training.yaml")
cli_args = parser.parse_args()
CLI_CFG: dict = {}


def main():
    mlm_rules = CLI_CFG["data_collator"]["mlm_masking_rules"]
    run_id = "-".join([f"{k[0]}{int(v * 100):02d}" for k, v in mlm_rules.items()])
    base_result_dir = Path(CLI_CFG["result_dir"]) / run_id / "finetuning"
    print(f"Loading results from: {base_result_dir}")

    for task_key, task_specs in CLI_CFG["prediction_tasks"].items():
        print(f"Plotting results for task: {task_key}")
        plot_one_task(task_key, task_specs, base_result_dir)


def plot_one_task(task_key: str, task_specs: dict[str, dict], base_dir: Path):
    """Plot finetuning results for a specific prediction task
    """
    # Retrieve task specifics
    task_result_dir = base_dir / task_key
    horizons = task_specs["horizons"]
    all_fups = task_specs["fups"]
    
    # Figure and axes setup
    num_metrics = len(TARGET_METRICS)
    fig_width = BAR_WIDTH * 2 * (1 + (1 + len(all_fups)) * len(horizons))
    fig_height = 4 * num_metrics
    _, axes = plt.subplots(num_metrics, 1, figsize=(fig_width, fig_height), dpi=300, sharex=True)
    axes = np.atleast_1d(axes)
    x = np.arange(len(all_fups))

    # Plot a bar-plot for all metrics
    for idx, metric_suffix in enumerate(TARGET_METRICS):
        ax = axes[idx]
        y_config = Y_AXIS_DICT.get(metric_suffix, {"label": metric_suffix, "lim": (0, 1)})
        data = {h: [] for h in horizons}
        for h in horizons:

            # Format helper: [30, 90] -> "0030-0090"
            fmt_fn = lambda x: "-".join(f"{i:04d}" for i in sorted(([x] if isinstance(x, int) else x or [])))
            
            # Single run with all training samples, and multiple validation
            if not TRAIN_DATA_FILTERED:
                # Path construction: hrz(XXXX)_fut(all)_fuv(0000-0030-0060...)
                fut_str = "all"
                fuv_str = fmt_fn(all_fups)  # matching config list passed to finetuner                
                run_id_str = f"hrz({h:04d})_fut({fut_str})_fuv({fuv_str})"
                run_path = task_result_dir / run_id_str
                
                # Extract metric for each stratified follow-up period
                log_history, best_step = get_run_history(run_path)
                for f in all_fups:
                    metric_key = f"eval_fup_{f:04d}_{metric_suffix}"
                    val = extract_metric_from_history(log_history, best_step, metric_key)
                    data[h].append(val)

            # Separate runs for each follow-up period, with filtered training data
            else:
                for f in all_fups:
                    # Path construction: hrz(XXXX)_fut(0090)_fuv(0090)
                    str_f = fmt_fn(f)
                    run_id_str = f"hrz({h:04d})_fut({str_f})_fuv({str_f})"
                    run_path = task_result_dir / run_id_str
                    
                    # Extract metric for this follow-up period
                    log_history, best_step = get_run_history(run_path)
                    metric_key = f"eval_fup_{f:04d}_{metric_suffix}"
                    val = extract_metric_from_history(log_history, best_step, metric_key)
                    data[h].append(val)

        # Plot bars
        for i, horizon in enumerate(horizons):
            offset = (i - 1) * BAR_WIDTH 
            color = COLORS[i % len(COLORS)]
            plot_vals = [v if v is not None else 0.0 for v in data[horizon]]
            rects = ax.bar(x + offset, plot_vals, BAR_WIDTH, label=f"{horizon} days", edgecolor='black', color=color, linewidth=0.5)
            ax.bar_label(rects, fmt='%.2f', padding=2, fontsize=9)

        y_label = f"{task_key.replace('_', ' ').capitalize()} - {y_config['label']}"
        ax.set_ylabel(y_label, fontsize=12, labelpad=10)
        ax.set_ylim(*y_config['lim'])
        ax.grid(axis='y', linestyle='--', alpha=0.7, zorder=0)
        if idx == 0:
            ax.legend(title="Prediction horizon", title_fontsize=11, fontsize=11, loc='upper left')
        if idx == num_metrics - 1:
            ax.set_xlabel('Follow-up period (days)', fontsize=12, labelpad=10)
            ax.set_xticks(x)
            ax.set_xticklabels(all_fups, fontsize=11)

    os.makedirs(task_result_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(task_result_dir / "combined_results.png")
    plt.close()


def get_run_history(run_dir: Path):
    if not run_dir.exists():
        print(f"Warning: Directory not found {run_dir}")
        return [], None
    checkpoints = sorted(run_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split('-')[-1]))
    if checkpoints:
        json_path = checkpoints[-1] / "trainer_state.json"
    elif (run_dir / "trainer_state.json").exists():
        json_path = run_dir / "trainer_state.json"
    else:
        print(f"Warning: No trainer_state.json found in {run_dir}")
        return [], None

    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        return data.get("log_history", []), data.get("best_global_step")
    except Exception as e:
        print(f"Error reading JSON {json_path}: {e}")
        return [], None


def extract_metric_from_history(log_history, best_step, metric_name):
    if not log_history: return 0.0
    if best_step is not None:
        best_log = next((log for log in log_history if log.get("step") == best_step and metric_name in log), None)
        if best_log: return best_log[metric_name]
    for log in reversed(log_history):
        if metric_name in log: return log[metric_name]
    # Silent fail for cleaner output, or print warning if debugging
    return 0.0


if __name__ == "__main__":
    with open(cli_args.config, 'r') as f:
        CLI_CFG = yaml.safe_load(f)
    main()