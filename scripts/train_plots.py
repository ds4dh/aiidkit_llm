import os
import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

#==========================#
# To report for Mélanie:   #
# FUP 00: HRZ: 30, 90, 365 #
# FUP 30: HRZ: __, 60, 335 #
# FUP 90: HRZ: __, __, 275 #
#==========================#

# Plotting configuration
TRAIN_DATA_FILTERED = False
MAIN_RESULTS_DIR = "results_all_possible_fups_train"
TARGET_METRICS = ["roc_auc", "brier_raw", "brier_cal"]
MLM_RULES_SET_TO_PLOT = [
    # {"entity_id": 0.00, "attribute_id": 0.00, "value_id": 0.15},
    {"entity_id": 0.00, "attribute_id": 0.00, "value_id": 0.25},
#   {"entity_id": 0.00, "attribute_id": 0.00, "value_id": 0.45},
#   {"entity_id": 0.00, "attribute_id": 0.00, "value_id": 0.75},
#   {"entity_id": 0.00, "attribute_id": 0.05, "value_id": 0.15},
#   {"entity_id": 0.00, "attribute_id": 0.15, "value_id": 0.45},
#   {"entity_id": 0.00, "attribute_id": 0.25, "value_id": 0.75},
    {"entity_id": 0.05, "attribute_id": 0.15, "value_id": 0.25},
    {"entity_id": 0.15, "attribute_id": 0.25, "value_id": 0.45},
    # {"entity_id": 0.05, "attribute_id": 0.05, "value_id": 0.05},
    {"entity_id": 0.15, "attribute_id": 0.15, "value_id": 0.15},
    {"entity_id": 0.25, "attribute_id": 0.25, "value_id": 0.25},
    # {"entity_id": 0.35, "attribute_id": 0.35, "value_id": 0.35},
]
TASKS_TO_PLOT = {
    "infection_bacteria": {"fups": [0, 30, 90], "horizons": [30, 60, 90, 275, 335, 365]},
    # "infection_bacteria": {"fups": [0, 30, 60, 90, 180, 365], "horizons": [30, 60, 90]},
    # "infection_virus": {"fups": [0, 30, 60, 90, 180, 365], "horizons": [30, 60, 90]},
    # "infection_fungi": {"fups": [0, 30, 60, 90, 180, 365], "horizons": [30, 60, 90]},
    # "infection": {"fups": [0, 30, 60, 90, 180, 365], "horizons": [30, 60, 90]},
    # "graft_loss": {"fups": [0, 180, 365, 730, 1095], "horizons": [180, 365, 730]},
    # "death": {"fups": [0, 180, 365, 730, 1095], "horizons": [180, 365, 730]},
}

Y_AXIS_DICT = {
    "roc_auc": {"label": "ROC AUC", "lim": (0.5, 1.0)},
    "pr_auc": {"label": "PR AUC", "lim": (0.0, 1.0)},
    "recall": {"label": "Recall", "lim": (0.0, 1.0)},
    "precision": {"label": "Precision", "lim": (0.0, 1.0)},
    "f1": {"label": "F1 score", "lim": (0.0, 1.0)},
    "acc": {"label": "Accuracy", "lim": (0.0, 1.0)},
    "bal_acc": {"label": "Balanced accuracy", "lim": (0.5, 1.0)},
}
COLORS = ["tab:blue", "tab:green", "tab:orange", "tab:red", "tab:purple", "tab:brown"]

parser = argparse.ArgumentParser(description="Plot fine-tuning results.")
parser.add_argument("--test", "-t", action="store_true", help="Plots final test results.")
cli_args = parser.parse_args()


def main():
    for mlm_rules in MLM_RULES_SET_TO_PLOT:
        run_id = "-".join([f"{k[0]}{int(v * 100):02d}" for k, v in mlm_rules.items()])
        base_result_dir = Path(MAIN_RESULTS_DIR) / run_id / "finetuning"
        
        mode_str = "test" if cli_args.test else "validation"
        print(f"Loading {mode_str} results from: {base_result_dir}")

        for task_key, task_specs in TASKS_TO_PLOT.items():
            print(f"Plotting results for task: {task_key}")
            plot_one_task(task_key, task_specs, base_result_dir)


def plot_one_task(task_key: str, task_specs: dict[str, dict], base_dir: Path):
    """Plot finetuning results for a specific prediction task"""
    # Retrieve task specifics
    task_result_dir = base_dir / task_key
    horizons = task_specs["horizons"]
    all_fups = task_specs["fups"]
    
    # Figure and axes setup
    n_horizons = len(horizons)
    num_metrics = len(TARGET_METRICS)
    bar_width = 0.8 / n_horizons
    group_width = n_horizons * bar_width
    fig_width = max(8, (len(all_fups) * group_width * 1.5))
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
            fmt_fn = lambda x: "-".join(f"{i:04d}" for i in sorted(([x] if isinstance(x, int) else x or [])))
            
            # Construct run path
            if not TRAIN_DATA_FILTERED:
                fut_str = "all"
                fuv_str = fmt_fn(all_fups)                
                run_id_str = f"hrz({h:04d})_fut({fut_str})_fuv({fuv_str})"
            else:
                pass 

            # Single run path with all follow-up periods trained simultaneously
            if not TRAIN_DATA_FILTERED:
                
                # Load correct set of results
                run_path = task_result_dir / run_id_str
                if cli_args.test:
                    source_data = load_test_results(run_path)
                    prefix = "test"
                else:
                    source_data, best_step = get_run_history(run_path)
                    prefix = "eval"

                # Extract values in the correct way
                for f in all_fups:
                    metric_key = f"{prefix}_fup_{f:04d}_{metric_suffix}"
                    if cli_args.test:
                        val = source_data.get(metric_key, 0.0)
                    else:
                        val = extract_metric_from_history(source_data, best_step, metric_key)
                    data[h].append(val)

            # Separate runs for each follow-up period
            else:
                for f in all_fups:
                    str_f = fmt_fn(f)
                    run_id_str = f"hrz({h:04d})_fut({str_f})_fuv({str_f})"
                    run_path = task_result_dir / run_id_str
                    
                    # Load correct set of results
                    if cli_args.test:
                        source_data = load_test_results(run_path)
                        prefix = "test"
                    else:
                        source_data, best_step = get_run_history(run_path)
                        prefix = "eval"
                    
                    # Extract values in the correct way
                    metric_key = f"{prefix}_fup_{f:04d}_{metric_suffix}"
                    if cli_args.test:
                        val = source_data.get(metric_key, 0.0)
                    else:
                        val = extract_metric_from_history(source_data, best_step, metric_key)
                        
                    data[h].append(val)

        # Plot bars
        for i, horizon in enumerate(horizons):
            offset = (i - (n_horizons - 1) / 2) * bar_width
            plot_vals = [v if v is not None else 0.0 for v in data[horizon]]
            rects = ax.bar(
                x + offset, plot_vals, bar_width, label=f"{horizon} days", 
                edgecolor='black', color=COLORS[i % len(COLORS)], linewidth=0.5
            )
            ax.bar_label(rects, fmt='%.2f', padding=2, fontsize=9)

        y_label = f"{task_key.replace('_', ' ').capitalize()} - {y_config['label']}"
        ax.set_ylabel(y_label, fontsize=12, labelpad=10)
        ax.set_ylim(*y_config['lim'])
        ax.grid(axis='y', linestyle='--', alpha=0.7, zorder=0)
        ax.set_xticks(x)
        ax.set_xticklabels(all_fups, fontsize=11)
        ax.tick_params(axis='x', labelbottom=True) 
        if idx == 0:
            ax.legend(title="Prediction horizon", title_fontsize=11, fontsize=11, loc='upper left')
        if idx == num_metrics - 1:
            ax.set_xlabel('Follow-up period (days)', fontsize=12, labelpad=10)

    # Save final plot
    os.makedirs(task_result_dir, exist_ok=True)
    mode_str = "test" if cli_args.test else "validation"
    save_path = task_result_dir / f"combined_{mode_str}_results.png"
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved plot to {save_path}")
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
    return 0.0


def load_test_results(run_dir: Path) -> dict:
    """Helper to load the flat JSON test results file"""
    json_path = run_dir / "test_results.json"
    if not json_path.exists():
        return {}  # quiet warning if test results do not exist yet
    try:
        with open(json_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error reading {json_path}: {e}")
        return {}


if __name__ == "__main__":
    main()