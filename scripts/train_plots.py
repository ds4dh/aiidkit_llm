import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Configuration
CLASSIFICATION_TASK = "Bacterial infection prediction"  # to be defined manually (for now!)
TARGET_METRIC = "eval_roc_auc"  # specific key to extract from log_history
CONDITION = "generative_nft"  # discriminative, generative, generative_nft
HORIZONS = [30, 60, 90]
FOLLOW_UPS = [0, 30, 90, 180, 365]
Y_AXIS_DICT = {
    "eval_roc_auc": {"label": "ROC AUC", "lim": (0.5, 0.85)},
    "eval_pr_auc": {"label": "PR AUC", "lim": (0.0, 0.5)},
    "eval_recall": {"label": "Recall", "lim": (0.0, 1.0)},
    "eval_precision": {"label": "Precision", "lim": (0.0, 1.0)},
    "eval_f1": {"label": "F1 score", "lim": (0.0, 1.0)},
    "eval_acc": {"label": "Accuracy", "lim": (0.0, 1.0)},
    "eval_bal_acc": {"label": "Balanced accuracy", "lim": (0.5, 1.0)},
}
Y_AXIS_METRIC = Y_AXIS_DICT[TARGET_METRIC]["label"]
Y_AXIS_LABEL = f"{CLASSIFICATION_TASK} - {Y_AXIS_METRIC}"
Y_LIM = Y_AXIS_DICT[TARGET_METRIC]["lim"]
RESULT_SUBDIR_DICT = {
    "discriminative": "finetuning/pse(False)_pte(False)",
    "generative": "generative/nft(False)",
    "generative_nft": "generative/nft(True)",
}
RESULT_DIR = Path("results") / RESULT_SUBDIR_DICT[CONDITION]
BAR_WIDTH = 0.25
FIG_WIDTH = BAR_WIDTH * 2 * (1 + (1 + len(FOLLOW_UPS)) * len(HORIZONS))
FIG_HEIGHT = 4
COLORS = ["tab:blue", "tab:green", "tab:orange"]


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


def main():
    """
    Generate a plot combining different horizon and follow-up periods.
    for a given modelling framework, and a given evaluation metric
    """
    # Extract data
    data = {h: [] for h in HORIZONS}
    for h in HORIZONS:
        for f in FOLLOW_UPS:
            folder_name = f"hrz({h})_fut()_fuv({f})"
            run_path = RESULT_DIR / folder_name
            metric = get_best_metric_from_run(run_path, TARGET_METRIC)
            data[h].append(metric)

    # Create grouped bar plot
    _, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=300)
    x = np.arange(len(FOLLOW_UPS))
    
    for i, horizon in enumerate(HORIZONS):
        offset = (i - 1) * BAR_WIDTH  # Center the groups
        values = data[horizon]
        
        rects = ax.bar(
            x + offset, values, BAR_WIDTH, 
            label=f"{horizon} days", edgecolor='black',
            color=COLORS[i], linewidth=0.5
        )
        ax.bar_label(rects, fmt='%.2f', padding=2, fontsize=9)

    # Formatting
    ax.set_xlabel('Follow-up period (days)', fontsize=12, labelpad=10)
    ax.set_ylabel(Y_AXIS_LABEL, fontsize=12, labelpad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(FOLLOW_UPS, fontsize=11)
    ax.set_ylim(*Y_LIM) 
    ax.grid(axis='y', linestyle='--', alpha=0.7, zorder=0)
    ax.legend(title="Prediction horizon", title_fontsize=11, fontsize=11, loc='upper left')
    
    plt.tight_layout()
    output_path = RESULT_DIR / "combined_results.png"
    plt.savefig(output_path)
    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()