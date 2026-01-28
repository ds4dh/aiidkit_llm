import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from itertools import chain

COLORS = [f"tab:{c}" for c in ["blue", "green", "orange", "red", "purple", "brown"]]
METRICS_TO_PLOT = ["roc_auc", "pr_auc", "brier"]
Y_AXIS_CONFIG = {
    "roc_auc": {"label": "ROC AUC", "lim": (0.5, 1.0)},
    "pr_auc": {"label": "PR AUC", "lim": (0.0, 0.5)},
    "brier": {"label": "Brier score", "lim": (0.0, 0.30)},
    "nb_t10": {"label": "Net benefit", "lim": (-0.25, 0.25)},
    "recall_t10": {"label": "Recall", "lim": (0.0, 1.0)},
    "precision_t10": {"label": "Precision", "lim": (0.0, 1.0)},
    "f1_t10": {"label": "F1 score", "lim": (0.0, 1.0)},
    "acc_t10": {"label": "Accuracy", "lim": (0.0, 1.0)},
    "bal_acc_t10": {"label": "Balanced accuracy", "lim": (0.5, 1.0)},
}


def main():
    """Main execution loop over MLM configs and Tasks."""
    results_dir = "results"
    finetuning_subdir = "finetuning"
    train_data_filtered = False
    mlm_configs = [
        {"entity_id": 0.05, "attribute_id": 0.15, "value_id": 0.25},
        # ...
    ]
    tasks_dict = {
        "infection_bacteria": {"fups": [0, 30, 60, 90, 180, 365], "horizons": [[30, 60, 90]]},
        "graft_loss": {"fups": [0, 180, 365, 730, 1095], "horizons": [[180, 365, 730, 1095]]},
        "death": {"fups": [0, 180, 365, 730, 1095], "horizons": [[180, 365, 730, 1095]]},
        # ...
    }

    # Generate plots for all ways in which pretraining was performed
    for rule in mlm_configs:
        
        # Construct run ID (e.g. "e05-a15-v25")
        run_id = "-".join([f"{k[0]}{int(v * 100):02d}" for k, v in rule.items()])
        finetuning_dir = Path(results_dir) / run_id / finetuning_subdir
        
        print(f"\n> Processing Config: {run_id} | Filtered={train_data_filtered}")
        if not finetuning_dir.exists():
            print(f"  [Skip] Path not found: {finetuning_dir}")
            continue

        for task_key, task_specs in tasks_dict.items():
            print(f"  -- Plotting Task: {task_key}")
            plot_task_results(task_key, task_specs, finetuning_dir, train_data_filtered)


def plot_task_results(
    task_key: str,
    task_specs: dict,
    finetuning_dir: Path,
    train_data_augment: str = "all",
):
    """Generates and saves the bar plot for a single task_key."""
    task_dir = finetuning_dir / task_key
    fups = task_specs["fups"]
    
    # Flatten the plotting structure for the loop, but keep grouping logic for folder finding
    # horizons_struct is the raw list (e.g. [[30, 60]] or [30, 60])
    horizons_struct = task_specs["horizons"]
    
    # Flatten strictly for plotting iteration
    if horizons_struct and isinstance(horizons_struct[0], list):
        flat_horizons = list(chain.from_iterable(horizons_struct))
    else:
        flat_horizons = horizons_struct

    # --- Data collection ---
    # Map: horizon -> [value_at_fup_0, value_at_fup_1, ...]
    plot_data = {metric: {h: [] for h in flat_horizons} for metric in METRICS_TO_PLOT}

    # Iterate over the structure provided in config to respect "Joint" vs "Single" training
    for h_group in horizons_struct:
        
        # Normalize group to list for consistent processing
        # If h_group is int (30), it implies hrz(0030). If list [30, 60], it implies hrz(0030-0060).
        current_horizons = [h_group] if isinstance(h_group, int) else h_group
        
        # Retrieve data for this group of horizons
        for fup in fups:
            # Find the run folder containing this FUP and Horizon Group
            run_path = find_run_folder(task_dir, current_horizons, fup, train_data_augment)
            results = load_results(run_path)

            # Extract metrics for each horizon in this group
            for h in current_horizons:
                for metric in METRICS_TO_PLOT:
                    # Construct strict key: e.g. test_fup_0030_roc_auc_label_infection_bacteria_0060d
                    key = f"test_fup_{fup:04d}_{metric}_label_{task_key}_{h:04d}d"
                    val = results.get(key, 0.0)
                    plot_data[metric][h].append(val)

    # Plotting
    n_h = len(flat_horizons)
    n_m = len(METRICS_TO_PLOT)
    
    # Dynamic layout
    _, axes = plt.subplots(n_m, 1, figsize=(max(8, len(fups)*2), 4*n_m), sharex=True, dpi=300)
    axes = np.atleast_1d(axes)
    x_pos = np.arange(len(fups))
    bar_w = 0.8 / n_h

    for i, metric in enumerate(METRICS_TO_PLOT):
        ax = axes[i]
        cfg = Y_AXIS_CONFIG.get(metric, {"label": metric, "lim": (0, 1)})
        
        for j, h in enumerate(flat_horizons):
            # Calculate offset to center bars
            offset = (j - (n_h - 1) / 2) * bar_w
            values = plot_data[metric][h]
            
            # Draw bars
            rects = ax.bar(
                x_pos + offset, values, bar_w, 
                label=f"{h} days", color=COLORS[j % len(COLORS)], 
                edgecolor='black', linewidth=0.6, alpha=0.85
            )
            
            # Label bars (only valid values)
            labels = [f"{v:.2f}" if v > 0.01 else "" for v in values]
            ax.bar_label(rects, labels=labels, padding=2, fontsize=8)

        # Styling
        ax.set_ylabel(cfg["label"], fontsize=11, fontweight='bold')
        ax.set_ylim(cfg["lim"])
        ax.grid(axis='y', linestyle='--', alpha=0.4)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f"FUP {f}d" for f in fups], fontsize=10)
        if i == 0:  # legend on top only
            ax.legend(
                title="Prediction Horizon", loc='upper center', 
                bbox_to_anchor=(0.5, 1.25), ncol=min(n_h, 6), frameon=False
            )

    # Save
    task_dir.mkdir(parents=True, exist_ok=True)
    fname = task_dir / f"results_tda-{train_data_augment}.png"
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()
    print(f"  -> Saved: {fname}")


def find_run_folder(
    base_dir: Path,
    horizons: list[int],
    fup: int,
    train_data_augment: str,
) -> Path | None:
    """
    Locates the specific run folder based on horizon configuration and FUP strategy.
    """
    if not base_dir.exists(): return None

    # Format the 'hrz' string part (e.g. "0030-0060" or just "0030")
    hrz_str = "-".join(f"{h:04d}" for h in sorted(horizons))
    
    # Format the 'fut/fuv' string part
    # Filtered   -> fut(0030)_fuv(0030)
    # Unfiltered -> fut(all)_fuv(...)  (Note: fuv part is variable, handled by glob)
    if train_data_augment == "none":
        suffix_pattern = f"fut({fup:04d})_fuv({fup:04d})"
    else:
        suffix_pattern = f"fut({train_data_augment})_fuv(*)" 

    # Construct glob pattern
    # Look for: hrz(0030-0060)_fut(all)_fuv(...)
    pattern = f"hrz({hrz_str})_{suffix_pattern}"
    
    # Search
    candidates = list(base_dir.glob(pattern))
    if candidates:
        return candidates[0]  # return first match
    
    return None


def load_results(run_path: Path | None) -> dict:
    """Safely loads test_results.json"""
    if not run_path: return {}
    
    path = run_path / "test_results.json"
    if not path.exists():
        print(f"     [!] Missing results file: {path}")
        return {}
        
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"     [Error] Corrupt JSON {path}: {e}")
        return {}


if __name__ == "__main__":
    main()