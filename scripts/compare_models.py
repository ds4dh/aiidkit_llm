import argparse
import json
import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Optional
from pathlib import Path

TASKS_TO_PLOT = [
    "infection_bacteria", 
    "infection_virus", 
    "death",
    "graft_loss",
]
METRICS_OF_INTEREST = {
    "roc_auc": "ROC AUC",
    "pr_auc": "PR AUC",
    # "brier": "Brier Score",
    # "f1_best_f1": "F1 (Best)",
    # "acc_best_f1": "Accuracy",
}
LOWER_IS_BETTER = ["brier"]
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
DEFAULT_BASE_DIR_CLASSIC_ML = "BACKUP - results_classic_ml_tda_all"
# DEFAULT_BASE_DIR_TRANSFORMER = "BACKUP - results_transformer_tda_all/e00-a00-v15"
# DEFAULT_BASE_DIR_TRANSFORMER = "BACKUP - results_transformer_tda_all/e05-a15-v25"
# DEFAULT_BASE_DIR_TRANSFORMER = "BACKUP - results_transformer_tda_all/e15-a25-v45"
# DEFAULT_BASE_DIR_TRANSFORMER = "BACKUP - results_transformer_tda_all/e25-a25-v25"
# DEFAULT_BASE_DIR_TRANSFORMER = "BACKUP - results_transformer_tda_all/e35-a35-v35"
# DEFAULT_BASE_DIR_TRANSFORMER = "BACKUP - results_transformer_tda_all/e00-a00-v15"
# DEFAULT_BASE_DIR_TRANSFORMER = "BACKUP - results_transformer_tda_all/e15-a15-v15"
# DEFAULT_BASE_DIR_TRANSFORMER = "BACKUP - results_transformer_tda_all/no_pretrain"
# DEFAULT_BASE_DIR_TRANSFORMER = "results_optuna/trial_003/e15-a15-v35"
DEFAULT_BASE_DIR_TRANSFORMER = "results_optuna/trial_015/e15-a05-v15"


def parse_args():
    parser = argparse.ArgumentParser(description="Compare transformer vs classic ML baselines.")
    parser.add_argument(
        "--baselines_dir", "-b", type=str, default=DEFAULT_BASE_DIR_CLASSIC_ML,
        help="Path to the classic ML results root (e.g., results_classic_ml)",
    )
    parser.add_argument(
        "--transformer_dir", "-t", type=str, default=DEFAULT_BASE_DIR_TRANSFORMER,
        help="Path to the transformer run directory (e.g., results/e15-a25-v45/finetuning)",
    )
    parser.add_argument(
        "--output_dir", "-o", type=str, default="results_compared",
        help="Directory to save tables and plots.",
    )
    parser.add_argument(
        "--best_baseline", type=str, default="random_forest",  # "random_forest",
        help="Name of the baseline model to use for direct plotting comparisons.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    transformer_path = Path(args.transformer_dir)
    baselines_path = Path(args.baselines_dir)
    
    # Load Data
    print(">>> Loading transformer results...")
    df_trans = pd.DataFrame(load_results_from_dir(transformer_path, model_name_override="Transformer"))
    print(">>> Loading classic ML baseline results...")
    df_base = pd.DataFrame(load_results_from_dir(baselines_path))
    if df_trans.empty or df_base.empty:
        print("Some results were not found. Check your paths!")
        return

    # Combine
    full_df = pd.concat([df_trans, df_base], ignore_index=True)
    full_df = full_df[full_df["Task"].isin(TASKS_TO_PLOT)]
    
    # Generate performance summary and detailed tables
    print("\n>>> Generating comparison tables...")
    generate_overall_summary(full_df, output_path)
    generate_table(full_df, output_path)
    
    # Generate plots
    print("\n>>> Generating advantage scatterplots...")
    plot_advantage_scatter(
        full_df, 
        baseline_name=args.best_baseline, 
        transformer_name="Transformer",
        output_dir=output_path
    )
    
    plot_delta_heatmap(
        full_df, 
        baseline_name=args.best_baseline, 
        transformer_name="Transformer",
        output_dir=output_path
    )
    
    print("\nDone.")


def parse_json_key(key: str) -> Optional[Dict[str, Any]]:
    """
    Parses a results JSON key like: 'test_fup_0030_roc_auc_label_infection_bacteria_0090d'
    Returns a dict with metadata or None if irrelevant.
    """
    # Regex to capture: fup, metric, task, horizon
    # Pattern looks for: test_fup_{INT}_{METRIC_NAME}_label_{TASK}_{INT}d
    
    # Extract follow-up period
    fup_match = re.search(r"test_fup_(\d+)_", key)
    if not fup_match:
        return None
    fup = int(fup_match.group(1))
    
    # Extract horizon
    hrz_match = re.search(r"_(\d+)d$", key)
    if not hrz_match:
        return None
    horizon = int(hrz_match.group(1))
    
    # Extract task and metric (strip prefix/suffix to isolate "metric_label_task")
    core = key[fup_match.end():hrz_match.start()] # e.g., "roc_auc_label_infection_bacteria"
    if "_label_" not in core:
        return None        
    metric_raw, task_raw = core.split("_label_", 1)
    
    # Filter only metrics we care about
    if metric_raw not in METRICS_OF_INTEREST:
        return None
        
    return {
        "fup": fup,
        "horizon": horizon,
        "task": task_raw,
        "metric": metric_raw
    }


def load_results_from_dir(root_path: Path, model_name_override: str = None) -> List[Dict]:
    """
    Recursively finds test_results.json files and extracts metrics.
    """
    data_records = []
    
    # Find all test_results.json files
    json_files = list(root_path.rglob("test_results.json"))
    
    print(f"Scanning {root_path}: Found {len(json_files)} result files.")

    for file_path in json_files:
        try:
            with open(file_path, 'r') as f:
                results = json.load(f)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue

        # Determine model name if not provided
        # For baselines: .../results_classic_ml/xgboost/task/... -> model is parent of task
        # For transformers: provided as "Transformer" usually
        if model_name_override:
            model_name = model_name_override
        else:
            # Heuristic: For baselines, the model name is usually 2 levels up from the file
            # e.g., xgboost/infection_bacteria/hrz.../test_results.json
            parts = file_path.parts
            try:
                # Find index of 'prediction_tasks' keys (tasks) to identify relative position
                # Assuming standard structure: root / model / task / subfolder / file
                model_name = parts[-4] 
            except IndexError:
                model_name = "Unknown"

        for key, value in results.items():
            meta = parse_json_key(key)
            if meta:
                data_records.append({
                    "Model": model_name,
                    "Task": meta["task"],
                    "Horizon": meta["horizon"],
                    "FUP": meta["fup"],
                    "Metric": meta["metric"],
                    "Value": value
                })
                
    return data_records


def generate_table(df: pd.DataFrame, output_dir: Path):
    """
    Generates 4 separate rich comparison tables:
        ROC AUC - Infections
        PR AUC  - Infections
        ROC AUC - Other Tasks (Death, Graft Loss, etc.)
        PR AUC  - Other Tasks (Death, Graft Loss, etc.)
    """
    # Pivot data to wide format
    pivot_df = df.pivot_table(
        index=["Task", "Horizon", "FUP", "Metric"], 
        columns="Model", 
        values="Value"
    )
    
    # Enforce column order: LogReg -> RF -> XGB -> Transformer
    desired_order = ["logistic_regression", "random_forest", "xgboost", "Transformer"]
    existing_cols = [c for c in desired_order if c in pivot_df.columns]
    remaining_cols = [c for c in pivot_df.columns if c not in existing_cols]
    final_cols = existing_cols + remaining_cols
    
    pivot_df = pivot_df[final_cols]

    # Define Filters
    # Helper to check if task is an infection task
    is_infection = lambda x: x.startswith("infection")
    
    # ROC AUC - Infections
    df_roc_inf = pivot_df[
        (pivot_df.index.get_level_values("Metric") == "roc_auc") & 
        (pivot_df.index.get_level_values("Task").map(is_infection))
    ]
    
    # PR AUC - Infections
    df_pr_inf = pivot_df[
        (pivot_df.index.get_level_values("Metric") == "pr_auc") & 
        (pivot_df.index.get_level_values("Task").map(is_infection))
    ]
    
    # ROC AUC - Others
    df_roc_oth = pivot_df[
        (pivot_df.index.get_level_values("Metric") == "roc_auc") & 
        (~pivot_df.index.get_level_values("Task").map(is_infection))
    ]
    
    # PR AUC - Others
    df_pr_oth = pivot_df[
        (pivot_df.index.get_level_values("Metric") == "pr_auc") & 
        (~pivot_df.index.get_level_values("Task").map(is_infection))
    ]

    # Generate and Save Tables
    print(f"\n>>> Saving split tables to {output_dir}...")
    
    _save_formatted_table(df_roc_inf, output_dir / "table_roc_auc_infections.md", "ROC AUC (Infections)")
    _save_formatted_table(df_pr_inf,  output_dir / "table_pr_auc_infections.md",  "PR AUC (Infections)")
    _save_formatted_table(df_roc_oth, output_dir / "table_roc_auc_others.md",     "ROC AUC (Other Tasks)")
    _save_formatted_table(df_pr_oth,  output_dir / "table_pr_auc_others.md",      "PR AUC (Other Tasks)")


def _save_formatted_table(df: pd.DataFrame, filepath: Path, title: str):
    """
    Helper to apply bold formatting to best values and save as Markdown.
    """
    if df.empty:
        print(f"   [Skipping] No data for {title}")
        return

    # Apply bold formatting
    def highlight_best(row):
        metric = row.name[3] # Index level 3 is Metric
        is_lower_better = metric in LOWER_IS_BETTER
        
        valid_vals = row.dropna()
        if len(valid_vals) == 0: return row
        
        target_val = valid_vals.min() if is_lower_better else valid_vals.max()
        
        out = row.copy().astype(object)
        for col in row.index:
            val = row[col]
            if pd.isna(val):
                out[col] = "-"
            else:
                s_val = f"{val:.4f}"
                if abs(val - target_val) < 1e-9:
                    out[col] = f"**{s_val}**"
                else:
                    out[col] = s_val
        return out

    pretty_df = df.apply(highlight_best, axis=1)
    
    # Reset index to make Task/Horizon/FUP real columns
    md_df = pretty_df.reset_index()
    
    # Clean up repetitive values for readability
    md_df['Horizon'] = md_df['Horizon'].astype(str)
    md_df.loc[md_df['Task'].duplicated(), 'Task'] = ''
    
    # Only clear Horizon if Task is also empty (same block)
    mask_task_empty = md_df['Task'] == ''
    md_df.loc[mask_task_empty & md_df['Horizon'].duplicated(), 'Horizon'] = ''

    # Save to file
    with open(filepath, "w") as f:
        f.write(f"# {title}\n\n")
        f.write(md_df.to_markdown(index=False))
        
    print(f"   [Saved] {filepath.name}")
    

def plot_advantage_scatter(
    df: pd.DataFrame, 
    baseline_name: str, 
    transformer_name: str,
    output_dir: Path
):
    """
    Plots the advantage (Delta) of the Transformer over the Baseline.
    Delta = Transformer - Baseline.
    """
    # Filter and Pivot
    subset = df[(df["Model"].isin([baseline_name, transformer_name]))].copy()
    if subset.empty:
        print("No data found for plotting.")
        return

    # Pivot to get side-by-side columns
    pivot = subset.pivot_table(
        index=["Task", "Horizon", "FUP", "Metric"], 
        columns="Model", 
        values="Value"
    ).reset_index()

    if baseline_name not in pivot.columns or transformer_name not in pivot.columns:
        print(f"Skipping plots: Models {baseline_name} or {transformer_name} not found in data.")
        return

    # Calculate Delta
    pivot["Delta"] = pivot[transformer_name] - pivot[baseline_name]
    
    # Prepare plotting attributes
    # Convert Horizon to string for discrete categorization in Legend
    pivot = pivot.sort_values("Horizon") # Ensure numeric sort first
    pivot["Horizon_Str"] = pivot["Horizon"].astype(str) + "d"
    
    unique_horizons = pivot["Horizon_Str"].unique()
    # Ensure palette matches the number of horizons
    current_palette = COLORS[:len(unique_horizons)]

    tasks = pivot["Task"].unique()
    sns.set_theme(style="whitegrid")

    for task in tasks:
        task_data = pivot[pivot["Task"] == task]
        metrics = task_data["Metric"].unique()

        for metric in metrics:
            metric_data = task_data[task_data["Metric"] == metric]
            if metric_data.empty: continue
            
            plt.figure(figsize=(8, 6))
            
            # SCATTERPLOT
            # Z-order ensures dots are on top of lines
            pal = current_palette[:len(metric_data.value_counts("Horizon_Str"))]
            ax = sns.scatterplot(
                data=metric_data, x="FUP", y="Delta", hue="Horizon_Str",
                style="Horizon_Str", palette=pal, s=120, alpha=0.9, zorder=2
            )
            
            # Optional: Faint line connecting the dots to help track the horizon trend
            sns.lineplot(
                data=metric_data, x="FUP", y="Delta", hue="Horizon_Str",
                palette=pal, legend=False, alpha=0.3, linewidth=1.5, ax=ax, zorder=1
            )
            
            # Add Zero Line (Baseline Reference)
            plt.axhline(0, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
            
            # Labels
            readable_metric = METRICS_OF_INTEREST.get(metric, metric)
            plt.title(f"Advantage vs {baseline_name}: {readable_metric}\nTask: {task}", fontsize=14, fontweight='bold')
            plt.xlabel("Follow-up Period (Days)", fontsize=12)
            plt.ylabel(f"Delta ({transformer_name} - {baseline_name})", fontsize=12)
            
            # Move legend outside if crowded, or keep inside best loc
            plt.legend(title="Horizon", bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
            
            # Save
            filename = f"scatter_advantage_{task}_{metric}.png"
            plt.savefig(output_dir / filename, bbox_inches='tight', dpi=150)
            plt.close()
            
    print(f"Scatter plots saved to {output_dir}")


def plot_delta_heatmap(
    df: pd.DataFrame, 
    baseline_name: str, 
    transformer_name: str, 
    output_dir: Path
):
    """
    Creates a single figure with 2x2 subplots showing Delta.
    Uses a shared and symmetric color scale for all subplots.
    """
    # Pre-process all data to find the global min/max for the scale
    df = df[df["Model"].isin([baseline_name, transformer_name])].copy()
    if df.empty:
        print("No data found for heatmaps.")
        return

    # Pivot everything first to calculate all Deltas at once
    full_pivot = df.pivot_table(
        index=["Task", "Horizon", "FUP", "Metric"], 
        columns="Model", 
        values="Value"
    )
    
    if baseline_name not in full_pivot.columns or transformer_name not in full_pivot.columns:
        print("Baseline or Transformer model not found in data.")
        return

    full_pivot["Delta"] = full_pivot[transformer_name] - full_pivot[baseline_name]
    
    # Calculate global symmetric scale limits
    max_delta = full_pivot["Delta"].max()
    min_delta = full_pivot["Delta"].min()
    limit = max(abs(max_delta), abs(min_delta))
    
    # Add a small buffer (e.g., 10%) so the most intense color isn't maxed out
    limit = limit * 1.1 
    vmin, vmax = -limit, limit
    print(f"Heatmap scale set to: {vmin:.3f} to {vmax:.3f}")

    # Setup plot
    cmap_choice = "RdBu"
    is_infection = lambda t: t.startswith("infection")
    
    setup = [
        ("roc_auc", "Infections", 0, 0, is_infection),
        ("roc_auc", "Other Tasks", 0, 1, lambda t: not is_infection(t)),
        ("pr_auc",  "Infections", 1, 0, is_infection),
        ("pr_auc",  "Other Tasks", 1, 1, lambda t: not is_infection(t)),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    plt.subplots_adjust(hspace=0.4, wspace=0.3)
    fig.suptitle(
        f"Performance delta: {transformer_name} - {baseline_name}",
        fontsize=20, fontweight='bold', y=0.95,
    )

    plot_generated = False

    for metric, task_group_name, row, col, task_filter in setup:
        ax = axes[row, col]
        
        # We perform the same pivot logic but on the filtered subset for the specific plot
        subset = df[(df["Metric"] == metric) & (df["Task"].map(task_filter))]
        
        if subset.empty:
            ax.text(0.5, 0.5, "No data", ha='center', va='center', fontsize=14, color='gray')
            ax.set_axis_off()
            continue

        # Pivot for this specific subplot
        pivot = subset.pivot_table(index=["Task", "Horizon", "FUP"], columns="Model", values="Value")
        pivot["Delta"] = pivot[transformer_name] - pivot[baseline_name]
        
        matrix_df = pivot.reset_index()
        matrix_df["Y_Label"] = matrix_df["Task"] + " (" + matrix_df["Horizon"].astype(str) + "d)"
        final_matrix = matrix_df.pivot(index="Y_Label", columns="FUP", values="Delta")
        final_matrix.sort_index(inplace=True)

        # Plot with Shared Scale
        sns.heatmap(
            final_matrix, annot=True, fmt=".3f", cmap=cmap_choice, center=0, 
            vmin=vmin, vmax=vmax, cbar_kws={'label': 'Delta'}, ax=ax
        )
        
        readable_metric = METRICS_OF_INTEREST.get(metric, metric)
        ax.set_title(f"{readable_metric} - {task_group_name}", fontsize=15, fontweight='bold')
        ax.set_xlabel("Follow-up Period (Days)", fontsize=11)
        ax.set_ylabel("Task (Horizon)", fontsize=11)
        
        plot_generated = True

    if plot_generated:
        filename = "heatmap_subplots_delta.png"
        output_file = output_dir / filename
        plt.savefig(output_file, bbox_inches='tight', dpi=150)
        plt.close()
        print(f"Heatmap subplots saved to: {output_file}")


def generate_overall_summary(df: pd.DataFrame, output_dir: Path):
    """
    Calculates aggregated performance (Mean +/- Std) for each model and metric,
    pooling over all Tasks, Horizons, and Follow-up periods.
    """
    print("\n>>> Generating Overall Performance Summary...")
    
    # Group by Metric and Model, calculating Mean and Std
    # We aggregate 'Value' column
    grouped = df.groupby(["Metric", "Model"])["Value"].agg(["mean", "std"])
    
    # Format as string: "0.1234 ± 0.0056"
    def fmt(row):
        return f"{row['mean']:.4f} ± {row['std']:.4f}"
    
    grouped["Formatted"] = grouped.apply(fmt, axis=1)
    
    # Pivot to make it a nice table: Rows=Metrics, Cols=Models
    summary_table = grouped.reset_index().pivot(index="Metric", columns="Model", values="Formatted")
    
    # Remap metric codes to readable names
    summary_table.index = summary_table.index.map(lambda x: METRICS_OF_INTEREST.get(x, x))
    
    # Enforce Column Order: LogReg -> RF -> XGB -> Transformer
    desired_order = ["logistic_regression", "random_forest", "xgboost", "Transformer"]
    existing_cols = [c for c in desired_order if c in summary_table.columns]
    remaining_cols = [c for c in summary_table.columns if c not in existing_cols]
    final_cols = existing_cols + remaining_cols
    
    summary_table = summary_table[final_cols]
    
    # Print to console
    print(summary_table.to_markdown())
    
    # Save to file
    out_file = output_dir / "overall_performance_summary.md"
    with open(out_file, "w") as f:
        f.write("# Overall Performance Summary\n")
        f.write("*Pooled across all Tasks, Horizons, and Follow-up Periods*\n\n")
        f.write(summary_table.to_markdown())
        
    print(f"Summary saved to {out_file}")
    

if __name__ == "__main__":
    main()