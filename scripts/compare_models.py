import argparse
import json
import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Optional
from pathlib import Path

TRANSFORMER_DIRS = {
    "infection_bacteria": "results_optuna_infection_bacteria/trial_032/e05-a35-v35",
    # "infection_virus":    "results_optuna_infection_virus/trial_015/best_run",
    # "death":              "results_optuna_death/trial_005/e20-a10",
    # "graft_loss":         "results_optuna_graft_loss/trial_099/run_v2",
}
TASKS_TO_PLOT = list(TRANSFORMER_DIRS.keys())
METRICS_OF_INTEREST = {
    "roc_auc": "ROC AUC (↑)",
    # "pr_auc": "PR AUC (↑)",
    "ece": "ECE (↓)",
    # "brier": "Brier Score (↓)",
    "nb_t10": "Net benefit - thresh. 50% (↑)",
    "bal_acc_t10": "Balanced acc. - thresh. 50%  (↑)",
}
LOWER_IS_BETTER = ["brier", "ece"]
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
DEFAULT_BASE_DIR_CLASSIC_ML = "results_classic_ml"


def parse_args():
    parser = argparse.ArgumentParser(description="Compare transformer vs classic ML baselines.")
    parser.add_argument(
        "--baselines_dir", "-b", type=str, default=DEFAULT_BASE_DIR_CLASSIC_ML,
        help="Path to the classic ML results root (e.g., results_classic_ml)",
    )
    # Note: Transformer paths are now handled via the TRANSFORMER_DIRS dict at the top of the script
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
    baselines_path = Path(args.baselines_dir)
    
    # Load zransformer data from multiple task-specific folders
    print(">>> Loading transformer results...")
    trans_records = []
    
    for task_name, dir_path in TRANSFORMER_DIRS.items():
        if task_name not in TASKS_TO_PLOT:
            continue
            
        t_path = Path(dir_path)
        if not t_path.exists():
            print(f"   [WARNING] Path for '{task_name}' not found: {t_path}")
            continue
            
        print(f"   Loading '{task_name}' from: {t_path}")
        # Force the model name to "Transformer" so they all group together in plots
        task_records = load_results_from_dir(t_path, model_name_override="Transformer")
        trans_records.extend(task_records)

    df_trans = pd.DataFrame(trans_records)
    
    if not df_trans.empty:
        df_trans = df_trans[df_trans["Task"].isin(TASKS_TO_PLOT)]

    # Load classic ML data from single root folder
    print(f">>> Loading classic ML baseline results from {baselines_path}...")
    df_base = pd.DataFrame(load_results_from_dir(baselines_path))
    
    if df_trans.empty and df_base.empty:
        print("No results found. Check your paths in the script configuration!")
        return

    # --- Combine ---
    full_df = pd.concat([df_trans, df_base], ignore_index=True)
    
    # Global filter for tasks
    full_df = full_df[full_df["Task"].isin(TASKS_TO_PLOT)]
    
    if full_df.empty:
        print("Data loaded but no matching tasks found. Check TASKS_TO_PLOT.")
        return

    # --- Generate performance summary and detailed tables ---
    print("\n>>> Generating comparison tables...")
    generate_overall_summary(full_df, output_path)
    generate_table(full_df, output_path)
    
    # --- Generate plots ---
    print("\n>>> Generating advantage scatterplots...")
    plot_advantage_scatter(
        full_df, 
        baseline_name=args.best_baseline, 
        transformer_name="Transformer",
        output_dir=output_path
    )
    
    print(">>> Generating delta heatmaps...")
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
    
    # Only print scanning message if there are actually files, to reduce clutter in the loop
    if json_files:
        # print(f"Scanning {root_path}: Found {len(json_files)} result files.")
        pass

    for file_path in json_files:
        try:
            with open(file_path, 'r') as f:
                results = json.load(f)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue

        # Determine model name if not provided
        if model_name_override:
            model_name = model_name_override
        else:
            # Heuristic: For baselines, the model name is usually 2 levels up from the file
            # e.g., xgboost/infection_bacteria/hrz.../test_results.json
            parts = file_path.parts
            try:
                # Find relative position. Assuming structure: root / model / task / subfolder / file
                # You might need to adjust this index based on your exact folder depth
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
    Generates 4 separate rich comparison tables.
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
        # print(f"   [Skipping] No data for {title}")
        return

    # Apply bold formatting
    def highlight_best(row):
        metric = row.name[3]  # index level 3 is metric
        valid_vals = row.dropna()
        if len(valid_vals) == 0: return row
        
        if metric in LOWER_IS_BETTER:
            target_val = valid_vals.min()
        else:
            target_val = valid_vals.max()
        
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
            
            # Z-order ensures dots are on top of lines
            plt.figure(figsize=(8, 6))
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
    Creates a figure with subplots showing Delta (Transformer - Baseline).
    """
    # Filter for relevant models
    df = df[df["Model"].isin([baseline_name, transformer_name])].copy()
    if df.empty:
        print("No data found for heatmaps.")
        return

    # Pivot to calculate Deltas
    full_pivot = df.pivot_table(
        index=["Task", "Horizon", "FUP", "Metric"], 
        columns="Model", 
        values="Value"
    )
    
    if baseline_name not in full_pivot.columns or transformer_name not in full_pivot.columns:
        print("Baseline or Transformer model not found in data.")
        return

    full_pivot["Delta"] = full_pivot[transformer_name] - full_pivot[baseline_name]
    
    # Calculate global symmetric scale limits for consistent coloring
    max_delta = full_pivot["Delta"].max()
    min_delta = full_pivot["Delta"].min()
    limit = max(abs(max_delta), abs(min_delta)) if not pd.isna(max_delta) else 1.0
    limit = limit * 1.1 # Add buffer
    vmin, vmax = -limit, limit
    print(f"Heatmap scale set to: {vmin:.3f} to {vmax:.3f}")

    # Define Column Logic
    metrics_list = list(METRICS_OF_INTEREST.keys())
    n_metrics = len(metrics_list)
    is_infection = lambda t: t.startswith("infection")
    col_setup = [
        ("Infections", is_infection),
        ("Other Tasks", lambda t: not is_infection(t))
    ]
    
    # Create Figure: Height scales with number of metrics (approx 6 inches per row)
    fig, axes = plt.subplots(n_metrics, 2, figsize=(20, 6 * n_metrics))
    plt.subplots_adjust(hspace=0.4, wspace=0.3)
    
    fig.suptitle(
        f"Performance delta: {transformer_name} - {baseline_name}",
        fontsize=20, fontweight='bold', y=0.98 if n_metrics > 2 else 0.95,
    )

    # Ensure axes is always a 2D array (n_rows, 2) even if n_metrics is 1
    if n_metrics == 1:
        axes = axes.reshape(1, -1)

    # Iterate over Metrics (Rows)
    plot_generated = False
    for row_idx, metric_key in enumerate(metrics_list):
        metric_name = METRICS_OF_INTEREST[metric_key]
        
        # Iterate over Task Groups (Columns)
        for col_idx, (group_name, task_filter) in enumerate(col_setup):
            ax = axes[row_idx, col_idx]
            
            # Filter data for specific Metric AND specific Task Group
            subset = df[(df["Metric"] == metric_key) & (df["Task"].map(task_filter))]
            if subset.empty:
                ax.text(
                    0.5, 0.5, f"No data for\n{metric_name}\n({group_name})",
                    ha='center', va='center', fontsize=14, color='gray',
                )
                ax.set_axis_off()
                continue

            # Pivot for heatmap matrix
            pivot = subset.pivot_table(index=["Task", "Horizon", "FUP"], columns="Model", values="Value")
            if baseline_name not in pivot.columns or transformer_name not in pivot.columns:
                 ax.text(0.5, 0.5, "Missing Model Data", ha='center', va='center')
                 ax.set_axis_off()
                 continue

            pivot["Delta"] = pivot[transformer_name] - pivot[baseline_name]
            matrix_df = pivot.reset_index()
            
            # Create Y-Axis Label: "Task_Name (90d)"
            matrix_df["Y_Label"] = matrix_df["Task"] + " (" + matrix_df["Horizon"].astype(str) + "d)"
            
            final_matrix = matrix_df.pivot(index="Y_Label", columns="FUP", values="Delta")
            final_matrix.sort_index(inplace=True)

            # Plot heatmap
            sns.heatmap(
                final_matrix, annot=True, fmt=".3f", cmap="RdBu", center=0, 
                vmin=vmin, vmax=vmax, cbar_kws={'label': 'Delta'}, ax=ax
            )
            
            ax.set_title(f"{metric_name} - {group_name}", fontsize=15, fontweight='bold')
            ax.set_xlabel("Follow-up Period (Days)", fontsize=11)
            ax.set_ylabel("Task (Horizon)", fontsize=11)
            
            plot_generated = True

    if plot_generated:
        filename = "heatmap_subplots_delta.png"
        output_file = output_dir / filename
        plt.savefig(output_file, bbox_inches='tight', dpi=150)
        plt.close()
        print(f"Heatmap subplots saved to: {output_file}")
    else:
        print("No plots were generated (empty intersections).")
        plt.close()
        
        
def generate_overall_summary(df: pd.DataFrame, output_dir: Path):
    """
    Calculates aggregated performance (Mean +/- Std) for each model and metric.
    """
    print("\n>>> Generating Overall Performance Summary...")
    
    grouped = df.groupby(["Metric", "Model"])["Value"].agg(["mean", "std"])
    
    def fmt(row):
        return f"{row['mean']:.4f} ± {row['std']:.4f}"
    
    grouped["Formatted"] = grouped.apply(fmt, axis=1)
    
    summary_table = grouped.reset_index().pivot(index="Metric", columns="Model", values="Formatted")
    
    summary_table.index = summary_table.index.map(lambda x: METRICS_OF_INTEREST.get(x, x))
    
    desired_order = ["logistic_regression", "random_forest", "xgboost", "Transformer"]
    existing_cols = [c for c in desired_order if c in summary_table.columns]
    remaining_cols = [c for c in summary_table.columns if c not in existing_cols]
    final_cols = existing_cols + remaining_cols
    
    summary_table = summary_table[final_cols]
    
    print(summary_table.to_markdown())
    
    out_file = output_dir / "overall_performance_summary.md"
    with open(out_file, "w") as f:
        f.write("# Overall Performance Summary\n")
        f.write("*Pooled across all Tasks, Horizons, and Follow-up Periods*\n\n")
        f.write(summary_table.to_markdown())
        
    print(f"Summary saved to {out_file}")
    

if __name__ == "__main__":
    main()