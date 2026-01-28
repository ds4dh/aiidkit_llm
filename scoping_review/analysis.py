import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from pathlib import Path
from mappings import COLUMN_MAPPING_INVERSED, QUALITATIVE_FEATURES, ENTRY_MAPPING
from scope_module import (
    ScopingReviewData,
    plot_temporal_trend,
    plot_bar_distribution,
    plot_metric_reporting,
    plot_reporting_gaps,
    plot_bubble_chart,
    plot_sankey,
    plot_continent_countries_stack,
)

# Suppress minor warnings for cleaner output
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("tab10")

def main():
    """
    Main function to perform scoping review analysis and generate plots.
    """
    INPUT_FILENAME = "data/CHARMS Simplified Checklist - True run - Corrected - MA - Full.xlsx"
    OUTPUT_DIR = Path("scoping_review_results")
    
    # Setup directories
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dirs = {
        'landscape': OUTPUT_DIR / '1_landscape',
        'methods': OUTPUT_DIR / '2_methods',
        'results': OUTPUT_DIR / '3_results_quality',
        'interactions': OUTPUT_DIR / '4_interactions'
    }
    for d in dirs.values(): d.mkdir(exist_ok=True)

    # Load data
    try:
        # Check current dir first, then data/ subdir
        fpath = Path(INPUT_FILENAME)
        if not fpath.exists(): fpath = Path("data") / INPUT_FILENAME.name
        
        if fpath.exists():
            data = ScopingReviewData(fpath)
        else:
            print(f"Error: Could not find '{INPUT_FILENAME}'.")
            return
    except Exception as e:
        print(f"Critical Error loading data: {e}")
        return

    # ===============
    # LANDSCAPE PLOTS
    # ===============
    print("Generating landscape plots...")
    
    plot_temporal_trend(
        data, 
        dirs['landscape'] / "publication_trend.png"
    )
    
    plot_continent_countries_stack(
        data, 
        dirs['landscape'] / "sites_by_continent_stacked.png"
    )
    
    plot_bar_distribution(
        data, 'transplant_type', 
        "Distribution of Transplant Type", 
        dirs['landscape'] / "transplant_type.png",
        value_mapper=ENTRY_MAPPING
    )
    
    plot_bar_distribution(
        data, 'transplant_infections_predicted', 
        "Distribution of Specific Infections Predicted", 
        dirs['landscape'] / "transplant_infections_predicted.png",
        value_mapper=ENTRY_MAPPING
    )

    # ====================
    # METHODOLOGICAL PLOTS
    # ====================
    print("Generating methodology plots...")
    
    plot_bar_distribution(
        data, 'model_best_main', 
        "Primary Machine Learning Algorithms",
        dirs['methods'] / "main_models.png", 
        top_n=10, 
        value_mapper=ENTRY_MAPPING
    )
    
    plot_bar_distribution(
        data, 'data_type', 
        "Data Type", 
        dirs['methods'] / "data_type.png",
        value_mapper=ENTRY_MAPPING
    )
    
    plot_bar_distribution(
        data, 'data_source', 
        "Source of Data", 
        dirs['methods'] / "data_source.png",
        value_mapper=ENTRY_MAPPING
    )
    
    plot_bar_distribution(
        data, 'data_splitting_strategy', 
        "Data Splitting Strategy", 
        dirs['methods'] / "data_splitting_strategy.png",
        value_mapper=ENTRY_MAPPING
    )
    
    plot_bar_distribution(
        data, 'model_external_validation', 
        "External Evaluation Strategy", 
        dirs['methods'] / "model_external_validation.png",
        value_mapper=ENTRY_MAPPING
    )

    # =========================
    # RESULTS AND QUALITY PLOTS
    # =========================
    print("Generating quality and results plots...")
    
    plot_metric_reporting(
        data, 
        dirs['results'] / "metric_reporting_frequency.png"
    )
    
    plot_reporting_gaps(
        data, 
        dirs['results'] / "reporting_gaps.png"
    )
    
    # Qualitative Themes (Loop used here as these are identical calls for similar features)
    for feat in QUALITATIVE_FEATURES:
        title = COLUMN_MAPPING_INVERSED.get(feat, feat)
        plot_bar_distribution(
            data, feat, 
            f"Common Themes: {title}",
            dirs['results'] / f"{feat}_themes.png", 
            top_n=10,
            value_mapper=ENTRY_MAPPING
        )

    # ============
    # INTERACTIONS
    # ============
    print("Generating interaction plots...")
    
    plot_bubble_chart(
        data, 
        dirs['interactions'] / "bubble_performance_size.html"
    )
    
    plot_sankey(
        data, 
        dirs['interactions'] / "sankey_transplant_data_model.html"
    )
    
    print(f"Done! All figures saved in: {OUTPUT_DIR.resolve()}")

if __name__ == "__main__":
    main()