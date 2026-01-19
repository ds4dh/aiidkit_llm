import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from pathlib import Path
from scope_module import (
    ScopingReviewData,
    plot_temporal_trend,
    plot_bar_distribution,
    plot_metric_reporting,
    plot_reporting_gaps,
    plot_bubble_chart,
    plot_sankey,
    COLUMN_MAPPING_INVERSED,
    QUALITATIVE_FEATURES,
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
    # INPUT_FILENAME = "data/CHARMS Simplified Checklist - True run - Simplified.xlsx"
    INPUT_FILENAME = "data/CHARMS Simplified Checklist - True run - Corrected - MA - Full.xlsx"
    OUTPUT_DIR = Path("scoping_review_results")
    
    # Setup with subdirectories for organization
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
        if Path(INPUT_FILENAME).exists():
            data = ScopingReviewData(INPUT_FILENAME)
        elif (Path("data") / INPUT_FILENAME).exists():
            data = ScopingReviewData(Path("data") / INPUT_FILENAME)
        else:
            print(f"Error: Could not find '{INPUT_FILENAME}'. Check file name.")
            return
    except Exception as e:
        print(f"Critical Error loading data: {e}")
        return

    # Generate landscape plots
    print("Generating landscape plots...")
    plot_temporal_trend(data, dirs['landscape'] / "publication_trend.png")
    
    for feat in ['transplant_type', 'transplant_infections_predicted', 'data_sites_region']:
        title = COLUMN_MAPPING_INVERSED.get(feat, feat)
        plot_bar_distribution(
            data, feat, f"Distribution of {title}", dirs['landscape'] / f"{feat}.png",
        )

    # Generate methodological plots
    print("Generating methodology plots...")
    plot_bar_distribution(
        data, 'model_best_main', "Primary Machine Learning Algorithms",
        dirs['methods'] / "main_models.png", top_n=10,
    )
    
    for feat in ['data_type', 'data_source', 'data_splitting_strategy', 'model_external_validation']:
        title = COLUMN_MAPPING_INVERSED.get(feat, feat)
        plot_bar_distribution(data, feat, title, dirs['methods'] / f"{feat}.png")

    # Generate results and quality plots
    print("Generating quality and results plots...")
    plot_metric_reporting(data, dirs['results'] / "metric_reporting_frequency.png")
    plot_reporting_gaps(data, dirs['results'] / "reporting_gaps.png")
    
    # Qualitative histograms (inclusions, limitations)
    for feat in QUALITATIVE_FEATURES:
        title = COLUMN_MAPPING_INVERSED.get(feat, feat)
        plot_bar_distribution(
            data, feat, f"Common Themes: {title}",
            dirs['results'] / f"{feat}_themes.png", top_n=10,
        )

    # Generate complex interactions
    print("Generating interaction plots...")
    plot_bubble_chart(data, dirs['interactions'] / "bubble_performance_size.html")
    plot_sankey(data, dirs['interactions'] / "sankey_transplant_data_model.html")
    
    # Goodbye message
    print(f"Done! All figures saved in: {OUTPUT_DIR.resolve()}")

if __name__ == "__main__":
    main()