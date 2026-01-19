import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from collections import Counter

# ============================================================================
# FEATURE MAPPING
# ============================================================================

COLUMN_MAPPING_INVERSED = {
    # Publications
    'publication_authors': 'Authors',
    'publication_year': 'Publication year',
    'publication_title': 'Publication title',
    'publication_pmid': 'Publication PMID',
    'publication_doi': 'Publication DOI',
    'publication_journal': 'Publication journal',
    
    # Data
    'data_source': 'Source of data (e.g., EHR, registry, clinical trial, bespoke cohort)',
    'data_type': 'Data type (e.g., structured, unstructured text, imaging, time-series, genomics)',
    'data_available': 'Is data publicly available? (Yes, no, on request)',
    'data_sites_region': 'Study sites (region)',
    'data_sites_number': 'Study sites (number of centers)',
    'data_inclusion_criteria': 'Inclusion criteria',
    'data_exclusion_criteria': 'Exclusion criteria',
    'data_ethics_approval': 'Ethical approval',
    'data_consent_obtaiend': 'Consent',
    'data_study_settings': 'Study settings (e.g., inpatient, outpatient, ICU)',
    'data_recruitment_method': 'Recruitment method',
    'data_recruitment_dates': 'Recruitment dates',
    'data_collection_dates': 'Data collection dates',
    'data_participant_characteristics': 'Key participant characteristics',
    'data_n_participants': 'Number of participants (total in study)',
    'data_n_events': 'Number of outcomes/events (total events in study)',
    'data_epv': 'Number of outcomes/events wrt the number of candidate predictors (EPV)',
    'data_n_missing': 'Number of participants with any missing value',
    'data_missing_handling': 'Handling of missing data (e.g., complete-case analysis)',
    'data_splitting_strategy': 'Data splitting strategy',
    'data_splitting_ratios': 'Data splitting ratios',
    
    # Outcome
    'outcome_predicted': 'Outcome(s) predicted',
    'outcome_definition': 'Definition of each possible outcome',
    'outcome_assessment': 'How was outcome assessed (e.g., clinical diagnosis, microbiology, ICD codes)',
    'outcome_prevalence': 'Prevalence of each outcome',
    'outcome_def_same': 'Was the outcome definition the same for all participants? (yes, no)',
    'outcome_type': 'Type of outcome (e.g., single, combined, survival, binary, multi-class)',
    'outcome_blinded': 'Was the outcome assessed without knowledge of the candidate predictors? (yes, no)',
    'outcome_timing': 'Timing of outcome measurement/occurence (e.g., follow-up duration, time to event)',
    
    # Predictors
    'predictors_number': 'Number of (candidate) predictors assessed',
    'predictors_type': 'Type of predictors',
    'predictors_timing': 'Timing of predictor measurement',
    'predictors_similar': 'Were predictors and measurements similar for all participants? (yes, no)',
    'predictors_blinded': 'Were predictors assessed blinded for outcome? (yes, no)',
    'predictors_continuous_handling': 'Handling of continuous predictors in the modelling',
    'predictors_candidate_selection': 'Method for selection of candidate predictors',
    'predictors_selection': 'Method for selection of predictors during multivariable modelling',
    'predictors_imbalance_handling': 'Handling of class imbalance (e.g., oversampling)',
    'predictors_in_outcome': 'Were candidate predictors part of the outcome? (yes, no)',
    'predictors_n_final': 'Number of predictors/features used in the final model',
    
    # Model
    'models_used': 'Model name',
    'model_type': 'Model type',
    'model_software': 'Software/programming language/libraries used',
    'model_hyperparameter_tuning': 'Hyperparameter tuning method',
    'model_code_available': 'Is model code available? (yes, no, on request)',
    'model_internal_validation': 'Internal validation strategy',
    'model_external_validation': 'External evaluation strategy',
    'model_best_main': 'Main or best model',
    'model_architecture_provided': 'Final model architecture/equation provided',
    'model_n_parameters': 'Number of parameters in the final model',
    'model_baseline_used': 'Was a baseline model/method used',
    'model_interpretability': 'Feature importance/interpretability method',
    'model_top_features': 'If above is not NI, top-N most important features',
    
    # Metrics
    'metrics_c_statistic': 'C-Statistic',
    'metrics_calibration': 'Calibration metrics',
    'metrics_clinical_utility': 'Clinical utility metrics',
    'metrics_auroc': 'AUROC graph / value',
    'metrics_auprc': 'AUPRC (PR) graph / value',
    'metrics_log_rank': 'Log-rank test',
    'metrics_risk_curves': 'Risk group curves',
    'metrics_accuracy': 'Accuracy',
    'metrics_balanced_accuracy': 'Balanced accuracy',
    'metrics_specificity': 'Specificity',
    'metrics_recall': 'Recall (sensitivity)',
    'metrics_precision': 'Precision (PPV)',
    'metrics_npv': 'Negative predicted value - NPV',
    'metrics_f1_score': 'F1-Score',
    'metrics_other': 'Other',
    
    # Study results
    'study_key_findings': 'Key findings of the study',
    'study_evidence_strength': 'Strength of evidence for drivers',
    'study_strengths': 'Study strengths',
    'study_limitations': 'Study limitations',
    'study_future_research': 'Future research directions',
    'study_clinical_implications': 'Clinical implications and utility',
    'study_implementation_barriers': 'Potential barriers to implementation',
    
    # Transplant-specific
    'transplant_type': 'Type of transplant',
    'transplant_post_time': 'Time post-transplant covered by prediction',
    'transplant_infections_predicted': 'Specific infections predicted',
    'transplant_immunosuppression': 'Immunosuppression regimen details',
    'transplant_biomarkers': 'Any specific biomarkers or transplant-related predictors used',
    'transplant_opportunistic_infections': 'Consideration of opportunistic infections (yes/no)',
    'transplant_risk_stratification': 'Patient stratification by risk (if applicable)',
}

# Create a normalization lookup dict ("publicationyear" -> "publication_year")
NORMALIZED_LOOKUP = {}
for internal_key, nice_label in COLUMN_MAPPING_INVERSED.items():
    # Map the "nice label" (lowercased, no spaces) to internal key
    clean_nice = re.sub(r'[^a-zA-Z0-9]', '', nice_label.lower())
    NORMALIZED_LOOKUP[clean_nice] = internal_key

    # Also map the internal key itself (just in case)
    clean_internal = re.sub(r'[^a-zA-Z0-9]', '', internal_key.lower())
    NORMALIZED_LOOKUP[clean_internal] = internal_key

# Groups for looping
CATEGORICAL_FEATURES = [
    'transplant_type', 'transplant_infections_predicted', 'transplant_post_time',
    'data_source', 'data_sites_region', 'data_type', 
    'model_software', 'model_external_validation', 'model_interpretability',
    'model_best_main', 'data_splitting_strategy'
]

QUALITATIVE_FEATURES = [
    'data_inclusion_criteria', 'data_exclusion_criteria', 
    'study_limitations', 'study_key_findings'
]

# ============================================================================
# DATA PROCESSING CLASS
# ============================================================================

class ScopingReviewData:
    def __init__(self, filepath):
        self.filepath = Path(filepath)
        self.df = self._load_and_process()
        self.n_papers = len(self.df)
        print(f"Data loaded successfully. Total Papers: {self.n_papers}")
        print(f"Columns identified: {list(self.df.columns[:5])}...")

    def _load_and_process(self):
        # Load Data
        try:
            # Try loading as CSV
            if self.filepath.suffix == '.xlsx':
                 raw = pd.read_excel(self.filepath, header=None)
            else:
                 # Try typical encodings
                 try:
                     raw = pd.read_csv(self.filepath, header=None, encoding='utf-8-sig')
                 except:
                     raw = pd.read_csv(self.filepath, header=None, encoding='latin1')
        except Exception as e:
            raise ValueError(f"Could not read file: {e}")

        # Filter rows based on instructions in column 0
        # 1. Identify Paper IDs from header row (Row 0, Col 2 onwards)
        # 2. Remove rows where Col 0 contains "Ignore feature"
        headers = raw.iloc[0, :].values.astype(str)
        paper_ids = [h for h in headers[2:] if str(h).lower() != 'nan']
        valid_rows = []
        for _, row in raw.iloc[1:, :].iterrows():
            instruction = str(row[0]).lower()
            if "ignore feature" in instruction: continue
            valid_rows.append(row)
            
        clean_df = pd.DataFrame(valid_rows)
        
        # Transpose (rows become Papers, columns become Features)
        features_raw = clean_df.iloc[:, 0].values.astype(str)
        data_values = clean_df.iloc[:, 1:].values
        if data_values.shape[1] > len(paper_ids):
            data_values = data_values[:, :len(paper_ids)]         
        df_transposed = pd.DataFrame(data_values.T, columns=features_raw)
        df_transposed['PAPER_ID'] = paper_ids
        
        # Map column names
        new_columns = {}
        for col in df_transposed.columns:
            col_clean = re.sub(r'[^a-zA-Z0-9]', '', str(col).lower())
            if col_clean in NORMALIZED_LOOKUP:
                mapped_name = NORMALIZED_LOOKUP[col_clean]
                new_columns[col] = mapped_name
            else:
                new_columns[col] = col.strip()

        df_transposed.rename(columns=new_columns, inplace=True)
        
        return df_transposed


    @staticmethod
    def parse_list(cell_value):
        """Splits comma/semicolon separated strings into a clean list."""
        if pd.isna(cell_value): return []
        s = str(cell_value).strip()
        if s.lower() in ['nan', 'ni', 'not specified', 'none', '', 'n/a']: return []
        
        # Split by comma or semicolon
        items = re.split(r'[,;]', s)
        # Clean up parentheses or extra whitespace
        clean_items = []
        for i in items:
            i = i.strip()
            if i: clean_items.append(i)
        return clean_items

    @staticmethod
    def parse_numeric(cell_value):
        if pd.isna(cell_value): return None
        s = str(cell_value)
        try:
            return float(s)
        except:
            # Try to extract the first number found
            match = re.search(r"(\d+(\.\d+)?)", s)
            if match:
                return float(match.group(1))
        return None

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_bar_distribution(data_obj, feature, title, output_path, top_n=15):
    if feature not in data_obj.df.columns:
        print(f"Feature '{feature}' not found in data. Skipping plot.")
        return

    all_items = []
    for val in data_obj.df[feature]:
        items = data_obj.parse_list(val)
        items = [i.title() for i in items]
        all_items.extend(items)

    if not all_items: return

    counts = Counter(all_items).most_common(top_n)
    if not counts: return
    
    df_counts = pd.DataFrame(counts, columns=['Category', 'Count'])

    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=df_counts, x='Count', y='Category', palette='viridis')
    ax.bar_label(ax.containers[0])
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Number of Papers')
    plt.ylabel('')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_metric_reporting(data_obj, output_path):
    """
    Plots a bar chart of how many studies reported specific performance metrics.
    """
    # Find columns starting with metrics_
    metric_cols = [c for c in data_obj.df.columns if c.startswith('metrics_')]
    if not metric_cols: return
    
    # Define terms that count as "Not Reported"
    NEGATIVE_TERMS = {
        'no', 'none', 'ni', 'not specified', 'nan', 'n/a', 
        'unknown', 'not considered', 'not reported', 'not applicable'
    }

    results = {}
    for col in metric_cols:
        valid_count = 0
        for val in data_obj.df[col]:
            items = data_obj.parse_list(val)
            is_reported = False
            for item in items:
                clean_item = str(item).strip().lower()
                if clean_item and clean_item not in NEGATIVE_TERMS:
                    is_reported = True
                    break
            
            if is_reported:
                valid_count += 1
            
        label = COLUMN_MAPPING_INVERSED.get(col, col.replace('metrics_', '').title())
        results[label] = valid_count

    # Filter out metrics with 0 reported studies to keep plot clean
    results = {k: v for k, v in results.items() if v > 0}

    if not results: return
    sorted_res = dict(sorted(results.items(), key=lambda x: x[1], reverse=True))
    
    plt.figure(figsize=(12, 7))
    bars = plt.bar(sorted_res.keys(), sorted_res.values(), color='#4c72b0', edgecolor='black')
    plt.bar_label(bars)
    plt.xticks(rotation=45, ha='right')
    plt.title("Frequency of Reported Performance Metrics", fontsize=14)
    plt.ylabel("Number of Studies")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_reporting_gaps(data_obj, output_path):
    gaps = {}
    target_cols = ['model_code_available', 'data_available', 'model_external_validation', 'data_ethics_approval']
    
    for col in target_cols:
        if col not in data_obj.df.columns: continue
        label = COLUMN_MAPPING_INVERSED.get(col, col)
        neg_count = 0
        for val in data_obj.df[col]:
            s = str(val).lower()
            if any(x in s for x in ['no', 'none', 'ni', 'not specified', 'nan', 'request']):
                neg_count += 1
        gaps[label] = neg_count

    if not gaps: return

    plt.figure(figsize=(8, 5))
    bars = plt.bar(gaps.keys(), gaps.values(), color='#c44e52', alpha=0.8)
    plt.bar_label(bars)
    plt.title(f"Reporting Gaps (Negative or Not Specified) / N={data_obj.n_papers}")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_bubble_chart(data_obj, output_path):
    # Require N and AUROC
    if 'data_n_participants' not in data_obj.df.columns or 'metrics_auroc' not in data_obj.df.columns:
        print("Missing N or AUROC columns for bubble chart.")
        return

    rows = []
    for _, row in data_obj.df.iterrows():
        n = data_obj.parse_numeric(row.get('data_n_participants'))
        auroc = data_obj.parse_numeric(row.get('metrics_auroc'))
        
        # If AUROC > 1 (e.g., 85%), convert to 0.85
        if auroc and auroc > 1.0: auroc = auroc / 100.0
        
        if n and auroc:
            # Events
            events = data_obj.parse_numeric(row.get('data_n_events'))
            if not events: events = 20 # Default size
            
            # Transplant type
            t_type = "Unknown"
            if 'transplant_type' in row:
                t_list = data_obj.parse_list(row['transplant_type'])
                if t_list: t_type = t_list[0].title()

            rows.append({
                'N': n, 'AUROC': auroc, 'Events': events, 
                'Transplant': t_type, 'Paper': str(row.get('PAPER_ID', ''))
            })

    if not rows: return
    df_plot = pd.DataFrame(rows)

    try:
        fig = px.scatter(
            df_plot, x="N", y="AUROC", size="Events", color="Transplant",
            hover_name="Paper", log_x=True,
            title="Model Performance vs. Sample Size",
            height=600
        )
        fig.write_html(output_path)
        png_path = str(output_path).replace('.html', '.png')
        fig.write_image(png_path, scale=2)
        
    except Exception as e:
        print(f"Plotly save error (ensure 'kaleido' is installed): {e}")


def plot_sankey(data_obj, output_path):
    # Check columns
    req_cols = ['transplant_type', 'data_type', 'model_best_main']
    if not all(c in data_obj.df.columns for c in req_cols):
        print("Missing columns for Sankey diagram.")
        return

    paths = []
    for _, row in data_obj.df.iterrows():
        path = []
        for c in req_cols:
            vals = data_obj.parse_list(row[c])
            if not vals: 
                val = "Unknown"
            else:
                val = vals[0].title()
            if len(val) > 20: val = val[:17] + "..."
            path.append(val)
        paths.append(tuple(path))
        
    counts = Counter(paths)
    if not counts: return
    
    labels = []
    source, target, value = [], [], []
    label_map = {}

    def get_idx(name, stage):
        uid = f"{stage}_{name}"
        if uid not in label_map:
            label_map[uid] = len(labels)
            labels.append(name)
        return label_map[uid]

    for (t_type, d_type, model), count in counts.items():
        s1 = get_idx(t_type, 0)
        t1 = get_idx(d_type, 1)
        source.append(s1); target.append(t1); value.append(count)
        
        s2 = get_idx(d_type, 1)
        t2 = get_idx(model, 2)
        source.append(s2); target.append(t2); value.append(count)

    try:
        fig = go.Figure(data=[go.Sankey(
            node=dict(pad=15, thickness=20, line=dict(color="black", width=0.5),
                      label=labels, color="skyblue"),
            link=dict(source=source, target=target, value=value)
        )])
        fig.update_layout(title_text="Research Flow", font_size=12)
        fig.write_html(output_path)
        png_path = str(output_path).replace('.html', '.png')
        fig.write_image(png_path, scale=2)
        
    except Exception as e:
        print(f"Sankey save error (ensure 'kaleido' is installed): {e}")

def plot_temporal_trend(data_obj, output_path):
    if 'publication_year' not in data_obj.df.columns:
        print("publication_year not found.")
        return

    years = []
    for val in data_obj.df['publication_year']:
        y = data_obj.parse_numeric(val)
        if y and 2000 < y < 2030: years.append(int(y))
        
    if not years: return
    
    counts = Counter(years)
    if not counts: return
    
    min_y, max_y = min(years), max(years)
    all_years = list(range(min_y, max_y + 1))
    vals = [counts.get(y, 0) for y in all_years]
    
    plt.figure(figsize=(10, 5))
    plt.plot(all_years, vals, marker='o', linewidth=2, color='darkblue')
    plt.xticks(all_years)
    plt.title("Publication Trend", fontsize=14)
    plt.ylabel("Count")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
