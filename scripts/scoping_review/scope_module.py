import re
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from mappings import (
    NORMALIZED_LOOKUP, COUNTRY_TO_CONTINENT,
    COLUMN_MAPPING_INVERSED, FEATURE_STRATEGIES,
)


# ============================================================================
# DATA PROCESSING
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
            if self.filepath.suffix == '.xlsx':  # try loading as CSV
                 raw = pd.read_excel(self.filepath, header=None)
            else:  # try typical encodings
                 try:
                     raw = pd.read_csv(self.filepath, header=None, encoding='utf-8-sig')
                 except:
                     raw = pd.read_csv(self.filepath, header=None, encoding='latin1')
        except Exception as e:
            raise ValueError(f"Could not read file: {e}")

        # Filter rows based on instructions in column 0
        # - First, identify Paper IDs from header row (Row 0, Col 2 onwards)
        # - Then, remove rows where Col 0 contains "Ignore feature"
        headers = raw.iloc[0, :].values.astype(str)
        paper_ids = [h for h in headers[1:] if str(h).lower() != 'nan']
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
            match = re.search(r"(\d+(\.\d+)?)", s)  # first number found
            if match: return float(match.group(1))
        return None


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_bar_distribution(data_obj, feature, title, output_path, top_n=15, value_mapper=None):
    if feature not in data_obj.df.columns: return
    parsed_series = data_obj.df[feature].apply(data_obj.parse_list)

    # Apply value mapping, if provided
    if value_mapper:
        lookup = {v.lower(): k for k, vals in value_mapper.items() for v in vals}
        map_items = lambda items: [lookup.get(str(i).strip().lower(), i) for i in items]
        parsed_series = parsed_series.apply(map_items)

    # Apply strategy
    strategy = FEATURE_STRATEGIES.get(feature, 'aggregate')
    if strategy == 'take_first':  
        series = parsed_series.apply(lambda x: x[0] if x else np.nan)
    elif strategy == 'join':      
        series = parsed_series.apply(lambda x: " & ".join(sorted(set(x))) if x else np.nan) # Added sorted/set for cleaner joins
    elif strategy == 'binarize':  
        neg_terms = {'no', 'none', 'ni', 'not specified', 'nan', 'n/a', 'unknown'}
        series = parsed_series.apply(lambda x: "Reported" if x and any(str(i).strip().lower() not in neg_terms for i in x) else "Not Reported")
    else: # 'aggregate'
        series = parsed_series.explode()

    # Clean and plot
    series = series.dropna().astype(str).str.strip().str.title()
    if series.empty: return
    df_counts = series.value_counts().head(top_n).reset_index()
    df_counts.columns = ['Category', 'Count']
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=df_counts, x='Count', y='Category', palette='Set2' if strategy == 'binarize' else 'viridis')
    ax.bar_label(ax.containers[0])    
    plt.title(f"{title}", fontsize=14, fontweight='bold')
    plt.xlabel('Number of Papers'); plt.ylabel(''); plt.tight_layout()
    plt.savefig(output_path); plt.close()


def plot_metric_reporting(data_obj, output_path):
    metric_cols = [c for c in data_obj.df.columns if c.startswith('metrics_')]
    neg_terms = {
        'no', 'none', 'ni', 'not specified', 'nan', 'n/a', 'unknown',
        'not considered', 'not reported', 'not applicable',
    }
    results = {}
    for col in metric_cols:
        valid_mask = data_obj.df[col].apply(
            lambda x: any(
                str(i).strip().lower() not in neg_terms
                for i in data_obj.parse_list(x))
            )
        if (count := valid_mask.sum()) > 0:
            results[COLUMN_MAPPING_INVERSED.get(col, col.replace('metrics_', '').title())] = count

    if not results: return
    plt.figure(figsize=(12, 7))
    bars = plt.bar(*zip(*sorted(results.items(), key=lambda x: x[1], reverse=True)), color='#4c72b0', edgecolor='black')
    plt.bar_label(bars); plt.xticks(rotation=45, ha='right')
    plt.title("Frequency of Reported Performance Metrics", fontsize=14)
    plt.ylabel("Number of Studies")
    plt.tight_layout(); plt.savefig(output_path); plt.close()


def plot_reporting_gaps(data_obj, output_path):
    cols = ['model_code_available', 'data_available', 'model_external_validation', 'data_ethics_approval']
    gaps = {}
    for col in [c for c in cols if c in data_obj.df.columns]:
        gap_str = data_obj.df[col].astype(str).str.lower().str
        gap = gap_str.contains(r'no|none|ni|not specified|nan|request').sum()
        gaps[COLUMN_MAPPING_INVERSED.get(col, col)] = gap

    if not gaps: return
    plt.figure(figsize=(8, 5))
    plt.bar_label(plt.bar(gaps.keys(), gaps.values(), color='#c44e52', alpha=0.8))
    plt.title(f"Reporting Gaps (Negative or Not Specified) / N={data_obj.n_papers}"); plt.tight_layout()
    plt.savefig(output_path); plt.close()


def plot_bubble_chart(data_obj, output_path):
    if not {'data_n_participants', 'metrics_auroc'}.issubset(data_obj.df.columns): return
    
    def extract_row(row):
        n = data_obj.parse_numeric(row.get('data_n_participants'))
        auroc = data_obj.parse_numeric(row.get('metrics_auroc'))
        if not (n and auroc): return None
        return {
            'N': n, 'AUROC': auroc / 100.0 if auroc > 1.0 else auroc,
            'Events': data_obj.parse_numeric(row.get('data_n_events')) or 20,
            'Transplant': (data_obj.parse_list(row.get('transplant_type')) or ["Unknown"])[0].title(),
            'Paper': str(row.get('PAPER_ID', ''))
        }

    df_plot = pd.DataFrame([r for r in data_obj.df.apply(extract_row, axis=1) if r])
    if df_plot.empty: return

    try:
        fig = px.scatter(
            df_plot, x="N", y="AUROC", size="Events", color="Transplant", hover_name="Paper",
            log_x=True, title="Model Performance vs. Sample Size", height=600,
        )
        fig.write_html(output_path)
        fig.write_image(str(output_path).replace('.html', '.png'), scale=2)
    except Exception as e: print(f"Plotly save error: {e}")


def plot_sankey(data_obj, output_path):
    cols = ['transplant_type', 'data_type', 'model_best_main']
    if not all(c in data_obj.df.columns for c in cols): return

    # Create paths
    paths = data_obj.df[cols].apply(
        lambda row: tuple(
            (data_obj.parse_list(val) or ["Unknown"])[0].title()[:17]
            + ("..." if len((data_obj.parse_list(val) or ["Unknown"])[0]) > 20 else "")
            for val in row
        ), axis=1)
    counts = paths.value_counts()
    if counts.empty: return

    labels, source, target, value, label_map = [], [], [], [], {}
    def get_idx(n, s): 
        uid = f"{s}_{n}"; idx = label_map.setdefault(uid, len(labels))
        if idx == len(labels): labels.append(n)
        return idx

    for (t, d, m), c in counts.items():
        source.extend([get_idx(t, 0), get_idx(d, 1)])
        target.extend([get_idx(d, 1), get_idx(m, 2)])
        value.extend([c, c])

    try:
        node = dict(pad=15, thickness=20, line=dict(color="black", width=0.5), label=labels, color="skyblue")
        go.Figure(
            data=[go.Sankey(node=node, link=dict(source=source, target=target, value=value))]
        ).update_layout(title_text="Research Flow", font_size=12).write_html(output_path)
        
    except Exception as e: print(f"Sankey save error: {e}")


def plot_temporal_trend(data_obj, output_path):
    if 'publication_year' not in data_obj.df.columns: return
    years = data_obj.df['publication_year'].apply(data_obj.parse_numeric).dropna().astype(int)
    years = years[(years > 2000) & (years < 2030)].value_counts().sort_index()
    if years.empty: return
    years = years.reindex(range(years.index.min(), years.index.max() + 1), fill_value=0)
    
    plt.figure(figsize=(10, 5))
    plt.plot(years.index, years.values, marker='o', linewidth=2, color='darkblue')
    plt.xticks(years.index); plt.title("Publication Trend", fontsize=14); plt.ylabel("Count")
    plt.grid(True, alpha=0.3); plt.tight_layout(); plt.savefig(output_path); plt.close()


def plot_continent_countries_stack(data_obj, output_path):
    if 'data_sites_region' not in data_obj.df.columns: return
    
    # Flatten country list and map to continent
    exploded = data_obj.df['data_sites_region'].apply(data_obj.parse_list).explode()
    exploded = exploded.dropna().str.strip().str.title()
    if exploded.empty: return
    
    def get_continent(loc):
        s = str(loc).lower().strip()
        return next((v for k, v in COUNTRY_TO_CONTINENT.items() if k in s), "Unknown")
    
    df = pd.DataFrame({'Country': exploded, 'Continent': exploded.apply(get_continent)})
    df_pivot = pd.crosstab(df['Continent'], df['Country'])
    
    # Sort by row sum then column sum
    df_pivot = df_pivot.loc[
        df_pivot.sum(1).sort_values(ascending=False).index,
        df_pivot.sum().sort_values(ascending=False).index,
    ]

    cmap = ListedColormap(np.vstack((plt.get_cmap('Pastel1').colors, plt.get_cmap('Set3').colors)))
    ax = df_pivot.plot(
        kind='bar', stacked=True, figsize=(12, 8), colormap=cmap,
        edgecolor='black', linewidth=0.3, width=0.8, legend=False,
    )
    
    min_h = ax.get_ylim()[1] * 0.02
    for c_idx, container in enumerate(ax.containers):
        name = df_pivot.columns[c_idx]
        for rect in container:
            if (h := rect.get_height()) > min_h:
                ax.text(
                    rect.get_x() + rect.get_width()/2, rect.get_y() + h/2,
                    f"{name[:18]+'.' if len(name)>20 else name} ({int(h)})",
                    ha='center', va='center', color='black', fontsize=11, fontweight='medium',
                )

    plt.title("Distribution of Study Sites", fontsize=16, fontweight='bold'); plt.xlabel(""); plt.ylabel("Number of Study Sites", fontsize=12)
    plt.xticks(rotation=0, fontsize=12); plt.tight_layout(); plt.savefig(output_path, dpi=300); plt.close()