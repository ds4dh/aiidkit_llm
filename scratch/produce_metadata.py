#!/usr/bin/env python3
"""
generate_vocab_and_domains.py

Comprehensive metadata extraction script that generates:
1. Full Categorized & Sorted Vocabulary CSV (tokens, role, occurrence counts, IDs).
2. Feature Domain Profile CSV (one row per (entity, attribute) showing exact data domains,
   variable types, intervals, allowed categories, and summary stats).
3. Human-readable Markdown Data Dictionary (ready to send directly to your professor).
4. Machine-readable JSON Data Domain specification.
"""

import csv
import json
import pickle
import yaml
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
from datasets import concatenate_datasets, load_from_disk

CONFIG_PATH = Path("configs/discriminative_training.yaml")
BIN_LABELS = ["Lowest", "Lower", "Low", "Middle", "High", "Higher", "Highest"]
SPECIAL_TOKENS = ["[PAD]", "[MASK]", "[BOS]", "[EOS]", "[UNK]"]


def main():
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Missing config at {CONFIG_PATH.resolve()}")

    with open(CONFIG_PATH, "r") as f:
        cfg = yaml.safe_load(f)

    data_split = cfg.get("data_split_type", "temporal_split")
    data_dir = Path(cfg["data_dir"]) / data_split
    meta_dir = data_dir / "processed_cache" / "pretraining_metadata"

    vocab_pkl = meta_dir / "vocab.pkl"
    bin_intervals_pkl = meta_dir / "bin_intervals.pkl"

    if not vocab_pkl.exists():
        raise FileNotFoundError(
            f"Cannot find vocab.pkl at {vocab_pkl}. "
            "Please run scripts/metadata_recovery.py first."
        )

    print(f"Loading cached metadata from: {meta_dir}")
    with open(vocab_pkl, "rb") as f:
        vocab: dict[str, int] = pickle.load(f)

    bin_intervals: dict[str, pd.IntervalIndex] = {}
    if bin_intervals_pkl.exists():
        with open(bin_intervals_pkl, "rb") as f:
            bin_intervals = pickle.load(f)

    # -------------------------------------------------------------------------
    # 1. Load Dataset for Full Population Profile Scan
    # -------------------------------------------------------------------------
    raw_pt_dir = data_dir / "fup_None"
    if not raw_pt_dir.exists():
        raw_pt_dir = next(data_dir.glob("fup_*"), None)

    if raw_pt_dir is None:
        raise FileNotFoundError(f"No valid fup_* directories found inside {data_dir}")

    print(f"Scanning dataset sequences from: {raw_pt_dir.name} ...")
    ds_dict = load_from_disk(str(raw_pt_dir))
    splits_to_scan = [ds_dict[s] for s in ["train", "validation", "test"] if s in ds_dict]
    cohort = concatenate_datasets(splits_to_scan) if splits_to_scan else ds_dict["train"]

    total_patients = len(cohort)
    print(f"Total cohort size: {total_patients:,} patients.")

    # -------------------------------------------------------------------------
    # 2. Extract Token Counts & Attribute Data Domains
    # -------------------------------------------------------------------------
    entity_counts = Counter()
    attribute_counts = Counter()
    value_counts = Counter()

    # Per-feature data collector: (entity, attribute) -> list of raw values
    feature_values = defaultdict(list)
    feature_patients = defaultdict(set)
    feature_obs_per_pt = defaultdict(lambda: defaultdict(int))

    labels_map = {i: label for i, label in enumerate(BIN_LABELS)}

    print("Computing feature distributions, coverage, and token assignments...")
    for sample in cohort:
        pid = sample.get("patientid", sample.get("patientkey", "unknown"))
        entities = sample.get("entity", [])
        attributes = sample.get("attribute", [])
        values = sample.get("value", [])

        entity_counts.update(entities)
        attribute_counts.update(attributes)

        for e, a, v in zip(entities, attributes, values):
            feat_key = (e, a)
            feature_patients[feat_key].add(pid)
            feature_obs_per_pt[feat_key][pid] += 1
            feature_values[feat_key].append(v)

            token_str = str(v)
            if a in bin_intervals:
                try:
                    f_val = float(v)
                    idx = bin_intervals[a].get_loc(f_val)
                    token_str = labels_map[idx]
                except (ValueError, KeyError):
                    try:
                        if float(v) > bin_intervals[a].right.max():
                            token_str = labels_map[len(BIN_LABELS) - 1]
                        else:
                            token_str = labels_map[0]
                    except ValueError:
                        pass
            else:
                try:
                    token_str = str(int(float(v)))
                except (ValueError, TypeError):
                    pass

            value_counts[token_str] += 1

    # -------------------------------------------------------------------------
    # 3. Export Vocabulary (Categorized, Counted, and Sorted)
    # -------------------------------------------------------------------------
    out_dir = Path("scratch/metadata_exports")
    out_dir.mkdir(parents=True, exist_ok=True)
    vocab_csv_path = out_dir / "vocab_categorized_sorted.csv"

    token_rows = []
    for token, token_id in vocab.items():
        if token in SPECIAL_TOKENS:
            token_rows.append({
                "id": token_id,
                "token": token,
                "type": "Special Token",
                "count": 0,
                "section": "SPECIAL TOKENS",
            })
            continue

        c_ent = entity_counts.get(token, 0)
        c_attr = attribute_counts.get(token, 0)
        c_val = value_counts.get(token, 0)
        tot = c_ent + c_attr + c_val

        types = []
        if c_ent > 0:
            types.append("Entity")
        if c_attr > 0:
            types.append("Attribute")
        if c_val > 0 or token in BIN_LABELS:
            types.append("Value")

        role_str = " / ".join(types) if types else "Value"
        section = "ENTITY TOKENS" if "Entity" in types else ("ATTRIBUTE TOKENS" if "Attribute" in types else "VALUE TOKENS")

        token_rows.append({
            "id": token_id,
            "token": token,
            "type": role_str,
            "count": tot,
            "section": section,
        })

    sections = ["SPECIAL TOKENS", "ENTITY TOKENS", "ATTRIBUTE TOKENS", "VALUE TOKENS"]
    with open(vocab_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["token_id", "token", "token_type", "occurrence_count"])
        for sec in sections:
            sec_items = [r for r in token_rows if r["section"] == sec]
            if sec == "SPECIAL TOKENS":
                sec_items.sort(key=lambda x: x["id"])
            else:
                sec_items.sort(key=lambda x: (-x["count"], x["id"]))

            writer.writerow([])
            writer.writerow([f"# --- {sec} ({len(sec_items)}) ---", "", "", ""])
            for item in sec_items:
                writer.writerow([item["id"], item["token"], item["type"], item["count"]])

    print(f"[SUCCESS] Exported sorted vocabulary -> {vocab_csv_path}")

    # -------------------------------------------------------------------------
    # 4. Generate Feature Data Domains
    # -------------------------------------------------------------------------
    feature_domain_records = []
    json_domain_dict = {}

    sorted_features = sorted(feature_values.keys(), key=lambda x: (x[0], x[1]))

    for entity, attribute in sorted_features:
        feat_key = (entity, attribute)
        vals = feature_values[feat_key]
        n_obs = len(vals)
        pts = feature_patients[feat_key]
        n_pts = len(pts)
        coverage_pct = (n_pts / total_patients) * 100.0

        pt_densities = list(feature_obs_per_pt[feat_key].values())
        med_obs = float(np.median(pt_densities)) if pt_densities else 0.0
        q25_obs, q75_obs = np.percentile(pt_densities, [25, 75]) if pt_densities else (0.0, 0.0)

        # Determine Variable Type and Boundaries
        is_binned = attribute in bin_intervals
        domain_desc = ""
        summary_stats = ""
        var_type = "Categorical"

        if is_binned:
            var_type = "Numerical continuous (Binned to Vocabulary)"
            intervals = bin_intervals[attribute]
            float_vals = []
            for v in vals:
                try:
                    float_vals.append(float(v))
                except (ValueError, TypeError):
                    pass

            if float_vals:
                f_arr = np.array(float_vals)
                mean_val = float(np.mean(f_arr))
                std_val = float(np.std(f_arr))
                med_val = float(np.median(f_arr))
                q25, q75 = np.percentile(f_arr, [25, 75])
                p1, p99 = np.percentile(f_arr, [1, 99])
                min_val = float(np.min(f_arr))
                max_val = float(np.max(f_arr))

                summary_stats = (
                    f"Mean: {mean_val:.2f} ± {std_val:.2f} | "
                    f"Median: {med_val:.2f} [IQR: {q25:.2f} - {q75:.2f}] | "
                    f"Min-Max: [{min_val:.2f}, {max_val:.2f}] | "
                    f"P1-P99: [{p1:.2f}, {p99:.2f}]"
                )

            # Map the 7 bin intervals
            bin_strs = []
            json_bins = []
            for lbl, ival in zip(BIN_LABELS, intervals):
                bin_strs.append(f"{lbl}: [{ival.left:.2f}, {ival.right:.2f})")
                json_bins.append({
                    "bin_label": lbl,
                    "left": float(ival.left),
                    "right": float(ival.right),
                    "token_id": vocab.get(lbl, None),
                })
            domain_desc = " ; ".join(bin_strs)

            json_domain_dict[f"{entity} - {attribute}"] = {
                "entity": entity,
                "attribute": attribute,
                "variable_type": var_type,
                "patient_coverage_pct": round(coverage_pct, 2),
                "total_observations": n_obs,
                "obs_per_patient": {"median": med_obs, "iqr": [float(q25_obs), float(q75_obs)]},
                "summary_statistics": summary_stats,
                "binned_vocabulary_intervals": json_bins,
            }

        else:
            # Categorical or Discrete Integer
            str_vals = []
            for v in vals:
                try:
                    str_vals.append(str(int(float(v))))
                except (ValueError, TypeError):
                    str_vals.append(str(v))

            val_counter = Counter(str_vals)
            distinct_k = len(val_counter)
            var_type = "Binary / Categorical" if distinct_k <= 2 else "Categorical"

            # Sort categories by frequency
            cat_breakdown = []
            json_cats = []
            for c_name, c_cnt in val_counter.most_common(15):
                pct = (c_cnt / n_obs) * 100.0
                cat_breakdown.append(f"{c_name} ({pct:.1f}%)")
                json_cats.append({
                    "category": c_name,
                    "count": c_cnt,
                    "percentage": round(pct, 2),
                    "token_id": vocab.get(c_name, vocab.get("[UNK]")),
                })

            if distinct_k > 15:
                cat_breakdown.append(f"+ {distinct_k - 15} more...")

            domain_desc = "Categories: " + ", ".join(cat_breakdown)
            summary_stats = f"Total distinct classes: {distinct_k}"

            json_domain_dict[f"{entity} - {attribute}"] = {
                "entity": entity,
                "attribute": attribute,
                "variable_type": var_type,
                "patient_coverage_pct": round(coverage_pct, 2),
                "total_observations": n_obs,
                "obs_per_patient": {"median": med_obs, "iqr": [float(q25_obs), float(q75_obs)]},
                "distinct_categories_count": distinct_k,
                "categories": json_cats,
            }

        feature_domain_records.append({
            "Entity": entity,
            "Attribute": attribute,
            "Variable_Type": var_type,
            "Patient_Coverage": f"{n_pts:,} ({coverage_pct:.1f}%)",
            "Obs_Per_Patient (Median [IQR])": f"{med_obs:.1f} [{int(q25_obs)}-{int(q75_obs)}]",
            "Total_Observations": n_obs,
            "Domain_Specification_Or_Intervals": domain_desc,
            "Distribution_Summary": summary_stats,
        })

    # Save Feature Domains CSV
    domain_df = pd.DataFrame(feature_domain_records)
    domain_csv_path = out_dir / "feature_data_domains.csv"
    domain_df.to_csv(domain_csv_path, index=False)
    print(f"[SUCCESS] Exported Feature Data Domains CSV -> {domain_csv_path}")

    # Save JSON Specification
    domain_json_path = out_dir / "feature_data_domains.json"
    with open(domain_json_path, "w", encoding="utf-8") as f:
        json.dump(json_domain_dict, f, indent=2, ensure_ascii=False)
    print(f"[SUCCESS] Exported Feature Data Domains JSON -> {domain_json_path}")

    # -------------------------------------------------------------------------
    # 5. Generate Markdown Data Dictionary (for sending to professor)
    # -------------------------------------------------------------------------
    md_path = out_dir / "data_domains_and_vocabulary_report.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Clinical Feature Domains & Vocabulary Specification\n\n")
        f.write(f"- **Total Vocabulary Size:** {len(vocab):,} tokens\n")
        f.write(f"- **Total Patients:** {total_patients:,}\n")
        f.write(f"- **Distinct Clinical Features (Entity-Attribute pairs):** {len(sorted_features):,}\n")
        f.write(f"- **Data Split Engine:** `{data_split}`\n\n")
        f.write("---\n\n")
        f.write("## 1. Feature Data Domains\n\n")
        f.write(
            "| Entity | Attribute | Variable Type | Patient Coverage | Obs / Pt (Median [IQR]) | Domain Specification / Vocabulary Intervals |\n"
        )
        f.write(
            "| :--- | :--- | :--- | :--- | :--- | :--- |\n"
        )
        for _, row in domain_df.iterrows():
            f.write(
                f"| **{row['Entity']}** | `{row['Attribute']}` | {row['Variable_Type']} | "
                f"{row['Patient_Coverage']} | {row['Obs_Per_Patient (Median [IQR])']} | "
                f"{row['Domain_Specification_Or_Intervals']} |\n"
            )

        f.write("\n---\n\n")
        f.write("## 2. Vocabulary Breakdown Summary\n\n")
        for sec in sections:
            sec_items = [r for r in token_rows if r["section"] == sec]
            f.write(f"### {sec} ({len(sec_items)} tokens)\n\n")
            top_sample = sec_items[:12]
            f.write(", ".join([f"`{x['token']}` (ID: {x['id']})" for x in top_sample]))
            if len(sec_items) > 12:
                f.write(f", ... *(and {len(sec_items) - 12} more)*")
            f.write("\n\n")

    print(f"[SUCCESS] Exported Markdown Clinical Dictionary -> {md_path}")
    print("\nAll deliverables generated in `metadata_exports/`:")
    print(" 1. vocab_categorized_sorted.csv")
    print(" 2. feature_data_domains.csv")
    print(" 3. feature_data_domains.json")
    print(" 4. data_domains_and_vocabulary_report.md")


if __name__ == "__main__":
    main()