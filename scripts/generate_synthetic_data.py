"""
generate_synthetic_data.py

Generates realistic synthetic datasets for the AIIDKIT pipeline by programmatically
incorporating empirical statistics from an actual dataset profile (data/teav_feature_profile.csv).

Features:
1. Ingests 180+ entity-attribute pairs with empirical coverage, observation frequencies
   (median [IQR]), variable types (Categorical, Categorical index, Numerical continuous),
   and distributions (classes/weights, mean, std, median, IQR, P1-P99).
2. Generates coherent patient trajectories: baseline events at time 0.0 and longitudinal
   routine visits and acute events (labs, vitals, meds, infections, biopsies) across time.
3. Exports both HuggingFace DatasetDict objects (arrow format) across follow-up windows
   (fup_None, fup_0000, fup_0030, ...) and individual patient CSV files (columns: time, entity, attribute, value).
4. Provides a CLI toggle (--generate-classic-ml) to control
   whether classic ML tabular datasets are generated (disabled by default).
"""

import argparse
import csv
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict


# ==========================================
# 1. Profile Parser & Statistical Containers
# ==========================================

@dataclass
class AttributeProfile:
    entity: str
    attribute: str
    coverage: float  # fraction in [0, 1]
    median_obs: float
    iqr_low: float
    iqr_high: float
    total_obs: int
    var_type: str  # "Categorical", "Categorical index", "Numerical continuous"
    categories: list[str] = field(default_factory=list)
    category_weights: list[float] = field(default_factory=list)
    mean: float = 0.0
    std: float = 1.0
    median_val: float = 0.0
    iqr_low_val: float = 0.0
    iqr_high_val: float = 0.0
    p1: float = 0.0
    p99: float = 1.0
    is_baseline: bool = False


class FeatureProfileParser:
    """Parses and models empirical dataset statistics from teav_feature_profile.csv."""

    def __init__(self, profile_path: str | Path = "data/teav_feature_profile.csv"):
        self.profile_path = Path(profile_path)
        self.profiles: list[AttributeProfile] = []
        self.by_entity: dict[str, list[AttributeProfile]] = {}
        self.by_key: dict[tuple[str, str], AttributeProfile] = {}
        self.baseline_attrs: list[AttributeProfile] = []
        self.longitudinal_attrs: list[AttributeProfile] = []
        self._load()

    def _load(self):
        if not self.profile_path.exists():
            raise FileNotFoundError(f"Feature profile file not found at: {self.profile_path.resolve()}")

        try:
            df = pd.read_csv(self.profile_path, encoding="utf-8")
        except UnicodeDecodeError:
            df = pd.read_csv(self.profile_path, encoding="latin1")

        baseline_entities = {
            "Patient info", "Donor info", "Transplant info",
            "Mismatch info", "Serology info", "Patient serology", "History"
        }

        for _, row in df.iterrows():
            entity = str(row["Entity"]).strip()
            attribute = str(row["Attribute"]).strip()
            cov_str = str(row["Patient coverage (%)"]).strip()
            obs_str = str(row["Obs / patient (median [IQR])"]).strip()
            total_obs = int(str(row["Total observations"]).replace(",", "").strip())
            var_type = str(row["Variable type"]).strip()
            val_range_str = str(row["Value range / distinct classes (up to 10)"]).strip()

            # Coverage parsing (e.g. "3,312 (100.0%)" -> 1.0)
            m_cov = re.search(r"([\d\.]+)%", cov_str)
            coverage = float(m_cov.group(1)) / 100.0 if m_cov else 1.0

            # Obs / patient parsing (e.g. "1.0 [1-1]" or "20.0 [15-27]")
            m_obs = re.search(r"([\d\.]+)\s*\[(\d+)-(\d+)\]", obs_str)
            if m_obs:
                median_obs = float(m_obs.group(1))
                iqr_low = float(m_obs.group(2))
                iqr_high = float(m_obs.group(3))
            else:
                median_obs = 1.0
                iqr_low = 1.0
                iqr_high = 1.0

            prof = AttributeProfile(
                entity=entity,
                attribute=attribute,
                coverage=coverage,
                median_obs=median_obs,
                iqr_low=iqr_low,
                iqr_high=iqr_high,
                total_obs=total_obs,
                var_type=var_type,
            )

            # Parse value distributions
            if var_type in ["Categorical", "Categorical index"]:
                # Matches: label (count, pct%)
                matches = re.findall(r"(?:^|,\s*)(.*?)\s*\(\s*([\d,]+)\s*,\s*([\d\.]+)%\s*\)", val_range_str)
                if matches:
                    cats = [m[0].strip() for m in matches]
                    weights = [float(m[2]) for m in matches]
                    tot = sum(weights)
                    if tot > 0:
                        weights = [w / tot for w in weights]
                    else:
                        weights = [1.0 / len(cats)] * len(cats)
                    prof.categories = cats
                    prof.category_weights = weights
                else:
                    prof.categories = ["0", "1"]
                    prof.category_weights = [0.5, 0.5]

            elif var_type == "Numerical continuous":
                # Matches Mean: X ± Y | Median: M [IQR: Q1-Q3] | 98% range [P1-P99]: [P1, P99]
                m_mean = re.search(r"Mean:\s*([-\d\.]+)\s*[±\xb1\?]\s*([-\d\.]+)", val_range_str)
                m_med = re.search(r"Median:\s*([-\d\.]+)", val_range_str)
                m_iqr = re.search(r"IQR:\s*([-\d\.]+)-([-\d\.]+)", val_range_str)
                m_range = re.search(r"\[P1-P99\]:\s*\[([-\d\.]+),\s*([-\d\.]+)\]", val_range_str)

                if m_mean:
                    prof.mean = float(m_mean.group(1))
                    prof.std = max(0.001, float(m_mean.group(2)))
                if m_med:
                    prof.median_val = float(m_med.group(1))
                if m_iqr:
                    prof.iqr_low_val = float(m_iqr.group(1))
                    prof.iqr_high_val = float(m_iqr.group(2))
                if m_range:
                    prof.p1 = float(m_range.group(1))
                    prof.p99 = float(m_range.group(2))

            # Classify baseline vs longitudinal
            prof.is_baseline = (entity in baseline_entities) or (median_obs <= 1.0 and iqr_high <= 1.0 and entity not in ["Patient vitals", "Patient lab", "Organ lab", "Medication", "Infection", "Biopsy"])

            self.profiles.append(prof)
            self.by_entity.setdefault(entity, []).append(prof)
            self.by_key[(entity, attribute)] = prof
            if prof.is_baseline:
                self.baseline_attrs.append(prof)
            else:
                self.longitudinal_attrs.append(prof)

    def sample_value(self, prof: AttributeProfile) -> str:
        """Samples a concrete observation value matching the empirical distribution."""
        if prof.var_type in ["Categorical", "Categorical index"]:
            if prof.categories and prof.category_weights:
                return random.choices(prof.categories, weights=prof.category_weights)[0]
            return "0"

        # Numerical continuous
        val = np.random.normal(prof.mean, prof.std)
        if prof.p1 < prof.p99:
            val = np.clip(val, prof.p1, prof.p99)
        if prof.p1 >= 0:
            val = max(0.0, float(val))

        # Format integer-like continuous features cleanly, otherwise 2 decimal places
        if prof.mean.is_integer() and prof.std.is_integer() and prof.p1.is_integer():
            return str(int(round(val)))
        return f"{round(float(val), 2):.2f}"


# ==========================================
# 2. Patient Trajectory & Label Generator
# ==========================================

LABEL_TASKS = [
    "label_infection_bacteria_0030d",
    "label_infection_bacteria_0060d",
    "label_infection_bacteria_0090d",
    "label_infection_virus_0030d",
    "label_infection_virus_0060d",
    "label_infection_virus_0090d",
]


def generate_single_patient_trajectory(
    parser: FeatureProfileParser,
    patient_id: str,
    max_days: float = 365.0,
) -> tuple[list[tuple[float, str, str, str]], dict[str, int]]:
    """
    Generates a full chronological (time, entity, attribute, value) trajectory
    and coordinated infection prediction task labels for one patient.
    """
    events: list[tuple[float, str, str, str]] = []

    # 1. Baseline measurements at time 0.0 (pre-transplant or at transplant)
    for prof in parser.baseline_attrs:
        if random.random() < prof.coverage:
            val = parser.sample_value(prof)
            events.append((0.0, prof.entity, prof.attribute, val))

    # 2. Longitudinal routine clinic visits (Vitals and Labs)
    # Post-transplant follow-up schedule: frequent early, spread across the year
    n_visits = random.randint(6, 14)
    # Log-linear distribution of visits so early days have higher density
    u = np.sort(np.random.uniform(0.05, 1.0, size=n_visits))
    visit_times = sorted(list(set(round(float(t), 1) for t in (u ** 1.5) * max_days)))
    if not visit_times or visit_times[0] <= 0:
        visit_times = [7.0, 14.0, 30.0, 60.0, 90.0, 180.0, 360.0]

    for t in visit_times:
        # Patient vitals
        for prof in parser.by_entity.get("Patient vitals", []):
            if random.random() < min(1.0, prof.coverage * 1.1):
                if prof.attribute == "Transplant age (years)":
                    val = f"{t / 365.25:.2f}"
                else:
                    val = parser.sample_value(prof)
                events.append((t, prof.entity, prof.attribute, val))

        # Patient labs
        for prof in parser.by_entity.get("Patient lab", []):
            if random.random() < min(1.0, prof.coverage * 1.0):
                val = parser.sample_value(prof)
                events.append((t, prof.entity, prof.attribute, val))

        # Organ labs
        for prof in parser.by_entity.get("Organ lab", []):
            if random.random() < min(1.0, prof.coverage * 0.9):
                val = parser.sample_value(prof)
                events.append((t, prof.entity, prof.attribute, val))

    # 3. Longitudinal Medications (starts and stops)
    for prof in parser.by_entity.get("Medication", []):
        if random.random() < prof.coverage:
            k = max(1, int(round(np.random.normal(prof.median_obs, 0.8))))
            k = min(k, 4)
            for _ in range(k):
                t_med = round(float(random.choice([0.0, 1.0, 7.0, 14.0, 30.0, 60.0, 90.0, 180.0])), 1)
                val = parser.sample_value(prof)
                events.append((t_med, prof.entity, prof.attribute, val))

    # 4. Acute Events: Infections
    bacterial_days: list[float] = []
    viral_days: list[float] = []

    # Overall infection coverage in real data is ~92%
    inf_prof = parser.by_key.get(("Infection", "Category"))
    inf_cov = inf_prof.coverage if inf_prof else 0.92
    if random.random() < inf_cov:
        num_infections = random.choices([1, 2, 3], weights=[0.65, 0.25, 0.10])[0]
        for _ in range(num_infections):
            t_inf = round(float(random.uniform(5.0, max_days)), 1)
            # Sample infection category
            cat_prof = parser.by_key.get(("Infection", "Category"))
            category = parser.sample_value(cat_prof) if cat_prof else random.choice(["Bacteria", "Virus"])

            if category == "Bacteria":
                bacterial_days.append(t_inf)
            elif category == "Virus":
                viral_days.append(t_inf)

            # Record related infection attributes at this exact time
            for attr_name in ["Type", "Category", "Site", "Pathogen", "Clinically significant", "Resistance - MDR"]:
                prof = parser.by_key.get(("Infection", attr_name))
                if prof:
                    val = category if attr_name == "Category" else parser.sample_value(prof)
                    events.append((t_inf, "Infection", attr_name, val))

    # 5. Acute Events: Biopsies
    biop_prof = parser.by_key.get(("Biopsy", "Rejection diagnosis"))
    biop_cov = biop_prof.coverage if biop_prof else 0.79
    if random.random() < biop_cov:
        num_biopsies = random.choices([1, 2], weights=[0.8, 0.2])[0]
        for _ in range(num_biopsies):
            t_bio = round(float(random.uniform(7.0, max_days)), 1)
            for prof in parser.by_entity.get("Biopsy", []):
                if random.random() < min(1.0, prof.coverage * 1.1):
                    val = parser.sample_value(prof)
                    events.append((t_bio, "Biopsy", prof.attribute, val))

    # 6. Virology screenings
    for prof in parser.by_entity.get("Virology", []):
        if random.random() < prof.coverage:
            for _ in range(random.randint(1, 3)):
                t_vir = round(float(random.uniform(14.0, max_days)), 1)
                val = parser.sample_value(prof)
                events.append((t_vir, "Virology", prof.attribute, val))

    # 7. Other longitudinal complications (Comorbidity, Organ event, Malignancy)
    for ent_name in ["Comorbidity", "Organ event", "Malignancy"]:
        for prof in parser.by_entity.get(ent_name, []):
            if random.random() < prof.coverage:
                t_ev = round(float(random.uniform(10.0, max_days)), 1)
                val = parser.sample_value(prof)
                events.append((t_ev, ent_name, prof.attribute, val))

    # Chronologically sort all events for this patient
    events.sort(key=lambda x: x[0])

    # Compute binary prediction labels based on actual simulated events
    labels: dict[str, int] = {}

    def get_binary_status(days_list: list[float], horizon: float) -> int:
        has_event = any(d <= horizon for d in days_list)
        # Small probability (5%) of missing/censored label (-100)
        if random.random() < 0.05:
            return -100
        return 1 if has_event else 0

    labels["label_infection_bacteria_0030d"] = get_binary_status(bacterial_days, 30.0)
    labels["label_infection_bacteria_0060d"] = get_binary_status(bacterial_days, 60.0)
    labels["label_infection_bacteria_0090d"] = get_binary_status(bacterial_days, 90.0)

    labels["label_infection_virus_0030d"] = get_binary_status(viral_days, 30.0)
    labels["label_infection_virus_0060d"] = get_binary_status(viral_days, 60.0)
    labels["label_infection_virus_0090d"] = get_binary_status(viral_days, 90.0)

    return events, labels


# ==========================================
# 3. Dataset Assembly & Slicing
# ==========================================

def generate_cohort(
    parser: FeatureProfileParser,
    num_samples: int = 60,
) -> dict[str, list[dict[str, Any]]]:
    """Generates full patient trajectories and labels partitioned by split."""
    split_counts = {
        "train": max(12, int(num_samples * 0.6)),
        "validation": max(6, int(num_samples * 0.2)),
        "test": max(6, int(num_samples * 0.2)),
    }

    patient_records_by_split: dict[str, list[dict[str, Any]]] = {}
    patient_id_counter = 1

    for split_name, count in split_counts.items():
        cohort = []
        for _ in range(count):
            pid = f"P{patient_id_counter:04d}"
            patient_id_counter += 1
            events, labels = generate_single_patient_trajectory(parser, pid, max_days=365.0)
            record = {
                "patientid": pid,
                "patientkey": f"{pid}_seq",
                "events": events,
                **labels,
            }
            cohort.append(record)

        # Guarantee at least 4 positive and 4 negative labels in every split
        # so downstream ROC-AUC metrics and cross-entropy evaluate cleanly
        for lbl in LABEL_TASKS:
            pos_indices = [i for i, r in enumerate(cohort) if r[lbl] == 1]
            neg_indices = [i for i, r in enumerate(cohort) if r[lbl] == 0]

            needed_pos = max(0, 4 - len(pos_indices))
            if needed_pos > 0:
                for idx in range(min(needed_pos, len(cohort))):
                    cohort[idx][lbl] = 1

            needed_neg = max(0, 4 - len(neg_indices))
            if needed_neg > 0:
                for idx in range(min(needed_neg, len(cohort) - 4)):
                    cohort[4 + idx][lbl] = 0

        patient_records_by_split[split_name] = cohort

    return patient_records_by_split


def build_teav_dataset_dict(
    patient_records_by_split: dict[str, list[dict[str, Any]]],
    fup: int | None = None,
) -> DatasetDict:
    """
    Converts full patient trajectories into a HuggingFace DatasetDict.
    For a given follow-up window (fup), events are sliced to time <= fup.
    Baseline events (time == 0.0) are always preserved.
    """
    splits = {}

    for split_name, cohort in patient_records_by_split.items():
        data = {
            "entity": [],
            "attribute": [],
            "value": [],
            "time": [],
            "patientid": [],
            "patientkey": [],
        }
        for lbl in LABEL_TASKS:
            data[lbl] = []

        for record in cohort:
            pid = record["patientid"]
            pkey = f"{pid}_seq" if fup is None else f"{pid}_{fup:04d}d"

            # Filter events by follow-up window
            if fup is None:
                filtered_events = record["events"]
            elif fup == 0:
                filtered_events = [e for e in record["events"] if e[0] <= 0.0]
            else:
                filtered_events = [e for e in record["events"] if e[0] <= float(fup)]

            # Always ensure at least baseline events exist
            if not filtered_events:
                filtered_events = [e for e in record["events"] if e[0] <= 0.0]

            data["entity"].append([e[1] for e in filtered_events])
            data["attribute"].append([e[2] for e in filtered_events])
            data["value"].append([e[3] for e in filtered_events])
            data["time"].append([e[0] for e in filtered_events])
            data["patientid"].append(pid)
            data["patientkey"].append(pkey)

            for lbl in LABEL_TASKS:
                data[lbl].append(record[lbl])

        splits[split_name] = Dataset.from_dict(data)

    return DatasetDict(splits)


# ==========================================
# 4. Individual Patient Sample CSVs
# ==========================================

def save_individual_patient_samples(
    cohort_splits: dict[str, list[dict[str, Any]]],
    output_dir: Path,
    num_samples: int = 10,
):
    """
    Exports individual patient timeline CSV files with columns (time, entity, attribute, value).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    all_patients = []
    for split_name in ["train", "validation", "test"]:
        all_patients.extend(cohort_splits.get(split_name, []))

    selected = all_patients[:num_samples]
    print(f"Exporting {len(selected)} individual patient CSV samples to: {output_dir.resolve()}")

    for i, patient in enumerate(selected, 1):
        filename = output_dir / f"patient_{i:04d}.csv"
        with open(filename, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["time", "entity", "attribute", "value"])
            for t, ent, attr, val in patient["events"]:
                writer.writerow([t, ent, attr, val])

    print(f"[SUCCESS] Exported {len(selected)} sample CSVs successfully.")


# ==========================================
# 5. Classic ML Tabular Data Generator
# ==========================================

def generate_classic_ml_data(
    num_samples: int = 60,
    fup: int = 30,
    parser: FeatureProfileParser | None = None,
) -> pd.DataFrame:
    """
    Generates synthetic tabular DataFrame for classic ML models using
    empirical profile distributions.
    """
    records = []

    for i in range(num_samples):
        pid = f"P{i+1:04d}"
        pkey = f"{pid}_{fup:04d}d"

        # Empirical feature sampling from profile if available
        if parser:
            age_prof = parser.by_key.get(("Patient info", "Age at transplant (years)"))
            age_val = float(parser.sample_value(age_prof)) if age_prof else random.uniform(18, 75)

            bmi_prof = parser.by_key.get(("Patient vitals", "Patient BMI (kg/m²)"))
            bmi_val = float(parser.sample_value(bmi_prof)) if bmi_prof else random.uniform(18.5, 35.0)

            creat_prof = parser.by_key.get(("Patient lab", "Creatinine (µmol/l)"))
            creat_val = float(parser.sample_value(creat_prof)) if creat_prof else random.uniform(60, 200)

            sex_prof = parser.by_key.get(("Patient info", "Sex"))
            is_female = 1 if (sex_prof and parser.sample_value(sex_prof) == "Female") else random.choice([0, 1])

            dgf_prof = parser.by_key.get(("Transplant info", "Delayed graft function"))
            dgf_val = 1 if (dgf_prof and parser.sample_value(dgf_prof) == "1") else random.choice([0, 1])
        else:
            age_val = random.uniform(18, 75)
            bmi_val = random.uniform(18.5, 35.0)
            creat_val = random.uniform(60, 200)
            is_female = random.choice([0, 1])
            dgf_val = random.choice([0, 1])

        rec = {
            "patientkey": pkey,
            "patientid": pid,
            "obs_end_days": float(fup + 30),
            "fup": fup,
            # Feature columns
            "age": age_val,
            "bmi": bmi_val,
            "creatinine_baseline": creat_val,
            "tacrolimus_mean": random.uniform(3.0, 12.0),
            "wbc_min": random.uniform(2.0, 11.0),
            "gender_female": is_female,
            "delayed_graft_function": dgf_val,
            "donor_deceased": random.choice([0, 1]),
            "hypertension_history": random.choice([0, 1]),
            "diabetes_history": random.choice([0, 1]),
        }

        for lbl in LABEL_TASKS:
            r = random.random()
            if r < 0.40:
                rec[lbl] = 1
            elif r < 0.95:
                rec[lbl] = 0
            else:
                rec[lbl] = -100

        records.append(rec)

    df = pd.DataFrame(records)
    for lbl in LABEL_TASKS:
        if (df[lbl] == 1).sum() < 4:
            df.loc[:3, lbl] = 1
        if (df[lbl] == 0).sum() < 4:
            df.loc[4:7, lbl] = 0

    return df


# ==========================================
# 6. Main Pipeline
# ==========================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate profile-driven synthetic data for AIIDKIT pipeline testing.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--profile_path", type=str, default="data/teav_feature_profile.csv", help="Path to empirical dataset profile CSV.")
    parser.add_argument("--output_dir", type=str, default="data/synthetic", help="Base output directory.")
    parser.add_argument("--samples", type=int, default=60, help="Number of patient samples to generate per split strategy.")
    parser.add_argument("--num_individual_samples", type=int, default=10, help="Number of individual patient CSV samples to export.")
    parser.add_argument(
        "--generate_classic_ml",
        "--generate-classic-ml",
        "--classic_ml",
        "--classic-ml",
        dest="generate_classic_ml",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to generate classic ML tabular datasets (parquet format). Disabled by default; use --generate-classic-ml to enable.",
    )
    parser.add_argument(
        "--split_strategies",
        nargs="+",
        default=["temporal_split", "random_split", "center_split"],
        help="Split strategies to generate datasets for.",
    )
    args = parser.parse_args()

    base_dir = Path(args.output_dir)
    profile_file = Path(args.profile_path)
    print(f"Loading empirical dataset statistics from: {profile_file.resolve()}")
    profile_parser = FeatureProfileParser(profile_file)
    print(f"Loaded {len(profile_parser.profiles)} feature profiles across {len(profile_parser.by_entity)} clinical entities.")
    print(f"  - Baseline attributes: {len(profile_parser.baseline_attrs)}")
    print(f"  - Longitudinal attributes: {len(profile_parser.longitudinal_attrs)}")

    fup_list = [0, 30, 60, 90, 120, 150, 180, 360]

    # Generate one shared cohort for individual CSV sample export
    cohort_for_samples = generate_cohort(profile_parser, num_samples=max(args.samples, args.num_individual_samples))
    if args.num_individual_samples > 0:
        samples_dir = base_dir / "teav" / "_individual_samples"
        save_individual_patient_samples(cohort_for_samples, samples_dir, num_samples=args.num_individual_samples)

    for split_strat in args.split_strategies:
        print(f"\n=======================================================")
        print(f"Generating synthetic data for split: [{split_strat}]")
        print(f"=======================================================")

        # 1. Generate tEAV Datasets
        teav_dir = base_dir / "teav" / split_strat
        teav_dir.mkdir(parents=True, exist_ok=True)

        # Generate realistic cohort for this split
        cohort = generate_cohort(profile_parser, num_samples=args.samples)

        # Pretraining fup_None (full trajectories)
        print(f"  -> Building tEAV pretraining dataset (fup_None)...")
        ds_none = build_teav_dataset_dict(cohort, fup=None)
        ds_none.save_to_disk(str(teav_dir / "fup_None"))

        # Finetuning follow-up windows (fup_0000, fup_0030, ...)
        for fup in fup_list:
            print(f"  -> Building tEAV finetuning dataset (fup_{fup:04d})...")
            ds_fup = build_teav_dataset_dict(cohort, fup=fup)
            ds_fup.save_to_disk(str(teav_dir / f"fup_{fup:04d}"))

        # 2. Generate Classic ML Datasets (if requested)
        if args.generate_classic_ml:
            ml_dir = base_dir / "classic_ml" / split_strat
            ml_dir.mkdir(parents=True, exist_ok=True)

            for fup in fup_list:
                fup_folder = ml_dir / f"fup_{fup:04d}d"
                fup_folder.mkdir(parents=True, exist_ok=True)
                print(f"  -> Generating Classic ML parquet files in {fup_folder.name}...")

                df_full = generate_classic_ml_data(num_samples=args.samples, fup=fup, parser=profile_parser)

                n_train = int(len(df_full) * 0.6)
                n_val = int(len(df_full) * 0.2)

                df_train = df_full.iloc[:n_train].copy()
                df_train["split"] = "train"

                df_val = df_full.iloc[n_train:n_train + n_val].copy()
                df_val["split"] = "validation"

                df_test = df_full.iloc[n_train + n_val:].copy()
                df_test["split"] = "test"

                df_train.to_parquet(fup_folder / "train.parquet", index=False)
                df_val.to_parquet(fup_folder / "validation.parquet", index=False)
                df_test.to_parquet(fup_folder / "test.parquet", index=False)
        else:
            print(f"  [SKIPPED] Classic ML dataset generation skipped (--no-generate-classic-ml).")

    print("\n[SUCCESS] Synthetic data generation complete!")
    print(f"tEAV path: {(base_dir / 'teav').resolve()}")
    if args.generate_classic_ml:
        print(f"Classic ML path: {(base_dir / 'classic_ml').resolve()}")
    if args.num_individual_samples > 0:
        print(f"Individual CSV samples path: {(base_dir / 'teav' / '_individual_samples').resolve()}")


if __name__ == "__main__":
    main()

