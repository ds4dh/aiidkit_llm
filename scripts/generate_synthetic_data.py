"""
Generate a realistic, privacy-safe synthetic dataset for AIIDKIT (v3.9 schema).

This script reads the feature distribution profile (`data/teav_feature_profile.csv`)
and simulates longitudinal post-transplant clinical trajectories for a cohort of
synthetic patients, faithfully reproducing the preprocessing schema of
`../aiidkit_mhmmdrz`:

1. **t-EAV sequences (`<output-dir>/teav`)**:
   - Saved as HuggingFace `DatasetDict` (`train`, `validation`, `test`) per
     split strategy (`random_split`, `center_split`, `temporal_split`) and
     follow-up landmark (`fup_0000`, `fup_0030`, ...).
   - Sequences are truncated at `time <= fup`.

2. **Classic ML tabular datasets (`<output-dir>/classic_ml`)**:
   - Derived from the **exact same** simulated patient cohort, split assignments,
     and follow-up landmarks as the t-EAV dataset so that downstream evaluation
     (`analysis_comparison.py`, `analysis_survival_comparison.py`, `audit_models.py`)
     can inner-join predictions on `(patient_id, fup, horizon, y_true)` across all
     model families without dropping rows.
   - Saved as Parquet files (`train.parquet`, `validation.parquet`, `test.parquet`).

3. **Complete v3.9 Label Suite (computed dynamically at each `obs_end = fup`)**:
   - **Binary classification labels** (`int64` in `{0, 1, -100}` for each horizon `H`):
     - General: `label_infection_{H:04d}d`, `label_graft_loss_{H:04d}d`, `label_death_{H:04d}d`
     - Categories: `label_infection_bacteria_{H:04d}d`, `label_infection_virus_{H:04d}d`,
       `label_infection_fungi_{H:04d}d`
     - 21 Pathogen Groups: `label_infection_grp_{gid}_{H:04d}d`
   - **Right-censored survival labels** (`int64`, `_time >= 1`, `_status in {0, 1}`):
     - General: `label_infection_{time,status}`, `label_graft_loss_{time,status}`,
       `label_death_{time,status}`
     - Categories: `label_infection_bacteria_{time,status}`,
       `label_infection_virus_{time,status}`, `label_infection_fungi_{time,status}`
     - 21 Pathogen Groups: `label_grp_{gid}_{time,status}`

Usage:
    python scripts/generate_synthetic_data.py
    python scripts/generate_synthetic_data.py --num-patients 120 --max-fup 360
"""

import argparse
import re
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict, Features, Sequence, Value


# ==============================================================================
# 1. PATHOGEN GROUP TAXONOMY (v3.9 — 21 Groups across Bacteria, Virus, Fungi)
#    Recovered from ../aiidkit_mhmmdrz/fusion_pathogens/preprocess/add_group_labels.py
#    and synchronized with src/data/survival_labels.py
# ==============================================================================

PATHOGEN_GROUPS: Dict[str, List[str]] = {
    # --- Bacteria (8 groups) ---
    "anaerobic_bacteria": [
        "Anaerobic bacteria", "Actinomyces",
    ],
    "clostridium": [
        "CDI", "Clostridium sp",
    ],
    "enterobacterales": [
        "E. coli", "Klebsiella sp", "Enterobacter", "Proteus sp",
        "Citrobacter", "Serratia", "Morganella", "Campylobacter",
        "Salmonella sp", "Yersinia pseudotuberculosis", "Other enterobacteria",
    ],
    "gn_nonfermenters": [
        "Pseudomonas aeruginosa", "Other Pseudomonas", "Acinetobacter",
        "Stenotrophomonas", "Burkholderia",
    ],
    "gram_positive": [
        "Enterococcus", "Staph aureus", "MSSA", "MRSA",
        "St. coagulase negative", "Pneumococcus", "Streptococcus sp",
        "Streptococcus  sp", "Other GP bacteria",
    ],
    "other_bacteria": [
        "Other bacteria",
    ],
    "other_gn_bacteria": [
        "Haemophilus influenzae", "Legionella", "Moraxella",
        "Other non-enteric GN bacteria",
    ],
    "other_opportunistic": [
        "Nocardia", "Mycobacteria non-tuberculosis",
        "M. tuberculosis group", "Listeria monocytogenes",
    ],

    # --- Virus (9 groups) ---
    "bkpyv": [
        "BKV",
    ],
    "cmv": [
        "CMV",
    ],
    "enteric_viruses": [
        "Noro", "Rota", "Astrovirus", "Sapovirus", "Enterovirus",
    ],
    "hepatitis_viruses": [
        "HBV", "HCV", "HEV", "HAV",
    ],
    "herpes_viruses": [
        "HSV", "VZV", "EBV", "HHV6", "HHV8",
    ],
    "influenza": [
        "Influenza",
    ],
    "other_resp_viruses": [
        "Rhinovirus", "RSV", "Parainfluenza", "Coronavirus",
        "Metapneumovirus", "Adenovirus", "Other respiratory virus",
    ],
    "other_viruses": [
        "JCV", "Parvo B19", "HPV", "HIV", "MPox", "Other virus",
    ],
    "sars_cov_2": [
        "SARS-CoV-2",
    ],

    # --- Fungi (4 groups) ---
    "aspergillus": [
        "Aspergillus fumigatus", "Aspergillus non-fumigatus",
    ],
    "candida": [
        "Candida albicans", "Candida non albicans",
    ],
    "other_fungi": [
        "Zygomycetes", "Cryptococcus neoformans", "Other fungi",
        "Alternaria sp", "Microsporidia",
    ],
    "pneumocystis": [
        "Pneumocystis sp",
    ],
}

GROUP_TO_CATEGORY: Dict[str, str] = {
    # Bacteria (8)
    "anaerobic_bacteria": "Bacteria",
    "clostridium": "Bacteria",
    "enterobacterales": "Bacteria",
    "gn_nonfermenters": "Bacteria",
    "gram_positive": "Bacteria",
    "other_bacteria": "Bacteria",
    "other_gn_bacteria": "Bacteria",
    "other_opportunistic": "Bacteria",
    # Virus (9)
    "bkpyv": "Virus",
    "cmv": "Virus",
    "enteric_viruses": "Virus",
    "hepatitis_viruses": "Virus",
    "herpes_viruses": "Virus",
    "influenza": "Virus",
    "other_resp_viruses": "Virus",
    "other_viruses": "Virus",
    "sars_cov_2": "Virus",
    # Fungi (4)
    "aspergillus": "Fungi",
    "candida": "Fungi",
    "other_fungi": "Fungi",
    "pneumocystis": "Fungi",
}

CATEGORY_TO_GROUPS: Dict[str, List[str]] = {
    "Bacteria": [g for g, c in GROUP_TO_CATEGORY.items() if c == "Bacteria"],
    "Virus": [g for g, c in GROUP_TO_CATEGORY.items() if c == "Virus"],
    "Fungi": [g for g, c in GROUP_TO_CATEGORY.items() if c == "Fungi"],
}

PARASITE_SPECIES: List[str] = [
    "Toxo", "Cryptosporidium sp", "Giardia lamblia",
    "Echinococcus sp", "Leishmania", "Strongyloides", "Other parasites",
]

SPECIES_TO_GROUP: Dict[str, str] = {
    sp: gid for gid, species_list in PATHOGEN_GROUPS.items() for sp in species_list
}

GENERAL_OUTCOMES = ["infection", "graft_loss", "death"]
CATEGORY_OUTCOMES = ["infection_bacteria", "infection_virus", "infection_fungi"]
GROUP_OUTCOMES = [f"grp_{gid}" for gid in sorted(PATHOGEN_GROUPS.keys())]
BINARY_GROUP_OUTCOMES = [f"infection_grp_{gid}" for gid in sorted(PATHOGEN_GROUPS.keys())]


# ==============================================================================
# 2. CLINICAL KNOWLEDGE BASE & FEATURE PROFILE LOADER
# ==============================================================================

STATIC_ENTITIES = [
    "Patient info", "Donor info", "Transplant info", "Mismatch info",
    "Serology info", "Patient serology", "Immunology", "History",
]

ROUTINE_LAB_ENTITIES = ["Patient vitals", "Patient lab", "Organ lab", "Virology"]

MEDICATION_CATEGORIES: Dict[str, List[str]] = {
    "CNI": ["Tacrolimus", "Cyclosporine"],
    "Antimetabolite": ["Mycophenolate Mofetil", "Mycophenolic Acid", "Azathioprine"],
    "Steroids": ["Glucocorticoid", "Prednisone", "Methylprednisolone"],
    "mTOR_Inhibitor": ["Everolimus", "Sirolimus"],
    "Induction": ["Basiliximab", "Rabbit ATG", "Rituximab", "IVIG"],
    "Antiviral": ["Valganciclovir", "Valaciclovir"],
    "Antibiotic_Proph": ["Cotrimoxazole", "Atovaquone", "Pentamidine"],
    "Antibiotic_Treat": ["Beta-Lactame", "Quinolone", "Cephalosporin", "Nitrofurantoin"],
    "Antifungal": ["Other drugs"],
    "Antihypertensive": ["Calcium channel blocker", "Beta-blocker", "ACE inhibitor", "Angiotensin receptor blocker"],
    "Diuretic": ["Other drugs"],
    "Lipid_Lowering": ["Statin"],
}

ORDINAL_LEVELS = ["Lowest", "Lower", "Low", "Middle", "High", "Higher", "Highest"]
ORDINAL_PROBS = np.array([0.05, 0.10, 0.20, 0.30, 0.20, 0.10, 0.05], dtype=np.float64)
ORDINAL_LOW = ["Lowest", "Lower", "Low"]
ORDINAL_MID = ["Middle"]
ORDINAL_HIGH = ["High", "Higher", "Highest"]


class FeatureProfileSampler:
    """Samples realistic attribute-value pairs from `teav_feature_profile.csv`."""

    def __init__(self, csv_path: Path, rng: np.random.Generator):
        self.rng = rng
        df = pd.read_csv(csv_path)
        self.tree: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]] = {}
        self.num_stats: Dict[Tuple[str, str], Tuple[float, float, float, float]] = {}

        for _, row in df.iterrows():
            ent = str(row["Entity"]).strip()
            attr = str(row["Attribute"]).strip()
            vtype = str(row["Variable type"]).strip()
            val_str = str(row["Value range / distinct classes (up to 10)"]).strip()

            if ent not in self.tree:
                self.tree[ent] = {}

            if vtype == "Numerical continuous":
                self.tree[ent][attr] = (np.array(ORDINAL_LEVELS), ORDINAL_PROBS.copy())
                m_mean = re.search(r"Mean:\s*([\d.-]+)\s*±\s*([\d.-]+)", val_str)
                m_range = re.search(r"\[P1-P99\]:\s*\[([\d.-]+),\s*([\d.-]+)\]", val_str)
                if m_mean and m_range:
                    mu, sigma = float(m_mean.group(1)), max(0.01, float(m_mean.group(2)))
                    p1, p99 = float(m_range.group(1)), float(m_range.group(2))
                    self.num_stats[(ent, attr)] = (mu, sigma, p1, p99)
                else:
                    self.num_stats[(ent, attr)] = (50.0, 15.0, 5.0, 95.0)
            else:
                matches = re.findall(r"([^,]+?)\s*\(\d+,\s*([\d.]+)%\)", val_str)
                if matches:
                    vals = np.array([m[0].strip() for m in matches])
                    probs = np.array([float(m[1]) for m in matches], dtype=np.float64)
                    prob_sum = probs.sum()
                    if prob_sum > 0:
                        probs = probs / prob_sum
                    else:
                        probs = np.ones(len(vals), dtype=np.float64) / len(vals)
                    self.tree[ent][attr] = (vals, probs)
                else:
                    self.tree[ent][attr] = (np.array(["0", "1"]), np.array([0.8, 0.2]))

        # Augment Infection -> Pathogen with all 21 pathogen group species
        if "Infection" in self.tree:
            all_species = [sp for g in PATHOGEN_GROUPS.values() for sp in g] + PARASITE_SPECIES
            self.tree["Infection"]["Pathogen"] = (
                np.array(all_species),
                np.ones(len(all_species), dtype=np.float64) / len(all_species),
            )

    def get_attributes(self, entity: str) -> List[str]:
        return list(self.tree.get(entity, {}).keys())

    def sample_value(self, entity: str, attribute: str, shift_sigma: float = 0.0) -> Tuple[str, str]:
        """Returns (raw_value_str, binned_value_str)."""
        if entity not in self.tree or attribute not in self.tree[entity]:
            return "Unknown", "Unknown"
        if (entity, attribute) in self.num_stats:
            mu, sigma, p1, p99 = self.num_stats[(entity, attribute)]
            z = float(self.rng.normal(shift_sigma, 1.0))
            v = float(np.clip(mu + z * sigma, p1, p99))
            b_idx = int(np.clip(int((z + 2.5) / 5.0 * len(ORDINAL_LEVELS)), 0, len(ORDINAL_LEVELS) - 1))
            return f"{v:.2f}", ORDINAL_LEVELS[b_idx]
        vals, probs = self.tree[entity][attribute]
        chosen = str(self.rng.choice(vals, p=probs))
        return chosen, chosen


# ==============================================================================
# 3. PATIENT TRAJECTORY & EVENT SIMULATION
# ==============================================================================

def _append_event(
    seq: Dict[str, list],
    time: int,
    entity: str,
    attribute: str,
    value: str | Tuple[str, str],
):
    seq["time"].append(int(time))
    seq["entity"].append(str(entity))
    seq["attribute"].append(str(attribute))
    if isinstance(value, tuple):
        raw_v, bin_v = value
    else:
        raw_v, bin_v = str(value), str(value)
    seq["value"].append(str(raw_v))
    seq["value_binned"].append(str(bin_v))


def simulate_patient(
    patient_idx: int,
    sampler: FeatureProfileSampler,
    rng: np.random.Generator,
    max_fup: int = 360,
    max_horizon: int = 360,
) -> dict:
    """
    Simulate a complete longitudinal patient record up to `max_sim_day`,
    including latent risk factors, EAV event history, tabular summaries,
    and ground-truth event timestamps for all general, category, and
    21 pathogen-group endpoints.
    """
    pid = str(10001 + patient_idx)
    pkey = f"{pid}_1"

    # Latent risk profile in [0, 1]: high-risk patients have worse baseline/lab markers
    # and higher hazard of post-transplant infections, graft loss, and death.
    latent_risk = float(rng.beta(2.0, 2.5))
    is_high_risk = latent_risk >= 0.55

    # Ensure censoring happens well after max_fup + max_horizon for most patients
    # so binary labels at max_fup are predominantly observed (0 or 1, not -100).
    max_sim_day = max(max_fup + max_horizon + 365, 1800)
    censor_day = int(rng.integers(max_fup + max_horizon + 30, max_sim_day + 1))

    seq = {"time": [], "entity": [], "attribute": [], "value": [], "value_binned": []}

    # --- 1. Static / Baseline Attributes at time = 0 ---
    pat_age = int(np.clip(rng.normal(58 if is_high_risk else 46, 12), 18, 82))
    pat_sex = "Male" if rng.random() < 0.62 else "Female"
    don_age = int(np.clip(rng.normal(56 if is_high_risk else 44, 14), 18, 80))
    don_type = "DBD" if rng.random() < (0.75 if is_high_risk else 0.45) else "Living related"
    tx_organ = str(rng.choice(["Kidney tpx", "Liver", "Heart tpx", "Lung tpx"], p=[0.65, 0.18, 0.09, 0.08]))
    cold_ischemia_min = float(np.clip(rng.normal(720 if don_type == "DBD" else 180, 150), 30, 1800))
    hla_mismatch = int(rng.integers(3, 7) if is_high_risk else rng.integers(0, 4))
    cmv_match = "D+/R-" if (is_high_risk and rng.random() < 0.55) else str(rng.choice(["D+/R+", "D-/R+", "D-/R-"]))
    ebv_match = "D+/R-" if (is_high_risk and rng.random() < 0.35) else "D+/R+"

    for ent in STATIC_ENTITIES:
        for attr in sampler.get_attributes(ent):
            val = sampler.sample_value(ent, attr)
            # Inject structured clinical signal into baseline tokens
            if ent == "Patient info" and "sex" in attr.lower():
                val = pat_sex
            elif ent == "Donor info" and "type" in attr.lower():
                val = don_type
            elif ent == "Transplant info" and "listing organ" in attr.lower():
                val = tx_organ
            elif ent == "Serology info" and attr == "CMV":
                val = cmv_match
            elif ent == "Serology info" and attr == "EBV":
                val = ebv_match
            _append_event(seq, 0, ent, attr, val)

    # --- 2. Longitudinal Routine Visits (Vitals & Labs) ---
    visit_days = sorted(set(
        [1, 3, 7, 14, 21, 30, 45, 60, 75, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360]
        + list(range(420, censor_day + 1, 90))
    ))
    visit_days = [d for d in visit_days if d <= censor_day]

    # Store numeric lab/vital trajectories for Classic ML aggregation
    numeric_trajectories: List[Dict[str, float]] = []

    for d in visit_days:
        # Simulate underlying numeric values correlated with latent_risk
        creat = float(np.clip(rng.normal(190 if is_high_risk else 115, 35), 50, 600))
        egfr = float(np.clip(rng.normal(38 if is_high_risk else 68, 15), 8, 120))
        crp = float(np.clip(rng.exponential(28 if is_high_risk else 6), 0.5, 250))
        leuk = float(np.clip(rng.normal(11.5 if is_high_risk else 6.8, 2.5), 1.5, 30.0))
        hgb = float(np.clip(rng.normal(102 if is_high_risk else 128, 14), 65, 175))
        plt_cnt = float(np.clip(rng.normal(180 if is_high_risk else 240, 55), 30, 550))
        alb = float(np.clip(rng.normal(33 if is_high_risk else 41, 4), 20, 52))
        bmi = float(np.clip(rng.normal(27.5 if is_high_risk else 24.5, 4.0), 16.0, 42.0))
        sbp = float(np.clip(rng.normal(142 if is_high_risk else 126, 14), 90, 200))
        hr = float(np.clip(rng.normal(84 if is_high_risk else 72, 10), 50, 130))

        numeric_trajectories.append({
            "day": float(d),
            "Creatinine": creat,
            "eGFR": egfr,
            "CRP": crp,
            "Leukocytes": leuk,
            "Hemoglobin": hgb,
            "Platelets": plt_cnt,
            "Albumin": alb,
            "BMI": bmi,
            "Systolic_BP": sbp,
            "Heart_Rate": hr,
        })

        for ent in ROUTINE_LAB_ENTITIES:
            attrs = sampler.get_attributes(ent)
            if not attrs:
                continue
            k = min(len(attrs), max(2, int(len(attrs) * 0.5)))
            chosen_attrs = rng.choice(attrs, size=k, replace=False)
            for attr in chosen_attrs:
                attr_low = attr.lower()
                shift = 0.0
                if "creatinine" in attr_low or "crp" in attr_low or "leukocyte" in attr_low:
                    shift = 0.9 if is_high_risk else -0.8
                elif "egfr" in attr_low or "hemoglobin" in attr_low or "albumin" in attr_low:
                    shift = -0.9 if is_high_risk else 0.8
                val = sampler.sample_value(ent, attr, shift_sigma=shift)
                _append_event(seq, d, ent, attr, val)

    # --- 3. Medications (Maintenance + Prophylaxis + Acute Courses) ---
    med_records: List[Tuple[int, int, str, str]] = []  # (start_day, stop_day, cat, drug)
    for cat in ["CNI", "Antimetabolite", "Steroids", "Antibiotic_Proph", "Antiviral"]:
        drug = str(rng.choice(MEDICATION_CATEGORIES[cat]))
        start_d = int(rng.integers(0, 5))
        is_proph = cat in ("Antibiotic_Proph", "Antiviral")
        stop_d = int(rng.integers(90, 360)) if is_proph else censor_day
        med_records.append((start_d, stop_d, cat, drug))
        start_attr = "Start (infection prophylaxis)" if is_proph else "Start (immunosuppression)"
        stop_attr = "Stop (infection prophylaxis)" if is_proph else "Stop (immunosuppression)"
        _append_event(seq, start_d, "Medication", start_attr, drug)
        if stop_d < censor_day:
            _append_event(seq, stop_d, "Medication", stop_attr, drug)

    if is_high_risk and rng.random() < 0.70:
        ind_drug = str(rng.choice(MEDICATION_CATEGORIES["Induction"]))
        med_records.append((0, 7, "Induction", ind_drug))
        _append_event(seq, 0, "Medication", "Start (induction)", ind_drug)
        _append_event(seq, 7, "Medication", "Stop (induction)", ind_drug)

    # --- 4. Biopsy / Rejection / Comorbidity Events ---
    biopsy_days: List[int] = []
    rejection_days: List[int] = []
    n_biopsies = int(rng.poisson(1.8 if is_high_risk else 0.5))
    for _ in range(n_biopsies):
        b_day = int(rng.integers(5, min(censor_day, max_fup + 180)))
        biopsy_days.append(b_day)
        for attr in sampler.get_attributes("Biopsy"):
            val = sampler.sample_value("Biopsy", attr)
            if "rejection" in attr.lower() and is_high_risk:
                val = "Biops proven rj"
                rejection_days.append(b_day)
            _append_event(seq, b_day, "Biopsy", attr, val)

    if rng.random() < (0.65 if is_high_risk else 0.25):
        c_day = int(rng.integers(1, min(censor_day, max_fup + 90)))
        for attr in sampler.get_attributes("Comorbidity"):
            _append_event(seq, c_day, "Comorbidity", attr, sampler.sample_value("Comorbidity", attr))

    # --- 5. Infection Events Across Categories & All 21 Pathogen Groups ---
    # We record every infection event as (day, category, group_id, species, clinically_sig)
    infection_events: List[Dict[str, object]] = []
    all_group_ids = sorted(PATHOGEN_GROUPS.keys())

    # Base number of infection episodes over the follow-up timeline
    n_infections = int(rng.poisson(5.5 if is_high_risk else 1.8))
    # Guarantee that across the cohort, all 21 pathogen groups are well represented
    preferred_groups = [
        all_group_ids[(patient_idx * 3 + k) % len(all_group_ids)]
        for k in range(3)
    ]

    candidate_landmarks = [0, 30, 60, 90, 180, 360]
    for ep_idx in range(n_infections):
        if ep_idx < len(preferred_groups) and rng.random() < 0.75:
            gid = preferred_groups[ep_idx]
        else:
            gid = str(rng.choice(all_group_ids))
        cat = GROUP_TO_CATEGORY[gid]
        species = str(rng.choice(PATHOGEN_GROUPS[gid]))

        # Place events inside clinically relevant windows (e.g., fup + 5..85 days)
        anchor = int(rng.choice(candidate_landmarks))
        offset = int(rng.integers(4, 88))
        inf_day = min(censor_day - 1, max(1, anchor + offset))

        clin_sig = "True" if (cat != "Virus" or rng.random() < 0.85) else "False"
        location = str(rng.choice(["Urinary tract", "RT", "Blood", "GI", "Mucocutaneous"]))
        effect = str(rng.choice(["Proven disease", "Probable disease", "Viral syndrome"]))

        infection_events.append({
            "day": inf_day,
            "category": cat,
            "group_id": gid,
            "species": species,
            "clinically_significant": clin_sig == "True",
        })

        _append_event(seq, inf_day, "Infection", "Category", cat)
        _append_event(seq, inf_day, "Infection", "Pathogen", species)
        _append_event(seq, inf_day, "Infection", "Site", location)
        _append_event(seq, inf_day, "Infection", "Type", effect)
        _append_event(seq, inf_day, "Infection", "Clinically significant", clin_sig)

        # Trigger corresponding anti-infective treatment
        treat_cat = (
            "Antibiotic_Treat" if cat == "Bacteria"
            else ("Antiviral" if cat == "Virus" else "Antifungal")
        )
        treat_drug = str(rng.choice(MEDICATION_CATEGORIES[treat_cat]))
        _append_event(seq, inf_day, "Medication", "Start (infection prophylaxis)", treat_drug)
        if inf_day + 14 < censor_day:
            _append_event(seq, inf_day + 14, "Medication", "Stop (infection prophylaxis)", treat_drug)

    # Occasional parasite event for vocabulary completeness
    if rng.random() < 0.12:
        p_day = int(rng.integers(10, min(censor_day, max_fup + 60)))
        p_sp = str(rng.choice(PARASITE_SPECIES))
        _append_event(seq, p_day, "Infection", "Category", "Parasite")
        _append_event(seq, p_day, "Infection", "Pathogen", p_sp)
        _append_event(seq, p_day, "Infection", "Clinically significant", "True")
        infection_events.append({
            "day": p_day,
            "category": "Parasite",
            "group_id": None,
            "species": p_sp,
            "clinically_significant": True,
        })

    # --- 6. Terminal / Competing Outcomes: Graft Loss & Death ---
    graft_loss_day: Optional[int] = None
    death_day: Optional[int] = None

    if rng.random() < (0.28 if is_high_risk else 0.08):
        anchor = int(rng.choice(candidate_landmarks))
        graft_loss_day = min(censor_day - 2, anchor + int(rng.integers(10, 85)))
        for attr in sampler.get_attributes("Graft Loss"):
            _append_event(seq, graft_loss_day, "Graft Loss", attr, sampler.sample_value("Graft Loss", attr))

    if rng.random() < (0.24 if is_high_risk else 0.06):
        anchor = int(rng.choice(candidate_landmarks))
        min_death_d = (graft_loss_day + 5) if (graft_loss_day and rng.random() < 0.5) else (anchor + int(rng.integers(12, 85)))
        if min_death_d < censor_day:
            death_day = int(min_death_d)
            censor_day = death_day
            for attr in sampler.get_attributes("Death"):
                _append_event(seq, death_day, "Death", attr, sampler.sample_value("Death", attr))

    # Filter out any events after effective censor_day and sort chronologically
    order = [
        i for i, t in enumerate(seq["time"]) if t <= censor_day
    ]
    order.sort(key=lambda i: (seq["time"][i], i))
    for k in seq:
        seq[k] = [seq[k][i] for i in order]

    infection_events = [ev for ev in infection_events if int(ev["day"]) <= censor_day]
    infection_events.sort(key=lambda ev: int(ev["day"]))

    return {
        "patientid": pid,
        "patientkey": pkey,
        "patient_idx": patient_idx,
        "latent_risk": latent_risk,
        "is_high_risk": is_high_risk,
        "censor_day": censor_day,
        "graft_loss_day": graft_loss_day,
        "death_day": death_day,
        "infection_events": infection_events,
        "numeric_trajectories": numeric_trajectories,
        "med_records": med_records,
        "biopsy_days": biopsy_days,
        "rejection_days": rejection_days,
        "static_features": {
            "pat_age_at_tx": float(pat_age),
            "pat_sex": 1 if pat_sex == "Male" else 0,
            "don_age": float(don_age),
            "don_type": 1 if don_type == "DBD" else 0,
            "tx_organ": {"Kidney tpx": 0, "Liver": 1, "Heart tpx": 2, "Lung tpx": 3}.get(tx_organ, 0),
            "tx_cold_ischemia_min": cold_ischemia_min,
            "hla_mismatch_total": float(hla_mismatch),
            "sero_cmv_match": {"D-/R-": 0, "D-/R+": 1, "D+/R+": 2, "D+/R-": 3}.get(cmv_match, 0),
            "sero_ebv_match": 1 if ebv_match == "D+/R-" else 0,
        },
        "sequence": seq,
    }


# ==============================================================================
# 4. DYNAMIC LABEL COMPUTATION AT FOLLOW-UP LANDMARK `obs_end = fup`
#    Mirrors `ClinicalTransplant.compute_binary_labels` and
#    `ClinicalTransplant.compute_survival_labels` in `../aiidkit_mhmmdrz/src/clinical_objects.py`
# ==============================================================================

def _collect_future_event_days(patient: dict, fup: int) -> Dict[str, List[int]]:
    """Return sorted future event days (`day > fup`) for every endpoint."""
    censor_day = int(patient["censor_day"])
    future: Dict[str, List[int]] = {
        "infection": [],
        "graft_loss": [],
        "death": [],
        "infection_bacteria": [],
        "infection_virus": [],
        "infection_fungi": [],
    }
    for gid in PATHOGEN_GROUPS:
        future[f"grp_{gid}"] = []

    for ev in patient["infection_events"]:
        d = int(ev["day"])
        if fup < d <= censor_day:
            future["infection"].append(d)
            cat = str(ev["category"])
            if cat == "Bacteria":
                future["infection_bacteria"].append(d)
            elif cat == "Virus":
                future["infection_virus"].append(d)
            elif cat == "Fungi":
                future["infection_fungi"].append(d)
            gid = ev.get("group_id")
            if gid and f"grp_{gid}" in future:
                future[f"grp_{gid}"].append(d)

    gl_day = patient["graft_loss_day"]
    if gl_day is not None and fup < int(gl_day) <= censor_day:
        future["graft_loss"].append(int(gl_day))

    d_day = patient["death_day"]
    if d_day is not None and fup < int(d_day) <= censor_day:
        future["death"].append(int(d_day))

    for k in future:
        future[k] = sorted(set(future[k]))
    return future


def compute_labels_at_fup(
    patient: dict,
    fup: int,
    horizons: List[int],
) -> Dict[str, int]:
    """
    Compute all binary horizon labels and right-censored survival `(time, status)`
    labels relative to landmark `obs_end = fup`.
    """
    censor_day = int(patient["censor_day"])
    death_day = patient["death_day"]
    future_days = _collect_future_event_days(patient, fup)
    labels: Dict[str, int] = {}

    # 1. Binary labels for each horizon H
    # Mapping from binary label task key -> future_days key
    binary_task_map = {
        "infection": "infection",
        "graft_loss": "graft_loss",
        "death": "death",
        "infection_bacteria": "infection_bacteria",
        "infection_virus": "infection_virus",
        "infection_fungi": "infection_fungi",
    }
    for gid in sorted(PATHOGEN_GROUPS.keys()):
        binary_task_map[f"infection_grp_{gid}"] = f"grp_{gid}"

    for task_name, fkey in binary_task_map.items():
        ev_days = future_days[fkey]
        for h in horizons:
            col = f"label_{task_name}_{h:04d}d"
            win_end = fup + h
            has_event = any(fup < d <= win_end for d in ev_days)
            if has_event:
                labels[col] = 1
            elif censor_day >= win_end:
                labels[col] = 0
            elif death_day is not None and fup < int(death_day) <= win_end:
                # Competing death before horizon end counts as known non-event (0) for non-death endpoints
                labels[col] = 0
            else:
                labels[col] = -100

    # 2. Survival labels (time > 0 in days after fup, status in {0, 1})
    survival_outcomes = (
        GENERAL_OUTCOMES
        + CATEGORY_OUTCOMES
        + [f"grp_{gid}" for gid in sorted(PATHOGEN_GROUPS.keys())]
    )
    follow_up_remaining = max(1, censor_day - fup)

    for outcome in survival_outcomes:
        ev_days = future_days[outcome]
        if ev_days:
            first_day = ev_days[0]
            labels[f"label_{outcome}_time"] = max(1, int(first_day - fup))
            labels[f"label_{outcome}_status"] = 1
        else:
            labels[f"label_{outcome}_time"] = int(follow_up_remaining)
            labels[f"label_{outcome}_status"] = 0

    return labels


def ensure_split_label_diversity(
    records: List[dict],
    horizons: List[int],
    rng: np.random.Generator,
) -> None:
    """
    In small synthetic cohorts (e.g. 9 validation / 9 test patients), rare endpoints
    or pathogen groups might randomly have all-0 or all-1 labels in a split slice,
    which causes AUROC / PR-AUC / Platt scaling / Cox NLL to emit warnings or NaN.
    This helper guarantees at least one positive (1) and one negative (0) sample per
    endpoint in `records` when len(records) >= 2.
    """
    if len(records) < 2:
        return

    # Sort indices by latent risk so highest-risk patient gets the positive event
    sorted_idx = sorted(
        range(len(records)),
        key=lambda i: records[i].get("_latent_risk", 0.5),
        reverse=True,
    )
    hi_idx = sorted_idx[0]
    lo_idx = sorted_idx[-1]

    # Collect all outcome keys (survival + binary horizon)
    survival_outcomes = (
        GENERAL_OUTCOMES
        + CATEGORY_OUTCOMES
        + [f"grp_{gid}" for gid in sorted(PATHOGEN_GROUPS.keys())]
    )

    for outcome in survival_outcomes:
        status_col = f"label_{outcome}_status"
        time_col = f"label_{outcome}_time"
        statuses = [r[status_col] for r in records]
        if 1 not in statuses:
            records[hi_idx][status_col] = 1
            records[hi_idx][time_col] = int(rng.integers(5, 28))
        elif 0 not in statuses:
            records[lo_idx][status_col] = 0
            records[lo_idx][time_col] = max(records[lo_idx][time_col], 180)

        # Corresponding binary prefix
        bin_prefix = (
            f"infection_{outcome}" if outcome.startswith("grp_") else outcome
        )
        for h in sorted(horizons):
            bcol = f"label_{bin_prefix}_{h:04d}d"
            if bcol not in records[0]:
                continue
            valid_vals = [r[bcol] for r in records if r[bcol] in (0, 1)]
            if 1 not in valid_vals:
                records[hi_idx][bcol] = 1
            if 0 not in valid_vals:
                records[lo_idx][bcol] = 0

        # Enforce horizon monotonicity on modified records (30d <= 60d <= 90d ...)
        for idx in (hi_idx, lo_idx):
            running_pos = 0
            for h in sorted(horizons):
                bcol = f"label_{bin_prefix}_{h:04d}d"
                if bcol in records[idx]:
                    if running_pos == 1 and records[idx][bcol] == 0:
                        records[idx][bcol] = 1
                    elif records[idx][bcol] == 1:
                        running_pos = 1


# ==============================================================================
# 5. TABULAR FEATURE AGGREGATION FOR CLASSIC ML (`ClassicMLDatasetBuilder` schema)
# ==============================================================================

def build_classic_ml_row(
    patient: dict,
    fup: int,
    labels: Dict[str, int],
    rng: np.random.Generator,
) -> dict:
    """
    Aggregate patient history up to `obs_end_days = fup` into a flat tabular row
    matching `ClassicMLDatasetBuilder` (`../aiidkit_mhmmdrz/src/builders.py`).
    """
    row: Dict[str, object] = {
        "patientid": patient["patientid"],
        "patientkey": patient["patientkey"],
        "obs_end_days": int(fup),
    }

    # 1. Static features
    row.update(patient["static_features"])

    # 2. Lab & Vital aggregations up to `fup`
    past_visits = [v for v in patient["numeric_trajectories"] if v["day"] <= fup]
    past_30d = [v for v in past_visits if v["day"] >= max(0, fup - 30)]
    if not past_visits:
        # Fallback to first simulated visit if fup == 0
        past_visits = patient["numeric_trajectories"][:1]
    if not past_30d:
        past_30d = past_visits[-2:]

    lab_vital_names = [
        ("Lab", "Creatinine"),
        ("Lab", "eGFR"),
        ("Lab", "CRP"),
        ("Lab", "Leukocytes"),
        ("Lab", "Hemoglobin"),
        ("Lab", "Platelets"),
        ("Lab", "Albumin"),
        ("Vital", "BMI"),
        ("Vital", "Systolic_BP"),
        ("Vital", "Heart_Rate"),
    ]
    for prefix, name in lab_vital_names:
        vals_all = np.array([v[name] for v in past_visits], dtype=np.float64)
        vals_30d = np.array([v[name] for v in past_30d], dtype=np.float64)
        row[f"{prefix}_{name}_last"] = float(vals_all[-1])
        row[f"{prefix}_{name}_mean_all"] = float(np.mean(vals_all))
        row[f"{prefix}_{name}_min_all"] = float(np.min(vals_all))
        row[f"{prefix}_{name}_max_all"] = float(np.max(vals_all))
        row[f"{prefix}_{name}_std_all"] = float(np.std(vals_all)) if len(vals_all) > 1 else 0.0
        row[f"{prefix}_{name}_mean_30d"] = float(np.mean(vals_30d))
        row[f"{prefix}_{name}_slope_30d"] = (
            float(vals_30d[-1] - vals_30d[0]) / max(1.0, float(len(vals_30d)))
            if len(vals_30d) > 1 else 0.0
        )

    # 3. Medication active status & exposure counts up to `fup`
    for cat, drugs in MEDICATION_CATEGORIES.items():
        active = any(
            start_d <= fup <= stop_d and m_cat == cat
            for start_d, stop_d, m_cat, _ in patient["med_records"]
        )
        ever = any(
            start_d <= fup and m_cat == cat
            for start_d, stop_d, m_cat, _ in patient["med_records"]
        )
        row[f"MedCat_{cat}_active"] = int(active)
        row[f"MedCat_{cat}_ever"] = int(ever)

    # 4. Prior infection history up to `fup`
    past_infs = [ev for ev in patient["infection_events"] if int(ev["day"]) <= fup]
    row["Inf_prior_total_count"] = int(len(past_infs))
    for cat in ["Bacteria", "Virus", "Fungi", "Parasite"]:
        cat_infs = [ev for ev in past_infs if ev["category"] == cat]
        row[f"Inf_prior_{cat}_count"] = int(len(cat_infs))
        row[f"Inf_prior_{cat}_30d"] = int(any(int(ev["day"]) >= fup - 30 for ev in cat_infs))

    if past_infs:
        last_inf_day = max(int(ev["day"]) for ev in past_infs)
        row["Inf_days_since_last"] = float(fup - last_inf_day)
    else:
        row["Inf_days_since_last"] = float(fup + 365.0)

    # 5. Prior biopsy & rejection counts up to `fup`
    row["Biopsy_prior_count"] = int(sum(1 for d in patient["biopsy_days"] if d <= fup))
    row["Rej_prior_count"] = int(sum(1 for d in patient["rejection_days"] if d <= fup))

    # 6. Attach the exact same binary & survival labels as the t-EAV dataset
    row.update(labels)
    return row


# ==============================================================================
# 6. DATASET BUILDERS (t-EAV HuggingFace DatasetDict + Classic ML Parquet)
# ==============================================================================

def build_hf_features(sample_labels: Dict[str, int]) -> Features:
    """Construct the exact HuggingFace `Features` schema expected by `patient_dataset.py`."""
    feature_dict = {
        "patientid": Value("string"),
        "patientkey": Value("string"),
        "time": Sequence(Value("int32")),
        "entity": Sequence(Value("string")),
        "attribute": Sequence(Value("string")),
        "value": Sequence(Value("string")),
        "value_binned": Sequence(Value("string")),
    }
    for k in sorted(sample_labels.keys()):
        feature_dict[k] = Value("int64")
    return Features(feature_dict)


def generate_all_datasets(
    cohort: List[dict],
    output_dir: Path,
    fups: List[int],
    horizons: List[int],
    generate_classic_ml: bool,
    seed: int,
) -> None:
    """
    Generate both `teav` and `classic_ml` synthetic datasets across all 3 split
    strategies (`random_split`, `center_split`, `temporal_split`) and all follow-up
    landmarks (`fups`).
    """
    teav_root = output_dir / "teav"
    classic_root = output_dir / "classic_ml"
    if teav_root.exists():
        shutil.rmtree(teav_root, ignore_errors=True)
    if generate_classic_ml and classic_root.exists():
        shutil.rmtree(classic_root, ignore_errors=True)

    n_total = len(cohort)
    n_train = max(1, int(0.70 * n_total))
    n_val = max(1, int(0.15 * n_total))

    # Construct 3 distinct split permutations of the same cohort
    rng_split = np.random.default_rng(seed + 101)
    idx_all = np.arange(n_total)

    perm_random = rng_split.permutation(idx_all)
    perm_center = np.roll(perm_random, shift=max(1, n_total // 4))
    perm_temporal = idx_all.copy()  # ordered by simulated transplant index

    split_indices_map = {
        "random_split": {
            "train": perm_random[:n_train],
            "validation": perm_random[n_train:n_train + n_val],
            "test": perm_random[n_train + n_val:],
        },
        "center_split": {
            "train": perm_center[:n_train],
            "validation": perm_center[n_train:n_train + n_val],
            "test": perm_center[n_train + n_val:],
        },
        "temporal_split": {
            "train": perm_temporal[:n_train],
            "validation": perm_temporal[n_train:n_train + n_val],
            "test": perm_temporal[n_train + n_val:],
        },
    }

    sample_labels = compute_labels_at_fup(cohort[0], fups[0], horizons)
    hf_features = build_hf_features(sample_labels)

    for split_type, part_indices in split_indices_map.items():
        print(f"  -> Building split strategy: {split_type}")
        cache_dir = teav_root / split_type / "processed_cache"
        if cache_dir.exists():
            shutil.rmtree(cache_dir, ignore_errors=True)

        for fup in [*fups, None]:
            fup_name = f"fup_{fup:04d}" if fup is not None else "fup_None"
            teav_ds_dict = {}
            classic_dfs = {}

            for part_name, indices in part_indices.items():
                rng_part = np.random.default_rng(seed + hash((split_type, fup, part_name)) % 100000)
                teav_records: List[dict] = []
                classic_rows: List[dict] = []

                for idx in indices:
                    p = cohort[int(idx)]
                    # Keep patient if still under observation at `fup`
                    if fup is not None and int(p["censor_day"]) <= fup and len(indices) > 4:
                        continue

                    # Truncate EAV sequence to `time <= fup` (or keep full sequence for fup_None)
                    seq = p["sequence"]
                    if fup is not None:
                        valid_pos = [i for i, t in enumerate(seq["time"]) if t <= fup]
                        trunc_seq = {
                            k: [seq[k][i] for i in valid_pos]
                            for k in ("time", "entity", "attribute", "value", "value_binned")
                        }
                    else:
                        trunc_seq = {k: list(seq[k]) for k in ("time", "entity", "attribute", "value", "value_binned")}

                    labels = compute_labels_at_fup(p, fup if fup is not None else 0, horizons)

                    rec = {
                        "patientid": p["patientid"],
                        "patientkey": p["patientkey"],
                        "_latent_risk": p["latent_risk"],
                        **trunc_seq,
                        **labels,
                    }
                    teav_records.append(rec)

                # Ensure every split/fup slice has at least 1 positive and 1 negative per label
                ensure_split_label_diversity(teav_records, horizons, rng_part)

                # Build matching Classic ML rows using the exact same post-diversity labels
                idx_by_pid = {cohort[int(i)]["patientid"]: cohort[int(i)] for i in indices}
                for rec in teav_records:
                    latent_r = rec.pop("_latent_risk", 0.5)
                    p_orig = idx_by_pid[rec["patientid"]]
                    rec_labels = {k: v for k, v in rec.items() if k.startswith("label_")}
                    if generate_classic_ml and fup is not None:
                        c_row = build_classic_ml_row(p_orig, fup, rec_labels, rng_part)
                        classic_rows.append(c_row)

                # Convert to HuggingFace Dataset
                if teav_records:
                    col_dict = {col: [r[col] for r in teav_records] for col in hf_features.keys()}
                else:
                    col_dict = {col: [] for col in hf_features.keys()}
                teav_ds_dict[part_name] = Dataset.from_dict(col_dict, features=hf_features)

                if generate_classic_ml:
                    classic_dfs[part_name] = pd.DataFrame(classic_rows)

            # Save t-EAV DatasetDict to disk
            teav_out = teav_root / split_type / fup_name
            teav_out.mkdir(parents=True, exist_ok=True)
            DatasetDict(teav_ds_dict).save_to_disk(str(teav_out))

            # Save Classic ML Parquet files to disk
            if generate_classic_ml and fup is not None:
                classic_out = classic_root / split_type / fup_name
                classic_out.mkdir(parents=True, exist_ok=True)
                for part_name, df_part in classic_dfs.items():
                    df_part.to_parquet(classic_out / f"{part_name}.parquet", index=False)


# ==============================================================================
# 7. CLI ENTRYPOINT
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate v3.9 synthetic t-EAV and Classic ML datasets with full pathogen group & survival labels."
    )
    parser.add_argument(
        "--profile-csv",
        type=Path,
        default=Path("data/teav_feature_profile.csv"),
        help="Path to the feature distribution profile CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/synthetic"),
        help="Root output directory for generated synthetic datasets.",
    )
    parser.add_argument(
        "--num-patients",
        type=int,
        default=60,
        help="Total number of synthetic patients to simulate.",
    )
    parser.add_argument(
        "--max-fup",
        type=int,
        default=360,
        help="Maximum follow-up landmark day (e.g. 360).",
    )
    parser.add_argument(
        "--fup-step",
        type=int,
        default=180,
        help="Step size in days between follow-up landmarks (also includes standard landmarks 0,30,60,90,180,360).",
    )
    parser.add_argument(
        "--horizons",
        type=int,
        nargs="+",
        default=[30, 60, 90, 180, 360, 720, 1800],
        help="Prediction horizons in days for binary labels (default: 30, 60, 90, 180, 360, 720, 1800).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--generate-classic-ml",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Generate matching Classic ML tabular Parquet datasets (default: True).",
    )
    args = parser.parse_args()

    if not args.profile_csv.exists():
        raise FileNotFoundError(
            f"Feature profile CSV not found at {args.profile_csv}. "
            "Please run from the repository root."
        )

    rng = np.random.default_rng(args.seed)

    # Build follow-up landmark list: include standard landmarks [0, 30, 60, 90, 180, 360]
    # plus regular grid up to max_fup so both train_all.sh and custom fup ranges work.
    standard_fups = [f for f in [0, 30, 60, 90, 180, 360] if f <= args.max_fup]
    grid_fups = list(range(0, args.max_fup + 1, args.fup_step))
    fups = sorted(set(standard_fups + grid_fups))
    horizons = sorted(set(args.horizons))

    print(f"Loading feature distribution profile from {args.profile_csv}...")
    sampler = FeatureProfileSampler(args.profile_csv, rng=rng)

    print(f"Simulating {args.num_patients} longitudinal synthetic patients (seed={args.seed})...")
    cohort = [
        simulate_patient(
            patient_idx=i,
            sampler=sampler,
            rng=rng,
            max_fup=max(fups),
            max_horizon=max(horizons),
        )
        for i in range(args.num_patients)
    ]

    sample_labels = compute_labels_at_fup(cohort[0], 0, horizons)
    n_bin = sum(1 for k in sample_labels if k.endswith("d"))
    n_surv = sum(1 for k in sample_labels if k.endswith("_status"))
    print(
        f"Label schema per landmark: {n_bin} binary columns across horizons {horizons} "
        f"+ {n_surv} survival outcomes ({n_surv * 2} time/status columns) "
        f"[including all {len(PATHOGEN_GROUPS)} pathogen groups]."
    )

    print(f"Writing datasets to {args.output_dir.resolve()} (fups={fups})...")
    generate_all_datasets(
        cohort=cohort,
        output_dir=args.output_dir,
        fups=fups,
        horizons=horizons,
        generate_classic_ml=args.generate_classic_ml,
        seed=args.seed,
    )
    print(f"Done! Synthetic datasets successfully generated under {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
