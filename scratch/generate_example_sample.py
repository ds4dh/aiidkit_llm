import csv
import json
import pickle
import yaml
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict

# --- PATHS & CONFIGURATION ---
DATA_DIR = Path("/home/shares/ds4dh/aiidkit_project/data_new/processed/v3.6/teav")
DATA_SPLIT_TYPE = "temporal_split"
CONFIG_PATH = Path("configs/discriminative_training.yaml")
OUTPUT_DIR = Path("web_tool/synthetic_patients")

MAX_SURVIVAL_DAYS = 365  # Focus timeline on the first year post-transplant
ORDERED_PHASES = [
    "Baseline_01_Patient_Info",
    "Baseline_02_Donor",
    "Baseline_03_Mismatch",
    "Baseline_04_Serology",
    "Baseline_05_Transplant_Details",
    "PreTransplant_History",
    "Longitudinal_Medications",
    "Longitudinal_Infections",
    "Longitudinal_Vitals_and_Labs",
    "Longitudinal_Diagnoses_and_Hospitalizations"
]

def determine_clinical_phase(time, entity_str):
    lower_ent = entity_str.lower()
    if time == 0 and any(x in lower_ent for x in ["donor", "mismatch", "serology", "procedure", "baseline", "info"]):
        if "donor" in lower_ent: return "Baseline_02_Donor"
        elif "mismatch" in lower_ent: return "Baseline_03_Mismatch"
        elif "serology" in lower_ent: return "Baseline_04_Serology"
        elif "transplant" in lower_ent or "procedure" in lower_ent: return "Baseline_05_Transplant_Details"
        else: return "Baseline_01_Patient_Info"
    if time < 0:
        return "PreTransplant_History"
    if "med" in lower_ent or "immuno" in lower_ent:
        return "Longitudinal_Medications"
    elif "infection" in lower_ent or "bacteria" in lower_ent or "virus" in lower_ent or "fungal" in lower_ent:
        return "Longitudinal_Infections"
    elif "vital" in lower_ent or "test" in lower_ent or "lab" in lower_ent or "idx" in lower_ent:
        return "Longitudinal_Vitals_and_Labs"
    else:
        return "Longitudinal_Diagnoses_and_Hospitalizations"

def load_profile_data():
    cache_pkl_path = OUTPUT_DIR / "dataset_profile_cache.pkl"
    if cache_pkl_path.exists():
        print(f"--> Loading dataset structural profiles from cache: {cache_pkl_path}")
        with open(cache_pkl_path, "rb") as f:
            return pickle.load(f)
            
    # Fallback to scanning if cache missing
    print("--> Profiles not cached. Compiling distributions from training split...")
    from src.data.patient_dataset import load_hf_data_and_metadata
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)
    
    dataset, _, vocab = load_hf_data_and_metadata(
        data_dir=DATA_DIR / DATA_SPLIT_TYPE, fup_train=None, fup_valid=None, fup_test=None,
        time_mapping=config["data_collator"].get("time_mapping", None),
        eav_mappings=config["data_collator"].get("eav_mappings", None),
    )
    
    real_data = dataset["train"]
    id2token = {v: k for k, v in vocab.items()}
    
    phase_sequence_lengths = defaultdict(list)
    phase_time_deltas = defaultdict(list)
    phase_triplet_counts = defaultdict(Counter)

    for sample in real_data:
        entities, attributes, values, times = sample["entity_id"], sample["attribute_id"], sample["value_id"], sample["days_since_tpx"]
        valid_len = sum(1 for e in entities if e != 0)
        sample_phase_lengths = defaultdict(int)
        last_time_per_phase = {}

        for e, a, v, t in zip(entities[:valid_len], attributes[:valid_len], values[:valid_len], times[:valid_len]):
            e, a, v, t = int(e), int(a), int(v), int(round(float(t)))
            ent_str = id2token.get(e, "[UNK]")
            phase = determine_clinical_phase(t, ent_str)
            
            phase_triplet_counts[phase][(e, a, v)] += 1
            sample_phase_lengths[phase] += 1
            if phase in last_time_per_phase:
                phase_time_deltas[phase].append(t - last_time_per_phase[phase])
            last_time_per_phase[phase] = t
            
        for phase, length in sample_phase_lengths.items():
            phase_sequence_lengths[phase].append(length)

    payload = {
        "vocab": vocab, "id2token": id2token, "phase_triplet_counts": phase_triplet_counts,
        "phase_sequence_lengths": phase_sequence_lengths, "phase_time_deltas": phase_time_deltas
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(cache_pkl_path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    return payload

def generate_patient_trajectory(risk_profile, profile_data):
    id2token = profile_data["id2token"]
    vocab = profile_data["vocab"]
    token2id = profile_data["vocab"]
    phase_triplet_counts = profile_data["phase_triplet_counts"]
    phase_sequence_lengths = profile_data["phase_sequence_lengths"]
    phase_time_deltas = profile_data["phase_time_deltas"]

    # Reverse mapping helper to find exact token combination slots dynamically
    def locate_triplet_ids(ent, attr, val):
        for (e_id, a_id, v_id) in token_lookup_cache:
            if id2token.get(e_id) == ent and id2token.get(a_id) == attr and id2token.get(v_id) == val:
                return (e_id, a_id, v_id)
        # Dynamic fallback construction if specific combination was never seen together
        return (token2id.get(ent, 0), token2id.get(attr, 0), token2id.get(val, 0))

    final_timeline = []

    # Enforced clinical anchors extracted from Cluster Analysis
    anchors = {
        "high": {
            "Patient info": {"Sex": "Female"},
            "Transplant info": {"Delayed graft function": "1", "Graft recovery score": "3", "Hosp. stay duration so far (days)": "Highest", "Previous pregnancy": "1", "Previous transfusion": "1"},
            "Patient vitals": {"Patient BMI (kg/m²)": "Highest", "Patient age (years)": "Higher"},
            "Infections": [
                (12, "Infection", "Type", "Proven disease"),
                (12, "Infection", "Category", "Bacteria"),
                (12, "Infection", "Clinically significant", "True"),
                (12, "Infection", "Site", "Urinary tract"),
                (12, "Infection", "Pathogen", "E. coli"),
                (45, "Infection", "Type", "Proven disease"),
                (45, "Infection", "Category", "Bacteria"),
                (45, "Infection", "Clinically significant", "True"),
                (45, "Infection", "Site", "Urinary tract"),
                (45, "Infection", "Pathogen", "E. coli"),
                (45, "Organ event", "Diagnosis", "Graft pyelonephritis")
            ]
        },
        "medium": {
            "Patient info": {"Sex": "Male"},
            "Transplant info": {"Delayed graft function": "0", "Previous pregnancy": "0"},
            "Infections": []
        },
        "low": {
            "Patient info": {"Age at transplant (years)": "Lowest", "Sex": "Male", "Previous transplant count (any organ)": "0", "Previous transplanted organ count": "0"},
            "Donor info": {"Type": "Living related"},
            "Transplant info": {"Delayed graft function": "0", "Graft recovery score": "0", "Hosp. stay duration so far (days)": "Lowest", "24h Urine Collection": "0", "Previous pregnancy": "0", "Previous transfusion": "0"},
            "Patient vitals": {"Patient age (years)": "Lowest"},
            "Infections": []
        }
    }

    current_profile = anchors[risk_profile]
    current_pre_tx_cursor = -120
    current_post_tx_cursor = 0

    for phase in ORDERED_PHASES:
        if phase not in phase_triplet_counts or not phase_sequence_lengths[phase]:
            continue

        # Extract standard sequence lengths
        phase_len = max(5, int(np.random.choice(phase_sequence_lengths[phase])))
        if phase.startswith("Baseline"):
            phase_len = min(phase_len, 15) # Keep baseline block concise

        triplets_list = list(phase_triplet_counts[phase].keys())
        total_triplets_sum = sum(phase_triplet_counts[phase].values())
        triplet_probs = [count / total_triplets_sum for count in phase_triplet_counts[phase].values()]
        
        # Build raw random block pool
        chosen_idx = np.random.choice(len(triplets_list), size=phase_len, p=triplet_probs)
        phase_triplets = [triplets_list[idx] for idx in chosen_idx]

        # -------------------------------------------------------------
        # BACKGROUND PROBABILISTIC GENERATION & ANCHOR REPLACEMENT
        # -------------------------------------------------------------
        global token_lookup_cache
        token_lookup_cache = triplets_list

        processed_triplets = []
        for (e, a, v) in phase_triplets:
            ent_s, attr_s, val_s = id2token.get(e), id2token.get(a), id2token.get(v)
            
            # Check if this feature has an active hardcoded override constraint
            if ent_s in current_profile and attr_s in current_profile[ent_s]:
                override_val = current_profile[ent_s][attr_s]
                processed_triplets.append(locate_triplet_ids(ent_s, attr_s, override_val))
            else:
                processed_triplets.append((e, a, v))

        # -------------------------------------------------------------
        # CHRONOLOGICAL TIME STEP COMPUTATION
        # -------------------------------------------------------------
        phase_times = []
        if phase.startswith("Baseline"):
            phase_times = [0] * len(processed_triplets)
        elif phase == "PreTransplant_History":
            deltas = np.random.choice(phase_time_deltas[phase], size=len(processed_triplets)) if phase_time_deltas[phase] else [14]*len(processed_triplets)
            for d in deltas:
                current_pre_tx_cursor = min(-1, current_pre_tx_cursor + max(1, int(round(d))))
                phase_times.append(int(current_pre_tx_cursor))
        else:
            deltas = np.random.choice(phase_time_deltas[phase], size=len(processed_triplets)) if phase_time_deltas[phase] else [15]*len(processed_triplets)
            for d in deltas:
                current_post_tx_cursor = min(MAX_SURVIVAL_DAYS, current_post_tx_cursor + max(1, int(round(d))))
                phase_times.append(int(current_post_tx_cursor))

        for t, trip in zip(phase_times, processed_triplets):
            final_timeline.append((t, trip))

    # Append deterministic longitudinal infection risk events directly to the timeline
    for (t_day, ent_s, attr_s, val_s) in current_profile["Infections"]:
        trip_ids = locate_triplet_ids(ent_s, attr_s, val_s)
        final_timeline.append((t_day, trip_ids))

    # Ensure complete strict chronological sequencing across tracking arrays
    final_timeline.sort(key=lambda x: x[0])
    return final_timeline

def write_synthetic_output(filename_prefix, risk_label, profile_data):
    timeline = generate_patient_trajectory(risk_label, profile_data)
    id2token = profile_data["id2token"]
    
    output_path = OUTPUT_DIR / f"{filename_prefix}_risk.csv"
    
    with open(output_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["days_since_tpx", "entity", "attribute", "value_binned"])
        
        for t, (e, a, v) in timeline:
            writer.writerow([
                t,
                id2token.get(e, "[UNK]"),
                id2token.get(a, "[UNK]"),
                id2token.get(v, "[UNK]")
            ])
    print(f"[SUCCESS] Synthetic EAV trajectory saved to: {output_path} ({len(timeline)} records)")

if __name__ == "__main__":
    print("Initializing Clinical Cohort Synthetic Trajectory Simulator...")
    profiles = load_profile_data()
    
    # Generate the 3 target demonstration validation suites
    write_synthetic_output("high", "high", profiles)
    write_synthetic_output("medium", "medium", profiles)
    write_synthetic_output("low", "low", profiles)
    print("\nAll demonstration scripts processed successfully.")