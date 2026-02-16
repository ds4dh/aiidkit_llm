import os
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Any
from pathlib import Path
from filelock import FileLock
from tqdm import tqdm
from collections import defaultdict
from datasets import (
    Dataset, DatasetDict,
    load_from_disk, concatenate_datasets, disable_caching,
)

disable_caching()
SAFE_NUM_PROCS = 4  # max(1, len(os.sched_getaffinity(0)) - 2)
BIN_LABELS = ["Lowest", "Lower", "Low", "Middle", "High", "Higher", "Highest"]


def load_hf_data_and_metadata(
    data_dir: str,
    sanity_check_output_dir: str | None = None,
    fup_train: int | list[int] | None = None,
    fup_valid: int | list[int] | None = None,
    fup_test: int | list[int] | None = None,
    target_undersampling_ratio: float | None = None,
    label_keys: str | list[str] | None = None,
    time_mapping: dict[str, str]={"days_since_tpx": "time"},
    eav_mappings: dict[str, str]={
        "entity_id": "entity",
        "attribute_id": "attribute",
        "value_id": "value_binned",
    },
) -> tuple[DatasetDict, dict[str, pd.IntervalIndex], dict[str, int]]:
    """ Build data, using a consumer/creator pattern
        - if `metadata_cache_key` not provided (pretraining), compute and save metadata
        - if `metadata_cache_key` provided (finetuning), load metadata from cache
    """
    # Safety checks
    if isinstance(label_keys, str):
        label_keys = [label_keys]     
    if target_undersampling_ratio is not None and not label_keys:
        raise ValueError("Cannot perform undersampling without valid 'label_keys'.")

    # Initialization
    root_path = Path(data_dir)
    metadata_path = root_path / "processed_cache" / "pretraining_metadata"
    is_pretraining = (fup_train is None)
    vocab = None
    bin_intervals = None

    # Load data
    print(f"Loading raw data from disk...")
    dataset = DatasetDict({
        "train": _load_and_tag(
            root_path, fup_train, "train", label_keys=label_keys,
            target_undersampling_ratio=target_undersampling_ratio,
        ),
        "validation": _load_and_tag(root_path, fup_valid, "validation"),
        "test": _load_and_tag(root_path, fup_test, "test"),
    })

    # Metadata: load or compute
    if not is_pretraining:
        print(f"Finetuning mode. Loading pretraining metadata from {metadata_path}")
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found at {metadata_path}. Run pre-training first.")
        with open(metadata_path / "vocab.pkl", "rb") as f:
            vocab = pickle.load(f)
        with open(metadata_path / "bin_intervals.pkl", "rb") as f:
            bin_intervals = pickle.load(f)
    else:
        print("Pretraining mode. Computing statistics and vocabulary...")
        scanned_data = concatenate_datasets([dataset["train"], dataset["validation"]])
        bin_intervals, vocab = scan_and_compute_metadata(scanned_data)
        
        # Save metadata
        print(f"Saving new pre-training metadata to: {metadata_path}")
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        with FileLock(metadata_path.parent / "pretraining_metadata.lock"):
            metadata_path.mkdir(parents=True, exist_ok=True)
            with open(metadata_path / "bin_intervals.pkl", "wb") as f: 
                pickle.dump(bin_intervals, f)
            with open(metadata_path / "vocab.pkl", "wb") as f: 
                pickle.dump(vocab, f)

    # Sanity check, if required
    if sanity_check_output_dir is not None:
        plot_category_distributions(dataset, bin_intervals, output_dir=sanity_check_output_dir)

    # Preprocessing (bin values, map tokens to IDs, format time entries)
    dataset = DatasetDict({
        split: ds.map(
            preprocess_batch, batched=True, batch_size=1000, desc=f"Preprocessing {split} set",
            num_proc=SAFE_NUM_PROCS, load_from_cache_file=False, keep_in_memory=True,
            fn_kwargs={
                "vocab": vocab, "bin_intervals": bin_intervals, "bin_labels": BIN_LABELS,
                "time_mapping": time_mapping, "eav_mappings": eav_mappings,
            },
        )
        for split, ds in dataset.items()
    })

    # Final formatting and filtering
    all_cols = dataset["train"].column_names
    label_cols = [c for c in all_cols if c.startswith("label_")]
    time_key = list(time_mapping.keys())[0]
    teav_cols = list(eav_mappings.keys()) + [time_key] 
    cols_to_keep = [*teav_cols, *label_cols, "patientid", "fup"]
    available_cols = [c for c in cols_to_keep if c in all_cols]
    dataset.set_format(type="numpy", columns=available_cols)

    return dataset, bin_intervals, vocab


def preprocess_batch(batch, vocab, bin_intervals, time_mapping, eav_mappings, bin_labels):
    """
    Module-level function to clean time, bin values, and map tokens to IDs in a single pass.
    """
    unk_id = vocab["[UNK]"]
    labels_map = {i: label for i, label in enumerate(bin_labels)}
    
    # Process time
    time_out_key = list(time_mapping.keys())[0]
    time_in_key = list(time_mapping.values())[0]    
    out_time = []
    for seq in batch[time_in_key]:  # handle time cleaning (with NaN -> 0.0)
        out_time.append([0.0 if (t is None or t != t) else t for t in seq])

    # Process (entity, attribute, value) entries by batch
    ent_in = eav_mappings["entity_id"]      # usually "entity"
    attr_in = eav_mappings["attribute_id"]  # usually "attribute"
    val_in = "value"                        # Raw values
    out_ent_ids, out_attr_ids, out_val_ids = [], [], []
    for entities, attributes, values in zip(batch[ent_in], batch[attr_in], batch[val_in]):
        
        # Map entity and attribute directly
        out_ent_ids.append([vocab.get(e, unk_id) for e in entities])
        out_attr_ids.append([vocab.get(a, unk_id) for a in attributes])
        
        # Map values (more complex: binning -> string -> token ID)
        row_val_ids = []
        for v, attr in zip(values, attributes):
            token_str = str(v)
            
            # Numerical attribute (needs binning)
            if attr in bin_intervals:
                try:
                    f_val = float(v)
                    idx = bin_intervals[attr].get_loc(f_val)
                    token_str = labels_map[idx]
                except (ValueError, KeyError):
                    try:  # handle outliers outside training range
                        if float(v) > bin_intervals[attr].right.max():
                            token_str = labels_map[len(bin_labels)-1]
                        else:
                            token_str = labels_map[0]
                    except ValueError:
                        pass  # keep original string if not a float
            
            # Categorical attribute (clean formatting)
            else:
                try:
                    token_str = str(int(float(v)))  # "1.0" -> "1" for categorical integers
                except ValueError:
                    pass
            
            # Map the resulting token string to an ID
            row_val_ids.append(vocab.get(token_str, unk_id))
            
        out_val_ids.append(row_val_ids)

    # Return dictionary matching the model's expected input format
    return {
        time_out_key: out_time,
        "entity_id": out_ent_ids,
        "attribute_id": out_attr_ids,
        "value_id": out_val_ids
    }


def scan_and_compute_metadata(dataset, attribute_key="attribute", value_key="value"):
    """
    Read-only scan of the dataset to calculate bin intervals and vocabulary.
    """    
    values_by_attr = defaultdict(list)
    unique_tokens = set()
    attribute_types = defaultdict(lambda: "numerical")
    
    # Single pass to collect data
    for sample in tqdm(dataset, desc="Compute dataset metadata"):
        unique_tokens.update(sample["entity"])  # add any entity immediately
        unique_tokens.update(sample["attribute"])  # add any attribute immediately
        
        # Collect values
        for attr, val in zip(sample[attribute_key], sample[value_key]):
            try:
                values_by_attr[attr].append(float(val))
            except ValueError:
                attribute_types[attr] = "categorical"
                # If categorical, clean it and add to vocab candidates
                try:
                    clean_val = str(int(float(val)))
                except ValueError:
                    clean_val = str(val)
                unique_tokens.add(clean_val)

    # Compute bin intervals
    bin_intervals = {}
    for attr, values in values_by_attr.items():
        # If numerical and enough unique values -> bin it
        if attribute_types[attr] == "numerical" and len(set(values)) > 10:
            try:
                values_arr = np.array(values)
                noise = np.random.uniform(-1e-6, 1e-6, size=values_arr.shape)
                binned = pd.qcut(x=values_arr + noise, q=len(BIN_LABELS), duplicates='drop')
            except ValueError:
                binned = pd.cut(x=values, bins=len(BIN_LABELS))
            
            bin_intervals[attr] = binned.categories
        else:
            # Reclassify as categorical if too few values (e.g. {0, 1, 2})
            attribute_types[attr] = "categorical"
            for v in set(values):
                unique_tokens.add(str(int(v)))

    # Build vocabulary (always include special tokens and bin labels)
    base_vocab = {"[PAD]": 0, "[MASK]": 1, "[BOS]": 2, "[EOS]": 3, "[UNK]": 4}
    vocab = dict(base_vocab)
    unique_tokens.update(BIN_LABELS)  # e.g. "Highest", "Low", ...
    start_idx = len(vocab)
    for idx, term in enumerate(sorted(list(unique_tokens))):
        vocab[term] = idx + start_idx

    return bin_intervals, vocab


def _load_and_tag(
    root_path: Path,
    fup_input: int | list[int] | None,
    split: str,
    label_keys: list[str] | None = None,
    target_undersampling_ratio: float | None = None,
):
    """Loads specific follow-up folders, tags follow-up, concatenates in-memory."""
    fups = fup_input if isinstance(fup_input, list) else [fup_input]
    datasets = []
    
    for fup in tqdm(fups, desc=f"Loading {split} data from listed follow-up periods"):
        suffix = f"{fup:04d}" if fup is not None else "None"
        folder_path = root_path / f"fup_{suffix}"
        if not folder_path.exists():
            raise FileNotFoundError(f"Dataset not found: {folder_path}")
        
        # Load the dataset (initially memory-mapped from disk)
        ds = load_from_disk(str(folder_path))[split]
        
        # Soft undersampling, usually only for training data
        if target_undersampling_ratio is not None:
            ds = undersample_dataset(ds, label_keys, target_undersampling_ratio)
            ds = ds.flatten_indices(keep_in_memory=True)
        
        # Tag samples with fup-integer (use -1 for None)
        fup_val = fup if fup is not None else -1
        ds = ds.add_column("fup", [fup_val] * len(ds))        
        datasets.append(ds)
        
    # Concatenation is now a fast memory copy operation
    return concatenate_datasets(datasets) if len(datasets) > 0 else Dataset.from_dict({})


def undersample_dataset(
    dataset: Dataset,
    label_keys: list[str],
    target_ratio: float = 1.0,
    seed: int = 1234,
    balancing_label_key: str | None = None,
):
    """
    Multi-label undersampling.
    If 'balancing_label_key' is provided:
        Undersamples based on that specific key (binary logic).
    Else:
        Undersamples "clean" patients (0 on all labels) to match 
        patients with "at least one event" (1 on any label).
    """
    rng = np.random.default_rng(seed)
    
    # Balance on a specific key
    if balancing_label_key is not None:
        if balancing_label_key not in label_keys:
            raise ValueError(f"Balancing key {balancing_label_key} not in label_keys")
        
        # Standard binary logic
        labels = np.array(dataset[balancing_label_key])
        pos_indices = np.where(labels == 1)[0]
        neg_indices = np.where(labels == 0)[0]
        
    # Clever Union (balance "any Positive" vs "all negative")
    else:
        # Identify "Any Positive" (The valuable minority)
        label_matrix = np.stack([dataset[k] for k in label_keys], axis=1)
        any_positive_mask = (np.max(label_matrix, axis=1) == 1)
        pos_indices = np.where(any_positive_mask)[0]
        neg_indices = np.where(~any_positive_mask)[0]  # all zeros (or -100s)

    # Common undersampling logic
    n_pos = len(pos_indices)
    n_neg = len(neg_indices)
    # print(f"Undersampling: Found {n_pos} any-pos samples and {n_neg} all-neg samples.")
    if n_pos < n_neg:
        majority_indices = neg_indices
        minority_indices = pos_indices
        keep_n = int(n_pos * target_ratio)
    else:
        majority_indices = pos_indices
        minority_indices = neg_indices
        keep_n = int(n_neg * target_ratio)

    # Check if we even need to drop anything
    if len(majority_indices) <= keep_n:
        # print("Majority class is already smaller than target ratio. No undersampling applied.")
        return dataset

    # Select random subset of majority
    selected_majority = rng.choice(majority_indices, size=keep_n, replace=False)
    
    # Combine and shuffle
    final_indices = np.concatenate([selected_majority, minority_indices])
    rng.shuffle(final_indices)
    
    return dataset.select(final_indices, keep_in_memory=True)


def plot_category_distributions(
    dataset: DatasetDict,
    bin_intervals: dict[str, pd.IntervalIndex],
    output_dir: str,
    attribute_key: str = "attribute",
    value_key: str = "value_binned"
):
    """
    Plots the distribution of values for binned attributes
    """
    print(f"Running sanity check: plotting distributions to {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    # We only care about attributes that were actually binned (exist in bin_intervals)
    binned_attrs = set(bin_intervals.keys())
    
    # Initialize counters for the relevant attributes
    counts = defaultdict(lambda: defaultdict(int))
    
    # Aggregate counts from the training set
    for sample in tqdm(dataset["train"], desc="Aggregating plot data"):
        attrs = sample[attribute_key]
        vals = sample[value_key]
        
        # Zip attributes and values to count pairs
        for a, v in zip(attrs, vals):
            if a in binned_attrs:
                counts[a][v] += 1
    
    # Generate and save plots
    for attr, val_counts in tqdm(counts.items(), desc="Generating plots"):

        # Calculate total for proportions
        total = sum(val_counts.values())
        if total == 0:
            continue

        # Plot raw counts
        data_y = [val_counts.get(label, 0) for label in BIN_LABELS]
        plt.figure(figsize=(10, 6))
        sns.barplot(x=BIN_LABELS, y=data_y, palette="viridis")
        
        plt.title(f"Distribution for: {attr}")
        plt.xlabel("Category")
        plt.ylabel("Count")
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Sanitize filename
        safe_name = "".join(c if c.isalnum() or c in (' ', '-', '_') else "_" for c in attr).strip()
        plt.savefig(os.path.join(output_dir, f"{safe_name}.png"))
        plt.close()

    print(f"Sanity check complete. Plots saved to {output_dir}")
    
    
def format_eav_to_tree(df: pd.DataFrame) -> list[str]:
    """
    Format a dataframe group into a list of tree-like strings
    - input df should have columns: "entity", "attribute", "value"
    """
    if not all(col in df.columns for col in ["entity", "attribute", "value"]):
        raise ValueError("Dataframe must contain 'entity', 'attribute', and 'value' columns")
    
    lines = []
    entities = df["entity"].unique()
    for i, entity_name in enumerate(entities):
        is_last_entity = (i == len(entities) - 1)
        entity_prefix = "└──" if is_last_entity else "├──"
        lines.append(f"{entity_prefix} Entity: {entity_name}")

        child_prefix = "   " if is_last_entity else "│  "
        entity_subgroup = df[df["entity"] == entity_name]
        for j, (_, row) in enumerate(entity_subgroup.iterrows()):
            is_last_row = (j == len(entity_subgroup) - 1)
            attr_prefix = "└──" if is_last_row else "├──"
            lines.append(f"{child_prefix}{attr_prefix} {row['attribute']}: {row['value']}")
    
    return lines


def format_eav_to_markdown(df: pd.DataFrame) -> list[str]:
    """
    Format a dataframe group into a list of markdown strings
    """
    if not all(col in df.columns for col in ["entity", "attribute", "value"]):
        raise ValueError("Dataframe must contain 'entity', 'attribute', and 'value' columns")

    lines = []
    for entity_name in df["entity"].unique():
        lines.append(f"### Entity: {entity_name}")  # h3 for each entity
        entity_subgroup = df[df["entity"] == entity_name]
        for _, row in entity_subgroup.iterrows():
            lines.append(f"- **{row['attribute']}**: {row['value']}")  # bullet points for attributes
        lines.append("")  # add a newline for spacing between entities
    
    return lines


def construct_patient_card(
    patient_data: dict[str, Any],
) -> dict[str, str]:
    """
    Construct a hierarchical representation of the patient record, both in
    raw text and markdown formats
    """
    # Create a new dataframe from patient data features
    patient_df = pd.DataFrame({
        "entity": patient_data["entity"],
        "attribute": patient_data["attribute"],
        "value": patient_data["value"],
        "time": patient_data["time"],
    })

    # Fill-in patient card dictionary (keys being both formats)
    patient_card_dict = {}
    for use_markdown in [True, False]:
        
        # Split data into baseline (no time) and timed events
        patient_df["time"] = pd.to_datetime(patient_df["time"], errors="coerce")
        baseline_df = patient_df[patient_df["time"].isnull()]
        timed_df = patient_df.dropna(subset=['time']).sort_values(by="time")

        # Choose formatting function
        format_fn = format_eav_to_markdown if use_markdown else format_eav_to_tree

        # Initiate record text with baseline patient data
        record_text = []
        if not baseline_df.empty:
            baseline_str = "Baseline data"
            if use_markdown: baseline_str = f"## {baseline_str}"
            record_text.append(baseline_str)
            record_text.extend(format_fn(baseline_df))

        # Process and format timed data
        if not timed_df.empty:
            tpx_event_mask = timed_df["attribute"] == "Transplantation event"
            tpx_times = timed_df.loc[tpx_event_mask, "time"].unique()
            first_transplant_date = min(tpx_times) if len(tpx_times) > 0 else None

            # if max_days_after_first_transplant is not None and first_transplant_date:
            #     end_date = first_transplant_date + pd.Timedelta(days=max_days_after_first_transplant)
            #     timed_df = timed_df[timed_df["time"] <= end_date]
            
            # Corrected the duplicated loop here
            for time, time_group in timed_df.groupby("time"):
                time_str = f"Time: {time.strftime('%Y-%m-%d')}"
                if time == first_transplant_date:
                    time_str += " /!\\ TRANSPLANTATION DAY"
                if use_markdown: time_str = f"## {time_str}"
                record_text.append(time_str)
                record_text.extend(format_fn(time_group))
        
        patient_card_key = f"patient_card_{'markdown' if use_markdown else 'text'}"
        patient_card_dict[patient_card_key] = "\n".join(record_text)
    
    return patient_card_dict
