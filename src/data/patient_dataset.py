import pickle
import numpy as np
import pandas as pd
from typing import Any
from pathlib import Path
from filelock import FileLock
from collections import defaultdict
from datasets import DatasetDict, load_from_disk, concatenate_datasets
import src.constants as constants
csts = constants.ConstantsNamespace()


def load_hf_data_and_metadata(
    data_dir: str,
    fup_train: int | list[int] | None = None,
    fup_valid: int | list[int] | None = None,
    label_key: str | None = None,
    do_undersampling: bool = False,
) -> tuple[DatasetDict, dict[str, pd.IntervalIndex], dict[str, int]]:
    """
    Consumer/creator pattern
    - if `metadata_cache_key` provided (finetuning), load metadata from cache
    - if `metadata_cache_key` not provided (pretraining) compute and save metadata
    """
    # Initialization
    root_path = Path(data_dir)
    metadata_path = root_path / "processed_cache" / "pretraining_metadata"
    is_pretraining = (fup_train is None)
    vocab = None
    bin_intervals = None
    
    # In finetuning mode, load reference metadata
    if not is_pretraining:
        print(f"Finetuning mode. Loading pretraining metadata from {metadata_path}")
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found at {metadata_path}. Run pre-training first.")
        with open(metadata_path / "vocab.pkl", "rb") as f:
            vocab = pickle.load(f)
        with open(metadata_path / "bin_intervals.pkl", "rb") as f:
            bin_intervals = pickle.load(f)
    else:
        print("Pretraining mode. Bins and vocabulary will be computed and saved.")
        
    # Load raw data
    print(f"Loading raw data from disk...")
    dataset = DatasetDict({
        "train": _load_and_tag(
            root_path, fup_train, "train",
            label_key=label_key, do_undersampling=do_undersampling,
        ),
        "validation": _load_and_tag(root_path, fup_valid, "validation"),
    })
    
    # Create bin intervals in pretraining mode, apply existing ones in finetuning mode
    dataset, bin_intervals = bin_values_by_attribute(dataset, existing_bins=bin_intervals)
    if vocab is None:  # in pretraining mode, compute new vocabulary
        base_vocab = {"[PAD]": 0, "[MASK]": 1, "[BOS]": 2, "[EOS]": 3, "[UNK]": 4}
        vocab = get_vocab(dataset, base_vocab=base_vocab)
    
    # Map tokens to IDs using vocabulary
    unk_id = vocab["[UNK]"]
    def map_to_ids(example):
        return {
            "entity_id": [vocab.get(t, unk_id) for t in example["entity"]],
            "attribute_id": [vocab.get(t, unk_id) for t in example["attribute"]],
            "value_id": [vocab.get(t, unk_id) for t in example["value_binned"]],
        }
    dataset = dataset.map(map_to_ids, desc="Mapping tokens to Unified IDs", num_proc=8)

    # Final formatting
    dataset = dataset.map(lambda x: {"days_since_tpx": [0.0 if pd.isna(t) else t for t in x["time"]]})
    all_cols = dataset["train"].column_names
    label_cols = [c for c in all_cols if c.startswith("label_")]
    cols_to_keep = [
        "entity_id", "attribute_id", "value_id", "days_since_tpx",
        "entity", "attribute", "value_binned", "patientid", "fup", *label_cols,
    ]
    available_cols = [c for c in cols_to_keep if c in dataset["train"].column_names]
    dataset.set_format(type="numpy", columns=available_cols)

    # Save metadata only in pretraining mode
    if is_pretraining:
        print(f"Saving new pre-training metadata to: {metadata_path}")
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = metadata_path.parent / "pretraining_metadata.lock"
        with FileLock(lock_path):
            metadata_path.mkdir(parents=True, exist_ok=True)
            with open(metadata_path / "bin_intervals.pkl", "wb") as f: 
                pickle.dump(bin_intervals, f)
            with open(metadata_path / "vocab.pkl", "wb") as f: 
                pickle.dump(vocab, f)

    return dataset, bin_intervals, vocab


def _load_and_tag(
    root_path: Path,
    fup_input: int | list[int] | None,
    split: str,
    label_key: str | None = None,
    do_undersampling: bool = False,
):
    """Loads specific follow-up folders, tags follow-up, concatenates."""
    fups = fup_input if isinstance(fup_input, list) else [fup_input]
    datasets = []
    for fup in fups:
        suffix = f"{fup:04d}" if fup is not None else "None"
        folder_path = root_path / f"fup_{suffix}"
        if not folder_path.exists():
            raise FileNotFoundError(f"Dataset not found: {folder_path}")
        ds = load_from_disk(str(folder_path))[split]
        if do_undersampling: ds = undersample_dataset(ds, label_key)
        
        # Tag samples with fup-integer (use -1 for None, even if not used anyways)
        fup_val = fup if fup is not None else -1
        ds = ds.add_column("fup", [fup_val] * len(ds))
        
        datasets.append(ds)
        
    return concatenate_datasets(datasets)


def get_vocab(dataset: DatasetDict, base_vocab: dict) -> dict[str, int]:
    """Computes vocabulary from current dataset."""
    print("Computing new vocabulary...")
    
    # Combine train and validation for vocab building
    unique_tokens = set()
    for sample in concatenate_datasets([dataset["train"], dataset["validation"]]):
        unique_tokens.update(sample["entity"])
        unique_tokens.update(sample["attribute"])
        unique_tokens.update(sample["value_binned"])

    vocab = dict(base_vocab)
    start_idx = len(vocab)
    for idx, term in enumerate(sorted(list(unique_tokens))):
        vocab[term] = idx + start_idx
    
    return vocab


def bin_values_by_attribute(
    dataset: DatasetDict,
    existing_bins: dict[str, pd.IntervalIndex] | None = None,
    bin_labels: list[str] = None,
) -> tuple[DatasetDict, dict[str, np.ndarray]]:
    """
    Bins numerical values. If existing bins are provided, uses those intervals
    instead of computing new ones (important for finetuning vs pretraining).
    """
    if bin_labels is None:
        bin_labels = ["Lowest", "Lower", "Low", "Middle", "High", "Higher", "Highest"]
        
    # Group training and validation longitudinal sample values by attribute
    train_val_data = concatenate_datasets([dataset["train"], dataset["validation"]])
    values_by_attr = defaultdict(list)
    attribute_types = defaultdict(lambda: "numerical")
    
    # Only scan data if we need to compute bins (existing_bins is None)
    # or if we need to know which attributes are categorical vs numerical for processing
    for sample in train_val_data:
        for entity, attribute, value in zip(sample["entity"], sample["attribute"], sample["value"]):
            try:
                if existing_bins is None: 
                    values_by_attr[attribute].append(float(value))
            except ValueError:
                attribute_types[attribute] = "categorical"

    # Compute or Assign Bin Intervals
    bin_intervals: dict[str, pd.IntervalIndex] = {}
    
    if existing_bins is not None:
        # Fine-tuning path: Use the intervals from pre-training
        bin_intervals = existing_bins
        # We assume attributes in existing_bins are numerical
        for attr in existing_bins:
            attribute_types[attr] = "numerical"
    else:
        # Pre-training path: Compute new intervals
        for attribute, values in values_by_attr.items():
            if attribute_types[attribute] == "numerical" and len(set(values)) > 10:
                try:
                    binned = pd.qcut(x=values, q=len(bin_labels))
                except ValueError:
                    binned = pd.cut(x=values, bins=len(bin_labels))
                bin_intervals[attribute] = binned.categories
            else:
                attribute_types[attribute] = "categorical"

    # Define numerical value binning function
    def bin_sample_values(values: list, attributes: list) -> dict[str, list]:
        values_binned = []
        labels_map = {i: label for i, label in enumerate(bin_labels)}
        
        # Ensure bin_labels map to indices correctly
        for val, attr in zip(values, attributes):
            processed_val = str(val)

            # Numerical attribute with known bins
            if attr in bin_intervals:
                try:
                    f_val = float(val)
                    # Check intervals
                    idx = bin_intervals[attr].get_loc(f_val)
                    processed_val = labels_map[idx]
                except KeyError:
                    # Outlier handling (outside training range)
                    if f_val > bin_intervals[attr].right.max():
                        processed_val = labels_map[len(bin_labels)-1] # Highest
                    else:
                        processed_val = labels_map[0] # Lowest
                except ValueError:
                    pass  # keep original string if conversion fails

            # Categorical or unknown attribute
            else:
                try:
                    processed_val = str(int(float(val)))
                except ValueError:
                    pass
            
            values_binned.append(processed_val)

        return {"value_binned": values_binned}

    # Apply the binning
    binned_dataset = dataset.map(
        function=lambda s: bin_sample_values(s["value"], s["attribute"]),
        desc="Binning values", load_from_cache_file=False,
    )

    return binned_dataset, bin_intervals


def undersample_dataset(dataset, label_key, seed=1234):
    """
    Optimized undersampling using index selection instead of row filtering.
    """
    # Computes positive and negative samples
    labels = np.array(dataset[label_key])
    pos_indices = np.where(labels == 1)[0]
    neg_indices = np.where(labels == 0)[0]
    n_pos, n_neg = len(pos_indices), len(neg_indices)
    if n_pos == n_neg: return dataset  # no undersampling required

    # Identify majority and minority indices
    if n_pos > n_neg:
        majority_indices = pos_indices
        minority_indices = neg_indices
        n_min = n_neg
    else:
        majority_indices = neg_indices
        minority_indices = pos_indices
        n_min = n_pos

    # Undersample shuffled majority indices
    rng = np.random.default_rng(seed)
    selected_majority_indices = rng.choice(majority_indices, size=n_min, replace=False)
    final_indices = np.concatenate([selected_majority_indices, minority_indices])
    rng.shuffle(final_indices)
    
    return dataset.select(final_indices)


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
    Format a dataframe group into a list of markdown strings.
    Input dataframe should have columns: "entity", "attribute", "value"
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
