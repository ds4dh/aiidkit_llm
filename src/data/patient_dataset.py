import shutil
import pickle
import numpy as np
import pandas as pd
from typing import Any
from pathlib import Path
from collections import defaultdict
from datasets import DatasetDict, load_from_disk, concatenate_datasets
import src.constants as constants
csts = constants.ConstantsNamespace()


def load_hf_data_and_metadata(
    data_dir: str,
    fup_train: int | list[int] | None = None,
    fup_valid: int | list[int] | None = None,
    overwrite_cache: bool = False,
    metadata_cache_key: str | None = None,
    **kwargs,
) -> tuple[DatasetDict, dict[str, pd.IntervalIndex], dict[str, int]]:
    """
    Loads, concatenates, and processes patient datasets with caching support.
    Ensures Vocabulary and Binning consistency if metadata_cache_key is provided.
    """
    root_path = Path(data_dir)
    train_folders = _resolve_folders(root_path, fup_train)
    valid_folders = _resolve_folders(root_path, fup_valid)

    # Generate cache name with the reference key to prevent collisions
    # where the same data is processed with different vocabularies/bins
    cache_name = _generate_cache_name(train_folders, valid_folders)
    if metadata_cache_key:
        cache_name = f"{cache_name}_ref-{metadata_cache_key}"

    cache_path = root_path / "processed_cache" / cache_name
    bins_path = cache_path / "bin_intervals.pkl"
    vocab_path = cache_path / "vocab.pkl"
    
    # import ipdb; ipdb.set_trace()

    # Try loading from cache
    if cache_path.exists() and not overwrite_cache:
        print(f"Loading cached dataset from: {cache_path}")
        try:
            dataset = DatasetDict.load_from_disk(str(cache_path))
            with open(bins_path, "rb") as f: bin_intervals = pickle.load(f)
            with open(vocab_path, "rb") as f: vocab = pickle.load(f)
            return dataset, bin_intervals, vocab
        except Exception as e:
            print(f"Cache load failed ({e}), reprocessing raw data")
            shutil.rmtree(cache_path, ignore_errors=True)

    # Load reference data if finetuning
    ref_vocab, ref_bins = None, None
    if metadata_cache_key:
        print(f"Aligning to reference metadata from: {metadata_cache_key}")
        ref_path = root_path / "processed_cache" / metadata_cache_key        
        if not ref_path.exists():
            raise FileNotFoundError(f"Cache not found at {ref_path}. Run pre-training first.")
        with open(ref_path / "vocab.pkl", "rb") as f: ref_vocab = pickle.load(f)
        with open(ref_path / "bin_intervals.pkl", "rb") as f: ref_bins = pickle.load(f)

    # Load and concatenate raw data
    print(f"Loading raw data from disk...")
    raw_train = [load_from_disk(str(p))["train"] for p in train_folders]
    raw_valid = [load_from_disk(str(p))["validation"] for p in valid_folders]
    dataset = DatasetDict({
        "train": concatenate_datasets(raw_train),
        "validation": concatenate_datasets(raw_valid),
    })

    # Bin dataset and compute bins + vocabulary, using reference if provided
    dataset, bin_intervals = bin_values_by_attribute(dataset, existing_bins=ref_bins)
    if ref_vocab:
        print("Using reference vocabulary.")
        vocab = ref_vocab
    else:
        # base_vocab = {v['token']: v['id'] for v in base_vocab_cfg.values()}
        base_vocab = {"[PAD]": 0, "[MASK]": 1, "[BOS]": 2, "[EOS]": 3, "[UNK]": 4}
        vocab = get_vocab(dataset, base_vocab=base_vocab, root_path=root_path, **kwargs)

    # Map to unified IDs
    unk_id = vocab["[UNK]"]
    def map_to_ids(example):
        return {
            "entity_id": [vocab.get(t, unk_id) for t in example["entity"]],
            "attribute_id": [vocab.get(t, unk_id) for t in example["attribute"]],
            "value_id": [vocab.get(t, unk_id) for t in example["value_binned"]],
        }
    dataset = dataset.map(map_to_ids, desc="Mapping tokens to Unified IDs", num_proc=4)

    # Final formatting
    dataset = dataset.map(lambda x: {"days_since_tpx": [0.0 if pd.isna(t) else t for t in x["time"]]})
    all_cols = dataset["train"].column_names
    label_cols = [c for c in all_cols if c.startswith("label_")]
    cols_to_keep = [
        "entity_id", "attribute_id", "value_id", "days_since_tpx",
        "entity", "attribute", "value_binned", "patientid", *label_cols,
    ]
    available_cols = [c for c in cols_to_keep if c in dataset["train"].column_names]
    dataset.set_format(type="numpy", columns=available_cols)

    # Save to cache
    print(f"Saving processed dataset and metadata to: {cache_path}")
    dataset.save_to_disk(str(cache_path))
    with open(bins_path, "wb") as f: pickle.dump(bin_intervals, f)
    with open(vocab_path, "wb") as f: pickle.dump(vocab, f)

    return dataset, bin_intervals, vocab


def _resolve_folders(
    root_path: Path,
    selection: int | list[int] | None,
) -> list[Path]:
    """
    Helper to convert int/list/None into a list of folder paths.
    'None' is treated as a literal suffix -> 'fup_None'.
    """
    # Normalize input to a list (wrap int or None to a list if not a list)
    fup_list = selection if isinstance(selection, list) else [selection]

    # Scan for available fup folders (to ensure we match existing ones)
    all_fups = sorted([p for p in root_path.iterdir() if p.is_dir() and p.name.startswith("fup_")])
    selected_paths = []
    for fups in fup_list:
        suffix = f"{fups:04d}" if fups is not None else "None"
        target_name = f"fup_{suffix}"
        match = next((p for p in all_fups if p.name == target_name), None)
        if match:
            selected_paths.append(match)
        else:
            raise FileNotFoundError(f"Dataset folder '{target_name}' not found in {root_path}")
            
    return selected_paths


def _generate_cache_name(train_paths: list[Path], valid_paths: list[Path]) -> str:
    """Creates a deterministic folder name based on input sources."""
    def get_suffixes(paths):
        # Extracts '90', '180', 'None' from 'fup_90', 'fup_None'
        names = sorted([p.name.replace("fup_", "") for p in paths])
        return "-".join(names) if names else "empty"

    t_sig = get_suffixes(train_paths)
    v_sig = get_suffixes(valid_paths)
    
    # Example result: "processed_train-90-180_val-90"
    return f"processed_train-{t_sig}_val-{v_sig}"


def get_vocab(
    dataset: DatasetDict,
    root_path: Path | None = None,
    base_vocab: dict = {"[PAD]": 0, "[MASK]": 1, "[BOS]": 2, "[EOS]": 3, "[UNK]": 4},
    **kwargs
) -> dict[str, int]:
    """
    Collects unique tokens. If 'metadata_cache_key' is provided in kwargs,
    loads the vocabulary from that cached dataset instead of computing it.
    """
    # Load vocabulary using cache key (e.g., to ensure same vocab as pre-training)
    if "metadata_cache_key" in kwargs and root_path is not None:
        meta_fup = kwargs["metadata_cache_key"]
        print(f"Loading reference vocabulary using follow-up: {meta_fup}")
        
        # Resolve the path to the master vocabulary
        ref_t_folders = _resolve_folders(root_path, meta_fup)
        ref_v_folders = _resolve_folders(root_path, meta_fup)
        ref_cache_name = _generate_cache_name(ref_t_folders, ref_v_folders)
        ref_vocab_path = root_path / "processed_cache" / ref_cache_name / "vocab.pkl"
        
        if not ref_vocab_path.exists():
            raise FileNotFoundError(f"Reference vocab not found at {ref_vocab_path}. Run pretraining loading first.")
            
        with open(ref_vocab_path, "rb") as f:
            return pickle.load(f)

    # Compute vocabulary from current dataset
    print("Computing vocabulary from current dataset.")
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
    Bins numerical values. If existing_bins is provided, uses those intervals 
    instead of computing new ones (CRITICAL for fine-tuning consistency).
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
        # Ensure bin_labels map to indices correctly
        labels_map = {i: label for i, label in enumerate(bin_labels)}
        
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
