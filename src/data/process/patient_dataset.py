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
    cutoff_days_train: int | list[int] | None = None,
    cutoff_days_valid: int | list[int] | None = None,
    overwrite_cache: bool = False,
    **kwargs,
) -> tuple[DatasetDict, dict[str, pd.IntervalIndex], dict[str, int]]:
    """
    Loads, concatenates, and processes patient datasets with caching support.
    """
    # Resolve which subfolders to load for each split
    root_path = Path(data_dir)
    train_folders = _resolve_folders(root_path, cutoff_days_train)
    valid_folders = _resolve_folders(root_path, cutoff_days_valid)

    # Compute the cache name (e.g., "processed_train-90-180_val-90") and paths
    cache_name = _generate_cache_name(train_folders, valid_folders)
    cache_path = root_path / "processed_cache" / cache_name
    bins_path = cache_path / "bin_intervals.pkl"
    vocab_path = cache_path / "vocab.pkl"
    
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

    # Load and concatenate raw data
    print(f"Loading raw data")
    raw_train = [load_from_disk(p)["train"] for p in train_folders]
    raw_valid = [load_from_disk(p)["validation"] for p in valid_folders]
    dataset = DatasetDict({
        "train": concatenate_datasets(raw_train),
        "validation": concatenate_datasets(raw_valid),
    })

    # Bin dataset, compute vocab, and map tokens to unified IDs
    dataset, bin_intervals = bin_values_by_attribute(dataset)
    vocab = get_vocab(dataset, root_path=root_path, **kwargs)
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

    # Save data and metadata to cache
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
    days_list = selection if isinstance(selection, list) else [selection]

    # Scan for available fup folders (to ensure we match existing ones)
    all_fups = sorted([p for p in root_path.iterdir() if p.is_dir() and p.name.startswith("fup_")])
    selected_paths = []
    for day in days_list:
        target_name = f"fup_{day}"
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
    **kwargs
) -> dict[str, int]:
    """
    Collects unique tokens. If 'metadata_cache_key' is provided in kwargs,
    loads the vocabulary from that cached dataset instead of computing it.
    """
    # Load vocabulary using cache key (e.g., to ensure same vocab as pre-training)
    if "metadata_cache_key" in kwargs and root_path is not None:
        meta_cut = kwargs["metadata_cache_key"]
        print(f"Loading reference vocabulary using cutoff: {meta_cut}")
        
        # Resolve the path to the master vocabulary
        ref_t_folders = _resolve_folders(root_path, meta_cut)
        ref_v_folders = _resolve_folders(root_path, meta_cut)
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

    vocab = dict(csts.BASE_VOCAB)
    start_idx = len(csts.BASE_VOCAB)
    for idx, term in enumerate(sorted(list(unique_tokens))):
        vocab[term] = idx + start_idx

    return vocab


def bin_values_by_attribute(
    dataset: DatasetDict,
    bin_labels: list[str]={
        0: "Lowest", 1: "Lower", 2: "Low", 3: "Middle",
        4: "High", 5: "Higher", 6: "Highest",
    },
) -> tuple[DatasetDict, dict[str, np.ndarray]]:
    """
    Post-processing a huggingface dataset dictionary to bin values by quantiles
    computed over each attribute
    """
    # Group training and validation longitudinal sample values by attribute
    train_val_data = concatenate_datasets([dataset["train"], dataset["validation"]])
    values_by_attr = defaultdict(list)
    attribute_types = defaultdict(lambda: "numerical")
    feature_types = {}
    for sample in train_val_data:

        # Fill-in groups by attribute type (either category or value)
        for entity, attribute, value in zip(sample["entity"], sample["attribute"], sample["value"]):
            feature_types[attribute] = entity
            try:
                values_by_attr[attribute].append(float(value))
            except ValueError:  # i.e., if str value is not a number
                values_by_attr[attribute].append(value)
                attribute_types[attribute] = "categorical"

    # Compute bin intervals for continuous data
    bin_intervals: dict[str, pd.IntervalIndex] = {}
    for attribute, values in values_by_attr.items():
        if attribute_types[attribute] == "numerical" and len(set(values)) > 10:
            try:
                binned = pd.qcut(x=values,  q=len(bin_labels))
            except ValueError:  # required for very skewed data, like with many 0.0
                binned = pd.cut(x=values, bins=len(bin_labels))
            bin_intervals[attribute] = binned.categories
        else:  # correcting the type of numerical values with low numerosity
            attribute_types[attribute] = "categorical"

    # Define numerical value binning, given bin intervals
    def bin_sample_values(values: list, attributes: list) -> dict[str, list]:
        values_binned = []
        sorted_labels = [bin_labels[i] for i in sorted(bin_labels.keys())]
        for val, attr in zip(values, attributes):
            processed_val = str(val)

            # Numerical attribute (known from training + validation sets)
            if attr in bin_intervals:
                try:  # normal path: value is within known intervals
                    f_val = float(val)
                    bin_idx = bin_intervals[attr].get_loc(f_val)
                    processed_val = sorted_labels[bin_idx]
                except KeyError:  # outlier path: value outside training range
                    is_high = f_val > bin_intervals[attr].right.max()
                    processed_val = sorted_labels[-1] if is_high else sorted_labels[0]
                except ValueError:  # error path: keep raw string
                    pass

            # Categorical or unknown attribute (e.g., new in test set)
            else:
                try:  # clean up integer-floats (e.g., "1.0" -> "1")
                    processed_val = str(int(float(val)))
                except ValueError:  # keep original string if not a number
                    pass

            values_binned.append(processed_val)

        return {"value_binned": values_binned}

    # Apply the binning to all dataset samples (including test set)
    bin_fn = lambda s: bin_sample_values(s["value"], s["attribute"])
    binned_dataset = dataset.map(
        function=bin_fn,
        desc="Binning values",
        load_from_cache_file=False,
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
