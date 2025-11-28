import os
import numpy as np
import pandas as pd
import torch
from typing import Any
from pathlib import Path
from collections import defaultdict
from vllm import LLM, SamplingParams
from transformers import pipeline
from tqdm.contrib.concurrent import process_map
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict, concatenate_datasets
import src.constants as constants
csts = constants.ConstantsNamespace()


def load_hf_data_and_metadata(
    data_dir: str,
    metadata_dir: str|None=None,
    prediction_horizon: int|None=None,
    cutoff_days_train: list[int]|int|None=None,
    cutoff_days_valid: list[int]|int|None=None,
) -> tuple[DatasetDict, dict[str, dict[int, str]]|None]:
    """
    Load a huggingface dataset from disk, with or without associated metadata
    """
    # Re-set data directory if a prediction horizon is set (labelled dataset)
    if prediction_horizon is not None:

        # Input argument checks
        if not isinstance(prediction_horizon, int):
            raise ValueError("Prediction horizon must be an integer.")
        if prediction_horizon not in csts.PREDICTION_HORIZONS:
            raise ValueError(
                f"Prediction horizon {prediction_horizon} is not supported. "
                f"Supported horizons are: {csts.PREDICTION_HORIZONS}"
            )

        # Set the data directory to the one with infection labels
        horizon_dir = f"task_{prediction_horizon}_days_horizon"
        dataset_dir = os.path.join(os.path.dirname(data_dir), horizon_dir)
    
    # Else, take the full sequence dataset, without labels
    else:
        dataset_dir = data_dir

    # Load the dataset from disk
    dataset = DatasetDict.load_from_disk(dataset_dir)
    bin_intervals = None
    vocabs = None
    if metadata_dir is not None:
        bin_intervals = pd.read_pickle(os.path.join(metadata_dir, "bin_intervals.pkl"))
        vocabs = pd.read_pickle(os.path.join(metadata_dir,"vocabs.pkl"))

    # If prediction_horizon is set, a classification task dataset is used, this means
    # the samples should be filtered according to the dataset to only include samples
    # that are within the cutoff days from the transplantation event. The argument
    # cutoff_days can be list[int|str] or None, i.e., no filtering occurs
    cutoff_days_dict = {
        "train": cutoff_days_train,
        "validation": cutoff_days_valid,
        "test": cutoff_days_valid,
    }
    for split, cutoff_days in cutoff_days_dict.items():

        # None meaning "no filtering", i.e., is equivalent to csts.CUTOFF_DAYS
        if cutoff_days is not None:

            # Input argument checks
            if prediction_horizon is None:
                raise ValueError("Cutoff days can only be set if prediction horizon is also set.")
            if isinstance(cutoff_days, int):
                cutoff_days = [cutoff_days]
            if not all(cd in csts.CUTOFF_DAYS for cd in cutoff_days):
                raise ValueError(
                    f"One or more value in cutoff days {cutoff_days} are not supported. "
                    f"Supported cutoff days are: {csts.CUTOFF_DAYS}"
                )

            # Make sure cutoff_days is a list of str
            cutoff_days = [str(cd) for cd in cutoff_days]

            # Filter the dataset to only include samples with the specified cutoff days
            dataset[split] = dataset[split].filter(
                function=lambda sample: sample["cutoff"] in cutoff_days,
                desc=f"Filtering samples with cutoff day in {cutoff_days}",
                num_proc=min(16, os.cpu_count() - 1),
            )

    return dataset, bin_intervals, vocabs


def build_huggingface_patient_dataset(
    input_data_dir: str,
    output_data_dir: str,
    metadata_dir: str,
    create_patient_cards: bool=True,
    create_patient_card_summaries: bool=False,
) -> None:
    """
    Build a huggingface dataset from individual patient csv files
    """ 
    # Create list of patient dictionaries read from patient csv files
    patient_csv_paths = [str(p) for p in Path(input_data_dir).rglob("patient_*.csv")]
    data = process_map(
        build_patient_sample, patient_csv_paths, max_workers=8, chunksize=100,
        desc="Reading patient files",
    )

    # Create train, validation, and test splits by patient (70% // 15% // 15%)
    train_data, valtest_data = train_test_split(data, test_size=0.3, random_state=1)
    val_data, test_data = train_test_split(valtest_data, test_size=0.5, random_state=1)

    # Create huggingface dataset dictionary that includes all splits
    dataset = DatasetDict({
        "train": Dataset.from_list(train_data),
        "validation": Dataset.from_list(val_data),
        "test": Dataset.from_list(test_data)
    })

    # Bins values in the dataset, compute vocabularies for all feature types
    dataset, bin_intervals, vocabs = format_patient_dataset(dataset)

    # Add a structured card for each patient (both raw text and markdown formats)
    if create_patient_cards:
        dataset = dataset.map(
            function=construct_patient_card,
            desc="Creating patient cards",
            num_proc=min(16, os.cpu_count() - 1),
        )

    # Summarize all patient cards as a new column field
    if create_patient_card_summaries:
        dataset = generate_patient_card_summaries(dataset)

    # Save everything to disk
    dataset.save_to_disk(output_data_dir)
    os.makedirs(metadata_dir, exist_ok=True)
    if bin_intervals is not None:
        pd.to_pickle(bin_intervals, os.path.join(metadata_dir, "bin_intervals.pkl"))
    if vocabs is not None:
        pd.to_pickle(vocabs, os.path.join(metadata_dir, "vocabs.pkl"))
    print("Huggingface dataset created and saved to disk")


def build_patient_sample(patient_csv_path: str) -> dict[str, np.ndarray|list|str]:
    """
    Function to read a patient csv file to a dictionary of input (entity, attribute,
    value, and time) and infection label events
    - Time is also formatted as the number of days since the first transplante event
    """
    # Load patient data as a pandas dataframe
    patient_df = pd.read_csv(patient_csv_path)

    # Format and extract time
    patient_df["time"] = patient_df["time"].astype(str)  # to avoid type inconsistency with nans
    patient_times = pd.to_datetime(patient_df["time"])

    # Fill-in static values with first transplantation date
    tpx_event_mask = patient_df["attribute"] == "Transplanted organ"
    first_tpx_time = patient_times.loc[tpx_event_mask].iloc[0]
    patient_times = patient_times.fillna(first_tpx_time)

    # Compute the difference of days with the transplantation date and sort events
    days_since_tpx = (patient_times - first_tpx_time).dt.days
    patient_df["days_since_tpx"] = days_since_tpx
    patient_df = patient_df.sort_values("days_since_tpx").reset_index(drop=True)

    # Handle mixed-type and NaN values (types are added later)
    patient_df = patient_df.dropna(subset=["value", "attribute"])
    patient_df["value"] = patient_df["value"].astype(str)

    # Join input features and infection labels into a sample dictionary
    sample = patient_df[["entity", "attribute", "value", "time", "days_since_tpx"]]
    sample = sample.to_dict(orient="list")
    sample.update({"infection_events": get_all_infection_events(patient_df)})
    sample.update({"patient_csv_path": patient_csv_path})

    return sample


def format_patient_dataset(
    dataset: DatasetDict,
) -> tuple[DatasetDict, dict[str, np.ndarray], dict[str, dict[int, str]]]:
    """
    Take in a patient dataset with columns "entity", "attribute", "value",
    compute value bins for categorization, then map all these columns to the
    corresponding ids, using one vocabulary per column
    """
    # Bin numerical values into "value_binned" (keeping if already categorical)
    binned_dataset, bin_intervals = bin_values_by_attribute(dataset)

    # Map the dataset using one vocabulary per column
    vocabs = get_vocabs(binned_dataset)  # compute vocab now that values are binned
    unk_id = csts.BASE_VOCAB["[UNK]"]  # for tokens not from the training data
    binned_dataset = binned_dataset.map(
        lambda s: {
            "entity_id": [vocabs["entity"].get(e, unk_id) for e in s["entity"]],
            "attribute_id": [vocabs["attribute"].get(a, unk_id) for a in s["attribute"]],
            "value_id": [vocabs["value_binned"].get(v, unk_id) for v in s["value_binned"]],
        },
        desc="Mapping tokens to IDs",
        load_from_cache_file=False,
    )

    # Set some features to numpy, for easier handling by the data collators
    # and also when creating infection prediction task datasets
    binned_dataset.set_format(
        type="numpy",
        columns=[
            "entity_id", "attribute_id", "value_id", "days_since_tpx",
            "entity", "attribute", "value", "value_binned", "time",
            "infection_events", "patient_csv_path",
        ],
    )

    return binned_dataset, bin_intervals, vocabs


def get_vocabs(dataset: DatasetDict) -> dict[str, dict[str, int]]:
    """
    Compute vocabularies for entity, attribute, and binned value columns
    """
    # Collect all possible expressions for entities, attributes, and binned values
    vocab_sets = {
        "entity": set(),     # "entity_id": set(),
        "attribute": set(),  # "attribute_id": set(),
        "value_binned": set(),      # "value_id": set(),
    }
    for sample in concatenate_datasets([dataset["train"], dataset["validation"]]):
        for key, vocab_set in vocab_sets.items():
            vocab_set.update(sample[key])
    # vocab_sets["joined"] = set().union(*vocab_sets.values())

    # Build vocabularies by extending the "base vocabulary"
    num_base_tokens = len(csts.BASE_VOCAB)
    vocabs = {key: dict(csts.BASE_VOCAB) for key in vocab_sets.keys()}
    for key, vocab_set in vocab_sets.items():
        for idx, term in enumerate(sorted(list(vocab_set))):
            vocabs[key][term] = idx + num_base_tokens

    return vocabs


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
            except ValueError: # might fail for very skewed data, like with many 0.0
                binned = pd.cut(x=values, bins=len(bin_labels))
            bin_intervals[attribute] = binned.categories
        else:  # correcting the type of numerical values with low numerosity
            attribute_types[attribute] = "categorical"

    # Define numerical value binning, given bin intervals
    def bin_sample_values(values, attributes):
        values_binned = []
        for value, attribute in zip(values, attributes):
            if attribute_types[attribute] == "numerical":
                try: bin_idx = bin_intervals[attribute].get_loc(float(value))
                except KeyError: bin_idx = len(bin_intervals[attribute]) - 1
                values_binned.append(bin_labels[bin_idx])
                
            elif attribute_types[attribute] == "categorical":
                try: category = str(int(float(value)))
                except ValueError: category = value
                values_binned.append(category)
            
        return {"value_binned": values_binned}

    # Apply the binning to all dataset samples (including test set)
    bin_fn = lambda s: bin_sample_values(s["value"], s["attribute"])
    binned_dataset = dataset.map(
        function=bin_fn,
        desc="Binning values",
        load_from_cache_file=False,
    )

    return binned_dataset, bin_intervals


def get_all_infection_events(
    patient_df: pd.DataFrame,
) -> dict[str, list]:
    """
    Extract all clinically significant infection events from a patient
    TODO: MAKE THE CHECK MORE ROBUST TO CODE CHANGE:
          FOR NOW, IT RELIES ON FUNCTION SIGNATURE!!
    """
    clinically_significant_inf_events = patient_df.loc[
        patient_df["entity"].str.contains("infection", case=False)
        & (patient_df["attribute"] == "Clinically significant")
        & (patient_df["value"] == "True")
    ]
    return {
        "infection_time": clinically_significant_inf_events["days_since_tpx"].tolist(),
        "infection_type": clinically_significant_inf_events["entity"].tolist(),
    }


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


def get_summarization_fn_with_vllm(
    llm_model_path: str="Qwen/Qwen3-8B",  # "Qwen/Qwen3-4B-AWQ"
    patient_card_key: str="patient_card_text",
) -> None:
    """
    Generate patient summaries using vLLM for efficient inference, as a new column
    """
    # Initialize the vLLM model
    quantization = "awq_marlin" if "awq" in llm_model_path.lower() else None
    llm = LLM(
        model=llm_model_path,
        trust_remote_code=True,
        quantization=quantization,
        gpu_memory_utilization=0.90,
        max_num_seqs=1,
        swap_space=4.0,
        cpu_offload_gb=4.0,
    )

    # Define sampling parameters
    sampling_params = SamplingParams(
        temperature=0.1,
        top_p=0.9,
        max_tokens=512,
        stop=["\n\n---\n\n"],  # optional stop sequences
    )

    # Define a clear prompt template for the summarization task
    prompt_template = (
        "You are an expert clinical analyst specializing in infectious diseases in "
        "immunocompromised patients. Your task is to process the patient's medical record "
        "and create a short and concise summary of relevant clinical events and features "
        "useful to predict the risk of an imminent bacterial, viral, or fungal infection. "
        "Very important: the summary should not exceed 300 words."
        "\n\n---\nPATIENT RECORD:\n{patient_card}"
        "\n\n---\nPATIENT RECORD SUMMARY:\n"
    )

    # Define a function to summarize patient card texts
    def generate_patient_summaries_batch(batch: dict[str, Any]) -> dict[str, list[str]]:
        patient_cards = batch[patient_card_key]
        prompts = [prompt_template.format(patient_card=card) for card in patient_cards]
        outputs = llm.generate(prompts, sampling_params)
        summaries = [output.outputs[0].text.strip() for output in outputs]
        return {"patient_card_summary": summaries}

    return generate_patient_summaries_batch


def get_summarization_fn_without_vllm(
    llm_model_path: str="Falconsai/medical_summarization",
    patient_card_key: str="patient_card_text",
) -> dict[str, list[str]]:
    """
    Generate summaries for a batch using a transformers summarization pipeline
    Note: this pipeline is already made for summarization, so no prompt template
    """
    # Initialize the summarization pipeline
    summarizer = pipeline(
        task="summarization",
        model=llm_model_path,
        device="cuda" if torch.cuda.is_available() else "cpu",
        trust_remote_code=True,
    )

    # Define a function to summarize patient card texts
    def generate_summaries_pipeline(batch: dict[str, list]) -> dict[str, list[str]]:
        batch_list = [s for s in batch[patient_card_key]]
        outputs = summarizer(
            batch_list, truncation=True, early_stopping=True,
            max_new_tokens=512, min_length=30, num_beams=4, no_repeat_ngram_size=3, 
        )
        summaries = [output["summary_text"].strip() for output in outputs]
        return {"patient_card_summary": summaries}

    return generate_summaries_pipeline


def generate_patient_card_summaries(
    dataset: DatasetDict,
    use_vllm: bool=True,
    inference_batch_size: int=16,
):
    """
    Generate summaries for each patient card in the dataset
    """
    # Define the function that will summarize patient card batches
    if use_vllm:
        summarize_fn = get_summarization_fn_with_vllm()
    else:
        summarize_fn = get_summarization_fn_without_vllm()
    
    # Apply the summarization function to all splits in the dataset
    dataset = dataset.map(
        function=summarize_fn,
        batched=True,
        batch_size=inference_batch_size,
        desc="Summarizing patient cards with LLM",
    )

    return dataset