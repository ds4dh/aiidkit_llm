import os
import torch
from datasets import DatasetDict
from functools import partial

from src.data.process.patient_dataset import (
    construct_patient_card,
    generate_patient_card_summaries,
)
import src.constants as constants
csts = constants.ConstantsNamespace()

LABEL_KEYS = [
    "infection_label_binary_any",
    "infection_label_binary_bacterial",
    "infection_label_binary_viral",
    "infection_label_binary_fungal",
    "infection_label_categorical",
    "infection_label_one_hot",
]
CONTEXT_KEYS = ["cutoff", "horizon", "sequence_id"]


def create_infection_datasets(
    patient_sequence_dir: str,
    create_patient_cards: bool=True,
    create_patient_card_summaries: bool=False,
) -> DatasetDict:
    """
    Transform a dataset of full patient sequences into datasets of partial sequences,
    each with corresponding infection prediction labels for multiple horizons
    """
    for prediction_horizon in csts.PREDICTION_HORIZONS:

        # Load the full patient sequence dataset
        patient_dataset = DatasetDict.load_from_disk(patient_sequence_dir)

        # Define the task given the prediction horizon and the cutoff days
        task_creation_fn = partial(
            create_prediction_sequences_batched,
            prediction_horizon=prediction_horizon,
            cutoff_days=csts.CUTOFF_DAYS,
        )

        # Create the task dataset for this prediction horizon
        task_dataset = patient_dataset.map(
            task_creation_fn,
            with_indices=True,  # to keep track of sequence ids
            batched=True,  # important, since each patient creates several samples
            load_from_cache_file=False,
            remove_columns=patient_dataset["train"].column_names,
            desc=f"Generating task sequences for {prediction_horizon} days horizon",
        )

        # Add a structured card for each patient sequence (both raw text and markdown formats)
        if create_patient_cards:
            task_dataset = task_dataset.map(
                function=construct_patient_card,
                desc="Creating patient cards",
                num_proc=min(16, os.cpu_count() - 1),
            )

        # Summarize all patient cards as a new column field
        if create_patient_card_summaries:
            dataset = generate_patient_card_summaries(dataset)

        # Save task dataset in a new directory, next to the one with full patient sequences
        output_dir = os.path.join(
            os.path.dirname(patient_sequence_dir),
            f"task_{prediction_horizon}_days_horizon",
        )
        task_dataset.save_to_disk(output_dir)


def get_one_hot_target(future_inf_types: list[str]) -> list[int]:
    """
    Compute the multi-hot encoded label for future infections
    """
    label = [0] * len(csts.INFECTION_TYPES)
    for i, inf_type in enumerate(csts.INFECTION_TYPES):
        if any(inf_type in s.lower() for s in future_inf_types):
            label[i] = 1

    return label


def get_categorical_target(
    future_inf_times: list[int],
    future_inf_types: list[str],
) -> int:
    """
    Compute the categorical label for the first future infection
    """
    if not future_inf_times:
        return csts.LABEL_CLASSES.index("healthy")

    first_infection_type = min(zip(future_inf_times, future_inf_types))[1].lower()
    for inf_type in csts.INFECTION_TYPES:
        if inf_type in first_infection_type:
            return csts.LABEL_CLASSES.index(inf_type)
    
    return csts.LABEL_CLASSES.index("healthy")  # fallback


def get_binary_target(
    future_types: list[str],
    type_constraint: str|None = None,
) -> int:
    """
    Compute the binary label for the first future infection
    """
    if not future_types:
        return 0
    if type_constraint is None:
        return 1
    for inf_type in future_types:
        if type_constraint.lower() in inf_type.lower():
            return 1
    return 0


def create_partial_sequence_labels(
    infection_events: dict[str, str|int],
    cutoff_day: int,
    prediction_horizon: int,
) -> tuple[list[int], list[list[int]]]:
    """
    Create all labels for a partial sequence given by a cutoff and an horizon
    """
    # Define the prediction window
    prediction_start_day = cutoff_day + 1
    prediction_end_day = cutoff_day + prediction_horizon

    # Find all infections that occur within this future window
    future_times, future_types = [], []
    for time, type in zip(
        infection_events["infection_time"],
        infection_events["infection_type"],
    ):
        if prediction_start_day <= time <= prediction_end_day:
            future_times.append(time)
            future_types.append(type)

    # Build all required labels
    label_dict = {
        "infection_label_binary_any": get_binary_target(future_types),
        "infection_label_binary_bacterial": get_binary_target(future_types, "Bacterial Infection"),
        "infection_label_binary_viral": get_binary_target(future_types, "Viral Infection"),
        "infection_label_binary_fungal": get_binary_target(future_types, "Fungal Infection"),
        "infection_label_categorical": get_categorical_target(future_times, future_types),
        "infection_label_one_hot": get_one_hot_target(future_types),
    }

    # Sanity check
    if not all([label_key in LABEL_KEYS for label_key in label_dict.keys()]):
        raise ValueError("One or more keys in LABEL_KEYS are not defined.")
        
    return label_dict


def create_prediction_sequences(
    patient_sample: dict[str, torch.Tensor],
    sequence_id: int,
    prediction_horizon: int,
    cutoff_days: list[int],
    non_sequence_keys: list[str]=[
        "patient_csv_path", "infection_events",
        "patient_card_markdown", "patient_card_text",
    ],
) -> dict[str, list]:
    """
    Generate multiple training samples (partial sequences with labels) from a single
    patient's full sequence for various cutoff days and prediction horizons.
    """
    # Unpack patient data
    days_since_tpx = patient_sample["days_since_tpx"]
    infection_events = patient_sample["infection_events"]
    max_time = days_since_tpx.max() if len(days_since_tpx) > 0 else 0

    # Initialize utilities for sequence creation
    data_columns = list(patient_sample.keys()) + LABEL_KEYS + CONTEXT_KEYS
    new_samples = {key: [] for key in data_columns}

    # Iterate through different cutoff days
    previous_mask_sum = -1
    for cutoff_day in cutoff_days:

        # If cutoff day is "full", take the last horizon days for label assessment
        if cutoff_day == "full":
            cutoff_day_key = "full"
            cutoff_day = max_time - prediction_horizon
            if cutoff_day <= 0:
                continue  # skipping if unrelated to transplantation
        else:
            cutoff_day_key = str(cutoff_day)
            cutoff_day = int(cutoff_day)

        # Skip samples where there is not enough future data to assess a label
        if cutoff_day + prediction_horizon > max_time:
            continue

        # Create the mask as a tensor
        time_mask = (days_since_tpx <= cutoff_day)
        
        # More efficient check: if the number of True values in the mask hasn't changed,
        # the resulting sequence is the same.
        current_mask_sum = time_mask.sum()
        if current_mask_sum == previous_mask_sum:
            continue
        previous_mask_sum = current_mask_sum

        # Get all labels for this partial sequence (this part remains the same)
        label_dict = create_partial_sequence_labels(
            infection_events=infection_events,
            cutoff_day=cutoff_day,
            prediction_horizon=prediction_horizon,
        )

        # Apply the mask to all relevant tensors
        for key, values in patient_sample.items():
            if key not in non_sequence_keys:
                new_samples[key].append(values[time_mask]) 
            else:
                new_samples[key].append(values)  # non-sequence data

        # Add computed labels and label information as new fields
        for label_key in LABEL_KEYS:
            new_samples[label_key].append(label_dict[label_key])

        # Add context info to keep track from which sequence samples were generated
        new_samples["cutoff"].append(cutoff_day_key)
        new_samples["horizon"].append(prediction_horizon)
        new_samples["sequence_id"].append(sequence_id)

    return new_samples


def create_prediction_sequences_batched(
    patient_batch: dict[str, list],
    sequence_ids: list[int],
    prediction_horizon: int,
    cutoff_days: list[int],
) -> dict[str, list]:
    """
    Processes a batch of patient records to generate prediction windows,
    iterating over each patient to generate multiple training samples
    """
    # Iterate over each patient in the batch
    num_patients = len(patient_batch[next(iter(patient_batch))])
    output_keys = list(patient_batch.keys()) + LABEL_KEYS + CONTEXT_KEYS
    output_batch = {key: [] for key in output_keys}
    for i in range(num_patients):

        # Reconstruct a single patient sample from the batched dictionary
        patient_sample = {key: values[i] for key, values in patient_batch.items()}

        # Generate new samples for this single patient
        new_samples_for_patient = create_prediction_sequences(
            patient_sample=patient_sample,
            sequence_id=sequence_ids[i],
            prediction_horizon=prediction_horizon,
            cutoff_days=cutoff_days,
        )

        # Extend the batch output with the newly generated samples
        for key, value_list in new_samples_for_patient.items():
            output_batch[key].extend(value_list)

    return output_batch


if __name__ == "__main__":
    create_infection_datasets(patient_sequence_dir=csts.HUGGINGFACE_DIR_PATH)