import os
import pandas as pd

from src.data.preprocess.consent_data import Consent, get_patient_consent, get_patients_IDs
from src.data.preprocess.kidney_fup_data import pool_kidney_fup_data
from src.data.preprocess.kidney_bl_data import pool_kidney_bl_data
from src.data.preprocess.patient_bl_data import pool_patient_bl_data
from src.data.preprocess.patient_drug_data import pool_patient_drug_data
from src.data.preprocess.organ_base_data import pool_organ_base_data
from src.data.preprocess.patient_infectious_disease_data import pool_patient_infection_data
from src.data.process.patient_dataset import build_huggingface_patient_dataset
from src.data.process.infection_task import create_infection_datasets
from src.data.process.explore_utils import (
    generate_sex_distribution_plot,
    generate_age_distribution_plot,
    generate_infection_type_plots,
    generate_infection_test_plot,
    generate_survival_analysis_plots,
)

from functools import partial
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

import argparse
import src.constants as constants
csts = constants.ConstantsNamespace()
parser = argparse.ArgumentParser()
parser.add_argument("-d", "--debug", action="store_true", help="Enable debug mode")
parser.add_argument("-e", "--explore", action="store_true", help="Enable exploration mode")
parser.add_argument("-p", "--postprocess", action="store_true", help="Only post-process dataset")
args = parser.parse_args()
DEBUG_FLAG = args.debug
EXPLORE_FLAG = args.explore
ONLY_POST_PROCESS_FLAG = args.postprocess


def main():
    """ Create patient records from the raw STCS data file
        TODO: CHECK HOW THINGS WORK FOR PATIENTS WITH MULTIPLE TRANSPLANTS AND
              CONSIDER USING ORGAN_ID INSTEAD OF / IN COMBINATION WITH PATIENT_ID
    """
    # Create output directory if it does not exist
    preprocessed_dir_path = os.path.dirname(csts.PREPROCESSED_DIR_PATH)
    if not os.path.exists(preprocessed_dir_path):
        os.makedirs(preprocessed_dir_path)

    # Load patient data sheets from the pickle file
    data_dict = pd.read_pickle(csts.PICKLE_DATA_PATH)
    patients_IDs = get_patients_IDs(data_dict[csts.CONSENT_SHEET])

    # Process some patients, using only one process, if DEBUG_FLAG is enabled
    if DEBUG_FLAG:
        for patient_ID in tqdm(patients_IDs[0:1000], "Creating patient records"):
            print(f"Processing patient {patient_ID}")
            create_patient_record(patient_ID, data_dict)
    
    # Allow raw data file exploration, if EXPLORE_FLAG is enabled
    elif EXPLORE_FLAG:
        exploration_fn(data_dict)

    else:
        # Process all patients using multiprocesing to create csv records
        if not ONLY_POST_PROCESS_FLAG:
            num_workers = min(os.cpu_count() - 1, 16)
            chunksize = max(1, len(patients_IDs) // (4 * num_workers))
            pooled_fn = partial(create_patient_record, data_dict=data_dict)
            process_map(
                pooled_fn,
                patients_IDs,
                max_workers=num_workers,
                desc=f"Creating patient records with {num_workers} workers and chunksize {chunksize}",
                chunksize=chunksize,
            )

        # Format the csv files to a huggingface dataset with train / val / test splits
        build_huggingface_patient_dataset(
            input_data_dir=csts.PREPROCESSED_DIR_PATH,
            output_data_dir=csts.HUGGINGFACE_DIR_PATH,
            metadata_dir=csts.METADATA_DIR_PATH,
            create_patient_cards=False,
            create_patient_card_summaries=False,
        )

        # Create infection task datasets
        create_infection_datasets(
            patient_sequence_dir=csts.HUGGINGFACE_DIR_PATH,
            create_patient_cards=True,
            create_patient_card_summaries=False,
        )

    # Success!
    print("All datasets created successfully")


def create_patient_record(
    patient_ID: int,
    data_dict: pd.DataFrame,
) -> pd.DataFrame:
    """ Create a patient record from the raw data
    """
    # Only process patients who have given consent
    consent_status = get_patient_consent(patient_ID, data_dict[csts.CONSENT_SHEET])
    if consent_status == Consent.CONSENT_REFUSED: return None

    # Load patient information from all the sheets
    patient_df = pd.concat([
        pool_patient_bl_data(
            patient_ID,
            data_dict[csts.PATIENT_BL_SHEET],
            data_dict[csts.PATIENT_PSQ_SHEET],
            data_dict[csts.KIDNEY_BL_SHEET],
        ),
        pool_patient_drug_data(
            patient_ID,
            data_dict[csts.PATIENT_DRUG_SHEET],
            csts.START_TIME_IMPUTATION_STRATEGY,
            csts.STOP_TIME_IMPUTATION_STRATEGY,
            csts.OTHER_DRUG_TYPE_MAPPING_STRATEGY,
        ),
        pool_kidney_bl_data(patient_ID, data_dict[csts.KIDNEY_BL_SHEET]),
        pool_kidney_fup_data(patient_ID, data_dict[csts.KIDNEY_FUP_SHEET]),
        pool_organ_base_data(patient_ID, data_dict[csts.ORGAN_BASE_SHEET]),
        # pool_patient_stop_data(patient_ID, data_dict[csts.PATIENT_STOP_SHEET]),
        pool_patient_infection_data(patient_ID, data_dict[csts.PATIENT_INFECTION_SHEET]),
    ], ignore_index=True)

    # Post-process the patient record
    patient_df["time"] = pd.to_datetime(patient_df["time"], errors="coerce")
    patient_df["time"] = patient_df["time"].dt.strftime("%Y-%m-%d")  # time is always a date
    patient_df = patient_df.drop_duplicates(subset=["attribute", "value", "time"])
    patient_df = patient_df.sort_values(by=["time", "entity", "attribute", "value"])

    # Save the patient record to a CSV file
    if 1:  # not DEBUG_FLAG:
        save_path = os.path.join(csts.PREPROCESSED_DIR_PATH, f"patient_{patient_ID}.csv")
        os.makedirs(csts.PREPROCESSED_DIR_PATH, exist_ok=True)
        patient_df.to_csv(save_path, index=False)


def print_index_range(df, field, with_date=True):
    """
    Utils function to explore indexed data fields
    """
    indexed_fields = [c for c in df.columns if field + "_" in c]
    print(indexed_fields)
    print(f"Num indices for {field}: {len(indexed_fields)}")
    if with_date:
        print_index_range(df, field + "date", with_date=False)


def exploration_fn(data_dict: pd.DataFrame):
    """ Explore the raw data file
    """
    pat_cst = data_dict[csts.CONSENT_SHEET]
    pat_bl = data_dict[csts.PATIENT_BL_SHEET]
    pat_psq = data_dict[csts.PATIENT_PSQ_SHEET]
    pat_drg = data_dict[csts.PATIENT_DRUG_SHEET]
    pat_stop = data_dict[csts.PATIENT_STOP_SHEET]
    pat_inf = data_dict[csts.PATIENT_INFECTION_SHEET]
    kid_bl = data_dict[csts.KIDNEY_BL_SHEET]
    kid_fup = data_dict[csts.KIDNEY_FUP_SHEET]
    org_base = data_dict[csts.ORGAN_BASE_SHEET]
    print("To explore: pat_cst, pat_bl, pat_psq, pat_drg, pat_stop, pat_inf, kid_bl, kid_fup, org_base")
    import ipdb; ipdb.set_trace()

    # Generate various distribution plots
    generate_sex_distribution_plot(pat_bl)
    generate_age_distribution_plot(pat_bl, kid_bl)
    generate_infection_type_plots(pat_inf)
    generate_infection_test_plot(pat_inf)

    # Generate various survival analysis plots
    generate_survival_analysis_plots(pat_inf, pat_stop, kid_bl)


if __name__ == "__main__":
    main()
