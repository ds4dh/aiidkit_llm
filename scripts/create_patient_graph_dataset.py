import os
import pickle
import argparse
from tqdm import tqdm 

# Import internal functions
from src.data.process.patient_dataset import load_hf_data_and_metadata

from src.data.process.graph_patient_dataset import create_HDF5_file_graph_DS, is_float_string

#====================================================================================================#
#====================================================================================================#
#====================================================================================================#
def main():
    #==========================================================================#
    # Construct the argument parser
    ap = argparse.ArgumentParser()
    # Add the arguments to the parser
    ap.add_argument('--store_path', required=True, help="Path where the generated dataset is going to be stored.", type=str)
    ap.add_argument('--huggingface_dir_path', required=True, help="Path to the directory containing the Hugging Face dataset.", type=str)
    ap.add_argument('--metadata_dir_path', required=True, help="Path to the directory containing the Hugging Face dataset metadata.", type=str)
    ap.add_argument('--prediction_horizon', default=30, help="Prediction horizon used to define the label of the samples.", type=int)
    ap.add_argument('--cutoff_days_train', nargs='+', default=[30, 90], help="Train cutoff days.", type=int)
    ap.add_argument('--cutoff_days_valid', nargs='+', default=[30, 90], help="Validation cutoff days.", type=int)
    args = vars(ap.parse_args())

    # Getting the value of the arguments
    store_path = args['store_path']
    huggingface_dir_path = args['huggingface_dir_path']
    metadata_dir_path = args['metadata_dir_path']
    prediction_horizon = args['prediction_horizon']
    cutoff_days_train = args['cutoff_days_train']
    cutoff_days_valid = args['cutoff_days_valid']

    #==========================================================================#
    # Creating folder to store the data
    if (not os.path.exists(f"{store_path}/graph_h5/")):
        os.mkdir(f"{store_path}/graph_h5/")
    store_path = f"{store_path}/graph_h5/"
    print(f"\n=========> Generated HDF5 will be stored in {store_path}\n<=========")

    #==========================================================================#
    # Loading the patients dataset
    pats_dataset, bin_intervals, vocabs = load_hf_data_and_metadata(
                                                                        data_dir=huggingface_dir_path,
                                                                        metadata_dir=metadata_dir_path,
                                                                        prediction_horizon=prediction_horizon,
                                                                        cutoff_days_train=cutoff_days_train,
                                                                        cutoff_days_valid=cutoff_days_valid,
                                                                    )
    

    #==========================================================================#
    # Getting all the possible values for each pair Entity-Attribute in the DS
    # USE ONLY TRAIN dataset
    possible_values_all_patients = {}
    for data_split in ['train']:
    #for data_split in tqdm(pats_dataset):
        entity = pats_dataset[data_split]['entity']
        attribute = pats_dataset[data_split]['attribute']
        value = pats_dataset[data_split]['value']
        times = pats_dataset[data_split]['time']
        for pat_ID in tqdm(range(entity.shape[0])):
            n_vals = entity[pat_ID].shape[0]
            for i in range(n_vals):
            #for i in tqdm(range(n_vals)):
                # Get EAV
                ent = entity[pat_ID][i]
                attr = attribute[pat_ID][i]
                val = value[pat_ID][i]
                time = times[pat_ID][i]
                # Add to dict of entity-attribute values
                if ((ent, attr) not in possible_values_all_patients):
                    possible_values_all_patients[(ent, attr)] = set()
                possible_values_all_patients[(ent, attr)].add(val)

    # Separate numerical variables from continuous ones
    categorical_ent_attr_pairs = {}
    continuous_ent_attr_pairs = {}
    for ent, attr in possible_values_all_patients:
        # Test if continuous: if all possible values are floats
        are_all_floats = True
        for val in possible_values_all_patients[(ent, attr)]:
            # Test if float
            is_float = is_float_string(val)
            if (not is_float):
                are_all_floats = False
                break
        # Separating
        if (are_all_floats):
            continuous_ent_attr_pairs[(ent, attr)] = possible_values_all_patients[(ent, attr)]
        else:
            categorical_ent_attr_pairs[(ent, attr)] = possible_values_all_patients[(ent, attr)]
            
    # Number of continuous and categorical entity-attr pairs
    print(f"\n===> Number of entity-attr pairs with continuous values: {len(continuous_ent_attr_pairs)}")
    print(f"\n===> Number of entity-attr pairs with categorical values: {len(categorical_ent_attr_pairs)}")

    #==========================================================================#
    # Creating HDF5 file        
    for data_split in ["Train", "Validation", "Test"]:
        possible_ent_attr_pairs = list(possible_values_all_patients)
        ids_ent_attr_pairs = {possible_ent_attr_pairs[i]: i for i in range(len(possible_ent_attr_pairs))}
        h5_fn = create_HDF5_file_graph_DS(
                                            store_path=store_path,
                                            patients_dataset=pats_dataset,
                                            ids_ent_attr_pairs=ids_ent_attr_pairs,
                                            continuous_ent_attr_pairs=continuous_ent_attr_pairs,
                                            categorical_ent_attr_pairs=categorical_ent_attr_pairs,
                                            data_split=data_split,
                                            perc_patients_keep=1,
                                            #perc_patients_keep=0.01,
                                        )
        
    #==========================================================================#
    # Save other variables necessary to use the GNN models (possible_values_all_patients and categorical_ent_attr_pairs)
    possible_values_all_patients_path = store_path + "/possible_values_all_patients.pkl"
    with open(possible_values_all_patients_path, "wb") as fp:   
        pickle.dump(possible_values_all_patients, fp)
    categorical_ent_attr_pairs_path = store_path + "/categorical_ent_attr_pairs.pkl"
    with open(categorical_ent_attr_pairs_path, "wb") as fp:   #Pickling
        pickle.dump(categorical_ent_attr_pairs, fp)


if __name__=="__main__":
    main()
