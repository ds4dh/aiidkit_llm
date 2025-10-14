import os
import h5py
from copy import deepcopy
from datetime import datetime
import numpy as np
from tqdm import tqdm 
import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData


#====================================================================================================#
#====================================================================================================#
#====================================================================================================#
def get_patient_data_dicts(patient_ID, patients_dataset, data_split='Train'):
    """
        Get the data dicts of a patient. Two data dicts
        are return, one regrouping all the information
        of the patient in an EAV format (with supplementary
        keys such as days_since_tpx or infection_label_binary_any).
        The second one regroups the main features per date (no
        label information).

        Parameters:
        -----------
        patient_ID: int
            Patient ID int the Hugging Face dataset.
        patients_dataset: datasets.dataset_dict.DatasetDict
            Dataset containing all the patients in an EAV
            format.
        data_split: str
            Split from which we are going to extract the patient
            (usually 'train', 'validation', and 'test')

        Returns:
        --------
        pat_data_dict: dict
            Dictionary regrouping the patient information in an
            EAV format, having the following keys: 
                - 'entity',
                - 'attribute',
                - 'value',
                - 'time',
                - 'days_since_tpx',
                - 'infection_events',
                - 'patient_csv_path',
                - 'value_binned',
                - 'entity_id',
                - 'attribute_id',
                - 'value_id',
                - 'infection_label_binary_any',
                - 'infection_label_binary_bacterial',
                - 'infection_label_binary_viral',
                - 'infection_label_binary_fungal',
                - 'infection_label_categorical',
                - 'infection_label_one_hot',
                - 'cutoff',
                - 'horizon',
                - 'sequence_id'
        pat_data_dict_regrouped: dict
            Dictionary regrouping the main features of the patient
            per date. The keys are the different dates of measurments.
    """
    # Getting patient data
    pat_data_dict = {}
    for tmp_key in list(patients_dataset[data_split.lower()].features.keys()):
        pat_data_dict[tmp_key] = patients_dataset[data_split.lower()][tmp_key][patient_ID]

    # Regrouping patient data by date
    pat_data_dict_regrouped = {}
    for i in range(pat_data_dict['entity'].shape[0]):
        time = pat_data_dict['time'][i]
        if (time.lower() != 'nan'):
            if (pat_data_dict['time'][i] not in pat_data_dict_regrouped):
                pat_data_dict_regrouped[time] = []
            node = {
                        "entity": pat_data_dict['entity'][i],
                        "attribute": pat_data_dict['attribute'][i],
                        "days_since_tpx": pat_data_dict['days_since_tpx'][i],
                        "value": pat_data_dict['value'][i],
                        "time": pat_data_dict['time'][i]
                   }
            pat_data_dict_regrouped[time].append(node)

    return pat_data_dict, pat_data_dict_regrouped


def date_difference_in_days(date_str1, date_str2):
    """
    Computes the difference in days between two date strings (YYYY-MM-DD).

    Parameters:
    -----------
        date_str1: str
            The first date string.
        date_str2: str
            The second date string.

    Returns:
    -----------
        time_diff: int
            The difference in days.
    """
    # Define the expected format
    date_format = "%Y-%m-%d"

    # Convert the strings to datetime objects
    date1 = datetime.strptime(date_str1, date_format)
    date2 = datetime.strptime(date_str2, date_format)

    # Compute the difference (a timedelta object)
    time_difference = date2 - date1

    # Extract the total number of days from the timedelta object
    time_diff = time_difference.days 
    
    return time_diff

# Function indicating if a str is a float encoded as a string or a real string
def is_float_string(s):
    try:
        float(s)
        return True
    except ValueError:
        return False
    

def build_hetero_graph_without_embeddings(
                                        pat_data_dict_regrouped,
                                        ids_ent_attr_pairs,
                                        continuous_ent_attr_pairs,
                                        categorical_ent_attr_pairs,
                                        #constant_edges_without_attr=0.
                                        constant_edges_without_attr=1.
                                      ):
    """
        Returns a Pytorch Geometric heterogeneous data representing
        the patient trajectory.

        Parameters:
        -----------
        pat_data_dict_regrouped: dict
            Dictionary regrouping the patient's features by date 
            (the keys are the times of measurments of patient
            information).
        ids_ent_attr_pairs: dict
            Dictionary giving unique IDs to pairs (entity, attribute),
            necessary to use embeddings. The keys are (entity, attribute)
            pairs and the values are the IDs.
        continuous_ent_attr_pairs: dict
            Subset of ids_ent_attr_pairs getting only the (entity, attribute)
            pairs having continuous (float) values.
        categorical_ent_attr_pairs: dict
            Subset of ids_ent_attr_pairs getting only the (entity, attribute)
            pairs having categorical values.
        constant_edges_without_attr: float
            Value to use in attribute in edges that does not have attributes.
            Using a value of 0 does not necessarily means that there is 
            no edge between nodes.

        Returns:
        --------
        pat_data_graph: HeteroData
            Heterogeneous graph representing the patient's trajectory.
    """
    # Creating graph Pytorch Geometric structure for patient
    pat_data_graph = HeteroData()
    
    # Central nodes (time feature only, float)
    central_nodes = torch.from_numpy(np.unique([int(pat_data_dict_regrouped[key][i]['days_since_tpx']) for key in pat_data_dict_regrouped for i in range(len(pat_data_dict_regrouped[key]))]).reshape(-1, 1))
    pat_data_graph["central"].x = central_nodes.float()
    # TO BE ABLE TO USE BATCHING, WE ADD DUMMY FEATURES SO THAT ALL THE NODE TYPES HAVE THE SAME ATTRIBUTES
    pat_data_graph['central'].ent_attr_ids = torch.zeros(pat_data_graph["central"].x.shape)
    pat_data_graph['central'].vals = torch.zeros(pat_data_graph["central"].x.shape)
    pat_data_graph['central'].vocab_ids = torch.zeros(pat_data_graph["central"].x.shape)
    pat_data_graph['central'].days_since_tpx = torch.zeros(pat_data_graph["central"].x.shape)
    
    # Creating vocab. for categorical features
    categorical_vals_vocabs = {}
    for ent, attr in categorical_ent_attr_pairs:
        # Remove Unknown if exists
        categorical_ent_attr_pairs[(ent, attr)].discard('Unknown')
        categorical_ent_attr_pairs[(ent, attr)].discard('unknown')
        # Add Unknown token <UNK>
        categorical_ent_attr_pairs[(ent, attr)].add('<UNK>')
        categorical_vals_vocabs[(ent, attr)] = {value: index for index, value in enumerate(categorical_ent_attr_pairs[(ent, attr)])}
    
    # Child nodes IDs and values
    children_ent_attr_continuous_ids = []
    children_ent_attr_continuous_values = []
    children_ent_attr_continuous_times = []
    children_ent_attr_continuous_days_since_tpx = []
    children_ent_attr_categorical_ids = []
    children_ent_attr_categorical_values = []
    children_ent_attr_categorical_times = []
    children_ent_attr_categorical_days_since_tpx = []
    for day in pat_data_dict_regrouped:
        n_childs = len(pat_data_dict_regrouped[day])
        for child_local_ID in range(n_childs):
            # Getting the child entity-attribute ID
            child = pat_data_dict_regrouped[day][child_local_ID]
            ent_attr_pair = (child['entity'], child['attribute'])
            id_ent_attr_pair = ids_ent_attr_pairs[ent_attr_pair]
    
            # Getting the value
            if (ent_attr_pair in continuous_ent_attr_pairs):
                children_ent_attr_continuous_ids.append(id_ent_attr_pair)
                children_ent_attr_continuous_values.append(float(child['value']))
                children_ent_attr_continuous_times.append(child['time'])
                children_ent_attr_continuous_days_since_tpx.append(child['days_since_tpx'])
            elif (ent_attr_pair in categorical_ent_attr_pairs):
                children_ent_attr_categorical_ids.append(id_ent_attr_pair)
                children_ent_attr_categorical_values.append([ent_attr_pair, child['value']])
                children_ent_attr_categorical_times.append(child['time'])
                children_ent_attr_categorical_days_since_tpx.append(child['days_since_tpx'])
    
                
    children_ent_attr_continuous_ids = torch.tensor(children_ent_attr_continuous_ids)
    children_ent_attr_continuous_values = torch.tensor(children_ent_attr_continuous_values)
    children_ent_attr_continuous_days_since_tpx = torch.tensor(children_ent_attr_continuous_days_since_tpx)
    children_ent_attr_categorical_ids = torch.tensor(children_ent_attr_categorical_ids)
    children_ent_attr_categorical_vocab_ids = torch.stack([torch.tensor(categorical_vals_vocabs[c[0]].get(c[1], categorical_vals_vocabs[c[0]]["<UNK>"])) for c in children_ent_attr_categorical_values])
    children_ent_attr_categorical_days_since_tpx = torch.tensor(children_ent_attr_categorical_days_since_tpx)
    
    # Child node features
    # Continuous
    pat_data_graph['child_cont'].ent_attr_ids = children_ent_attr_continuous_ids
    pat_data_graph['child_cont'].vals = children_ent_attr_continuous_values # .values cannot be used as it is already used by Pytorch Geometric
    pat_data_graph['child_cont'].days_since_tpx = children_ent_attr_continuous_days_since_tpx
    # TO BE ABLE TO USE BATCHING, WE ADD DUMMY FEATURES SO THAT ALL THE NODE TYPES HAVE THE SAME ATTRIBUTES
    pat_data_graph['child_cont'].vocab_ids = torch.zeros(pat_data_graph['child_cont'].days_since_tpx.shape)
    pat_data_graph["child_cont"].x = torch.zeros(pat_data_graph['child_cont'].days_since_tpx.shape)
    
    # Categorical
    pat_data_graph['child_categ'].ent_attr_ids = children_ent_attr_categorical_ids
    pat_data_graph['child_categ'].vocab_ids = children_ent_attr_categorical_vocab_ids
    pat_data_graph['child_categ'].days_since_tpx = children_ent_attr_categorical_days_since_tpx
    # TO BE ABLE TO USE BATCHING, WE ADD DUMMY FEATURES SO THAT ALL THE NODE TYPES HAVE THE SAME ATTRIBUTES
    pat_data_graph['child_categ'].vals = torch.zeros(pat_data_graph['child_categ'].days_since_tpx.shape)
    pat_data_graph["child_categ"].x = torch.zeros(pat_data_graph['child_categ'].days_since_tpx.shape)
    
    # Create the edges
    # Between central nodes
    central_nodes_edge_idx = torch.tensor([ [i for i in range(len(central_nodes)-1)], [j for j in range(1, len(central_nodes))]])
    central_nodes_times = [key for key in pat_data_dict_regrouped]
    edges_central_nodes_edge_attr = torch.tensor([date_difference_in_days(central_nodes_times[i], central_nodes_times[i+1]) for i in range(len(central_nodes_times)-1)]).unsqueeze(1)
    pat_data_graph["central", "sequence", "central"].edge_index = central_nodes_edge_idx
    pat_data_graph["central", "sequence", "central"].edge_attr = edges_central_nodes_edge_attr
    # Between central nodes and child nodes
    central_to_child_cont_edge_idx = [[], []]
    central_to_child_categ_edge_idx = [[], []]
    for central_node_i in range(central_nodes.shape[0]):
        time_central_node = central_nodes_times[central_node_i]
        # Continuous nodes
        for child_cont_j in range(len(children_ent_attr_continuous_times)):
            time_child_cont = children_ent_attr_continuous_times[child_cont_j]
            if (time_central_node == time_child_cont):
                central_to_child_cont_edge_idx[0].append(central_node_i)
                central_to_child_cont_edge_idx[1].append(child_cont_j)
        # Categorical nodes
        for child_categ_j in range(len(children_ent_attr_categorical_times)):
            time_child_categ = children_ent_attr_categorical_times[child_categ_j]
            if (time_central_node == time_child_categ):
                central_to_child_categ_edge_idx[0].append(central_node_i)
                central_to_child_categ_edge_idx[1].append(child_categ_j)
    central_to_child_cont_edge_idx = torch.tensor(central_to_child_cont_edge_idx)
    central_to_child_categ_edge_idx = torch.tensor(central_to_child_categ_edge_idx)
    pat_data_graph["central", "has_child", "child_cont"].edge_index = central_to_child_cont_edge_idx
    pat_data_graph["central", "has_child", "child_cont"].edge_attr = torch.ones((1, central_to_child_cont_edge_idx.shape[1]))*constant_edges_without_attr
    pat_data_graph["central", "has_child", "child_categ"].edge_index = central_to_child_categ_edge_idx
    pat_data_graph["central", "has_child", "child_categ"].edge_attr = torch.ones((1, central_to_child_categ_edge_idx.shape[1]))*constant_edges_without_attr

    return pat_data_graph
    

def create_HDF5_file_graph_DS(
                                store_path,
                                patients_dataset,
                                ids_ent_attr_pairs,
                                continuous_ent_attr_pairs,
                                categorical_ent_attr_pairs,
                                data_split,
                                perc_patients_keep=1.0,
                                metadata={}
                             ):
    """
        Creates an HDF5 file with the graph structure
        necessary to create Pytorch Geometric graph
        datasets.

        Parameters:
        -----------
        store_path: str
            Path to store the created HDF5 file.
        patients_dataset: datasets.dataset_dict.DatasetDict
            Dataset containing all the patients in an EAV
            format.
        ids_ent_attr_pairs: dict
            Dictionary giving unique IDs to pairs (entity, attribute),
            necessary to use embeddings. The keys are (entity, attribute)
            pairs and the values are the IDs.
        continuous_ent_attr_pairs: dict
            Subset of ids_ent_attr_pairs getting only the (entity, attribute)
            pairs having continuous (float) values.
        categorical_ent_attr_pairs: dict
            Subset of ids_ent_attr_pairs getting only the (entity, attribute)
            pairs having categorical values.
        data_split: str
            Split from which we are going to extract the patient
            (usually 'train', 'validation', and 'test')
        perc_patients_keep: float
            Percentage of patients to keep.
        metadata: dict
            Dictionary containing information about the dataset that is going
            to be stored in the HDF5 file (cuttoff days for train, cutoff 
            days for validation and prediction horizon)

        Returns:
        --------
        h5_fn: str
            Path to the HDF5 containing the dataset.
        
    """
    # HDF5 filename
    i = 0
    h5_fn = store_path + f"/AIIDKIT_TEAV_Graph_{data_split.upper()}_"
    for key in metadata:
        h5_fn += f"{key}-{metadata[key]}_"
    while (os.path.exists(h5_fn + str(i) + '.hdf5')):
        i += 1
    h5_fn = h5_fn + str(i) + '.hdf5'

    # Creating the file
    hdf5_file = h5py.File(h5_fn, "w")

    # Iterating over the patients
    n_patients = patients_dataset[data_split.lower()].shape[0]
    if (perc_patients_keep < 1.0):
        n_patients = int(perc_patients_keep*n_patients)
    for pat_ID in tqdm(range(n_patients)):
        # Create group for the current patient
        current_pat_group = hdf5_file.create_group(str(pat_ID))
        
        # Get data
        pat_data_dict, pat_data_dict_regrouped = get_patient_data_dicts(
                                                                            patient_ID=pat_ID,
                                                                            patients_dataset=patients_dataset,
                                                                            data_split=data_split
                                                                        )
    
        # Build HeteroData
        pat_data_graph = build_hetero_graph_without_embeddings(
                                                                    pat_data_dict_regrouped=pat_data_dict_regrouped,
                                                                    ids_ent_attr_pairs=ids_ent_attr_pairs,
                                                                    continuous_ent_attr_pairs=continuous_ent_attr_pairs,
                                                                    categorical_ent_attr_pairs=categorical_ent_attr_pairs
                                                                )

        # Store in HDF5
        # Create datasets for Central nodes
        central_subgroup = current_pat_group.create_group("central")
        central_subgroup.create_dataset("x", data=pat_data_graph["central"].x.cpu().numpy())
        central_subgroup.create_dataset("ent_attr_ids", data=pat_data_graph['central'].ent_attr_ids.cpu().numpy())
        central_subgroup.create_dataset("vals", data=pat_data_graph['central'].vals.cpu().numpy())
        central_subgroup.create_dataset("vocab_ids", data=pat_data_graph['central'].vocab_ids.cpu().numpy())
        central_subgroup.create_dataset("days_since_tpx", data=pat_data_graph['central'].days_since_tpx.cpu().numpy())
        # Create datasets for Child nodes
        # Continuous nodes
        child_cont_subgroup = current_pat_group.create_group("child_cont")
        child_cont_subgroup.create_dataset("x", data=pat_data_graph["child_cont"].x.cpu().numpy())
        child_cont_subgroup.create_dataset("ent_attr_ids", data=pat_data_graph['child_cont'].ent_attr_ids.cpu().numpy())
        child_cont_subgroup.create_dataset("vals", data=pat_data_graph['child_cont'].vals.cpu().numpy())
        child_cont_subgroup.create_dataset("vocab_ids", data=pat_data_graph['child_cont'].vocab_ids.cpu().numpy())
        child_cont_subgroup.create_dataset("days_since_tpx", data=pat_data_graph['child_cont'].days_since_tpx.cpu().numpy())
        # Categorical nodes
        child_categ_subgroup = current_pat_group.create_group("child_categ")
        child_categ_subgroup.create_dataset("x", data=pat_data_graph["child_categ"].x.cpu().numpy())
        child_categ_subgroup.create_dataset("ent_attr_ids", data=pat_data_graph['child_categ'].ent_attr_ids.cpu().numpy())
        child_categ_subgroup.create_dataset("vals", data=pat_data_graph['child_categ'].vals.cpu().numpy())
        child_categ_subgroup.create_dataset("vocab_ids", data=pat_data_graph['child_categ'].vocab_ids.cpu().numpy())
        child_categ_subgroup.create_dataset("days_since_tpx", data=pat_data_graph['child_categ'].days_since_tpx.cpu().numpy())
        # Create datasets for edges between central nodes
        central_to_central_subgroup = current_pat_group.create_group("central_to_central")
        central_to_central_subgroup.create_dataset("edge_index", data=pat_data_graph["central", "sequence", "central"].edge_index.cpu().numpy())
        central_to_central_subgroup.create_dataset("edge_attr", data=pat_data_graph["central", "sequence", "central"].edge_attr.cpu().numpy())
        # Create datasets between central nodes and children nodes
        central_to_child_cont_subgroup = current_pat_group.create_group("central_to_child_cont")
        central_to_child_cont_subgroup.create_dataset("edge_index", data=pat_data_graph["central", "has_child", "child_cont"].edge_index.cpu().numpy())
        central_to_child_cont_subgroup.create_dataset("edge_attr", data=pat_data_graph["central", "has_child", "child_cont"].edge_attr.cpu().numpy())
        central_to_child_categ_subgroup = current_pat_group.create_group("central_to_child_categ")
        central_to_child_categ_subgroup.create_dataset("edge_index", data=pat_data_graph["central", "has_child", "child_categ"].edge_index.cpu().numpy())
        central_to_child_categ_subgroup.create_dataset("edge_attr", data=pat_data_graph["central", "has_child", "child_categ"].edge_attr.cpu().numpy())
    
        # Add label
        possible_labels_types = [
                                    "infection_label_binary_any",
                                    "infection_label_binary_bacterial",
                                    "infection_label_binary_viral",
                                    "infection_label_binary_fungal",
                                    "infection_label_categorical",
                                    "infection_label_one_hot"
                                ]
        for label_type in possible_labels_types:
            label = pat_data_dict[label_type]
            current_pat_group.create_dataset(label_type, data=label)
        
    return h5_fn

def create_PyGeo_Graph_DS_from_HDF5(h5_fn, label_type="infection_label_binary_bacterial"):
    """
        Creates a list of Pytorch Geometric HeteroData graphs
        that can be used for training.

        Parameters:
        -----------
        h5_fn: str
            Path to the HDF5 containing the dataset.
        label_type: str
            Type of label to use. The following options are available:
                - infection_label_binary_any
                - infection_label_binary_bacterial
                - infection_label_binary_viral
                - infection_label_binary_fungal
                - infection_label_categorical
                - infection_label_one_hot

        Returns:
        --------
        dataset: list
            List of HeteroData.
    """
    # Load HDF5 file
    hdf5_file = h5py.File(h5_fn, 'r')

    # Iterating over the file
    dataset = []
    pats_ids = sorted([int(str_pat_ID) for str_pat_ID in hdf5_file]) # To iterate the patients in the same order
    for pat_ID in tqdm(pats_ids):
        # Transform pat_ID to string 
        str_pat_ID = str(pat_ID)
        
        # Create patient hetero data
        pat_data_graph = HeteroData()

        # Central nodes
        pat_data_graph["central"].x = torch.from_numpy(hdf5_file[str_pat_ID]["central"]["x"][:])
        pat_data_graph['central'].ent_attr_ids = torch.from_numpy(hdf5_file[str_pat_ID]["central"]["ent_attr_ids"][:])
        pat_data_graph['central'].vals = torch.from_numpy(hdf5_file[str_pat_ID]["central"]["vals"][:])
        pat_data_graph['central'].vocab_ids = torch.from_numpy(hdf5_file[str_pat_ID]["central"]["vocab_ids"][:])
        pat_data_graph['central'].days_since_tpx = torch.from_numpy(hdf5_file[str_pat_ID]["central"]["days_since_tpx"][:])

        # Continuous children nodes
        pat_data_graph["child_cont"].x = torch.from_numpy(hdf5_file[str_pat_ID]["child_cont"]["x"][:])
        pat_data_graph['child_cont'].ent_attr_ids = torch.from_numpy(hdf5_file[str_pat_ID]["child_cont"]["ent_attr_ids"][:])
        pat_data_graph['child_cont'].vals = torch.from_numpy(hdf5_file[str_pat_ID]["child_cont"]["vals"][:])
        pat_data_graph['child_cont'].vocab_ids = torch.from_numpy(hdf5_file[str_pat_ID]["child_cont"]["vocab_ids"][:])
        pat_data_graph['child_cont'].days_since_tpx = torch.from_numpy(hdf5_file[str_pat_ID]["child_cont"]["days_since_tpx"][:])

        # Categorical children nodes
        pat_data_graph["child_categ"].x = torch.from_numpy(hdf5_file[str_pat_ID]["child_categ"]["x"][:])
        pat_data_graph['child_categ'].ent_attr_ids = torch.from_numpy(hdf5_file[str_pat_ID]["child_categ"]["ent_attr_ids"][:])
        pat_data_graph['child_categ'].vals = torch.from_numpy(hdf5_file[str_pat_ID]["child_categ"]["vals"][:])
        pat_data_graph['child_categ'].vocab_ids = torch.from_numpy(hdf5_file[str_pat_ID]["child_categ"]["vocab_ids"][:])
        pat_data_graph['child_categ'].days_since_tpx = torch.from_numpy(hdf5_file[str_pat_ID]["child_categ"]["days_since_tpx"][:])

        # Central to central edges
        pat_data_graph["central", "sequence", "central"].edge_index = torch.from_numpy(hdf5_file[str_pat_ID]["central_to_central"]["edge_index"][:])
        pat_data_graph["central", "sequence", "central"].edge_attr = torch.from_numpy(hdf5_file[str_pat_ID]["central_to_central"]["edge_attr"][:])
        # Central to central child_cont
        pat_data_graph["central", "has_child", "child_cont"].edge_index = torch.from_numpy(hdf5_file[str_pat_ID]["central_to_child_cont"]["edge_index"][:])
        pat_data_graph["central", "has_child", "child_cont"].edge_attr = torch.from_numpy(hdf5_file[str_pat_ID]["central_to_child_cont"]["edge_attr"][:]).T # Transpose as should be of shape (num_features, 1) and not (1, num_features) to be able to do batching
        # Central to central child_categ
        pat_data_graph["central", "has_child", "child_categ"].edge_index = torch.from_numpy(hdf5_file[str_pat_ID]["central_to_child_categ"]["edge_index"][:])
        pat_data_graph["central", "has_child", "child_categ"].edge_attr = torch.from_numpy(hdf5_file[str_pat_ID]["central_to_child_categ"]["edge_attr"][:]).T # Transpose as should be of shape (num_features, 1) and not (1, num_features) to be able to do batching

        # Label
        try:
            label = torch.tensor(hdf5_file[str_pat_ID][label_type][()])
        except:
            label = torch.tensor(hdf5_file[str_pat_ID][label_type][:])
            
        pat_data_graph.y = torch.tensor(label)

        # Add to dataset
        dataset.append(deepcopy(pat_data_graph))

    return dataset
    
def normalize_dataset(dataset, statistics):
    """
        Normalize the features of a dataset using the statistics
        in the argument statistics.

        Parameters:
        -----------
        dataset: list
            List of HeteroData graphs from which we want to
            normalize the features.

        Returns:
        --------
        normalized_dataset: list
            List of HeteroData with normalized features.
    """
    # Creating new list for normalized dataset
    normalized_dataset = []

    # Iterating over the dataset
    for hetero_graph in tqdm(dataset):
        # Clone graph to normalize it
        normalized_hetero_graph = deepcopy(hetero_graph)

        # Normalize features of central nodes (days since transplant) using min-max scaled to -1 and 1 to keep the importance of negative sign
        normalized_hetero_graph['central'].x = 2*((normalized_hetero_graph['central'].x - statistics['central_nodes_x']['Min'])/(statistics['central_nodes_x']['Max'] - statistics['central_nodes_x']['Min'])) - 1 
        
        # For continuous variable we do Standarization (z-score)
        child_cont_ent_attr_ids_np = hetero_graph['child_cont'].ent_attr_ids.cpu().numpy()
        for i in range(len(child_cont_ent_attr_ids_np)):
            cont_feat_id = int(child_cont_ent_attr_ids_np[i])
            if (cont_feat_id not in statistics["child_cont_nodes_vals"]):
                print(f"\nPROBLEM: continuous feature of ID {cont_feat_id} is not in the statistics dict so it cannot be normalized.\n")
                raise RuntimeError(f"PROBLEM: continuous feature of ID {cont_feat_id} is not in the statistics dict so it cannot be normalized")
            else:
                if not (type(statistics["child_cont_nodes_vals"][cont_feat_id]["Mean"]) == np.float32):
                    if (torch.isnan(statistics["child_cont_nodes_vals"][cont_feat_id]["Mean"])): # In this case, all the values of the feature are 0 so we do nothing
                        pass
                else:
                    normalized_hetero_graph['child_cont'].vals[i] = (normalized_hetero_graph['child_cont'].vals[i] - statistics["child_cont_nodes_vals"][cont_feat_id]["Mean"])/(statistics["child_cont_nodes_vals"][cont_feat_id]["Std"])
                if (torch.any(torch.isnan(normalized_hetero_graph['child_cont'].vals[i]))):
                    print(f"\nPROBLEM: NaN value when normalizing {hetero_graph['child_cont'].vals[i]} using Mean {statistics['child_cont_nodes_vals'][cont_feat_id]['Mean']} and Std {statistics['child_cont_nodes_vals'][cont_feat_id]['Std']}")

        # For days since tpx we do min-max scaled to -1 and 1 to keep the importance of negative sign
        normalized_hetero_graph['child_cont'].days_since_tpx = 2*((normalized_hetero_graph['child_cont'].days_since_tpx - statistics['child_cont_nodes_days_since_tpx']['Min'])/(statistics['child_cont_nodes_days_since_tpx']['Max'] - statistics['child_cont_nodes_days_since_tpx']['Min'])) - 1 
        normalized_hetero_graph['child_categ'].days_since_tpx = 2*((normalized_hetero_graph['child_categ'].days_since_tpx - statistics['child_categ_nodes_days_since_tpx']['Min'])/(statistics['child_categ_nodes_days_since_tpx']['Max'] - statistics['child_categ_nodes_days_since_tpx']['Min'])) - 1 

        # For times between measurments (central to central edge attribute) we to Log transform + standarziation or min-max
        normalized_hetero_graph[('central', 'sequence', 'central')].edge_attr = (torch.log(normalized_hetero_graph[('central', 'sequence', 'central')].edge_attr + 1) - statistics["central_to_central_edge_attr"]["LogMin"])/(statistics["central_to_central_edge_attr"]["LogMax"] - statistics["central_to_central_edge_attr"]["LogMin"])

        # Adding the normalized hetero graph into the new list of normalized graphs
        normalized_dataset.append(normalized_hetero_graph)

    return normalized_dataset



def compute_statistics_dataset(dataset):
    """
        Normalize the features of a dataset using the statistics
        in the argument statistics.

        Parameters:
        -----------
        dataset: list
            List of HeteroData graphs from which we want to
            normalize the features.

        Returns:
        --------
        statistics: dict
            Dictionary containing the computed statistics of 
            the dataset.
    """
    # Initialize statistics dict
    statistics = {
                    "central_nodes_x": {"Min": None, "Max": None, "Mean": None, "Std": None},
                    "child_cont_nodes_vals": {}, # All metrics are computed per feature (each value in hetero_graph['child_cont'].vals is one feature, but not always in the same order)
                    "child_cont_nodes_days_since_tpx": {"Min": None, "Max": None, "Mean": None, "Std": None},
                    "child_categ_nodes_days_since_tpx": {"Min": None, "Max": None, "Mean": None, "Std": None},
                    "central_to_central_edge_attr": {"Min": None, "Max": None, "Mean": None, "Std": None, "LogMin": None, "LogMax": None, "LogMean": None, "LogStd": None}
                }

    # Iterating over all the graphs in the dataset
    central_nodes_x_list = []
    child_cont_nodes_vals_dict = {}
    child_cont_nodes_days_since_tpx_list = []
    child_categ_nodes_days_since_tpx_list = []
    central_to_central_edge_attr_list = []
    for hetero_graph in tqdm(dataset):
        # Getting the values of the features that we want to normalize
        # Central nodes
        central_nodes_x_list.extend(hetero_graph['central'].x.cpu().numpy().squeeze().tolist())
        # Children with continuous/float values
        # IMPORTANT: for child_cont_nodes_vals_list we do append as we are going to normalize PER FEATURE and each value is a different feature
        # Values 
        child_cont_ent_attr_ids_np = hetero_graph['child_cont'].ent_attr_ids.cpu().numpy()
        child_cont_vals_np = hetero_graph['child_cont'].vals.cpu().numpy()
        for i in range(len(child_cont_ent_attr_ids_np)):
            cont_feat_id = int(child_cont_ent_attr_ids_np[i])
            cont_feat_val = child_cont_vals_np[i]
            if (cont_feat_id not in statistics["child_cont_nodes_vals"]):
                statistics["child_cont_nodes_vals"][cont_feat_id] = {"Min": None, "Max": None, "Mean": None, "Std": None}
                child_cont_nodes_vals_dict[cont_feat_id] = []
            child_cont_nodes_vals_dict[cont_feat_id].append(cont_feat_val)
        # Days since transplantation
        child_cont_days_since_tpx_vals = hetero_graph['child_cont'].days_since_tpx.cpu().numpy().squeeze().tolist()
        if (type(child_cont_days_since_tpx_vals) == int): # Sometimes it can happen when hetero_graph['child_cont'].days_since_tpx.shape is of shape [1]
            child_cont_days_since_tpx_vals = [child_cont_days_since_tpx_vals]
        child_cont_nodes_days_since_tpx_list.extend(child_cont_days_since_tpx_vals)
        # Children with categorical values
        child_categ_nodes_days_since_tpx_list.extend(hetero_graph['child_categ'].days_since_tpx.cpu().numpy().squeeze().tolist()) 
        # Edges between central nodes
        central_to_central_edge_attr_list.extend(hetero_graph[('central', 'sequence', 'central')].edge_attr.cpu().numpy().squeeze().tolist())

    # Transform into np.array
    central_nodes_x_list = np.array(central_nodes_x_list)
    #print(f"\n central_nodes_x_list.shape = {central_nodes_x_list.shape} \n")
    for cont_feat_id in child_cont_nodes_vals_dict:
        child_cont_nodes_vals_dict[cont_feat_id] = np.array(child_cont_nodes_vals_dict[cont_feat_id])
        #print(f"\n child_cont_nodes_vals_dict[{cont_feat_id}] = {child_cont_nodes_vals_dict[cont_feat_id].shape} \n")
    child_cont_nodes_days_since_tpx_list = np.array(child_cont_nodes_days_since_tpx_list)
    #print(f"\n child_cont_nodes_days_since_tpx_list.shape = {child_cont_nodes_days_since_tpx_list.shape} \n")
    child_categ_nodes_days_since_tpx_list = np.array(child_categ_nodes_days_since_tpx_list)
    #print(f"\n child_categ_nodes_days_since_tpx_list.shape = {child_categ_nodes_days_since_tpx_list.shape} \n")
    central_to_central_edge_attr_list = np.array(central_to_central_edge_attr_list)
    #print(f"\n central_to_central_edge_attr_list.shape = {central_to_central_edge_attr_list.shape} \n")

    # Compute statistics
    statistics["central_nodes_x"] = {"Min": np.min(central_nodes_x_list), "Max": np.max(central_nodes_x_list), "Mean": np.mean(central_nodes_x_list), "Std": np.std(central_nodes_x_list)}
    for cont_feat_id in statistics["child_cont_nodes_vals"]:
        if ((child_cont_nodes_vals_dict[cont_feat_id] == 0.0).all()): # In this case all the values are 0.
            statistics["child_cont_nodes_vals"][cont_feat_id] = {"Min": torch.tensor(torch.nan), "Max": torch.tensor(torch.nan), "Mean": torch.tensor(torch.nan), "Std": torch.tensor(torch.nan)}
        else:
            statistics["child_cont_nodes_vals"][cont_feat_id] = {"Min": np.min(child_cont_nodes_vals_dict[cont_feat_id]), "Max": np.max(child_cont_nodes_vals_dict[cont_feat_id]), "Mean": np.mean(child_cont_nodes_vals_dict[cont_feat_id]), "Std": np.std(child_cont_nodes_vals_dict[cont_feat_id])}
    statistics["child_cont_nodes_days_since_tpx"] = {"Min": np.min(child_cont_nodes_days_since_tpx_list), "Max": np.max(child_cont_nodes_days_since_tpx_list), "Mean": np.mean(child_cont_nodes_days_since_tpx_list), "Std": np.std(child_cont_nodes_days_since_tpx_list)}
    statistics["child_categ_nodes_days_since_tpx"] = {"Min": np.min(child_categ_nodes_days_since_tpx_list), "Max": np.max(child_categ_nodes_days_since_tpx_list), "Mean": np.mean(child_categ_nodes_days_since_tpx_list), "Std": np.std(child_categ_nodes_days_since_tpx_list)}
    statistics["central_to_central_edge_attr"] = {"Min": np.min(central_to_central_edge_attr_list), "Max": np.max(central_to_central_edge_attr_list), "Mean": np.mean(central_to_central_edge_attr_list), "Std": np.std(central_to_central_edge_attr_list)}
    statistics["central_to_central_edge_attr"] = {"LogMin": np.min(np.log(central_to_central_edge_attr_list+1)), "LogMax": np.max(np.log(central_to_central_edge_attr_list+1)), "LogMean": np.mean(np.log(central_to_central_edge_attr_list+1)), "LogStd": np.std(np.log(central_to_central_edge_attr_list+1))}

    return statistics