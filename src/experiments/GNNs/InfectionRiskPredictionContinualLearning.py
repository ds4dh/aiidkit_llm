#!/usr/bin/env python3
"""
    Class for an experiment training a Deep Learning model
    to fo infection risk prediction
"""
import os
import pickle
import yaml
import argparse
from datetime import datetime
from tqdm import tqdm
from random import shuffle
import shutil

from copy import deepcopy

import h5py

import random

import numpy as np

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.utils.class_weight import compute_class_weight

import torch
import torch_geometric

# Internal imports
from src.experiments.GenericExperiment import GenericExperiment
from src.data.process.graph_patient_dataset import create_PyGeo_Graph_DS_from_HDF5,\
                                                   normalize_dataset,\
                                                   compute_statistics_dataset
from src.model.GraphBased.HeteroGNN import HeteroGNN
from src.model.GraphBased.HeteroGraphSage import HeteroGraphSage
from src.model.GraphBased.HeteroGAT import HeteroGAT
from src.experiments.Utils.EvidentialClassification import EvidentialClassification
from src.experiments.Utils.tools import get_uncertainties
from src.experiments.GNNs.InfectionRiskPrediction import InfectionRiskPred


#====================================================================================================#
#====================================================================================================#
#====================================================================================================#
# Define some useful functions
def split_into_n_lists(lst, N):
    """
        Generated with ChatGPT, splits a list into into N smaller lists, each containing (as evenly 
        as possible) distinct elements from the original list — i.e., no integer appears twice and the
        groups are balanced in size.
    """
    k, m = divmod(len(lst), N)
    return [lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(N)]


#====================================================================================================#
#====================================================================================================#
#====================================================================================================#
# Abstract class
class InfectionRiskPredContinualLearning(InfectionRiskPred):
    def __init__(self, parameters_exp):
        """
            Class for infection risk prediction DL experiment.

            Arguments:
            ----------
            parameters_exp: dict
                Dictionary containing the parameters of the experiment.
        """
        # Parent constructor
        super().__init__(parameters_exp)    

        # Parameters for the number of incremental/continual DS to create
        if ('n_incrementa_ds' not in self.parameters_exp):
            # It includes the first training dataset (50% of the training patients and the rest of the datasets)
            self.parameters_exp['n_incrementa_ds'] = 4
        if ('incremental_ds_strategy' not in self.parameters_exp):
            #self.parameters_exp['incremental_ds_strategy'] = 'EqualNoPatients'
            self.parameters_exp['incremental_ds_strategy'] = 'FirstDSHalfPatients'

        # Parameters for continual learning
        if ('ContinualLearning' not in self.parameters_exp):
            self.parameters_exp['ContinualLearning'] = {
                                                            'replay': False,
                                                            'memory_capacity': 0
                                                       }

        # Parameters for the replay-based strategy
        self.memory = None   
        if ('replay' not in self.parameters_exp['ContinualLearning']):
            self.parameters_exp['ContinualLearning']['replay'] = False
        if (self.parameters_exp['ContinualLearning']['replay']):
            # Memory capacity
            if ('memory_capacity' not in self.parameters_exp['ContinualLearning']):
                self.parameters_exp['ContinualLearning']['memory_capacity'] = 100
            # Replay strategy
            if ('replay_strategy' not in self.parameters_exp['ContinualLearning']):
                self.parameters_exp['ContinualLearning']['replay_strategy'] = None
            # Weights of the losses
            if ('lambda_dataset' not in self.parameters_exp['ContinualLearning']):
                self.parameters_exp['ContinualLearning']['lambda_dataset'] = 1
            if ('lambda_replay' not in self.parameters_exp['ContinualLearning']):
                self.parameters_exp['ContinualLearning']['lambda_replay'] = 1e-1


    def createIncrementalDatasets(self):
        """
            Create the incremental torch datasets associated needed to train and evaluate the model
            in a Continual Learning context.
            Two strategies:
                - EqualNoPatients: each dataset contain the same (or almost) number of patients. 
                For instance, if we have 2000 training patients and self.parameters_exp['n_incrementa_ds'] = 4
                Then each dataset will contain 500 patients (non overlapping, one patient can only
                be in one dataset).
                - FirstDSHalfPatients : In this case, the first dataset contain half of the total training patients and the
                rest of the datasets contain the same number of patients.
        """
        #====================================================================================================#
        #====================================================================================================#
        # Creating full datasets
        self.createTorchDatasets()
        self.full_train_ds = deepcopy(self.train_ds)

        # Create a copy of the val and test datasets (as normalization changes over time)
        self.raw_val_ds = deepcopy(self.val_ds)
        self.raw_test_ds = deepcopy(self.test_ds)

        # Getting the patients IDs
        patients_IDs = []
        for pat_graph in self.full_train_ds:
            patients_IDs.append(pat_graph.pat_ID)
        patients_IDs = np.unique(patients_IDs)
        np.random.shuffle(patients_IDs) # Shuffle in-place the list of indices

        # Creating the distribution of patients among the different datasets
        if (self.parameters_exp['incremental_ds_strategy'].lower() == 'equalnopatients'):
            pats_IDs_per_DS = split_into_n_lists(patients_IDs, self.parameters_exp['n_incrementa_ds'])
        elif (self.parameters_exp['incremental_ds_strategy'].lower() == 'firstdshalfpatients'):
            first_DS_pats_IDs = patients_IDs[:len(patients_IDs)//2]
            other_DS_pats_IDs = split_into_n_lists(patients_IDs[len(patients_IDs)//2:], self.parameters_exp['n_incrementa_ds']-1) # -1 as the first dataset was created manually
            pats_IDs_per_DS = [first_DS_pats_IDs]
            pats_IDs_per_DS.extend(other_DS_pats_IDs)
        else:
            ValueError(f"Strategy {self.parameters_exp['incremental_ds_strategy']} to create the incremental DS is not valid")

        # Creating the different Pytorch datasets datasets
        self.train_incremental_ds = [[] for _ in range(self.parameters_exp['n_incrementa_ds'])]
        for pat_seq_graph in self.full_train_ds:
            # Copying the sample in the right DS
            for DS_ID in range(self.parameters_exp['n_incrementa_ds']):
                if (pat_seq_graph.pat_ID in pats_IDs_per_DS[DS_ID]):
                    self.train_incremental_ds[DS_ID].append(deepcopy(pat_seq_graph))

        #====================================================================================================#
        #====================================================================================================#
        # Print number of patients and samples per DS
        n_DS = len(self.train_incremental_ds)
        for DS_ID in range(n_DS):
            # Number of patients
            n_patients_in_current_DS = len(pats_IDs_per_DS[DS_ID])
            print(f"\n=========> In DS {DS_ID} there are {n_patients_in_current_DS} patients (computed using pats_IDs_per_DS[{DS_ID}])")
            # Number of samples
            n_samples_in_current_DS = len(self.train_incremental_ds[DS_ID])
            print(f"\t=========> In DS {DS_ID} there are {n_samples_in_current_DS} samples")
            # Number of samples per class
            n_samples_per_class_current_DS = {}
            for pat_seq_graph in self.train_incremental_ds[DS_ID]:
                tmp_class = int(pat_seq_graph.y)
                if (tmp_class not in n_samples_per_class_current_DS):
                    n_samples_per_class_current_DS[tmp_class] = 0
                n_samples_per_class_current_DS[tmp_class] += 1
            print(f"\t=========> In DS {DS_ID} there are:")
            for tmp_class in n_samples_per_class_current_DS:
                print(f"\t===> {n_samples_per_class_current_DS[tmp_class]} samples of class {tmp_class}")

        # # Sanity check: the number of different patients and their IDs in each dataset of self.train_incremental_ds should be the same as pats_IDs_per_DS
        # for DS_ID in range(self.parameters_exp['n_incrementa_ds']):
        #     # Getting the patients IDs in the dataset
        #     tmp_pats_IDs = []
        #     for pat_seq_graph in self.train_incremental_ds[DS_ID]:
        #         tmp_pats_IDs.append(int(pat_seq_graph.pat_ID))
        #     tmp_pats_IDs = np.array(tmp_pats_IDs)
        #     print(f"\n=========> In DS {DS_ID} there are {len(tmp_pats_IDs)} patients (computed using self.train_incremental_ds[{DS_ID}])")
        #     print(f"\t=========> Same patients IDs in pats_IDs_per_DS[{DS_ID}] and self.train_incremental_ds[{DS_ID}]: {np.array_equal(np.unique(pats_IDs_per_DS[DS_ID]), np.unique(tmp_pats_IDs))}")

    
    def addClassWeightsLoss(self, multiclass_strategy='balanced'):
        """
            Compute class weights and redefines the loss function using
            these class weights to handle imbalanced classes.
        """
        # Get the current training dataset
        self.createCombinedTrainingDS()

        # Get class weights and new losses for the current training dataset
        super().addClassWeightsLoss(multiclass_strategy='balanced')


    def computeTrainDSStatistics(self):
        """
            Computes the statistics necessary to normalize the dataset.
            IMPORTANT: In a continual learning setting, we do not have access
            to all the data at once. Therefore, we cannot compute the global 
            statistics of the datasets at once.
            Instead we propose to compute Online normalization statistics
            as the new data arrive, and do a replay-buffer (memory) aided
            normalization.
        """
        print(f"\n\n==========> NORMALIZING DATASETS <==========\n")
        if (self.parameters_exp['Dataset']['dataset_name'].lower() == 'aiidkit') and\
            (self.parameters_exp['Dataset']['subdataset'].lower() == 'teav_static_graph_v1'):
                # Compute statistics on the current incremental dataset                    
                current_combined_DS_statistics = compute_statistics_dataset(self.train_ds) # Normally, here the train_ds is the combination of the memory and current incremental DS (if the memory is empty, it is only the current incremental DS)
                self.n_seen_samples += len(self.train_ds)
                if (self.train_statistics is None):
                    self.train_statistics = current_combined_DS_statistics
                else:
                    # Updating the statistics per feature
                    for feature_type in self.train_statistics:
                        # Number of features in the current DS
                        n_current_samples = len(self.train_ds)

                        # Update the global statistics
                        if ('Mean' in current_combined_DS_statistics[feature_type]) or ('LogMean' in current_combined_DS_statistics[feature_type]):
                            try:
                                # Mean
                                new_mean = ( self.n_seen_samples * self.train_statistics[feature_type]['Mean'] + n_current_samples*current_combined_DS_statistics['Mean'] ) / (self.n_seen_samples + n_current_samples)
                                # Std
                                new_std = (
                                            self.n_seen_samples*(self.train_statistics[feature_type]['Std'] + (self.train_statistics[feature_type]['Mean'] - new_mean)**2) + n_current_samples*(current_combined_DS_statistics['Std'] + (current_combined_DS_statistics['Mean'] - new_mean)**2)
                                        ) / (self.n_seen_samples + n_current_samples)
                                # Min
                                new_min = min(self.train_statistics[feature_type]['Min'], current_combined_DS_statistics['Min'])
                                # Max
                                new_max = min(self.train_statistics[feature_type]['Max'], current_combined_DS_statistics['Max'])
                            except:
                                pass
                        

                            # Particular case of central_to_central_edge_attr where we also have to compute the min, max, mean, and std of the logs
                            try:
                                # Log Mean
                                new_log_mean = ( self.n_seen_samples * self.train_statistics[feature_type]['LogMean'] + n_current_samples*current_combined_DS_statistics['LogMean'] ) / (self.n_seen_samples + n_current_samples)
                                # Log Std
                                new_log_std = (
                                            self.n_seen_samples*(self.train_statistics[feature_type]['LogStd'] + (self.train_statistics[feature_type]['LogMean'] - new_mean)**2) + n_current_samples*(current_combined_DS_statistics['LogStd'] + (current_combined_DS_statistics['LogMean'] - new_mean)**2)
                                        ) / (self.n_seen_samples + n_current_samples)
                                # Log Min
                                new_log_min = min(self.train_statistics[feature_type]['LogMin'], current_combined_DS_statistics['LogMin'])
                                # Log Max
                                new_log_max = min(self.train_statistics[feature_type]['LogMax'], current_combined_DS_statistics['LogMax'])
                            
                                # Update the with the new values
                                self.train_statistics[feature_type]['LogMean'] = new_log_mean
                                self.train_statistics[feature_type]['LogStd'] = new_log_std
                                self.train_statistics[feature_type]['LogMin'] = new_log_min
                                self.train_statistics[feature_type]['LogMax'] = new_log_max
                            except:
                                pass


                            # Update with the new values
                            try:
                                self.train_statistics[feature_type]['Mean'] = new_mean
                                self.train_statistics[feature_type]['Std'] = new_std
                                self.train_statistics[feature_type]['Min'] = new_min
                                self.train_statistics[feature_type]['Max'] = new_max
                            except:
                                pass
                    else:
                        # We do what we do before for each subfeature
                        for tmp_cont_feat_id in current_combined_DS_statistics[feature_type]:
                            try:
                                # Mean
                                new_mean = ( self.n_seen_samples * self.train_statistics[feature_type][tmp_cont_feat_id]['Mean'] + n_current_samples*current_combined_DS_statistics[tmp_cont_feat_id]['Mean'] ) / (self.n_seen_samples + n_current_samples)
                                # Std
                                new_std = (
                                            self.n_seen_samples*(self.train_statistics[feature_type][tmp_cont_feat_id]['Std'] + (self.train_statistics[feature_type][tmp_cont_feat_id]['Mean'] - new_mean)**2) + n_current_samples*(current_combined_DS_statistics[tmp_cont_feat_id]['Std'] + (current_combined_DS_statistics[tmp_cont_feat_id]['Mean'] - new_mean)**2)
                                        ) / (self.n_seen_samples + n_current_samples)
                                # Min
                                new_min = min(self.train_statistics[feature_type][tmp_cont_feat_id]['Min'], current_combined_DS_statistics[tmp_cont_feat_id]['Min'])
                                # Max
                                new_max = min(self.train_statistics[feature_type][tmp_cont_feat_id]['Max'], current_combined_DS_statistics[tmp_cont_feat_id]['Max'])
                            except:
                                pass
                        

                            # Particular case of central_to_central_edge_attr where we also have to compute the min, max, mean, and std of the logs
                            try:
                                # Log Mean
                                new_log_mean = ( self.n_seen_samples * self.train_statistics[feature_type][tmp_cont_feat_id]['LogMean'] + n_current_samples*current_combined_DS_statistics[tmp_cont_feat_id]['LogMean'] ) / (self.n_seen_samples + n_current_samples)
                                # Log Std
                                new_log_std = (
                                            self.n_seen_samples*(self.train_statistics[feature_type][tmp_cont_feat_id]['LogStd'] + (self.train_statistics[feature_type][tmp_cont_feat_id]['LogMean'] - new_mean)**2) + n_current_samples*(current_combined_DS_statistics[tmp_cont_feat_id]['LogStd'] + (current_combined_DS_statistics[tmp_cont_feat_id]['LogMean'] - new_mean)**2)
                                        ) / (self.n_seen_samples + n_current_samples)
                                # Log Min
                                new_log_min = min(self.train_statistics[feature_type][tmp_cont_feat_id]['LogMin'], current_combined_DS_statistics[tmp_cont_feat_id]['LogMin'])
                                # Log Max
                                new_log_max = min(self.train_statistics[feature_type][tmp_cont_feat_id]['LogMax'], current_combined_DS_statistics[tmp_cont_feat_id]['LogMax'])
                            
                                # Update the with the new values
                                self.train_statistics[feature_type][tmp_cont_feat_id]['LogMean'] = new_log_mean
                                self.train_statistics[feature_type][tmp_cont_feat_id]['LogStd'] = new_log_std
                                self.train_statistics[feature_type][tmp_cont_feat_id]['LogMin'] = new_log_min
                                self.train_statistics[feature_type][tmp_cont_feat_id]['LogMax'] = new_log_max
                            except:
                                pass


                            # Update with the new values
                            try:
                                self.train_statistics[feature_type][tmp_cont_feat_id]['Mean'] = new_mean
                                self.train_statistics[feature_type][tmp_cont_feat_id]['Std'] = new_std
                                self.train_statistics[feature_type][tmp_cont_feat_id]['Min'] = new_min
                                self.train_statistics[feature_type][tmp_cont_feat_id]['Max'] = new_max
                            except:
                                pass

        else:
            raise ValueError(f"Combination of dataset {self.parameters_exp['Dataset']['dataset_name']} and subdataset type {self.parameters_exp['Dataset']['subdataset']} is not valid.")

    def memoryUpdate(self):
        """
            Updates the memory for a replay-based CL experiment.
            IMPORTANT: The memory stores the raw samples, not the normalized ones, as they are going to 
            be normalized in the combined dataset.
        """
        if (self.parameters_exp['ContinualLearning']['replay']):
            # Initialize memory if necessary
            if (self.memory is None):
                self.memory = {}
            # Add samples if the memory is not full
            if (len(self.memory) < self.parameters_exp['ContinualLearning']['memory_capacity']):
                # Iterating over the sample in the current incremental dataset
                for i in range(len(self.train_ds)):
                    # Getting the original sample (not the normalized one as it is going to be re-normalized with the updated statistics in the next round)
                    sample = self.train_ds[i]
                    ID_in_origin = self.train_samples_IDs_in_origin[i]

                    # Getting the origin
                    sample_origin = sample.data_origin

                    # Getting the original sample
                    if (sample_origin == 'dataset'):
                        original_sample = self.train_incremental_ds[self.current_DS_ID][ID_in_origin]
                        self.memory[len(self.memory)] = deepcopy(original_sample) # len(self.memory) is always the ID of the last sample as the memory increases
            else:
                if (self.parameters_exp['ContinualLearning']['replay_strategy'] is None):
                    self.memory = {}
                elif (self.parameters_exp['ContinualLearning']['replay_strategy'].lower() == 'random'):
                    # Iterating over the sample in the current incremental dataset
                    for i in range(len(self.train_ds)):
                        # Getting the original sample (not the normalized one as it is going to be re-normalized with the updated statistics in the next round)
                        sample = self.train_ds[i]
                        ID_in_origin = self.train_samples_IDs_in_origin[i]

                        # Getting the origin
                        sample_origin = sample.data_origin

                        # Getting the original sample
                        if (sample_origin == 'dataset'):
                            original_sample = self.train_incremental_ds[self.current_DS_ID][ID_in_origin]
                        elif (sample_origin == 'memory'):
                            original_sample = self.memory[ID_in_origin]
                        
                        # If it is a new sample, we randomly decide to add it or not to the memory
                        if (sample_origin == 'dataset'):
                            if (np.random.random() < 0.5):
                                # Randomly choose a sample in memory to remove
                                list_mem_samples_IDs = list(self.memory.keys())
                                sample_ID_memory_remove = np.random.choice(list_mem_samples_IDs)
                                self.memory[sample_ID_memory_remove] = deepcopy(original_sample)


                elif (self.parameters_exp['ContinualLearning']['replay_strategy'].lower() == 'sims'):
                    pass
                    # TODO
                    raise NotImplementedError()
                elif (self.parameters_exp['ContinualLearning']['replay_strategy'].lower() == 'simsmodel'):
                    pass
                    # TODO
                    raise NotImplementedError()
                else:
                    raise ValueError(f"Replay strategy {self.parameters_exp['ContinualLearning']['replay_strategy']} is not valid.")
        else:
            self.memory = {}
            
        # TODO: FIND A WAY TO TRACK MEMORY TO SEE IF WE CAN SEE THE CATASTROPHIC FORGETTING IN THE MEMORY???

    def createCombinedTrainingDS(self):
        """
            Combines the current available training data with the data in the memory to 
            create the current training DS. It stores the origin of the data to be able
            to weight the different losses.
        """
        #======================================================================#
        # Current training DS
        self.train_ds = []

        # Ids of the samples in their original dataset or memory
        self.train_samples_IDs_in_origin = []


        # Iterating over the samples in the current incremental DS
        for tmp_sample_ID in range(len(self.train_incremental_ds[self.current_DS_ID])):
            tmp_sample = self.train_incremental_ds[self.current_DS_ID][tmp_sample_ID]
            sample_to_add = deepcopy(tmp_sample)
            sample_to_add.data_origin = "dataset"
            self.train_ds.append(sample_to_add)
            self.train_samples_IDs_in_origin.append(tmp_sample_ID)

        # Iterating over the samples in the memory
        if (self.memory is not None):
            for tmp_sample_ID in self.memory:
                sample_to_add = self.memory[tmp_sample_ID]
                sample_to_add.data_origin = "memory"
                self.train_ds.append(deepcopy(sample_to_add))
                self.train_samples_IDs_in_origin.append(tmp_sample_ID)



        # Mix the samples
        tmp_idx_train_samples = [i for i in range(len(self.train_ds))]
        shuffle(tmp_idx_train_samples)
        self.train_ds = [self.train_ds[i] for i in tmp_idx_train_samples]
        self.train_samples_IDs_in_origin = [self.train_samples_IDs_in_origin[i] for i in tmp_idx_train_samples]

        #======================================================================#
        # Get the original unnormalized validation and test datasets as they 
        # are going to be re-normalized with the updated statistics
        self.val_ds = self.raw_val_ds
        self.test_ds = self.raw_test_ds
        
    

    def computeForwardPass(self, batch, epoch_nb, batch_ID=None):
        #======================================================================#
        #======================================================================#
        #======================================================================#
        if (self.parameters_exp['Dataset']['dataset_name'].lower() == 'aiidkit') and\
            (self.parameters_exp['Dataset']['subdataset'].lower() == 'teav_static_graph_v1'):

            # Transform edge indices into ints
            for edge_type, edge_index in batch.edge_index_dict.items():
                # Ensure the tensor is integer type
                if (batch.edge_index_dict[edge_type].dtype == torch.float32):
                    batch.edge_index_dict[edge_type] = edge_index.to(torch.long)

            # Putting batch to correct device
            batch = batch.to(self.device)

            # Computing output
            out = self.model(batch) # shape [batch_size, out_channels]

            # Separating the samples by origin (memory or current dataset) to compute the different losses, it necessary
            if (self.parameters_exp['ContinualLearning']['replay']):
                # Separate predictions
                unique_data_origins = np.unique(batch.data_origin)

                # Creating masks for eac
                grouped_outputs = {i: out[batch.data_origin == np.array(i)] for i in unique_data_origins}
                grouped_labels = {i: batch.y[batch.data_origin == np.array(i)] for i in unique_data_origins}

                # Get the loss
                loss_replay = 0
                if (self.num_classes == 2):
                    if (not self.use_evidential_learning):
                        loss_dataset = self.criterion(grouped_outputs['dataset'].view(-1), grouped_labels['dataset'].float())
                        if ('memory' in grouped_outputs):
                            loss_replay = self.criterion(grouped_outputs['memory'].view(-1), grouped_labels['memory'].float())
                    else:
                        loss_dataset = self.criterion(grouped_outputs['dataset'], grouped_labels['dataset'])
                        if ('memory' in grouped_outputs):
                            loss_replay = self.criterion(grouped_outputs['memory'], grouped_labels['memory'])
                else:
                    loss_dataset = self.criterion(grouped_outputs['dataset'], grouped_labels['dataset'].squeeze())
                    if ('memory' in grouped_outputs):
                        loss_replay = self.criterion(grouped_outputs['memory'], grouped_labels['memory'].squeeze())

                total_loss = self.parameters_exp['ContinualLearning']['lambda_dataset']*loss_dataset + self.parameters_exp['ContinualLearning']['lambda_replay']*loss_replay
            else:
                # Getting the loss 
                if (self.num_classes == 2):
                    if (not self.use_evidential_learning):
                        loss_classif = self.criterion(out.view(-1), batch.y.float())
                    else:
                        loss_classif = self.criterion(out, batch.y)
                else:
                    loss_classif = self.criterion(out, batch.y.squeeze())
                total_loss = loss_classif

            # Adding the loss to the dictionary of losses
            loss = {
                        "total_loss": total_loss # DO NOT USE .item() unless you do not want to compute the gradient on it!
                    }
            if (self.parameters_exp['ContinualLearning']['replay']):
                loss['loss_dataset'] = loss_dataset
                loss['loss_replay'] = loss_replay

            # Getting the predictions
            if (self.num_classes == 2):
                if (self.use_evidential_learning):
                    # Getting the prediction probabilities
                    alphas = out # out is the alpha (evidence+1) as indicated in the code of Dirichlet layer of edl_pytorch
                    predictions_probs = alphas / torch.sum(alphas, dim=1, keepdim=True)
                    # Predictions
                    predictions = torch.argmax(predictions_probs, dim=1).detach().cpu().numpy()
                    predictions_probs = predictions_probs.detach().cpu().numpy()
                else:
                    # Prediction probabilities
                    decision_treshold = 0.5
                    if (out.ndim > 1 and out.size(-1) == 1):
                        predictions_probs = torch.sigmoid(out.view(-1)).detach().cpu().numpy()
                    else:
                        predictions_probs = torch.sigmoid(out).view(-1).detach().cpu().numpy()
                    # Predictions
                    predictions = (predictions_probs >= decision_treshold).astype(int)

                # Get true labels (must be 0/1 tensor)
                targets = batch.y.view(-1).cpu().numpy()

                # Fill the predictions dictionary
                preds = {
                            "targets": targets,
                            "predictions": predictions,
                            "predictions_probs": predictions_probs
                        }
                if (self.use_evidential_learning):
                    epistemic_uncert, aleatoric_uncert, total_uncert = get_uncertainties(alphas)
                    preds["epistemic_uncert"] = epistemic_uncert.detach().cpu().numpy()
                    preds["aleatoric_uncert"] = aleatoric_uncert.detach().cpu().numpy()
                    preds["total_uncert"] = total_uncert.detach().cpu().numpy()
            else:
                raise NotImplementedError("Not implemented yet, see code epidemio_informed_gnns.")
        else:
            raise ValueError(f"Combination of dataset {self.parameters_exp['Dataset']['dataset_name']} and subdataset type {self.parameters_exp['Dataset']['subdataset']} is not valid.")

        return loss, preds

    def initSingleTrain(self, create_new_model=True):
        """
            Initialize the parameters for a single train
        """
        print(f"\n\n==========> Initialize the single traininf for a CONTINUAL LEARNING setting <==========")
        # Create new combined training daset
        self.createCombinedTrainingDS()

        # Initialize the training with th classical parameters
        super().initSingleTrain(create_new_model=create_new_model)

        # Add data origin to val and test datasets for consistence


    def singleTrain(self, rep_ID, create_new_model=True):
        """
            Trains a model one time during self.parameters_exp['TrainingParams']['n_epochs'] epochs
        """
        # Indicate that no statistics have been compute to noramlize the datasets
        self.train_statistics = None
        self.n_seen_samples = 0

        # Reinitialize the model's weights
        reinitialize_model_weights = True

        # Iterating over the incremental datasets
        original_repetition_id = self.repetition_id
        for train_DS_ID in range(self.parameters_exp['n_incrementa_ds']):
            # Creating an ID for the training with the current DS
            self.repetition_id = f"{original_repetition_id}_DS-{train_DS_ID}"

            # Dataset is not normalized when new data arrive
            self.are_DS_normalized = False 

            # Set the current incremental DS to use
            self.current_DS_ID = train_DS_ID

            # Normally, the initSingleTrain should be done inside super().singleTrain
            # # Initialize the training WITHOUT reinitialization of the weights of the model
            # self.initSingleTrain(create_new_model=reinitialize_model_weights)
            # reinitialize_model_weights = False # We intialize the model's weights only once per training

            # Do a classic training with the current combined dataset
            super().singleTrain(rep_ID=rep_ID, create_new_model=reinitialize_model_weights)
            reinitialize_model_weights = False # We intialize the model's weights only once per training

            # Update memory
            self.memoryUpdate()


    def holdoutTrain(self, save_results=True):
        """
            Does a holdout training repeated self.parameters_exp['TrainingParams']['nb_repetitions'] times
        """
        # Reinitialite memory
        self.memory = None 

        # Do the holdout
        super().holdoutTrain(save_results=save_results)
    


#==============================================================================#
#================================ Main Function ================================#
#==============================================================================#
def main():
    #==========================================================================#
    # IGNOEING WARNING
    import warnings
    warnings.filterwarnings("ignore")
    #==========================================================================#


    print("\n\n==================== Beginning of the experiment ====================\n\n")
    #==========================================================================#
    # Fixing the random seed
    #seed = 42
    # seed = 7
    seed = 1
    random.seed(seed) # For reproducibility purposes
    np.random.seed(seed) # For reproducibility purposes
    torch.manual_seed(seed) # For reproducibility purposes
    if torch.cuda.is_available(): # For reproducibility purposes
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True) # For reproducibility purposes
    os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
    #os.environ["CUBLAS_WORKSPACE_CONFIG"]=":16:8"

    #==========================================================================#
    # Construct the argument parser
    ap = argparse.ArgumentParser()
    # Add the arguments to the parser
    ap.add_argument('--parameters_file', required=True, help="Yaml parameters for the experiment", type=str)
    args = vars(ap.parse_args())

    # Getting the value of the arguments
    parameters_file = args['parameters_file']
    with open(parameters_file, 'r') as file:
        parameters_exp = yaml.safe_load(file)

    #==========================================================================#
    # Creating an instance of the experiment
    exp = InfectionRiskPredContinualLearning(parameters_exp)

    # Creating directory to save the results
    inc = 0
    current_datetime = datetime.now().strftime("%d.%m.%Y_%H%M%S")
    # If execution is from the folder where the code is
    #resultsFolder = '../../../results/InfectionRiskPredContinualLearning/' + parameters_exp['exp_id'] + '_' + current_datetime
    # If execution is from src
    resultsFolder = './results/InfectionRiskPredContinualLearning/' + parameters_exp['exp_id'] + '_' + current_datetime
    while (os.path.isdir(resultsFolder+ '_' + str(inc))):
        inc += 1
    resultsFolder = resultsFolder + '_' + str(inc)
    os.mkdir(resultsFolder)
    exp.setResultsFolder(resultsFolder)
    print("===> Saving the results of the experiment in {}".format(resultsFolder))

    # Dataset Loading
    exp.createIncrementalDatasets()


    # Creating directories for the trained models, the training and testing metrics
    # and the parameters of the model (i.e. the training parameters and the network
    # architecture)
    os.mkdir(resultsFolder + '/model/')
    os.mkdir(resultsFolder + '/params_exp/')
    os.mkdir(resultsFolder + '/metrics/')

    # Normalizing the dataset
    # Done each time a new dataset is received

    # Saving the training parameters in the folder of the results
    inc = 0
    parameters_file_new = resultsFolder + '/params_exp/params_beginning' + '_'
    while (os.path.isfile(parameters_file_new + str(inc) + '.yaml')):
        inc += 1
    parameters_file_new = parameters_file_new + str(inc) +'.yaml'
    try:
        with open(parameters_file_new, 'w') as file:
            # Use yaml.dump() to write the data to the file
            # The 'sort_keys=False' argument keeps dictionary keys in the order they were defined (Python 3.7+),
            # which can make the file more readable.
            yaml.dump(parameters_exp, file, sort_keys=False)
        print(f"Successfully saved data to {parameters_file_new}")
    except Exception as e:
        print(f"An error occurred: {e}")


    # Doing holdout evaluation
    # Optuna or classical holdout
    if (parameters_exp['Optimization']['Optuna']['use_optuna']):
        exp.optuna_tunning() 
    else:
        # Doing holdout evaluation
        exp.holdoutTrain(save_results=True)
        #exp.holdoutTrain(save_results=False)

    # Saving the training parameters in the folder of the results
    inc = 0
    parameters_file_new = resultsFolder + '/params_exp/params' + '_'
    while (os.path.isfile(parameters_file_new + str(inc) + '.yaml')):
        inc += 1
    parameters_file_new = parameters_file_new + str(inc) +'.yaml'
    try:
        with open(parameters_file_new, 'w') as file:
            # Use yaml.dump() to write the data to the file
            # The 'sort_keys=False' argument keeps dictionary keys in the order they were defined (Python 3.7+),
            # which can make the file more readable.
            yaml.dump(parameters_exp, file, sort_keys=False)
        print(f"Successfully saved data to {parameters_file_new}")
    except Exception as e:
        print(f"An error occurred: {e}")

    # Saving the python file containing the network architecture
    if (parameters_exp['Model']['model_to_use'].lower() == "simpleheterognn"):
        shutil.copy2('src/model/GraphBased/AiidkitTEAVGraphEmbedder.py', resultsFolder + '/model/graph_embedder_architecture.py')
        shutil.copy2('src/model/GraphBased/HeteroGNN.py', resultsFolder + '/model/architecture.py')
    elif (parameters_exp['Model']['model_to_use'].lower() == "heterographsage"):
        shutil.copy2('src/model/GraphBased/AiidkitTEAVGraphEmbedder.py', resultsFolder + '/model/graph_embedder_architecture.py')
        shutil.copy2('src/model/GraphBased/HeteroGraphSage.py', resultsFolder + '/model/graph_classifier_architecture.py')
    else:
        raise ValueError()
    
    # Save the data distribution
    # TODO: Copy HDF5 dataset files?

    #==========================================================================#
    # Writing or modifying params files if Optuna was used
    if ('Optimization' in parameters_exp):
        if ('Optuna' in parameters_exp['Optimization']):
            if ('use_optuna' in parameters_exp['Optimization']['Optuna']):
                if (parameters_exp['Optimization']['Optuna']['use_optuna']):
                    # Get Cutoff Value and pred Horizon
                    cutoff_val = int([el.split('-') for el in parameters_file.split('/') if 'Cutoff' in el][0][-1])
                    pred_hor_val = int([el.split('-') for el in parameters_file.split('/') if 'PredHorizon' in el][0][-1])
                    if ('Evidential_Optuna' not in parameters_file):
                        # Getting the path of the file to modify or create
                        # Simple holdout train (without Optuna nor Evidential Learning)
                        simple_train_yaml = ''.join(parameters_file.split('_Optuna'))
                        if (os.path.exists(simple_train_yaml)):
                            with open(simple_train_yaml, "r") as file:
                                simple_train_params = yaml.safe_load(file)  # Loads YAML as Python dict 
                        else:
                            simple_train_params = deepcopy(parameters_exp)
                        # Optuna for evidential learning
                        optuna_evidential_yaml = parameters_file.split('_Optuna')[0] + "Evidential_Optuna.yaml"
                        if (os.path.exists(optuna_evidential_yaml)):
                            with open(optuna_evidential_yaml, "r") as file:
                                optuna_evidential_params = yaml.safe_load(file)  # Loads YAML as Python dict 
                        else:
                            optuna_evidential_params = deepcopy(parameters_exp)

                        # Getting the parameters of the best trial
                        best_params = exp.study.best_params
                        # Define global parameters (common to all experiments)
                        simple_train_params['TrainingParams']['lr'] = best_params['learning_rate']
                        #simple_train_params['TrainingParams']['weight_decay'] = best_params['weight_decay']
                        #simple_train_params['Optimization']['optimizer'] = best_params['optimizer']
                        # Define per-model parameters
                        if (not exp.use_evidential_learning):
                            simple_train_params['Model']['hidden_channels'] = best_params["hidden_channels"]
                            #simple_train_params['Model']['graph_pool_strategy'] = best_params["graph_pool_strategy"]
                            simple_train_params['Model']['graph_pool_fusion'] = best_params["graph_pool_fusion"]
                            if (simple_train_params['Model']['model_to_use'].lower() == 'simpleheterognn'):
                                pass # All the parameters where already defined
                            elif (simple_train_params['Model']['model_to_use'].lower() == 'heterographsage'):
                                simple_train_params['Model']['dropout'] = best_params["dropout"]
                                simple_train_params['Model']["num_layers"] = best_params["num_layers"]
                            elif (simple_train_params['Model']['model_to_use'].lower() == "heterogat"):
                                simple_train_params['Model']['dropout'] = best_params["dropout"]
                                simple_train_params['Model']["num_layers"] = best_params["num_layers"]
                                simple_train_params['Model']["heads"] = best_params["heads"]
                            else:
                                raise ValueError(f"\nModel to use {simple_train_params['Model']['model_to_use'].lower()} is not valid for Optuna hyper-parameter tuning\n.")    
                        simple_train_params['Optimization']['Optuna']['use_optuna'] = False
                        simple_train_params['Optimization']['Optuna']['optuna_starting_point_fn'] = None
                        simple_train_params['Optimization']['EarlyStopping']['use_early_stopping'] = False
                        simple_train_params['exp_id'] = simple_train_yaml.split('/')[-1].split('.yaml')[0] + f"_Cutoff-{cutoff_val}_PredHor-{pred_hor_val}"
                            
                        # Add the LR for the single train parameters
                        optuna_evidential_params = deepcopy(simple_train_params)
                        optuna_evidential_params['TrainingParams']['lr'] = None
                        optuna_evidential_params['Optimization']['Optuna']['use_optuna'] = True
                        optuna_evidential_params['Optimization']['EarlyStopping']['use_early_stopping'] = True
                        optuna_evidential_params['Optimization']['loss_function'] = "EvidentialLearningLoss"
                        optuna_evidential_params['Optimization']['EvidentialLoss'] = {
                                                                                        "lambda_evidential_sched": True,
                                                                                        "lambda_evidential": None
                                                                                    }

                        optuna_evidential_params['exp_id'] = optuna_evidential_yaml.split('/')[-1].split('.yaml')[0] + f"_Cutoff-{cutoff_val}_PredHor-{pred_hor_val}"
                        
                        # Write back the Yaml params files
                        # Simple holdout train (without Optuna nor Evidential Learning)
                        with open(simple_train_yaml, "w") as file:
                            yaml.safe_dump(simple_train_params, file)
                        print(f"\n ===> New config file created or modified at location: {simple_train_yaml} \n")
                        # Optuna for evidential learning
                        with open(optuna_evidential_yaml, "w") as file:
                            yaml.safe_dump(optuna_evidential_params, file)
                        print(f"\n ===> New config file created or modified at location: {optuna_evidential_yaml} \n")
                            
                    else:
                        # Getting the path of the file to modify or create
                        # Simple holdout train (without Optuna but with Evidential Learning)
                        simple_evidential_train_yaml = ''.join(parameters_file.split('_Optuna'))
                        if (os.path.exists(simple_evidential_train_yaml)):
                            with open(simple_evidential_train_yaml, "r") as file:
                                simple_evidential_train_params = yaml.safe_load(file)  # Loads YAML as Python dict 
                        else:
                            simple_evidential_train_params = deepcopy(parameters_exp)

                        # Getting the parameters of the best trial
                        best_params = exp.study.best_params
                        # Define global parameters (common to all experiments)
                        simple_evidential_train_params['TrainingParams']['lr'] = best_params['learning_rate']
                        #simple_evidential_train_params['TrainingParams']['weight_decay'] = best_params['weight_decay']
                        #simple_evidential_train_params['Optimization']['optimizer'] = best_params['optimizer']
                        simple_evidential_train_params['Optimization']['EvidentialLoss']['lambda_evidential'] = best_params['lambda_evidential']
                        simple_evidential_train_params['Optimization']['Optuna']['use_optuna'] = False
                        simple_evidential_train_params['Optimization']['Optuna']['optuna_starting_point_fn'] = None
                        simple_evidential_train_params['Optimization']['EarlyStopping']['use_early_stopping'] = False
                        simple_evidential_train_params['exp_id'] = simple_evidential_train_yaml.split('/')[-1].split('.yaml')[0] + f"_Cutoff-{cutoff_val}_PredHor-{pred_hor_val}"

                        # Write back the Yaml params files
                        with open(simple_evidential_train_yaml, "w") as file:
                            yaml.safe_dump(simple_evidential_train_params, file)
                        print(f"\n ===> New config file created or modified at location: {simple_evidential_train_yaml} \n")



    #==========================================================================#
    print("\n\n==================== End of the experiment ====================\n\n")



if __name__=="__main__":
    main()