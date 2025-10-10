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
from src.experiments.Utils.EvidentialClassification import EvidentialClassification
from src.experiments.Utils.tools import get_uncertainties

#====================================================================================================#
#====================================================================================================#
#====================================================================================================#
# Abstract class
class InfectionRiskPred(GenericExperiment):
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


    def createTorchDatasets(self, verbose=True):
        """
            Create the torch datasets associated needed to train and evaluate the model
        """
        if ('Train' not in self.parameters_exp['Dataset']['hdf5_dataset_filename']) or ('Test' not in self.parameters_exp['Dataset']['hdf5_dataset_filename']):
            raise RuntimeError("Train and Test datasets must be specified (Validation is optional).")
        else:
            if (self.parameters_exp['Dataset']['dataset_name'].lower() == 'aiidkit'):
                if (self.parameters_exp['Dataset']['subdataset'].lower() == 'teav_static_graph_v1'):
                    # Train
                    self.train_ds = create_PyGeo_Graph_DS_from_HDF5(
                                                                        h5_fn=self.parameters_exp['Dataset']['hdf5_dataset_filename']['Train'],
                                                                        label_type=self.parameters_exp['Dataset']['label_type']
                                                                    )
                    # Validation
                    if ('Validation' in self.parameters_exp['Dataset']['hdf5_dataset_filename']):
                        self.val_ds = create_PyGeo_Graph_DS_from_HDF5(
                                                                            h5_fn=self.parameters_exp['Dataset']['hdf5_dataset_filename']['Validation'],
                                                                            label_type=self.parameters_exp['Dataset']['label_type']
                                                                        ) 
                    else:
                        self.val_ds = []
                    # Test
                    self.test_ds = create_PyGeo_Graph_DS_from_HDF5(
                                                                        h5_fn=self.parameters_exp['Dataset']['hdf5_dataset_filename']['Test'],
                                                                        label_type=self.parameters_exp['Dataset']['label_type']
                                                                    )
                    
                    # Load important data for GNNs
                    graph_ds_folder_name = '/'.join(self.parameters_exp['Dataset']['hdf5_dataset_filename']['Train'].split('/')[:-1]) + '/'
                    possible_values_all_patients_fn = graph_ds_folder_name + 'possible_values_all_patients.pkl'
                    with open(possible_values_all_patients_fn, mode='rb') as pf:
                        self.possible_values_all_patients = pickle.load(pf)
                    categorical_ent_attr_pairs_fn = graph_ds_folder_name + 'categorical_ent_attr_pairs.pkl'
                    with open(categorical_ent_attr_pairs_fn, mode='rb') as pf:
                        self.categorical_ent_attr_pairs = pickle.load(pf)
                else:
                    raise ValueError(f"Subsdataset {self.parameters_exp['Dataset']['subdataset']} is not valid")
            else:
                raise ValueError(f"Dataset {self.parameters_exp['Dataset']['dataset_name']} is not valid")
            
    
    def addClassWeightsLoss(self, multiclass_strategy='balanced'):
        """
            Compute class weights and redefines the loss function using
            these class weights to handle imbalanced classes.
        """
        # Compute class weights
        if (self.parameters_exp['Dataset']['dataset_name'].lower() == 'aiidkit') and\
            (self.parameters_exp['Dataset']['subdataset'].lower() == 'teav_static_graph_v1'):
                # Getting all the labels
                y_all = []
                for data in self.train_ds:
                    y = data.y.view(-1).cpu()
                    y_all.append(y)
                y_all = torch.cat(y_all).numpy()

                # Computing the number of classes if not given
                if self.num_classes is None:
                    self.num_classes = int(y_all.max()) + 1

                # Number of samples per class
                counts = np.bincount(y_all, minlength=self.num_classes).astype(float)
                total = counts.sum()

                # Computing weights
                if (self.num_classes == 2) and (not self.use_evidential_learning):
                    num_neg = counts[0]
                    num_pos = counts[1]
                    if num_pos == 0 or num_neg == 0:
                        print("WARNING: Dataset has only one class. Defaulting pos_weight=1.0")
                        self.class_weights_to_use = torch.tensor(1.0, device=self.device)
                    else:
                        self.class_weights_to_use = torch.tensor(num_neg / num_pos, device=self.device)

                    print(f"\n===> Found {num_pos} positive and {num_neg} negative samples.")
                    print(f"\n===> pos_weight = {self.class_weights_to_use.item():.4f}") 
                else:
                    if multiclass_strategy == "balanced":
                        weights = total / (self.num_classes * counts)
                    elif multiclass_strategy == "inverse":
                        weights = 1.0 / counts
                    elif multiclass_strategy == "sqrt_inv":
                        weights = 1.0 / np.sqrt(counts)
                    else:
                        raise ValueError(f"Unknown multiclass strategy '{multiclass_strategy}'")

                    # Normalize to mean 1.0 (optional but helps stability)
                    weights = weights / weights.mean()

                    self.class_weights_to_use = torch.tensor(weights, dtype=torch.float32, device=self.device)

                    print(f"\n===> Class counts: {counts.astype(int)}")
                    print(f"\n===> Weights ({multiclass_strategy}): {weights.tolist()}")
        else:
            raise ValueError(f"Combination of dataset {self.parameters_exp['Dataset']['dataset_name']} and subdataset type {self.parameters_exp['Dataset']['subdataset']} is not valid.")

        # Defining the new loss
        if (self.parameters_exp['Optimization']['loss_function'].lower() == "ce"):
            if (self.num_classes > 2):
                self.criterion = torch.nn.CrossEntropyLoss(weight=self.class_weights_to_use)
            else:
                self.criterion = torch.nn.BCEWithLogitsLoss(pos_weight=self.class_weights_to_use)
        elif (self.use_evidential_learning):
            self.criterion = EvidentialClassification(lamb=self. parameters_exp['Optimization']['EvidentialLoss']['lambda_evidential'], class_weights=self.class_weights_to_use)
    
    def normalizeDataset(self):
        """
            Normalize the dataset by substracting the mean and dividing by
            the std
        """
        if (self.parameters_exp['Dataset']['dataset_name'].lower() == 'aiidkit') and\
            (self.parameters_exp['Dataset']['subdataset'].lower() == 'teav_static_graph_v1'):
                # Compute statistics on the training dataset
                train_statistics = compute_statistics_dataset(self.train_ds)

                # Normalize datasets
                self.train_ds = normalize_dataset(dataset=self.train_ds, statistics=train_statistics)
                if (len(self.val_ds) > 0):
                    self.val_ds = normalize_dataset(dataset=self.val_ds, statistics=train_statistics)
                self.test_ds = normalize_dataset(dataset=self.test_ds, statistics=train_statistics)
        else:
            raise ValueError(f"Combination of dataset {self.parameters_exp['Dataset']['dataset_name']} and subdataset type {self.parameters_exp['Dataset']['subdataset']} is not valid.")

    
    def dataloadersCreation(self):
        """
            Create the train and test dataloader necessary to train and test a
            deep learning model
        """
        if (self.parameters_exp['Dataset']['dataset_name'].lower() == 'aiidkit'):
            if (self.parameters_exp['Dataset']['subdataset'].lower() == 'teav_static_graph_v1'):
                # Train
                self.train_loader = torch_geometric.loader.DataLoader(self.train_ds, batch_size=self.parameters_exp['TrainingParams']['batch_size_train'])  # batching heterogeneous graphs
                
                # Validation
                self.val_loader = torch_geometric.loader.DataLoader(self.val_ds, batch_size=self.parameters_exp['TrainingParams']['batch_size_val'])  # batching heterogeneous graphs

                # Test
                self.test_loader = torch_geometric.loader.DataLoader(self.test_ds, batch_size=self.parameters_exp['TrainingParams']['batch_size_test'])  # batching heterogeneous graphs
            else:
                raise ValueError(f"Subsdataset {self.parameters_exp['Dataset']['subdataset']} is not valid")
        else:
            raise ValueError(f"Dataset {self.parameters_exp['Dataset']['dataset_name']} is not valid")

    
    def modelCreation(self):
        """
            Creates a model to be trained on the selected time-frequency
            representation
        """
        if (self.parameters_exp['Model']['model_type'].lower() == "gnn"):
            if (self.parameters_exp['Model']['model_to_use'].lower() == "simpleheterognn"):
                print("\n==========> Using HeteroGNN GNN <==========\n")
                self.model = HeteroGNN(
                                    hidden_channels=self.parameters_exp['Model']['hidden_channels'],
                                    out_channels=self.parameters_exp['Model']['out_channels'],
                                    possible_values_all_patients=self.possible_values_all_patients,
                                    categorical_ent_attr_pairs=self.categorical_ent_attr_pairs,
                                    metadata=self.train_ds[0].metadata(),
                                    graph_pool_strategy=self.parameters_exp['Model']['graph_pool_strategy'],
                                    graph_pool_fusion=self.parameters_exp['Model']['graph_pool_fusion'],
                                    evidential=self.use_evidential_learning
                                ).to(self.device)
            elif (self.parameters_exp['Model']['model_to_use'].lower() == "heterographsage"):
                print("\n==========> Using HeteroGraphSage GNN <==========\n")
                self.model = HeteroGraphSage(
                                            in_channels={"central": 1, "child_cont": 9, "child_categ": 8},
                                            hidden_channels=self.parameters_exp['Model']['hidden_channels'],
                                            out_channels=self.parameters_exp['Model']['out_channels'],
                                            dropout=self.parameters_exp['Model']["dropout"],
                                            num_layers=self.parameters_exp['Model']["num_layers"],
                                            possible_values_all_patients=self.possible_values_all_patients,
                                            categorical_ent_attr_pairs=self.categorical_ent_attr_pairs,
                                            metadata=self.train_ds[0].metadata(),
                                            graph_pool_strategy=self.parameters_exp['Model']['graph_pool_strategy'],
                                            graph_pool_fusion=self.parameters_exp['Model']['graph_pool_fusion'],
                                            act='ReLU',
                                            aggr='mean',
                                            evidential=self.use_evidential_learning
                                        ).to(self.device) 
            else:
                raise ValueError("Model to use {} is not valid".format(self.parameters_exp['Model']['model_to_use']))


    def computeForwardPass(self, batch, epoch_nb, batch_ID=None):
        #======================================================================#
        #======================================================================#
        #======================================================================#
        if (self.parameters_exp['Dataset']['dataset_name'].lower() == 'aiidkit') and\
            (self.parameters_exp['Dataset']['subdataset'].lower() == 'teav_static_graph_v1'):
            # Putting batch to correct device
            batch = batch.to(self.device)

            # Computing output
            out = self.model(batch) # shape [batch_size, out_channels]

            # Getting the loss function
            if (self.num_classes == 2):
                if (not self.use_evidential_learning):
                    loss_classif = self.criterion(out.view(-1), batch.y.float())
                else:
                    loss_classif = self.criterion(out, batch.y)
            else:
                loss_classif = self.criterion(out, batch.y.squeeze())

            # Adding the loss to the dictionary of losses
            loss = {
                        "total_loss": loss_classif # DO NOT USE .item() unless you do not want to compute the gradient on it!
                    }

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
                raise NotImplementedError("Not implemented yes, see code epidemio_informed_gnns.")
        else:
            raise ValueError(f"Combination of dataset {self.parameters_exp['Dataset']['dataset_name']} and subdataset type {self.parameters_exp['Dataset']['subdataset']} is not valid.")

        return loss, preds

    
    def singleTrain(self, rep_ID):
        """
            Trains a model one time during self.parameters_exp['TrainingParams']['n_epochs'] epochs
        """
        # Doing all on the hdf5 files result
        with h5py.File(self.repetitions_results_fn, 'a') as h5f_rep_results:
            # Initialization
            self.initSingleTrain()

            # Data structures for the losses and predictions
            # Losses
            h5f_rep_results.create_group(f"Rep-{self.repetition_id}/Loss")
            h5f_rep_results.create_group(f"Rep-{self.repetition_id}/Loss/Train")
            h5f_rep_results.create_group(f"Rep-{self.repetition_id}/Loss/Val")
            h5f_rep_results.create_group(f"Rep-{self.repetition_id}/Loss/Test")
            # Predictions
            h5f_rep_results.create_group(f"Rep-{self.repetition_id}/Preds")
            h5f_rep_results.create_group(f"Rep-{self.repetition_id}/Preds/Train")
            h5f_rep_results.create_group(f"Rep-{self.repetition_id}/Preds/Val")
            h5f_rep_results.create_group(f"Rep-{self.repetition_id}/Preds/Test")

            # Epochs
            seen_loss_types = []
            for epoch in tqdm(range(self.parameters_exp['TrainingParams']['n_epochs'])):
                # Update lambda evidential learning if asked
                if (self.use_evidential_learning) and\
                   (self.parameters_exp['Optimization']['EvidentialLoss']['lambda_evidential_sched']):
                    self.scheduling_lambda_evidential(epoch)
                # Training
                self.test_mode = False
                self.model.train()
                tmp_train_losses = {}
                tmp_train_preds = []
                for batch in tqdm(self.train_loader):
                    # Compute forward pass
                    train_loss, train_preds = self.updateModel(batch, epoch)
            
                    # Add predictions and losses to list
                    tmp_train_preds.append(train_preds)
                    for loss_type in train_loss:
                        if (loss_type not in tmp_train_losses):
                            tmp_train_losses[loss_type] = []
                        try:
                            tmp_train_losses[loss_type].append(train_loss[loss_type].detach().data.cpu().numpy())
                        except:
                            tmp_train_losses[loss_type].append(train_loss[loss_type])
                # Mean performance in the batch
                for loss_type in tmp_train_losses:
                    if (f"Rep-{self.repetition_id}/Loss/Train/{loss_type}" not in h5f_rep_results):
                        h5f_rep_results.create_dataset(f"Rep-{self.repetition_id}/Loss/Train/{loss_type}", shape=(self.parameters_exp['TrainingParams']['n_epochs'],), dtype='f')
                        seen_loss_types.append(loss_type)
                    h5f_rep_results[f"Rep-{self.repetition_id}/Loss/Train/{loss_type}"][epoch] = np.mean(tmp_train_losses[loss_type])
                # Preds
                if (epoch % self.epochs_step_save_preds == 0) or (epoch == self.parameters_exp['TrainingParams']['n_epochs']-1):
                    # Create group for the current epoch
                    h5f_rep_results.create_group(f"Rep-{self.repetition_id}/Preds/Train/Epoch-{epoch}")
                    
                    # Concatenating the results of the batches
                    for pred_type in list(tmp_train_preds[0].keys()):
                        tmp_pred = np.concatenate([tmp_train_preds[batch_ID][pred_type] for batch_ID in range(len(tmp_train_preds))])
                        h5f_rep_results.create_dataset(f"Rep-{self.repetition_id}/Preds/Train/Epoch-{epoch}/{pred_type}", data=tmp_pred)


                # Validation
                self.test_mode = False # As the validation set has the same characteristics as the training set
                tmp_val_losses, tmp_val_preds = self.evalCurrentModel(self.val_loader, epoch)
                # Loss
                for loss_type in tmp_val_losses:
                    if (f"Rep-{self.repetition_id}/Loss/Val/{loss_type}" not in h5f_rep_results):
                        h5f_rep_results.create_dataset(f"Rep-{self.repetition_id}/Loss/Val/{loss_type}", shape=(self.parameters_exp['TrainingParams']['n_epochs'],), dtype='f')
                    h5f_rep_results[f"Rep-{self.repetition_id}/Loss/Val/{loss_type}"][epoch] = np.mean(tmp_val_losses[loss_type])
                # Preds
                if (epoch % self.epochs_step_save_preds == 0) or (epoch == self.parameters_exp['TrainingParams']['n_epochs']-1):
                    # Create group for the current epoch
                    h5f_rep_results.create_group(f"Rep-{self.repetition_id}/Preds/Val/Epoch-{epoch}")
                    
                    # Concatenating the results of the batches
                    for pred_type in list(tmp_val_preds[0].keys()):
                        tmp_pred = np.concatenate([tmp_val_preds[batch_ID][pred_type] for batch_ID in range(len(tmp_val_preds))])
                        h5f_rep_results.create_dataset(f"Rep-{self.repetition_id}/Preds/Val/Epoch-{epoch}/{pred_type}", data=tmp_pred)
                
                # LR scheduler
                # if (self.parameters_exp['Optimization']['optimizer'].lower() in ['adam', 'adamw']):
                #     # print("\n\n (BEFORE) Doing scheduler update (lr = {})".format(self.sched.get_last_lr()))
                #     self.sched.step(loss_values['Val'][epoch]) # For ReduceLROnPlateau
                #     # self.sched.step() # For MultiStepLR
                #     # print("(AFTER) Doing scheduler update (lr = {})\n\n".format(self.sched.get_last_lr()))

                # Test the model
                self.test_mode = True
                tmp_test_losses, tmp_test_preds = self.evalCurrentModel(self.test_loader, epoch)
                # Loss
                for loss_type in tmp_test_losses:
                    if (f"Rep-{self.repetition_id}/Loss/Test/{loss_type}" not in h5f_rep_results):
                        h5f_rep_results.create_dataset(f"Rep-{self.repetition_id}/Loss/Test/{loss_type}", shape=(self.parameters_exp['TrainingParams']['n_epochs'],), dtype='f')
                    h5f_rep_results[f"Rep-{self.repetition_id}/Loss/Test/{loss_type}"][epoch] = np.mean(tmp_test_losses[loss_type])
                # Preds
                if (epoch % self.epochs_step_save_preds == 0) or (epoch == self.parameters_exp['TrainingParams']['n_epochs']-1):
                    # Create group for the current epoch
                    h5f_rep_results.create_group(f"Rep-{self.repetition_id}/Preds/Test/Epoch-{epoch}")
                    
                    # Concatenating the results of the batches
                    for pred_type in list(tmp_test_preds[0].keys()):
                        tmp_pred = np.concatenate([tmp_test_preds[batch_ID][pred_type] for batch_ID in range(len(tmp_test_preds))])
                        h5f_rep_results.create_dataset(f"Rep-{self.repetition_id}/Preds/Test/Epoch-{epoch}/{pred_type}", data=tmp_pred)

                print("================================================================================")
                print("LOSS AND METRICS\n")
                for loss_type in seen_loss_types:
                    print(f"\n=========> Loss type {loss_type} <=========")
                    print("\tTrain loss at epoch {} is {}".format(epoch, h5f_rep_results[f"Rep-{self.repetition_id}/Loss/Train/{loss_type}"][epoch]))
                    print("\t\tVal loss at epoch {} is {}".format(epoch, h5f_rep_results[f"Rep-{self.repetition_id}/Loss/Val/{loss_type}"][epoch]))
                    print("\t\tTest loss at epoch {} is {}".format(epoch, h5f_rep_results[f"Rep-{self.repetition_id}/Loss/Test/{loss_type}"][epoch]))
                # TODO: IMPLEMENT OTHER METRICS SUCH AS:
                print("================================================================================\n\n")

                # IMPORTANT: SAVING JIT MODELS FOR STM GNN MODELS DOES NOT WORK AS STM GNN INFER THE SHAPE OF AN INPUT BASED ON WHAT WE PASS TO THE MODEL AND JIT DOES NOT SUPPORT THAT
                # # Saving the current model(s)
                # if hasattr(self, 'model'):
                #     if (self.model is not None):
            #             model_jit = torch.jit.script(self.model) # Export to TorchScript
            #             model_jit.save(self.results_folder + '/model/JIT_current_model-{}_epoch-{}_holdout-{}_rep-{}.pth'.format(self.exp_id, epoch, self.holdout_train_id, self.repetition_id)) # Save
            #             # if (os.path.exists(self.results_folder + '/model/JIT_current_model-{}_epoch-{}_holdout-{}_rep-{}.pth'.format(self.exp_id, epoch-1, self.holdout_train_id, self.repetition_id))):
            #                 # os.remove(self.results_folder + '/model/JIT_current_model-{}_epoch-{}_holdout-{}_rep-{}.pth'.format(self.exp_id, epoch-1, self.holdout_train_id, self.repetition_id))
    

    def evalCurrentModel(self, dataloader, epoch):
        # Evaluation
        self.model.eval()
        tmp_losses = {}
        tmp_preds = []
        # IMPORTANT: HERE WE DO NOT USE WITH TORCH NO GRAD AS WE NEED TO COMPUTE THE GRADIENTS FOR THE VALIDATION AND TEST LOSSES
        for batch in tqdm(dataloader):
            # Forward pass
            loss, preds = self.computeForwardPass(batch, epoch)

            # Add predictions and losses to list
            tmp_preds.append(preds)
            for loss_type in loss:
                if (loss_type not in tmp_losses):
                    tmp_losses[loss_type] = []
                try:
                    tmp_losses[loss_type].append(loss[loss_type].detach().data.cpu().numpy())
                except:
                    tmp_losses[loss_type].append(loss[loss_type])
        self.optimizer.zero_grad()

        return tmp_losses, tmp_preds
    



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
    exp = InfectionRiskPred(parameters_exp)

    # Creating directory to save the results
    inc = 0
    current_datetime = datetime.now().strftime("%d.%m.%Y_%H%M%S")
    # If execution is from the folder where the code is
    #resultsFolder = '../../../results/InfectionRiskPred/' + parameters_exp['exp_id'] + '_' + current_datetime
    # If execution is from src
    resultsFolder = './results/InfectionRiskPred/' + parameters_exp['exp_id'] + '_' + current_datetime
    while (os.path.isdir(resultsFolder+ '_' + str(inc))):
        inc += 1
    resultsFolder = resultsFolder + '_' + str(inc)
    os.mkdir(resultsFolder)
    exp.setResultsFolder(resultsFolder)
    print("===> Saving the results of the experiment in {}".format(resultsFolder))

    # Dataset Loading
    exp.createTorchDatasets()

    # Add class weights
    exp.addClassWeightsLoss(multiclass_strategy='balanced')
    #exp.addClassWeightsLoss(multiclass_strategy='inverse')
    #exp.addClassWeightsLoss(multiclass_strategy='sqrt_inv')

    # Creating directories for the trained models, the training and testing metrics
    # and the parameters of the model (i.e. the training parameters and the network
    # architecture)
    os.mkdir(resultsFolder + '/model/')
    os.mkdir(resultsFolder + '/params_exp/')
    os.mkdir(resultsFolder + '/metrics/')

    # Normalizing the dataset
    #exp.computeDatasetMeanStd()
    #exp.normalizeDataset()

    # Saving the training parameters in the folder of the results
    inc = 0
    parameters_file = resultsFolder + '/params_exp/params_beginning' + '_'
    while (os.path.isfile(parameters_file + str(inc) + '.yaml')):
        inc += 1
    parameters_file = parameters_file + str(inc) +'.yaml'
    try:
        with open(parameters_file, 'w') as file:
            # Use yaml.dump() to write the data to the file
            # The 'sort_keys=False' argument keeps dictionary keys in the order they were defined (Python 3.7+),
            # which can make the file more readable.
            yaml.dump(parameters_exp, file, sort_keys=False)
        print(f"Successfully saved data to {parameters_file}")
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
    parameters_file = resultsFolder + '/params_exp/params' + '_'
    while (os.path.isfile(parameters_file + str(inc) + '.yaml')):
        inc += 1
    parameters_file = parameters_file + str(inc) +'.yaml'
    try:
        with open(parameters_file, 'w') as file:
            # Use yaml.dump() to write the data to the file
            # The 'sort_keys=False' argument keeps dictionary keys in the order they were defined (Python 3.7+),
            # which can make the file more readable.
            yaml.dump(parameters_exp, file, sort_keys=False)
        print(f"Successfully saved data to {parameters_file}")
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
    print("\n\n==================== End of the experiment ====================\n\n")



if __name__=="__main__":
    main()