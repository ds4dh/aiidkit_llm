#!/usr/bin/env python3
"""
    This code implements some basic classes and functions for a generic deep
    learning experiment
    This code CANNOT BE USE BY ITSELF, as there is no main function because
    the loss function to use will depend on what we want to do, so it cannot
    be generic.
"""
import os
import yaml
from abc import ABC, abstractmethod
from tqdm import tqdm
from random import shuffle
from pathlib import Path
import h5py
import numpy as np
from sklearn.metrics import matthews_corrcoef,\
                            f1_score,\
                            balanced_accuracy_score,\
                            classification_report,\
                            roc_auc_score
import torch
import optuna

# Internal imports
from src.experiments.Utils.EvidentialClassification import EvidentialClassification


#====================================================================================================#
#====================================================================================================#
#====================================================================================================#
# Function to get the classification metrics
def get_classification_metrics(targets, preds, data_split='Test', n_unique_classes=None, verbose=False, print_classification_report=False):
    # Getting predictions for random classifier
    if (n_unique_classes is None):
        n_unique_classes = np.unique(targets)
    random_pred = torch.randint(low=0, high=max(n_unique_classes), size=( len(preds), 1)).detach().cpu().numpy()

    # Balanced accuracy
    balanced_acc = balanced_accuracy_score(targets, preds, adjusted=False)
    balanced_acc_adjusted = balanced_accuracy_score(targets, preds, adjusted=True)
    balanced_acc_random = balanced_accuracy_score(targets, random_pred, adjusted=False)
    balanced_acc_adjusted_random = balanced_accuracy_score(targets, random_pred, adjusted=True)
    if (verbose):
        print(f"\n{data_split} balanced accuracy: {balanced_acc*100}%")
        print(f"\n{data_split} balanced accuracy (adjusted): {balanced_acc_adjusted*100}%")
        print(f"\t{data_split} balanced accuracy random classifier: {balanced_acc_random*100}%")
        print(f"\t{data_split} balanced accuracy random classifier (adjusted): {balanced_acc_adjusted_random*100}%")
    # MCC
    mcc = matthews_corrcoef(targets, preds)
    mcc_random = matthews_corrcoef(targets, random_pred)
    if (verbose):
        print(f"\n{data_split} MCC: {mcc*100}%")
        print(f"\t{data_split} MCC random classifier: {mcc_random*100}%")
    # F1 Score
    if (len(n_unique_classes) == 2):
        f1_score_val = f1_score(targets, preds, average="binary")
        f1_score_val_random = f1_score(targets, random_pred, average="binary")
    else:
        f1_score_val = f1_score(targets, preds, average="micro")
        f1_score_val_random = f1_score(targets, random_pred, average="micro")
    if (verbose):
        print(f"\n{data_split} F1 Score: {f1_score_val*100}%")
        print(f"\t{data_split} F1 Score random classifier: {f1_score_val_random*100}%")
    
    # Performance per class
    if (print_classification_report):
        if (len(n_unique_classes) == 2):
            target_names = ["Not Infection", "Infection"]
        else:
            raise ValueError(f"{len(n_unique_classes)} is not valid.")
        print("\n\n{} classification report: {}".format(data_split, classification_report(targets, preds, target_names=target_names, labels=n_unique_classes) ))
        print("\n\t{} classification report random classifier: {}".format(data_split, classification_report(targets, random_pred, target_names=target_names, labels=n_unique_classes)))

    return mcc*100, f1_score_val*100, balanced_acc*100, balanced_acc_adjusted*100


# Class for early stopping
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0, restore_best_weights=True):
        """
            GENERATED WITH CHAT-GPT

            Parameters:
            -----------
            patience: int
                How many epochs to wait after last improvement.
            min_delta: float
                Minimum change in the monitored metric to qualify as improvement.
            restore_best_weights: bool
                Whether to restore the model weights from the best epoch.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_state_dicts = {}

    def __call__(self, val_loss, models):
        score = -val_loss  # since lower loss is better

        if self.best_score is None:
            self.best_score = score
            for submodel in models:
                self.best_state_dicts[submodel] = models[submodel].state_dict()
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                if self.restore_best_weights:
                    for submodel in models:
                        models[submodel].load_state_dict(self.best_state_dicts[submodel])
        else:
            self.best_score = score
            for submodel in models:
                self.best_state_dicts[submodel] = models[submodel].state_dict()
            self.counter = 0

# Abstract class
class GenericExperiment(ABC):
    def __init__(self, parameters_exp):
        """
            Class for a generic DL experiment (train, validate, test
            a certain model on a given dataset)

            Arguments:
            ----------
            parameters_exp: dict
                Dictionary containing the parameters of the experiment.
        """
        # Defining some attributes of the experiment
        self.exp_id = parameters_exp['exp_id']
        self.results_folder = None

        # Dataset type
        # Main dataset
        if ('Dataset' not in parameters_exp):
            parameters_exp['Dataset'] = {}
            if ('dataset_name' not in parameters_exp['Dataset']):
                parameters_exp['Dataset']['dataset_name'] = 'AIIDKIT'
        # Subdataset
        if ('subdataset' not in parameters_exp['Dataset']):
            parameters_exp['Dataset']['subdataset'] = 'TEAV_Static_Graph_v1'

        # Label type to use
        if ('label_type' not in parameters_exp):
            parameters_exp['Dataset']['label_type'] = "infection_label_binary_bacterial" 

        # HDF5 file name
        if ('hdf5_dataset_filename' not in parameters_exp['Dataset']):
            raise RuntimeError("HDF5 files containing the datasets shoudl be specified")
        else:
            if ('Train' not in parameters_exp['Dataset']['hdf5_dataset_filename']) or ('Test' not in parameters_exp['Dataset']['hdf5_dataset_filename']):
                raise RuntimeError("Train and Test datasets must be specified (Validation is optional).")

        # Save predictions every X epochs (to reduce storage requirements)
        if ("epochs_step_save_preds" not in parameters_exp):
            parameters_exp["epochs_step_save_preds"] = 10
        self.epochs_step_save_preds = parameters_exp["epochs_step_save_preds"]

        # Number of classes
        if ("binary" in parameters_exp['Dataset']['label_type'].lower()):
            self.num_classes = 2
        else:
            raise ValueError(f"Label type {parameters_exp['Dataset']['label_type']} is not supported")

        # Parameter to know if we have to normalize or not the dataset
        if ("normalize_ds" not in parameters_exp['Dataset']):
            parameters_exp['Dataset']['normalize_ds'] = False
        self.are_DS_normalized = False
        
            
                
        # Optuna params
        if ('Optimization' not in parameters_exp):
            parameters_exp['Optimization'] = {}
        if ('Optuna' not in parameters_exp['Optimization']):
            parameters_exp['Optimization']['Optuna'] = {}
        if ('use_optuna' not in parameters_exp['Optimization']['Optuna']):
            parameters_exp['Optimization']['Optuna']['use_optuna'] = False
        if (parameters_exp['Optimization']['Optuna']['use_optuna']):
            # N trials to do
            if ('n_trials' not in parameters_exp['Optimization']['Optuna']):
                parameters_exp['Optimization']['Optuna']['n_trials'] = 30
            # File path to Optuna DB to continue optimization
            if ('optuna_starting_point_fn' not in parameters_exp['Optimization']['Optuna']):
                parameters_exp['Optimization']['Optuna']['optuna_starting_point_fn'] = None

        # Training params
        if ('device' not in parameters_exp):
            parameters_exp['device'] = "cuda:0" if torch.cuda.is_available() else "cpu"
            #parameters_exp['device'] = "cpu"
        self.device = torch.device(parameters_exp['device'])
        #self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # self.device = torch.device("cpu")
        
        if ('TrainingParams' not in parameters_exp):
            parameters_exp['TrainingParams'] = {}
        if ('lr' not in parameters_exp['TrainingParams']):
            parameters_exp['TrainingParams']['lr'] = 1e-2
        if ('nb_repetitions' not in parameters_exp['TrainingParams']):
            parameters_exp['TrainingParams']['nb_repetitions'] = 1
        if (parameters_exp['Optimization']['Optuna']['use_optuna'] and parameters_exp['TrainingParams']['nb_repetitions'] > 1):
            raise RecursionError("When doing Optuna hyperparameter search, you cannot select more than one repetition of a single training and evaluation")
        if ('weight_decay' not in parameters_exp['TrainingParams']):
            parameters_exp['TrainingParams']['weight_decay'] = 1e-5
        if ('batch_size_train' not in parameters_exp['TrainingParams']):
            parameters_exp['TrainingParams']['batch_size_train'] = 4
        if ('batch_size_val' not in parameters_exp['TrainingParams']):
            parameters_exp['TrainingParams']['batch_size_val'] = 4
        if ('batch_size_test' not in parameters_exp['TrainingParams']):
            parameters_exp['TrainingParams']['batch_size_test'] = 4
        if ('n_epochs' not in parameters_exp['TrainingParams']):
            parameters_exp['TrainingParams']['n_epochs'] = 10
        if (not parameters_exp['Optimization']['Optuna']['use_optuna']):
            self.lr = parameters_exp['TrainingParams']['lr']

        # Results file
        self.repetitions_results_fn = None

        # Optimizer
        if ('optimizer' not in parameters_exp['Optimization']):
            parameters_exp['Optimization']['optimizer'] = "Adam"
            #parameters_exp['Optimization']['optimizer'] = "AdamW"
        
        # Main loss function
        self.use_evidential_learning = False
        if ('loss_function' not in parameters_exp['Optimization']):
            parameters_exp['Optimization']['loss_function'] = "CE"
        if (parameters_exp['Optimization']['loss_function'].lower() == "ce"):
            if (self.num_classes > 2):
                self.criterion = torch.nn.CrossEntropyLoss()
            else:
                self.criterion = torch.nn.BCEWithLogitsLoss()
        elif (parameters_exp['Optimization']['loss_function'].lower() == "evidentiallearningloss"):
            self.use_evidential_learning = True
            if ('EvidentialLoss' not in parameters_exp['Optimization']):
                parameters_exp['Optimization']['EvidentialLoss'] = {}
            if ('lambda_evidential_sched' not in parameters_exp['Optimization']['EvidentialLoss']):
                parameters_exp['Optimization']['EvidentialLoss']['lambda_evidential_sched'] = False
            if ('lambda_evidential' not in parameters_exp['Optimization']['EvidentialLoss']):
                if (parameters_exp['Optimization']['EvidentialLoss']['lambda_evidential_sched']):
                    parameters_exp['Optimization']['EvidentialLoss']['lambda_evidential'] = 0
                else:
                    parameters_exp['Optimization']['EvidentialLoss']['lambda_evidential'] = 1e-3

            if(parameters_exp['Optimization']['EvidentialLoss']['lambda_evidential_sched']) and (parameters_exp['Optimization']['EvidentialLoss']['lambda_evidential'] is not None):
                if (parameters_exp['Optimization']['EvidentialLoss']['lambda_evidential'] > 0):
                    print(f"\nWARNING: Trying to used both scheduling and fixed value for lambda_evidential. In this case, scheduling is used by default.")
                else:
                    parameters_exp['Optimization']['EvidentialLoss']['lambda_evidential'] = 0
            self.criterion = EvidentialClassification(lamb=parameters_exp['Optimization']['EvidentialLoss']['lambda_evidential'])
        else:
            raise ValueError(f"\nLoss function {parameters_exp['Optimization']['loss_function']} is not valid\n")
        
        # Model type to use (Transformer, GNN, etc.)
        if ('Model' not in parameters_exp):
            parameters_exp['Model'] = {}
        if ('model_type' not in parameters_exp['Model']):
            parameters_exp['Model']['model_type'] = 'GNN'
        self.model_type = parameters_exp['Model']['model_type']

        # Precise model to use
        if ('model_to_use' not in parameters_exp['Model']):
            if (parameters_exp['Model']['model_type'].lower() == 'GNN'):
                #parameters_exp['Model']['model_to_use'] = 'SimpleHeteroGNN'
                parameters_exp['Model']['model_to_use'] = 'HeteroGraphSage'
            else:
                raise ValueError("Model type {} is not valid".format(parameters_exp['Model']['model_type']))

        # Parameters of the model
        if (parameters_exp['Model']['model_to_use'].lower() == 'simpleheterognn'):
            if ('hidden_channels' not in parameters_exp['Model']):
                parameters_exp['Model']['hidden_channels'] = 16
            if ('graph_pool_strategy' not in parameters_exp['Model']):
                parameters_exp['Model']['graph_pool_strategy'] = "mean"
            if ('graph_pool_fusion' not in parameters_exp['Model']):
                parameters_exp['Model']['graph_pool_fusion'] = "concatenation"
        elif (parameters_exp['Model']['model_to_use'].lower() == 'heterographsage'):
            if ('hidden_channels' not in parameters_exp['Model']):
                parameters_exp['Model']['hidden_channels'] = 16
            if ('dropout' not in parameters_exp['Model']):
                parameters_exp['Model']['dropout'] = 0.1
            if ('num_layers' not in parameters_exp['Model']):
                parameters_exp['Model']['num_layers'] = 2
            if ('hidden_channels' not in parameters_exp['Model']):
                parameters_exp['Model']['hidden_channels'] = 16
            if ('graph_pool_strategy' not in parameters_exp['Model']):
                parameters_exp['Model']['graph_pool_strategy'] = "mean"
            if ('graph_pool_fusion' not in parameters_exp['Model']):
                parameters_exp['Model']['graph_pool_fusion'] = "concatenation"
        # Output channels (number of classes for prediction)
        if (self.num_classes == 2):
            if (self.use_evidential_learning):
                parameters_exp['Model']['out_channels'] = 2
            else:
                parameters_exp['Model']['out_channels'] = 1
        else:
            parameters_exp['Model']['out_channels'] = self.num_classes

        # Holdout train ID (in case we use the method several times)
        self.holdout_train_id = -1

        # Early stopping
        if ('EarlyStopping' not in parameters_exp['Optimization']):
            parameters_exp['Optimization']['EarlyStopping'] = {}
        if ('use_early_stopping' not in parameters_exp['Optimization']['EarlyStopping']):
            parameters_exp['Optimization']['EarlyStopping']['use_early_stopping'] = False
        if (parameters_exp['Optimization']['EarlyStopping']['use_early_stopping']):
            if ('patience' not in parameters_exp['Optimization']['EarlyStopping']):
                parameters_exp['Optimization']['EarlyStopping']['patience'] = 5
            if ('min_delta' not in parameters_exp['Optimization']['EarlyStopping']):
                parameters_exp['Optimization']['EarlyStopping']['min_delta'] = 1e-4

        # Parameters of the exp
        self.parameters_exp = parameters_exp

    @abstractmethod
    def createTorchDatasets(self, verbose=True, prefix_paths=""):
        """
            Create the torch datasets associated needed to train and evaluate the model
        """
        pass

    @abstractmethod
    def addClassWeightsLoss(self):
        """
            Compute class weights and redefines the loss function using
            these class weights to handle imbalanced classes.
        """
        pass

    @abstractmethod
    def computeTrainDSStatistics(self):
        """
            Computes the train DS statistics used to normalize the datasets
        """
        # TODO
        raise NotImplementedError()

    @abstractmethod
    def normalizeDataset(self):
        """
            Computes the statistics necessary to normalize the dataset.
        """
        # TODO
        raise NotImplementedError()

    @abstractmethod
    def dataloadersCreation(self):
        """
            Create the train and test dataloader necessary to train and test a
            deep learning model
        """
        # TODO
        raise NotImplementedError()

    @abstractmethod
    def modelCreation(self):
        """
            Creates a model to be trained on the selected time-frequency
            representation
        """
        raise NotImplementedError()
    
    def scheduling_lambda_evidential(self, current_epoch):
        """
            Annealed scheduling of the KL lambda parameter (term of the uncertainty
            regularizer) for evidential classification, controlling the trade-off
            between uncertainty and classificaiton perfromance.
        """
        # Update lambda evidential
        self.parameters_exp['Optimization']['EvidentialLoss']['lambda_evidential'] = min(1, current_epoch/self.parameters_exp['TrainingParams']['n_epochs'])

        # Re-create the loss
        if (self.use_evidential_learning):
            self.criterion = EvidentialClassification(lamb=self.parameters_exp['Optimization']['EvidentialLoss']['lambda_evidential'], class_weights=self.class_weights_to_use)

    
    def createOptimizer(self):
        # Parameters to optimize
        model_params = self.model.parameters()

        # Creating the optimizer
        if (self.parameters_exp['Optimization']['optimizer'].lower() == 'adam'):
            self.optimizer = torch.optim.Adam(
                                                model_params,
                                                lr=self.parameters_exp['TrainingParams']['lr'],
                                                weight_decay=self.parameters_exp['TrainingParams']['weight_decay']
                                            )
        elif (self.parameters_exp['Optimization']['optimizer'].lower() == 'adamw'):
            self.optimizer = torch.optim.AdamW(
                                                model_params,
                                                lr=self.parameters_exp['TrainingParams']['lr'],
                                                betas=(0.9, 0.999),
                                                weight_decay=self.parameters_exp['TrainingParams']['weight_decay']
                                              )
        else:
            raise ValueError(f"\nOptimizer {self.parameters_exp['Optimization']['optimizer']} is not valid")
        
        # Learning rate scheduler
        # Noam already has a learning rate scheduler
        if (self.parameters_exp['Optimization']['optimizer'].lower() in ['adam', 'adamw']):
            self.sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
                                                                        self.optimizer,
                                                                        mode='min',\
                                                                        factor=0.1,
                                                                        patience=5,\
                                                                        threshold=1e-4,
                                                                        threshold_mode='rel',\
                                                                        cooldown=0,\
                                                                        min_lr=0,\
                                                                        eps=1e-08
                                                                        #eps=1e-04
                                                                    )
            # self.sched = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[1, 10], gamma=0.1, verbose=True)
            # self.sched = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.1, last_epoch=- 1, verbose=True)
        
    def computeForwardPass(self, batch, epoch_nb, batch_ID=None):
        # Loss
        loss = None
        preds = None

        return loss, preds
    
    def updateModel(self, batch, epoch, batch_ID=None):
        """
            Updates the weights of a model
        """
        # Zero the parameters gradients
        self.optimizer.zero_grad()

        # Forward pass
        loss, preds = self.computeForwardPass(batch, epoch, batch_ID=batch_ID)
        
        # Backward pass for the gradient computation
        if (type(loss) == dict):
            loss["total_loss"].backward()
        else:
            loss.backward()

        # Applying gradient clipping to solve some
        # gradient problems
        # nn.utils.clip_grad_value_(self.model.parameters(), clip_value=0.1)

        # Updating the weights
        self.optimizer.step()

        return loss, preds

    def initSingleTrain(self, create_new_model=True):
        """
            Initialize the parameters for a single train
        """
        # Normalize the datasets
        if (not self.are_DS_normalized):
            self.normalizeDataset()
            self.are_DS_normalized = True

        # Creating the dataloaders
        self.dataloadersCreation()

        # Add class weights
        self.addClassWeightsLoss(multiclass_strategy='balanced')
        #self.addClassWeightsLoss(multiclass_strategy='inverse')
        #self.addClassWeightsLoss(multiclass_strategy='sqrt_inv')

        # Creating the model
        if (create_new_model):
            self.modelCreation()

        # Creating the optimizer
        self.createOptimizer()

        # Early stopping if asked
        if (self.parameters_exp['Optimization']['EarlyStopping']['use_early_stopping']):
            if (type(self.val_ds) == list) and (len(self.val_ds) == 0):
                RuntimeError("When using early stopping, a validation dataset is needed!")
            else:
                self.use_early_stopping = EarlyStopping(
                                                        patience=self.parameters_exp['Optimization']['EarlyStopping']['patience'],
                                                        min_delta=self.parameters_exp['Optimization']['EarlyStopping']['min_delta'],
                                                        restore_best_weights=True
                                                        #restore_best_weights=False
                                                    )

    @abstractmethod
    def singleTrain(self, rep_ID, create_new_model=True):
        """
            Trains a model one time during self.parameters_exp['TrainingParams']['n_epochs'] epochs
        """
        raise NotImplementedError()

    def evalCurrentModel(self, dataloader, epoch):
        # Evaluation
        self.model.eval()
        tmp_losses = []
        tmp_preds = []
        with torch.no_grad():
            batch_ID = 0
            for batch in tqdm(dataloader):
                loss, preds = self.computeForwardPass(batch, epoch, batch_ID=batch_ID)
                tmp_losses.append(loss.detach().data.cpu().numpy())
                tmp_preds.append(preds)
                batch_ID += 1

        return tmp_losses, tmp_preds

    def holdoutTrain(self, save_results=True):
        """
            Does a holdout training repeated self.parameters_exp['TrainingParams']['nb_repetitions'] times
        """
        # Holdout ID
        self.holdout_train_id += 1 # The initial value is initialize in the constructor, and is -1

        # Creating HDF5 file for the repetitions over time
        if (self.repetitions_results_fn is None):
            self.repetitions_results_fn = self.results_folder + '/metrics/final_results_all_repetitions_0.hdf5'
            with h5py.File(self.repetitions_results_fn, 'w') as h5file:
                for rep_id in range(self.parameters_exp['TrainingParams']['nb_repetitions']):
                    h5file.create_group(f"Rep-{rep_id}")
        
        # Iterating over the repetitions
        for nb_repetition in range(self.parameters_exp['TrainingParams']['nb_repetitions']):
            print("\n\n\n\n======================================================================")
            print("=======> Repetitions {} <=======".format(nb_repetition))
            print("======================================================================")
            # Doing single train
            self.repetition_id = nb_repetition
            self.singleTrain(nb_repetition, create_new_model=True)

            # Saving the final model and the results
            # Model
            if (self.model is not None):
                # Model file to save
                model_file = self.results_folder + '/model/final_model-{}_rep-{}_holdout-{}'.format(self.exp_id, self.repetition_id, self.holdout_train_id)
                model_file = model_file + ".pth"
                # Save
                torch.save({
                                'model_state_dict': self.model.state_dict(),
                                'model': self.model
                            }, model_file)
                # # IMPORTANT: SAVING JIT MODELS FOR STM GNN MODELS DOES NOT WORK AS STM GNN INFER THE SHAPE OF AN INPUT BASED ON WHAT WE PASS TO THE MODEL AND JIT DOES NOT SUPPORT THAT
                # # Model exported with torch.jit
                # model_jit = torch.jit.script(self.model) # Export to TorchScript
                # model_jit.save(self.results_folder + '/model/final-JIT_model-{}_rep-{}.pth'.format(self.exp_id, self.repetition_id)) # Save

    def optuna_objective(self, trial):
        """
            Objective to use with Optuna for hyper-parameter optimization
        """
        #======================================================================#
        #=======================Hyper-parameters to tune=======================#
        #======================================================================#
        # Define global hyper-parameters to tune
        self.parameters_exp['TrainingParams']['lr'] = trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True)
        #self.parameters_exp['TrainingParams']['weight_decay'] = trial.suggest_float("weight_decay", 1e-5, 1e-1, log=True)
        #self.parameters_exp['Optimization']['optimizer'] = trial.suggest_categorical("optimizer", ["Adam", "AdamW"])
        if (self.use_evidential_learning):
            self.parameters_exp['Optimization']['EvidentialLoss']['lambda_evidential'] = trial.suggest_float("lambda_evidential", 1e-4, 1e-1, log=True)

        # Define per-model hyper-parameters to tune
        if (not self.use_evidential_learning):
            self.parameters_exp['Model']['hidden_channels'] = trial.suggest_categorical("hidden_channels", [8, 16, 32, 64])
            #self.parameters_exp['Model']['graph_pool_strategy'] = trial.suggest_categorical("graph_pool_strategy", ["mean", "topk"])
            self.parameters_exp['Model']['graph_pool_fusion'] = trial.suggest_categorical("graph_pool_fusion", ["stack", "concatenation"])
            if (self.parameters_exp['Model']['model_to_use'].lower() == 'simpleheterognn'):
                pass # All the parameters where already defined
            elif (self.parameters_exp['Model']['model_to_use'].lower() == 'heterographsage'):
                self.parameters_exp['Model']['dropout'] = trial.suggest_float("dropout", 0.0, 0.5, log=False)
                self.parameters_exp['Model']["num_layers"] = trial.suggest_categorical("num_layers", [1, 2, 4])
            elif (self.parameters_exp['Model']['model_to_use'].lower() == "heterogat"):
                self.parameters_exp['Model']['dropout'] = trial.suggest_float("dropout", 0.0, 0.5, log=False)
                self.parameters_exp['Model']["num_layers"] = trial.suggest_categorical("num_layers", [1, 2, 4])
                self.parameters_exp['Model']["heads"] = trial.suggest_categorical("heads", [2, 4, 8])
            else:
                raise ValueError(f"\nModel to use {self.parameters_exp['Model']['model_to_use'].lower()} is not valid for Optuna hyper-parameter tuning\n.")
            
        #======================================================================#
        #==================Creating HDF5 for temporary results==================#
        #======================================================================#
        # Creating HDF5 file to temporarily store the results
        self.repetitions_results_fn = self.results_folder + f'/metrics/tmp_optuna_res_file.hdf5'
        with h5py.File(self.repetitions_results_fn, 'w') as h5file:
            for rep_id in range(self.parameters_exp['TrainingParams']['nb_repetitions']):
                # IMPORTANT: For Optuna optimization, we do not use several datasets as it
                # IMPORTANT: to expensive to do it.
                h5file.create_group(f"Rep-{rep_id}")

        #======================================================================#
        #==========================Getting the metric==========================#
        #======================================================================#
        # Verify that validation dataset is available (necessary for hyper-parameter tuning)
        if (type(self.val_ds) == list) and (len(self.val_ds) == 0):
            raise RuntimeError("\nHyper-parameter optimization with Optuna cannot be done without validation set\n")
        try:
            # Training the model
            for nb_repetition in range(self.parameters_exp['TrainingParams']['nb_repetitions']):
                self.singleTrain(nb_repetition, create_new_model=True)

            # Load results HDF5 file
            results_h5_file = h5py.File(self.repetitions_results_fn, 'r')

            # Getting the validation metric
            n_repetitions = len(results_h5_file)
            epochs_list = sorted([int(epoch_str.split('-')[-1]) for epoch_str in list(results_h5_file[f"Rep-0"]["Preds"]["Train"].keys())])
            epochs_to_use = max(epochs_list)
            metrics = {
                            "MCC": [None for _ in range(n_repetitions)],
                            "F1Score": [None for _ in range(n_repetitions)],
                            "BalancedAccuracy": [None for _ in range(n_repetitions)],
                            "BalancedAccuracyAdjusted": [None for _ in range(n_repetitions)],
                            "AUC": [None for _ in range(n_repetitions)]
                        }
            for rep_id in range(n_repetitions):
                # Filling results
                data_split = "Val"
                if (len(results_h5_file[f"Rep-{rep_id}"]["Preds"][data_split].keys()) > 0):
                    # Getting targets and predictions
                    targets = results_h5_file[f"Rep-{rep_id}"]["Preds"][data_split][f"Epoch-{epochs_to_use}"]['targets']
                    if (self.num_classes > 2):
                        raise NotImplementedError("Not implemented yet, see code epidemio_informed_gnns.")
                    else:
                        predictions = results_h5_file[f"Rep-{rep_id}"]["Preds"][data_split][f"Epoch-{epochs_to_use}"]['predictions']
                    predictions_probs = results_h5_file[f"Rep-{rep_id}"]["Preds"][data_split][f"Epoch-{epochs_to_use}"]['predictions_probs']

                    # Getting the metrics for the epoch
                    # MCC, F1-Score and Balanced Accuracy
                    mcc, f1_score_val, balanced_acc, balanced_acc_adjusted = get_classification_metrics(
                                                                                                            targets,
                                                                                                            predictions,
                                                                                                            data_split,
                                                                                                            [i for i in range(self.num_classes)],
                                                                                                            verbose=False,
                                                                                                            print_classification_report=False
                                                                                                        )
                    # AUC
                    if (self.num_classes == 2):
                        if (self.use_evidential_learning):
                            new_targets = np.zeros((targets.shape[0], 2), dtype=int)
                            new_targets[targets[:] == 0., 0] = 1
                            new_targets[targets[:] == 1., 1] = 1
                            auc = roc_auc_score(new_targets, predictions_probs, average="macro")
                        else:
                            auc = roc_auc_score(targets, predictions_probs, average="macro")
                    else:
                        # According to Scikit-learn roc_auc_score with multi_class='ovo' and average="macro" is insensitive to class imbalance 
                        auc = roc_auc_score(targets, predictions_probs, multi_class='ovo', average="macro")

                    # Add to metrics list
                    metrics["MCC"][rep_id] = mcc
                    metrics["F1Score"][rep_id] = f1_score_val
                    metrics["BalancedAccuracy"][rep_id] = balanced_acc
                    metrics["BalancedAccuracyAdjusted"][rep_id] = balanced_acc_adjusted
                    metrics["AUC"][rep_id] = auc

            # Erase the results HDF5 file as it is not necessary for hyper-parameter tuning
            os.remove(self.repetitions_results_fn)

            # Selecting metric to use
            if (self.metric_use_optuna.lower() == "mcc"):
                val_metric = np.mean(metrics["MCC"])
            elif (self.metric_use_optuna.lower() == "f1score"):
                val_metric = np.mean(metrics["F1Score"])
            elif (self.metric_use_optuna.lower() == "balancedaccuracy"):
                val_metric = np.mean(metrics["BalancedAccuracy"])
            elif (self.metric_use_optuna.lower() == "BalancedAccuracyAdjusted"):
                val_metric = np.mean(metrics["BalancedAccuracyAdjusted"])
            elif (self.metric_use_optuna.lower() == "auc"):
                val_metric = np.mean(metrics["AUC"])
            
            return val_metric
        except Exception as e:
            print(f"Trial {trial.number} failed with error: {e}. Returning 0.0 as metric.")

            #raise e

            return 0.0  # <- only works for metrics to maximize and > 0.0!
        
    def optuna_tunning(self):
        """
            Hyper-parameter optimization using Optuna
        """
        print(f"\n\n\n\n======================================================================")
        print(f"======================================================================")
        print(f"======================================================================")
        print(f"============DOING HYPER-PARAMETER OPTIMIZATION WITH OPTUNA============")
        print(f"======================================================================")
        print(f"======================================================================")
        print(f"======================================================================\n\n\n\n")
        # As we are going to do several sub-experiments, we are going to have
        # one ID per sub-experiment
        self.repetition_id = 0

        # Metric to use for hyper-parameter optimization
        #self.metric_use_optuna = "MCC"
        #self.metric_use_optuna = "F1Score"
        self.metric_use_optuna = "BalancedAccuracy"
        #self.metric_use_optuna = "AUC"

        # Creating the study
        if (self.parameters_exp['Optimization']['Optuna']['optuna_starting_point_fn'] is None): # Create study from scratch
            print("\n\n=========> CREATING OPTUNA STUDY FROM SCRATCH <=========\n")
            db_dir = Path(self.results_folder) / 'metrics'
            db_dir.mkdir(parents=True, exist_ok=True)
            storage = f"sqlite:///{db_dir / 'optuna_results.db'}"
            self.study = optuna.create_study(
                                            direction="maximize",
                                            study_name=f'OptunaHyperParamOptim_{self.exp_id}',
                                            sampler=optuna.samplers.TPESampler(seed=42), # Fix seed for reproducibility
                                            storage=storage
                                        )
        else: # Continue study
            print(f"\n\n=========> LOADING OPTUNA STUDY TO CONTINUE IT ({self.parameters_exp['Optimization']['Optuna']['optuna_starting_point_fn']}) <=========\n")
            # Getting the path of the study file
            optuna_study_path = Path(self.parameters_exp['Optimization']['Optuna']['optuna_starting_point_fn'])
            # Getting the name of the study
            tmp_params_file = '/'.join(self.parameters_exp['Optimization']['Optuna']['optuna_starting_point_fn'].split('/')[:-2]) + '/params_exp/params_beginning_0.yaml'
            with open(tmp_params_file, 'r') as file:
                tmp_params_exp = yaml.safe_load(file)
                tmp_exp_id = tmp_params_exp['exp_id']
            study_name = f'OptunaHyperParamOptim_{tmp_exp_id}'
            self.study = optuna.load_study(
                                            study_name=study_name,
                                            sampler=optuna.samplers.TPESampler(seed=42), # Fix seed for reproducibility
                                            storage=f"sqlite:///{optuna_study_path}"
                                        )
        self.normalized_ds = False
        self.study.optimize(self.optuna_objective, n_trials=self.parameters_exp['Optimization']['Optuna']['n_trials'])
        print("Best Trial:")
        trial = self.study.best_trial
        print(f"{self.metric_use_optuna.upper()}: {trial.value:.4f}")
        print("\tParams:")
        for key, value in trial.params.items():
            print(f"\t\t{key}: {value}")


    def setResultsFolder(self, results_folder):
        """
            Set the folder where the results are going to be stored
        """
        self.results_folder = results_folder
