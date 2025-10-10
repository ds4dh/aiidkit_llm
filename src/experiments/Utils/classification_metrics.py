"""
    Plot the metrics and results of an experiment
"""
import os
import h5py
import argparse
import pickle
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import matthews_corrcoef,\
                            f1_score,\
                            balanced_accuracy_score,\
                            classification_report,\
                            roc_auc_score,\
                            average_precision_score
import torch

def plot_metric_epochs(metric_dict, metric_name="Loss"):
    """
        Plots a metric of the results of a model over the epochs

        Parameters:
        -----------
        metric_dict: dict
            Dictionary where the keys are the data splits and the values are
            the lists with the values of the different losses over the epochs
            and repetitions for that data split.
            The structure is the following:
            data_split -> epochs -> repetitions
    """
    # Creating the figure
    fig = plt.figure()
    # Iterating over the different data splits
    for data_split in metric_dict:
        # Getting the number of epochs
        if (type(metric_dict[data_split]) == dict):
            epochs = sorted(list(metric_dict[data_split].keys()))
        else:
            epochs = list(range(len(metric_dict[data_split])))

        # Getting the mean and std value per epoch
        mean_per_epoch = []
        std_per_epoch = []
        for values_per_rep in metric_dict[data_split]:
            if (type(metric_dict[data_split]) == dict):
                if (metric_dict[data_split][values_per_rep][0] is not None):
                    mean_per_epoch.append(np.mean(metric_dict[data_split][values_per_rep]))
                    std_per_epoch.append(np.std(metric_dict[data_split][values_per_rep])) 
            else:
                mean_per_epoch.append(np.mean(values_per_rep))
                std_per_epoch.append(np.std(values_per_rep))
        if (len(mean_per_epoch) > 0):
            plt.errorbar(epochs, mean_per_epoch, std_per_epoch, label=f"{data_split} {metric_name}")
    plt.xlabel("Epochs")
    plt.ylabel(metric_name)
    plt.title(metric_name)
    plt.legend()
    plt.show()


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


def compute_ece(probs, labels, n_bins=10, evidential=False):
    """
        IMPORTANT: Generated using ChatGPT.
        Computes the expected calibratione error (ECE).

        Parameters:
        -----------
        probs: np.array or torch.tensor
            [N, num_classes] softmax probabilities
        probs: np.array or torch.tensor
            [N] true labels
        evidential: bool
            Boolean indicating if evidential learning was used
            (in that case binary classification gives an 
            output vector with two elements, one per class)
    """
    # Getting the unique labels
    unique_labels = np.unique(labels)

    # Get the "confindence" of the model on the predictions (i.e. max prob per sample, allowing to choose the predicted class)
    if (len(unique_labels) == 2) and (not evidential): # Binary classification
        decision_treshold = 0.5
        confidences = (probs >= decision_treshold)*probs + (probs < decision_treshold)*(1-probs)
        predictions = (probs >= decision_treshold).astype(int)
    else:
        confidences = np.max(probs, axis=1)
        predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels)

    # Computing the ECE
    ece = 0.0
    bin_boundaries = np.linspace(0.0, 1.0, n_bins+1)
    for i in range(n_bins):
        # Select samples in bin
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i+1])
        prop_in_bin = np.mean(in_bin)
        
        if prop_in_bin > 0:
            acc_in_bin = np.mean(accuracies[in_bin])
            avg_conf_in_bin = np.mean(confidences[in_bin])
            ece += prop_in_bin * abs(acc_in_bin - avg_conf_in_bin)
    
    return ece

#======================================================================#
#=================================MAIN=================================#
#======================================================================#
def main():
    #======================================================================#
    #============================Argument Parser============================#
    #======================================================================#
    # Construct the argument parser
    ap = argparse.ArgumentParser()
    # Add the arguments to the parser
    ap.add_argument('--results_folder', required=True, help="Path to the folder containing the results of the experiment", type=str)
    ap.add_argument('--print_classification_report', help="Use it if want to print the classification report per epochs", action='store_true')
    ap.add_argument('-v', '--verbose', help="Use it to get details about the performances per epoch", action='store_true')
    args = vars(ap.parse_args())

    # Getting the value of the arguments
    results_folder = args['results_folder']
    print_classification_report = args['print_classification_report']
    verbose = args['verbose']
    
    #======================================================================#
    #===============================Load data===============================#
    #======================================================================#
    # Open the results file
    results_file = results_folder + "/metrics/final_results_all_repetitions_0.hdf5"
    results_h5_file = h5py.File(results_file, 'r')

    # Number of unique classes (necessary for performances of random classifier)
    if ('murcia' in results_folder.lower()):
        N_UNIQUE_CLASSES = [tmp_i for tmp_i in range(6)]
    else:
        N_UNIQUE_CLASSES = [tmp_i for tmp_i in range(4)]

    #======================================================================#
    #===============================Plot loss===============================#
    #======================================================================#
    # Getting the metrics dicts for the different losses
    if ('Rep-0' in results_h5_file):
        base_name_main_group = 'Rep-'
        multiple_datasets = False
    else:
        base_name_main_group = 'Rep-0_Dataset-'
        multiple_datasets = True
    loss_names = list(results_h5_file[base_name_main_group+"0"]['Loss']['Train'].keys())
    n_repetitions = len(results_h5_file)

    # Creating the loss dict
    losses_dict = {}
    for loss_name in loss_names:
        losses_dict[loss_name] = {}
        for data_split in ["Train", "Val", "Test"]:
            if (data_split in results_h5_file[base_name_main_group+"0"]['Loss']):
                if (len(results_h5_file[base_name_main_group+"0"]['Loss'][data_split].keys()) > 0):
                    n_epochs = len(results_h5_file[base_name_main_group+"0"]['Loss'][data_split][loss_name])
                    losses_dict[loss_name][data_split] = [[None for _ in range(n_repetitions)] for _ in range(n_epochs)]
    # Filling the loss dict
    for loss_name in loss_names:
        for rep_id in range(n_repetitions):
            for data_split in list(results_h5_file[base_name_main_group+str(rep_id)]['Loss'].keys()):
                if (len(results_h5_file[base_name_main_group+"0"]['Loss'][data_split].keys()) > 0):
                    n_epochs = len(results_h5_file[base_name_main_group+str(rep_id)]['Loss'][data_split][loss_name])
                    for epoch in range(n_epochs):
                        losses_dict[loss_name][data_split][epoch][rep_id] = results_h5_file[base_name_main_group+str(rep_id)]['Loss'][data_split][loss_name][epoch]

    # Plotting the different losses
    for loss_type in loss_names:
        plot_metric_epochs(metric_dict=losses_dict[loss_type], metric_name=loss_type)

    #======================================================================#
    #========================Get performance metrics========================#
    #======================================================================#
    # Getting the predictions for each epoch (concatenate days)
    n_repetitions = len(results_h5_file)
    epochs_list = sorted([int(epoch_str.split('-')[-1]) for epoch_str in list(results_h5_file[base_name_main_group+"0"]["Preds"]["Train"].keys())])
    metrics_per_data_split = {
                                "MCC": {"Train": {epoch: [None for _ in range(n_repetitions)] for epoch in epochs_list}, "Val": {epoch: [None for _ in range(n_repetitions)] for epoch in epochs_list}, "Test": {epoch: [None for _ in range(n_repetitions)] for epoch in epochs_list}},
                                "F1Score": {"Train": {epoch: [None for _ in range(n_repetitions)] for epoch in epochs_list}, "Val": {epoch: [None for _ in range(n_repetitions)] for epoch in epochs_list}, "Test": {epoch: [None for _ in range(n_repetitions)] for epoch in epochs_list}},
                                "BalancedAccuracy": {"Train": {epoch: [None for _ in range(n_repetitions)] for epoch in epochs_list}, "Val": {epoch: [None for _ in range(n_repetitions)] for epoch in epochs_list}, "Test": {epoch: [None for _ in range(n_repetitions)] for epoch in epochs_list}},
                                "AUC": {"Train": {epoch: [None for _ in range(n_repetitions)] for epoch in epochs_list}, "Val": {epoch: [None for _ in range(n_repetitions)] for epoch in epochs_list}, "Test": {epoch: [None for _ in range(n_repetitions)] for epoch in epochs_list}},
                                "PerClassAUC": {"Train": {epoch: [None for _ in range(n_repetitions)] for epoch in epochs_list}, "Val": {epoch: [None for _ in range(n_repetitions)] for epoch in epochs_list}, "Test": {epoch: [None for _ in range(n_repetitions)] for epoch in epochs_list}},
                            }
    if (multiple_datasets):
        base_group_name = "Rep"
    targets_all_days_last_epoch = {rep_id : {data_split: None for data_split in list(results_h5_file[base_name_main_group+str(rep_id)]["Preds"].keys())} for rep_id in range(n_repetitions)}
    preds_all_days_last_epoch = {rep_id : {data_split: None for data_split in list(results_h5_file[base_name_main_group+str(rep_id)]["Preds"].keys())} for rep_id in range(n_repetitions)}
    preds_probs_all_days_last_epoch = {rep_id : {data_split: None for data_split in list(results_h5_file[base_name_main_group+str(rep_id)]["Preds"].keys())} for rep_id in range(n_repetitions)}
    targets_inf_risk_all_days_last_epoch = None
    preds_inf_risk_all_days_last_epoch = None
    preds_probs_inf_risk_all_days_last_epoch = None
    if ("true_inf_risk_class" in results_h5_file[base_name_main_group+"0"]["Preds"]["Train"][f"Epoch-{epoch}"][f"Day-0"]):
        targets_inf_risk_all_days_last_epoch = {rep_id : {data_split: None for data_split in list(results_h5_file[base_name_main_group+str(rep_id)]["Preds"].keys())} for rep_id in range(n_repetitions)}
        preds_inf_risk_all_days_last_epoch = {rep_id : {data_split: None for data_split in list(results_h5_file[base_name_main_group+str(rep_id)]["Preds"].keys())} for rep_id in range(n_repetitions)}
        preds_probs_inf_risk_all_days_last_epoch = {rep_id : {data_split: None for data_split in list(results_h5_file[base_name_main_group+str(rep_id)]["Preds"].keys())} for rep_id in range(n_repetitions)}
        metrics_per_data_split["MCCInfectionRisk"] = {"Train": {epoch: [None for _ in range(n_repetitions)] for epoch in epochs_list}, "Val": {epoch: [None for _ in range(n_repetitions)] for epoch in epochs_list}, "Test": {epoch: [None for _ in range(n_repetitions)] for epoch in epochs_list}}
        metrics_per_data_split["F1ScoreInfectionRisk"] = {"Train": {epoch: [None for _ in range(n_repetitions)] for epoch in epochs_list}, "Val": {epoch: [None for _ in range(n_repetitions)] for epoch in epochs_list}, "Test": {epoch: [None for _ in range(n_repetitions)] for epoch in epochs_list}}
        metrics_per_data_split["BalancedAccuracyInfectionRisk"] = {"Train": {epoch: [None for _ in range(n_repetitions)] for epoch in epochs_list}, "Val": {epoch: [None for _ in range(n_repetitions)] for epoch in epochs_list}, "Test": {epoch: [None for _ in range(n_repetitions)] for epoch in epochs_list}}
        metrics_per_data_split["AUCInfectionRisk"] = {"Train": {epoch: [None for _ in range(n_repetitions)] for epoch in epochs_list}, "Val": {epoch: [None for _ in range(n_repetitions)] for epoch in epochs_list}, "Test": {epoch: [None for _ in range(n_repetitions)] for epoch in epochs_list}}
        metrics_per_data_split["PerClassAUCInfectionRisk"] = {"Train": {epoch: [None for _ in range(n_repetitions)] for epoch in epochs_list}, "Val": {epoch: [None for _ in range(n_repetitions)] for epoch in epochs_list}, "Test": {epoch: [None for _ in range(n_repetitions)] for epoch in epochs_list}}
    for rep_id in range(n_repetitions):
        for data_split in list(results_h5_file[base_name_main_group+str(rep_id)]["Preds"].keys()):
            if (len(results_h5_file[base_name_main_group+str(rep_id)]["Preds"][data_split].keys()) > 0):
                for epoch in epochs_list:
                    targets_all_days = []
                    preds_all_days = []
                    preds_all_days_probs = []
                    targets_inf_risk_all_days = []
                    preds_inf_risk_all_days = []
                    preds_inf_risk_all_days_probs = []
                    n_days = len(results_h5_file[base_name_main_group+str(rep_id)]["Preds"][data_split][f"Epoch-{epoch}"].keys())
                    for day in range(n_days):
                        # Getting targets and predictions
                        targets_all_days.append(results_h5_file[base_name_main_group+str(rep_id)]["Preds"][data_split][f"Epoch-{epoch}"][f"Day-{day}"]['true_seir_states'])
                        preds_all_days.append(np.argmax(results_h5_file[base_name_main_group+str(rep_id)]["Preds"][data_split][f"Epoch-{epoch}"][f"Day-{day}"]['pred_seir_states_probs'], axis=1))
                        preds_all_days_probs.append(results_h5_file[base_name_main_group+str(rep_id)]["Preds"][data_split][f"Epoch-{epoch}"][f"Day-{day}"]['pred_seir_states_probs'])
                        if (targets_inf_risk_all_days_last_epoch is not None):
                            targets_inf_risk_all_days.append(results_h5_file[base_name_main_group+str(rep_id)]["Preds"][data_split][f"Epoch-{epoch}"][f"Day-{day}"]['true_inf_risk_class'])
                            preds_inf_risk_all_days.append(np.argmax(results_h5_file[base_name_main_group+str(rep_id)]["Preds"][data_split][f"Epoch-{epoch}"][f"Day-{day}"]['seir_states_probs_inf_risk'], axis=1))
                            preds_inf_risk_all_days_probs.append(results_h5_file[base_name_main_group+str(rep_id)]["Preds"][data_split][f"Epoch-{epoch}"][f"Day-{day}"]['seir_states_probs_inf_risk'])
                    targets_all_days = np.concatenate(targets_all_days, axis=0)
                    preds_all_days = np.concatenate(preds_all_days, axis=0)
                    preds_all_days_probs = np.concatenate(preds_all_days_probs, axis=0)
                    if (targets_inf_risk_all_days_last_epoch is not None):
                        targets_inf_risk_all_days = np.concatenate(targets_inf_risk_all_days, axis=0)
                        preds_inf_risk_all_days = np.concatenate(preds_inf_risk_all_days, axis=0)
                        preds_inf_risk_all_days_probs = np.concatenate(preds_inf_risk_all_days_probs, axis=0)

                    # Reshaping targets_all_days, preds_all_days, and preds_all_days_probs in case of having a forecast horizon > 1
                    if (len(preds_all_days_probs.shape) > 2):
                        targets_all_days = targets_all_days.reshape(-1)
                        preds_all_days = preds_all_days.reshape(-1)
                        preds_all_days_probs = preds_all_days_probs.transpose(0, 2, 1).reshape(-1, preds_all_days_probs.shape[1])

                    # Last epochs targets and preds
                    if (epoch == max(epochs_list)):
                        # States preds
                        targets_all_days_last_epoch[rep_id][data_split] = targets_all_days
                        preds_all_days_last_epoch[rep_id][data_split] = preds_all_days
                        preds_probs_all_days_last_epoch[rep_id][data_split] = preds_all_days_probs
                        # Infection risks preds
                        if (targets_inf_risk_all_days_last_epoch is not None):
                            targets_inf_risk_all_days_last_epoch[rep_id][data_split] = targets_inf_risk_all_days
                            preds_inf_risk_all_days_last_epoch[rep_id][data_split] = preds_inf_risk_all_days
                            preds_probs_inf_risk_all_days_last_epoch[rep_id][data_split] = preds_inf_risk_all_days_probs
                    
                    # Getting the metrics for the epoch
                    print(f"\n\n=========> {data_split.upper()} DATA SPLIT FOR EPOCH {epoch} <=========\n")
                    # STATES predictions metrics
                    # MCC, F1-Score and Balanced Accuracy
                    N_UNIQUE_CLASSES = [i for i in range(preds_all_days_probs.shape[1])]
                    mcc, f1_score_val, balanced_acc = get_classification_metrics(
                                                                                    targets_all_days,
                                                                                    preds_all_days,
                                                                                    data_split,
                                                                                    N_UNIQUE_CLASSES,
                                                                                    verbose=verbose,
                                                                                    print_classification_report=print_classification_report
                                                                                )
                    # AUC
                    if (preds_all_days_probs.shape[1] == 2): # Binary classification
                        auc = roc_auc_score(targets_all_days, preds_all_days_probs[:, 1], average="macro")
                        random_preds_probs = np.random.dirichlet(alpha=np.ones(2), size=len(targets_all_days))
                        auc_random = roc_auc_score(targets_all_days, random_preds_probs[:, 1], average="macro") 
                    else:
                        # According to Scikit-learn roc_auc_score with multi_class='ovo' and average="macro" is insensitive to class imbalance 
                        auc = roc_auc_score(targets_all_days, preds_all_days_probs, multi_class='ovo', average="macro")
                        random_preds_probs = np.random.dirichlet(alpha=np.ones(max(N_UNIQUE_CLASSES)+1), size=len(targets_all_days))
                        auc_random = roc_auc_score(targets_all_days, random_preds_probs, multi_class='ovo', average="macro")
                    if (verbose):
                        print(f"\n{data_split} AUC: {auc}")
                        print(f"\t{data_split} AUC random classifier: {auc_random}")
                    # Per-class PR AUCs
                    if (preds_all_days_probs.shape[1] > 2):
                        per_class_pr_auc = []
                        per_class_pr_auc_random = []
                        for c in N_UNIQUE_CLASSES:
                            y_true_c = (targets_all_days == c).astype(int)
                            y_score_c = preds_all_days_probs[:, c]
                            y_score_random_c = random_preds_probs[:, c]
                            per_class_pr_auc.append(average_precision_score(y_true_c, y_score_c))
                            per_class_pr_auc_random.append(average_precision_score(y_true_c, y_score_random_c))
                        if (verbose):
                            print(f"\n{data_split} Per class AUC: {per_class_pr_auc}")
                            print(f"\t{data_split} Per class AUC random classifier: {per_class_pr_auc_random}")
                    
                    metrics_per_data_split["MCC"][data_split][epoch][rep_id] = mcc 
                    metrics_per_data_split["F1Score"][data_split][epoch][rep_id] = f1_score_val 
                    metrics_per_data_split["BalancedAccuracy"][data_split][epoch][rep_id] = balanced_acc 
                    metrics_per_data_split["AUC"][data_split][epoch][rep_id] = auc 
                    if (preds_all_days_probs.shape[1] > 2):
                        metrics_per_data_split["PerClassAUC"][data_split][epoch][rep_id] = per_class_pr_auc 
                    # INFECTION RISK prediction metrics
                    if (targets_inf_risk_all_days_last_epoch is not None):
                        # MCC, F1-Score and Balanced Accuracy
                        N_UNIQUE_CLASSES_INF_RISK = [i for i in range(preds_inf_risk_all_days_probs.shape[1])]
                        mcc_inf_risk, f1_score_val_inf_risk, balanced_acc_inf_risk = get_classification_metrics(
                                                                                        targets_inf_risk_all_days,
                                                                                        preds_inf_risk_all_days,
                                                                                        data_split,
                                                                                        N_UNIQUE_CLASSES_INF_RISK,
                                                                                        verbose=verbose,
                                                                                        print_classification_report=print_classification_report
                                                                                    )
                        # AUC
                        if (preds_inf_risk_all_days_probs.shape[1] == 2): # Binary classification
                            auc_inf_risk = roc_auc_score(targets_inf_risk_all_days, preds_inf_risk_all_days_probs[:, 1], average="macro")
                            random_preds_probs_inf_risk = np.random.dirichlet(alpha=np.ones(2), size=len(targets_inf_risk_all_days))
                            auc_random_inf_risk = roc_auc_score(targets_inf_risk_all_days, random_preds_probs_inf_risk[:, 1], average="macro") 
                        else:
                            # According to Scikit-learn roc_auc_score with multi_class='ovo' and average="macro" is insensitive to class imbalance 
                            auc_inf_risk = roc_auc_score(targets_inf_risk_all_days, preds_inf_risk_all_days_probs, multi_class='ovo', average="macro")
                            random_preds_probs_inf_risk = np.random.dirichlet(alpha=np.ones(max(N_UNIQUE_CLASSES_INF_RISK)+1), size=len(targets_inf_risk_all_days))
                            auc_random_inf_risk = roc_auc_score(targets_inf_risk_all_days, random_preds_probs_inf_risk, multi_class='ovo', average="macro")
                        if (verbose):
                            print(f"\n{data_split} AUC: {auc}")
                            print(f"\t{data_split} AUC random classifier: {auc_random}")
                        # Per-class PR AUCs
                        if (preds_inf_risk_all_days_probs.shape[1] > 2):
                            per_class_pr_auc_inf_risk = []
                            per_class_pr_auc_random_inf_risk = []
                            for c in N_UNIQUE_CLASSES_INF_RISK:
                                y_true_c_inf_risk = (targets_inf_risk_all_days == c).astype(int)
                                y_score_c_inf_risk = preds_inf_risk_all_days_probs[:, c]
                                y_score_random_c_inf_risk = random_preds_probs_inf_risk[:, c]
                                per_class_pr_auc_inf_risk.append(average_precision_score(y_true_c_inf_risk, y_score_c_inf_risk))
                                per_class_pr_auc_random_inf_risk.append(average_precision_score(y_true_c_inf_risk, y_score_random_c_inf_risk))
                            if (verbose):
                                print(f"\n{data_split} Per class AUC: {per_class_pr_auc_inf_risk}")
                                print(f"\t{data_split} Per class AUC random classifier: {per_class_pr_auc_random_inf_risk}")
                        
                        metrics_per_data_split["MCCInfectionRisk"][data_split][epoch][rep_id] = mcc_inf_risk
                        metrics_per_data_split["F1ScoreInfectionRisk"][data_split][epoch][rep_id] = f1_score_val_inf_risk
                        metrics_per_data_split["BalancedAccuracyInfectionRisk"][data_split][epoch][rep_id] = balanced_acc_inf_risk
                        metrics_per_data_split["AUCInfectionRisk"][data_split][epoch][rep_id] = auc_inf_risk
                        if (preds_inf_risk_all_days_probs.shape[1] > 2):
                            metrics_per_data_split["PerClassAUCInfectionRisk"][data_split][epoch][rep_id] = per_class_pr_auc_inf_risk

    # Plotting the different metrics over the epohs
    for metric_type in metrics_per_data_split:
        if (metric_type != 'PerClassAUC'):
            plot_metric_epochs(metric_dict=metrics_per_data_split[metric_type], metric_name=metric_type)

    # Plot the TEST metrics for the last epoc
    # STATES PREDICTIONS
    last_epoch = sorted(list(metrics_per_data_split["MCC"]["Test"].keys()))[-1]
    last_mcc_test_mean = np.mean(metrics_per_data_split["MCC"]["Test"][last_epoch])
    last_mcc_test_std = np.std(metrics_per_data_split["MCC"]["Test"][last_epoch])
    last_f1_score_test_mean = np.mean(metrics_per_data_split["F1Score"]["Test"][last_epoch])
    last_f1_score_test_std = np.std(metrics_per_data_split["F1Score"]["Test"][last_epoch])
    last_balanced_acc_test_mean = np.mean(metrics_per_data_split["BalancedAccuracy"]["Test"][last_epoch])
    last_balanced_acc_test_std = np.std(metrics_per_data_split["BalancedAccuracy"]["Test"][last_epoch])
    last_auc_test_mean = np.mean(metrics_per_data_split["AUC"]["Test"][last_epoch])
    last_auc_test_std = np.std(metrics_per_data_split["AUC"]["Test"][last_epoch])
    if (preds_all_days_probs.shape[1] > 2):
        last_per_class_auc_test_mean = np.mean(metrics_per_data_split["PerClassAUC"]["Test"][last_epoch], axis=0)
        last_per_class_auc_test_std = np.std(metrics_per_data_split["PerClassAUC"]["Test"][last_epoch], axis=0)
    print("\n=========> TEST MCC in the last epoch: {} +- {}%".format(last_mcc_test_mean, last_mcc_test_std))
    print("\tTEST F1 Score in the last epoch: {} +- {}%".format(last_f1_score_test_mean, last_f1_score_test_std))
    print("\tTEST Balanced Accuracy in the last epoch: {} +- {}%".format(last_balanced_acc_test_mean, last_balanced_acc_test_std))
    print("\tTEST AUC in the last epoch: {} +- {}%".format(last_auc_test_mean, last_auc_test_std))
    if (preds_all_days_probs.shape[1] > 2):
        print("\tTEST PER CLASS AUC in the last epoch: {} +- {}%".format(last_per_class_auc_test_mean, last_per_class_auc_test_std))
    # INFECTION RISK predictions
    if (targets_inf_risk_all_days_last_epoch is not None):
        last_mcc_inf_risk_test_mean = np.mean(metrics_per_data_split["MCCInfectionRisk"]["Test"][last_epoch])
        last_mcc_inf_risk_test_std = np.std(metrics_per_data_split["MCCInfectionRisk"]["Test"][last_epoch])
        last_f1_score_inf_risk_test_mean = np.mean(metrics_per_data_split["F1ScoreInfectionRisk"]["Test"][last_epoch])
        last_f1_score_inf_risk_test_std = np.std(metrics_per_data_split["F1ScoreInfectionRisk"]["Test"][last_epoch])
        last_balanced_acc_inf_risk_test_mean = np.mean(metrics_per_data_split["BalancedAccuracyInfectionRisk"]["Test"][last_epoch])
        last_balanced_acc_inf_risk_test_std = np.std(metrics_per_data_split["BalancedAccuracyInfectionRisk"]["Test"][last_epoch])
        last_auc_inf_risk_test_mean = np.mean(metrics_per_data_split["AUCInfectionRisk"]["Test"][last_epoch])
        last_auc_inf_risk_test_std = np.std(metrics_per_data_split["AUCInfectionRisk"]["Test"][last_epoch])
        if (preds_inf_risk_all_days_probs.shape[1] > 2):
            last_per_class_auc_inf_risk_test_mean = np.mean(metrics_per_data_split["PerClassAUCInfectionRisk"]["Test"][last_epoch], axis=0)
            last_per_class_auc_inf_risk_test_std = np.std(metrics_per_data_split["PerClassAUCInfectionRisk"]["Test"][last_epoch], axis=0)
        print("\n=========> TEST MCC INFECTION RISK in the last epoch: {} +- {}%".format(last_mcc_inf_risk_test_mean, last_mcc_inf_risk_test_std))
        print("\tTEST F1 Score INFECTION RISK in the last epoch: {} +- {}%".format(last_f1_score_inf_risk_test_mean, last_f1_score_inf_risk_test_std))
        print("\tTEST Balanced Accuracy INFECTION RISK in the last epoch: {} +- {}%".format(last_balanced_acc_inf_risk_test_mean, last_balanced_acc_inf_risk_test_std))
        print("\tTEST AUC INFECTION RISK in the last epoch: {} +- {}%".format(last_auc_inf_risk_test_mean, last_auc_inf_risk_test_std))
        if (preds_inf_risk_all_days_probs.shape[1] > 2):
            print("\tTEST PER CLASS AUC in the last epoch: {} +- {}%".format(last_per_class_auc_inf_risk_test_mean, last_per_class_auc_inf_risk_test_std))



    #======================================================================#
    #=====================Plot SEIR evolution over time=====================#
    #======================================================================#
    # Opening the parameters used for this experiment
    parameters_file = results_folder + f"/params_exp/params_0.pth"
    with open(parameters_file, 'rb') as pf:
        params_exp = pickle.load(pf)

    # Mapping
    if (params_exp['dataset_name'].lower() == 'sociopatterns'):
        SUSCEPTIBLE, EXPOSED, INFECTIOUS, RECOVERED = 0, 1, 2, 3 # states of the nodes
        MAPPING = {'S' : SUSCEPTIBLE, 'E' : EXPOSED, 'I' : INFECTIOUS, 'R' : RECOVERED}
    elif (params_exp['dataset_name'].lower() == 'murcia'):
        SUSCEPTIBLE, EXPOSED, INFECTIOUS, RECOVERED, DECEASED, NONSUSCEPTIBLE = 0, 1, 2, 3, 4, 5 # states of the nodes
        MAPPING = {'S' : SUSCEPTIBLE, 'E' : EXPOSED, 'I' : INFECTIOUS, 'R' : RECOVERED, 'D': DECEASED, 'NS': NONSUSCEPTIBLE}
    INV_MAPPING = {v: k for k, v in MAPPING.items()}

    # Data to use
    #epoch_to_use = 0
    #data_split = "Train"
    #data_split = "Val"
    data_split = "Test"
    last_epoch = max([int(epoch_str.split('-')[-1]) for epoch_str in list(results_h5_file[base_name_main_group+"0"]["Preds"][data_split].keys())])
    epoch_to_use = last_epoch

    # Plot
    plot_pred_epidemic_evolution(
                                    h5_results_file=results_h5_file,
                                    epoch_to_use=epoch_to_use,
                                    states_mapping=MAPPING,
                                    data_split=data_split
                                )
        


    #======================================================================#
    #=====================Calibration error=====================#
    #======================================================================#
    # Get the expected calibration error ECE
    #data_split_to_use = 'Val'
    data_split_to_use = 'Test'
    ece_per_rep = []
    N_BINS = 10 # Finer bins give more detail but can be noisy; coarser bins are smoother but less precise.
    #N_BINS = 15 # Finer bins give more detail but can be noisy; coarser bins are smoother but less precise.
    for rep_ID in preds_probs_all_days_last_epoch:
        ece_rep = compute_ece(
                                    probs=preds_probs_all_days_last_epoch[rep_ID][data_split_to_use],
                                    labels=targets_all_days_last_epoch[rep_ID][data_split_to_use],
                                    n_bins=N_BINS
                            )
        ece_per_rep.append(ece_rep)

    # Getting the mean ECE
    mean_ece = np.mean(ece_per_rep)
    std_ece = np.std(ece_per_rep)
    print(f"\nThe {data_split_to_use} ECE in the LAST epoch is: {mean_ece} +- {std_ece}\n")
    if (mean_ece < 0.05):
        print(f"\n\t=========>The model is relatively well calibrated\n")
    elif (mean_ece >= 0.05) and (mean_ece <= 0.1):
        print(f"\n\t=========>The model has acceptable calibrated BUT it has noticeable miscalibration\n")
    else:
        print(f"\n\t=========>The model has a significant calibration issue (common for deep networks, especially overconfident ones)\n")
        


if (__name__=='__main__'):
    main()