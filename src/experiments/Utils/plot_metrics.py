"""
    Plot the metrics and results of an experiment
"""
import h5py
import argparse
import yaml
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score,\
                            average_precision_score
from src.experiments.Utils.classification_metrics import get_classification_metrics,\
                                                         compute_ece
from src.experiments.Utils.uncertainty_quantification import get_evidential_uncertainties,\
                                                              plot_metric_vs_uncertainty,\
                                                              plot_calibration_uncertainty_correctness,\
                                                              plot_calibrated_uncertainty_correctness

# To use if plots are not showing after plt.shoy()
# import matplotlib
# matplotlib.use("Qt5Agg")  # or "Qt5Agg" if you have PyQt installed


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
    ap.add_argument('--plot_loss', help="Use it if want to plot the losses", action='store_true')
    ap.add_argument('--print_classification_report', help="Use it if want to print the classification report per epochs", action='store_true')
    ap.add_argument('--uncertainties', help="Use it if want to see the uncertainties of evidential learning experiments", action='store_true')
    ap.add_argument('-v', '--verbose', help="Use it to get details about the performances per epoch", action='store_true')
    args = vars(ap.parse_args())

    # Getting the value of the arguments
    results_folder = args['results_folder']
    plot_loss = args['plot_loss']
    print_classification_report = args['print_classification_report']
    uncertainties = args['uncertainties']
    verbose = args['verbose']
    
    #======================================================================#
    #===============================Load data===============================#
    #======================================================================#
    # Open the results file
    results_file = results_folder + "/metrics/final_results_all_repetitions_0.hdf5"
    results_h5_file = h5py.File(results_file, 'r')

    #======================================================================#
    #===========================Load params file===========================#
    #======================================================================#
    # Load params file to know if evidential learning was used during training
    parameters_file = results_folder + f"/params_exp/params_0.yaml"
    with open(parameters_file, 'r') as file:
        params_exp = yaml.safe_load(file)
    if (params_exp['Optimization']['loss_function'].lower() == 'evidentiallearningloss'):
        use_evidential_learning = True 
    else:
        use_evidential_learning = False

    #======================================================================#
    #===============================Plot loss===============================#
    #======================================================================#
    if (plot_loss):
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
    # Getting the predictions for each epoch 
    n_repetitions = len(results_h5_file)
    epochs_list = sorted([int(epoch_str.split('-')[-1]) for epoch_str in list(results_h5_file[base_name_main_group+"0"]["Preds"]["Train"].keys())])
    metrics_per_data_split = {
                                "MCC": {"Train": {epoch: [None for _ in range(n_repetitions)] for epoch in epochs_list}, "Val": {epoch: [None for _ in range(n_repetitions)] for epoch in epochs_list}, "Test": {epoch: [None for _ in range(n_repetitions)] for epoch in epochs_list}},
                                "F1Score": {"Train": {epoch: [None for _ in range(n_repetitions)] for epoch in epochs_list}, "Val": {epoch: [None for _ in range(n_repetitions)] for epoch in epochs_list}, "Test": {epoch: [None for _ in range(n_repetitions)] for epoch in epochs_list}},
                                "BalancedAccuracy": {"Train": {epoch: [None for _ in range(n_repetitions)] for epoch in epochs_list}, "Val": {epoch: [None for _ in range(n_repetitions)] for epoch in epochs_list}, "Test": {epoch: [None for _ in range(n_repetitions)] for epoch in epochs_list}},
                                "BalancedAccuracyAdjusted": {"Train": {epoch: [None for _ in range(n_repetitions)] for epoch in epochs_list}, "Val": {epoch: [None for _ in range(n_repetitions)] for epoch in epochs_list}, "Test": {epoch: [None for _ in range(n_repetitions)] for epoch in epochs_list}},
                                "AUC": {"Train": {epoch: [None for _ in range(n_repetitions)] for epoch in epochs_list}, "Val": {epoch: [None for _ in range(n_repetitions)] for epoch in epochs_list}, "Test": {epoch: [None for _ in range(n_repetitions)] for epoch in epochs_list}},
                                "PerClassAUC": {"Train": {epoch: [None for _ in range(n_repetitions)] for epoch in epochs_list}, "Val": {epoch: [None for _ in range(n_repetitions)] for epoch in epochs_list}, "Test": {epoch: [None for _ in range(n_repetitions)] for epoch in epochs_list}},
                            }
    targets_last_epoch = {rep_id : {data_split: None for data_split in list(results_h5_file[base_name_main_group+str(rep_id)]["Preds"].keys())} for rep_id in range(n_repetitions)}
    preds_last_epoch = {rep_id : {data_split: None for data_split in list(results_h5_file[base_name_main_group+str(rep_id)]["Preds"].keys())} for rep_id in range(n_repetitions)}
    preds_probs_last_epoch = {rep_id : {data_split: None for data_split in list(results_h5_file[base_name_main_group+str(rep_id)]["Preds"].keys())} for rep_id in range(n_repetitions)}
    for rep_id in range(n_repetitions):
        for data_split in list(results_h5_file[base_name_main_group+str(rep_id)]["Preds"].keys()):
            if (len(results_h5_file[base_name_main_group+str(rep_id)]["Preds"][data_split].keys()) > 0):
                for epoch in epochs_list:
                    # Get targets, preds and prediction scores
                    targets = results_h5_file[base_name_main_group+str(rep_id)]['Preds'][data_split][f'Epoch-{epoch}']['targets']
                    N_UNIQUE_CLASSES = np.unique(targets)
                    if (len(N_UNIQUE_CLASSES) == 2): # Bianry classificaiton
                        preds = results_h5_file[base_name_main_group+str(rep_id)]['Preds'][data_split][f'Epoch-{epoch}']['predictions']
                    else:
                        preds = np.argmax(results_h5_file[base_name_main_group+str(rep_id)]['Preds'][data_split][f'Epoch-{epoch}']['predictions'], axis=1)
                    preds_probs = results_h5_file[base_name_main_group+str(rep_id)]['Preds'][data_split][f'Epoch-{epoch}']['predictions_probs']

                    # Last epochs targets and preds
                    if (epoch == max(epochs_list)):
                        # States preds
                        targets_last_epoch[rep_id][data_split] = targets[:]
                        preds_last_epoch[rep_id][data_split] = preds[:]
                        preds_probs_last_epoch[rep_id][data_split] = preds_probs[:]

                    # Getting the metrics for the epoch
                    print(f"\n\n=========> {data_split.upper()} DATA SPLIT FOR EPOCH {epoch} <=========\n")
                    # STATES predictions metrics
                    # MCC, F1-Score and Balanced Accuracy
                    mcc, f1_score_val, balanced_acc, balanced_acc_adjusted = get_classification_metrics(
                                                                                                            targets,
                                                                                                            preds,
                                                                                                            data_split,
                                                                                                            N_UNIQUE_CLASSES,
                                                                                                            verbose=verbose,
                                                                                                            print_classification_report=print_classification_report
                                                                                                        )
                    # AUC
                    if (len(N_UNIQUE_CLASSES) == 2): # Binary classification
                        if (use_evidential_learning):
                            new_targets = np.zeros((targets.shape[0], 2), dtype=int)
                            new_targets[targets[:] == 0., 0] = 1
                            new_targets[targets[:] == 1., 1] = 1
                            auc = roc_auc_score(new_targets, preds_probs, average="macro")
                            random_preds_probs = np.random.dirichlet(alpha=np.ones(2), size=len(targets))
                            auc_random = roc_auc_score(new_targets, random_preds_probs, average="macro") 
                        else:
                            auc = roc_auc_score(targets, preds_probs, average="macro")
                            random_preds_probs = np.random.dirichlet(alpha=np.ones(2), size=len(targets))
                            auc_random = roc_auc_score(targets, random_preds_probs[:, 1], average="macro") 
                    else:
                        # According to Scikit-learn roc_auc_score with multi_class='ovo' and average="macro" is insensitive to class imbalance 
                        auc = roc_auc_score(targets, preds_probs, multi_class='ovo', average="macro")
                        random_preds_probs = np.random.dirichlet(alpha=np.ones(max(N_UNIQUE_CLASSES)+1), size=len(targets))
                        auc_random = roc_auc_score(targets, random_preds_probs, multi_class='ovo', average="macro")
                    if (verbose):
                        print(f"\n{data_split} AUC: {auc}")
                        print(f"\t{data_split} AUC random classifier: {auc_random}")
                    # Per-class PR AUCs
                    if (len(N_UNIQUE_CLASSES) > 2):
                        per_class_pr_auc = []
                        per_class_pr_auc_random = []
                        for c in N_UNIQUE_CLASSES:
                            y_true_c = (targets == c).astype(int)
                            y_score_c = preds_probs[:, c]
                            y_score_random_c = random_preds_probs[:, c]
                            per_class_pr_auc.append(average_precision_score(y_true_c, y_score_c))
                            per_class_pr_auc_random.append(average_precision_score(y_true_c, y_score_random_c))
                        if (verbose):
                            print(f"\n{data_split} Per class AUC: {per_class_pr_auc}")
                            print(f"\t{data_split} Per class AUC random classifier: {per_class_pr_auc_random}")
                    
                    metrics_per_data_split["MCC"][data_split][epoch][rep_id] = mcc 
                    metrics_per_data_split["F1Score"][data_split][epoch][rep_id] = f1_score_val 
                    metrics_per_data_split["BalancedAccuracy"][data_split][epoch][rep_id] = balanced_acc 
                    metrics_per_data_split["BalancedAccuracyAdjusted"][data_split][epoch][rep_id] = balanced_acc_adjusted
                    metrics_per_data_split["AUC"][data_split][epoch][rep_id] = auc 
                    if (len(N_UNIQUE_CLASSES) > 2):
                        metrics_per_data_split["PerClassAUC"][data_split][epoch][rep_id] = per_class_pr_auc 

    # Plotting the different metrics over the epohs
    for metric_type in metrics_per_data_split:
        if (metric_type != 'PerClassAUC'):
            plot_metric_epochs(metric_dict=metrics_per_data_split[metric_type], metric_name=metric_type)

    # Plot the TEST metrics for the last epoc
    # Predictions
    last_epoch = sorted(list(metrics_per_data_split["MCC"]["Test"].keys()))[-1]
    last_mcc_test_mean = np.mean(metrics_per_data_split["MCC"]["Test"][last_epoch])
    last_mcc_test_std = np.std(metrics_per_data_split["MCC"]["Test"][last_epoch])
    last_f1_score_test_mean = np.mean(metrics_per_data_split["F1Score"]["Test"][last_epoch])
    last_f1_score_test_std = np.std(metrics_per_data_split["F1Score"]["Test"][last_epoch])
    last_balanced_acc_test_mean = np.mean(metrics_per_data_split["BalancedAccuracy"]["Test"][last_epoch])
    last_balanced_acc_test_std = np.std(metrics_per_data_split["BalancedAccuracy"]["Test"][last_epoch])
    last_balanced_acc_adj_test_mean = np.mean(metrics_per_data_split["BalancedAccuracyAdjusted"]["Test"][last_epoch])
    last_balanced_acc_adj_test_std = np.std(metrics_per_data_split["BalancedAccuracyAdjusted"]["Test"][last_epoch])
    last_auc_test_mean = np.mean(metrics_per_data_split["AUC"]["Test"][last_epoch])
    last_auc_test_std = np.std(metrics_per_data_split["AUC"]["Test"][last_epoch])
    if (len(N_UNIQUE_CLASSES) > 2):
        last_per_class_auc_test_mean = np.mean(metrics_per_data_split["PerClassAUC"]["Test"][last_epoch], axis=0)
        last_per_class_auc_test_std = np.std(metrics_per_data_split["PerClassAUC"]["Test"][last_epoch], axis=0)
    print("\n=========> TEST MCC in the last epoch: {} +- {}%".format(last_mcc_test_mean, last_mcc_test_std))
    print("\tTEST F1 Score in the last epoch: {} +- {}%".format(last_f1_score_test_mean, last_f1_score_test_std))
    print("\tTEST Balanced Accuracy in the last epoch: {} +- {}%".format(last_balanced_acc_test_mean, last_balanced_acc_test_std))
    print("\tTEST Balanced Accuracy Adjusted in the last epoch: {} +- {}%".format(last_balanced_acc_adj_test_mean, last_balanced_acc_adj_test_std))
    print("\tTEST AUC in the last epoch: {} +- {}%".format(last_auc_test_mean, last_auc_test_std))
    if (len(N_UNIQUE_CLASSES) > 2):
        print("\tTEST PER CLASS AUC in the last epoch: {} +- {}%".format(last_per_class_auc_test_mean, last_per_class_auc_test_std))
    
    #======================================================================#
    #===========================Calibration error===========================#
    #======================================================================#
    # Get the expected calibration error ECE
    #data_split_to_use = 'Val'
    data_split_to_use = 'Test'
    ece_per_rep = []
    N_BINS = 10 # Finer bins give more detail but can be noisy; coarser bins are smoother but less precise.
    #N_BINS = 15 # Finer bins give more detail but can be noisy; coarser bins are smoother but less precise.
    for rep_ID in preds_probs_last_epoch:
        ece_rep = compute_ece(
                                    probs=preds_probs_last_epoch[rep_ID][data_split_to_use],
                                    labels=targets_last_epoch[rep_ID][data_split_to_use],
                                    n_bins=N_BINS,
                                    evidential=use_evidential_learning
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
        

    #======================================================================#
    #======================Uncertainty quantification======================#
    #======================================================================#
    if (uncertainties):
        if (use_evidential_learning):
            # Getting the average epistemic, aleatoric and total uncertainty for the different repetitions
            epoch_to_use = max(epochs_list)
            uncertainties_dict = get_evidential_uncertainties(
                                                                h5_results_file=results_h5_file,
                                                                epoch_to_use=epoch_to_use
                                                            )

            # Analyzing some uncertainties
            # Min and max possible values
            n_classes = len(N_UNIQUE_CLASSES)
            print("\n=========> EPISTEMIC uncertainty ranges between 0 and 1")
            print(f"\t=========> ALEATORIC uncertainty ranges between 0 and {np.log(n_classes)}")
            print(f"\t=========> TOTAL uncertainty ranges between 0 and {np.log(n_classes)}")


            # Get the average uncertainties per class
            # Group by class
            #data_split_to_use = 'Train' 
            #data_split_to_use = 'Val' 
            data_split_to_use = 'Test'
            uncert_by_class = {int(targ): {"Epistemic": [ [] for _ in range(n_repetitions) ], "Aleatoric": [ [] for _ in range(n_repetitions) ], "Total": [ [] for _ in range(n_repetitions) ]} for targ in np.unique(list(uncertainties_dict['Targets'].keys()))}
            for rep_ID in range(len(uncertainties_dict["Epistemic"])):
                for sample_ID in range(uncertainties_dict['Targets'][rep_ID][data_split_to_use].shape[0]):
                    # Target label
                    target_label = int(uncertainties_dict["Targets"][rep_ID][data_split_to_use][sample_ID])
                    
                    # Uncertainties
                    uncert_by_class[target_label]["Epistemic"][rep_ID].append(uncertainties_dict["Epistemic"][rep_ID][data_split_to_use][sample_ID])
                    uncert_by_class[target_label]["Aleatoric"][rep_ID].append(uncertainties_dict["Aleatoric"][rep_ID][data_split_to_use][sample_ID])
                    uncert_by_class[target_label]["Total"][rep_ID].append(uncertainties_dict["Total"][rep_ID][data_split_to_use][sample_ID])

            #======================================================================#
            #==================Uncertainty PLOTS (CALIBRATION)==================#
            #======================================================================#
            # Parameters for ALL THE PLOTS
            rep_ID_to_use = 0
            #metric_to_use = "MCC"
            #metric_to_use = "F1Score"
            metric_to_use = "BalancedAccuracy"
            #metric_to_use = "AUC"
            # Plot metric vs uncertainty
            #UNCERT_TYPE = "Epistemic"
            #UNCERT_TYPE = "Aleatoric"
            UNCERT_TYPE = "Total"
            plot_metric_vs_uncertainty(
                                        uncertainty_dict_per_rep=uncertainties_dict[UNCERT_TYPE],
                                        targets_dict_per_rep=targets_last_epoch,
                                        preds_dict_per_rep=preds_last_epoch,
                                        preds_probs_dict_per_rep=preds_probs_last_epoch,
                                        metric_to_use=metric_to_use,
                                        rep_id=rep_ID_to_use,
                                        data_split=data_split_to_use
                                    )
            
            # Uncertainty vs Correctness
            plot_calibration_uncertainty_correctness(
                                                        uncertainty_dict_per_rep=uncertainties_dict[UNCERT_TYPE],
                                                        targets_dict_per_rep=targets_last_epoch,
                                                        preds_dict_per_rep=preds_last_epoch,
                                                        rep_id=rep_ID_to_use,
                                                        data_split=data_split_to_use
                                                )
            
            # Plot calibrated uncertainty using the probability of correctness threshold
            plot_calibrated_uncertainty_correctness(
                                                    uncertainty_dict_per_rep=uncertainties_dict[UNCERT_TYPE],
                                                    targets_dict_per_rep=targets_last_epoch,
                                                    preds_dict_per_rep=preds_last_epoch,
                                                    preds_probs_dict_per_rep=preds_probs_last_epoch,
                                                    metric_to_use=metric_to_use,
                                                    rep_id=rep_ID_to_use,
                                                    data_split=data_split_to_use
                                                )
        else:
            print("\nUncertainty quantification can only be done for evidential learning experiments")
    
if (__name__=='__main__'):
    main()