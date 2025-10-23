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
from src.experiments.Utils.plot_metrics import plot_metric_epochs
from src.experiments.Utils.classification_metrics import get_classification_metrics,\
                                                         compute_ece
from src.experiments.Utils.uncertainty_quantification import get_evidential_uncertainties,\
                                                              plot_metric_vs_uncertainty,\
                                                              plot_calibration_uncertainty_correctness,\
                                                              plot_calibrated_uncertainty_correctness


# Ignore all warnings
import warnings
warnings.filterwarnings("ignore")

# To use if plots are not showing after plt.shoy()
# import matplotlib
# matplotlib.use("Qt5Agg")  # or "Qt5Agg" if you have PyQt installed

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
        if ('Rep-0_DS-0' in results_h5_file):
            base_name_main_group = 'Rep-'
            multiple_datasets = False
        else:
            raise RuntimeError("HDF5 results files for CL should contain groups of the form Rep-ID1_DS-ID2")
        loss_names = list(results_h5_file[base_name_main_group+"0_DS-0"]['Loss']['Train'].keys())
        reps_names = []
        for key in results_h5_file:
            reps_names.append(key.split('_')[0])
        reps_names = np.unique(reps_names)
        n_repetitions = len(reps_names)

        # Getting the IDs of the incremental DS
        incremental_DS_IDs = []
        for key in results_h5_file:
            tmp_ID = key.split('DS-')[-1]
            if ('Rep' not in tmp_ID):
                incremental_DS_IDs.append(int(tmp_ID))
        incremental_DS_IDs = np.unique(incremental_DS_IDs)

        # Getting the number of iterations per repetition
        # As we are in a CL framework, the total number of iterations is the sum of the number of epochs per training dataset
        n_iterations_per_rep_per_data_split = {} 
        for rep_id in range(n_repetitions):
            n_iterations_per_rep_per_data_split[rep_id] = {}
            for data_split in list(results_h5_file[base_name_main_group+f'{rep_id}_DS-0']['Loss'].keys()):
                n_iterations = 0
                for DS_ID in incremental_DS_IDs: # WE HAVE TO ITERATE FIRST OVER DS AS THE MODELS ARE TRAINED SEQUENTIALLY ON THE INCREMENTAL DS (so the epochs of model in DS-0 are the first iterations, then there are those of DS-1, etc.)
                    if (len(results_h5_file[base_name_main_group+f'{rep_id}_DS-{DS_ID}']['Loss'][data_split].keys()) > 0):
                        n_epochs = len(results_h5_file[base_name_main_group+f'{rep_id}_DS-{DS_ID}']['Loss'][data_split]['total_loss'])
                        n_iterations += n_epochs
                n_iterations_per_rep_per_data_split[rep_id][data_split] = n_iterations

        # Creating the loss dict
        losses_dict = {}
        for loss_name in loss_names:
            losses_dict[loss_name] = {}
            for data_split in ["Train", "Val", "Test"]:
                if (data_split in results_h5_file[base_name_main_group+"0_DS-0"]['Loss']):
                    if (len(results_h5_file[base_name_main_group+"0_DS-0"]['Loss'][data_split].keys()) > 0):
                        n_iter = n_iterations_per_rep_per_data_split[rep_id][data_split]
                        losses_dict[loss_name][data_split] = [[None for _ in range(n_repetitions)] for _ in range(n_iter)]
        # Filling the loss dict
        for loss_name in loss_names:
            for rep_id in range(n_repetitions):
                for data_split in list(results_h5_file[base_name_main_group+f'{rep_id}_DS-0']['Loss'].keys()):
                    current_iteration = 0
                    for DS_ID in incremental_DS_IDs: # WE HAVE TO ITERATE FIRST OVER DS AS THE MODELS ARE TRAINED SEQUENTIALLY ON THE INCREMENTAL DS (so the epochs of model in DS-0 are the first iterations, then there are those of DS-1, etc.)
                        if (len(results_h5_file[base_name_main_group+f'{rep_id}_DS-{DS_ID}']['Loss'][data_split].keys()) > 0):
                            n_epochs = len(results_h5_file[base_name_main_group+f'{rep_id}_DS-{DS_ID}']['Loss'][data_split][loss_name])
                            for epoch in range(n_epochs):
                                losses_dict[loss_name][data_split][current_iteration][rep_id] = results_h5_file[base_name_main_group+f'{rep_id}_DS-{DS_ID}']['Loss'][data_split][loss_name][epoch]
                                current_iteration += 1

        # Plotting the different losses
        for loss_type in loss_names:
            plot_metric_epochs(metric_dict=losses_dict[loss_type], metric_name=loss_type)

    #======================================================================#
    #========================Get performance metrics========================#
    #======================================================================#
    # Getting the number of iterations per repetition FOR THE PREDICTIONS 
    # As we are in a CL framework, the total number of iterations is the sum of the number of epochs per training dataset
    n_iterations_preds_per_rep_per_data_split = {} 
    for rep_id in range(n_repetitions):
        n_iterations_preds_per_rep_per_data_split[rep_id] = {}
        for data_split in list(results_h5_file[base_name_main_group+f'{rep_id}_DS-0']['Preds'].keys()):
            n_iterations = 0
            for DS_ID in incremental_DS_IDs: # WE HAVE TO ITERATE FIRST OVER DS AS THE MODELS ARE TRAINED SEQUENTIALLY ON THE INCREMENTAL DS (so the epochs of model in DS-0 are the first iterations, then there are those of DS-1, etc.)
                if (len(results_h5_file[base_name_main_group+f'{rep_id}_DS-{DS_ID}']['Preds'][data_split].keys()) > 0):
                    n_epochs = len(results_h5_file[base_name_main_group+f'{rep_id}_DS-{DS_ID}']['Preds'][data_split])
                    n_iterations += n_epochs
            n_iterations_preds_per_rep_per_data_split[rep_id][data_split] = n_iterations

    # Getting the predictions for each iteration 
    metrics_per_data_split = {
                                "MCC": {"Train": {iteration: [None for _ in range(n_repetitions)] for iteration in range(n_iterations_preds_per_rep_per_data_split[0]['Train'])}, "Val": {iteration: [None for _ in range(n_repetitions)] for iteration in range(n_iterations_preds_per_rep_per_data_split[0]['Val'])}, "Test": {iteration: [None for _ in range(n_repetitions)] for iteration in range(n_iterations_preds_per_rep_per_data_split[0]['Test'])}},
                                "F1Score": {"Train": {iteration: [None for _ in range(n_repetitions)] for iteration in range(n_iterations_preds_per_rep_per_data_split[0]['Train'])}, "Val": {iteration: [None for _ in range(n_repetitions)] for iteration in range(n_iterations_preds_per_rep_per_data_split[0]['Val'])}, "Test": {iteration: [None for _ in range(n_repetitions)] for iteration in range(n_iterations_preds_per_rep_per_data_split[0]['Test'])}},
                                "BalancedAccuracy": {"Train": {iteration: [None for _ in range(n_repetitions)] for iteration in range(n_iterations_preds_per_rep_per_data_split[0]['Train'])}, "Val": {iteration: [None for _ in range(n_repetitions)] for iteration in range(n_iterations_preds_per_rep_per_data_split[0]['Val'])}, "Test": {iteration: [None for _ in range(n_repetitions)] for iteration in range(n_iterations_preds_per_rep_per_data_split[0]['Test'])}},
                                "BalancedAccuracyAdjusted": {"Train": {iteration: [None for _ in range(n_repetitions)] for iteration in range(n_iterations_preds_per_rep_per_data_split[0]['Train'])}, "Val": {iteration: [None for _ in range(n_repetitions)] for iteration in range(n_iterations_preds_per_rep_per_data_split[0]['Val'])}, "Test": {iteration: [None for _ in range(n_repetitions)] for iteration in range(n_iterations_preds_per_rep_per_data_split[0]['Test'])}},
                                "AUC": {"Train": {iteration: [None for _ in range(n_repetitions)] for iteration in range(n_iterations_preds_per_rep_per_data_split[0]['Train'])}, "Val": {iteration: [None for _ in range(n_repetitions)] for iteration in range(n_iterations_preds_per_rep_per_data_split[0]['Val'])}, "Test": {iteration: [None for _ in range(n_repetitions)] for iteration in range(n_iterations_preds_per_rep_per_data_split[0]['Test'])}},
                                "PerClassAUC": {"Train": {iteration: [None for _ in range(n_repetitions)] for iteration in range(n_iterations_preds_per_rep_per_data_split[0]['Train'])}, "Val": {iteration: [None for _ in range(n_repetitions)] for iteration in range(n_iterations_preds_per_rep_per_data_split[0]['Val'])}, "Test": {iteration: [None for _ in range(n_repetitions)] for iteration in range(n_iterations_preds_per_rep_per_data_split[0]['Test'])}}
                            }
    targets_last_iteration = {rep_id : {data_split: None for data_split in list(results_h5_file[base_name_main_group+f'{rep_id}_DS-0']["Preds"].keys())} for rep_id in range(n_repetitions)}
    preds_last_iteration = {rep_id : {data_split: None for data_split in list(results_h5_file[base_name_main_group+f'{rep_id}_DS-0']["Preds"].keys())} for rep_id in range(n_repetitions)}
    preds_probs_last_iteration = {rep_id : {data_split: None for data_split in list(results_h5_file[base_name_main_group+f'{rep_id}_DS-0']["Preds"].keys())} for rep_id in range(n_repetitions)}
    for rep_id in range(n_repetitions):
        for data_split in list(results_h5_file[base_name_main_group+f'{rep_id}_DS-0']["Preds"].keys()):
            current_iteration = 0
            n_iter = n_iterations_preds_per_rep_per_data_split[rep_id][data_split]
            for DS_ID in incremental_DS_IDs: # WE HAVE TO ITERATE FIRST OVER DS AS THE MODELS ARE TRAINED SEQUENTIALLY ON THE INCREMENTAL DS (so the epochs of model in DS-0 are the first iterations, then there are those of DS-1, etc.)
                if (len(results_h5_file[base_name_main_group+f'{rep_id}_DS-{DS_ID}']["Preds"][data_split].keys()) > 0):
                    epochs_list = sorted([int(epoch_str.split('-')[-1]) for epoch_str in list(results_h5_file[base_name_main_group+f'{rep_id}_DS-{DS_ID}']["Preds"][data_split].keys())])
                    for epoch in epochs_list:
                        # Get targets, preds and prediction scores
                        targets = results_h5_file[base_name_main_group+f'{rep_id}_DS-{DS_ID}']['Preds'][data_split][f'Epoch-{epoch}']['targets']
                        N_UNIQUE_CLASSES = np.unique(targets)
                        if (len(N_UNIQUE_CLASSES) == 2): # Bianry classificaiton
                            preds = results_h5_file[base_name_main_group+f'{rep_id}_DS-{DS_ID}']['Preds'][data_split][f'Epoch-{epoch}']['predictions']
                        else:
                            preds = np.argmax(results_h5_file[base_name_main_group+f'{rep_id}_DS-{DS_ID}']['Preds'][data_split][f'Epoch-{epoch}']['predictions'], axis=1)
                        preds_probs = results_h5_file[base_name_main_group+f'{rep_id}_DS-{DS_ID}']['Preds'][data_split][f'Epoch-{epoch}']['predictions_probs']

                        # Last iterations targets and preds
                        if (current_iteration == (n_iter-1)):
                            # States preds
                            targets_last_iteration[rep_id][data_split] = targets[:]
                            preds_last_iteration[rep_id][data_split] = preds[:]
                            preds_probs_last_iteration[rep_id][data_split] = preds_probs[:]

                        # Getting the metrics for the iteration
                        print(f"\n\n=========> {data_split.upper()} DATA SPLIT FOR ITERATION {current_iteration} <=========\n")
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
                        
                        metrics_per_data_split["MCC"][data_split][current_iteration][rep_id] = mcc 
                        metrics_per_data_split["F1Score"][data_split][current_iteration][rep_id] = f1_score_val 
                        metrics_per_data_split["BalancedAccuracy"][data_split][current_iteration][rep_id] = balanced_acc 
                        metrics_per_data_split["BalancedAccuracyAdjusted"][data_split][current_iteration][rep_id] = balanced_acc_adjusted
                        metrics_per_data_split["AUC"][data_split][current_iteration][rep_id] = auc 
                        if (len(N_UNIQUE_CLASSES) > 2):
                            metrics_per_data_split["PerClassAUC"][data_split][current_iteration][rep_id] = per_class_pr_auc 
                        current_iteration += 1

    # Plotting the different metrics over the epohs
    for metric_type in metrics_per_data_split:
        if (metric_type != 'PerClassAUC'):
            plot_metric_epochs(metric_dict=metrics_per_data_split[metric_type], metric_name=metric_type)

    # Plot the TEST metrics for the last epoc
    # Predictions
    last_iteration = sorted(list(metrics_per_data_split["MCC"]["Test"].keys()))[-1]
    last_mcc_test_mean = np.mean(metrics_per_data_split["MCC"]["Test"][last_iteration])
    last_mcc_test_std = np.std(metrics_per_data_split["MCC"]["Test"][last_iteration])
    last_f1_score_test_mean = np.mean(metrics_per_data_split["F1Score"]["Test"][last_iteration])
    last_f1_score_test_std = np.std(metrics_per_data_split["F1Score"]["Test"][last_iteration])
    last_balanced_acc_test_mean = np.mean(metrics_per_data_split["BalancedAccuracy"]["Test"][last_iteration])
    last_balanced_acc_test_std = np.std(metrics_per_data_split["BalancedAccuracy"]["Test"][last_iteration])
    last_balanced_acc_adj_test_mean = np.mean(metrics_per_data_split["BalancedAccuracyAdjusted"]["Test"][last_iteration])
    last_balanced_acc_adj_test_std = np.std(metrics_per_data_split["BalancedAccuracyAdjusted"]["Test"][last_iteration])
    last_auc_test_mean = np.mean(metrics_per_data_split["AUC"]["Test"][last_iteration])
    last_auc_test_std = np.std(metrics_per_data_split["AUC"]["Test"][last_iteration])
    if (len(N_UNIQUE_CLASSES) > 2):
        last_per_class_auc_test_mean = np.mean(metrics_per_data_split["PerClassAUC"]["Test"][last_iteration], axis=0)
        last_per_class_auc_test_std = np.std(metrics_per_data_split["PerClassAUC"]["Test"][last_iteration], axis=0)
    print("\n=========> TEST MCC in the last iteration: {} +- {}%".format(last_mcc_test_mean, last_mcc_test_std))
    print("\tTEST F1 Score in the last iteration: {} +- {}%".format(last_f1_score_test_mean, last_f1_score_test_std))
    print("\tTEST Balanced Accuracy in the last iteration: {} +- {}%".format(last_balanced_acc_test_mean, last_balanced_acc_test_std))
    print("\tTEST Balanced Accuracy Adjusted in the last iteration: {} +- {}%".format(last_balanced_acc_adj_test_mean, last_balanced_acc_adj_test_std))
    print("\tTEST AUC in the last iteration: {} +- {}%".format(last_auc_test_mean, last_auc_test_std))
    if (len(N_UNIQUE_CLASSES) > 2):
        print("\tTEST PER CLASS AUC in the last iteration: {} +- {}%".format(last_per_class_auc_test_mean, last_per_class_auc_test_std))

    #======================================================================#
    #===========================Calibration error===========================#
    #======================================================================#
    # Get the expected calibration error ECE
    #data_split_to_use = 'Val'
    data_split_to_use = 'Test'
    ece_per_rep = []
    N_BINS = 10 # Finer bins give more detail but can be noisy; coarser bins are smoother but less precise.
    #N_BINS = 15 # Finer bins give more detail but can be noisy; coarser bins are smoother but less precise.
    for rep_ID in preds_probs_last_iteration:
        ece_rep = compute_ece(
                                    probs=preds_probs_last_iteration[rep_ID][data_split_to_use],
                                    labels=targets_last_iteration[rep_ID][data_split_to_use],
                                    n_bins=N_BINS,
                                    evidential=use_evidential_learning
                            )
        ece_per_rep.append(ece_rep)

    # Getting the mean ECE
    mean_ece = np.mean(ece_per_rep)
    std_ece = np.std(ece_per_rep)
    print(f"\nThe {data_split_to_use} ECE in the LAST iteration is: {mean_ece} +- {std_ece}\n")
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
            raise NotImplementedError()
        else:
            print("\nUncertainty quantification can only be done for evidential learning experiments")
    
if (__name__=='__main__'):
    main()