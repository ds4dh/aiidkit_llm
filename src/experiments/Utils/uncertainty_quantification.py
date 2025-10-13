"""
    Plot the metrics and results of an experiment
"""
import h5py
import argparse
import yaml
import numpy as np
import matplotlib.pyplot as plt
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import matthews_corrcoef,\
                            f1_score,\
                            balanced_accuracy_score,\
                            roc_auc_score



def get_evidential_uncertainties(h5_results_file, epoch_to_use):
    # Getting the average epistemic, aleatoric and total uncertainty for the different repetitions
    n_repetitions = len(h5_results_file)
    data_splits = list(h5_results_file["Rep-0"]["Preds"].keys())
    uncertainties_dict = {
                                'Epistemic': { rep_ID: {data_split: None for data_split in data_splits} for rep_ID in range(n_repetitions) },
                                'Aleatoric': { rep_ID: {data_split: None for data_split in data_splits} for rep_ID in range(n_repetitions) },
                                'Total': { rep_ID: {data_split: None for data_split in data_splits} for rep_ID in range(n_repetitions) },
                                'Targets': { rep_ID: {data_split: None for data_split in data_splits} for rep_ID in range(n_repetitions) }
                            }  
    for data_split in data_splits:
        if ("aleatoric_uncert" in h5_results_file["Rep-0"]["Preds"][data_split]["Epoch-0"]):
            for rep_ID in range(n_repetitions):
                # Getting the uncertainties
                uncertainties_dict["Epistemic"][rep_ID][data_split] = h5_results_file[f"Rep-{rep_ID}"]["Preds"][data_split][f"Epoch-{epoch_to_use}"]['epistemic_uncert'][:]
                uncertainties_dict["Aleatoric"][rep_ID][data_split] = h5_results_file[f"Rep-{rep_ID}"]["Preds"][data_split][f"Epoch-{epoch_to_use}"]['aleatoric_uncert'][:]
                uncertainties_dict["Total"][rep_ID][data_split] = h5_results_file[f"Rep-{rep_ID}"]["Preds"][data_split][f"Epoch-{epoch_to_use}"]['total_uncert'][:]
                uncertainties_dict["Targets"][rep_ID][data_split] = h5_results_file[f"Rep-{rep_ID}"]["Preds"][data_split][f"Epoch-{epoch_to_use}"]['targets'][:]
        
    return uncertainties_dict


def plot_metric_vs_uncertainty(
                                uncertainty_dict_per_rep,
                                targets_dict_per_rep,
                                preds_dict_per_rep,
                                preds_probs_dict_per_rep=None,
                                metric_to_use='BalancedAccuracy',
                                rep_id=0,
                                data_split='Test'
                             ):
    """
        IMPORTANT: GENERATED WITH THE HELP OF CHAT-GPT.
        
        Plots a perfomance metric against the uncertainty of the predictions

        Parameters:
        -----------
        uncertainty_dict_per_rep: dict
            Dictionary containing the uncertainty all the samples.
            The keys are the repetitions IDs and the values
            are dicts having as keys the data splits, and as values
            a np.ndarray containing the uncertainties for all
            the days, for all the individuals (single dimension
            array combining all of this information).
        targets_dict_per_rep: dict
            Dictionary containing the targets of all the samples.
            The keys are the repetitions IDs and the values
            are dicts having as keys the data splits, and as values
            a np.ndarray containing the uncertainties for all
            the days, for all the individuals (single dimension
            array combining all of this information).
        preds_dict_per_rep: dict
            Dictionary containing the predictions of all the samples.
            The keys are the repetitions IDs and the values
            are dicts having as keys the data splits, and as values
            a np.ndarray containing the uncertainties for all
            the days, for all the individuals (single dimension
            array combining all of this information).
        preds_probs_dict_per_rep: dict
            Dictionary containing the predictions probabilities of all the samples.
            The keys are the repetitions IDs and the values
            are dicts having as keys the data splits, and as values
            a np.ndarray containing the uncertainties for all
            the days, for all the individuals (single dimension
            array combining all of this information). ONLY NECESSARY
            when metric_to_use is AUC.
        metric_to_use: str
            Metric to use for the plot.
        rep_id: int
            Repetition ID of the experiment to consider
        data_split: str
            Data split to use.
    """
    # Getting the useful data
    uncertainties = uncertainty_dict_per_rep[rep_id][data_split]
    true_labels = targets_dict_per_rep[rep_id][data_split]
    predictions = preds_dict_per_rep[rep_id][data_split]
    predictions_probs = preds_probs_dict_per_rep[rep_id][data_split]
    all_classes = np.unique(true_labels)
    
    # Define thresholds over uncertainty
    thresholds = np.linspace(uncertainties.min(), uncertainties.max(), 50)
    
    metric_values = []
    num_kept = []
    
    for t in thresholds:
        # Keep only predictions below the threshold
        mask = uncertainties <= t
        num_kept.append(np.sum(mask))
        
        if np.sum(mask) == 0:
            metric_values.append(np.nan)  # avoid empty slice
            continue
        
        filtered_true = true_labels[mask]
        filtered_pred = predictions[mask]
        filtered_pred_probs = predictions_probs[mask]

        if (metric_to_use.lower() == 'mcc'):
            metric = matthews_corrcoef(filtered_true, filtered_pred)
        elif (metric_to_use.lower() == 'balancedaccuracy'):
            metric = balanced_accuracy_score(filtered_true, filtered_pred)
        elif (metric_to_use.lower() == 'f1score'):
            metric = f1_score(filtered_true, filtered_pred, average="micro") 
        elif (metric_to_use.lower() == 'auc'):
            if (filtered_pred_probs.shape[1] == 2): # Binary classification
                metric = roc_auc_score(filtered_true, filtered_pred_probs[:, 1]) 
            else:
                metric = roc_auc_score(filtered_true, filtered_pred_probs, multi_class='ovo', average="macro", labels=all_classes)
        else:
            raise NotImplementedError()
        metric_values.append(metric)

    # Comput the metric for all the data
    if (metric_to_use.lower() == 'mcc'):
        metric_all = matthews_corrcoef(true_labels, predictions)
    elif (metric_to_use.lower() == 'balancedaccuracy'):
        metric_all = balanced_accuracy_score(true_labels, predictions)
    elif (metric_to_use.lower() == 'f1score'):
        metric_all = f1_score(true_labels, predictions, average="micro") 
    elif (metric_to_use.lower() == 'auc'):
        if (predictions_probs.shape[1] == 2): # Binary classification
            metric_all = roc_auc_score(true_labels, predictions_probs[:, 1])
        else:
            metric_all = roc_auc_score(true_labels, predictions_probs, multi_class='ovo', average="macro", labels=all_classes)
    else:
        raise NotImplementedError()

    
    # Plot metric
    fig, ax1 = plt.subplots(figsize=(10,6))
    
    color = 'tab:blue'
    ax1.set_xlabel("Uncertainty threshold")
    ax1.set_ylabel(f"{metric_to_use.upper()}", color=color)
    ax1.plot(thresholds, metric_values, marker='o', color=color, label=metric_to_use.upper())
    ax1.axhline(metric_all, color='gray', linestyle=':', label=f'{metric_to_use.upper()} (all data) = {metric_all:.3f}')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True)
    
    # Plot number of predictions kept on a second y-axis
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel("N° predictions kept", color=color)
    ax2.plot(thresholds, num_kept, marker='x', linestyle='--', color=color, label='Num kept')
    ax2.tick_params(axis='y', labelcolor=color)
    
    fig.tight_layout()
    plt.title(f"{metric_to_use.upper()} and N° Predictions Kept vs Uncertainty Threshold")
    plt.show()


def plot_calibration_uncertainty_correctness(
                                                uncertainty_dict_per_rep,
                                                targets_dict_per_rep,
                                                preds_dict_per_rep,
                                                rep_id,
                                                data_split
                                           ):
    """
        IMPORTANT: GENERATED WITH THE HELP OF CHAT-GPT  
        
        Plots the uncertainty vs the correctness.

        Parameters:
        -----------
        uncertainty_dict_per_rep: dict
            Dictionary containing the uncertainty all the samples.
            The keys are the repetitions IDs and the values
            are dicts having as keys the data splits, and as values
            a np.ndarray containing the uncertainties for all
            the days, for all the individuals (single dimension
            array combining all of this information).
        targets_dict_per_rep: dict
            Dictionary containing the targets of all the samples.
            The keys are the repetitions IDs and the values
            are dicts having as keys the data splits, and as values
            a np.ndarray containing the uncertainties for all
            the days, for all the individuals (single dimension
            array combining all of this information).
        preds_dict_per_rep: dict
            Dictionary containing the predictions of all the samples.
            The keys are the repetitions IDs and the values
            are dicts having as keys the data splits, and as values
            a np.ndarray containing the uncertainties for all
            the days, for all the individuals (single dimension
            array combining all of this information).
        rep_id: int
            Repetition ID of the experiment to consider
        data_split: str
            Data split to use.
    """
    # Getting the data
    uncertainties = uncertainty_dict_per_rep[rep_id][data_split]
    true_labels = targets_dict_per_rep[rep_id][data_split]
    predictions = preds_dict_per_rep[rep_id][data_split]
    correct = (predictions == true_labels).astype(int)

    # Plot
    plt.figure(figsize=(6,4))
    plt.scatter(uncertainties, correct, alpha=0.3)
    plt.xlabel("Uncertainty")
    plt.ylabel("Correct prediction")
    plt.title("Uncertainty vs correctness (1=True, 0=False)")
    plt.grid(True)
    plt.show()

    # Bin the uncertainties
    num_bins = 20
    bins = np.linspace(uncertainties.min(), uncertainties.max(), num_bins + 1)
    bin_indices = np.digitize(uncertainties, bins) - 1  # indices from 0 to num_bins-1
    
    # Compute average correctness per bin
    bin_centers = (bins[:-1] + bins[1:]) / 2
    prob_correct = np.zeros(num_bins)
    counts = np.zeros(num_bins)
    
    for i in range(num_bins):
        in_bin = bin_indices == i
        counts[i] = in_bin.sum()
        if counts[i] > 0:
            prob_correct[i] = correct[in_bin].mean()
        else:
            prob_correct[i] = np.nan  # no data in bin
    
    # Plot
    plt.figure(figsize=(8,5))
    plt.plot(bin_centers, prob_correct, marker='o')
    plt.xlabel("Uncertainty")
    plt.ylabel("Observed prob. of correctness")
    plt.title("Calibration: Uncertainty vs Prob. of Correctness")
    plt.grid(True)
    plt.show()
    
    # Optional: show counts
    plt.figure(figsize=(8,3))
    plt.bar(bin_centers, counts, width=(bins[1]-bins[0])*0.9)
    plt.xlabel("Uncertainty")
    plt.ylabel("N° of predictions per bin")
    plt.title("N° of predictions per uncertainty bin")
    plt.show()



def plot_calibrated_uncertainty_correctness(
                                                uncertainty_dict_per_rep,
                                                targets_dict_per_rep,
                                                preds_dict_per_rep,
                                                preds_probs_dict_per_rep=None,
                                                metric_to_use='BalancedAccuracy',
                                                rep_id=0,
                                                data_split='Test'
                                           ):
    """
        IMPORTANT: GENERATED WITH THE HELP OF CHAT-GPT  
        
        Plots the CALIBRATED uncertainty vs probability
        of the correctness THRESHOLD.

        Parameters:
        -----------
        uncertainty_dict_per_rep: dict
            Dictionary containing the uncertainty all the samples.
            The keys are the repetitions IDs and the values
            are dicts having as keys the data splits, and as values
            a np.ndarray containing the uncertainties for all
            the days, for all the individuals (single dimension
            array combining all of this information).
        targets_dict_per_rep: dict
            Dictionary containing the targets of all the samples.
            The keys are the repetitions IDs and the values
            are dicts having as keys the data splits, and as values
            a np.ndarray containing the uncertainties for all
            the days, for all the individuals (single dimension
            array combining all of this information).
        preds_dict_per_rep: dict
            Dictionary containing the predictions of all the samples.
            The keys are the repetitions IDs and the values
            are dicts having as keys the data splits, and as values
            a np.ndarray containing the uncertainties for all
            the days, for all the individuals (single dimension
            array combining all of this information).
        preds_probs_dict_per_rep: dict
            Dictionary containing the predictions probabilities of all the samples.
            The keys are the repetitions IDs and the values
            are dicts having as keys the data splits, and as values
            a np.ndarray containing the uncertainties for all
            the days, for all the individuals (single dimension
            array combining all of this information). ONLY NECESSARY
            when metric_to_use is AUC.
        metric_to_use: str
            Metric to use for the plot.
        rep_id: int
            Repetition ID of the experiment to consider
        data_split: str
            Data split to use.
    """
    # Step 1: Compute correctness (1 if correct, 0 if wrong)
    uncertainties = uncertainty_dict_per_rep[rep_id][data_split]
    true_labels = targets_dict_per_rep[rep_id][data_split]
    predictions = preds_dict_per_rep[rep_id][data_split]
    predictions_probs = preds_probs_dict_per_rep[rep_id][data_split]
    all_classes = np.unique(true_labels)
    correct = (predictions == true_labels).astype(int)
    
    # Step 2: Fit isotonic regression: uncertainty -> probability of correctness
    # We invert uncertainty because higher uncertainty should correspond to lower correctness
    iso_reg = IsotonicRegression(y_min=0, y_max=1, increasing=False)
    iso_reg.fit(uncertainties, correct)
    
    # Step 3: Get calibrated probabilities
    calibrated_probs = iso_reg.predict(uncertainties)
    
    # Optional: Plot calibration curve
    plt.figure(figsize=(8,5))
    plt.scatter(uncertainties, correct, alpha=0.2, label='Original data')
    plt.plot(np.sort(uncertainties), iso_reg.predict(np.sort(uncertainties)),
             color='red', label='Isotonic calibration')
    plt.xlabel("Uncertainty")
    plt.ylabel("Observed probability of correctness")
    plt.title("Calibration: Uncertainty vs Correctness")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Step 4: Filter predictions based on calibrated probability threshold to maximize metric
    thresholds = np.linspace(0, 1, 50)
    metric_values = []
    num_kept = []
    
    for t in thresholds:
        mask = calibrated_probs >= t  # keep predictions with high probability of being correct
        num_kept.append(np.sum(mask))
        
        if np.sum(mask) == 0:
            metric_values.append(np.nan)
            continue
        
        filtered_true = true_labels[mask]
        filtered_pred = predictions[mask]
        filtered_pred_probs = predictions_probs[mask]

        if (metric_to_use.lower() == 'mcc'):
            metric = matthews_corrcoef(filtered_true, filtered_pred)
        elif (metric_to_use.lower() == 'balancedaccuracy'):
            metric = balanced_accuracy_score(filtered_true, filtered_pred)
        elif (metric_to_use.lower() == 'f1score'):
            metric = f1_score(filtered_true, filtered_pred, average="micro") 
        elif (metric_to_use.lower() == 'auc'):
            if (filtered_pred_probs.shape[1] == 2): # Binary classification
                metric = roc_auc_score(filtered_true, filtered_pred_probs[:, 1]) 
            else:
                metric = roc_auc_score(filtered_true, filtered_pred_probs, multi_class='ovo', average="macro", labels=all_classes)
        else:
            raise NotImplementedError()
        metric_values.append(metric)
    
    # Step 5: Plot metric vs number of predictions kept
    fig, ax1 = plt.subplots(figsize=(10,6))
    
    # Step 6: Compute metric using all data (baseline)
    if (metric_to_use.lower() == 'mcc'):
        metric_all = matthews_corrcoef(true_labels, predictions)
    elif (metric_to_use.lower() == 'balancedaccuracy'):
        metric_all = balanced_accuracy_score(true_labels, predictions)
    elif (metric_to_use.lower() == 'f1score'):
        metric_all = f1_score(true_labels, predictions, average="micro") 
    elif (metric_to_use.lower() == 'auc'):
        if (predictions_probs.shape[1] == 2): # Binary classification
            metric_all = roc_auc_score(true_labels, predictions_probs[:, 1])
        else:
            metric_all = roc_auc_score(true_labels, predictions_probs, multi_class='ovo', average="macro", labels=all_classes)
    else:
        raise NotImplementedError()
    
    color = 'tab:blue'
    ax1.set_xlabel("Porb. of correctness threshold")
    ax1.set_ylabel(f"{metric_to_use.upper()}", color=color)
    ax1.plot(thresholds, metric_values, marker='o', color=color, label=metric_to_use.upper())
    ax1.axhline(metric_all, color='gray', linestyle=':', label=f'{metric_to_use.upper()} (all data) = {metric_all:.3f}')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True)
    
    # Optional: plot number of predictions kept on a second y-axis
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.plot(thresholds, num_kept, marker='x', linestyle='--', color=color, label='N° kept')
    ax2.set_ylabel("N° of preds kept", color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    
    fig.tight_layout()
    plt.title(f"{metric_to_use.upper()} and N° of Preds Kept vs Calibrated Prob. Threshold")
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
    args = vars(ap.parse_args())

    # Getting the value of the arguments
    results_folder = args['results_folder']
    
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
    #===========================Use of multiple DS===========================#
    #======================================================================#
    # Getting the metrics dicts for the different losses
    if ('Rep-0' in results_h5_file):
        base_name_main_group = 'Rep-'
        multiple_datasets = False
    else:
        base_name_main_group = 'Rep-0_Dataset-'
        multiple_datasets = True


    #======================================================================#
    #========================Get performance metrics========================#
    #======================================================================#
    # Getting the predictions for each epoch 
    n_repetitions = len(results_h5_file)
    epochs_list = sorted([int(epoch_str.split('-')[-1]) for epoch_str in list(results_h5_file[base_name_main_group+"0"]["Preds"]["Train"].keys())])
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
                    
    #======================================================================#
    #==================Uncertainty PLOTS (NO CALIBRATION)==================#
    #======================================================================#
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