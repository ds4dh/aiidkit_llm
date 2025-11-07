"""
    This code implements some useful functions to train
    GNN models for infection risk prediction on graph-based
    datasets
"""
import torch
from torch.special import digamma


def get_uncertainties(dirichlet_alpha):
    """
        Computes the epistemic, aleatoric and total uncertainties
        for the output of an evidential learning Dirichlet-based
        classificaiton model.

        IMPORTANT: The formulas for the quantification of epistemic
        and aleatoric uncertainties come from equations 21 and 23
        of the following paper: https://journals.ametsoc.org/view/journals/aies/3/4/AIES-D-23-0093.1.xml

        Parameters:
        -----------
        dirichlet_alpha: torch.tensor
            Alphas of the Dirichlet distribution, prediction of 
            a Dirichlet-based evidential learning model. It is 
            of shape (batch_size, n_classes)

        Returns:
        --------
        epistemic: torch.Tensor
            Epistemic (model-dependent and reducible) uncertainty.
            It ranges between 0 and 1, and smaller values are better.
        aleatoric: torch.Tensor
            Aleatoric (data-dependent and non-reducible) uncertainty.
            It ranges between 0 and log_e(K) where K is the number of
            classes (for 4 classes log_e(K) ~ 1.386). Smaller values are better.
        total: torch.Tensor
            Total uncertainty.
            It ranges between 0 and log_e(K) where K is the number of
            classes (for 4 classes log_e(K) ~ 1.386). Smaller values are better.
    """
    # Getting number of samples N and number of classes K
    N = dirichlet_alpha.shape[0]
    K = dirichlet_alpha.shape[1]

    # Getting the sum of alphas per sample
    S = torch.sum(dirichlet_alpha, dim=1, keepdim=True)

    # Getting the probability score for each sample and class
    probs = dirichlet_alpha / S

    # Total uncertainty
    total = probs - probs**2 # Obtained by summing Eqs. 21 and 23 of https://journals.ametsoc.org/view/journals/aies/3/4/AIES-D-23-0093.1.xml

    # Epistemic uncertainty
    epistemic = probs*(1-probs)/(S+1)

    # Aleatoric uncertainty
    aleatoric = total - epistemic

    return epistemic, aleatoric, total