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
    K = dirichlet_alpha.size(1)
    S = torch.sum(dirichlet_alpha, dim=1, keepdim=True)
    probs = dirichlet_alpha / S

    # Epistemic: high when total evidence is low
    epistemic = K / S.squeeze()

    # Aleatoric: expected entropy under Dirichlet
    psi_alpha_plus_1 = digamma(dirichlet_alpha + 1)
    psi_S_plus_1 = digamma(S + 1)
    aleatoric = -torch.sum(probs * (psi_alpha_plus_1 - psi_S_plus_1), dim=1)

    # Total: entropy of the mean
    total = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)

    return epistemic, aleatoric, total
