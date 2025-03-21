import math

import torch
from torch import nn, Tensor

import matplotlib.pyplot as plt
import numpy as np

from permutect.architecture.spectra.artifact_spectra import ArtifactSpectra
from permutect.utils.enums import Variation
from permutect.utils.stats_utils import beta_binomial_log_lk

EPSILON = 0.001


class NormalArtifactSpectrum(nn.Module):
    """
    Joint distribution for normal and tumor.  P(normal alt, tumor alt | normal depth, tumor depth) is modeled as
    P(normal alt | normal depth) * P(tumor alt | normal allele fraction, tumor depth)

    The first part P(normal alt | normal depth) is an instance of ArtifactSpectra.  Importantly, it's independent of
    the ArtifactSpectra used in the posterior model for tumor artifact allele fractions.
    """
    def __init__(self):
        super(NormalArtifactSpectrum, self).__init__()
        V = len(Variation)

        self.normal_spectrum = ArtifactSpectra()

        # mean of tumor beta is type-dependent multiplier of normal AF.  Sigmoid will map it onto [0,1]
        self.mean_multiplier_pre_sigmoid_v = torch.nn.Parameter(torch.zeros(V))
        self.concentration_pre_exp_v = torch.nn.Parameter(torch.log(30*torch.ones(V)))  # alpha + beta parameters

    def forward(self, var_types_b, tumor_alt_counts_b: Tensor, tumor_depths_b: Tensor, normal_alt_counts_b: Tensor, normal_depths_b: Tensor):
        normal_log_lks_b = self.normal_spectrum.forward(var_types_b, normal_depths_b, normal_alt_counts_b)

        mean_multiplier_v = torch.sigmoid(self.mean_multiplier_pre_sigmoid_v)
        mean_multiplier_b = mean_multiplier_v[var_types_b]
        concentration_v = torch.exp(self.concentration_pre_exp_v)
        concentration_b = concentration_v[var_types_b]

        normal_af_b = normal_alt_counts_b / (normal_depths_b + 0.001)
        tumor_mean_b = normal_af_b * mean_multiplier_b

        # mean = alpha / (alpha + beta)
        alpha_b = 0.001 + tumor_mean_b * concentration_b
        beta_b = concentration_b - alpha_b
        tumor_log_lks_b = beta_binomial_log_lk(n=tumor_depths_b, k=tumor_alt_counts_b, alpha=alpha_b, beta=beta_b)

        return tumor_log_lks_b, normal_log_lks_b


    # TODO: move this method to plotting
    #def density_plot_on_axis(self, ax):
    #    tumor_fractions_2d, normal_fractions_2d = self.get_tumor_and_normal_fractions_bs(batch_size=1, num_samples=100000)
    #    tumor_f = torch.squeeze(tumor_fractions_2d).detach().numpy()
    #    normal_f = torch.squeeze(normal_fractions_2d).detach().numpy()
    #
    #    ax.hist2d(tumor_f, normal_f, bins=(100, 100), range=[[0, 1], [0, 1]], density=True, cmap=plt.cm.jet)
