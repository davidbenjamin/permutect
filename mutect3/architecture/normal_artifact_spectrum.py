import math

import torch
from torch import nn

import matplotlib.pyplot as plt
import numpy as np

EPSILON = 0.001


class NormalArtifactSpectrum(nn.Module):
    def __init__(self, num_samples: int):
        super(NormalArtifactSpectrum, self).__init__()

        self.num_samples = num_samples

        self.W = nn.Linear(in_features=2, out_features=2)

    def forward(self, tumor_alt_1d: torch.Tensor, tumor_ref_1d: torch.Tensor, normal_alt_1d: torch.Tensor, normal_ref_1d: torch.Tensor):
        if torch.sum(normal_alt_1d) < 1:    # shortcut if no normal alts in the whole batch
            print("debug, no normal alts in batch")
            return -9999 * torch.ones_like(tumor_alt_1d)
        batch_size = len(tumor_alt_1d)
        tumor_fractions_2d, normal_fractions_2d = self.get_tumor_and_normal_fraction(batch_size, self.num_samples)

        log_likelihoods_2d = torch.reshape(tumor_alt_1d, (batch_size, 1)) * torch.log(tumor_fractions_2d) \
            + torch.reshape(tumor_ref_1d, (batch_size, 1)) * torch.log(1 - tumor_fractions_2d) \
            + torch.reshape(normal_alt_1d, (batch_size, 1)) * torch.log(normal_fractions_2d) \
            + torch.reshape(normal_ref_1d, (batch_size, 1)) * torch.log(1 - normal_fractions_2d)

        # average over sample dimension
        log_likelihoods_1d = torch.logsumexp(log_likelihoods_2d, dim=1) - math.log(self.num_samples)

        # zero likelihood if no alt in normal
        no_alt_in_normal_mask = normal_alt_1d < 1
        return -9999 * no_alt_in_normal_mask + log_likelihoods_1d * torch.logical_not(no_alt_in_normal_mask)

    def get_tumor_and_normal_fraction(self, batch_size, num_samples):
        gaussian_3d = torch.randn(batch_size, num_samples, 2)
        correlated_gaussian_3d = self.W.forward(gaussian_3d)
        # to prevent nans, map onto [EPSILON, 1 - EPSILON]
        tumor_fractions_2d = EPSILON + (1 - 2 * EPSILON) * torch.sigmoid(correlated_gaussian_3d[:, :, 0])
        normal_fractions_2d = EPSILON + (1 - 2 * EPSILON) * torch.sigmoid(correlated_gaussian_3d[:, :, 1])
        return tumor_fractions_2d, normal_fractions_2d

    # TODO: move this method to plotting
    def density_plot_on_axis(self, ax):
        tumor_fractions_2d, normal_fractions_2d = self.get_tumor_and_normal_fraction(batch_size=1, num_samples=100000)
        tumor_f = torch.squeeze(tumor_fractions_2d).numpy()
        normal_f = torch.squeeze(normal_fractions_2d).numpy()

        ax.hist2d(tumor_f, normal_f, bins=(100, 100), range=[[0, 1], [0, 1]], density=True, cmap=plt.cm.jet)
