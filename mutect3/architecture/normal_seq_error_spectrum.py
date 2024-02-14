import math

import torch
from torch import nn

import matplotlib.pyplot as plt
import numpy as np

EPSILON = 0.001


# we can't use a beta binomial for normal seq error because betas have such long tails that even if we constrain the mean
# to be small there is too large a probability of a large allele fraction.  Here we assume an underlying exponential distribution on
# the allele fraction ie it is an exponential-binomial.  Since these are not conjugate we have to explicitly sample and
# essentially perform a brute force Monte Carlo integral.
class NormalSeqErrorSpectrum(nn.Module):
    def __init__(self, num_samples: int, max_mean: float):
        super(NormalSeqErrorSpectrum, self).__init__()

        self.num_samples = num_samples

        self.max_mean = max_mean

        # this is 1/lambda parameter
        # TODO: magic constant initialization!!!
        self.mean = torch.nn.Parameter(torch.Tensor(0.001))

    def forward(self, alt_counts_1d: torch.Tensor, ref_counts_1d: torch.Tensor):
        batch_size = len(alt_counts_1d)
        fractions_2d = self.get_fractions(batch_size, self.num_samples)

        log_likelihoods_2d = torch.reshape(alt_counts_1d, (batch_size, 1)) * torch.log(fractions_2d) \
            + torch.reshape(ref_counts_1d, (batch_size, 1)) * torch.log(1 - fractions_2d)

        # average over sample dimension
        log_likelihoods_1d = torch.logsumexp(log_likelihoods_2d, dim=1) - math.log(self.num_samples)

        return log_likelihoods_1d

    def get_fractions(self, batch_size, num_samples):
        actual_mean = self.max_mean * torch.tanh(self.mean / self.max_mean)
        uniform_samples = torch.rand(batch_size, num_samples)
        # map from U([0,1]) samples to exponential distribution samples by applying the inverse CDF
        fractions_2d = -actual_mean * torch.log(uniform_samples)
        return fractions_2d

    # TODO: move this method to plotting
    def density_plot_on_axis(self, ax):
        fractions = torch.squeeze(self.get_fractions(1, 100000)).detach().numpy()
        ax.hist(fractions, bins=1000, range=[0, 1])
