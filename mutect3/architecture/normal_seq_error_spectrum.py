import math

import torch
from torch import nn

import matplotlib.pyplot as plt
import numpy as np

EPSILON = 0.001

# the mean of a half-normal distribution is related to the standard deviation sigma of its corresponding normal distribution by
# sigma = mean * sqrt(pi/2)
SQRT_PI_OVER_2 = math.sqrt(math.pi / 2)


# we can't use a beta binomial for normal seq error because betas have such long tails that even if we constrain the mean
# to be small there is too large a probability of a large allele fraction.  Here we assume an underlying half normal distribution on
# the allele fraction ie it is a half normal-binomial.  Since these are not conjugate we have to explicitly sample and
# essentially perform a brute force Monte Carlo integral.
class NormalSeqErrorSpectrum(nn.Module):
    def __init__(self, num_samples: int, max_mean: float):
        super(NormalSeqErrorSpectrum, self).__init__()

        self.num_samples = num_samples

        self.max_mean = max_mean

        # this is 1/lambda parameter
        # TODO: magic constant initialization!!!
        self.mean_pre_sigmoid = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, alt_counts_1d: torch.Tensor, ref_counts_1d: torch.Tensor):
        print("mean pre sigmoid " + str(self.mean_pre_sigmoid))
        batch_size = len(alt_counts_1d)
        fractions_2d = self.get_fractions(batch_size, self.num_samples)

        log_likelihoods_2d = torch.reshape(alt_counts_1d, (batch_size, 1)) * torch.log(fractions_2d) \
            + torch.reshape(ref_counts_1d, (batch_size, 1)) * torch.log(1 - fractions_2d)

        # average over sample dimension
        log_likelihoods_1d = torch.logsumexp(log_likelihoods_2d, dim=1) - math.log(self.num_samples)

        if log_likelihoods_1d.isnan().any():
            print("nan in normal seq error spectrum forward")
            print("alt counts: " + str(alt_counts_1d.detach().numpy()))
            print("ref counts: " + str(ref_counts_1d.detach().numpy()))
            print("fractions: " + str(fractions_2d.detach().numpy()))
            print("min fraction: " + str(torch.min(fractions_2d)))
            print("max fraction: " + str(torch.max(fractions_2d)))
            print("mean pre sigmoid " + str(self.mean_pre_sigmoid))

            assert 5 < 4, "NAN CRASH!!!"

        return log_likelihoods_1d

    def get_fractions(self, batch_size, num_samples):
        actual_mean = torch.sigmoid(self.mean_pre_sigmoid) * self.max_mean
        actual_sigma = SQRT_PI_OVER_2 * actual_mean
        normal_samples = torch.randn(batch_size, num_samples)
        half_normal_samples = torch.abs(normal_samples)
        fractions_2d_unbounded = actual_sigma * half_normal_samples
        # apply tanh to constrain fractions to [0, 1)
        fractions_2d = torch.tanh(fractions_2d_unbounded)
        return fractions_2d

    # TODO: move this method to plotting
    def density_plot_on_axis(self, ax):
        fractions = torch.squeeze(self.get_fractions(1, 100000)).detach().numpy()
        ax.hist(fractions, bins=1000, range=[0, 1])
