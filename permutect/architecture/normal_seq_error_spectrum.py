import math

import torch
from torch import nn, Tensor, IntTensor

from permutect.utils.enums import Variation
from permutect.utils.stats_utils import uniform_binomial_log_lk

EPSILON = 0.0001


# we can't use a beta binomial for normal seq error because betas have such long tails that even if we constrain the mean
# to be small there is too large a probability of a large allele fraction.  Instead, we use a uniform binomial.
class NormalSeqErrorSpectrum(nn.Module):
    def __init__(self, max_mean: float):
        super(NormalSeqErrorSpectrum, self).__init__()
        self.max_mean = max_mean
        self.means_pre_sigmoid_v = torch.nn.Parameter(torch.zeros(len(Variation)))

    def forward(self, ref_counts_b: Tensor, alt_counts_b: Tensor, var_types_b: IntTensor):
        x1_b = torch.zeros_like(alt_counts_b)
        x2_b = 2 * self.max_mean * torch.sigmoid(self.mean_pre_sigmoid_v[var_types_b])
        return uniform_binomial_log_lk(n=alt_counts_b + ref_counts_b, k=alt_counts_b, x1=x1_b, x2=x2_b)

