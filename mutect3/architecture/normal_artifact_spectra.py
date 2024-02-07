import math

import torch
from mutect3 import utils
from torch import nn, exp, unsqueeze, logsumexp
from torch.nn.functional import softmax, log_softmax

from mutect3.metrics.plotting import simple_plot
from mutect3.utils import beta_binomial


class NormalArtifactSpectra(nn.Module):
    def __init__(self, input_size: int, num_samples: int):
        super(NormalArtifactSpectra, self).__init__()

        self.num_samples = num_samples

        self.W = nn.Linear(in_features=2, out_features=2)

    def forward(self, tumor_alt_1d: torch.Tensor, tumor_ref_1d: torch.Tensor, normal_alt_1d: torch.Tensor, normal_ref_1d: torch.Tensor):
        batch_size = len(tumor_alt_1d)
        gaussian_3d = torch.randn(batch_size, self.num_samples, 2)
        correlated_gaussian_3d = self.W.forward(gaussian_3d)
        tumor_fractions_2d = torch.sigmoid(correlated_gaussian_3d[:, :, 0])
        normal_fractions_2d = torch.sigmoid(correlated_gaussian_3d[:, :, 1])

        log_likelihoods_2d = torch.reshape(tumor_alt_1d, (batch_size, 1)) * torch.log(tumor_fractions_2d) \
            + torch.reshape(tumor_ref_1d, (batch_size, 1)) * torch.log(1 - tumor_fractions_2d) \
            + torch.reshape(normal_alt_1d, (batch_size, 1)) * torch.log(normal_fractions_2d) \
            + torch.reshape(normal_ref_1d, (batch_size, 1)) * torch.log(1 - normal_fractions_2d)