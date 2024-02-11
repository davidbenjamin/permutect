import math

import torch
from torch import nn


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
        gaussian_3d = torch.randn(batch_size, self.num_samples, 2)
        correlated_gaussian_3d = self.W.forward(gaussian_3d)
        tumor_fractions_2d = torch.sigmoid(correlated_gaussian_3d[:, :, 0])
        normal_fractions_2d = torch.sigmoid(correlated_gaussian_3d[:, :, 1])

        log_likelihoods_2d = torch.reshape(tumor_alt_1d, (batch_size, 1)) * torch.log(tumor_fractions_2d) \
            + torch.reshape(tumor_ref_1d, (batch_size, 1)) * torch.log(1 - tumor_fractions_2d) \
            + torch.reshape(normal_alt_1d, (batch_size, 1)) * torch.log(normal_fractions_2d) \
            + torch.reshape(normal_ref_1d, (batch_size, 1)) * torch.log(1 - normal_fractions_2d)

        if torch.rand(1) < 0.001:
            print("debug once every 1000 batches or so. . . ")
            print("average tumor f: " + str(torch.mean(tumor_fractions_2d)))
            print("average normal f: " + str(torch.mean(normal_fractions_2d)))
            print("min tumor f: " + str(torch.min(tumor_fractions_2d)))
            print("min normal f: " + str(torch.min(normal_fractions_2d)))
            print("weights are " + str(self.W.weight))
            print("bias is " + str(self.W.bias))

        # DEBUG, DELETE LATER
        if log_likelihoods_2d.isnan().any():
            offending_index = log_likelihoods_2d.isnan().nonzero()[0]
            offending_row = offending_index[0].item()
            print("normal artifact likelihoods contain a nan")
            print("offending indices: " + str(log_likelihoods_2d.isnan().nonzero()))
            print("tumor fractions sampled: " + str(tumor_fractions_2d[offending_row]))
            print("normal fractions sampled: " + str(normal_fractions_2d[offending_row]))
            print("log tumor, log 1 - tumor, log normal, log 1 - normal:")
            print(torch.log(tumor_fractions_2d[offending_row]))
            print(torch.log(1 - tumor_fractions_2d[offending_row]))
            print(torch.log(normal_fractions_2d[offending_row]))
            print(torch.log(1 - normal_fractions_2d[offending_row]))
            assert 5 < 4, "CRASH!!! normal artifact spectrum yields nan in forward pass"

        # average over sample dimension
        log_likelihoods_1d = torch.logsumexp(log_likelihoods_2d, dim=1) - math.log(self.num_samples)

        # zero likelihood if no alt in normal
        no_alt_in_normal_mask = normal_alt_1d < 1
        return -9999 * no_alt_in_normal_mask + log_likelihoods_1d * torch.logical_not(no_alt_in_normal_mask)
