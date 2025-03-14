import math

import torch
from torch import nn, IntTensor

from permutect.metrics.plotting import simple_plot
from permutect.misc_utils import backpropagate
from permutect.utils.stats_utils import uniform_binomial_log_lk
from permutect.utils.enums import Variation


DEPTH_CUTOFFS = [10, 20]
NUM_DEPTH_BINS = len(DEPTH_CUTOFFS) + 1


# the bin is the number of cutoffs that are met or exceeded
def depths_to_depth_bins(depths_b: IntTensor):
    assert len(DEPTH_CUTOFFS) == 2
    return (depths_b >= DEPTH_CUTOFFS[0]).long() + (depths_b >= DEPTH_CUTOFFS[1]).long()


class ArtifactSpectra(nn.Module):
    """
    Mixture of uniform-binomial compound distributions for each variant type
    """

    def __init__(self, num_components: int):
        super(ArtifactSpectra, self).__init__()
        self.K = num_components
        self.V = len(Variation)
        self.D = NUM_DEPTH_BINS

        self.weights_pre_softmax_dvk = torch.nn.Parameter(torch.ones(self.D, self.V, self.K))

        # the minima of the uniform-binomials, before before sent through a sigmoid
        # initialize to be essentially zero (after the sigmoid)
        initial_minima_pre_sigmoid_dvk = -10*torch.ones(self.D, self.V, self.K)
        self.min_pre_sigmoid_dvk = torch.nn.Parameter(initial_minima_pre_sigmoid_dvk)

        # to keep the maxima both greater than the minima and less than one, we say
        # maxima = sigmoid(minima_pre_sigmoid + lengths_in_logit_space)
        # initialize to
        initial_maxima_pre_sigmoid_dvk = -3 * torch.ones(self.D, self.V, self.K) + torch.rand(self.D, self.V, self.K)
        self.lengths_in_logit_space_pre_exp_dvk = torch.nn.Parameter(torch.log(initial_maxima_pre_sigmoid_dvk - initial_minima_pre_sigmoid_dvk))

    def get_minima_and_maxima_dvk(self):
        minima_dvk = torch.sigmoid(self.min_pre_sigmoid_dvk)
        maxima_dvk = torch.sigmoid(self.min_pre_sigmoid_dvk + torch.exp(self.lengths_in_logit_space_pre_exp_dvk))
        return minima_dvk, maxima_dvk

    '''
    here x is a 2D tensor, 1st dimension batch, 2nd dimension being features that determine which Beta mixture to use
    n and k are 1D tensors, the only dimension being batch.
    '''
    def forward(self, variant_types_b: IntTensor, depths_b: IntTensor, alt_counts_b):
        var_types_b = variant_types_b.long()
        depth_bins_b = depths_to_depth_bins(depths_b)

        minima_dvk, maxima_dvk = self.get_minima_and_maxima_dvk()
        # to select the appropriate depth bins 'd' and variant types 'v' over the batch, we flatten d and v and select
        # from the appropriate flattened indices
        flattened_dv = depth_bins_b * self.V + var_types_b
        minima_bk, maxima_bk = minima_dvk.view(-1, self.K)[flattened_dv], maxima_dvk.view(-1, self.K)[flattened_dv]

        depths_bk, alt_counts_bk = depths_b.view(-1, 1), alt_counts_b.view(-1, 1)

        log_lks_bk = uniform_binomial_log_lk(n=depths_bk, k=alt_counts_bk, x1=minima_bk, x2=maxima_bk)

        log_weights_dvk = torch.log_softmax(self.weights_pre_softmax_dvk, dim=-1)  # softmax over component dimension
        log_weights_bk = log_weights_dvk.view(-1, self.K)[flattened_dv]
        weighted_log_lks_bk = log_lks_bk + log_weights_bk
        result_b = torch.logsumexp(weighted_log_lks_bk, dim=-1)
        return result_b

    # TODO: utter code duplication with somatic spectrum
    def fit(self, num_epochs: int, types_b: IntTensor, depths_b: IntTensor, alt_counts_b: IntTensor, batch_size: int=64):
        optimizer = torch.optim.Adam(self.parameters())
        num_batches = math.ceil(len(alt_counts_b) / batch_size)

        for epoch in range(num_epochs):
            for batch in range(num_batches):
                batch_start = batch * batch_size
                batch_end = min(batch_start + batch_size, len(alt_counts_b))
                batch_slice = slice(batch_start, batch_end)
                loss = -torch.mean(self.forward(types_b[batch_slice], depths_b[batch_slice], alt_counts_b[batch_slice]))
                backpropagate(optimizer, loss)

    '''
    get raw data for a spectrum plot of probability density vs allele fraction for a particular variant type
    '''
    def spectrum_density_vs_fraction(self, variant_type: Variation, depth: int):
        depth_bin = depths_to_depth_bins(torch.tensor(depth)).item()
        fractions_f = torch.arange(0.01, 0.99, 0.001)  # 1D tensor
        weights_k = torch.softmax(self.weights_pre_softmax_dvk[depth_bin, variant_type], dim=-1).cpu()
        minima_dvk, maxima_dvk = self.get_minima_and_maxima_dvk()
        minima_k, maxima_k = minima_dvk[depth_bin, variant_type].cpu(), maxima_dvk[depth_bin, variant_type].cpu()

        densities_f = torch.zeros_like(fractions_f)
        for k in range(self.K):
            densities_f += weights_k[k] * (fractions_f < maxima_k[k]) * (fractions_f > minima_k[k])
        return fractions_f, densities_f

    '''
    here x is a 1D tensor, a single datum/row of the 2D tensors as above
    '''
    def plot_spectrum(self, variant_type: Variation, title, depth: int):
        fractions, densities = self.spectrum_density_vs_fraction(variant_type, depth)
        return simple_plot([(fractions.numpy(), densities.numpy(), " ")], "AF", "density", title)
