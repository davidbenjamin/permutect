import math

import torch
from permutect import misc_utils
from torch import nn, IntTensor

from permutect.metrics.plotting import simple_plot
from permutect.misc_utils import backpropagate
from permutect.utils.stats_utils import beta_binomial_log_lk, uniform_binomial_log_lk
from permutect.utils.enums import Variation


class ArtifactSpectra(nn.Module):
    """
    Mixture of uniform-binomial compound distributions for each variant type
    """

    def __init__(self, num_components: int):
        super(ArtifactSpectra, self).__init__()
        self.K = num_components
        self.V = len(Variation)

        self.weights_pre_softmax_vk = torch.nn.Parameter(torch.ones(self.V, self.K))

        # the minima of the uniform-binomials, before before sent through a sigmoid
        # initialize to be essentially zero (after the sigmoid)
        initial_minima_pre_sigmoid_vk = -10*torch.ones(self.V, self.K)
        self.min_pre_sigmoid_vk = torch.nn.Parameter(initial_minima_pre_sigmoid_vk)

        # to keep the maxima both greater than the minima and less than one, we say
        # maxima = sigmoid(minima_pre_sigmoid + lengths_in_logit_space)
        # initialize to
        initial_maxima_pre_sigmoid_vk = -3 * torch.ones(self.V, self.K) + torch.rand(self.V, self.K)
        self.lengths_in_logit_space_pre_exp_vk = torch.nn.Parameter(torch.log(initial_maxima_pre_sigmoid_vk - initial_minima_pre_sigmoid_vk))

    def get_minima_and_maxima_vk(self):
        minima_vk = torch.sigmoid(self.min_pre_sigmoid_vk)
        maxima_vk = torch.sigmoid(self.min_pre_sigmoid_vk + torch.exp(self.lengths_in_logit_space_pre_exp_vk))
        return minima_vk, maxima_vk

    '''
    here x is a 2D tensor, 1st dimension batch, 2nd dimension being features that determine which Beta mixture to use
    n and k are 1D tensors, the only dimension being batch.
    '''
    def forward(self, variant_types_b: torch.IntTensor, depths_b, alt_counts_b):
        var_types_b = variant_types_b.long()

        depths_bk, alt_counts_bk = depths_b.view(-1, 1), alt_counts_b.view(-1, 1)
        minima_vk, maxima_vk = self.get_minima_and_maxima_vk()
        minima_bk, maxima_bk = minima_vk[var_types_b], maxima_vk[var_types_b]
        log_lks_bk = uniform_binomial_log_lk(n=depths_bk, k=alt_counts_bk, x1=minima_bk, x2=maxima_bk)

        log_weights_vk = torch.log_softmax(self.weights_pre_softmax_vk, dim=-1)  # softmax over component dimension
        log_weights_bk = log_weights_vk[var_types_b]
        weighted_log_lks_bk = log_lks_bk + log_weights_bk
        result_b = torch.logsumexp(weighted_log_lks_bk, dim=-1)
        return result_b


    # TODO: utter code duplication with somatic spectrum
    def fit(self, num_epochs, types_b: IntTensor, depths_1d_tensor, alt_counts_1d_tensor, batch_size=64):
        optimizer = torch.optim.Adam(self.parameters())
        num_batches = math.ceil(len(alt_counts_1d_tensor) / batch_size)

        for epoch in range(num_epochs):
            for batch in range(num_batches):
                batch_start = batch * batch_size
                batch_end = min(batch_start + batch_size, len(alt_counts_1d_tensor))
                batch_slice = slice(batch_start, batch_end)
                loss = -torch.mean(self.forward(types_b[batch_slice], depths_1d_tensor[batch_slice], alt_counts_1d_tensor[batch_slice]))
                backpropagate(optimizer, loss)

    '''
    get raw data for a spectrum plot of probability density vs allele fraction for a particular variant type
    '''
    def spectrum_density_vs_fraction(self, variant_type: Variation, depth: int):
        fractions_f = torch.arange(0.01, 0.99, 0.001)  # 1D tensor
        weights_k = torch.softmax(self.weights_pre_softmax_vk[variant_type], dim=-1).cpu()
        minima_vk, maxima_vk = self.get_minima_and_maxima_vk()
        minima_k, maxima_k = minima_vk[variant_type].cpu(), maxima_vk[variant_type].cpu()

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
