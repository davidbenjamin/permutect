import math

import torch
from permutect import utils
from torch import nn

from permutect.metrics.plotting import simple_plot
from permutect.utils import beta_binomial, Variation


class ArtifactSpectra(nn.Module):
    """
    This model takes in 1D tensors (batch size, ) of alt counts and depths and a 2D tensor (batch size, len(Variation))
    of one-hot-encoded variant types and computes the log likelihoods
    log P(alt count | depth, variant type, spectrum parameters).

    The probability P(alt count | depth) is a K-component beta binomial mixture model where each component is a beta binomial
    P_k(a|d) = integral{Beta(f|alpha, beta) * Binom(a|d, f) df}.

    This integral is exact and is implemented in utils.beta_binomial()

    Importantly, the beta shape parameter is *not* learnable.  We fix it at a high constant value in order to force the spectrum
    to fall off very rapidly after its peak allele fraction.  Otherwise, the unrealistically long tail gives artificially
    high likelihoods for artifacts at high allele fractions.
    """

    def __init__(self, num_components: int):
        super(ArtifactSpectra, self).__init__()
        self.beta = 100 # not a learnable parameter!
        self.K = num_components
        self.V = len(Variation)

        self.weights0_pre_softmax_vk = torch.nn.Parameter(torch.ones(self.V, self.K))

        # for each component and variant type:
        # weight_pre_softmax = weight0_pre_softmax + gamma * sigmoid(depth * kappa)
        self.gamma_vk = torch.nn.Parameter(0.1 * torch.rand(self.V, self.K))
        self.kappa_vk = torch.nn.Parameter(0.02 * torch.ones(self.V, self.K))

        # initialize evenly spaced alphas from 1 to 7 for each variant type
        self.alpha0_pre_exp_vk = torch.nn.Parameter(torch.log(1 + 7 * (torch.arange(self.K) / self.K)).repeat(self.V, 1))

        self.eta_pre_exp_vk = torch.nn.Parameter(torch.ones(self.V, self.K))

        self.delta_pre_exp_vk = torch.nn.Parameter(torch.log(torch.ones(self.V, self.K)/50))

        # for each component and variant type:
        # alpha = exp(alpha0_pre_exp - exp(eta_pre_exp)*sigmoid(depth * exp(delta_pre_exp)))



    '''
    here x is a 2D tensor, 1st dimension batch, 2nd dimension being features that determine which Beta mixture to use
    n and k are 1D tensors, the only dimension being batch.
    '''
    def forward(self, types_one_hot_bv, depths_b, alt_counts_b):
        alt_counts_bk = torch.unsqueeze(alt_counts_b, dim=1).expand(-1, self.K - 1)
        depths_bk = torch.unsqueeze(depths_b, dim=1).expand(-1, self.K - 1)
        depths_bvk = depths_bk[:, None, :]

        eta_vk = torch.exp(self.eta_pre_exp_vk)
        delta_vk = torch.exp(self.delta_pre_exp_vk)
        alpha0_pre_exp_bvk, eta_bvk, delta_bvk = self.alpha0_pre_exp_vk[None, :, :], eta_vk[None, :, :], delta_vk[None, :, :]
        weights0_pre_softmax_bvk, gamma_bvk, kappa_bvk = self.weights0_pre_softmax_vk[None, :, :], self.gamma_vk[None, :, :], self.kappa_vk[None, :, :]

        alpha_bvk = torch.exp(alpha0_pre_exp_bvk - eta_bvk * torch.sigmoid(depths_bvk * delta_bvk))

        types_one_hot_bvk = torch.unsqueeze(types_one_hot_bv, dim=-1)   # gives it broadcastable length-1 component dimension
        alpha_bk = torch.sum(types_one_hot_bvk * alpha_bvk, dim=1)  # due to one-hotness only one v contributes to the sum
        beta_bk = self.beta * torch.ones_like(alpha_bk)

        if alpha_bk.isnan().any():
            print("NaN found in alpha_bk")
            assert 1 < 0, "FAIL"

        beta_binomial_likelihoods_bk = beta_binomial(depths_bk, alt_counts_bk, alpha_bk, beta_bk)

        weights_pre_softmax_bvk = weights0_pre_softmax_bvk + gamma_bvk * torch.sigmoid(depths_bvk * kappa_bvk)

        log_weights_bvk = torch.log_softmax(weights_pre_softmax_bvk, dim=-1)    # softmax over component dimension
        log_weights_bk = torch.sum(types_one_hot_bvk * log_weights_bvk, dim=1)  # same idea as above

        weighted_likelihoods_bk = log_weights_bk + beta_binomial_likelihoods_bk

        result_b = torch.logsumexp(weighted_likelihoods_bk, dim=1, keepdim=False)
        return result_b

    # TODO: utter code duplication with somatic spectrum
    def fit(self, num_epochs, types_one_hot_2d, depths_1d_tensor, alt_counts_1d_tensor, batch_size=64):
        optimizer = torch.optim.Adam(self.parameters())
        num_batches = math.ceil(len(alt_counts_1d_tensor) / batch_size)

        for epoch in range(num_epochs):
            for batch in range(num_batches):
                batch_start = batch * batch_size
                batch_end = min(batch_start + batch_size, len(alt_counts_1d_tensor))
                batch_slice = slice(batch_start, batch_end)
                loss = -torch.mean(self.forward(types_one_hot_2d[batch_slice], depths_1d_tensor[batch_slice], alt_counts_1d_tensor[batch_slice]))
                utils.backpropagate(optimizer, loss)

    '''
    get raw data for a spectrum plot of probability density vs allele fraction for a particular variant type
    '''
    def spectrum_density_vs_fraction(self, variant_type: Variation, depth: int):
        fractions_f = torch.arange(0.01, 0.99, 0.001)  # 1D tensor

        weights0_pre_softmax_k = self.weights0_pre_softmax_vk[variant_type]
        gamma_k = self.gamma_vk[variant_type]
        kappa_k = self.kappa_vk[variant_type]
        alpha0_pre_exp_k = self.alpha0_pre_exp_vk[variant_type]
        eta_k = torch.exp(self.eta_pre_exp_vk[variant_type])
        delta_k = torch.exp(self.delta_pre_exp_vk[variant_type])

        alpha_k = torch.exp(alpha0_pre_exp_k - eta_k * torch.sigmoid(depth * delta_k))
        weights_pre_softmax_k = weights0_pre_softmax_k + gamma_k * torch.sigmoid(depth * kappa_k)

        log_weights_k = torch.log_softmax(weights_pre_softmax_k, dim=0)
        beta_k = self.beta * torch.ones_like(alpha_k)

        # distribution done on CPU
        dist = torch.distributions.beta.Beta(alpha_k.cpu(), beta_k.cpu())
        log_densities_fk = dist.log_prob(fractions_f.unsqueeze(dim=-1))
        log_weights_fk = log_weights_k.unsqueeze(dim=0).cpu()

        log_weighted_densities_fk = log_weights_fk + log_densities_fk
        densities_f = torch.exp(torch.logsumexp(log_weighted_densities_fk, dim=1, keepdim=False))

        return fractions_f, densities_f

    '''
    here x is a 1D tensor, a single datum/row of the 2D tensors as above
    '''
    def plot_spectrum(self, variant_type: Variation, title, depth: int):
        fractions, densities = self.spectrum_density_vs_fraction(variant_type, depth)
        return simple_plot([(fractions.numpy(), densities.numpy(), " ")], "AF", "density", title)
