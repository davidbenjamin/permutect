import math

import torch
from permutect import utils
from torch import nn
from torch.nn.functional import log_softmax

from permutect.metrics.plotting import simple_plot
from permutect.utils import beta_binomial, binomial


class SomaticSpectrum(nn.Module):
    """
    This model takes in 1D tensor (batch size, ) alt counts and depths and computes the log likelihoods
    log P(alt count | depth, spectrum parameters).

    The probability P(alt count | depth) is a K-component mixture model where K-1 components are simple binomials
    P_k(a|d) = Binom(a|d, f_k) = (d C a) f_k^a (1-f_k)^(d-a), where f_k is the allele fraction associated with component
    k and the Kth component is a background beta binomial P_K(a|d) = integral{Beta(f|alpha, beta) * Binom(a|d, f) df}.

    This integral is exact and is implemented in utils.beta_binomial()

    We compute the binomial and beta binomial log likelihoods, then add in log space via logsumexp to get the overall
    mixture log likelihood.
    """

    def __init__(self, num_components: int):
        super(SomaticSpectrum, self).__init__()
        self.K = num_components

        # initialize equal weights for each binomial component and larger weight for beta binomial background (last component)
        weights_pre_softmax = torch.ones(self.K)
        weights_pre_softmax[-1] = 3

        self.weights_pre_softmax_k = torch.nn.Parameter(weights_pre_softmax)

        # initialize evenly spaced pre-sigmoid from -2 to 2
        self.f_pre_sigmoid_k = torch.nn.Parameter((4 * torch.arange(self.K - 1) / (self.K - 1)) - 2)

        # the alpha, beta shape parameters are exponentiated in the forward pass to ensure positive values
        self.alpha_pre_exp = torch.nn.Parameter(torch.tensor(1.0))
        self.beta_pre_exp = torch.nn.Parameter(torch.tensor(1.0))

    '''
    here alt counts and depths are 1D (batch size, ) tensors
    '''
    def forward(self, depths_b, alt_counts_b):
        batch_size = len(alt_counts_b)

        f_k = torch.sigmoid(self.f_pre_sigmoid_k)
        f_bk = f_k.expand(batch_size, -1)
        alt_counts_bk = torch.unsqueeze(alt_counts_b, dim=1).expand(-1, self.K - 1)
        depths_bk = torch.unsqueeze(depths_b, dim=1).expand(-1, self.K - 1)
        binomial_likelihoods_bk = binomial(depths_bk, alt_counts_bk, f_bk)

        alpha = torch.exp(self.alpha_pre_exp)
        beta = torch.exp(self.beta_pre_exp)
        alpha_b = alpha.expand(batch_size)
        beta_b = beta.expand(batch_size)

        beta_binomial_likelihoods_b = beta_binomial(depths_b, alt_counts_b, alpha_b, beta_b)
        beta_binomial_likelihoods_bk = torch.unsqueeze(beta_binomial_likelihoods_b, dim=1)

        likelihoods_bk = torch.hstack((binomial_likelihoods_bk, beta_binomial_likelihoods_bk))

        log_weights_k = log_softmax(self.weights_pre_softmax_k)  # these weights are normalized
        log_weights_bk = log_weights_k.expand(batch_size, -1)
        weighted_likelihoods_bk = log_weights_bk + likelihoods_bk

        result_b = torch.logsumexp(weighted_likelihoods_bk, dim=1, keepdim=False)
        return result_b

    def fit(self, num_epochs, depths_1d_tensor, alt_counts_1d_tensor, batch_size=64):
        optimizer = torch.optim.Adam(self.parameters())
        num_batches = math.ceil(len(alt_counts_1d_tensor) / batch_size)

        for epoch in range(num_epochs):
            for batch in range(num_batches):
                batch_start = batch * batch_size
                batch_end = min(batch_start + batch_size, len(alt_counts_1d_tensor))
                batch_slice = slice(batch_start, batch_end)
                loss = -torch.mean(self.forward(depths_1d_tensor[batch_slice], alt_counts_1d_tensor[batch_slice]))
                utils.backpropagate(optimizer, loss)

    '''
    get raw data for a spectrum plot of probability density vs allele fraction
    '''
    def spectrum_density_vs_fraction(self):
        fractions_f = torch.arange(0.01, 0.99, 0.001)  # 1D tensor

        f_k = torch.sigmoid(self.f_pre_sigmoid_k).cpu()

        # smear each binomial f into a narrow Gaussian for plotting
        gauss_k = torch.distributions.normal.Normal(f_k, 0.01 * torch.ones_like(f_k))
        log_gauss_fk = gauss_k.log_prob(fractions_f.unsqueeze(dim=1))

        alpha = torch.exp(self.alpha_pre_exp).cpu()
        beta = torch.exp(self.beta_pre_exp).cpu()

        beta = torch.distributions.beta.Beta(alpha, beta)
        log_beta_fk = beta.log_prob(fractions_f.unsqueeze(dim=1))

        log_densities_fk = torch.hstack((log_gauss_fk, log_beta_fk))

        log_weights_k = log_softmax(self.weights_pre_softmax_k).cpu()  # these weights are normalized
        log_weights_fk = log_weights_k.expand(len(fractions_f), -1)

        log_weighted_densities_fk = log_weights_fk + log_densities_fk
        densities_f = torch.exp(torch.logsumexp(log_weighted_densities_fk, dim=1, keepdim=False))

        return fractions_f, densities_f

    def plot_spectrum(self, title):
        fractions, densities = self.spectrum_density_vs_fraction()
        return simple_plot([(fractions.numpy(), densities.numpy(), " ")], "AF", "density", title)