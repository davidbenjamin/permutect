import math

import torch
from torch import nn, log, logsumexp
from torch.nn import Parameter
from torch.nn.functional import log_softmax

from permutect.metrics.plotting import simple_plot
from permutect.misc_utils import backpropagate
from permutect.utils.math_utils import add_in_log_space
from permutect.utils.stats_utils import beta_binomial_log_lk, uniform_binomial_log_lk

# exclude obvious germline, artifact, sequencing error etc from M step for speed
MIN_POSTERIOR_FOR_M_STEP = 0.2


class SomaticSpectrum(nn.Module):
    """
    This model takes in 1D tensor (batch size, ) alt counts and depths and computes the log likelihoods
    log P(alt count | depth, spectrum parameters).

    The probability P(alt count | depth) is a K+1-component mixture model where K components are uniform-binomial compound
    distributions (i.e. alt counts binomially distributed where the binomial probability p is drawn from a uniform
    distribution).

    the kth cluster's uniform distribution is c_k * Uniform[minor allele fraction, 1 - minor allele fraction, where the
    cell fraction c_k is a model parameter and the minor allele fraction depends on the variant's location in the genome.
    Thus the resulting mixture distribution is different depending on the location.

    The parameters are cell fractions c_0, c_1. . . c_(K-1) and cluster weights w_1, w_2. . .w_(K-1)

    The k = K background cluster DOES NOT have learned parameters because it represents not a biological property but rather
    a fudge factor for small CNVs we may have missed.  It's cluster likelihood is a broad beta binomial.


    We compute the uniform-binomial and beta binomial log likelihoods, then add in log space via logsumexp to get the overall
    mixture log likelihood.
    """

    def __init__(self, num_components: int):
        super(SomaticSpectrum, self).__init__()
        self.K = num_components

        # initialize evenly spaced cell fractions pre-sigmoid from -3 to 3
        self.cf_pre_sigmoid_k = Parameter((6 * ((torch.arange(num_components) / num_components) - 0.5)))

        # rough idea for initializing weights: the bigger the cell fraction 1) the more cells there are for mutations to arise
        # and 2) the longer the cluster has probably been around for mutations to arise
        # thus we initialize weights proportional to the square of the cell fraction
        # TODO: maybe this should just be linear instead of quadratic
        squared_cfs = torch.log(torch.square(torch.sigmoid(self.cf_pre_sigmoid_k.detach())))
        self.weights_pre_softmax_k = Parameter(squared_cfs)

        # TODO: this is an arbitrary guess
        background_weight = 0.0001
        self.log_background_weight = Parameter(log(torch.tensor(background_weight)), requires_grad=False)
        self.log_non_background_weight = Parameter(log(torch.tensor(1 - background_weight)), requires_grad=False)

        self.background_alpha = Parameter(torch.tensor([1]), requires_grad=False)
        self.background_beta = Parameter(torch.tensor([1]), requires_grad=False)

    '''
    here alt counts, depths, and minor allele fractions are 1D (batch size, ) tensors
    '''
    def forward(self, depths_b, alt_counts_b, mafs_b):
        # give batch tensors dummy length-1 k index for broadcasting
        alt_counts_bk = alt_counts_b.view(-1, 1)
        depths_bk = depths_b.view(-1, 1)
        mafs_bk = mafs_b.view(-1, 1)

        # lower and upper uniform distribution bounds
        cf_k = torch.sigmoid(self.cf_pre_sigmoid_k)
        cf_bk = cf_k.view(1, -1)         # dummy length-1 b index for broadcasting

        x1_bk, x2_bk = mafs_bk * cf_bk, (1-mafs_bk)*cf_bk
        uniform_binomial_log_lks_bk = uniform_binomial_log_lk(n=depths_bk, k=alt_counts_bk, x1=x1_bk, x2=x2_bk)

        log_weights_k = log_softmax(self.weights_pre_softmax_k, dim=-1)
        log_weights_bk = log_weights_k.view(1, -1)

        non_background_log_lks_b = logsumexp(log_weights_bk + uniform_binomial_log_lks_bk, dim=-1)
        background_log_lks_b = beta_binomial_log_lk(n=depths_b, k=alt_counts_b, alpha=self.background_alpha, beta=self.background_beta)

        result_b = add_in_log_space(self.log_non_background_weight + non_background_log_lks_b,
                                    self.log_background_weight + background_log_lks_b)
        return result_b


    '''
    here alt counts and depths are 1D (batch size, ) tensors
    '''
    """
    def weighted_likelihoods_by_cluster(self, depths_b, alt_counts_b):
        batch_size = len(alt_counts_b)

        f_k = torch.sigmoid(self.f_pre_sigmoid_k)
        f_bk = f_k.expand(batch_size, -1)
        alt_counts_bk = torch.unsqueeze(alt_counts_b, dim=1).expand(-1, self.K - 1)
        depths_bk = torch.unsqueeze(depths_b, dim=1).expand(-1, self.K - 1)
        binomial_likelihoods_bk = binomial_log_lk(depths_bk, alt_counts_bk, f_bk)

        alpha = torch.exp(self.alpha_pre_exp)
        beta = torch.exp(self.beta_pre_exp)
        alpha_b = alpha.expand(batch_size)
        beta_b = beta.expand(batch_size)

        beta_binomial_likelihoods_b = beta_binomial_log_lk(depths_b, alt_counts_b, alpha_b, beta_b)
        beta_binomial_likelihoods_bk = torch.unsqueeze(beta_binomial_likelihoods_b, dim=1)

        likelihoods_bk = torch.hstack((binomial_likelihoods_bk, beta_binomial_likelihoods_bk))

        log_weights_k = log_softmax(self.weights_pre_softmax_k, dim=-1)  # these weights are normalized
        log_weights_bk = log_weights_k.expand(batch_size, -1)
        weighted_likelihoods_bk = log_weights_bk + likelihoods_bk

        return weighted_likelihoods_bk
    """

    """
    # posteriors: responsibilities that each object is somatic
    def update_m_step(self, posteriors_n, alt_counts_n, depths_n):
        possible_somatic_indices = posteriors_n > MIN_POSTERIOR_FOR_M_STEP
        somatic_posteriors_n = posteriors_n[possible_somatic_indices]
        somatic_alt_counts_n = alt_counts_n[possible_somatic_indices]
        somatic_depths_n = depths_n[possible_somatic_indices]

        # TODO: make sure this all fits on GPU
        # TODO: maybe split it up into batches?
        weighted_likelihoods_nk = self.weighted_likelihoods_by_cluster(somatic_depths_n, somatic_alt_counts_n)
        cluster_posteriors_nk = somatic_posteriors_n[:, None] * torch.softmax(weighted_likelihoods_nk, dim=-1)
        cluster_totals_k = torch.sum(cluster_posteriors_nk, dim=0)

        with torch.no_grad():
            self.weights_pre_softmax_k.copy_(torch.log(cluster_totals_k + 0.00001))

            # update the binomial clusters -- we exclude the last cluster, which is beta binomial
            for k in range(self.K - 1):
                weights = cluster_posteriors_nk[:, k]
                f = torch.sum((weights * somatic_alt_counts_n)) / torch.sum((0.00001 + weights * somatic_depths_n))

                self.f_pre_sigmoid_k[k] = torch.log(f / (1-f))
    """

    def fit(self, num_epochs, depths_b, alt_counts_b, mafs_b, batch_size=64):
        optimizer = torch.optim.Adam(self.parameters())
        num_batches = math.ceil(len(alt_counts_b) / batch_size)

        for epoch in range(num_epochs):
            for batch in range(num_batches):
                batch_start = batch * batch_size
                batch_end = min(batch_start + batch_size, len(alt_counts_b))
                batch_slice = slice(batch_start, batch_end)
                loss = -torch.mean(self.forward(depths_b[batch_slice], alt_counts_b[batch_slice], mafs_b[batch_slice]))
                backpropagate(optimizer, loss)

    '''
    get raw data for a spectrum plot of probability density vs allele fraction
    '''
    def spectrum_density_vs_fraction(self):
        fractions_f = torch.arange(0.01, 0.99, 0.001)  # 1D tensor

        cf_k = torch.sigmoid(self.cf_pre_sigmoid_k).cpu()

        # smear each binomial f into a narrow Gaussian for plotting
        gauss_k = torch.distributions.normal.Normal(cf_k, 0.01 * torch.ones_like(cf_k))
        log_densities_fk = gauss_k.log_prob(fractions_f.unsqueeze(dim=1))

        log_weights_k = log_softmax(self.weights_pre_softmax_k, dim=-1).cpu()  # these weights are normalized
        log_weights_fk = log_weights_k.view(1, -1)

        log_weighted_densities_fk = log_weights_fk + log_densities_fk
        densities_f = torch.exp(torch.logsumexp(log_weighted_densities_fk, dim=-1))

        return fractions_f, densities_f

    def plot_spectrum(self, title):
        fractions, densities = self.spectrum_density_vs_fraction()
        return simple_plot([(fractions.numpy(), densities.numpy(), " ")], "AF", "density", title)