import math

import torch
from torch import log
from torch import logsumexp
from torch import nn
from torch.nn import Parameter
from torch.nn.parameter import Buffer
from torch.nn.utils import parametrize

from permutect.architecture.parameterizations import BoundedNumber
from permutect.architecture.parameterizations import LogWeights
from permutect.metrics.plotting import simple_plot
from permutect.misc_utils import backpropagate
from permutect.utils.math_utils import add_in_log_space
from permutect.utils.stats_utils import beta_binomial_log_lk
from permutect.utils.stats_utils import uniform_binomial_log_lk

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
        self.cf_k = Parameter(torch.sigmoid((6 * ((torch.arange(num_components) / num_components) - 0.5))))
        parametrize.register_parametrization(self, "cf_k", BoundedNumber(0, 1))

        # rough idea for initializing weights: the bigger the cell fraction 1) the more cells there are for mutations to arise
        # and 2) the longer the cluster has probably been around for mutations to arise
        # thus we initialize weights proportional to the square of the cell fraction
        # TODO: maybe this should just be linear instead of quadratic
        self.log_weights_k = Parameter(torch.log(torch.square(self.cf_k.detach())))
        parametrize.register_parametrization(self, "log_weights_k", LogWeights())

        # TODO: this is an arbitrary guess
        background_weight = 0.0001
        self.log_background_weight = Buffer(log(torch.tensor(background_weight)))
        self.log_non_background_weight = Buffer(log(torch.tensor(1 - background_weight)))

        self.background_alpha = Buffer(torch.tensor([1]))
        self.background_beta = Buffer(torch.tensor([1]))

    """
    here alt counts, depths, and minor allele fractions are 1D (batch size, ) tensors
    """

    def forward(self, depths_b, alt_counts_b, mafs_b):
        # give batch tensors dummy length-1 k index for broadcasting
        alt_counts_bk = alt_counts_b.view(-1, 1)
        depths_bk = depths_b.view(-1, 1)
        # we can't have maf = 0.5 exactly because then x1 = x2 and we get NaN
        mafs_bk = torch.clamp(mafs_b, max=0.49).view(-1, 1)

        # lower and upper uniform distribution bounds
        cf_bk = self.cf_k.view(1, -1)  # dummy length-1 b index for broadcasting

        x1_bk, x2_bk = mafs_bk * cf_bk, (1 - mafs_bk) * cf_bk
        uniform_binomial_log_lks_bk = uniform_binomial_log_lk(n=depths_bk, k=alt_counts_bk, x1=x1_bk, x2=x2_bk)

        log_weights_bk = self.log_weights_k.view(1, -1)

        non_background_log_lks_b = logsumexp(log_weights_bk + uniform_binomial_log_lks_bk, dim=-1)
        background_log_lks_b = beta_binomial_log_lk(
            n=depths_b, k=alt_counts_b, alpha=self.background_alpha, beta=self.background_beta
        )

        result_b = add_in_log_space(
            self.log_non_background_weight + non_background_log_lks_b,
            self.log_background_weight + background_log_lks_b,
        )
        return result_b

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

    """
    get raw data for a spectrum plot of probability density vs allele fraction
    """

    def spectrum_density_vs_fraction(self):
        fractions_f = torch.arange(0.01, 0.99, 0.001)  # 1D tensor

        # smear each binomial f into a narrow Gaussian for plotting
        cf_k_cpu = self.cf_k.cpu()
        gauss_k = torch.distributions.normal.Normal(cf_k_cpu, 0.01 * torch.ones_like(cf_k_cpu))
        log_densities_fk = gauss_k.log_prob(fractions_f.unsqueeze(dim=1))

        log_weights_fk = self.log_weights_k.view(1, -1).cpu()
        log_weighted_densities_fk = log_weights_fk + log_densities_fk
        densities_f = torch.exp(torch.logsumexp(log_weighted_densities_fk, dim=-1))

        return fractions_f, densities_f

    def plot_spectrum(self, title):
        fractions, densities = self.spectrum_density_vs_fraction()
        return simple_plot([(fractions.numpy(), densities.numpy(), " ")], "AF", "density", title)
