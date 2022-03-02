import torch
from torch import nn

import mutect3.metrics.plotting
from mutect3.data.read_set_batch import ReadSetBatch
from mutect3.utils import beta_binomial


class AFSpectrum(nn.Module):

    def __init__(self):
        super(AFSpectrum, self).__init__()
        # evenly-spaced beta binomials.  These are constants for now
        # TODO: should these be learned by making them Parameters?
        shapes = [(n + 1, 101 - n) for n in range(0, 100, 5)]
        self.a = torch.FloatTensor([shape[0] for shape in shapes])
        self.b = torch.FloatTensor([shape[1] for shape in shapes])

        # (pre-softmax) weights for the beta-binomial mixture.  Initialized as uniform.
        self.z = nn.Parameter(torch.ones(len(shapes)))

    # k "successes" out of n "trials" -- k and n are 1D tensors of the same size
    def log_likelihood(self, k, n):
        # note that we unsqueeze pi along the same dimension as alpha and beta
        log_pi = torch.unsqueeze(nn.functional.log_softmax(self.z, dim=0), 0)

        # n, k, a, b are all 1D tensors.  If we unsqueeze n,k along dim=1 and unsqueeze a,,b along dim=0
        # the resulting 2D log_likelihoods will have the structure
        # likelihoods[i,j] = beta_binomial(n[i],k[i],alpha[j],beta[j])
        log_likelihoods = beta_binomial(n.unsqueeze(1), k.unsqueeze(1), self.a.unsqueeze(0), self.b.unsqueeze(0))

        # by the convention above, the 0th dimension of log_likelihoods is n,k (batch) and the 1st dimension
        # is alpha, beta.  We sum over the latter, getting a 1D tensor corresponding to the batch
        return torch.logsumexp(log_pi + log_likelihoods, dim=1)

    # compute 1D tensor of log-likelihoods P(alt count|n, AF mixture model) over all data in batch
    def forward(self, batch: ReadSetBatch):
        return self.log_likelihood(batch.pd_tumor_alt_counts(), batch.pd_tumor_depths())

    # plot the mixture of beta densities
    def plot_spectrum(self, title):
        f = torch.arange(0.01, 0.99, 0.01)
        log_pi = nn.functional.log_softmax(self.z, dim=0)

        shapes = [(alpha, beta) for (alpha, beta) in zip(self.a.numpy(), self.b.numpy())]
        betas = [torch.distributions.beta.Beta(torch.FloatTensor([alpha]), torch.FloatTensor([beta])) for (alpha, beta)
                 in shapes]

        # list of tensors - the list is over the mixture components, the tensors are over AF values f
        unweighted_log_densities = torch.stack([beta.log_prob(f) for beta in betas], dim=0)
        # unsqueeze to make log_pi a column vector (2D tensor) for broadcasting
        weighted_log_densities = torch.unsqueeze(log_pi, 1) + unweighted_log_densities
        densities = torch.exp(torch.logsumexp(weighted_log_densities, dim=0))

        return mutect3.metrics.plotting.simple_plot([(f.detach().numpy(), densities.detach().numpy(), " ")], "AF", "density", title)