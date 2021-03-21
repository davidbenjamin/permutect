# Now let's learn the spectrum of artifact allele fractions in order to combine our calibrated classifier with priors
# to get a posterior probability
# the model is alt count ~ sum_j pi_j BetaBinomial(alpha_j, beta_j, depth)
# where the coefficients pi_j sum to 1 and alpha_j, beta_j are fixed as (1,101), (6,96), (11, 91). . .(101,1)
# and we want to learn pi_j by MLE
# we enforce the normaliztion constraint by letting pi = softmax(z)


# beta binomial log likelihood
# shape parameters are 1D tensors of the same length, and the log gamma is broadcast over all pairs (alpha,beta)
import torch
from torch import nn
import matplotlib.pyplot as plt


# note: this function works for alpha. beta tensors of the same shape, in which case it broadcasts
def beta_binomial(n, k, alpha, beta):
    return torch.lgamma(k + alpha) + torch.lgamma(n - k + beta) + torch.lgamma(alpha + beta) \
           - torch.lgamma(n + alpha + beta) - torch.lgamma(alpha) - torch.lgamma(beta)


class AFSpectrum:
    LEARNING_RATE = 1e-2

    def __init__(self):
        # evenly-spaced beta binomials.  These are constants for now
        # TODO: should these be learned?
        shapes = [(n + 1, 101 - n) for n in range(0, 100, 5)]
        self.a = torch.FloatTensor([shape[0] for shape in shapes])
        self.b = torch.FloatTensor([shape[1] for shape in shapes])

        # (pre-softmax) weights for the beta-binomial mixture.  Initialized as uniform.
        self.z = nn.Parameter(torch.ones(len(shapes)))

        self.optimizer = torch.optim.Adam([self.z], lr=AFSpectrum.LEARNING_RATE)

    # k "successes" out of n "trials"
    def log_likelihood(self, k, n):
        log_pi = nn.functional.log_softmax(self.z, dim=0)
        log_likelihoods = beta_binomial(n, k, self.a, self.b)
        return torch.logsumexp(log_pi + log_likelihoods, dim=0)

    def learn_epoch(self, k_n_tuples):
        for (k, n) in k_n_tuples:
            self.optimizer.zero_grad()
            nll = -self.log_likelihood(k, n)
            nll.backward()
            self.optimizer.step()

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
        log_densities = torch.logsumexp(weighted_log_densities, dim=0)
        fig = plt.figure()
        spec = fig.gca()
        spec.plot(f.detach().numpy(), torch.exp(log_densities).detach().numpy())
        spec.set_title(title)
        return fig, spec

# contains variant spectrum, artifact spectrum, and artifact/variant log prior ratio
class PriorModel:
    def __init__(self, initial_log_ratio = 0.0):
        self.variant_spectrum = AFSpectrum()
        self.artifact_spectrum = AFSpectrum()
        self.log_artifact_to_variant_ratio = 0.5

    def learn_epoch(self, model, loader, m2_filters_to_keep={}, threshold=0.0):
        artifact_k_n = []
        variant_k_n = []
        for batch in loader:
            depths = [datum.tumor_depth() for datum in batch.mutect_info()]
            filters = [m2.filters() for m2 in batch.mutect_info()]
            alt_counts = batch.alt_counts()
            predictions = model(batch)
            for n in range(batch.size()):
                is_artifact = predictions[n] > threshold or filters[n].intersection(m2_filters_to_keep)
                (artifact_k_n if is_artifact else variant_k_n).append((alt_counts[n].item(), depths[n]))

        self.artifact_spectrum.learn_epoch(artifact_k_n)
        self.variant_spectrum.learn_epoch(variant_k_n)
        self.log_artifact_to_variant_ratio = torch.log(len(artifact_k_n) / len(variant_k_n))

    # log prior ratio between artifact and variant
    def prior_log_odds(self, batch):
        alt_counts = batch.alt_counts().numpy()
        depths = [datum.tumor_depth() for datum in batch.mutect_info()]

        # these are relative log priors of artifacts and variants to have k alt reads out of n total
        spectrum_log_odds = torch.FloatTensor(
            [self.artifact_spectrum.log_likelihood(k, n).item() - self.variant_spectrum.log_likelihood(k, n).item() \
             for (k, n) in zip(alt_counts, depths)])

        return self.log_artifact_to_variant_ratio + spectrum_log_odds

    def plot_spectra(self):
        self.artifact_spectrum.plot_spectrum("Artifact AF spectrum")
        self.variant_spectrum.plot_spectrum("Variant AF spectrum")