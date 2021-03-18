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


def beta_binomial(n, k, alpha, beta):
    return torch.lgamma(k + alpha) + torch.lgamma(n - k + beta) + torch.lgamma(alpha + beta) \
           - torch.lgamma(n + alpha + beta) - torch.lgamma(alpha) - torch.lgamma(beta)


class AFSpectrum:
    LEARNING_RATE = 1e-2
    EPOCHS = 5

    def __init__(self):
        shapes = [(n + 1, 101 - n) for n in range(0, 100, 5)]
        self.a = torch.FloatTensor([shape[0] for shape in shapes])
        self.b = torch.FloatTensor([shape[1] for shape in shapes])
        self.z = nn.Parameter(torch.ones(len(shapes)))

    # k "successes" out of n "trials"
    def log_likelihood(self, k, n):
        log_pi = nn.functional.log_softmax(self.z, dim=0)
        log_likelihoods = beta_binomial(n, k, self.a, self.b)
        return torch.logsumexp(log_pi + log_likelihoods, dim=0)

    # k_n_tuples is an iterable of tuples (k,n) as above
    def learn(self, k_n_tuples):
        # note -- a, b are constants, not learned
        optimizer = torch.optim.Adam([self.z], lr=AFSpectrum.LEARNING_RATE)
        for epoch in range(AFSpectrum.EPOCHS):
            for (k, n) in k_n_tuples:
                optimizer.zero_grad()
                nll = -self.log_likelihood(k, n)
                nll.backward()
                optimizer.step()

    # plot the mixture of beta densities
    def plot_spectrum(self):
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

#TODO: move this into the class
def get_artifact_af_spectrum(dataset):
    artifact_af_spectrum = AFSpectrum()
    artifact_counts_and_depths = [(len(datum.alt_tensor()), datum.mutect2_data().tumor_depth()) for datum in dataset if datum.artifact_label() == 1]
    artifact_af_spectrum.learn(artifact_counts_and_depths)
    return artifact_af_spectrum


# learn artifact and variant AF spectra from a validation or test loader
def learn_af_spectra(model, loader, m2_filters_to_keep={}, threshold=0.0):
    artifact_k_n = []
    variant_k_n = []
    for batch in loader:
        depths = [datum.tumor_depth() for datum in batch.mutect2_data()]
        filters = [m2.filters() for m2 in batch.mutect2_data()]
        alt_counts = batch.alt_counts()
        predictions = model(batch)
        for n in range(batch.size()):
            is_artifact = predictions[n] > threshold or filters[n].intersection(m2_filters_to_keep)
            (artifact_k_n if is_artifact else variant_k_n).append((alt_counts[n].item(), depths[n]))

    artifact_af_spectrum = AFSpectrum()
    variant_af_spectrum = AFSpectrum()

    artifact_af_spectrum.learn(artifact_k_n)
    variant_af_spectrum.learn(variant_k_n)
    artifact_proportion = len(artifact_k_n) / (len(artifact_k_n) + len(variant_k_n))

    return artifact_proportion, artifact_af_spectrum, variant_af_spectrum