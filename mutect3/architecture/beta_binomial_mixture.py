import math
from torch import nn, lgamma, exp, unsqueeze, logsumexp
from torch.nn.functional import softmax, log_softmax
import torch

from mutect3.metrics.plotting import simple_plot


# note: this function works for n, k, alpha, beta tensors of the same shape
# the result is computed element-wise ie result[i,j. . .] = beta_binomial(n[i,j..], k[i,j..], alpha[i,j..], beta[i,j..)
# often n, k will correspond to a batch dimension and alpha, beta correspond to a model, in which case
# unsqueezing is necessary
def beta_binomial(n, k, alpha, beta):
    return lgamma(k + alpha) + lgamma(n - k + beta) + lgamma(alpha + beta) \
           - lgamma(n + alpha + beta) - lgamma(alpha) - lgamma(beta)


class BetaBinomialMixture(nn.Module):
    """
    This model takes in 2D tensor inputs (1st dimension is batch, 2nd is a feature vector that in practice is one-hot
    encoding of variant type) and as a function of input has a Beta mixture model.  That is, it computes for each input
    vector 1) a vector of mixture component weights 2) a vector of the alpha shape parameters of each component and 3) a
    vector of beta shape parameters of each component.  Due to batching these are all represented as 2D tensors.

    The computed likelihoods take in a 1D batch of total counts n and 1D batch of "success" counts k.
    """

    def __init__(self, input_size, num_components):
        super(BetaBinomialMixture, self).__init__()
        self.num_components = num_components
        self.input_size = input_size

        self.weights_pre_softmax = nn.Linear(in_features=input_size, out_features=num_components, bias=False)
        self.mean_pre_sigmoid = nn.Linear(in_features=input_size, out_features=num_components, bias=False)
        self.concentration_pre_exp = nn.Linear(in_features=input_size, out_features=num_components, bias=False)

        # the kth column of weights corresponds to the kth index of input = 1 and other inputs = 0
        # we are going to manually initialize to equal weights -- all zeroes
        # the alphas and betas will be equally spaced Beta distributions, for example, each column of alpha would be
        # 1, 11, 21, 31, 41, 51 and each column of beta would be 51, 41, 31, 21, 11, 1
        with torch.no_grad():
            self.weights_pre_softmax.weight.copy_(torch.zeros_like(self.weights_pre_softmax.weight))
            self.concentration_pre_exp.weight.copy_(torch.log(10*num_components*torch.ones_like(self.concentration_pre_exp.weight)))
            self.set_means((0.5 + torch.arange(num_components))/num_components)

    def set_means(self, means):
        assert len(means) == self.num_components
        each_mean_col_pre_sigmoid = torch.log(means / (1 - means))
        repeated_mean_cols = torch.hstack(self.input_size * [each_mean_col_pre_sigmoid.unsqueeze(dim=1)])
        with torch.no_grad():
            self.mean_pre_sigmoid.weight.copy_(repeated_mean_cols)

    def set_weights(self, weights):
        assert len(weights) == self.num_components
        pre_softmax_weights = torch.log(weights)
        repeated = torch.hstack(self.input_size * [pre_softmax_weights.unsqueeze(dim=1)])
        with torch.no_grad():
            self.weights_pre_softmax.weight.copy_(repeated)

    '''
    here x is a 2D tensor, 1st dimension batch, 2nd dimension being features that determine which Beta mixture to use
    n and k are 1D tensors, the only dimension being batch.
    '''
    def forward(self, x, n, k):
        log_weights = log_softmax(self.weights_pre_softmax(x), dim=1)

        # 2D tensors -- 1st dim batch, 2nd dim mixture component
        means = torch.sigmoid(self.mean_pre_sigmoid(x))
        concentrations = torch.exp(self.concentration_pre_exp(x))
        alphas = means * concentrations
        betas = (1 - means) * concentrations

        # we make them 2D, with 1st dim batch, to match alpha and beta.  A single column is OK because the single value of
        # n/k are broadcast over all mixture components
        n_2d = unsqueeze(n, dim=1)
        k_2d = unsqueeze(k, dim=1)

        log_component_likelihoods = beta_binomial(n_2d, k_2d, alphas, betas)
        log_weighted_component_likelihoods = log_weights + log_component_likelihoods

        # yields one number per batch, squeezed into 1D output tensor
        return logsumexp(log_weighted_component_likelihoods, dim=1, keepdim=False)

    # given 1D input tensor, return 1D tensors of component alphas and betas
    def component_shapes(self, input_1d):
        input_2d = input_1d.unsqueeze(dim=0)
        means = torch.sigmoid(self.mean_pre_sigmoid(input_2d)).squeeze()
        concentrations = torch.exp(self.concentration_pre_exp(input_2d)).squeeze()
        alphas = means * concentrations
        betas = (1 - means) * concentrations
        return alphas, betas

    def component_weights(self, input_1d):
        input_2d = input_1d.unsqueeze(dim=0)
        return softmax(self.weights_pre_softmax(input_2d), dim=1).squeeze()

    # given 1D input tensor, return the moments E[x], E[ln(x)], and E[x ln(x)] of the underlying beta mixture
    def moments_of_underlying_beta_mixture(self, input_1d):
        alphas, betas = self.component_shapes(input_1d)
        weights = self.component_weights(input_1d)

        # E[x]
        component_means = alphas / (alphas + betas)
        mixture_mean = torch.sum(weights * component_means)

        # E[ln(x)]
        component_log_means = torch.digamma(alphas) - torch.digamma(alphas + betas)   # digamma broadcasts to make 1D tensor
        mixture_log_mean = torch.sum(weights * component_log_means)

        # E[x ln(x)]
        component_log_linear_means = component_means * (torch.digamma(alphas + 1) - torch.digamma(alphas + betas + 1))
        mixture_log_linear_mean = torch.sum(weights * component_log_linear_means)

        return mixture_mean, mixture_log_mean, mixture_log_linear_mean

    '''
    here x is a 2D tensor, 1st dimension batch, 2nd dimension being features that determine which Beta mixture to use
    n is a 1D tensor, the only dimension being batch, and we sample a 1D tensor of k's
    '''
    def sample(self, x, n):
        # compute weights and select one mixture component from the corresponding multinomial for each datum / row
        weights = softmax(self.weights_pre_softmax(x).detach(), dim=1)  # 2D tensor
        component_indices = torch.multinomial(weights,  num_samples=1, replacement=True)    # 2D tensor with one column

        # get 1D tensors of one selected alpha and beta shape parameter per datum / row, then sample a fraction from each
        # It may be very wasteful computing everything and only using one component, but this is just for unit testing
        means = torch.sigmoid(self.mean_pre_sigmoid(x).detach()).gather(dim=1, index=component_indices).squeeze()
        concentrations = torch.exp(self.concentration_pre_exp(x).detach()).gather(dim=1, index=component_indices).squeeze()
        alphas = means * concentrations
        betas = (1 - means) * concentrations

        fractions = torch.distributions.beta.Beta(alphas, betas).sample()   # 1D tensor

        # recall, n and fractions are 1D tensors; result is also 1D tensor, one "success" count per datum
        return torch.distributions.binomial.Binomial(total_count=n, probs=fractions).sample()

    def fit(self, num_epochs, inputs_2d_tensor, depths_1d_tensor, alt_counts_1d_tensor, batch_size=64):
        optimizer = torch.optim.Adam(self.parameters())
        num_batches = math.ceil(len(alt_counts_1d_tensor)/batch_size)

        for epoch in range(num_epochs):
            for batch in range(num_batches):
                batch_start = batch * batch_size
                batch_end = min(batch_start + batch_size, len(alt_counts_1d_tensor))
                batch_slice = slice(batch_start, batch_end)
                loss = -torch.mean(self.forward(inputs_2d_tensor[batch_slice], depths_1d_tensor[batch_slice], alt_counts_1d_tensor[batch_slice]))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    '''
    here x is a 1D tensor, a single datum/row of the 2D tensors as above
    '''
    def plot_spectrum(self, x, title):
        f = torch.arange(0.01, 0.99, 0.01)  # 1D tensor

        unsqueezed = x.unsqueeze(dim=0)     # this and the three following tensors are 2D tensors with one row
        log_weights = log_softmax(self.weights_pre_softmax(unsqueezed).detach(), dim=1)
        means = torch.sigmoid(self.mean_pre_sigmoid(unsqueezed).detach())
        concentrations = torch.exp(self.concentration_pre_exp(unsqueezed).detach())
        alphas = means * concentrations
        betas = (1 - means) * concentrations

        # since f.unsqueeze(dim=1) is 2D column vector, log_prob produces 2D tensor where row index is f and column index is mixture component
        # adding the single-row 2D tensor log_weights broadcasts to each row / value of f
        # then we apply log_sum_exp, dim= 1, to sum over components and get a log_density for each f
        densities = exp(torch.logsumexp(log_weights + torch.distributions.beta.Beta(alphas, betas).log_prob(f.unsqueeze(dim=1)), dim=1, keepdim=False)) # 1D tensor

        return simple_plot([(f.numpy(), densities.numpy(), " ")], "AF", "density", title)


