from torch import nn, lgamma, exp, log_softmax, unsqueeze, logsumexp
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

        self.weights_pre_softmax = nn.Linear(in_features=input_size, out_features=num_components, bias=False)
        self.alpha_pre_exp = nn.Linear(in_features=input_size, out_features=num_components, bias=False)
        self.beta_pre_exp = nn.Linear(in_features=input_size, out_features=num_components, bias=False)

        # the kth column of weights corresponds to the kth index of input = 1 and other inputs = 0
        # we are going to manually initialize to equal weights -- all zeroes
        # the alphas and betas will be equally spaced Beta distributions, for example, each column of alpha would be
        # 1, 11, 21, 31, 41, 51 and each column of beta would be 51, 41, 31, 21, 11, 1
        with torch.no_grad():
            self.weights_pre_softmax.weight.copy_(torch.zeros_like(self.weights_pre_softmax.weight))

            each_alpha_col = 1.1 + torch.arange(num_components)*50/num_components
            each_beta_col = each_alpha_col.flip(dims=[0])
            repeated_alpha_cols = torch.hstack(input_size * [each_alpha_col.unsqueeze(dim=1)])
            repeated_beta_cols = torch.hstack(input_size * [each_beta_col.unsqueeze(dim=1)])

            self.alpha_pre_exp.weight.copy_(torch.log(repeated_alpha_cols))
            self.beta_pre_exp.weight.copy_(torch.log(repeated_beta_cols))

    '''
    here x is a 2D tensor, 1st dimension batch, 2nd dimension being features that determine which Beta mixture to use
    n and k are 1D tensors, the only dimension being batch.
    '''
    def forward(self, x, n, k):
        log_weights = log_softmax(self.weights_pre_softmax(x), dim=1)
        alphas = exp(self.alpha_pre_exp(x))
        betas = exp(self.beta_pre_exp(x))

        # we make them 2D, with 1st dim batch, to match alpha and beta.  A single column is OK because the single value of
        # n/k are broadcast over all mixture components
        n_2d = unsqueeze(n, dim=1)
        k_2d = unsqueeze(k, dim=1)

        log_component_likelihoods = beta_binomial(n_2d, k_2d, alphas, betas)
        log_weighted_component_likelihoods = log_weights + log_component_likelihoods

        # yields one number per batch, squeezed into 1D output tensor
        return logsumexp(log_weighted_component_likelihoods, dim=1, keepdim=False)

    '''
    here x is a 2D tensor, 1st dimension batch, 2nd dimension being features that determine which Beta mixture to use
    n is a 1D tensor, the only dimension being batch, and we sample a 1D tensor of k's
    '''
    def sample(self, x, n):
        # compute weights and select one mixture component from the corresponding multinomial for each datum / row
        weights = exp(log_softmax(self.weights_pre_softmax(x).detach(), dim=1))  # 2D tensor
        component_indices = torch.multinomial(weights,  num_samples=1, replacement=True)    # 2D tensor with one column

        # get 1D tensors of one selected alpha and beta shape parameter per datum / row, then sample a fraction from each
        # It may be very wasteful computing everything and only using one component, but this is just for unit testing
        alphas = exp(self.alpha_pre_exp(x).detach()).gather(dim=1, index=component_indices).squeeze()
        betas = exp(self.beta_pre_exp(x).detach()).gather(dim=1, index=component_indices).squeeze()
        fractions = torch.distributions.beta.Beta(alphas, betas).sample()   # 1D tensor

        # recall, n and fractions are 1D tensors; result is also 1D tensor, one "success" count per datum
        return torch.distributions.binomial.Binomial(total_count=n, probs=fractions).sample()

    '''
    here x is a 1D tensor, a single datum/row of the 2D tensors as above
    '''
    def plot_spectrum(self, x, title):
        f = torch.arange(0.01, 0.99, 0.01)  # 1D tensor

        unsqueezed = x.unsqueeze(dim=0)     # this and the three following tensors are 2D tensors with one row
        log_weights = log_softmax(self.weights_pre_softmax(unsqueezed).detach(), dim=1)
        alphas = exp(self.alpha_pre_exp(unsqueezed).detach())
        betas = exp(self.beta_pre_exp(unsqueezed).detach())

        # since f.unsqueeze(dim=1) is 2D column vector, log_prob produces 2D tensor where row index is f and column index is mixture component
        # adding the single-row 2D tensor log_weights broadcasts to each row / value of f
        # then we apply log_sum_exp, dim= 1, to sum over components and get a log_density for each f
        densities = exp(torch.logsumexp(log_weights + torch.distributions.beta.Beta(alphas, betas).log_prob(f.unsqueeze(dim=1)), dim=1, keepdim=False)) # 1D tensor

        return simple_plot([(f.numpy(), densities.numpy(), " ")], "AF", "density", title)


