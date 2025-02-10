import torch

"""
Note: all functions below operate on inputs of the same or broadcasting-compatible size.  When, for example, count tensors 
such as n and k pertain to a batch while parameters such as p, alpha, beta belong to a model, reshaping may be necessary.

Outputs have same shape as the input eg for the log binomial output[i,j. .] = log binomial(n[i,j..], k[i,j..], p[i,j..]).

All distributions are normalized (they contain nCk combinatorial factors etc) so that
sum_k exp (log prob(k | n, params)) = 1
"""


def binomial_log_lk(n, k, p):
    # binomial distribution log likelihood log Binom(k | n, p)
    # = log [nCk * p^k * (1-p)^(n-k)
    combinatorial_term = torch.lgamma(n + 1) - torch.lgamma(n - k + 1) - torch.lgamma(k + 1)
    return combinatorial_term + k * torch.log(p) + (n - k) * torch.log(1 - p)


def beta_binomial_log_lk(n, k, alpha, beta):
    # beta binomial distribution log likelihood  log P (k | n, alpha, beta)
    #                   = log integral_{0 to 1} Binomial(k | n, p) Beta(p | alpha, beta) dp
    combinatorial_term = torch.lgamma(n + 1) - torch.lgamma(n - k + 1) - torch.lgamma(k + 1)
    return combinatorial_term + torch.lgamma(k + alpha) + torch.lgamma(n - k + beta) + torch.lgamma(alpha + beta) \
           - torch.lgamma(n + alpha + beta) - torch.lgamma(alpha) - torch.lgamma(beta)


def gamma_binomial_log_lk(n, k, alpha, beta):
    # log likelihood of the gamma binomial distribution
    # same as the beta binomial, but with a gamma distribution overdispersion for the binomial distribution rather
    # than a beta distribution
    # WARNING: the approximations here only work if Gamma(f|alpha, beta) has very little density for f > 1
    # see pp 2 - 4 of my notebook
    alpha_tilde = (k + 1) * (n + 2) / (n - k + 1)
    beta_tilde = (n + 1) * (n + 2) / (n - k + 1)

    exponent_term = alpha_tilde * torch.log(beta_tilde) + alpha * torch.log(beta) -\
                    (alpha + alpha_tilde - 1) * torch.log(beta + beta_tilde)
    gamma_term = torch.lgamma(alpha + alpha_tilde - 1) - torch.lgamma(alpha) - torch.lgamma(alpha_tilde)
    return exponent_term + gamma_term - torch.log(n + 1)