import torch
from torch import lgamma

from permutect.utils.math_utils import subtract_in_log_space

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


def _incomplete_beta_coeff(n: int, a: torch.Tensor, b: torch.Tensor, x: torch.Tensor):
    """
    helper function for the incomplete beta continued fraction expansion
    n is the continued fraction coefficient, starting from n = 1
    a, b are the beta shape parameters
    x is the incompleteness i.e. how far to go in the CDF
    a, b, x should have the same or broadcast-compatible shape; n is an integer
    """
    if n % 2 == 0:
        m = n // 2
        return m*(b-m)*x / ((a+2*m-1) * (a+2*m))
    else:
        m = (n - 1) // 2
        return -(a+m)*(a+b+m)*x/((a+2*m)*(a+2*m+1))


def _incomplete_beta_cf_base(a: torch.Tensor, b: torch.Tensor, x: torch.Tensor):
    """
    continued fraction approximation to the incomplete beta function:
    I(a,b,x) = int_{0 to x} t^(a-1) (1-t)^(b-1) dt

    another helper function -- almost the complete thing, but this approximation, while technically correct, must be
    "flipped" for small x for good convergence
    a, b, x should have the same or broadcast-compatible shape
    """
    d1, d2, d3, d4, d5, d6 = (_incomplete_beta_coeff(k, a, b, x) for k in (1, 2, 3, 4, 5, 6))
    cf = 1 + d1/(1 + d2/(1 + d3/(1 + d4/(1 + d5/(1 + d6)))))
    return (x**a) * ((1-x)**b)/(a * cf)


def _log_incomplete_beta_cf_base(a: torch.Tensor, b: torch.Tensor, x: torch.Tensor):
    """
    log space continued fraction approximation to the incomplete beta function:
    log I(a,b,x) = log int_{0 to x} t^(a-1) (1-t)^(b-1) dt

    another helper function -- almost the complete thing, but this approximation, while technically correct, must be
    "flipped" for small x for good convergence
    a, b, x should have the same or broadcast-compatible shape
    """
    d1, d2, d3, d4, d5, d6, d7, d8 = (_incomplete_beta_coeff(k, a, b, x) for k in (1, 2, 3, 4, 5, 6, 7, 8))
    cf = 1 + d1 / (1 + d2 / (1 + d3 / (1 + d4 / (1 + d5 / (1 + d6 / (1 + d7 / (1 + d8)))))))
    return a * torch.log(x) + b * torch.log(1-x) - torch.log(a) - torch.log(cf)


def incomplete_beta(a: torch.Tensor, b: torch.Tensor, x: torch.Tensor):
    """
    continued fraction approximation to the incomplete beta function:
    I(a,b,x) = int_{0 to x} t^(a-1) (1-t)^(b-1) dt
    a, b, x should have the same or broadcast-compatible shape
    :return:
    """
    small_x = x < (a + 1)/(a+b+2)
    beta = torch.gamma(a) * torch.gamma(b) / torch.gamma(a+b)
    return _incomplete_beta_cf_base(a, b, x)*small_x + (beta - _incomplete_beta_cf_base(b, a, 1 - x))*(1-small_x)


def log_incomplete_beta(a: torch.Tensor, b: torch.Tensor, x: torch.Tensor):
    """
    log space continued fraction approximation to the incomplete beta function:
    log I(a,b,x) = log int_{0 to x} t^(a-1) (1-t)^(b-1) dt
    a, b, x should have the same or broadcast-compatible shape
    :return:
    """
    small_x = x < (a + 1)/(a+b+2)   # this is a mask, not a value, we it doesn't get logartihmed!!!
    log_beta = torch.lgamma(a) + torch.lgamma(b) - torch.lgamma(a+b)
    small_result = _log_incomplete_beta_cf_base(a, b, x)
    large_result = subtract_in_log_space(log_beta, _log_incomplete_beta_cf_base(b, a, 1 - x))

    return torch.where(small_x, small_result, large_result)


def log_regularized_incomplete_beta(a, b, x):
    """
        log space continued fraction approximation to the regularized incomplete beta function:
        log I(a,b,x) = log [(int_{0 to x} t^(a-1) (1-t)^(b-1) dt) / Beta(a,b)]

        This is also the cumulative distribution function of the beta distribution.
        a, b, x should have the same or broadcast-compatible shape
        :return:
        """
    return log_incomplete_beta(a, b, x) + lgamma(a+b) -lgamma(a) - lgamma(b)


def uniform_binomial_log_lk(n, k, x1, x2):
    # letting IB = incomplete binomial (not regularized)
    # uniform binomial distribution log likelihood  log P (k | n, x1, x2)
    #                   = log integral_{0 to 1} Binomial(k | n, p) Uniform(p | x1, x2) dp
    #                   = log(1/(x2-x1)) * log integral_{x1 to x2} Binomial(k | n, p) dp
    #                   = -log(x2-x1) + log(nCk) + log integral_{x1 to x2} p^k (1-p)^(n-k) dp
    #                   = -log(x2-x1) + log(nCk) + log [IB(k+1, n-k+1, x2) - IB(k+1, n-k+1, x1)]
    combinatorial_term = torch.lgamma(n + 1) - torch.lgamma(n - k + 1) - torch.lgamma(k + 1)
    uniform_normalization_term = -torch.log(x2-x1)
    incomplete_beta_term = subtract_in_log_space(log_incomplete_beta(k+1, n-k+1, x2), log_incomplete_beta(k+1, n-k+1, x1))

    return combinatorial_term + uniform_normalization_term + incomplete_beta_term
