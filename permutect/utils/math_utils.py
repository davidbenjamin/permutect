from __future__ import annotations

import math

import torch


def find_factors(n: int):
    result = []
    for m in range(1, int(math.sqrt(n)) + 1):
        if n % m == 0:
            result.append(m)
            if (n // m) > m:
                result.append(n // m)
    result.sort()
    return result


def prob_to_logit(prob: float):
    # the inverse of the sigmoid function.  Convert a probability to a logit.
    clipped_prob = 0.5 + 0.9999999 * (prob - 0.5)
    return math.log(clipped_prob / (1 - clipped_prob))


def inverse_sigmoid(probs: torch.Tensor) -> torch.Tensor:
    clipped_probs = 0.5 + 0.9999999 * (probs - 0.5)
    return torch.log(clipped_probs / (1 - clipped_probs))


def log_binomial_coefficient(n: torch.Tensor, k: torch.Tensor):
    return (n + 1).lgamma() - (k + 1).lgamma() - ((n - k) + 1).lgamma()


def f_score(tp, fp, total_true):
    fn = total_true - tp
    return tp / (tp + (fp + fn) / 2)


def add_in_log_space(x: torch.Tensor, y: torch.Tensor):
    """
    implement result = log(exp(x) + exp(y)) in a numerically stable way by extracting the elementwise max:
    with M = torch.maximum(x,y); result = log(exp(M)exp(x-M) + exp(M)exp(y-M)) = M + log(exp(x-M) + exp(y-M))
    this could be done by stacking x and y along a new dimension, then doing logsumexp along that dimension, but
    that is cumbersome.
    :return:
    """
    m = torch.maximum(x, y)
    return m + torch.log(torch.exp(x-m) + torch.exp(y-m))


def subtract_in_log_space(x: torch.Tensor, y: torch.Tensor):
    """
    implement result = log(exp(x) - exp(y)) in a numerically stable way by extracting the elementwise max:
    with M = torch.maximum(x,y); result = log(exp(M)exp(x-M) - exp(M)exp(y-M)) = M + log(exp(x-M) - exp(y-M))

    note: this only works if x > y, elementwise!
    :return:
    """
    m = torch.maximum(x, y)
    return m + torch.log(torch.exp(x-m) - torch.exp(y-m))