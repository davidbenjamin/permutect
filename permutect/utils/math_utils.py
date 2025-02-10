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


def log_binomial_coefficient(n: torch.Tensor, k: torch.Tensor):
    return (n + 1).lgamma() - (k + 1).lgamma() - ((n - k) + 1).lgamma()


def f_score(tp, fp, total_true):
    fn = total_true - tp
    return tp / (tp + (fp + fn) / 2)