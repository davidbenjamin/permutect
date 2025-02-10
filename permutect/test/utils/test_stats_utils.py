import torch
from permutect.utils.stats_utils import binomial_log_lk, beta_binomial_log_lk


def test_normalization():
    for p in [0.1, 0.5, 0.7]:
        for n in [1, 10, 50]:
            k = torch.tensor(range(0, n+1))

            # binomial
            binomial_log_sum = torch.logsumexp(binomial_log_lk(n=torch.tensor([n]), k=k, p=torch.tensor([p])), dim=-1)
            assert torch.abs(binomial_log_sum).item() < 10**(-5)

            # beta binomial
            for (alpha, beta) in [(1,1), (1,5), (5,1), (10,10)]:
                beta_binomial_log_sum = torch.logsumexp(beta_binomial_log_lk(n=torch.tensor([n]), k=k,
                    alpha=torch.tensor([alpha]), beta=torch.tensor([beta])), dim=-1)
                assert torch.abs(beta_binomial_log_sum).item() < 10 ** (-5)


def test_beta_binomial_sharp_limit():
    scale = 100
    for n in [1, 10, 25]:
        n_tensor = torch.tensor([n])
        k = torch.tensor(range(0, n + 1))
        for (alpha, beta) in [(2, 2), (2, 5), (5, 2), (10, 10)]:
            p = torch.tensor([alpha / (alpha+beta)])
            big_alpha, big_beta = torch.tensor([scale*alpha]), torch.tensor([scale*beta])
            beta_binom = beta_binomial_log_lk(n=n_tensor, k=k, alpha=big_alpha, beta=big_beta)
            binom = binomial_log_lk(n=n_tensor, k=k, p=p)
            diff = torch.exp(binom) - torch.exp(beta_binom)
            assert torch.sum(torch.abs(diff)).item() < 10 ** (-1)
