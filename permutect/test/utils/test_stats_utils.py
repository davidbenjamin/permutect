import torch
from scipy.special import betainc
from permutect.utils.stats_utils import binomial_log_lk, beta_binomial_log_lk, log_regularized_incomplete_beta, \
    uniform_binomial_log_lk


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

            # uniform binomial
            for (x1, x2) in [(0.1, 0.2), (0.1, 0.5), (0.4, 0.5), (0.5, 0.9)]:
                uniform_binomial_log_sum = torch.logsumexp(uniform_binomial_log_lk(n=torch.tensor([n]), k=k,
                    x1=torch.tensor([x1]), x2=torch.tensor([x2])), dim=-1)
                assert torch.abs(uniform_binomial_log_sum).item() < 0.005


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


def test_log_regularized_incomplete_beta():
    for (a, b) in [(1,1), (2,5), (5,2), (10, 10)]:
        atens, btens = torch.tensor([a]), torch.tensor([b])

        # it's a CDF, so should be 0 at 0 and 1 at 1
        res0 = log_regularized_incomplete_beta(atens, btens, torch.tensor([0]))
        assert torch.exp(res0).item() < 10**(-6)

        res1 = log_regularized_incomplete_beta(atens, btens, torch.tensor([1]))
        assert torch.abs(1 - torch.exp(res1)).item() < 10 ** (-6)

        # for symmetric beta, it's a CDF so should equal 0.5 at 0.5
        if a == b:
            res05 = log_regularized_incomplete_beta(atens, btens, torch.tensor([0.5]))
            assert torch.abs(0.5 - torch.exp(res05)).item() < 10 ** (-3)

        # test random values against scipy implementation:
        for x in [0.1, 0.2, 0.3, 0.7, 0.9]:
            res = torch.exp(log_regularized_incomplete_beta(atens, btens, torch.tensor([x])))
            expected = betainc(a, b, x)
            assert torch.abs(res - expected).item() < 10 ** (-3)


def test_uniform_binomial():
    # a very sharp uniform binomial reduces to a binomial
    for n in [1, 10, 25]:
        ntens = torch.tensor([n])
        k = torch.tensor(range(0, n + 1))
        for p in [0.1, 0.5, 0.7]:
            ptens = torch.tensor([p])
            x1, x2 = p - 0.001, p + 0.001
            x1tens, x2tens = torch.tensor([x1]), torch.tensor([x2])

            ub = torch.exp(uniform_binomial_log_lk(ntens, k, x1tens, x2tens))
            exact = torch.exp(binomial_log_lk(ntens, k, ptens))
            assert torch.sum(torch.abs(ub - exact)).item() < 0.005

            # the mixture of uniform [x-d, x] and [x,x+d] is the uniform [x-d,x+d]
            ub1 = torch.exp(uniform_binomial_log_lk(ntens, k, ptens-0.05, ptens))
            ub2 = torch.exp(uniform_binomial_log_lk(ntens, k, ptens, ptens+0.05))
            ub3 = torch.exp(uniform_binomial_log_lk(ntens, k, ptens-0.05, ptens + 0.05))
            assert torch.sum(torch.abs(ub3 - (ub1+ub2)/2)).item() < 0.005

        # when x1=0, x2=1 the integral is the normalization of the beta distribution
        # log(nCk) + log integral_{0 to 1} p ^ k (1 - p) ^ (n - k) dp
        # =log(nCk) + log Beta(k+1, n-k+1)
        # = lgamma(n + 1) - lgamma(n - k + 1) - lgamma(k + 1) +lgamma(k+1) + lgamma(n-k+1) - lgamma(n+2)
        # = lgamma(n + 1) - lgamma(n+2) = -log(gamma(n+2)/gamma(n+1)) = -log(n+1)
        ub = torch.exp(uniform_binomial_log_lk(ntens, k, torch.tensor([0]), torch.tensor([1])))
        exact = 1/(n+1)
        assert torch.sum(torch.abs(ub - exact)).item() < 0.01




