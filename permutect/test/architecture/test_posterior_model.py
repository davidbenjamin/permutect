import torch

from permutect.architecture.posterior_model import germline_log_likelihood


def test_germline_log_likelihood_normalization():
    for het_beta in (None, 5, 10):
        for depth in (5, 50, 100):
            alt_counts_b = torch.arange(depth + 1)
            afs_b = 0.1 * torch.ones_like(alt_counts_b)
            mafs_b = 0.4 * torch.ones_like(alt_counts_b)
            depths_b = depth * torch.ones_like(alt_counts_b)

            log_lks_b = germline_log_likelihood(afs_b, mafs_b, alt_counts_b, depths_b, het_beta=None)
            logsum = torch.logsumexp(log_lks_b, dim=0)
            assert torch.abs(logsum).item() < 0.00001

            r = 90