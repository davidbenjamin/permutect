import torch

from permutect.architecture.posterior_model import germline_log_likelihood, initialize_normal_artifact_spectra
from permutect.architecture.spectra.artifact_spectra import ArtifactSpectra
from permutect.architecture.spectra.somatic_spectrum import SomaticSpectrum


def test_log_likelihood_normalization():
    somatic_spectrum = SomaticSpectrum(num_components=5)
    artifact_spectra = ArtifactSpectra()
    normal_artifact_spectra = initialize_normal_artifact_spectra()
    for het_beta in (None, 5, 10):
        for depth in (5, 50, 100):
            alt_counts_b = torch.arange(depth + 1)
            afs_b = 0.1 * torch.ones_like(alt_counts_b)
            mafs_b = 0.4 * torch.ones_like(alt_counts_b)
            depths_b = depth * torch.ones_like(alt_counts_b)
            var_types_b = (2 * torch.ones_like(alt_counts_b)).int()

            germline_log_lks_b = germline_log_likelihood(afs_b, mafs_b, alt_counts_b, depths_b, het_beta=None)
            somatic_log_lks_b = somatic_spectrum.forward(depths_b, alt_counts_b, mafs_b)
            artifact_log_lks_b = artifact_spectra.forward(var_types_b, depths_b, alt_counts_b)
            norm_art_log_lks_b = normal_artifact_spectra.forward(var_types_b, depths_b, alt_counts_b)
            for log_lks_b in (germline_log_lks_b, somatic_log_lks_b, artifact_log_lks_b, norm_art_log_lks_b):
                lks_b = torch.exp(log_lks_b)
                assert torch.abs(torch.sum(lks_b) - 1).item() < 0.00001

            r = 90