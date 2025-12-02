import torch
from torch import nn

from permutect.architecture.spectra.artifact_spectra import ArtifactSpectra
from permutect.architecture.spectra.normal_artifact_spectrum import NormalArtifactSpectrum
from permutect.architecture.spectra.somatic_spectrum import SomaticSpectrum
from permutect.data.datum import DEFAULT_GPU_FLOAT, DEFAULT_CPU_FLOAT
from permutect.data.posterior_data import PosteriorBatch
from permutect.misc_utils import gpu_if_available
from permutect.utils.enums import Call
from permutect.utils.stats_utils import beta_binomial_log_lk


# TODO: write unit test asserting that this comes out to zero when counts are zero
# given germline, the probability of these particular reads being alt
def germline_log_likelihood(afs, mafs, alt_counts, depths, het_beta=None):
    hom_alpha, hom_beta = torch.tensor([98.0], device=depths.device), torch.tensor([2.0], device=depths.device)
    het_alpha, het_beta_to_use = (None, None) if het_beta is None else (torch.tensor([het_beta], device=depths.device), torch.tensor([het_beta], device=depths.device))
    het_probs = 2 * afs * (1 - afs)
    hom_probs = afs * afs
    het_proportion = het_probs / (het_probs + hom_probs)
    hom_proportion = 1 - het_proportion

    log_mafs = torch.log(mafs)
    log_1m_mafs = torch.log(1 - mafs)
    log_half_het_prop = torch.log(het_proportion / 2)

    ref_counts = depths - alt_counts

    combinatorial_term = torch.lgamma(depths + 1) - torch.lgamma(alt_counts + 1) - torch.lgamma(ref_counts + 1)
    # the following should both be 1D tensors of length batch size
    alt_minor_binomial = combinatorial_term + alt_counts * log_mafs + ref_counts * log_1m_mafs
    alt_major_binomial = combinatorial_term + ref_counts * log_mafs + alt_counts * log_1m_mafs
    alt_minor_ll = log_half_het_prop + (alt_minor_binomial if het_beta is None else beta_binomial_log_lk(depths, alt_counts, het_alpha, het_beta_to_use))
    alt_major_ll = log_half_het_prop + (alt_major_binomial if het_beta is None else beta_binomial_log_lk(depths, alt_counts, het_alpha, het_beta_to_use))
    hom_ll = torch.log(hom_proportion) + beta_binomial_log_lk(depths, alt_counts, hom_alpha, hom_beta)

    return torch.logsumexp(torch.vstack((alt_minor_ll, alt_major_ll, hom_ll)), dim=0)


class PosteriorModelSpectra(nn.Module):
    """
    Encapsulates all spectra of the posterior model:
        1) somatic variant spectrum
        2) artifact spectra
        3) normal artifact spectrum
        4) germline likelihoods
    Basically, anything that computes the likelihood of alt and ref read counts goes here.
    """

    def __init__(self, device=gpu_if_available(), het_beta: float = None):
        super(PosteriorModelSpectra, self).__init__()

        self._device = device
        self._dtype = DEFAULT_GPU_FLOAT if device != torch.device("cpu") else DEFAULT_CPU_FLOAT
        self.het_beta = het_beta

        # TODO introduce parameters class so that num_components is not hard-coded
        self.somatic_spectrum = SomaticSpectrum(num_components=5)
        self.artifact_spectra = ArtifactSpectra()
        self.normal_artifact_spectra = NormalArtifactSpectrum()

    def spectra_log_likelihoods_bc(self, batch: PosteriorBatch) -> torch.Tensor:
        """
        'bc' indexing denotes by datum within batch, then by Call type
        """
        var_types_b, afs_b, mafs_b = batch.get_variant_types(), batch.get_allele_frequencies(), batch.get_mafs()
        depths_b, alt_counts_b = batch.get_original_depths(), batch.get_original_alt_counts()
        normal_depths_b, normal_alt_counts_b = batch.get_original_normal_depths(), batch.get_original_normal_alt_counts()

        na_tumor_log_lks_b, na_normal_log_lks_b = self.normal_artifact_spectra.forward(var_types_b=var_types_b,
            tumor_alt_counts_b=alt_counts_b, tumor_depths_b=depths_b,
            normal_alt_counts_b=normal_alt_counts_b, normal_depths_b=normal_depths_b)

        spectra_log_lks_bc = torch.zeros((batch.size(), len(Call)), device=self._device, dtype=self._dtype)
        tumor_artifact_spectrum_log_lks_b = self.artifact_spectra.forward(batch.get_variant_types(), depths_b,
                                                                          alt_counts_b)
        spectra_log_lks_bc[:, Call.SOMATIC] = self.somatic_spectrum.forward(depths_b, alt_counts_b, mafs_b)
        spectra_log_lks_bc[:, Call.ARTIFACT] = tumor_artifact_spectrum_log_lks_b
        spectra_log_lks_bc[:, Call.NORMAL_ARTIFACT] = na_tumor_log_lks_b
        spectra_log_lks_bc[:, Call.SEQ_ERROR] = batch.get_seq_error_log_lks()
        spectra_log_lks_bc[:, Call.GERMLINE] = germline_log_likelihood(afs_b, mafs_b, alt_counts_b, depths_b, self.het_beta)

        normal_log_lks_bc = torch.zeros_like(spectra_log_lks_bc)
        normal_log_lks_bc[:, Call.SOMATIC] = batch.get_normal_seq_error_log_lks()
        normal_log_lks_bc[:, Call.ARTIFACT] = batch.get_normal_seq_error_log_lks()
        normal_log_lks_bc[:, Call.SEQ_ERROR] = batch.get_normal_seq_error_log_lks()
        normal_log_lks_bc[:, Call.NORMAL_ARTIFACT] = torch.where(normal_alt_counts_b < 1, -9999, na_normal_log_lks_b)
        normal_log_lks_bc[:, Call.GERMLINE] = germline_log_likelihood(afs_b, batch.get_normal_mafs(),
            normal_alt_counts_b, normal_depths_b, self.het_beta)

        return spectra_log_lks_bc, normal_log_lks_bc