import torch
from torch import nn, IntTensor, Tensor

from permutect.misc_utils import gpu_if_available
from permutect.utils.enums import Variation, Call


class PosteriorModelPriors(nn.Module):
    def __init__(self, variant_log_prior: float, artifact_log_prior: float, no_germline_mode: bool, device=gpu_if_available()):
        super(PosteriorModelPriors, self).__init__()
        self.no_germline_mode = no_germline_mode
        self._device = device

        # pre-softmax priors of different call types [log P(variant), log P(artifact), log P(seq error)] for each variant type
        self._unnormalized_priors_vc = torch.nn.Parameter(torch.ones(len(Variation), len(Call)))
        with torch.no_grad():
            self._unnormalized_priors_vc[:, Call.SOMATIC] = variant_log_prior
            self._unnormalized_priors_vc[:, Call.ARTIFACT] = artifact_log_prior
            self._unnormalized_priors_vc[:, Call.SEQ_ERROR] = 0
            self._unnormalized_priors_vc[:, Call.GERMLINE] = -9999 if self.no_germline_mode else 0
            self._unnormalized_priors_vc[:, Call.NORMAL_ARTIFACT] = artifact_log_prior

    def log_priors_bc(self, variant_types_b: IntTensor, allele_frequencies_1d: Tensor) -> Tensor:
        unnormalized_priors_bc = self._unnormalized_priors_vc[variant_types_b.long(), :]
        unnormalized_priors_bc[:, Call.SEQ_ERROR] = 0
        unnormalized_priors_bc[:, Call.GERMLINE] = -9999 if self.no_germline_mode else torch.log(1 - torch.square(1 - allele_frequencies_1d))     # 1 minus hom ref probability
        return torch.nn.functional.log_softmax(unnormalized_priors_bc, dim=-1)

    def update_priors_m_step(self, posteriors_nc, types_n, ignored_to_non_ignored_ratio):
        # update the priors in an EM-style M step.  We'll need the counts of each call type vs variant type
        total_nonignored = torch.sum(posteriors_nc).item()
        total_ignored = ignored_to_non_ignored_ratio * total_nonignored
        overall_total = total_ignored + total_nonignored

        with torch.no_grad():
            for c, call_type in enumerate(Call):
                if call_type == Call.SEQ_ERROR or call_type == Call.GERMLINE:
                    continue
                posteriors_n = posteriors_nc[:, c]

                for t, var_type in enumerate(Variation):
                    var_type_mask = (types_n == t)
                    total_for_this_call_and_var_type = torch.sum(posteriors_n * var_type_mask)

                    self._unnormalized_priors_vc[t, c] = torch.log(
                        total_for_this_call_and_var_type / (total_for_this_call_and_var_type + overall_total)).item()

            self._unnormalized_priors_vc[:, Call.SEQ_ERROR] = 0
            self._unnormalized_priors_vc[:, Call.GERMLINE] = -9999 if self.no_germline_mode else 0
