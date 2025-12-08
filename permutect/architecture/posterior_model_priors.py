from collections import defaultdict

import torch
from torch import nn, IntTensor, Tensor

from permutect.metrics import plotting
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

    def update_priors_m_step(self, posterior_totals_vc, ignored_to_non_ignored_ratio):
        # update the priors in an EM-style M step.  We'll need the counts of each call type vs variant type
        total_nonignored = torch.sum(posterior_totals_vc).item()
        total_ignored = ignored_to_non_ignored_ratio * total_nonignored
        overall_total = total_ignored + total_nonignored

        with torch.no_grad():
            self._unnormalized_priors_vc.copy_(torch.log(posterior_totals_vc/(posterior_totals_vc + overall_total)))
            self._unnormalized_priors_vc[:, Call.SEQ_ERROR] = 0
            self._unnormalized_priors_vc[:, Call.GERMLINE] = -9999 if self.no_germline_mode else 0

    def plot_log_priors(self):
        # bar plot of log priors -- data is indexed by call type name, and x ticks are variant types
        log_prior_bar_plot_data = defaultdict(list)
        for var_type_idx, variant_type in enumerate(Variation):
            log_priors = self.log_priors_bc(
                torch.LongTensor([var_type_idx]).to(device=self._device, dtype=self._unnormalized_priors_vc.dtype), torch.tensor([0.001]))
            log_priors_cpu = log_priors.squeeze().detach().cpu()
            for call_type in (Call.SOMATIC, Call.ARTIFACT, Call.NORMAL_ARTIFACT):
                log_prior_bar_plot_data[call_type.name].append(log_priors_cpu[call_type])

        prior_fig, prior_ax = plotting.grouped_bar_plot(log_prior_bar_plot_data, [v_type.name for v_type in Variation],
                                                        "log priors")
        return prior_fig, prior_ax
