from collections import defaultdict
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.nn import Parameter
from scipy import stats

from permutect.data.posterior_data import PosteriorBatch
from permutect.metrics import plotting
from permutect.misc_utils import gpu_if_available, backpropagate
from permutect.utils.array_utils import index_tensor, add_at_index
from permutect.utils.enums import Variation, Call


def get_ref_contexts_and_alt_bases(batch: PosteriorBatch):
    # each row is the ref sequence, followed by the alt sequence (hstacked) with A = 0, C = 1 . . . deletion = 4
    haplotypes_bs = batch.get_haplotypes_bs()
    seq_length = haplotypes_bs.shape[-1] // 2
    ref_center_idx = (seq_length - 1) // 2
    alt_center_idx = ref_center_idx + seq_length
    idx0 =  haplotypes_bs[:, ref_center_idx - 1]
    idx1 = haplotypes_bs[:, ref_center_idx].int()
    idx2 = haplotypes_bs[:, ref_center_idx + 1]
    idx3 = haplotypes_bs[:, alt_center_idx]
    return idx0, idx1, idx2, idx3


class PosteriorModelPriors(nn.Module):
    """
    Important technical point: the priors here really only apply to somatic variants, artifacts, and
    normal artifacts.

    The prior for germline variants depends not on this Module but on the population allele frequency of each
    particular variant.  Therefore the germline prior is actually implemented within the germline_log_likelihoods
    function of the PosteriorModelSpectra class.  In this model the log germline prior is set to zero (except in
    no_germline mode, where it is set to -9999 to effectively turn off germline calls) in order to have no effect.

    The sequencing error log prior is also set to zero because sequencing error essentially *always* happens!  More
    precisely, the sequencing error *generative process* of the sequencer having some stochastic process of misread
    bases is always occurring.  Usually this process does not cause error.  The question is not whether sequencing error
    exists, but whether sequencing error explains the amount of alt allele reads observed.  Thus the mathematically
    correct thing is to set the prior to one (log prior to zero) and allow the likelihoods (basically the TLOD from
    Mutect2) to distinguish.
    """
    def __init__(self, variant_log_prior: float, artifact_log_prior: float, no_germline_mode: bool, device=gpu_if_available()):
        super(PosteriorModelPriors, self).__init__()
        self.no_germline_mode = no_germline_mode
        self._device = device

        # pre-softmax priors of different call types [log P(variant), log P(artifact), log P(seq error)] for each variant type
        # although these vectors are defined for all variant types, somatic SNVs are handled separately
        self._unnormalized_priors_vc = torch.nn.Parameter(torch.zeros(len(Variation), len(Call)))
        with torch.no_grad():
            self._unnormalized_priors_vc[:, Call.SOMATIC] = variant_log_prior
            self._unnormalized_priors_vc[:, Call.ARTIFACT] = artifact_log_prior
            self._unnormalized_priors_vc[:, Call.GERMLINE] = -9999 if self.no_germline_mode else 0
            self._unnormalized_priors_vc[:, Call.NORMAL_ARTIFACT] = artifact_log_prior

        # context and substitution dependent (the 3 r's stand for ref context, the 'a' is for alt base)
        # we have 5 values just in case there is a D in the ref context (an indel next to a SNV, perhaps)
        self.somatic_snv_log_priors_rrra = Parameter(
            variant_log_prior * torch.ones((5, 5, 5, 5), device=self._device))

        self.NONTRIVIAL_CONTEXTS_rrra = torch.ones((5, 5, 5, 5), device=self._device)
        self.NONTRIVIAL_CONTEXTS_rrra[4, :, :, :] = 0
        self.NONTRIVIAL_CONTEXTS_rrra[:, 4, :, :] = 0
        self.NONTRIVIAL_CONTEXTS_rrra[:, :, 4, :] = 0
        self.NONTRIVIAL_CONTEXTS_rrra[:, :, :, 4] = 0
        for base in range(4):
            self.NONTRIVIAL_CONTEXTS_rrra[:, base, :, base] = 0
        self.NUM_NONTRIVIAL_CONTEXTS = 4 * 4 * 4 * 3

    @classmethod
    def initialize_snv_context_totals_rrra(cls, device=gpu_if_available()):
        return torch.zeros(5, 5, 5, 5, device=device)

    @classmethod
    def increment_somatic_snv_context_totals_rrra(cls, snv_totals_rrra, context_totals_rrra, batch: PosteriorBatch,
                                                  posteriors_bc) -> torch.Tensor:
        is_snv = (batch.get_variant_types() == Variation.SNV).float()
        somatic_snv_posteriors_b = posteriors_bc[:, Call.SOMATIC] * is_snv
        idx0, idx1, idx2, idx3 = get_ref_contexts_and_alt_bases(batch)
        add_at_index(tens=snv_totals_rrra, idx=(idx0, idx1, idx2, idx3), values=somatic_snv_posteriors_b)
        add_at_index(tens=context_totals_rrra, idx=(idx0, idx1, idx2, idx3), values=is_snv)

    def somatic_snv_log_priors(self, batch: PosteriorBatch) -> torch.Tensor:
        idx0, idx1, idx2, idx3 = get_ref_contexts_and_alt_bases(batch)
        return index_tensor(self.somatic_snv_log_priors_rrra, (idx0, idx1, idx2, idx3))

    def log_priors_bc(self, batch: PosteriorBatch) -> torch.Tensor:
        variant_types_b = batch.get_variant_types().long()
        allele_frequencies_1d = batch.get_allele_frequencies()
        is_snv = (variant_types_b == Variation.SNV).float()

        # seq error and germline initialized to 0 or -9999 as discussed above
        unnormalized_priors_bc = self._unnormalized_priors_vc[variant_types_b.long(), :]
        unnormalized_priors_bc[:, Call.SEQ_ERROR] = 0
        unnormalized_priors_bc[:, Call.GERMLINE] = -9999 if self.no_germline_mode else torch.log(
            1 - torch.square(1 - allele_frequencies_1d))  # 1 minus hom ref probability

        unnormalized_priors_bc[:, Call.SOMATIC] = is_snv * self.somatic_snv_log_priors(batch) + \
            (1 - is_snv) * unnormalized_priors_bc[:, Call.SOMATIC]
        return torch.nn.functional.log_softmax(unnormalized_priors_bc, dim=-1)

    def update_priors_m_step(self, posterior_totals_vc, somatic_snv_totals_rrra,
                             snv_context_totals_rrra, ignored_to_non_ignored_ratio):
        # update the priors in an EM-style M step.  We'll need the counts of each call type vs variant type
        # We need to correct for all the sites that didn't enter the training data, sites with neither somatic variants
        # nor artifacts.  For example, suppose our entire test dataset in a million-base genome consists of a single
        # somatic variant.  We want to somatic prior to be one in a million, not one in one!
        total_nonignored = torch.sum(posterior_totals_vc).item()
        total_ignored = ignored_to_non_ignored_ratio * total_nonignored
        overall_total = total_ignored + total_nonignored
        # coarse assumption that every context is equally likely
        total_ignored_per_context = total_nonignored * ignored_to_non_ignored_ratio / 64

        with torch.no_grad():
            self._unnormalized_priors_vc.copy_(torch.log(posterior_totals_vc/(posterior_totals_vc + overall_total)))
            self._unnormalized_priors_vc[:, Call.SEQ_ERROR] = 0
            self._unnormalized_priors_vc[:, Call.GERMLINE] = -9999 if self.no_germline_mode else 0

        # shared Beta(alpha, beta) prior on all context-dependent mutation rates.  In the M step for a particular
        # context these act as pseudocounts.
        # TODO: should we initialize this better?
        alpha, beta = torch.tensor([1.1], requires_grad=True, device=self._device), torch.tensor([1.1], requires_grad=True, device=self._device)

        shared_prior_optimizer = torch.optim.Adam([alpha, beta])
        for iteration in range(50):
            # closed form optimization (since shared beta prior is conjugate) of context-dependent priors
            # with shared prior parameters alpha, beta held fixed
            with torch.no_grad():
                self.somatic_snv_log_priors_rrra.copy_(torch.log((snv_context_totals_rrra + alpha - 1) /
                    (snv_context_totals_rrra + total_ignored_per_context + alpha + beta - 2)))

                # make a 1D tensor of the different nontrivial log SNV priors and fit alpha, beta to initialize
                nontrivial_log_priors = torch.exp(self.somatic_snv_log_priors_rrra.flatten()[self.NONTRIVIAL_CONTEXTS_rrra.flatten().nonzero()])
                new_alpha, new_beta, _, _ = stats.beta.fit(nontrivial_log_priors.cpu(), floc=0, fscale=1)
                alpha[0] = new_alpha
                beta[0] = new_beta

            # non-closed form gradient descent optimization of alpha, beta with context-dependent priors held fixed
            # the contribution of the shared prior to the log likelihood is
            # sum_context log Beta(context mutation rate | alpha, beta)
            # = sum_context {-log Beta(alpha, beta) + (alpha - 1) log(rate_contxt) + (beta-1)log(1-rate_context)
            # where the sum is over non-trivial contexts (exclude eg context = AGG, alt base = G)
            # that is
            sum_1 = torch.sum(self.somatic_snv_log_priors_rrra * self.NONTRIVIAL_CONTEXTS_rrra).detach()
            sum_2 = torch.sum(torch.log1p(-self.somatic_snv_log_priors_rrra) * self.NONTRIVIAL_CONTEXTS_rrra).detach()

            for subiteration in range(100):
                log_prob = self.NUM_NONTRIVIAL_CONTEXTS * (torch.lgamma(alpha + beta) - torch.lgamma(alpha) - torch.lgamma(beta)) + \
                           (alpha - 1) * sum_1 + (beta - 1) * sum_2
                backpropagate(shared_prior_optimizer, -log_prob)

    def make_priors_bar_plot(self, snv_context_totals_rrra):
        # bar plot of log priors -- data is indexed by call type name, and x ticks are variant types

        log_prior_bar_plot_data = {call.name : self._unnormalized_priors_vc[:, call].cpu().detach().numpy() \
                for call in [Call.SOMATIC, Call.ARTIFACT, Call.NORMAL_ARTIFACT]}

        somatic_snv_rates_rrra = self.NONTRIVIAL_CONTEXTS_rrra * torch.exp(self.somatic_snv_log_priors_rrra).detach()
        average_somatic_snv_rate = torch.sum(somatic_snv_rates_rrra) / self.NUM_NONTRIVIAL_CONTEXTS
        log_prior_bar_plot_data[Call.SOMATIC.name][Variation.SNV] = torch.log(average_somatic_snv_rate).item()

        prior_fig, prior_ax = plotting.grouped_bar_plot(log_prior_bar_plot_data, [v_type.name for v_type in Variation],
                                                        "log priors")
        return prior_fig, prior_ax

    def make_context_priors_plot(self):
        # the main structure is 4 rows A, C, G, T and 4 columns A, C, G, T defining the ref and alt bases of the SNV
        prior_fig, prior_ax = plt.subplots(4, 4, sharex='all', sharey='all', squeeze=False)
        row_names = ['A', 'C', 'G', 'T']
        col_names = ['A', 'C', 'G', 'T']
        bounds = np.array([0,1,2,3,4])

        common_colormesh = None
        for row, label in enumerate(row_names):
            for col, var_type in enumerate(col_names):
                if row != col:
                    # for each substitution, we have a 2D sub-array for left flank (l) and right flank (r)
                    values_lr = self.somatic_snv_log_priors_rrra[:, row, :, col].detach().cpu()
                    common_colormesh = plotting.color_plot_2d_on_axis(prior_ax[row, col], bounds, bounds, values_lr, None, None)

        prior_fig.colorbar(common_colormesh)
        plotting.tidy_subplots(prior_fig, prior_ax, x_label="alt base", y_label="ref base",
                               row_labels=row_names, column_labels=col_names)
        return prior_fig, prior_ax
