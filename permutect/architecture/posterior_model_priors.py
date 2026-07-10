import itertools

import numpy as np
import pymc as pm
import torch
from matplotlib import pyplot as plt
from torch import Tensor
from torch import nn
from torch.nn import Parameter

from permutect.data.batch import Batch
from permutect.data.datum import Data
from permutect.metrics import plotting
from permutect.misc_utils import gpu_if_available
from permutect.parameters import PosteriorModelParameters
from permutect.utils.array_utils import add_at_index
from permutect.utils.array_utils import index_tensor
from permutect.utils.enums import Call
from permutect.utils.enums import Variation


def get_ref_contexts_and_alt_bases(batch: Batch) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    # each row is the ref sequence, followed by the alt sequence (hstacked) with A = 0, C = 1 . . . deletion = 4
    haplotypes_bs = batch.get_haplotypes_bs()
    seq_length = haplotypes_bs.shape[-1] // 2
    ref_center_idx = (seq_length - 1) // 2
    alt_center_idx = ref_center_idx + seq_length
    idx0 = haplotypes_bs[:, ref_center_idx - 1]
    idx1 = haplotypes_bs[:, ref_center_idx].int()
    idx2 = haplotypes_bs[:, ref_center_idx + 1]
    idx3 = haplotypes_bs[:, alt_center_idx]
    return idx0, idx1, idx2, idx3


def convert_rrra_tensor_to_sc(input_rrra: torch.Tensor) -> torch.Tensor:
    """
    Input is a tensor indexed by the 3-base ref sequence rrr = (left flank, ref base of substitution, right flank)
    and alt base a, including the non-SNV index 4 for deletion and the unused elements where ref and alt bases
    are the same.  Output is indexed by 's' the flattened substitution taking on 4x3=12 values and by 'c', the
    flattened 4x4=16 values of the two flanking bases.
    """
    # step 1: left flank, right flank, ref base, alt base and discard non-SNV index 4
    tensor_ffra = torch.transpose(input_rrra, 1, 2)[0:4, 0:4, 0:4, 0:4]

    # step 2: flatten the flanking bases 'ff' into a single dimension 'c'
    tensor_cra = torch.flatten(tensor_ffra, start_dim=0, end_dim=1)

    # step 3: flatten the ref base 'r' and alt base 'a' into a single dimension 's' and omit r=a
    nontrivial_s_indices = torch.IntTensor([1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14])
    tensor_cs = torch.flatten(tensor_cra, start_dim=1, end_dim=2)[:, nontrivial_s_indices]

    # step 4: switch s and c
    return torch.transpose(tensor_cs, 0, 1)


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

    def __init__(self, posterior_params: PosteriorModelParameters, device=gpu_if_available()):
        super(PosteriorModelPriors, self).__init__()
        self.no_germline_mode = posterior_params.no_germline_mode
        self._device = device
        self.use_context_dependent_snv_priors = True

        variant_log_prior = posterior_params.initial_log_variant_prior
        artifact_log_prior = posterior_params.intial_log_artifact_prior

        # pre-softmax priors of different call types [log P(variant), log P(artifact), log P(seq error)] for each variant type
        # although these vectors are defined for all variant types, somatic SNVs are handled separately
        self.log_priors_vc = torch.nn.Parameter(torch.zeros(len(Variation), len(Call)))
        with torch.no_grad():
            self.log_priors_vc[:, Call.SOMATIC] = variant_log_prior
            self.log_priors_vc[:, Call.ARTIFACT] = artifact_log_prior
            self.log_priors_vc[:, Call.GERMLINE] = -9999 if self.no_germline_mode else 0
            self.log_priors_vc[:, Call.NORMAL_ARTIFACT] = artifact_log_prior

        # context and substitution dependent (the 3 r's stand for ref context, the 'a' is for alt base)
        # we have 5 values just in case there is a D in the ref context (an indel next to a SNV, perhaps)
        self.somatic_snv_log_priors_rrra = Parameter(variant_log_prior * torch.ones((5, 5, 5, 5), device=self._device))

        self.NONTRIVIAL_CONTEXTS_rrra = torch.ones((5, 5, 5, 5), device=self._device)
        self.NONTRIVIAL_CONTEXTS_rrra[4, :, :, :] = 0
        self.NONTRIVIAL_CONTEXTS_rrra[:, 4, :, :] = 0
        self.NONTRIVIAL_CONTEXTS_rrra[:, :, 4, :] = 0
        self.NONTRIVIAL_CONTEXTS_rrra[:, :, :, 4] = 0
        for base in range(4):
            self.NONTRIVIAL_CONTEXTS_rrra[:, base, :, base] = 0
        self.NUM_NONTRIVIAL_CONTEXTS = 4 * 4 * 4 * 3

    def enable_context_dependent_snv_priors(self) -> None:
        self.use_context_dependent_snv_priors = True

    def disable_context_dependent_snv_priors(self) -> None:
        self.use_context_dependent_snv_priors = False

    @classmethod
    def initialize_snv_context_totals_rrra(cls, device=gpu_if_available()):
        return torch.zeros(5, 5, 5, 5, device=device)

    @classmethod
    def increment_somatic_snv_context_totals_rrra(
        cls, snv_totals_rrra, context_totals_rrra, batch: Batch, posteriors_bc
    ):
        is_snv = (batch.get(Data.VARIANT_TYPE) == Variation.SNV).float()
        somatic_snv_posteriors_b = posteriors_bc[:, Call.SOMATIC] * is_snv
        idx0, idx1, idx2, idx3 = get_ref_contexts_and_alt_bases(batch)
        add_at_index(tens=snv_totals_rrra, idx=(idx0, idx1, idx2, idx3), values=somatic_snv_posteriors_b)
        add_at_index(tens=context_totals_rrra, idx=(idx0, idx1, idx2, idx3), values=is_snv)

    def somatic_snv_log_priors(self, batch: Batch) -> torch.Tensor:
        idx0, idx1, idx2, idx3 = get_ref_contexts_and_alt_bases(batch)
        return index_tensor(self.somatic_snv_log_priors_rrra, (idx0, idx1, idx2, idx3))

    def log_priors_bc(self, batch: Batch) -> torch.Tensor:
        variant_types_b = batch.get(Data.VARIANT_TYPE).long()
        allele_frequencies_b = batch.get(Data.ALLELE_FREQUENCY)
        is_snv_b = (variant_types_b == Variation.SNV).float()

        # seq error and germline initialized to 0 or -9999 as discussed above
        log_priors_bc = self.log_priors_vc[variant_types_b.long(), :]
        log_priors_bc[:, Call.SEQ_ERROR] = 0
        log_priors_bc[:, Call.GERMLINE] = (
            -9999 if self.no_germline_mode else torch.log(1 - torch.square(1 - allele_frequencies_b))
        )  # 1 minus hom ref probability

        if self.use_context_dependent_snv_priors:
            log_priors_bc[:, Call.SOMATIC] = (
                is_snv_b * self.somatic_snv_log_priors(batch) + (1 - is_snv_b) * log_priors_bc[:, Call.SOMATIC]
            )
        return torch.nn.functional.log_softmax(log_priors_bc, dim=-1)

    def update_priors_m_step(
        self,
        posterior_totals_vc,
        somatic_snv_totals_rrra,
        snv_context_totals_rrra,
        ignored_to_non_ignored_ratio,
    ):
        # update the priors in an EM-style M step.  We'll need the counts of each call type vs variant type
        # We need to correct for all the sites that didn't enter the training data, sites with neither somatic variants
        # nor artifacts.  For example, suppose our entire test dataset in a million-base genome consists of a single
        # somatic variant.  We want to somatic prior to be one in a million, not one in one!
        total_nonignored = torch.sum(posterior_totals_vc).item()
        total_ignored = ignored_to_non_ignored_ratio * total_nonignored
        overall_total = total_ignored + total_nonignored
        # coarse assumption that every context is equally likely
        total_ignored_per_context = total_ignored / 64

        with torch.no_grad():
            self.log_priors_vc.copy_(torch.log(posterior_totals_vc / (posterior_totals_vc + overall_total)))
            self.log_priors_vc[:, Call.SEQ_ERROR] = 0
            self.log_priors_vc[:, Call.GERMLINE] = -9999 if self.no_germline_mode else 0

        if self.use_context_dependent_snv_priors:
            total_sc = torch.round(convert_rrra_tensor_to_sc(snv_context_totals_rrra) + total_ignored_per_context).int()
            snv_sc = torch.round(convert_rrra_tensor_to_sc(somatic_snv_totals_rrra)).int()

            with pm.Model() as _mutation_rate_model:
                # NOTE: indices here are i) 's' for the 4x3=12 different substitutions, flattened so that
                # A->C = 0, A->G = 1. . .T->G = 11 and ii) 'c' for 4x4=16 different two bases of flanking
                # context, also flattened into one dimension rather than two.

                # overall mutation rate multiplier.  Approximately the overall average mutation rate.
                overall_rate = pm.Beta("overall_rate", alpha=1.0, beta=1e6)

                # concentration parameter for symmetric Dirichlet for relative mutation rates over the 4x3 = 12
                # different possible substitutions
                substitution_concentration = pm.Gamma("substitution_concentration", alpha=4.0, beta=5.0)
                concentration_s = pm.math.ones(shape=(12,)) * substitution_concentration
                theta_s = pm.Dirichlet("theta_s", a=concentration_s)

                # cocnentration parameter for symmetric Dirichlets for relative mutation rates over the 4x4 = 16
                # different possible two bases of flanking context within each substitution. Note that while we have
                # a single global concentration for all substitutions, within each substitution type there is an
                # independent Dirichlet random variable for the relative weights of different contexts.
                context_concentration = pm.Gamma("context_concentration", alpha=4.0, beta=5.0)
                concentration_c = pm.math.ones(shape=(16,)) * context_concentration
                theta_sc = pm.Dirichlet("theta_sc", a=concentration_c, shape=(12, 16))

                rate_sc = pm.Deterministic("rate_sc", overall_rate * 12 * theta_s.reshape((12, 1)) * 16 * theta_sc)
                _outcome = pm.Binomial(
                    "outcome",
                    n=total_sc.cpu().numpy().flatten(),
                    p=rate_sc.flatten(),
                    observed=snv_sc.cpu().numpy().flatten(),
                )

                mean_field = pm.fit(method="advi")

                idata = mean_field.sample(1000)

                mutation_rates_sc = np.mean(
                    idata.posterior["rate_sc"], axis=(0, 1)
                )  # axis 0 is the different MCMC samplers, axis 1 is the MCMC step
                mutation_rates_sc = torch.from_numpy(mutation_rates_sc.to_numpy())

                with torch.no_grad():
                    # lf, rf are left and right flanking bases, ref and alt are ref and alt bases of the substitution.
                    for lf, rf in itertools.product(range(4), range(4)):
                        for ref, alt in itertools.product(range(4), range(4)):
                            if ref != alt:
                                context = lf * 4 + rf
                                substitution_with_trivial = ref * 4 + alt
                                substitution = substitution_with_trivial - (substitution_with_trivial // 5) - 1
                                self.somatic_snv_log_priors_rrra[lf, ref, rf, alt] = torch.log(
                                    mutation_rates_sc[substitution, context]
                                )
        else:  # if not using context-dependent SNV priors
            with torch.no_grad():
                self.somatic_snv_log_priors_rrra.fill_(self.log_priors_vc[Variation.SNV, Call.SOMATIC])

    def make_priors_bar_plot(self, snv_context_totals_rrra):
        # bar plot of log priors -- data is indexed by call type name, and x ticks are variant types

        log_prior_bar_plot_data = {
            call.name: self.log_priors_vc[:, call].cpu().detach().numpy()
            for call in [Call.SOMATIC, Call.ARTIFACT, Call.NORMAL_ARTIFACT]
        }

        somatic_snv_rates_rrra = self.NONTRIVIAL_CONTEXTS_rrra * torch.exp(self.somatic_snv_log_priors_rrra).detach()
        average_somatic_snv_rate = torch.sum(somatic_snv_rates_rrra) / self.NUM_NONTRIVIAL_CONTEXTS
        log_prior_bar_plot_data[Call.SOMATIC.name][Variation.SNV] = torch.log(average_somatic_snv_rate).item()

        prior_fig, prior_ax = plotting.grouped_bar_plot(
            log_prior_bar_plot_data, [v_type.name for v_type in Variation], "log priors"
        )
        return prior_fig, prior_ax

    def make_context_priors_plot(self):
        # the main structure is 4 rows A, C, G, T and 4 columns A, C, G, T defining the ref and alt bases of the SNV
        prior_fig, prior_ax = plt.subplots(4, 4, sharex="all", sharey="all", squeeze=False)
        row_names = ["A", "C", "G", "T"]
        col_names = ["A", "C", "G", "T"]
        bounds = np.array([0, 1, 2, 3, 4])

        common_colormesh = None
        for row, label in enumerate(row_names):
            for col, var_type in enumerate(col_names):
                if row != col:
                    # for each substitution, we have a 2D sub-array for left flank (l) and right flank (r)
                    values_lr = self.somatic_snv_log_priors_rrra[:, row, :, col].detach().cpu()
                    common_colormesh = plotting.color_plot_2d_on_axis(
                        prior_ax[row, col], bounds, bounds, values_lr, None, None
                    )

        prior_fig.colorbar(common_colormesh)
        plotting.tidy_subplots(
            prior_fig,
            prior_ax,
            x_label="alt base",
            y_label="ref base",
            row_labels=row_names,
            col_labels=col_names,
        )
        return prior_fig, prior_ax
