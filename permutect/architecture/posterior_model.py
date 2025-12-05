from collections import defaultdict
from itertools import chain

import torch
from matplotlib import pyplot as plt
from torch import Tensor, IntTensor
from torch.utils.tensorboard import SummaryWriter
from tqdm.autonotebook import trange, tqdm

from permutect.architecture.posterior_model_priors import PosteriorModelPriors
from permutect.architecture.spectra.artifact_spectra import ArtifactSpectra
from permutect.architecture.spectra.normal_artifact_spectrum import NormalArtifactSpectrum
from permutect.architecture.spectra.somatic_spectrum import SomaticSpectrum
from permutect.architecture.spectra.posterior_model_spectra import PosteriorModelSpectra
from permutect.data.datum import DEFAULT_GPU_FLOAT, DEFAULT_CPU_FLOAT
from permutect.data.posterior_data import PosteriorBatch
from permutect.data.prefetch_generator import prefetch_generator
from permutect.metrics import plotting
from permutect.data.count_binning import NUM_ALT_COUNT_BINS, count_from_alt_bin_index, alt_count_bin_index
from permutect.misc_utils import StreamingAverage, gpu_if_available, backpropagate
from permutect.utils.enums import Variation, Call


class PosteriorModel(torch.nn.Module):
    """

    """
    def __init__(self, variant_log_prior: float, artifact_log_prior: float, num_base_features: int, no_germline_mode: bool = False, device=gpu_if_available(), het_beta: float = None):
        super(PosteriorModel, self).__init__()

        self._device = device
        self._dtype = DEFAULT_GPU_FLOAT if device != torch.device("cpu") else DEFAULT_CPU_FLOAT
        self.no_germline_mode = no_germline_mode
        self.num_base_features = num_base_features
        self.het_beta = het_beta

        self.spectra = PosteriorModelSpectra()
        self.priors = PosteriorModelPriors(variant_log_prior, artifact_log_prior, no_germline_mode, device)

        self.to(device=self._device, dtype=self._dtype)

    def posterior_probabilities_bc(self, batch: PosteriorBatch) -> Tensor:
        """
        :param batch:
        :return: non-log probabilities as a 2D tensor, indexed by batch 'b', Call type 'c'
        """
        return torch.nn.functional.softmax(self.log_relative_posteriors_bc(batch), dim=1)

    def error_probabilities_b(self, batch: PosteriorBatch, germline_mode: bool = False) -> Tensor:
        """
        :param germline_mode: if True, germline classification is not considered an error mode
        :param batch:
        :return: non-log error probabilities as a 1D tensor with length batch size
        """
        assert not (germline_mode and self.no_germline_mode), "germline mode and no-germline mode are incompatible"
        return 1 - self.posterior_probabilities_bc(batch)[:, Call.GERMLINE if germline_mode else Call.SOMATIC]     # 0th column is variant

    def log_posterior_and_ingredients(self, batch: PosteriorBatch) -> Tensor:
        """
        :param batch:
        :batch.seq_error_log_likelihoods() is the probability that these *particular* reads exhibit the alt allele given a
        sequencing error ie an error explainable in terms of base qualities.  For example if we have two alt reads with error
        probability of 0.1 and 0.2, and two ref reads with error probabilities 0.05 and 0.06 this quantity would be
        log(0.1*0.2*0.95*0.94).  This is an annotation emitted by the GATK and by the time it reaches here is a 1D tensor
        of length batch_size.
        :return:
        """
        var_types_b = batch.get_variant_types()

        # All log likelihood/relative posterior tensors below have shape batch.size() x len(CallType)
        # spectra tensors contain the likelihood that these *particular* reads (that is, not just the read count) are alt
        # normal log likelihoods contain everything going on in the matched normal sample
        log_priors_bc = self.priors.log_priors_bc(var_types_b, batch.get_allele_frequencies())
        spectra_log_lks_bc, normal_log_lks_bc = self.spectra.spectra_log_likelihoods_bc(batch)

        log_posteriors_bc = log_priors_bc + spectra_log_lks_bc + normal_log_lks_bc
        log_posteriors_bc[:, Call.ARTIFACT] += batch.get_artifact_logits()
        log_posteriors_bc[:, Call.NORMAL_ARTIFACT] += batch.get_artifact_logits()

        # TODO: HACK / EXPERIMENT: make it impossible to call an artifact when the artifact logits are negative
        log_posteriors_bc[:, Call.ARTIFACT] = torch.where(batch.get_artifact_logits() < 0, -9999, log_posteriors_bc[:, Call.ARTIFACT])
        # TODO: END OF HACK / EXPERIMENT

        return log_priors_bc, spectra_log_lks_bc, normal_log_lks_bc, log_posteriors_bc

    def log_relative_posteriors_bc(self, batch: PosteriorBatch) -> Tensor:
        _, _, _, log_posteriors_bc = self.log_posterior_and_ingredients(batch)
        return log_posteriors_bc

    def learn_priors_and_spectra(self, posterior_loader, num_iterations, ignored_to_non_ignored_ratio: float,
                                 summary_writer: SummaryWriter = None, learning_rate: float = 0.001):
        """
        :param summary_writer:
        :param num_iterations:
        :param posterior_loader:
        :param ignored_to_non_ignored_ratio: ratio of sites in which no evidence of variation was found to sites in which
        sufficient evidence was found to emit test data.  Without this parameter (i.e. if it were set to zero) we would
        underestimate the frequency of sequencing error, hence overestimate the prior probability of variation.
        """
        spectra_params = chain(self.spectra.parameters())
        optimizer = torch.optim.Adam(spectra_params, lr=learning_rate)

        for epoch in trange(1, num_iterations + 1, desc="AF spectra epoch"):
            epoch_loss = StreamingAverage()

            # store posteriors as a list (to be stacked at the end of the epoch) for an M step
            # 'l' for loader, 'b' for batch, 'c' for call type
            posteriors_lbc = []
            alt_counts_lb = []
            depths_lb = []
            types_lb = []

            batch: PosteriorBatch
            for batch in tqdm(prefetch_generator(posterior_loader), mininterval=10, total=len(posterior_loader)):
                relative_posteriors = self.log_relative_posteriors_bc(batch)
                log_evidence = torch.logsumexp(relative_posteriors, dim=1)

                posteriors_lbc.append(torch.softmax(relative_posteriors, dim=-1).detach())
                alt_counts_lb.append(batch.get_alt_counts().detach())
                depths_lb.append(batch.get_original_depths().detach())
                types_lb.append(batch.get_variant_types().detach())

                # confidence_mask = torch.logical_or(batch.get_artifact_logits() < 0, batch.get_artifact_logits() > 3)
                loss = -torch.mean(log_evidence)
                #loss = - torch.sum(confidence_mask * log_evidence) / (torch.sum(confidence_mask) + 0.000001)

                # note that we don't multiply by batch size because we take the mean of log evidence above
                # however, we must sum over variant types since each ignored site is simultaneously a missing non-SNV,
                # a missing non-INSERTION etc
                # we use a germline allele frequency of 0.001 for the missing sites but it doesn't really matter
                for var_type_idx, variant_type in enumerate(Variation):
                    log_priors = self.priors.log_priors_bc(torch.LongTensor([var_type_idx]).to(device=self._device, dtype=self._dtype), torch.tensor([0.001], device=self._device))
                    log_seq_error_prior = log_priors.squeeze()[Call.SEQ_ERROR]
                    missing_loss = -ignored_to_non_ignored_ratio * log_seq_error_prior  
                    loss += missing_loss

                backpropagate(optimizer, loss)

                epoch_loss.record_sum(batch.size() * loss.detach().item(), batch.size())
            # iteration over posterior dataloader finished

            # 'n' denotes index of data within entire Posterior Dataset
            posteriors_nc = torch.vstack(posteriors_lbc)
            alt_counts_n = torch.hstack(alt_counts_lb)
            depths_n = torch.hstack(depths_lb)
            types_n = torch.hstack(types_lb)

            self.priors.update_priors_m_step(posteriors_nc, types_n, ignored_to_non_ignored_ratio)
            # TODO: fix the M step for the new somatic spectrum?
            #self.somatic_spectrum.update_m_step(posteriors_nc[:, Call.SOMATIC], alt_counts_n, depths_n)

            if summary_writer is not None:
                summary_writer.add_scalar("spectrum negative log evidence", epoch_loss.get(), epoch)

                for depth in [9, 19, 30, 50, 100]:
                    art_spectra_fig, art_spectra_axs = self.spectra.artifact_spectra.plot_artifact_spectra(depth)
                    summary_writer.add_figure("Artifact AF Spectra at depth = " + str(depth), art_spectra_fig, epoch)

                #normal_artifact_spectra_fig, normal_artifact_spectra_axs = plot_artifact_spectra(self.normal_artifact_spectra)
                #summary_writer.add_figure("Normal Artifact AF Spectra", normal_artifact_spectra_fig, epoch)

                var_spectra_fig, var_spectra_axs = plt.subplots()
                frac, dens = self.spectra.somatic_spectrum.spectrum_density_vs_fraction()
                var_spectra_axs.plot(frac.detach().numpy(), dens.detach().numpy(), label="spectrum")
                var_spectra_axs.set_title("Variant AF Spectrum")
                summary_writer.add_figure("Variant AF Spectra", var_spectra_fig, epoch)

                # bar plot of log priors -- data is indexed by call type name, and x ticks are variant types
                log_prior_bar_plot_data = defaultdict(list)
                for var_type_idx, variant_type in enumerate(Variation):
                    log_priors = self.priors.log_priors_bc(torch.LongTensor([var_type_idx]).to(device=self._device, dtype=self._dtype), torch.tensor([0.001]))
                    log_priors_cpu = log_priors.squeeze().detach().cpu()
                    for call_type in (Call.SOMATIC, Call.ARTIFACT, Call.NORMAL_ARTIFACT):
                        log_prior_bar_plot_data[call_type.name].append(log_priors_cpu[call_type])

                prior_fig, prior_ax = plotting.grouped_bar_plot(log_prior_bar_plot_data, [v_type.name for v_type in Variation], "log priors")
                summary_writer.add_figure("log priors", prior_fig, epoch)

                # normal artifact joint tumor-normal spectra
                # na_fig, na_axes = plt.subplots(1, len(Variation), sharex='all', sharey='all', squeeze=False)
                # for variant_index, variant_type in enumerate(Variation):
                #    self.normal_artifact_spectra[variant_index].density_plot_on_axis(na_axes[0, variant_index])
                # plotting.tidy_subplots(na_fig, na_axes, x_label="tumor fraction", y_label="normal fraction",
                #                       row_labels=[""], column_labels=[var_type.name for var_type in Variation])
                # summary_writer.add_figure("normal artifact spectra", na_fig, epoch)

    # map of Variant type to probability threshold that maximizes F1 score
    # loader is a Dataloader whose collate_fn is the PosteriorBatch constructor
    def calculate_probability_thresholds(self, posterior_loader, summary_writer: SummaryWriter = None, germline_mode: bool = False):
        self.train(False)
        error_probs_by_type = {var_type: [] for var_type in Variation}   # includes both artifact and seq errors

        error_probs_by_type_by_cnt = {var_type: [[] for _ in range(NUM_ALT_COUNT_BINS)] for var_type in Variation}

        # TODO: use the EvaluationMetrics class to generate the theoretical ROC curve
        # TODO: then delete plotting.plot_theoretical_roc_on_axis
        batch: PosteriorBatch
        for batch in tqdm(prefetch_generator(posterior_loader), mininterval=10, total=len(posterior_loader)):
            # TODO: should this be the original alt counts instead?
            alt_counts_b = batch.get_alt_counts().cpu().tolist()
            # 0th column is true variant, subtract it from 1 to get error prob
            error_probs_b = self.error_probabilities_b(batch, germline_mode).cpu().tolist()

            for var_type, alt_count, error_prob in zip(batch.get_variant_types().cpu().tolist(), alt_counts_b, error_probs_b):
                error_probs_by_type[var_type].append(error_prob)
                error_probs_by_type_by_cnt[var_type][alt_count_bin_index(alt_count)].append(error_prob)

        thresholds_by_type = {}
        roc_fig, roc_axes = plt.subplots(1, len(Variation), sharex='all', sharey='all', squeeze=False)
        roc_by_cnt_fig, roc_by_cnt_axes = plt.subplots(1, len(Variation), sharex='all', sharey='all', squeeze=False, figsize=(10, 6), dpi=100)
        for var_type in Variation:
            # plot all count ROC curves for this variant type
            count_bin_labels = [str(count_from_alt_bin_index(count_bin)) for count_bin in range(NUM_ALT_COUNT_BINS)]
            _ = plotting.plot_theoretical_roc_on_axis(error_probs_by_type_by_cnt[var_type], count_bin_labels, roc_by_cnt_axes[0, var_type])
            best_threshold = plotting.plot_theoretical_roc_on_axis([error_probs_by_type[var_type]], [""], roc_axes[0, var_type])[0][0]

            # TODO: the theoretical ROC might need to return the best threshold for this
            thresholds_by_type[var_type] = best_threshold

        variation_types = [var_type.name for var_type in Variation]
        plotting.tidy_subplots(roc_by_cnt_fig, roc_by_cnt_axes, x_label="sensitivity", y_label="precision",
                               row_labels=[""], column_labels=variation_types)
        plotting.tidy_subplots(roc_fig, roc_axes, x_label="sensitivity", y_label="precision",
                               row_labels=[""], column_labels=variation_types)
        if summary_writer is not None:
            summary_writer.add_figure("theoretical ROC by variant type ", roc_fig)
            summary_writer.add_figure("theoretical ROC by variant type and alt count ", roc_by_cnt_fig)

        return thresholds_by_type
