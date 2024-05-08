from collections import defaultdict
from itertools import chain
from math import ceil

import torch
from intervaltree import IntervalTree
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from tqdm.autonotebook import trange, tqdm

from permutect import utils
from permutect.architecture.beta_binomial_mixture import BetaBinomialMixture, FeaturelessBetaBinomialMixture
from permutect.architecture.normal_seq_error_spectrum import NormalSeqErrorSpectrum
from permutect.data.posterior import PosteriorBatch
from permutect.metrics import plotting
from permutect.utils import Variation, Call
from permutect.metrics.evaluation_metrics import MAX_COUNT, NUM_COUNT_BINS, multiple_of_three_bin_index, multiple_of_three_bin_index_to_count

HOM_ALPHA, HOM_BETA = torch.Tensor([98.0]), torch.Tensor([2.0])


# TODO: write unit test asserting that this comes out to zero when counts are zero
# given germline, the probability of these particular reads being alt
def germline_log_likelihood(afs, mafs, alt_counts, ref_counts):
    het_probs = 2 * afs * (1 - afs)
    hom_probs = afs * afs
    het_proportion = het_probs / (het_probs + hom_probs)
    hom_proportion = 1 - het_proportion

    log_mafs = torch.log(mafs)
    log_1m_mafs = torch.log(1 - mafs)
    log_half_het_prop = torch.log(het_proportion / 2)

    depths = alt_counts + ref_counts

    combinatorial_term = torch.lgamma(depths + 1) - torch.lgamma(alt_counts + 1) - torch.lgamma(ref_counts + 1)
    # the following should both be 1D tensors of length batch size
    alt_minor_ll = combinatorial_term + log_half_het_prop + alt_counts * log_mafs + ref_counts * log_1m_mafs
    alt_major_ll = combinatorial_term + log_half_het_prop + ref_counts * log_mafs + alt_counts * log_1m_mafs
    hom_ll = torch.log(hom_proportion) + utils.beta_binomial(depths, alt_counts, HOM_ALPHA, HOM_BETA)

    return torch.logsumexp(torch.vstack((alt_minor_ll, alt_major_ll, hom_ll)), dim=0)


def initialize_artifact_spectra():
    return BetaBinomialMixture(input_size=len(Variation), num_components=5)


# TODO: max_mean is hard-coded magic constant!!
def initialize_normal_artifact_spectra():
    return BetaBinomialMixture(input_size=len(Variation), num_components=1, max_mean=0.1)


def plot_artifact_spectra(artifact_spectra: BetaBinomialMixture):
    # plot AF spectra in two-column grid with as many rows as needed
    art_spectra_fig, art_spectra_axs = plt.subplots(ceil(len(Variation) / 2), 2, sharex='all', sharey='all')
    for variant_type in Variation:
        n = variant_type
        row, col = int(n / 2), n % 2
        frac, dens = artifact_spectra.spectrum_density_vs_fraction(
            torch.from_numpy(variant_type.one_hot_tensor()).float())
        art_spectra_axs[row, col].plot(frac.numpy(), dens.numpy())
        art_spectra_axs[row, col].set_title(variant_type.name + " artifact AF spectrum")
    for ax in art_spectra_fig.get_axes():
        ax.label_outer()
    return art_spectra_fig, art_spectra_axs


class PosteriorModel(torch.nn.Module):
    """

    """
    def __init__(self, variant_log_prior: float, artifact_log_prior: float, segmentation=defaultdict(IntervalTree),
                 normal_segmentation=defaultdict(IntervalTree), no_germline_mode: bool = False):
        super(PosteriorModel, self).__init__()

        self.no_germline_mode = no_germline_mode

        # TODO: might as well give the normal segmentation as well
        self.segmentation = segmentation
        self.normal_segmentation = normal_segmentation

        # TODO introduce parameters class so that num_components is not hard-coded
        # featureless because true variant types share a common AF spectrum
        self.somatic_spectrum = FeaturelessBetaBinomialMixture(num_components=5)

        # artifact spectra for each variant type.  Variant type encoded as one-hot input vector.
        self.artifact_spectra = initialize_artifact_spectra()

        # normal sequencing error spectra for each variant type.
        self.normal_seq_error_spectra = torch.nn.ModuleList([NormalSeqErrorSpectrum(num_samples=50, max_mean=0.001) for _ in Variation])

        self.normal_artifact_spectra = initialize_normal_artifact_spectra()

        # TODO: num_samples is hard-coded magic constant!!!
        # self.normal_artifact_spectra = torch.nn.ModuleList([NormalArtifactSpectrum(num_samples=20) for variant_type in Variation])

        # pre-softmax priors [log P(variant), log P(artifact), log P(seq error)] for each variant type
        # linear layer with no bias to select the appropriate priors given one-hot variant encoding
        # in torch linear layers the weight is indexed by out-features, then in-features, so indices are
        # self.unnormalized_priors.weight[call type, variant type]
        self._unnormalized_priors = torch.nn.Linear(in_features=len(Variation), out_features=len(Call), bias=False)
        with torch.no_grad():
            initial = torch.zeros_like(self._unnormalized_priors.weight)
            # the following assignments are broadcast over rows; that is, each variant type gets the same prior
            initial[Call.SOMATIC] = variant_log_prior
            initial[Call.ARTIFACT] = artifact_log_prior
            initial[Call.SEQ_ERROR] = 0
            initial[Call.GERMLINE] = -9999 if self.no_germline_mode else 0
            initial[Call.NORMAL_ARTIFACT] = artifact_log_prior
            self._unnormalized_priors.weight.copy_(initial)

    def make_unnormalized_priors(self, variant_types_one_hot_2d: torch.Tensor, allele_frequencies_1d: torch.Tensor) -> torch.Tensor:
        result = self._unnormalized_priors(variant_types_one_hot_2d)
        result[:, Call.SEQ_ERROR] = 0
        result[:, Call.GERMLINE] = -9999 if self.no_germline_mode else torch.log(1 - torch.square(1 - allele_frequencies_1d))     # 1 minus hom ref probability
        return result   # batch size x len(CallType)

    def posterior_probabilities(self, batch: PosteriorBatch) -> torch.Tensor:
        """
        :param batch:
        :return: non-log probabilities as a 2D tensor, 1st index is batch, 2nd is variant/artifact/seq error
        """
        return torch.nn.functional.softmax(self.log_relative_posteriors(batch), dim=1)

    def error_probabilities(self, batch: PosteriorBatch, germline_mode: bool = False) -> torch.Tensor:
        """
        :param germline_mode: if True, germline classification is not considered an error mode
        :param batch:
        :return: non-log error probabilities as a 1D tensor with length batch size
        """
        assert not (germline_mode and self.no_germline_mode), "germline mode and no-germline mode are incompatible"
        return 1 - self.posterior_probabilities(batch)[:, Call.GERMLINE if germline_mode else Call.SOMATIC]     # 0th column is variant

    def log_posterior_and_ingredients(self, batch: PosteriorBatch) -> torch.Tensor:
        """
        :param batch:
        :batch.seq_error_log_likelihoods() is the probability that these *particular* reads exhibit the alt allele given a
        sequencing error ie an error explainable in terms of base qualities.  For example if we have two alt reads with error
        probability of 0.1 and 0.2, and two ref reads with error probabilities 0.05 and 0.06 this quantity would be
        log(0.1*0.2*0.95*0.94).  This is an annotation emitted by the GATK and by the time it reaches here is a 1D tensor
        of length batch_size.
        :return:
        """
        types = batch.variant_type_one_hot()

        # All log likelihood/relative posterior tensors below have shape batch.size() x len(CallType)
        # spectra tensors contain the likelihood that these *particular* reads (that is, not just the read count) are alt
        # normal log likelihoods contain everything going on in the matched normal sample
        # note that the call to make_unnormalized_priors ensures that no_germline_mode works
        log_priors = torch.nn.functional.log_softmax(self.make_unnormalized_priors(types, batch.allele_frequencies), dim=1)

        # defined as log [ int_0^1 Binom(alt count | depth, f) df ], including the combinatorial N choose N_alt factor
        flat_prior_spectra_log_likelihoods = -torch.log(batch.depths + 1)
        somatic_spectrum_log_likelihoods = self.somatic_spectrum.forward(batch.depths, batch.alt_counts)
        tumor_artifact_spectrum_log_likelihood = self.artifact_spectra.forward(types, batch.depths, batch.alt_counts)

        spectra_log_likelihoods = torch.zeros_like(log_priors)

        # essentially, this corrects the TLOD from M2, computed with a flat prior, to account for the precises somatic spectrum
        spectra_log_likelihoods[:, Call.SOMATIC] = somatic_spectrum_log_likelihoods - flat_prior_spectra_log_likelihoods
        spectra_log_likelihoods[:, Call.ARTIFACT] = tumor_artifact_spectrum_log_likelihood - flat_prior_spectra_log_likelihoods
        spectra_log_likelihoods[:, Call.NORMAL_ARTIFACT] = tumor_artifact_spectrum_log_likelihood - flat_prior_spectra_log_likelihoods # yup, it's the same spectrum
        spectra_log_likelihoods[:, Call.SEQ_ERROR] = -batch.tlods_from_m2
        # spectra_log_likelihoods[:, Call.GERMLINE] is computed below

        normal_log_likelihoods = torch.zeros_like(log_priors)
        normal_seq_error_log_likelihoods = torch.zeros_like(batch.alt_counts).float()

        for var_index, _ in enumerate(Variation):
            mask = types[:, var_index]
            log_likelihoods_for_this_type = self.normal_seq_error_spectra[var_index].forward(batch.normal_alt_counts, batch.normal_ref_counts())
            normal_seq_error_log_likelihoods += mask * log_likelihoods_for_this_type

        normal_log_likelihoods[:, Call.SOMATIC] = normal_seq_error_log_likelihoods
        normal_log_likelihoods[:, Call.ARTIFACT] = normal_seq_error_log_likelihoods
        normal_log_likelihoods[:, Call.SEQ_ERROR] = normal_seq_error_log_likelihoods

        no_alt_in_normal_mask = batch.normal_alt_counts < 1
        normal_log_likelihoods[:, Call.NORMAL_ARTIFACT] = -9999 * no_alt_in_normal_mask + \
            torch.logical_not(no_alt_in_normal_mask) * self.normal_artifact_spectra.forward(types, batch.normal_depths, batch.normal_alt_counts)

        # since this is a default dict, if there's no segmentation for the contig we will get no overlaps but not an error
        # In our case there is either one or zero overlaps, and overlaps have the form
        segmentation_overlaps = [self.segmentation[item.contig][item.position] for item in batch.original_list()]
        normal_segmentation_overlaps = [self.normal_segmentation[item.contig][item.position] for item in batch.original_list()]
        mafs = torch.Tensor([list(overlaps)[0].data if overlaps else 0.5 for overlaps in segmentation_overlaps])
        normal_mafs = torch.Tensor([list(overlaps)[0].data if overlaps else 0.5 for overlaps in normal_segmentation_overlaps])
        afs = batch.allele_frequencies
        spectra_log_likelihoods[:, Call.GERMLINE] = germline_log_likelihood(afs, mafs, batch.alt_counts, batch.ref_counts()) - flat_prior_spectra_log_likelihoods

        # it is correct not to subtract the flat prior likelihood from the normal term because this is an absolute likelihood, not
        # relative to seq error as the M2 TLOD is defined
        normal_log_likelihoods[:, Call.GERMLINE] = germline_log_likelihood(afs, normal_mafs, batch.normal_alt_counts, batch.normal_ref_counts())

        log_posteriors = log_priors + spectra_log_likelihoods + normal_log_likelihoods
        log_posteriors[:, Call.ARTIFACT] += batch.artifact_logits

        # we hedge our bets and average the two possibilities for a normal artifact: 1), that the tumor does not exhibit
        # an artifact, and 2), that it does
        log_normal_artifact_with_tumor_artifact = log_posteriors[:, Call.NORMAL_ARTIFACT] + batch.artifact_logits
        log_normal_artifact_without_tumor_artifact = log_posteriors[:, Call.NORMAL_ARTIFACT]
        log_posteriors[:, Call.NORMAL_ARTIFACT] = torch.logsumexp(torch.vstack((log_normal_artifact_with_tumor_artifact, log_normal_artifact_without_tumor_artifact)), dim=0) - torch.log(torch.Tensor([2]))

        return log_priors, spectra_log_likelihoods, normal_log_likelihoods, log_posteriors

    def log_relative_posteriors(self, batch: PosteriorBatch) -> torch.Tensor:
        _, _, _, log_posteriors = self.log_posterior_and_ingredients(batch)
        return log_posteriors

    def learn_priors_and_spectra(self, posterior_loader, num_iterations, ignored_to_non_ignored_ratio: float,
                                 summary_writer: SummaryWriter = None, artifact_log_priors=None, artifact_spectra_state_dict=None):
        """
        :param summary_writer:
        :param num_iterations:
        :param posterior_loader:
        :param ignored_to_non_ignored_ratio: ratio of sites in which no evidence of variation was found to sites in which
        sufficient evidence was found to emit test data.  Without this parameter (i.e. if it were set to zero) we would
        underestimate the frequency of sequencing error, hence overestimate the prior probability of variation.
        :param artifact_spectra_state_dict: (possibly None) if given, pretrained parameters of self.artifact_spectra
        from train_model.py.  In this case we make sure to freeze this part of the model
        :param artifact_log_priors: (possibly None) 1D tensor with length len(utils.Variation) containing log prior probabilities
        of artifacts for each variation type, from train_model.py.  If given, freeze these parameters.
        :return:
        """
        if artifact_spectra_state_dict is not None:
            self.artifact_spectra.load_state_dict(artifact_spectra_state_dict)
            utils.freeze(self.artifact_spectra.parameters())

        # TODO: UNSAFE! We're getting into the details of exactly how unnormalized priors work
        # TODO: and the whole approach breaks apart if we implement things without going through self._unnormalized_priors
        # TODO: also, this only ensures that the priors are frozen within this function, but what if they are modified elsewhere?
        # note that we can't just freeze normally because this is just one row of the log priors tensor
        if artifact_log_priors is not None:
            with torch.no_grad():
                self._unnormalized_priors.weight[Call.ARTIFACT] = artifact_log_priors
                self._unnormalized_priors.weight[Call.NORMAL_ARTIFACT] = artifact_log_priors

        artifact_spectra_params_to_learn = self.artifact_spectra.parameters() if artifact_spectra_state_dict is None else []
        spectra_and_prior_params = chain(self.somatic_spectrum.parameters(), artifact_spectra_params_to_learn,
                                         self._unnormalized_priors.parameters(), self.normal_seq_error_spectra.parameters(),
                                         self.normal_artifact_spectra.parameters())
        optimizer = torch.optim.Adam(spectra_and_prior_params)

        for epoch in trange(1, num_iterations + 1, desc="AF spectra epoch"):
            epoch_loss = utils.StreamingAverage()

            pbar = tqdm(enumerate(posterior_loader), mininterval=10)
            for n, batch in pbar:
                relative_posteriors = self.log_relative_posteriors(batch)
                log_evidence = torch.logsumexp(relative_posteriors, dim=1)
                loss = -torch.mean(log_evidence)

                # note that we don't multiply by batch size because we take the mean of log evidence above
                # however, we must sum over variant types since each ignored site is simultaneously a missing non-SNV,
                # a missing non-INSERTION etc
                # we use a germline allele frequency of 0.001 for the missing sites but it doesn't really matter
                for variant_type in Variation:
                    log_priors = torch.nn.functional.log_softmax(self.make_unnormalized_priors(torch.from_numpy(variant_type.one_hot_tensor()).float().unsqueeze(dim=0), torch.Tensor([0.001])), dim=1)
                    log_seq_error_prior = log_priors.squeeze()[Call.SEQ_ERROR]
                    missing_loss = -ignored_to_non_ignored_ratio * log_seq_error_prior  
                    loss += missing_loss

                utils.backpropagate(optimizer, loss)

                # TODO: INELEGANT! since we can't freeze just the artifact row of log priors, we have to reset it after each batch
                if artifact_log_priors is not None:
                    with torch.no_grad():
                        self._unnormalized_priors.weight[Call.ARTIFACT] = artifact_log_priors

                epoch_loss.record_sum(batch.size() * loss.detach(), batch.size())

            if summary_writer is not None:
                summary_writer.add_scalar("spectrum negative log evidence", epoch_loss.get(), epoch)

                for variant_index, variant_type in enumerate(Variation):
                    mean = self.normal_seq_error_spectra[variant_index].get_mean()
                    summary_writer.add_scalar("normal seq error mean fraction for " + variant_type.name, mean, epoch)

                art_spectra_fig, art_spectra_axs = plot_artifact_spectra(self.artifact_spectra)
                summary_writer.add_figure("Artifact AF Spectra", art_spectra_fig, epoch)

                #normal_seq_error_spectra_fig, normal_seq_error_spectra_axs = plot_artifact_spectra(self.normal_seq_error_spectra)
                #summary_writer.add_figure("Normal Seq Error AF Spectra", normal_seq_error_spectra_fig, epoch)

                normal_artifact_spectra_fig, normal_artifact_spectra_axs = plot_artifact_spectra(self.normal_artifact_spectra)
                summary_writer.add_figure("Normal Artifact AF Spectra", normal_artifact_spectra_fig, epoch)

                var_spectra_fig, var_spectra_axs = plt.subplots()
                frac, dens = self.somatic_spectrum.spectrum_density_vs_fraction()
                var_spectra_axs.plot(frac.numpy(), dens.numpy())
                var_spectra_axs.set_title("Variant AF Spectrum")
                summary_writer.add_figure("Variant AF Spectra", var_spectra_fig, epoch)

                # bar plot of log priors -- data is indexed by call type name, and x ticks are variant types
                log_prior_bar_plot_data = defaultdict(list)
                for variant_type in Variation:
                    log_priors = torch.nn.functional.log_softmax(self.make_unnormalized_priors(torch.from_numpy(variant_type.one_hot_tensor()).float().unsqueeze(dim=0), torch.Tensor([0.001])), dim=1)
                    for call_type in (Call.SOMATIC, Call.ARTIFACT, Call.NORMAL_ARTIFACT):
                        log_prior_bar_plot_data[call_type.name].append(log_priors.squeeze().detach()[call_type])

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
    def calculate_probability_thresholds(self, loader, summary_writer: SummaryWriter = None, germline_mode: bool = False):
        self.train(False)
        error_probs_by_type = {var_type: [] for var_type in Variation}   # includes both artifact and seq errors

        error_probs_by_type_by_cnt = {var_type: [[] for _ in range(NUM_COUNT_BINS)] for var_type in Variation}

        pbar = tqdm(enumerate(loader), mininterval=10)
        for n, batch in pbar:
            alt_counts = batch.alt_counts.tolist()
            # 0th column is true variant, subtract it from 1 to get error prob
            error_probs = self.error_probabilities(batch, germline_mode).tolist()
            types = [posterior_datum.variant_type for posterior_datum in batch.original_list()]

            for var_type, alt_count, error_prob in zip(types, alt_counts, error_probs):
                error_probs_by_type[var_type].append(error_prob)
                error_probs_by_type_by_cnt[var_type][multiple_of_three_bin_index(min(alt_count, MAX_COUNT))].append(error_prob)

        thresholds_by_type = {}
        roc_fig, roc_axes = plt.subplots(1, len(Variation), sharex='all', sharey='all', squeeze=False)
        roc_by_cnt_fig, roc_by_cnt_axes = plt.subplots(1, len(Variation), sharex='all', sharey='all', squeeze=False, figsize=(10, 6), dpi=100)
        for var_type in Variation:
            # plot all count ROC curves for this variant type
            count_bin_labels = [str(multiple_of_three_bin_index_to_count(count_bin)) for count_bin in range(NUM_COUNT_BINS)]
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
