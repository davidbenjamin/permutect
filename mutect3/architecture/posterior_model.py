from collections import defaultdict
from itertools import chain
from math import ceil

import torch
from intervaltree import IntervalTree
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from tqdm.autonotebook import trange, tqdm

from mutect3 import utils
from mutect3.architecture.beta_binomial_mixture import BetaBinomialMixture, FeaturelessBetaBinomialMixture
from mutect3.architecture.normal_artifact_spectrum import NormalArtifactSpectrum
from mutect3.data.posterior import PosteriorBatch
from mutect3.metrics import plotting
from mutect3.utils import Variation, Call
from mutect3.architecture.artifact_model import MAX_COUNT, NUM_COUNT_BINS, round_up_to_nearest_three, multiple_of_three_bin_index, multiple_of_three_bin_index_to_count

HOM_ALPHA, HOM_BETA = torch.Tensor([98.0]), torch.Tensor([2.0])


def compute_roc_data_and_threshold(error_probs):
    error_probs.sort()
    total_variants = len(error_probs) - sum(error_probs)

    # start by rejecting everything, then raise threshold one datum at a time
    threshold, tp, fp, best_f = 0.0, 0, 0, 0

    sens, prec = [], []
    for prob in error_probs:
        tp += (1 - prob)
        fp += prob
        sens.append(tp / (total_variants + 0.0001))
        prec.append(tp / (tp + fp + 0.0001))
        current_f = utils.f_score(tp, fp, total_variants)

        if current_f > best_f:
            best_f = current_f
            threshold = prob

    return sens, prec, threshold


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

    # the following should both be 1D tensors of length batch size
    alt_minor_ll = log_half_het_prop + alt_counts * log_mafs + ref_counts * log_1m_mafs
    alt_major_ll = log_half_het_prop + ref_counts * log_mafs + alt_counts * log_1m_mafs
    hom_ll = torch.log(hom_proportion) + utils.beta_binomial(depths, alt_counts, HOM_ALPHA, HOM_BETA)

    return torch.logsumexp(torch.vstack((alt_minor_ll, alt_major_ll, hom_ll)), dim=0)


def initialize_artifact_spectra():
    return BetaBinomialMixture(input_size=len(Variation), num_components=5)


# TODO: max_mean is hard-coded magic constant!!
def initialize_normal_seq_error_spectra():
    return BetaBinomialMixture(input_size=len(Variation), num_components=1, max_mean=0.01)


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
        self.normal_seq_error_spectra = initialize_normal_seq_error_spectra()

        # TODO: num_samples is hard-coded magic constant!!!
        self.normal_artifact_spectra = torch.nn.ModuleList([NormalArtifactSpectrum(num_samples=20) for variant_type in Variation])

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

        spectra_log_likelihoods = torch.zeros_like(log_priors)
        spectra_log_likelihoods[:, Call.SOMATIC] = self.somatic_spectrum.forward(batch.depths, batch.alt_counts)
        spectra_log_likelihoods[:, Call.ARTIFACT] = self.artifact_spectra.forward(types, batch.depths, batch.alt_counts)
        spectra_log_likelihoods[:, Call.SEQ_ERROR] = batch.seq_error_log_likelihoods

        normal_log_likelihoods = torch.zeros_like(log_priors)
        normal_seq_error_log_likelihoods = self.normal_seq_error_spectra.forward(types, batch.normal_depths, batch.normal_alt_counts)
        normal_log_likelihoods[:, Call.SOMATIC] = normal_seq_error_log_likelihoods
        normal_log_likelihoods[:, Call.ARTIFACT] = normal_seq_error_log_likelihoods
        normal_log_likelihoods[:, Call.SEQ_ERROR] = normal_seq_error_log_likelihoods

        normal_artifact_log_likelihoods = torch.zeros(batch.size())
        one_hot_types_2d = batch.variant_type_one_hot()
        for n, _ in enumerate(Variation):
            mask = one_hot_types_2d[:, n]
            assert mask.dim() == 1
            assert len(mask) == batch.size()
            na_log_lls_for_this_type = self.normal_artifact_spectra[n].forward(batch.alt_counts, batch.ref_counts(), batch.normal_alt_counts, batch.normal_ref_counts())
            assert na_log_lls_for_this_type.dim() == 1
            assert len(na_log_lls_for_this_type) == batch.size()
            normal_artifact_log_likelihoods += mask * na_log_lls_for_this_type

        # We assign half of the joint likelihood to tumor and half to normal
        spectra_log_likelihoods[:, Call.NORMAL_ARTIFACT] = normal_artifact_log_likelihoods / 2
        normal_log_likelihoods[:, Call.NORMAL_ARTIFACT] = normal_artifact_log_likelihoods / 2

        # since this is a default dict, if there's no segmentation for the contig we will get no overlaps but not an error
        # In our case there is either one or zero overlaps, and overlaps have the form
        segmentation_overlaps = [self.segmentation[item.contig][item.position] for item in batch.original_list()]
        normal_segmentation_overlaps = [self.normal_segmentation[item.contig][item.position] for item in batch.original_list()]
        mafs = torch.Tensor([list(overlaps)[0].data if overlaps else 0.5 for overlaps in segmentation_overlaps])
        normal_mafs = torch.Tensor([list(overlaps)[0].data if overlaps else 0.5 for overlaps in normal_segmentation_overlaps])
        afs = batch.allele_frequencies
        spectra_log_likelihoods[:, Call.GERMLINE] = germline_log_likelihood(afs, mafs, batch.alt_counts, batch.ref_counts())
        normal_log_likelihoods[:, Call.GERMLINE] = germline_log_likelihood(afs, normal_mafs, batch.normal_alt_counts, batch.normal_ref_counts())

        log_posteriors = log_priors + spectra_log_likelihoods + normal_log_likelihoods
        log_posteriors[:, Call.ARTIFACT] += batch.artifact_logits
        log_posteriors[:, Call.NORMAL_ARTIFACT] += batch.artifact_logits

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

                if loss.isnan().any():
                    print("epoch = " + str(epoch))
                    print("batch number " + str(n))
                    print("batch size " + str(batch.size()))
                    print("depths = " + str(batch.depths))
                    print("alt counts = " + str(batch.alt_counts))
                    print("normal depths = " + str(batch.normal_depths))
                    print("normal alt counts = " + str(batch.normal_alt_counts))
                    print("relative posteriors = " + str(relative_posteriors))
                    print("ingredients = " + str(self.log_posterior_and_ingredients(batch)))
                    assert 5 < 4, "CRASH BEFORE BACKPROP!!!"

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

                # DEBUG, DELETE LATER
                for variant_index, variant_type in enumerate(Variation):
                    linear = self.normal_artifact_spectra[variant_index].W
                    if linear.weight.isnan().any():
                        print("backprop yields nan in weights for " + variant_type.name)
                        print("epoch = " + str(epoch))
                        print("batch number " + str(n))
                        print("batch size " + str(batch.size()))
                        print("depths = " + str(batch.depths))
                        print("alt counts = " + str(batch.alt_counts))
                        print("normal depths = " + str(batch.normal_depths))
                        print("normal alt counts = " + str(batch.normal_alt_counts))
                        print("relative posteriors = " + str(relative_posteriors))
                        print("ingredients = " + str(self.log_posterior_and_ingredients(batch)))
                        assert 5 < 4, "CRASH BEFORE BACKPROP!!!"
                    if linear.bias.isnan().any():
                        print("backprop yields nan in bias for " + variant_type.name)
                        print("epoch = " + str(epoch))
                        print("batch number " + str(n))
                        print("batch size " + str(batch.size()))
                        print("depths = " + str(batch.depths))
                        print("alt counts = " + str(batch.alt_counts))
                        print("normal depths = " + str(batch.normal_depths))
                        print("normal alt counts = " + str(batch.normal_alt_counts))
                        print("relative posteriors = " + str(relative_posteriors))
                        print("ingredients = " + str(self.log_posterior_and_ingredients(batch)))
                        assert 5 < 4, "CRASH BEFORE BACKPROP!!!"


                # TODO: INELEGANT! since we can't freeze just the artifact row of log priors, we have to reset it after each batch
                if artifact_log_priors is not None:
                    with torch.no_grad():
                        self._unnormalized_priors.weight[Call.ARTIFACT] = artifact_log_priors

                epoch_loss.record_sum(batch.size() * loss.detach(), batch.size())

            if summary_writer is not None:
                summary_writer.add_scalar("spectrum negative log evidence", epoch_loss.get(), epoch)

                art_spectra_fig, art_spectra_axs = plot_artifact_spectra(self.artifact_spectra)
                summary_writer.add_figure("Artifact AF Spectra", art_spectra_fig, epoch)

                # DEBUG DELETE LATER
                try:
                    normal_seq_error_spectra_fig, normal_seq_error_spectra_axs = plot_artifact_spectra(self.normal_seq_error_spectra)
                    summary_writer.add_figure("Normal Seq Error AF Spectra", normal_seq_error_spectra_fig, epoch)
                except:
                    print("normal seq error weights pre softmax weight, bias: ")
                    print(self.normal_seq_error_spectra.weights_pre_softmax.weight)
                    print(self.normal_seq_error_spectra.weights_pre_softmax.bias)

                    print("normal seq error concentration pre exp weight, bias: ")
                    print(self.normal_seq_error_spectra.concentration_pre_exp.weight)
                    print(self.normal_seq_error_spectra.concentration_pre_exp.weight)

                    print("normal seq error mean pre sigmoid weight, bias: ")
                    print(self.normal_seq_error_spectra.mean_pre_sigmoid.weight)
                    print(self.normal_seq_error_spectra.mean_pre_sigmoid.weight)

                var_spectra_fig, var_spectra_axs = plt.subplots()
                frac, dens = self.somatic_spectrum.spectrum_density_vs_fraction()
                var_spectra_axs.plot(frac.numpy(), dens.numpy())
                var_spectra_axs.set_title("Variant AF Spectrum")
                summary_writer.add_figure("Variant AF Spectra", var_spectra_fig, epoch)

                # bar plot of log priors -- data is indexed by call type name, and x ticks are variant types
                log_prior_bar_plot_data = defaultdict(list)
                for variant_type in Variation:
                    log_priors = torch.nn.functional.log_softmax(self.make_unnormalized_priors(torch.from_numpy(variant_type.one_hot_tensor()).float().unsqueeze(dim=0), torch.Tensor([0.001])), dim=1)
                    for call_type in (Call.SOMATIC, Call.ARTIFACT):
                        log_prior_bar_plot_data[call_type.name].append(log_priors.squeeze().detach()[call_type])

                prior_fig, prior_ax = plotting.grouped_bar_plot(log_prior_bar_plot_data, [v_type.name for v_type in Variation], "log priors")
                summary_writer.add_figure("log priors", prior_fig, epoch)

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
            # stratified by alt count
            x_y_lab_tuples = []
            for count_bin in range(NUM_COUNT_BINS):
                error_probs = error_probs_by_type_by_cnt[var_type][count_bin]
                sens, prec, _ = compute_roc_data_and_threshold(error_probs) # we don't need the threshold here
                x_y_lab_tuples.append((sens, prec, str(multiple_of_three_bin_index_to_count(count_bin))))

            # plot all count ROC curves for this variant type
            plotting.simple_plot_on_axis(roc_by_cnt_axes[0, var_type], x_y_lab_tuples, None, None)

            # aggregated over all counts
            error_probs = error_probs_by_type[var_type]
            sens, prec, threshold = compute_roc_data_and_threshold(error_probs)
            plotting.simple_plot_on_axis(roc_axes[0, var_type], [(sens, prec, "")], None, None)
            thresholds_by_type[var_type] = threshold


        variation_types = [var_type.name for var_type in Variation]
        plotting.tidy_subplots(roc_by_cnt_fig, roc_by_cnt_axes, x_label="sensitivity", y_label="precision",
                               row_labels=[""], column_labels=variation_types)
        plotting.tidy_subplots(roc_fig, roc_axes, x_label="sensitivity", y_label="precision",
                               row_labels=[""], column_labels=variation_types)
        if summary_writer is not None:
            summary_writer.add_figure("theoretical ROC by variant type ", roc_fig)
            summary_writer.add_figure("theoretical ROC by variant type and alt count ", roc_by_cnt_fig)

        return thresholds_by_type
