import torch
from torch.utils.tensorboard import SummaryWriter

from itertools import chain
from collections import defaultdict
from intervaltree import IntervalTree
from tqdm.autonotebook import trange, tqdm
from matplotlib import pyplot as plt

from mutect3 import utils
from mutect3.data.posterior_batch import PosteriorBatch
from mutect3.utils import Variation, Call
from mutect3.architecture.beta_binomial_mixture import BetaBinomialMixture, FeaturelessBetaBinomialMixture
from mutect3.data.read_set_batch import ReadSetBatch
from mutect3.metrics import plotting

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

    # the following should both be 1D tensors of length batch size
    alt_minor_ll = log_half_het_prop + alt_counts * log_mafs + ref_counts * log_1m_mafs
    alt_major_ll = log_half_het_prop + ref_counts * log_mafs + alt_counts * log_1m_mafs
    hom_ll = torch.log(hom_proportion) + utils.beta_binomial(depths, alt_counts(), HOM_ALPHA, HOM_BETA)

    return torch.logsumexp(torch.vstack((alt_minor_ll, alt_major_ll, hom_ll)), dim=0)


class PosteriorModel(torch.nn.Module):
    """

    """
    def __init__(self, variant_log_prior: float, artifact_log_prior: float, segmentation=defaultdict(IntervalTree)):
        super(PosteriorModel, self).__init__()

        # TODO: might as well give the normal segmentation as well
        self.segmentation = segmentation

        # TODO introduce parameters class so that num_components is not hard-coded
        # featureless because true variant types share a common AF spectrum
        self.somatic_spectrum = FeaturelessBetaBinomialMixture(num_components=5)

        # artifact spectra for each variant type.  Variant type encoded as one-hot input vector.
        self.artifact_spectra = BetaBinomialMixture(input_size=len(Variation), num_components=5)

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
            initial[Call.GERMLINE] = 0
            self._unnormalized_priors.weight.copy_(initial)

    def make_unnormalized_priors(self, variant_types_one_hot_2d: torch.Tensor, allele_frequencies_1d: torch.Tensor) -> torch.Tensor:
        result = self._unnormalized_priors(variant_types_one_hot_2d)
        result[:, Call.SEQ_ERROR] = 0
        result[:, Call.GERMLINE] = torch.log(1 - torch.square(1 - allele_frequencies_1d))     # 1 minus hom ref probability
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
        return 1 - self.posterior_probabilities(batch)[:, Call.GERMLINE if germline_mode else Call.SOMATIC]     # 0th column is variant

    def log_relative_posteriors(self, batch: PosteriorBatch) -> torch.Tensor:
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

        # log priors of all call types for each datum -- batch.size() x len(CallType) tensor
        log_priors = torch.nn.functional.log_softmax(self.make_unnormalized_priors(types, batch.allele_frequencies()), dim=1)

        # log priors of all call types for each datum -- batch.size() x len(CallType) tensor
        # initialize (arbitrarily) to zero, then fill in the variant, artifact, and seq error columns
        log_likelihoods = torch.zeros_like(log_priors)

        # the AF spectrum's forward method uses a beta binomial that excludes the nCk combinatorial factor
        # Thus it gives the log likelihood that these *particular* reads exhibit the alt allele with no modification.
        somatic_ll = self.somatic_spectrum.forward(batch.depths(), batch.alt_counts())
        log_likelihoods[:, Call.SOMATIC] = somatic_ll + batch.normal_seq_error_log_likelihoods()

        # TODO: need to mix in possibility of artifact in normal as well as tumor -- currently assume artifact in tumor and nothing in normal
        log_likelihoods[:, Call.ARTIFACT] = self.artifact_spectra.forward(types, batch.depths(), batch.alt_counts()) \
            + batch.artifact_logits() + batch.normal_seq_error_log_likelihoods()

        # sample's and normal's reads both explained by sequencing error ie nothing going on in either
        # TODO: need to mix in possibility of nothing happening in tumor but artifact in normal -- currently assume seq error in tumor and seq error in normal
        log_likelihoods[:, Call.SEQ_ERROR] = batch.seq_error_log_likelihoods() + batch.normal_seq_error_log_likelihoods()

        # since this is a default dict, if there's no segmentation for the contig we will get no overlaps but not an error
        # In our case there is either one or zero overlaps, and overlaps have the form
        segmentation_overlaps = [self.segmentation[item.contig()][item.position()] for item in batch.original_list()]
        mafs = torch.Tensor([list(overlaps)[0].data if overlaps else 0.5 for overlaps in segmentation_overlaps])

        # TODO: allow for CNV / segmentation in normal
        normal_mafs = torch.Tensor([0.5 for _ in segmentation_overlaps])

        afs = batch.allele_frequencies()
        germline_ll = germline_log_likelihood(afs, mafs, batch.alt_counts(), batch.ref_counts())
        normal_germline_ll = germline_log_likelihood(afs, normal_mafs, batch.normal_alt_counts(), batch.normal_ref_counts())

        log_likelihoods[:, Call.GERMLINE] = germline_ll + normal_germline_ll

        return log_priors + log_likelihoods

    def learn_priors_and_spectra(self, posterior_loader, num_iterations, ignored_to_non_ignored_ratio: float,
                                 summary_writer: SummaryWriter = None):
        """
        :param summary_writer:
        :param num_iterations:
        :param posterior_loader:
        :param ignored_to_non_ignored_ratio: ratio of sites in which no evidence of variation was found to sites in which
        sufficient evidence was found to emit test data.  Without this parameter (i.e. if it were set to zero) we would
        underestimate the frequency of sequencing error, hence overestimate the prior probability of variation.

        :return:
        """
        spectra_and_prior_params = chain(self.somatic_spectrum.parameters(), self.artifact_spectra.parameters(),
                                         self._unnormalized_priors.parameters())
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
                    log_priors = torch.nn.functional.log_softmax(self.make_unnormalized_priors(variant_type.one_hot_tensor().unsqueeze(dim=0), torch.Tensor([0.001])), dim=1)
                    log_seq_error_prior = log_priors.squeeze()[Call.SEQ_ERROR]
                    missing_loss = -ignored_to_non_ignored_ratio * log_seq_error_prior  
                    loss += missing_loss

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                epoch_loss.record_sum(batch.size() * loss.detach(), batch.size())

            if summary_writer is not None:
                summary_writer.add_scalar("spectrum negative log evidence", epoch_loss.get(), epoch)

                # plot AF spectra in 2x2 grid
                spectra_fig, spectra_axs = plt.subplots(2, 2, sharex='all', sharey='all')
                frac, dens = self.somatic_spectrum.spectrum_density_vs_fraction()
                spectra_axs[0, 0].plot(frac.numpy(), dens.numpy())
                spectra_axs[0, 0].set_title("Variant AF Spectrum")

                for variant_type in Variation:
                    n = variant_type + 1    # +1 is the offset for variant
                    row, col = int(n/2), n % 2
                    frac, dens = self.artifact_spectra.spectrum_density_vs_fraction(variant_type.one_hot_tensor())
                    spectra_axs[row, col].plot(frac.numpy(), dens.numpy())
                    spectra_axs[row, col].set_title(variant_type.name + " artifact AF spectrum")

                for ax in spectra_fig.get_axes():
                    ax.label_outer()

                summary_writer.add_figure("Artifact and Variant AF Spectra", spectra_fig, epoch)

                # bar plot of log priors -- data is indexed by call type name, and x ticks are variant types
                log_prior_bar_plot_data = defaultdict(list)
                for variant_type in Variation:
                    log_priors = torch.nn.functional.log_softmax(self.make_unnormalized_priors(variant_type.one_hot_tensor().unsqueeze(dim=0), torch.Tensor([0.001])), dim=1)
                    for call_type in (Call.SOMATIC, Call.ARTIFACT):
                        log_prior_bar_plot_data[call_type.name].append(log_priors.squeeze().detach()[call_type])

                prior_fig, prior_ax = plotting.grouped_bar_plot(log_prior_bar_plot_data, [v_type.name for v_type in Variation], "log priors")
                summary_writer.add_figure("log priors", prior_fig, epoch)

    def calculate_probability_threshold(self, loader, summary_writer: SummaryWriter = None, germline_mode: bool = False):
        self.train(False)
        error_probs = []    # includes both artifact and seq errors

        pbar = tqdm(enumerate(loader), mininterval=10)
        for n, batch in pbar:
            # 0th column is true variant, subtract it from 1 to get error prob
            error_probs.extend(self.error_probabilities(batch, germline_mode).tolist())

        error_probs.sort()
        total_variants = len(error_probs) - sum(error_probs)

        # start by rejecting everything, then raise threshold one datum at a time
        threshold, tp, fp, best_f = 0.0, 0, 0, 0

        sens, prec = [], []
        for prob in error_probs:
            tp += (1 - prob)
            fp += prob
            sens.append(tp/(total_variants+0.0001))
            prec.append(tp/(tp+fp+0.0001))
            current_f = utils.f_score(tp, fp, total_variants)

            if current_f > best_f:
                best_f = current_f
                threshold = prob

        if summary_writer is not None:
            x_y_lab = [(sens, prec, "theoretical ROC curve according to M3's posterior probabilities")]
            fig, curve = plotting.simple_plot(x_y_lab, x_label="sensitivity", y_label="precision",
                                              title="theoretical ROC curve according to M3's posterior probabilities")
            summary_writer.add_figure("theoretical ROC curve", fig)

        return threshold
