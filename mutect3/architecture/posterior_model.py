import torch
from torch.utils.tensorboard import SummaryWriter

from itertools import chain
from collections import defaultdict
from tqdm.autonotebook import trange, tqdm
from matplotlib import pyplot as plt

from mutect3 import utils
from mutect3.utils import VariantType, CallType
from mutect3.architecture.beta_binomial_mixture import BetaBinomialMixture, FeaturelessBetaBinomialMixture
from mutect3.data.read_set_batch import ReadSetBatch
from mutect3.architecture.artifact_model import ArtifactModel
from mutect3.metrics import plotting


class PosteriorModel(torch.nn.Module):
    """

    """

    def __init__(self, artifact_model: ArtifactModel, variant_log_prior: float, artifact_log_prior: float):
        super(PosteriorModel, self).__init__()

        self.artifact_model = artifact_model
        utils.freeze(self.artifact_model.parameters())

        # TODO introduce parameters class so that num_components is not hard-coded
        # featureless because true variant types share a common AF spectrum
        self.variant_spectrum = FeaturelessBetaBinomialMixture(num_components=5)

        # artifact spectra for each variant type.  Variant type encoded as one-hot input vector.
        self.artifact_spectra = BetaBinomialMixture(input_size=len(VariantType), num_components=5)

        # pre-softmax priors [log P(variant), log P(artifact), log P(seq error)] for each variant type
        # linear layer with no bias to select the appropriate priors given one-hot variant encoding
        # in torch linear layers the weight is indexed by out-features, then in-features, so indices are
        # self.unnormalized_priors.weight[call type, variant type]
        self.unnormalized_priors = torch.nn.Linear(in_features=len(VariantType), out_features=len(CallType), bias=False)
        with torch.no_grad():
            initial = torch.zeros_like(self.unnormalized_priors.weight)
            # the following assignments are broadcast over rows; that is, each variant type gets the same prior
            initial[CallType.VARIANT] = variant_log_prior
            initial[CallType.ARTIFACT] = artifact_log_prior
            initial[CallType.SEQ_ERROR] = 0
            self.unnormalized_priors.weight.copy_(initial)

    def posterior_probabilities(self, batch: ReadSetBatch) -> torch.Tensor:
        """
        :param batch:
        :return: non-log probabilities as a 2D tensor, 1st index is batch, 2nd is variant/artifact/seq error
        """
        return torch.nn.functional.softmax(self.log_relative_posteriors(batch), dim=1)

    def error_probabilities(self, batch: ReadSetBatch) -> torch.Tensor:
        """
        :param batch:
        :return: non-log error probabilities as a 1D tensor with length batch size
        """
        return 1 - self.posterior_probabilities(batch)[:, CallType.VARIANT]     # 0th column is variant

    def log_relative_posteriors(self, batch: ReadSetBatch, artifact_logits: torch.Tensor = None) -> torch.Tensor:
        """
        :param artifact_logits: precomputed log odds ratio of artifact to non-artifact likelihoods.  If absent, it is computed
            from self.artifact_model
        :param batch:
        :batch.seq_error_log_likelihoods() is the probability that these *particular* reads exhibit the alt allele given a
        sequencing error ie an error explainable in terms of base qualities.  For example if we have two alt reads with error
        probability of 0.1 and 0.2, and two ref reads with error probabilities 0.05 and 0.06 this quantity would be
        log(0.1*0.2*0.95*0.94).  This is an annotation emitted by the GATK and by the time it reaches here is a 1D tensor
        of length batch_size.
        :return:
        """
        types = batch.variant_type_one_hot()
        log_combinatorial_factors = utils.log_binomial_coefficient(batch.pd_tumor_depths(), batch.pd_tumor_alt_counts())

        # log priors of all call types for each datum -- batch.size() x len(CallType) tensor
        log_priors = torch.nn.functional.log_softmax(self.unnormalized_priors(types), dim=1)

        # log priors of all call types for each datum -- batch.size() x len(CallType) tensor
        # initialize (arbitrarily) to zero, then fill in the variant, artifact, and seq error columns
        log_likelihoods = torch.zeros_like(log_priors)

        # the AF spectrum's forward method outputs the log likelihood of some number of alt reads, whereas we need
        # the probability that these *particular* reads exhibit the alt allele.  Since any read is (essentially) equally
        # likely to exhibit a variant, we simply divide by the combinatorial factor (depth)C(count)
        # this yields a 1D tensor of length batch_size
        log_likelihoods[:, CallType.VARIANT] = self.variant_spectrum.forward(batch.pd_tumor_depths(),
            batch.pd_tumor_alt_counts()) - log_combinatorial_factors

        # the artifact model gives the log likelihood ratio of these reads being alt given artifact, non-artifact
        # this is also a 1D tensor of length batch_size
        artifact_term = artifact_logits if artifact_logits is not None else self.artifact_model.forward(batch)
        log_likelihoods[:, CallType.ARTIFACT] = self.artifact_spectra.forward(types, batch.pd_tumor_depths(),
            batch.pd_tumor_alt_counts()) - log_combinatorial_factors + artifact_term

        log_likelihoods[:, CallType.SEQ_ERROR] = batch.seq_error_log_likelihoods()

        return log_priors + log_likelihoods

    def learn_priors_and_spectra(self, loader, num_iterations, ignored_to_non_ignored_ratio: float,
                                 summary_writer: SummaryWriter = None):
        """
        :param summary_writer:
        :param num_iterations:
        :param loader:
        :param ignored_to_non_ignored_ratio: ratio of sites in which no evidence of variation was found to sites in which
        sufficient evidence was found to emit test data.  Without this parameter (i.e. if it were set to zero) we would
        underestimate the frequency of sequencing error, hence overestimate the prior probability of variation.

        :return:
        """

        # precompute the logits of the artifact model
        artifact_logits_and_batches = [(self.artifact_model.forward(batch=batch).detach(), batch) for batch in loader]

        spectra_and_prior_params = chain(self.variant_spectrum.parameters(), self.artifact_spectra.parameters(),
                                         self.unnormalized_priors.parameters())
        optimizer = torch.optim.Adam(spectra_and_prior_params)

        for epoch in trange(1, num_iterations + 1, desc="AF spectra epoch"):
            epoch_loss = utils.StreamingAverage()

            pbar = tqdm(enumerate(artifact_logits_and_batches), mininterval=10)
            for n, (artifact_logits, batch) in pbar:
                relative_posteriors = self.log_relative_posteriors(batch, artifact_logits)
                log_evidence = torch.logsumexp(relative_posteriors, dim=1)
                loss = -torch.mean(log_evidence)

                # note that we don't multiply by batch size because we take the mean of log evidence above
                # however, we must sum over variant types since each ignored site is simultaneously a missing non-SNV,
                # a missing non-INSERTION etc
                for variant_type in VariantType:
                    log_priors = torch.nn.functional.log_softmax(self.unnormalized_priors(variant_type.one_hot_tensor().unsqueeze(dim=0)), dim=1)
                    log_seq_error_prior = log_priors.squeeze()[CallType.SEQ_ERROR]
                    missing_loss = -ignored_to_non_ignored_ratio * log_seq_error_prior  
                    loss += missing_loss

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                epoch_loss.record_sum(len(artifact_logits) * loss.detach(), len(artifact_logits))

            if summary_writer is not None:
                summary_writer.add_scalar("spectrum negative log evidence", epoch_loss.get(), epoch)

                # plot AF spectra in 2x2 grid
                spectra_fig, spectra_axs = plt.subplots(2, 2, sharex='all', sharey='all')
                frac, dens = self.variant_spectrum.spectrum_density_vs_fraction()
                spectra_axs[0, 0].plot(frac.numpy(), dens.numpy())
                spectra_axs[0, 0].set_title("Variant AF Spectrum")

                for variant_type in VariantType:
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
                for variant_type in VariantType:
                    log_priors = torch.nn.functional.log_softmax(self.unnormalized_priors(variant_type.one_hot_tensor().unsqueeze(dim=0)), dim=1)
                    for call_type in (CallType.VARIANT, CallType.ARTIFACT):
                        log_prior_bar_plot_data[call_type.name].append(log_priors.squeeze().detach()[call_type])

                prior_fig, prior_ax = plotting.grouped_bar_plot(log_prior_bar_plot_data, [v_type.name for v_type in VariantType], "log priors")
                summary_writer.add_figure("log priors", prior_fig, epoch)

    def calculate_probability_threshold(self, loader, summary_writer: SummaryWriter = None):
        self.train(False)
        error_probs = []    # includes both artifact and seq errors

        pbar = tqdm(enumerate(loader), mininterval=10)
        for n, batch in pbar:
            # 0th column is true variant, subtract it from 1 to get error prob
            error_probs.extend(self.error_probabilities(batch).tolist())

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
