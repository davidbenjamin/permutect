import torch
from torch.utils.tensorboard import SummaryWriter

from itertools import chain
from tqdm.autonotebook import trange, tqdm

from mutect3 import utils
from mutect3.architecture.beta_binomial_mixture import BetaBinomialMixture
from mutect3.data.read_set_batch import ReadSetBatch
from mutect3.architecture.artifact_model import ArtifactModel


class PosteriorModel(torch.nn.Module):
    """

    """

    def __init__(self, artifact_model: ArtifactModel, variant_log_prior: float, artifact_log_prior: float):
        super(PosteriorModel, self).__init__()

        self.artifact_model = artifact_model
        utils.freeze(self.artifact_model.parameters())

        # TODO introduce parameters class so that num_components is not hard-coded
        # input size = 1 because all true variant types share a common AF spectrum
        self.variant_spectrum = BetaBinomialMixture(input_size=1, num_components=5)

        # artifact spectra for each variant type.  Variant type encoded as one-hot input vector.
        self.artifact_spectra = BetaBinomialMixture(input_size=len(utils.VariantType), num_components=5)

        # pre-softmax priors [log P(variant), log P(artifact), log P(seq error)] for each variant type
        # linear layer with no bias to select the appropriate priors given one-hot variant encoding
        # in torch linear layers the weight is indexed by out-features, then in-features, so indices are
        # self.unnormalized_priors.weight[0 = variant, 1 = artifact, 2 = seq error; SNV = 0, INSERTION = 1, DELETION = 2]
        self.unnormalized_priors = torch.nn.Linear(in_features=len(utils.VariantType), out_features=3, bias=False)
        with torch.no_grad():
            initial = torch.zeros_like(self.unnormalized_priors.weight)
            # the following assignments are broadcast over rows; that is, each variant type gets the same prior
            initial[0] = variant_log_prior
            initial[1] = artifact_log_prior
            initial[2] = 0
            self.unnormalized_priors.weight.copy_(initial)

    def posterior_probabilities(self, batch: ReadSetBatch) -> torch.Tensor:
        """
        :param batch:
        :return: non-log probabilities as a 2D tensor, 1st index is batch, 2nd is variant/artifact/seq error
        """
        return torch.nn.functional.softmax(self.log_relative_posteriors(batch))

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

        # produce a batch_size x 3 tensor of log priors
        log_priors = torch.nn.functional.log_softmax(self.unnormalized_priors(types), dim=1)

        # we give the variant AF spectrum a dummy one-hot tensor with only one type because, unlike artifacts, true
        # variants of all types share a common biological AF spectrum
        # the AF spectrum's forward method outputs the log likelihood of some number of alt reads, whereas we need
        # the probability that these *particular* reads exhibit the alt allele.  Since any read is (essentially) equally
        # likely to exhibit a variant, we simply divide by the combinatorial factor (depth)C(count)
        # this yields a 1D tensor of length batch_size
        variant_log_likelihoods = self.variant_spectrum.forward(torch.ones((batch.size(), 1)), batch.pd_tumor_depths(),
            batch.pd_tumor_alt_counts()) - log_combinatorial_factors

        # the artifact model gives the log likelihood ratio of these reads being alt given artifact, non-artifact
        # this is also a 1D tensor of length batch_size
        artifact_term = artifact_logits if artifact_logits is not None else self.artifact_model.forward(batch)
        artifact_log_likelihoods = self.artifact_spectra.forward(types, batch.pd_tumor_depths(),
            batch.pd_tumor_alt_counts()) - log_combinatorial_factors + artifact_term

        # this is a batch_size x 3 tensor
        stacked_log_likelihoods = torch.hstack(
            [variant_log_likelihoods.unsqueeze(dim=1), artifact_log_likelihoods.unsqueeze(dim=1),
             batch.seq_error_log_likelihoods().unsqueeze(dim=1)])

        return log_priors + stacked_log_likelihoods

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
                for variant_type in utils.VariantType:
                    log_priors = torch.nn.functional.log_softmax(self.unnormalized_priors(variant_type.one_hot_tensor().unsqueeze(dim=0)), dim=1)
                    log_seq_error_prior = log_priors.squeeze()[2]
                    missing_loss = ignored_to_non_ignored_ratio * log_seq_error_prior
                    loss += missing_loss

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                epoch_loss.record_sum(len(artifact_logits) * loss.detach(), len(artifact_logits))

            if summary_writer is not None:
                summary_writer.add_scalar("spectrum negative log evidence", epoch_loss.get(), epoch)

                fig, curve = self.variant_spectrum.plot_spectrum(torch.Tensor([1]), "Variant AF spectrum")
                summary_writer.add_figure("Variant AF spectrum", fig, epoch)

                for variant_type in utils.VariantType:
                    fig, curve = self.artifact_spectra.plot_spectrum(variant_type.one_hot_tensor(),
                        variant_type.name + " artifact AF spectrum")
                    summary_writer.add_figure(variant_type.name + " artifact AF spectrum", fig, epoch)
