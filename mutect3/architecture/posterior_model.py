import torch

from mutect3 import utils
from mutect3.architecture.beta_binomial_mixture import BetaBinomialMixture
from mutect3.data.read_set_batch import ReadSetBatch
from mutect3.architecture.artifact_model import ArtifactModel


class PosteriorModel(torch.nn.Module):
    """

    """

    def __init__(self, artifact_model: ArtifactModel):
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
        self.unnormalized_priors = torch.nn.Linear(in_features=len(utils.VariantType), out_features=3, bias=False)
        with torch.no_grad():
            self.unnormalized_priors.weight.copy_(initial_log_ratio * torch.ones_like(self.unnormalized_priors.weight))

    def log_relative_posteriors(self, batch: ReadSetBatch, seq_error_log_likelihoods: torch.Tensor):
        '''

        :param batch:
        :param seq_error_log_likelihoods: the probability that these *particular* reads exhibit the alt allele given a
        sequencing error ie an error explainable in terms of base qualities.  For example if we have two alt reads with error
        probability of 0.1 and 0.2, and two ref reads with error probabilities 0.05 and 0.06 this quantity would be
        log(0.1*0.2*0.95*0.94).  This is an annotation emitted by the GATK and by the time it reaches here is a 1D tensor
        of length batch_size.
        :return:
        '''
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
        artifact_log_likelihoods = self.artifact_spectra.forward(types, batch.pd_tumor_depths(), batch.pd_tumor_alt_counts()) \
            - log_combinatorial_factors + self.artifact_model.forward(batch)

        # this is a batch_size x 3 tensor
        stacked_log_likelihoods = torch.hstack([variant_log_likelihoods.unsqueeze(dim=1), artifact_log_likelihoods.unsqueeze(dim=1),
                      seq_error_log_likelihoods.unsqueeze(dim=1)])

        return log_priors + stacked_log_likelihoods

    # TODO: this probably doesn't need ot be its own method -- just inline within the learning method
    # return 1D tensor of length batch_size of log_evidence, the target for maximizing when learning priors and spectra
    def log_evidence(self, batch: ReadSetBatch, seq_error_log_likelihoods: torch.Tensor):
        log_relative_posteriors = self.log_relative_posteriors(batch, seq_error_log_likelihoods)
        return torch.logsumexp(log_relative_posteriors, dim=1)

    def learn_priors_and_spectra(self, ignored_to_non_ignored_ratio: float):
        """
        :param ignored_to_non_ignored_ratio: ratio of sites in which no evidence of variation was found to sites in which
        sufficient evidence was found to emit test data.  Without this parameter (i.e. if it were set to zero) we would
        underestimate the frequency of sequencing error, hence overestimate the prior probability of variation.

        :return:
        """
