import torch
from torch import nn

from mutect3 import utils
from mutect3.architecture.beta_binomial_mixture import BetaBinomialMixture
from mutect3.data.read_set_batch import ReadSetBatch


class PriorModel(nn.Module):
    # contains variant spectrum, artifact spectrum, and artifact/variant log prior ratio
    def __init__(self, initial_log_ratio=0.0):
        super(PriorModel, self).__init__()
        # input size = 1 because all true variant types share a common AF spectrum
        self.variant_spectrum = BetaBinomialMixture(input_size=1, num_components=5)

        # artifact spectra for each variant type.  Variant type encoded as one-hot input vector.
        self.artifact_spectra = BetaBinomialMixture(input_size=len(utils.VariantType), num_components=5)

        # log prior ratio log[P(artifact)/P(variant)] for each type
        # linear layer with no bias to select the appropriate log odds given one-hot variant encoding
        self.prior_log_odds = nn.Linear(in_features=len(utils.VariantType), out_features=1, bias=False)

    def variant_log_likelihoods(self, batch: ReadSetBatch) -> torch.Tensor:
        dummy_input = torch.ones((batch.size(), 1))     # one-hot tensor with only one type
        return self.variant_spectrum.forward(dummy_input, batch.pd_tumor_depths(), batch.pd_tumor_alt_counts())

    def artifact_log_priors(self, batch: ReadSetBatch):
        return self.prior_log_odds(batch.variant_type_one_hot()).squeeze()

    def artifact_log_likelihoods(self, batch: ReadSetBatch, include_prior=True):
        var_types = batch.variant_type_one_hot()
        result = self.artifact_spectra.forward(var_types, batch.pd_tumor_depths(), batch.pd_tumor_alt_counts())

        if include_prior:
            result += self.prior_log_odds(var_types).squeeze()

        return result

    # forward pass returns posterior logits of being artifact given likelihood logits
    def forward(self, logits, batch):
        return logits + self.artifact_log_likelihoods(batch) - self.variant_spectrum(batch)

    # with fixed logits from the ReadSetClassifier, the log probability of seeing the observed tensors and counts
    # This is our objective to maximize when learning the prior model
    def log_evidence(self, logits, batch):
        term1 = torch.logsumexp(torch.column_stack((logits + self.artifact_log_likelihoods(batch), self.variant_spectrum(batch))), dim=1)

        prior_log_odds = self.artifact_log_priors(batch)
        term2 = torch.logsumexp(torch.column_stack((torch.zeros_like(logits), prior_log_odds)), dim=1)
        return term1 - term2
