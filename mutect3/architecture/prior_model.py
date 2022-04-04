import torch
from torch import nn

from mutect3 import utils
from mutect3.architecture.af_spectrum import AFSpectrum
from mutect3.data.read_set_batch import ReadSetBatch


class PriorModel(nn.Module):
    # contains variant spectrum, artifact spectrum, and artifact/variant log prior ratio
    def __init__(self, initial_log_ratio=0.0):
        super(PriorModel, self).__init__()
        self.variant_spectrum = AFSpectrum()

        # TODO: is there such thing as a ModuleMap?
        self.artifact_spectra = nn.ModuleList()
        self.prior_log_odds = nn.ParameterList()  # log prior ratio log[P(artifact)/P(variant)] for each type
        for _ in utils.VariantType:
            self.artifact_spectra.append(AFSpectrum(lambda_for_initial_z=lambda x: (1 - 10*x)))
            self.prior_log_odds.append(nn.Parameter(torch.tensor(initial_log_ratio)))

    # calculate log likelihoods for all variant types and then apply a mask to select the correct
    # type for each datum in a batch
    def artifact_log_likelihoods(self, batch: ReadSetBatch):
        result = torch.zeros(batch.size())
        for variant_type in utils.VariantType:
            output = self.prior_log_odds[variant_type.value] + self.artifact_spectra[variant_type.value](batch)
            mask = torch.tensor([1 if variant_type == datum.variant_type() else 0 for datum in batch.original_list()])
            result += mask * output

        return result

    # forward pass returns posterior logits of being artifact given likelihood logits
    def forward(self, logits, batch):
        return logits + self.artifact_log_likelihoods(batch) - self.variant_spectrum(batch)

    # with fixed logits from the ReadSetClassifier, the log probability of seeing the observed tensors and counts
    # This is our objective to maximize when learning the prior model
    def log_evidence(self, logits, batch):
        term1 = torch.logsumexp(torch.column_stack((logits + self.artifact_log_likelihoods(batch), self.variant_spectrum(batch))), dim=1)

        prior_log_odds = torch.zeros_like(logits)
        for variant_type in utils.VariantType:
            mask = torch.tensor([1 if variant_type == datum.variant_type() else 0 for datum in batch.original_list()])
            prior_log_odds += mask * self.prior_log_odds[variant_type.value]

        term2 = torch.logsumexp(torch.column_stack((torch.zeros_like(logits), prior_log_odds)), dim=1)
        return term1 - term2

    # returns list of fig, curve tuples
    def plot_spectra(self, title_prefix=""):
        result = []
        for variant_type in utils.VariantType:
            result.append(self.artifact_spectra[variant_type.value].plot_spectrum(title_prefix + variant_type.name + " artifact AF spectrum"))
        result.append(self.variant_spectrum.plot_spectrum(title_prefix + "Variant AF spectrum"))
        return result