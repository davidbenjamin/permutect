import torch
from torch.nn import Module, Parameter, BCEWithLogitsLoss

from permutect.data.batch import Batch
from permutect.data.reads_dataset import ratio_with_pseudocount, ReadsDataset
from permutect.metrics.loss_metrics import BatchProperty
from permutect.data.count_binning import count_bin_indices
from permutect.misc_utils import backpropagate
from permutect.utils.array_utils import index_4d_array, index_3d_array
from permutect.utils.enums import Label


class Balancer(Module):
    def __init__(self, dataset: ReadsDataset, learning_rate: float=0.001):
        super(Balancer, self).__init__()
        # general balancing idea: if total along some axis eg label is T and count for one particular label is C,
        # assign weight T/C -- then effective count is (T/C)*C = T, which is independent of label
        # we therefore need sums along certain axes:
        totals_slva = dataset.totals.get_marginal((BatchProperty.SOURCE, BatchProperty.LABEL, BatchProperty.VARIANT_TYPE, BatchProperty.ALT_COUNT_BIN))
        totals_sva = self.totals.get_marginal((BatchProperty.SOURCE, BatchProperty.VARIANT_TYPE, BatchProperty.ALT_COUNT_BIN))
        labeled_totals_sva = totals_sva - totals_slva[:, Label.UNLABELED, :, :]
        totals_va = torch.sum(totals_sva, dim=0)  # sum over label and source for source-balancing
        labeled_total = torch.sum(labeled_totals_sva)

        label_balancing_weights_slva = ratio_with_pseudocount(labeled_totals_sva[:, None, :, :], totals_slva)

        # next we want to normalize so that the average weight encountered on labeled data is 1 -- this way the learning rate
        # parameter has a fixed meaning.
        total_weight = torch.sum(totals_slva * label_balancing_weights_slva)
        total_supervised_weight = total_weight - torch.sum(totals_slva[:, Label.UNLABELED, :, :] * label_balancing_weights_slva[:, Label.UNLABELED, :, :])
        average_supervised_weight = total_supervised_weight / labeled_total

        # after the following line, average label-balancing weight encountered on labeled data is 1
        label_balancing_weights_slva = label_balancing_weights_slva / average_supervised_weight

        # the balancing process can reduce the influence of unlabeled data to match that of labeled data, but we don't want to
        # weight it strongly when there's little unlabeled data.  That is, if we have plenty of labeled data we are happy with
        # supervised learning!
        label_balancing_weights_slva[:, Label.UNLABELED, :, :] = \
            torch.clip(label_balancing_weights_slva[:, Label.UNLABELED, :, :], min=0, max=1)

        # at this point, average labeled weight is 1 and weights balance artifacts with non-artifacts for each combination
        # of source, count, and variant type

        # weights for adversarial source prediction task.  Balance over sources for each count and variant type
        source_balancing_weights_sva = ratio_with_pseudocount(totals_va[None, :, :], totals_sva)

        # we now normalize the source balancing weight to have the same total weights as supervised learning
        # the average supervised count has been normalized to 1 so the total supervised weight is just the total labeled
        # count.
        total_source_balancing_weight = torch.sum(totals_sva * source_balancing_weights_sva)
        source_balancing_weights_sva = self.source_balancing_weights_sva * labeled_total / total_source_balancing_weight

        # imbalanced unlabeled data can exert a bias just like labeled data.  These parameters keep track of the proportion
        # of unlabeled data that seem to be artifacts in order to weight losses appropriately.  Each source, count, and
        # variant type has its own proportion, stored as a logit-transformed probability
        # initialized as artifact, non-artifact equally likely
        self.totals_slva = Parameter(totals_slva, requires_grad=False)
        self.label_balancing_weights_slva = Parameter(label_balancing_weights_slva, requires_grad=False)
        self.source_balancing_weights_sva = Parameter(source_balancing_weights_sva, requires_grad=False)
        self.proportion_logits_sva = Parameter(torch.zeros_like(self.totals_slva[:, Label.UNLABELED, :, :]), requires_grad=True)
        self.optimizer = torch.optim.AdamW([self.proportion_logits_sva], lr=learning_rate)
        self.bce = BCEWithLogitsLoss(reduction='none')

    def calculate_batch_weights(self, batch: Batch):
        # TODO: we need a parameter to control the relative weight of unlabeled loss to labeled loss
        alt_count_bins = count_bin_indices(batch.get_alt_counts())
        return index_4d_array(self.label_balancing_weights_slva, batch.get_sources(), batch.get_labels(), batch.get_variant_types(), alt_count_bins)

    # calculate weights that adjust for the estimated proportion on unlabeled data that are actually artifacts, non-artifacts
    # as a side-effect of the calculation, also adjusts the estimated proportions with gradient descent
    def calculate_autobalancing_weights(self, batch: Batch, probabilities: torch.Tensor):
        # TODO: does this really need to be updated every batch?
        # effective totals are labeled plus estimated contributions from unlabeled
        # the proportion of unlabeled data that are artifacts
        proportions_sva = torch.sigmoid(self.proportion_logits_sva.detach())
        art_totals_sva = self.totals_slva[:, Label.ARTIFACT] + proportions_sva * self.totals_slva[:, Label.UNLABELED]
        nonart_totals_sva = self.totals_slva[:, Label.VARIANT] + (1 - proportions_sva) * self.totals_slva[:, Label.UNLABELED]
        totals_sva = art_totals_sva + nonart_totals_sva

        art_weights_sva = 0.5 * ratio_with_pseudocount(totals_sva, art_totals_sva)
        nonart_weights_sva = 0.5 * ratio_with_pseudocount(totals_sva, nonart_totals_sva)

        sources, variant_types = batch.get_sources(), batch.get_variant_types()
        alt_count_bins = count_bin_indices(batch.get_alt_counts())
        labels, is_labeled_mask = batch.get_labels(), batch.get_is_labeled_mask()

        # is_artifact is 1 / 0 if labeled as artifact / nonartifact; otherwise it's the estimated probability
        art_weights = index_3d_array(art_weights_sva, sources, variant_types, alt_count_bins)
        nonart_weights = index_3d_array(nonart_weights_sva, sources, variant_types, alt_count_bins)

        is_artifact = is_labeled_mask * labels + (1 - is_labeled_mask) * probabilities.detach()
        weights = is_artifact * art_weights + (1 - is_artifact) * nonart_weights

        # backpropagate our estimated proportions of artifacts among unlabeled data.  Note that we detach the computed probabilities!!
        artifact_prop_logits = index_3d_array(self.proportion_logits_sva, sources, variant_types, alt_count_bins)
        artifact_proportion_losses = (1 - is_labeled_mask) * self.bce(artifact_prop_logits, probabilities.detach())
        backpropagate(self.optimizer, torch.sum(artifact_proportion_losses))

        return weights.detach() # should already be detached, but just in case

    def calculate_batch_source_weights(self, batch: Batch):
        alt_count_bins = count_bin_indices(batch.get_alt_counts())
        return index_3d_array(self.source_balancing_weights_sva, batch.get_sources(), batch.get_variant_types(), alt_count_bins)