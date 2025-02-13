import torch
from torch.nn import Module, Parameter, BCEWithLogitsLoss

from permutect.data.reads_dataset import ALL_COUNTS_INDEX, ratio_with_pseudocount
from permutect.misc_utils import backpropagate
from permutect.utils.array_utils import index_4d_array, index_3d_array
from permutect.utils.enums import Label


class Balancer(Module):
    def __init__(self, dataset, learning_rate: float=0.001):
        super(Balancer, self).__init__()
        self.label_balancing_weights_sclt = Parameter(dataset.label_balancing_weights_sclt, requires_grad=False)
        self.source_balancing_weights_sct = Parameter(dataset.source_balancing_weights_sct, requires_grad=False)

        self.totals_sclt = Parameter(torch.from_numpy(dataset.totals_sclt), requires_grad=False)

        # imbalanced unlabeled data can exert a bias just like labeled data.  These parameters keep track of the proportion
        # of unlabeled data that seem to be artifacts in order to weight losses appropriately.  Each source, count, and
        # variant type has its own proportion, stored as a logit-transformed probability
        # initialized as artifact, non-artifact equally likely
        self.proportion_logits_sct = Parameter(torch.zeros_like(self.totals_sclt[:, :, Label.UNLABELED, :]), requires_grad=True)
        self.optimizer = torch.optim.AdamW([self.proportion_logits_sct], lr=learning_rate)

        self.bce = BCEWithLogitsLoss(reduction='none')

    # note: this works for both BaseBatch/BaseDataset AND ArtifactBatch/ArtifactDataset
    # each count is weighted separately for balanced loss within that count
    def calculate_batch_weights(self, batch):
        # TODO: we need a parameter to control the relative weight of unlabeled loss to labeled loss
        # For batch index n, we want weight[n] = dataset.weights[alt_counts[n], labels[n], variant_types[n]]
        sources = batch.get_sources()
        counts = batch.get_alt_counts()
        labels = batch.get_labels()
        variant_types = batch.get_variant_types()

        return index_4d_array(self.label_balancing_weights_sclt, sources, counts, labels, variant_types)

    # calculate weights that adjust for the estimated proportion on unlabeled data that are actually artifacts, non-artifacts
    # as a side-effect of the calculation, also adjusts the estimated proportions with gradient descent
    def calculate_autobalancing_weights(self, batch, probabilities):
        # TODO: does this really need to be updated every batch?
        # effective totals are labeled plus estimated contributions from unlabeled
        # the proportion of unlabeled data that are artifacts
        proportions_sct = torch.sigmoid(self.proportion_logits_sct.detach())
        art_totals_sct = self.totals_sclt[:, :, Label.ARTIFACT, :] + proportions_sct * self.totals_sclt[:, :, Label.UNLABELED, :]
        nonart_totals_sct = self.totals_sclt[:, :, Label.VARIANT, :] + (1 - proportions_sct) * self.totals_sclt[:, :, Label.UNLABELED, :]
        totals_sct = art_totals_sct + nonart_totals_sct

        art_weights_sct = 0.5 * ratio_with_pseudocount(totals_sct, art_totals_sct)
        nonart_weights_sct = 0.5 * ratio_with_pseudocount(totals_sct, nonart_totals_sct)

        sources, alt_counts, variant_types = batch.get_sources(), batch.get_alt_counts(), batch.get_variant_types()
        labels, is_labeled_mask = batch.get_labels(), batch.get_is_labeled_mask()

        # is_artifact is 1 / 0 if labeled as artifact / nonartifact; otherwise it's the estimated probability
        art_weights = index_3d_array(art_weights_sct, sources, alt_counts, variant_types)
        nonart_weights = index_3d_array(nonart_weights_sct, sources, alt_counts, variant_types)

        is_artifact = is_labeled_mask * labels + (1 - is_labeled_mask) * probabilities.detach()
        weights = is_artifact * art_weights + (1 - is_artifact) * nonart_weights

        # backpropagate our estimated proportions of artifacts among unlabeled data.  Note that we detach the computed probabilities!!
        artifact_prop_logits = index_3d_array(self.proportion_logits_sct, sources, alt_counts, variant_types)
        artifact_proportion_losses = (1 - is_labeled_mask) * self.bce(artifact_prop_logits, probabilities.detach())
        backpropagate(self.optimizer, torch.sum(artifact_proportion_losses))

        return weights.detach() # should already be detached, but just in case

    # note: this works for both BaseBatch/BaseDataset AND ArtifactBatch/ArtifactDataset
    # if by_count is True, each count is weighted separately for balanced loss within that count
    def calculate_batch_source_weights(self, batch, by_count: bool):
        # For batch index n, we want weight[n] = dataset.source_weights[sources[n], alt_counts[n], variant_types[n]]
        sources = batch.get_sources()
        counts = batch.get_alt_counts() if by_count else torch.full(size=(len(sources),), fill_value=ALL_COUNTS_INDEX, device=sources.device, dtype=torch.int)
        variant_types = batch.get_variant_types()

        return index_3d_array(self.source_balancing_weights_sct, sources, counts, variant_types)