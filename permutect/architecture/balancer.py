import torch
from torch.nn import Module, Parameter

from permutect.data.reads_dataset import ALL_COUNTS_INDEX
from permutect.utils.array_utils import index_4d_array, index_3d_array


class Balancer(Module):
    def __init__(self, dataset):
        super(Balancer, self).__init__()
        self.label_balancing_weights_sclt = Parameter(dataset.label_balancing_weights_sclt, requires_grad=False)
        self.source_balancing_weights_sct = Parameter(dataset.source_balancing_weights_sct, requires_grad=False)

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

    # note: this works for both BaseBatch/BaseDataset AND ArtifactBatch/ArtifactDataset
    # if by_count is True, each count is weighted separately for balanced loss within that count
    def calculate_batch_source_weights(self, batch, by_count: bool):
        # For batch index n, we want weight[n] = dataset.source_weights[sources[n], alt_counts[n], variant_types[n]]
        sources = batch.get_sources()
        counts = batch.get_alt_counts() if by_count else torch.full(size=(len(sources),), fill_value=ALL_COUNTS_INDEX, device=sources.device, dtype=torch.int)
        variant_types = batch.get_variant_types()

        return index_3d_array(self.source_balancing_weights_sct, sources, counts, variant_types)