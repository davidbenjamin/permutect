from typing import List

import torch
from mutect3.data.read_set import ReadSet
from mutect3.utils import Variation


# Read sets have different sizes so we can't form a batch by naively stacking tensors.  We need a custom way
# to collate a list of Datum into a Batch

# collated batch contains:
# 2D tensors of ALL ref (alt) reads, not separated by set.
# number of reads in ref (alt) read sets, in same order as read tensors
# info: 2D tensor of info fields, one row per variant
# labels: 1D tensor of 0 if non-artifact, 1 if artifact
# lists of original mutect2_data and site info
# Example: if we have two input data, one with alt reads [[0,1,2], [3,4,5] and the other with
# alt reads [[6,7,8], [9,10,11], [12,13,14] then the output alt reads tensor is
# [[0,1,2], [3,4,5], [6,7,8], [9,10,11], [12,13,14]] and the output counts are [2,3]
# inside the model, the counts will be used to separate the reads into sets
from mutect3.utils import Label


class ReadSetBatch:

    def __init__(self, data: List[ReadSet]):
        self.labeled = data[0].label() != Label.UNLABELED
        self.ref_count = len(data[0].ref_tensor)
        self.alt_count = len(data[0].alt_tensor)
        for datum in data:
            assert (datum.label() != Label.UNLABELED) == self.labeled, "Batch may not mix labeled and unlabeled"
            assert len(datum.ref_tensor) == self.ref_count, "batch may not mix different ref counts"
            assert len(datum.alt_tensor) == self.alt_count, "batch may not mix different alt counts"

        self._ref_sequences = torch.stack([item.ref_sequence_tensor for item in data])
        self._reads = torch.cat([item.ref_tensor for item in data] + [item.alt_tensor for item in data], dim=0)
        self._info = torch.stack([item.info_tensor for item in data], dim=0)
        self._labels = torch.FloatTensor([1.0 if item.label == Label.ARTIFACT else 0.0 for item in data]) if self.labeled else None
        self._size = len(data)

    # pin memory for all tensors that are sent to the GPU
    def pin_memory(self):
        self._ref_sequences = self._ref_sequences.pin_memory()
        self._reads = self._reads.pin_memory()
        self._info = self._info.pin_memory()
        self._labels = self._labels.pin_memory()
        return self

    def is_labeled(self) -> bool:
        return self.labeled

    def size(self) -> int:
        return self._size

    def ref_sequences(self) -> torch.Tensor:
        return self._ref_sequences

    def reads(self) -> torch.Tensor:
        return self._reads

    def info(self) -> torch.Tensor:
        return self._info

    def labels(self) -> torch.Tensor:
        return self._labels

    def variant_type_one_hot(self):
        return self._info[:, -len(Variation):]

    def variant_type_mask(self, variant_type: Variation):
        return self._info[:, -len(Variation) + variant_type.value] == 1
