from typing import List

import torch
from mutect3.data.read_set import ReadSet


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
        self._original_list = data  # keep this for downsampling augmentation
        self.labeled = data[0].label() != Label.UNLABELED
        for datum in data:
            if (datum.label() != Label.UNLABELED) != self.labeled:
                raise Exception("Batch may not mix labeled and unlabeled")

        # if ref read counts are 1, 2, 3 and alt read counts are 1, 2, 1, then end indices are 1, 3, 6, 7, 9, 10
        self._read_end_indices = torch.cumsum(torch.LongTensor([len(item.ref_tensor()) for item in data] + [len(item.alt_tensor()) for item in data]), dim=0)

        self._reads = torch.cat([item.ref_tensor() for item in data] + [item.alt_tensor() for item in data], dim=0)
        self._info = torch.stack([item.info_tensor() for item in data], dim=0)
        self._labels = torch.FloatTensor([1.0 if item.label() == Label.ARTIFACT else 0.0 for item in data]) if self.labeled else None
        self._size = len(data)

    # pin memory for all tensors that are sent to the GPU
    def pin_memory(self):
        self._reads = self._reads.pin_memory()
        self._info = self._info.pin_memory()
        self._labels = self._labels.pin_memory()
        self._read_end_indices = self._read_end_indices.pin_memory()
        return self

    def original_list(self) -> List[ReadSet]:
        return self._original_list

    def is_labeled(self) -> bool:
        return self.labeled

    def size(self) -> int:
        return self._size

    def read_end_indices(self):
        return self._read_end_indices

    def reads(self) -> torch.Tensor:
        return self._reads

    def alt_counts(self) -> torch.IntTensor:
        return torch.IntTensor([len(item.alt_tensor()) for item in self._original_list])

    def info(self) -> torch.Tensor:
        return self._info

    def labels(self) -> torch.Tensor:
        return self._labels

    def variant_type_one_hot(self):
        return torch.vstack([item.variant_type().one_hot_tensor() for item in self._original_list])

    def variant_type_mask(self, variant_type):
        return torch.BoolTensor([item.variant_type() == variant_type for item in self._original_list])
