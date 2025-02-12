from __future__ import annotations

import copy
from typing import List

import numpy as np
import torch
from torch import IntTensor, Tensor

from permutect.data.reads_datum import ReadsDatum
from permutect.utils.enums import Label


class ReadsBatch:
    """
    Read sets have different sizes so we can't form a batch by naively stacking tensors.  We need a custom way
    to collate a list of Datum into a Batch

    collated batch contains:
    2D tensors of ALL ref (alt) reads, not separated by set.
    number of reads in ref (alt) read sets, in same order as read tensors
    info: 2D tensor of info fields, one row per variant
    labels: 1D tensor of 0 if non-artifact, 1 if artifact
    lists of original mutect2_data and site info

    Example: if we have two input data, one with alt reads [[0,1,2], [3,4,5] and the other with
    alt reads [[6,7,8], [9,10,11], [12,13,14] then the output alt reads tensor is
    [[0,1,2], [3,4,5], [6,7,8], [9,10,11], [12,13,14]] and the output counts are [2,3]
    inside the model, the counts will be used to separate the reads into sets
    """

    def __init__(self, data: List[ReadsDatum]):
        # TODO: can we get rid of this potential bottleneck (might interact really badly with multiple workers)?
        self._original_list = data

        # num_classes = 5 for A, C, G, T, and deletion / insertion
        ref_alt = [torch.flatten(torch.permute(torch.nn.functional.one_hot(torch.from_numpy(np.vstack(item.get_ref_and_alt_sequences())).long(), num_classes=5), (0,2,1)), 0, 1) for item in data]    # list of 2D (2x5)xL
        # this is indexed by batch, length, channel (aka one-hot base encoding)
        ref_alt_bcl = torch.stack(ref_alt)

        self.ref_sequences_2d = ref_alt_bcl
        # TODO: probably easier just to stack the entire ParentDatum LongTensor

        list_of_ref_tensors = [item.get_ref_reads_2d() for item in data]
        list_of_alt_tensors = [item.get_alt_reads_2d() for item in data]
        self.reads_2d = torch.from_numpy(np.vstack(list_of_ref_tensors + list_of_alt_tensors))
        self.info_2d = torch.from_numpy(np.vstack([reads_datum.get_info_1d() for reads_datum in data]))


        ref_counts = IntTensor([len(datum.reads_2d) - datum.alt_count for datum in data])
        alt_counts = IntTensor([datum.alt_count for datum in data])
        labels = IntTensor([1 if item.label == Label.ARTIFACT else 0 for item in data])
        is_labeled_mask = IntTensor([0 if item.label == Label.UNLABELED else 1 for item in data])
        sources = IntTensor([item.source for item in data])
        variant_types = IntTensor([datum.get_variant_type() for datum in data])
        self.int_tensor = torch.vstack((ref_counts, alt_counts, labels, is_labeled_mask, sources, variant_types))

        self._size = len(data)

    # pin memory for all tensors that are sent to the GPU
    def pin_memory(self):
        self.ref_sequences_2d = self.ref_sequences_2d.pin_memory()
        self.reads_2d = self.reads_2d.pin_memory()
        self.info_2d = self.info_2d.pin_memory()
        self.int_tensor = self.int_tensor.pin_memory()

        return self

    def copy_to(self, device, dtype):
        is_cuda = device.type == 'cuda'
        # For all non-tensor attributes, shallow copy is sufficient
        new_batch = copy.copy(self)
        new_batch.ref_sequences_2d = self.ref_sequences_2d.to(device=device, dtype=dtype, non_blocking=is_cuda)
        new_batch.reads_2d = self.reads_2d.to(device=device, dtype=dtype, non_blocking=is_cuda)
        new_batch.info_2d = self.info_2d.to(device=device, dtype=dtype, non_blocking=is_cuda)
        new_batch.int_tensor = self.int_tensor.to(device=device, non_blocking=is_cuda)
        return new_batch

    def original_list(self):
        return self._original_list

    def get_reads_2d(self) -> Tensor:
        return self.reads_2d

    def get_ref_counts(self) -> IntTensor:
        return self.int_tensor[0, :]

    def get_alt_counts(self) -> IntTensor:
        return self.int_tensor[1, :]

    # the original IntEnum format
    def get_labels(self):
        return self.int_tensor[2, :]

    def get_training_labels(self):
        int_enum_labels = self.get_labels()
        return 1.0 * (int_enum_labels == Label.ARTIFACT) + 0.5 * (int_enum_labels == Label.UNLABELED)

    def get_is_labeled_mask(self) -> IntTensor:
        return self.int_tensor[3, :]

    def get_sources(self) -> IntTensor:
        return self.int_tensor[4, :]

    def get_variant_types(self) -> IntTensor:
        return self.int_tensor[5, :]

    def get_info_2d(self) -> Tensor:
        return self.info_2d

    def get_ref_sequences_2d(self) -> Tensor:
        return self.ref_sequences_2d

    def size(self) -> int:
        return self._size