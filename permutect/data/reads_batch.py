from __future__ import annotations

import copy
from typing import List

import numpy as np
import torch
from torch import Tensor

from permutect.data.batch import Batch
from permutect.data.reads_datum import ReadsDatum


class ReadsBatch(Batch):
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
        super().__init__(data)
        list_of_ref_tensors = [item.get_ref_reads_2d() for item in data]
        list_of_alt_tensors = [item.get_alt_reads_2d() for item in data]
        self.reads_2d = torch.from_numpy(np.vstack(list_of_ref_tensors + list_of_alt_tensors))

    # pin memory for all tensors that are sent to the GPU
    def pin_memory(self):
        super().pin_memory()
        self.reads_2d = self.reads_2d.pin_memory()

        return self

    def copy_to(self, device, dtype):
        is_cuda = device.type == 'cuda'
        new_batch = copy.copy(self)
        new_batch.reads_2d = self.reads_2d.to(device=device, dtype=dtype, non_blocking=is_cuda)
        new_batch.data = self.data.to(device, non_blocking=is_cuda)  # don't cast dtype -- needs to stay integral!
        return new_batch

    def get_reads_2d(self) -> Tensor:
        return self.reads_2d

    def get_one_hot_haplotypes_2d(self) -> Tensor:
        num_channels = 5
        # each row of haplotypes_2d is a ref haplotype concatenated horizontally with an alt haplotype of equal length
        # indices are b for batch, s index along DNA sequence, and later c for one-hot channel
        # h denotes horizontally concatenated sequences, first ref, then alt
        haplotypes_bh = self.get_haplotypes_2d()
        batch_size = len(haplotypes_bh)
        seq_length = haplotypes_bh.shape[1] // 2 # ref and alt have equal length and are h-stacked

        # num_classes = 5 for A, C, G, T, and deletion / insertion
        one_hot_haplotypes_bhc = torch.nn.functional.one_hot(haplotypes_bh, num_classes=num_channels)
        one_hot_haplotypes_bch = torch.permute(one_hot_haplotypes_bhc, (0, 2, 1))

        # interleave the 5 channels of ref and 5 channels of alt with a reshape
        # for each batch index we get 10 rows: the ref A channel sequence, then the alt A channel, then the ref C channel etc
        return one_hot_haplotypes_bch.reshape(batch_size, 2 * num_channels, seq_length)

    # useful for regenerating original data, for example in pruning.  Each original datum has its own reads_2d of ref
    # followed by alt
    def get_list_of_reads_2d(self):
        ref_counts, alt_counts = self.get_ref_counts(), self.get_alt_counts()
        total_ref = torch.sum(ref_counts).item()
        ref_reads, alt_reads = self.reads_2d[:total_ref], self.reads_2d[total_ref:]
        ref_splits, alt_splits = torch.cumsum(ref_counts)[:-1], torch.cumsum(alt_counts)[:-1]
        ref_list, alt_list = torch.tensor_split(ref_reads, ref_splits), torch.tensor_split(alt_reads, alt_splits)
        return [torch.vstack((refs, alts)).numpy() for refs, alts in zip(ref_list, alt_list)]

