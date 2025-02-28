from __future__ import annotations

from enum import IntEnum

import torch
from torch import Tensor

from permutect.data.batch import Batch
from permutect.data.count_binning import ref_count_bin_name, NUM_REF_COUNT_BINS, alt_count_bin_name, NUM_ALT_COUNT_BINS, \
    logit_bin_name, NUM_LOGIT_BINS, alt_count_bin_indices, ref_count_bin_indices, logit_bin_indices
from permutect.misc_utils import gpu_if_available
from permutect.utils.enums import Label, Variation


class BatchProperty(IntEnum):
    SOURCE = (0, None)
    LABEL = (1, [label.name for label in Label])
    VARIANT_TYPE = (2, [var_type.name for var_type in Variation])
    REF_COUNT_BIN = (3, [ref_count_bin_name(idx) for idx in range(NUM_REF_COUNT_BINS)])
    ALT_COUNT_BIN = (4, [alt_count_bin_name(idx) for idx in range(NUM_ALT_COUNT_BINS)])
    LOGIT_BIN = (5, [logit_bin_name(idx) for idx in range(NUM_LOGIT_BINS)])

    def __new__(cls, value, names_list):
        member = int.__new__(cls, value)
        member._value_ = value
        member.names_list = names_list
        return member

    def get_name(self, n: int):
        return str(n) if self.names_list is None else self.names_list[n]


class BatchIndices:
    def __init__(self, batch: Batch, logits: Tensor = None):
        self.sources = batch.get_sources()
        self.labels = batch.get_labels()
        self.var_types = batch.get_variant_types()
        self.ref_counts = batch.get_ref_counts()
        self.ref_count_bins = ref_count_bin_indices(self.ref_counts)
        self.alt_counts = batch.get_alt_counts()
        self.alt_count_bins = alt_count_bin_indices(self.alt_counts)
        self.logit_bins = logit_bin_indices(logits) if logits is not None else None





def make_batch_indexed_tensor(num_sources: int, device=gpu_if_available(), include_logits: bool=False, value:float = 0):
    # make tensor with indices slvra and optionally g:
    # 's' (source), 'l' (Label), 'v' (Variation), 'r' (ref count bin), 'a' (aklt count bin), 'g' (logit bin)
    if include_logits:
        return value * torch.ones((num_sources, len(Label), len(Variation), NUM_REF_COUNT_BINS, NUM_ALT_COUNT_BINS, NUM_LOGIT_BINS), device=device)
    else:
        return value * torch.ones((num_sources, len(Label), len(Variation), NUM_REF_COUNT_BINS, NUM_ALT_COUNT_BINS, NUM_LOGIT_BINS), device=device)