from __future__ import annotations

from enum import IntEnum

import torch
from torch import Tensor, IntTensor

from permutect.data.batch import Batch
from permutect.data.count_binning import ref_count_bin_name, NUM_REF_COUNT_BINS, alt_count_bin_name, NUM_ALT_COUNT_BINS, \
    logit_bin_name, NUM_LOGIT_BINS, alt_count_bin_indices, ref_count_bin_indices, logit_bin_indices
from permutect.misc_utils import gpu_if_available
from permutect.utils.array_utils import flattened_indices
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
    def __init__(self, batch: Batch, logits: Tensor = None, sources_override: IntTensor=None):
        """
        sources override is used for something of a hack where in filtering there is only one source, so we use the
        source dimensipn to instead represent the call type
        """
        self.sources = batch.get_sources() if sources_override is None else sources_override
        self.labels = batch.get_labels()
        self.var_types = batch.get_variant_types()
        self.ref_counts = batch.get_ref_counts()
        self.ref_count_bins = ref_count_bin_indices(self.ref_counts)
        self.alt_counts = batch.get_alt_counts()
        self.alt_count_bins = alt_count_bin_indices(self.alt_counts)
        self.logit_bins = logit_bin_indices(logits) if logits is not None else None

        # We do something kind of dangerous-seeming here: sources is the zeroth dimension and so the formula for
        # flattened indices *doesn't depend on the number of sources* since the stride from one source to the next is the
        # product of all the *other* dimensionalities.  Thus we can set the zeroth dimension to anythiong we want!
        # Just to make sure that this doesn't cause a silent error, we set it to None so that things will blow up
        # if my little analysis here is wrong
        dims = (None, len(Label), len(Variation), NUM_REF_COUNT_BINS, NUM_ALT_COUNT_BINS)
        idx = (self.sources, self.labels, self.var_types, self.ref_count_bins, self.alt_count_bins)
        self.flattened_idx = flattened_indices(dims, idx)

    def index_into_tensor(self, tens: Tensor):
        """
        given 5D batch-indexed tensor x_slvra get the 1D tensor
        result[i] = x_slvra[source[i], label[i], variant type[i], ref bin[i], alt bin[i]]
        This is equivalent to flattening x and indexing by the cached flattened indices
        :return:
        """
        assert tens.dim == 5, "Tensor must have 5 dimensions: source, label, variant type, ref count, alt count"
        return tens.view(-1)[self.flattened_idx]

    def increment_tensor(self, tens: Tensor, values: Tensor):
        # Similar, but implements: x_slvra[source[i], label[i], variant type[i], ref bin[i], alt bin[i]] += values[i]
        # Addition is in-place. The flattened view(-1) shares memory with the original tensor
        assert tens.dim == 5, "Tensor must have 5 dimensions: source, label, variant type, ref count, alt count"
        return tens.view(-1).index_add_(dim=0, index=self.flattened_idx, source=values)


class BatchIndicesWithLogits:
    def __init__(self, batch: Batch, logits: Tensor = None, sources_override: IntTensor=None):
        self.indices_without_logits = batch.batch_indices()
        self.logit_bins = logit_bin_indices(logits)

        # because logits are the last index, the flattened indices with logits are related to those without in a simple way
        self.flattened_idx_with_logits = self.logit_bins + NUM_LOGIT_BINS * self.indices_without_logits.flattened_idx

    def index_into_tensor(self, tens: Tensor):
        """
        given 6D batch-indexed tensor x_slvrag get the 1D tensor
        result[i] = x_slvra[source[i], label[i], variant type[i], ref bin[i], alt bin[i], logit bin[i]]
        This is equivalent to flattening x and indexing by the cached flattened indices
        :return:
        """
        assert tens.dim == 6, "Tensor must have 6 dimensions: source, label, variant type, ref count, alt count, logit"
        return tens.view(-1)[self.flattened_idx_with_logits]

    def increment_tensor(self, tens: Tensor, values: Tensor):
        # Similar, but implements: x_slvra[source[i], label[i], variant type[i], ref bin[i], alt bin[i], logit bin[i]] += values[i]
        # Addition is in-place. The flattened view(-1) shares memory with the original tensor
        assert tens.dim == 6, "Tensor must have 6 dimensions: source, label, variant type, ref count, alt count, logit"
        return tens.view(-1).index_add_(dim=0, index=self.flattened_idx_with_logits, source=values)


def make_batch_indexed_tensor(num_sources: int, device=gpu_if_available(), include_logits: bool=False, value:float = 0):
    # make tensor with indices slvra and optionally g:
    # 's' (source), 'l' (Label), 'v' (Variation), 'r' (ref count bin), 'a' (aklt count bin), 'g' (logit bin)
    if include_logits:
        return value * torch.ones((num_sources, len(Label), len(Variation), NUM_REF_COUNT_BINS, NUM_ALT_COUNT_BINS, NUM_LOGIT_BINS), device=device)
    else:
        return value * torch.ones((num_sources, len(Label), len(Variation), NUM_REF_COUNT_BINS, NUM_ALT_COUNT_BINS, NUM_LOGIT_BINS), device=device)