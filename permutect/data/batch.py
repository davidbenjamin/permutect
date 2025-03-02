from __future__ import annotations

from enum import IntEnum
from typing import List, Tuple

import torch
from matplotlib import pyplot as plt
from torch import IntTensor, FloatTensor, Tensor
import numpy as np

from permutect.data.count_binning import ref_count_bin_name, NUM_REF_COUNT_BINS, alt_count_bin_name, NUM_ALT_COUNT_BINS, \
    logit_bin_name, NUM_LOGIT_BINS, ref_count_bin_indices, alt_count_bin_indices, logit_bin_indices, \
    ref_count_bin_index, alt_count_bin_index, logits_from_bin_indices
from permutect.data.datum import Datum
from permutect.metrics import plotting
from permutect.misc_utils import gpu_if_available
from permutect.utils.array_utils import flattened_indices, select_and_sum
from permutect.utils.enums import Label, Variation


class Batch:
    def __init__(self, data: List[Datum]):
        self.data = torch.from_numpy(np.vstack([d.get_array_1d() for d in data])).to(dtype=torch.long)
        self._finish_initializiation_from_data_array()

    def _finish_initializiation_from_data_array(self):
        self._size = len(self.data)
        self.haplotypes_start = Datum.HAPLOTYPES_START_IDX
        self.haplotypes_end = Datum.HAPLOTYPES_START_IDX + self.data[0, Datum.HAPLOTYPES_LENGTH_IDX]
        self.info_start = self.haplotypes_end
        info_length = self.data[0, Datum.INFO_LENGTH_IDX]
        self.info_end = self.info_start + info_length
        self.lazy_batch_indices = None

    def batch_indices(self) -> BatchIndices:
        if self.lazy_batch_indices is not None:
            return self.lazy_batch_indices
        else:
            self.lazy_batch_indices = BatchIndices(sources=self.get_sources(), labels=self.get_labels(),
                var_types=self.get_variant_types(), ref_counts=self.get_ref_counts(), alt_counts=self.get_alt_counts())
            return self.lazy_batch_indices

    # get the original IntEnum format (VARIANT = 0, ARTIFACT = 1, UNLABELED = 2) labels
    def get_labels(self) -> IntTensor:
        return self.data[:, Datum.LABEL_IDX]

    # convert to the training format of 0.0 / 0.5 / 1.0 for variant / unlabeled / artifact
    # the 0.5 for unlabeled data is reasonable but should never actually be used due to the is_labeled mask
    def get_training_labels(self) -> FloatTensor:
        int_enum_labels = self.get_labels()
        return 1.0 * (int_enum_labels == Label.ARTIFACT) + 0.5 * (int_enum_labels == Label.UNLABELED)

    def get_is_labeled_mask(self) -> IntTensor:
        int_enum_labels = self.get_labels()
        return (int_enum_labels != Label.UNLABELED).int()

    def get_sources(self) -> IntTensor:
        return self.data[:, Datum.SOURCE_IDX]

    def get_variant_types(self) -> IntTensor:
        result = self.data[:, Datum.VARIANT_TYPE_IDX]
        return result

    def get_ref_counts(self) -> IntTensor:
        return self.data[:, Datum.REF_COUNT_IDX]

    def get_alt_counts(self) -> IntTensor:
        return self.data[:, Datum.ALT_COUNT_IDX]

    def get_original_alt_counts(self) -> IntTensor:
        return self.data[:, Datum.ORIGINAL_ALT_COUNT_IDX]

    def get_original_depths(self) -> IntTensor:
        return self.data[:, Datum.ORIGINAL_DEPTH_IDX]

    def get_original_normal_alt_counts(self) -> IntTensor:
        return self.data[:, Datum.ORIGINAL_NORMAL_ALT_COUNT_IDX]

    def get_original_normal_depths(self) -> IntTensor:
        return self.data[:, Datum.ORIGINAL_NORMAL_DEPTH_IDX]

    def get_info_be(self) -> Tensor:
        return self.data[:, self.info_start:self.info_end] / Datum.FLOAT_TO_LONG_MULTIPLIER

    def get_haplotypes_bs(self) -> IntTensor:
        # each row is 1D array of integer array reference and alt haplotypes concatenated -- A, C, G, T, deletion = 0, 1, 2, 3, 4
        return self.data[:, self.haplotypes_start:self.haplotypes_end]

    # pin memory for all tensors that are sent to the GPU
    def pin_memory(self):
        self.data = self.data.pin_memory()
        return self

    def get_data_be(self) -> np.ndarray:
        return self.data.cpu().numpy()

    def size(self) -> int:
        return self._size


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
    PRODUCT_OF_NON_SOURCE_DIMS_INCLUDING_LOGITS = len(Label) * len(Variation) * NUM_REF_COUNT_BINS * NUM_ALT_COUNT_BINS * NUM_LOGIT_BINS

    def __init__(self, sources: IntTensor, labels: IntTensor, var_types: IntTensor, ref_counts: IntTensor, alt_counts:IntTensor):
        """
        sources override is used for something of a hack where in filtering there is only one source, so we use the
        source dimensipn to instead represent the call type
        """
        self.sources = sources
        self.labels = labels
        self.var_types = var_types
        self.ref_count_bins = ref_count_bin_indices(ref_counts)
        self.alt_count_bins = alt_count_bin_indices(alt_counts)

        # We do something kind of dangerous-seeming here: sources is the zeroth dimension and so the formula for
        # flattened indices *doesn't depend on the number of sources* since the stride from one source to the next is the
        # product of all the *other* dimensionalities.  Thus we can set the zeroth dimension to anythiong we want!
        # Just to make sure that this doesn't cause a silent error, we set it to None so that things will blow up
        # if my little analysis here is wrong
        dims = (None, len(Label), len(Variation), NUM_REF_COUNT_BINS, NUM_ALT_COUNT_BINS)
        idx = (self.sources, self.labels, self.var_types, self.ref_count_bins, self.alt_count_bins)
        self.flattened_idx = flattened_indices(dims, idx)

    def _flattened_idx_with_logits(self, logits: Tensor):
        # because logits are the last index, the flattened indices with logits are related to those without in a simple way
        logit_bins = logit_bin_indices(logits)
        return logit_bins + NUM_LOGIT_BINS * self.flattened_idx

    def index_into_tensor(self, tens: Tensor, logits: Tensor = None):
        """
        given 5D batch-indexed tensor x_slvra get the 1D tensor
        result[i] = x_slvra[source[i], label[i], variant type[i], ref bin[i], alt bin[i]]
        This is equivalent to flattening x and indexing by the cached flattened indices
        :return:
        """
        if logits is None:
            assert tens.dim() == 5, "Tensor must have 5 dimensions: source, label, variant type, ref count, alt count"
            return tens.view(-1)[self.flattened_idx]
        else:
            assert tens.dim() == 6, "Tensor must have 6 dimensions: source, label, variant type, ref count, alt count, logit"
            return tens.view(-1)[self._flattened_idx_with_logits(logits)]

    def increment_tensor(self, tens: Tensor, values: Tensor, logits: Tensor = None):
        # Similar, but implements: x_slvra[source[i], label[i], variant type[i], ref bin[i], alt bin[i]] += values[i]
        # Addition is in-place. The flattened view(-1) shares memory with the original tensor
        if logits is None:
            assert tens.dim() == 5, "Tensor must have 5 dimensions: source, label, variant type, ref count, alt count"
            return tens.view(-1).index_add_(dim=0, index=self.flattened_idx, source=values)
        else:
            assert tens.dim() == 6, "Tensor must have 6 dimensions: source, label, variant type, ref count, alt count, logit"
            return tens.view(-1).index_add_(dim=0, index=self._flattened_idx_with_logits(logits), source=values)

    def increment_tensor_with_sources_and_logits(self, tens: Tensor, values: Tensor, sources_override: IntTensor, logits: Tensor):
        # we sometimes need to override the sources (in filter_variants.py there is a hack where we use the Call type
        # in place of the sources).  This is how we do that.
        assert tens.dim() == 6, "Tensor must have 6 dimensions: source, label, variant type, ref count, alt count, logit"
        indices_with_logits = self._flattened_idx_with_logits(logits)

        # eg, if the dimensions after source are 2, 3, 4 then every increase of the source by 1 is accompanied by an increase
        # of 2x3x4 = 24 in the flattened indices.
        indices = indices_with_logits + BatchIndices.PRODUCT_OF_NON_SOURCE_DIMS_INCLUDING_LOGITS * (sources_override - self.sources)
        return tens.view(-1).index_add_(dim=0, index=indices, source=values)


def make_batch_indexed_tensor(num_sources: int, device=gpu_if_available(), include_logits: bool=False, value:float = 0):
    # make tensor with indices slvra and optionally g:
    # 's' (source), 'l' (Label), 'v' (Variation), 'r' (ref count bin), 'a' (aklt count bin), 'g' (logit bin)
    if include_logits:
        return value * torch.ones((num_sources, len(Label), len(Variation), NUM_REF_COUNT_BINS, NUM_ALT_COUNT_BINS, NUM_LOGIT_BINS), device=device)
    else:
        return value * torch.ones((num_sources, len(Label), len(Variation), NUM_REF_COUNT_BINS, NUM_ALT_COUNT_BINS), device=device)


class BatchIndexedTensor:
    NUM_DIMS = 6
    """
    stores sums, indexed by batch properties source (s), label (l), variant type (v), ref count (r), alt count (a), logit (g)
    """
    def __init__(self, num_sources: int, device=gpu_if_available(), include_logits: bool = False):
        # note: if include logits is false, the g dimension is absent
        self.tens = make_batch_indexed_tensor(num_sources=num_sources, device=device, include_logits=include_logits, value=0)
        self.include_logits = include_logits
        self.device = device
        self.has_been_sent_to_cpu = False
        self.num_sources = num_sources

    def put_on_cpu(self):
        """
        Do this at the end of an epoch so that the whole tensor is on CPU in one operation rather than computing various
        marginals etc on GPU and sending them each to CPU for plotting etc.
        :return:
        """
        self.tens = self.tens.cpu()
        self.device = torch.device('cpu')
        self.has_been_sent_to_cpu = True
        return self

    def resize_sources(self, new_num_sources):
        old_num_sources = self.num_sources
        if new_num_sources < old_num_sources:
            self.tens = self.tens[:new_num_sources]
        elif new_num_sources > old_num_sources:
            new_totals = make_batch_indexed_tensor(num_sources=new_num_sources, device=self.device, include_logits=self.include_logits, value=0)
            new_totals[:old_num_sources] = self.tens
            self.tens = new_totals
        self.num_sources = new_num_sources

    def record_datum(self, datum: Datum, value: float = 1.0, grow_source_if_necessary: bool = True):
        assert not self.include_logits, "this only works when not including logits"
        source = datum.get_source()
        if source >= self.num_sources:
            if grow_source_if_necessary:
                self.resize_sources(source + 1)
            else:
                raise Exception("Datum source doesn't fit.")
        # no logits here
        ref_idx, alt_idx = ref_count_bin_index(datum.get_ref_count()), alt_count_bin_index(datum.get_alt_count())
        self.tens[source, datum.get_label(), datum.get_variant_type(), ref_idx, alt_idx] += value

    def record(self, batch: Batch, values: Tensor, logits: Tensor=None):
        assert not self.has_been_sent_to_cpu, "Can't record after already sending to CPU"
        batch.batch_indices().increment_tensor(self.tens, values=values, logits=logits)

    def record_with_sources_and_logits(self, batch: Batch, values: Tensor, sources_override: IntTensor, logits: Tensor):
        assert not self.has_been_sent_to_cpu, "Can't record after already sending to CPU"
        assert self.include_logits, "Tensor lacks a logit dimension"
        batch.batch_indices().increment_tensor_with_sources_and_logits(self.tens, values=values, sources_override=sources_override, logits=logits)

    def get_totals(self) -> Tensor:
        return self.tens

    def get_marginal(self, *properties: Tuple[BatchProperty, ...]) -> Tensor:
        """
        sum over all but one or more batch properties.
        For example self.get_marginal(BatchProperty.SOURCE, BatchProperty.LABEL) yields a (num sources x len(Label)) output
        """
        property_set = set(*properties)
        num_dims = len(BatchProperty) - (0 if self.include_logits else 1)
        other_dims = tuple(n for n in range(num_dims) if n not in property_set)
        return torch.sum(self.tens, dim=other_dims)

