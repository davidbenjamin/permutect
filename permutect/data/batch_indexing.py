from __future__ import annotations

from enum import IntEnum
from typing import List, Tuple

import torch
from matplotlib import pyplot as plt
from torch import Tensor, IntTensor

from permutect.data.batch import Batch
from permutect.data.count_binning import ref_count_bin_name, NUM_REF_COUNT_BINS, alt_count_bin_name, NUM_ALT_COUNT_BINS, \
    logit_bin_name, NUM_LOGIT_BINS, alt_count_bin_indices, ref_count_bin_indices, logit_bin_indices, \
    ref_count_bin_index, alt_count_bin_index, logits_from_bin_indices
from permutect.data.datum import Datum
from permutect.metrics import plotting
from permutect.misc_utils import gpu_if_available
from permutect.utils.array_utils import flattened_indices, select_and_sum
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
    PRODUCT_OF_NON_SOURCE_DIMS_INCLUDING_LOGITS = len(Label) * len(Variation) * NUM_REF_COUNT_BINS * NUM_ALT_COUNT_BINS * NUM_LOGIT_BINS

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
            assert tens.dim == 5, "Tensor must have 5 dimensions: source, label, variant type, ref count, alt count"
            return tens.view(-1)[self.flattened_idx]
        else:
            assert tens.dim == 6, "Tensor must have 6 dimensions: source, label, variant type, ref count, alt count, logit"
            return tens.view(-1)[self._flattened_idx_with_logits(logits)]

    def increment_tensor(self, tens: Tensor, values: Tensor, logits: Tensor = None):
        # Similar, but implements: x_slvra[source[i], label[i], variant type[i], ref bin[i], alt bin[i]] += values[i]
        # Addition is in-place. The flattened view(-1) shares memory with the original tensor
        if logits is None:
            assert tens.dim == 5, "Tensor must have 5 dimensions: source, label, variant type, ref count, alt count"
            return tens.view(-1).index_add_(dim=0, index=self.flattened_idx, source=values)
        else:
            assert tens.dim == 6, "Tensor must have 6 dimensions: source, label, variant type, ref count, alt count, logit"
            return tens.view(-1).index_add_(dim=0, index=self._flattened_idx_with_logits(logits), source=values)

    def increment_tensor_with_sources_and_logits(self, tens: Tensor, values: Tensor, sources_override: IntTensor, logits: Tensor):
        # we sometimes need to override the sources (in filter_variants.py there is a hack where we use the Call type
        # in place of the sources).  This is how we do that.
        assert tens.dim == 6, "Tensor must have 6 dimensions: source, label, variant type, ref count, alt count, logit"
        indices_with_logits = self._flattened_idx_with_logits(logits)

        # eg, if the dimensions after source are 2, 3, 4 then every increase of the source by 1 is accompanied by an increase
        # of 2x3x4 = 24 in the flattened indices.
        indices = BatchIndices.PRODUCT_OF_NON_SOURCE_DIMS_INCLUDING_LOGITS * (sources_override - self.sources)
        return tens.view(-1).index_add_(dim=0, index=indices, source=values)


def make_batch_indexed_tensor(num_sources: int, device=gpu_if_available(), include_logits: bool=False, value:float = 0):
    # make tensor with indices slvra and optionally g:
    # 's' (source), 'l' (Label), 'v' (Variation), 'r' (ref count bin), 'a' (aklt count bin), 'g' (logit bin)
    if include_logits:
        return value * torch.ones((num_sources, len(Label), len(Variation), NUM_REF_COUNT_BINS, NUM_ALT_COUNT_BINS, NUM_LOGIT_BINS), device=device)
    else:
        return value * torch.ones((num_sources, len(Label), len(Variation), NUM_REF_COUNT_BINS, NUM_ALT_COUNT_BINS, NUM_LOGIT_BINS), device=device)


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

    def split_over_sources(self) -> List[BatchIndexedTensor]:
        # split into single-source BatchIndexedTotals
        result = []
        for source in range(self.num_sources):
            element = BatchIndexedTensor(num_sources=1, device=self.device, include_logits=self.include_logits)
            element.tens[0].copy_(self.tens[source])
            result.append(element)
        return result

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

    def make_logit_histograms(self):
        assert self.has_been_sent_to_cpu, "Can't make plots before sending to CPU"
        fig, axes = plt.subplots(len(Variation), NUM_ALT_COUNT_BINS, sharex='all', sharey='all', squeeze=False,
                                 figsize=(2.5 * NUM_ALT_COUNT_BINS, 2.5 * len(Variation)), dpi=200)
        x_axis_logits = logits_from_bin_indices(torch.tensor(range(NUM_LOGIT_BINS)))

        num_sources = self.tens.shape[BatchProperty.SOURCE]
        multiple_sources = num_sources > 1
        for row, variation_type in enumerate(Variation):
            for count_bin in range(NUM_ALT_COUNT_BINS): # this is also the column index
                selection={BatchProperty.VARIANT_TYPE: variation_type, BatchProperty.ALT_COUNT_BIN: count_bin}
                totals_slg = select_and_sum(self.tens, select=selection, sum=(BatchProperty.REF_COUNT_BIN,))

                # The normalizing factor for each source, label is the sum over all logits for that source and label
                # This renders a histogram into a sort of probability density plot for each source, label
                normalization_slg = torch.sum(totals_slg, dim=-1, keepdim=True)
                normalized_totals_slg = (totals_slg + 0.000001) / normalization_slg

                # overlapping line plots for all source / label combinations
                # source 0 is filled; others are not
                ax = axes[row, count_bin]
                x_y_label_tuples = []
                for source in range(num_sources):
                    for label in Label:
                        if normalization_slg[source, label, 0].item() >= 1:
                            line_label = f"{label.name} ({source})" if multiple_sources else label.name
                            x_y_label_tuples.append((x_axis_logits.cpu().numpy(), normalized_totals_slg[source, label].cpu().numpy(), line_label))
                plotting.simple_plot_on_axis(ax, x_y_label_tuples, None, None)
                ax.legend()

        column_names = [alt_count_bin_name(count_idx) for count_idx in range(NUM_ALT_COUNT_BINS)]
        row_names = [var_type.name for var_type in Variation]
        plotting.tidy_subplots(fig, axes, x_label="predicted logit", y_label="frequency", row_labels=row_names, column_labels=column_names)
        return fig, axes