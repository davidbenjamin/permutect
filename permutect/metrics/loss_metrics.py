from enum import IntEnum
from typing import Tuple

import torch
from torch import IntTensor
from torch.utils.tensorboard import SummaryWriter

from permutect.data.batch import Batch
from permutect.data.reads_dataset import MAX_REF_COUNT, MAX_ALT_COUNT
from permutect.misc_utils import gpu_if_available
from permutect.utils.array_utils import add_to_5d_array
from permutect.utils.enums import Variation, Epoch, Label

# count bins are {0-2}, {3-5}, {6-8} etc up to the max alt and ref counts defined in ReadsDataset
COUNT_BIN_SKIP = 3
NUM_REF_COUNT_BINS = (MAX_REF_COUNT // COUNT_BIN_SKIP) + 1 # eg if max count is 10, the 10//3 + 1 = 4 bins are {0-2}, {3-5},{6-8},{9-10}
NUM_ALT_COUNT_BINS = (MAX_ALT_COUNT // COUNT_BIN_SKIP) + 1 # eg if max count is 10, the 10//3 + 1 = 4 bins are {0-2}, {3-5},{6-8},{9-10}


def bin_index(count: int) -> int:
    return count // COUNT_BIN_SKIP


def bin_indices(count_tensor: IntTensor) -> IntTensor:
    return torch.div(count_tensor, COUNT_BIN_SKIP, rounding_mode='floor')


def bin_name(bin_idx: int) -> str:
    return str(COUNT_BIN_SKIP * bin_idx + (COUNT_BIN_SKIP-1)//2)  # the center of the bin


class BatchProperty(IntEnum):
    SOURCE = (0, None)
    LABEL = (1, [label.name for label in Label])
    VARIANT_TYPE = (2, [var_type.name for var_type in Variation])
    REF_COUNT_BIN = (3, [bin_name(idx) for idx in range(NUM_REF_COUNT_BINS)])
    ALT_COUNT_BIN = (4, [bin_name(idx) for idx in range(NUM_ALT_COUNT_BINS)])

    def __new__(cls, value, names_list):
        member = int.__new__(cls, value)
        member._value_ = value
        member.names_list = names_list
        return member

    #def __init__(self, value, names_list):
    #    super().__init__()
    #    self.value = value
    #    self.names_list = names_list

    def get_name(self, n: int):
        return str(n) if self.names_list is None else self.names_list[n]


class BatchIndexedTotals:
    NUM_DIMS = 5
    """
    stores sums, indexed by batch properties source (s), label (l), variant type (v), ref count (r), alt count (a)
    """
    def __init__(self, num_sources: int, device=gpu_if_available()):
        self.totals_slvra = torch.zeros((num_sources, len(Label), len(Variation), NUM_REF_COUNT_BINS, NUM_ALT_COUNT_BINS), device=device)
        assert self.totals_slvra.dim() == len(BatchProperty)

    def record(self, batch: Batch, values: torch.Tensor):
        # values is a 1D tensor
        assert batch.size() == len(values)

        add_to_5d_array(self.totals_slvra, batch.get_sources(), batch.get_labels(), batch.get_variant_types(),
                        bin_indices(batch.get_ref_counts()), bin_indices(batch.get_alt_counts()), values)

    def get_totals(self) -> torch.Tensor:
        return self.totals_slvra

    def get_marginal(self, *properties: Tuple[BatchProperty, ...]) -> torch.Tensor:
        """
        sum over all but one or more batch properties.
        For example self.get_marginal(BatchProperty.SOURCE, BatchProperty.LABEL) yields a (num sources x len(Label)) output
        """
        property_set = set(*properties)
        other_dims = tuple(n for n in range(BatchIndexedTotals.NUM_DIMS) if n not in property_set)
        return torch.sum(self.totals_slvra, dim=other_dims)


class BatchIndexedAverages:
    def __init__(self, num_sources: int, device=gpu_if_available()):
        self.totals = BatchIndexedTotals(num_sources, device)
        self.counts = BatchIndexedTotals(num_sources, device)

    def record(self, batch: Batch, values: torch.Tensor, weights: torch.Tensor):
        self.totals.record(batch, (values*weights).detach())
        self.counts.record(batch, weights.detach())

    def get_averages(self) -> torch.Tensor:
        return self.totals.get_totals() / (0.001 + self.counts.get_totals())

    def get_marginal(self, *properties: Tuple[BatchProperty, ...]) -> torch.Tensor:
        return self.totals.get_marginal(properties) / self.counts.get_marginal(properties)

    def report_marginals(self, message: str):
        print(message)
        batch_property: BatchProperty
        for batch_property in BatchProperty:
            print(f"Marginalizing by {batch_property.name}")
            for n, ave in enumerate(self.get_marginal(batch_property).tolist()):
                print(f"{batch_property.get_name(n)}: {ave:.3f}")

    def write_to_summary_writer(self, epoch_type: Epoch, epoch: int, summary_writer: SummaryWriter, prefix: str):
        """
        write marginals for every batch property
        :return:
        """
        batch_property: BatchProperty
        for batch_property in BatchProperty:
            marginals = self.get_marginal(batch_property)
            for n, average in enumerate(marginals.tolist()):
                heading = f"{prefix}/{epoch_type.name}/{batch_property.name}/{batch_property.get_name(n)}"
                summary_writer.add_scalar(heading, average, epoch)
