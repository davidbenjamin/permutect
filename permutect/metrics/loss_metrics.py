from enum import IntEnum
from math import floor
from typing import Tuple

import torch
from torch import IntTensor
from torch.utils.tensorboard import SummaryWriter

from permutect.data.batch import Batch
from permutect.data.reads_dataset import MAX_REF_COUNT, MAX_ALT_COUNT
from permutect.misc_utils import gpu_if_available
from permutect.utils.array_utils import add_to_5d_array, add_to_6d_array
from permutect.utils.enums import Variation, Epoch, Label

# count bins are {0-2}, {3-5}, {6-8} etc up to the max alt and ref counts defined in ReadsDataset
COUNT_BIN_SKIP = 3
NUM_REF_COUNT_BINS = (MAX_REF_COUNT // COUNT_BIN_SKIP) + 1 # eg if max count is 10, the 10//3 + 1 = 4 bins are {0-2}, {3-5},{6-8},{9-10}
NUM_ALT_COUNT_BINS = (MAX_ALT_COUNT // COUNT_BIN_SKIP) + 1 # eg if max count is 10, the 10//3 + 1 = 4 bins are {0-2}, {3-5},{6-8},{9-10}

LOGIT_BIN_SKIP = 1
MIN_LOGIT, MAX_LOGIT = -10, 10
NUM_LOGIT_BINS = floor((MAX_LOGIT - MIN_LOGIT) / LOGIT_BIN_SKIP) + 1


def logit_bin_indices(logits_tensor: torch.Tensor) -> IntTensor:
    return torch.div(torch.clamp(logits_tensor, min=MIN_LOGIT, max=MAX_LOGIT) - MIN_LOGIT, LOGIT_BIN_SKIP, rounding_mode='floor')


def logit_bin_name(logit_bin_idx: int) -> str:
    return f"{MIN_LOGIT + (logit_bin_idx + 0.5) * LOGIT_BIN_SKIP}:.1f"


def count_bin_indices(count_tensor: IntTensor) -> IntTensor:
    return torch.div(count_tensor, COUNT_BIN_SKIP, rounding_mode='floor')


def count_bin_name(bin_idx: int) -> str:
    return str(COUNT_BIN_SKIP * bin_idx + (COUNT_BIN_SKIP-1)//2)  # the center of the bin


class BatchProperty(IntEnum):
    SOURCE = (0, None)
    LABEL = (1, [label.name for label in Label])
    VARIANT_TYPE = (2, [var_type.name for var_type in Variation])
    REF_COUNT_BIN = (3, [count_bin_name(idx) for idx in range(NUM_REF_COUNT_BINS)])
    ALT_COUNT_BIN = (4, [count_bin_name(idx) for idx in range(NUM_ALT_COUNT_BINS)])
    LOGIT_BIN = (5, [logit_bin_name(idx) for idx in range(NUM_LOGIT_BINS)])

    def __new__(cls, value, names_list):
        member = int.__new__(cls, value)
        member._value_ = value
        member.names_list = names_list
        return member

    def get_name(self, n: int):
        return str(n) if self.names_list is None else self.names_list[n]


class BatchIndexedTotals:
    NUM_DIMS = 6
    """
    stores sums, indexed by batch properties source (s), label (l), variant type (v), ref count (r), alt count (a), logit (g)
    """
    def __init__(self, num_sources: int, device=gpu_if_available(), include_logits: bool = False):
        self.totals_slvrag = torch.zeros((num_sources, len(Label), len(Variation), NUM_REF_COUNT_BINS, NUM_ALT_COUNT_BINS, NUM_LOGIT_BINS), device=device)
        assert self.totals_slvrag.dim() == len(BatchProperty)
        self.include_logits = include_logits

    def record(self, batch: Batch, logits: torch.Tensor, values: torch.Tensor):
        # values is a 1D tensor
        assert batch.size() == len(values)
        sources = batch.get_sources()
        logit_indices = logit_bin_indices(logits) if self.include_logits else torch.zeros_like(sources)

        add_to_6d_array(self.totals_slvrag, sources , batch.get_labels(), batch.get_variant_types(),
                        count_bin_indices(batch.get_ref_counts()), count_bin_indices(batch.get_alt_counts()), logit_indices, values)

    def get_totals(self) -> torch.Tensor:
        return self.totals_slvrag

    def get_marginal(self, *properties: Tuple[BatchProperty, ...]) -> torch.Tensor:
        """
        sum over all but one or more batch properties.
        For example self.get_marginal(BatchProperty.SOURCE, BatchProperty.LABEL) yields a (num sources x len(Label)) output
        """
        property_set = set(*properties)
        other_dims = tuple(n for n in range(BatchIndexedTotals.NUM_DIMS) if n not in property_set)
        return torch.sum(self.totals_slvrag, dim=other_dims)


class BatchIndexedAverages:
    def __init__(self, num_sources: int, device=gpu_if_available(), include_logits: bool = False):
        self.totals = BatchIndexedTotals(num_sources, device, include_logits)
        self.counts = BatchIndexedTotals(num_sources, device, include_logits)
        self.include_logits = include_logits

    def record(self, batch: Batch, logits: torch.Tensor, values: torch.Tensor, weights: torch.Tensor):
        self.totals.record(batch, logits, (values*weights).detach())
        self.counts.record(batch, logits, weights.detach())

    def get_averages(self) -> torch.Tensor:
        return self.totals.get_totals() / (0.001 + self.counts.get_totals())

    def get_marginal(self, *properties: Tuple[BatchProperty, ...]) -> torch.Tensor:
        return self.totals.get_marginal(properties) / self.counts.get_marginal(properties)

    def report_marginals(self, message: str):
        print(message)
        batch_property: BatchProperty
        for batch_property in BatchProperty:
            if batch_property == BatchProperty.LOGIT_BIN and not self.include_logits:
                continue
            values = self.get_marginal(batch_property).tolist()
            print(f"Marginalizing by {batch_property.name}")
            for n, ave in enumerate(values):
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


class AccuracyMetrics(BatchIndexedAverages):
    """
    Record should be called with values=tensor of 1 if correct, 0 if incorrect.  Accuracies are the averages of the correctness

    ROC curves can also be generated by calculating with labels and logit cumulative sums

    calibration can be done with accuracy vs logit
    """