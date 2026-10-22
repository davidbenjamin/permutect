import time
from typing import Iterator
from typing import Set

import cyvcf2
import psutil
import torch
from torch import Tensor
from torch import nn
from torch.nn import Parameter

from permutect.data.datum import Data
from permutect.data.datum import Datum
from permutect.utils.allele_utils import trim_alleles_on_right
from permutect.utils.allele_utils import truncate_bases_if_necessary


def report_memory_usage(message: str = ""):
    print(f"{message}  Memory usage: {psutil.virtual_memory().percent:.1f}%")


class ConsistentValue:
    """
    Tracks a value that once initialized, is consistent among eg all members of a dataset.  For example, all tensors
    must have the same number of columns.
    """

    def __init__(self, value=None):
        self.value = value

    def check(self, value):
        if self.value is None:
            self.value = value
        else:
            assert self.value == value, "inconsistent values"


class MutableInt:
    def __init__(self, value: int = 0):
        self.value = value

    def __str__(self):
        return str(self.value)

    def increment(self, amount: int = 1):
        self.value += amount

    def decrement(self, amount: int = 1):
        self.value -= amount

    def get_and_then_increment(self):
        self.value += 1
        return self.value - 1

    def get(self):
        return self.value

    def set(self, value: int):
        self.value = value


def gpu_if_available(exploit_mps=False) -> torch.device:
    if torch.cuda.is_available():
        d = "cuda"
    elif exploit_mps and torch.mps.is_available():
        d = "mps"
    else:
        d = "cpu"
    return torch.device(d)


def freeze(parameters):
    for parameter in parameters:
        parameter.requires_grad = False


def unfreeze(parameters):
    for parameter in parameters:
        if parameter.dtype.is_floating_point:  # an integer parameter isn't trainable by gradient descent
            parameter.requires_grad = True


class StreamingAverage:
    def __init__(self):
        self._count = 0.0
        self._sum = 0.0

    def is_empty(self) -> bool:
        return self._count == 0.0

    def get(self) -> float:
        return self._sum / (self._count + 0.0001)

    def record(self, value: float, weight: float = 1):
        self._count += weight
        self._sum += value * weight

    def record_sum(self, value_sum: float, count):
        self._count += count
        self._sum += value_sum

    # record only values masked as true
    def record_with_mask(self, values: Tensor, mask: Tensor):
        self._count += torch.sum(mask).item()
        self._sum += torch.sum(values * mask).item()

    # record values with different weights
    # values and mask should live on same device as self._sum
    def record_with_weights(self, values: Tensor, weights: Tensor):
        self._count += torch.sum(weights).item()
        self._sum += torch.sum(values * weights).item()


class Timer:
    def __init__(self, message: str = None):
        if message is not None:
            print(message)
        self._start_time = time.perf_counter()

    def report(self, message: str = ""):
        elapsed_time = time.perf_counter() - self._start_time
        print(f"{message}: {elapsed_time:0.4f} seconds")


def backpropagate(optimizer: torch.optim.Optimizer, loss: Tensor, params_to_clip: Iterator[Parameter] = []):
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(params_to_clip, max_norm=1.0)
    optimizer.step()


def get_first_numeric_element(variant, key):
    tuple_or_scalar = variant.INFO[key]
    return tuple_or_scalar[0] if type(tuple_or_scalar) is tuple else tuple_or_scalar


def encode(contig: str, position: int, ref: str, alt: str):
    # TODO: contigs stored as integer index must be converted back to string to compare VCF variants with dataset variants!!!
    trimmed_ref, trimmed_alt = trim_alleles_on_right(ref, alt)
    return contig + ":" + str(position) + ":" + truncate_bases_if_necessary(trimmed_alt)


def encode_datum(datum: Datum, contig_index_to_name_map):
    contig_name = contig_index_to_name_map[datum.get(Data.CONTIG)]
    return encode(contig_name, datum.get(Data.POSITION), datum.get_ref_allele(), datum.get_alt_allele())


def encode_variant(v: cyvcf2.Variant, zero_based=False):
    alt = v.ALT[0]  # TODO: we're assuming biallelic
    ref = v.REF
    start = (v.start + 1) if zero_based else v.start
    return encode(v.CHROM, start, ref, alt)


def overlapping_filters(v: cyvcf2.Variant, filters_set: Set[str]) -> Set[str]:
    return set([]) if v.FILTER is None else set(v.FILTER.split(";")).intersection(filters_set)


def get_tumor_sample_index(vcf: cyvcf2.VCF) -> int:
    # This is specific to Mutect2, which outputs a line:
    # ##tumor_sample=___________
    tumor_sample_name = ""
    all_samples = []
    for header_line in vcf.raw_header.split("\n"):
        if header_line.startswith("##tumor_sample"):
            tumor_sample_name = header_line.split("=")[-1]
            print(tumor_sample_name)
        elif header_line.startswith("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t"):
            prefix = "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t"
            all_samples = header_line[len(prefix) :].split()
    tumor_sample_index = 0
    for n, sample in enumerate(all_samples):
        if sample == tumor_sample_name:
            print(f"tumor sample found in VCF header: name {sample}, index {n}")
            tumor_sample_index = n
    return tumor_sample_index


def check_for_nan(model: torch.nn.Module):
    nan_found = False
    for name, param in model.named_parameters():
        if param.grad is not None:
            if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                print(f"Invalid gradient (NaN or Inf) found in parameter: {name}")
                nan_found = True
    assert not nan_found
