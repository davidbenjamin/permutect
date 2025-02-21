from math import floor

import torch
from torch import IntTensor


# ref bins are {0-2}, {3-5}, {6-8} etc
# alt bins are {1-3}, {4-6}, {7-9} etc
MAX_REF_COUNT = 10
MIN_ALT_COUNT = 1
MAX_ALT_COUNT = 15

# for ROC curves, it is important to choose values such that logit == 0 is at the boundary of bins, not inside a bin
MIN_LOGIT = -10
MAX_LOGIT = 10

COUNT_BIN_SKIP = 3
NUM_REF_COUNT_BINS = (MAX_REF_COUNT // COUNT_BIN_SKIP) + 1 # eg if max count is 10, the 10//3 + 1 = 4 bins are {0-2}, {3-5},{6-8},{9-10}
NUM_ALT_COUNT_BINS = ((MAX_ALT_COUNT - MIN_ALT_COUNT) // COUNT_BIN_SKIP) + 1 # eg if max count is 9 and min is 1, the 8//3 + 1 = 3 bins are {1-3}, {4-6},{7-9}
LOGIT_BIN_SKIP = 1
NUM_LOGIT_BINS = floor((MAX_LOGIT - MIN_LOGIT) / LOGIT_BIN_SKIP) + 1
ALT_COUNT_BIN_BOUNDS = [(MIN_ALT_COUNT + COUNT_BIN_SKIP * count_bin) for count_bin in range(NUM_ALT_COUNT_BINS + 1)]
REF_COUNT_BIN_BOUNDS = [COUNT_BIN_SKIP * count_bin for count_bin in range(NUM_REF_COUNT_BINS + 1)]



def cap_ref_count(ref_count: int) -> int:
    return min(ref_count, MAX_REF_COUNT)


def cap_alt_count(alt_count: int) -> int:
    return min(alt_count, MAX_ALT_COUNT)


def logit_bin_indices(logits_tensor: torch.Tensor) -> IntTensor:
    return torch.div(torch.clamp(logits_tensor, min=MIN_LOGIT, max=MAX_LOGIT) - MIN_LOGIT, LOGIT_BIN_SKIP, rounding_mode='floor').long()


def top_of_logit_bin(logit_bin_index: int) -> float:
    return MIN_LOGIT + (logit_bin_index +1) * LOGIT_BIN_SKIP


def logits_from_bin_indices(logit_bin_indices: IntTensor) -> torch.Tensor:
    return (MIN_LOGIT + LOGIT_BIN_SKIP/2) + LOGIT_BIN_SKIP * logit_bin_indices


def logit_bin_name(logit_bin_idx: int) -> str:
    return f"{MIN_LOGIT + (logit_bin_idx + 0.5) * LOGIT_BIN_SKIP}:.1f"


def ref_count_bin_indices(count_tensor: IntTensor) -> IntTensor:
    return torch.div(count_tensor, COUNT_BIN_SKIP, rounding_mode='floor')


def alt_count_bin_indices(count_tensor: IntTensor) -> IntTensor:
    return torch.div(count_tensor - 1, COUNT_BIN_SKIP, rounding_mode='floor')


def count_from_ref_bin_index(count_bin_index: int) -> int:
    return COUNT_BIN_SKIP * count_bin_index + (COUNT_BIN_SKIP // 2)


def count_from_alt_bin_index(count_bin_index: int) -> int:
    return MIN_ALT_COUNT + COUNT_BIN_SKIP * count_bin_index + (COUNT_BIN_SKIP // 2)


def ref_count_bin_index(count: int) -> int:
    return count // COUNT_BIN_SKIP


def alt_count_bin_index(count: int) -> int:
    return (count - MIN_ALT_COUNT) // COUNT_BIN_SKIP


def round_ref_count_to_bin_center(count: int) -> int:
    return count_from_ref_bin_index(ref_count_bin_index(count))


def round_alt_count_to_bin_center(count: int) -> int:
    return count_from_alt_bin_index(alt_count_bin_index(count))


def counts_from_ref_bin_indices(count_bin_indices: IntTensor) -> IntTensor:
    return COUNT_BIN_SKIP * count_bin_indices + (COUNT_BIN_SKIP // 2)


def counts_from_alt_bin_indices(count_bin_indices: IntTensor) -> IntTensor:
    return MIN_ALT_COUNT + COUNT_BIN_SKIP * count_bin_indices + (COUNT_BIN_SKIP // 2)


def ref_count_bin_name(bin_idx: int) -> str:
    return str(COUNT_BIN_SKIP * bin_idx + (COUNT_BIN_SKIP-1)//2)  # the center of the bin


def alt_count_bin_name(bin_idx: int) -> str:
    return str(MIN_ALT_COUNT + COUNT_BIN_SKIP * bin_idx + (COUNT_BIN_SKIP-1)//2)  # the center of the bin