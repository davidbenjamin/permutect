import torch
from torch import IntTensor

from permutect.utils.array_utils import cumsum_starting_from_zero


def test_cumsum_starting_from_zero():
    assert torch.sum(cumsum_starting_from_zero(IntTensor([1, 3, 2, 0, 3])) - IntTensor([0, 1, 4, 6, 6])).item() == 0