import torch
from torch import IntTensor, Tensor

from permutect.utils.array_utils import cumsum_starting_from_zero, add_to_2d_array, select_and_sum


def test_cumsum_starting_from_zero():
    assert torch.sum(cumsum_starting_from_zero(IntTensor([1, 3, 2, 0, 3])) - IntTensor([0, 1, 4, 6, 6])).item() == 0


def test_add_to_array():
    x = torch.zeros(4,5).int()
    add_to_2d_array(x, torch.LongTensor([0, 2]), torch.LongTensor([1, 3]), torch.IntTensor([7, 8]))
    assert x[0, 1] == 7
    assert x[2, 3] == 8
    assert torch.sum(x) == 15


def test_select_and_sum():
    x = Tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    assert torch.equal(select_and_sum(x, select={0: 0}, sum=(2, )), Tensor([3, 7]))
    assert torch.equal(select_and_sum(x, select={0: 1, 1: 0}), Tensor([5, 6]))
    assert torch.equal(select_and_sum(x, sum=(1, 2)), Tensor([10, 26]))
