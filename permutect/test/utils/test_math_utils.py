import torch

from permutect.utils.math_utils import find_factors, add_in_log_space, subtract_in_log_space


def test_find_factors():
    assert find_factors(1) == [1]
    assert find_factors(2) == [1, 2]
    assert find_factors(4) == [1, 2, 4]
    assert find_factors(7) == [1, 7]
    assert find_factors(24) == [1, 2, 3, 4, 6, 8, 12, 24]


def test_add_in_log_space():
    x = torch.rand((4, 5))
    y = torch.rand((4, 5))
    logx, logy = torch.log(x), torch.log(y)
    exact = x+y
    calc = torch.exp(add_in_log_space(logx, logy))
    assert torch.sum(torch.abs(exact - calc)).item() < 10**(-6)


def test_subtract_in_log_space():
    y = torch.rand((4, 5))
    x = y + torch.rand((4, 5)) # this guarantees y < x, elementwise and both are positive
    logx, logy = torch.log(x), torch.log(y)
    exact = x-y
    calc = torch.exp(subtract_in_log_space(logx, logy))
    assert torch.sum(torch.abs(exact - calc)).item() < 10**(-5)
    p = 9
