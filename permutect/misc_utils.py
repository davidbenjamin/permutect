import psutil
import tarfile
import os
import torch
from torch import Tensor


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
    def __init__(self, value:int = 0):
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
        d = 'cuda'
    elif exploit_mps and torch.mps.is_available():
        d = 'mps'
    else:
        d = 'cpu'
    return torch.device(d)


def freeze(parameters):
    for parameter in parameters:
        parameter.requires_grad = False


def unfreeze(parameters):
    for parameter in parameters:
        if parameter.dtype.is_floating_point:   # an integer parameter isn't trainable by gradient descent
            parameter.requires_grad = True


class StreamingAverage:
    def __init__(self):
        self._count = 0.0
        self._sum = 0.0

    def is_empty(self) -> bool:
        return self._count == 0.0

    def get(self) -> float:
        return self._sum / (self._count + 0.0001)

    def record(self, value: float, weight: float=1):
        self._count += weight
        self._sum += value * weight

    def record_sum(self, value_sum: float, count):
        self._count += count
        self._sum += value_sum

    # record only values masked as true
    def record_with_mask(self, values: Tensor, mask: Tensor):
        self._count += torch.sum(mask).item()
        self._sum += torch.sum(values*mask).item()

    # record values with different weights
    # values and mask should live on same device as self._sum
    def record_with_weights(self, values: Tensor, weights: Tensor):
        self._count += torch.sum(weights).item()
        self._sum += torch.sum(values * weights).item()


def backpropagate(optimizer: torch.optim.Optimizer, loss: Tensor):
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


def extract_to_temp_dir(tar_file, directory):
    tar = tarfile.open(tar_file)
    tar.extractall(directory)
    tar.close()
    return [os.path.abspath(os.path.join(directory, p)) for p in os.listdir(directory)]
