from __future__ import annotations

import copy
from typing import List

import torch
from torch import Tensor

from permutect.data.batch import Batch
from permutect.data.features_datum import FeaturesDatum


class FeaturesBatch(Batch):
    def __init__(self, data: List[FeaturesDatum]):
        super().__init__(data)
        self.representations_2d = torch.vstack([item.representation for item in data])

    def pin_memory(self):
        super().pin_memory()
        self.representations_2d = self.representations_2d.pin_memory()
        return self

    def copy_to(self, device, dtype):
        is_cuda = device.type == 'cuda'
        new_batch = copy.copy(self)
        new_batch.representations_2d = self.representations_2d.to(device=device, dtype=dtype, non_blocking=is_cuda)
        new_batch.data = self.data.to(device, non_blocking=is_cuda)   # don't cast dtype -- needs to stay integral!
        return new_batch

    def get_representations_2d(self) -> Tensor:
        return self.representations_2d