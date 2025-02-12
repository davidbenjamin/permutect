from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from permutect.data.datum import Datum, MAX_FLOAT_16, MIN_FLOAT_16


class FeaturesDatum(Datum):
    """
    """
    def __init__(self, datum_array: np.ndarray, representation: Tensor):
        super().__init__(datum_array)
        # Note: if changing any of the data fields below, make sure to modify the size_in_bytes() method below accordingly!
        assert representation.dim() == 1
        self.representation = torch.clamp(representation, MIN_FLOAT_16, MAX_FLOAT_16)
        self.set_features_dtype(torch.float16)

    def set_features_dtype(self, dtype):
        self.representation = self.representation.to(dtype=dtype)

    def size_in_bytes(self):
        return self.get_nbytes() + self.representation.nbytes