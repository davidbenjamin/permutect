from __future__ import annotations

import copy
from typing import List

import numpy as np
import torch
from torch import IntTensor, FloatTensor, Tensor

from permutect.data.datum import Datum
from permutect.data.features_datum import FeaturesDatum
from permutect.utils.enums import Label


class FeaturesBatch:
    def __init__(self, data: List[FeaturesDatum]):
        self.representations_2d = torch.vstack([item.representation for item in data])
        self.parent_data = torch.from_numpy(np.vstack([d.get_array_1d() for d in data])).to(dtype=torch.long)
        self._size = len(data)

    # get the original IntEnum format (VARIANT = 0, ARTIFACT = 1, UNLABELED = 2) labels
    def get_labels(self) -> IntTensor:
        return self.parent_data[:, Datum.LABEL_IDX]

    # convert to the training format of 0.0 / 0.5 / 1.0 for variant / unlabeled / artifact
    # the 0.5 for unlabeled data is reasonable but should never actually be used due to the is_labeled mask
    def get_training_labels(self) -> FloatTensor:
        int_enum_labels = self.get_labels()
        return 1.0 * (int_enum_labels == Label.ARTIFACT) + 0.5 * (int_enum_labels == Label.UNLABELED)

    def get_is_labeled_mask(self) -> IntTensor:
        int_enum_labels = self.get_labels()
        return (int_enum_labels != Label.UNLABELED).int()

    def get_sources(self) -> IntTensor:
        return self.parent_data[:, Datum.SOURCE_IDX]

    def get_variant_types(self) -> IntTensor:
        result = self.parent_data[:, Datum.VARIANT_TYPE_IDX]
        return result

    def get_ref_counts(self) -> IntTensor:
        return self.parent_data[:, Datum.REF_COUNT_IDX]

    def get_alt_counts(self) -> IntTensor:
        return self.parent_data[:, Datum.ALT_COUNT_IDX]

    # pin memory for all tensors that are sent to the GPU
    def pin_memory(self):
        self.representations_2d = self.representations_2d.pin_memory()
        self.parent_data = self.parent_data.pin_memory()
        return self

    def copy_to(self, device, dtype):
        is_cuda = device.type == 'cuda'
        # For all non-tensor attributes, shallow copy is sufficient
        # note that variants_array and counts_and_seq_lks_array are not used in training and are never sent to GPU
        new_batch = copy.copy(self)
        new_batch.representations_2d = self.representations_2d.to(device=device, dtype=dtype, non_blocking=is_cuda)
        new_batch.parent_data = self.parent_data.to(device, non_blocking=is_cuda)   # don't cast dtype -- needs to stay integral!

        return new_batch

    def get_parent_data_2d(self) -> np.ndarray:
        return self.parent_data.numpy()

    def get_representations_2d(self) -> Tensor:
        return self.representations_2d

    def size(self) -> int:
        return self._size