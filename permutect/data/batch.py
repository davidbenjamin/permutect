from typing import List

import torch
from torch import IntTensor, FloatTensor
import numpy as np

from permutect.data.datum import Datum
from permutect.utils.enums import Label


class Batch:
    def __init__(self, data: List[Datum]):
        self.data = torch.from_numpy(np.vstack([d.get_array_1d() for d in data])).to(dtype=torch.long)
        self._size = len(data)

        self.info_start = Datum.REF_SEQ_START_IDX + data[0].array[Datum.REF_SEQ_LENGTH_IDX]
        info_length = data[0].array[Datum.INFO_LENGTH_IDX]
        self.info_end = self.info_start + info_length

    # get the original IntEnum format (VARIANT = 0, ARTIFACT = 1, UNLABELED = 2) labels
    def get_labels(self) -> IntTensor:
        return self.data[:, Datum.LABEL_IDX]

    # convert to the training format of 0.0 / 0.5 / 1.0 for variant / unlabeled / artifact
    # the 0.5 for unlabeled data is reasonable but should never actually be used due to the is_labeled mask
    def get_training_labels(self) -> FloatTensor:
        int_enum_labels = self.get_labels()
        return 1.0 * (int_enum_labels == Label.ARTIFACT) + 0.5 * (int_enum_labels == Label.UNLABELED)

    def get_is_labeled_mask(self) -> IntTensor:
        int_enum_labels = self.get_labels()
        return (int_enum_labels != Label.UNLABELED).int()

    def get_sources(self) -> IntTensor:
        return self.data[:, Datum.SOURCE_IDX]

    def get_variant_types(self) -> IntTensor:
        result = self.data[:, Datum.VARIANT_TYPE_IDX]
        return result

    def get_ref_counts(self) -> IntTensor:
        return self.data[:, Datum.REF_COUNT_IDX]

    def get_alt_counts(self) -> IntTensor:
        return self.data[:, Datum.ALT_COUNT_IDX]

    def get_original_alt_counts(self) -> IntTensor:
        return self.data[:, Datum.ORIGINAL_ALT_COUNT_IDX]

    def get_original_depths(self) -> torch.IntTensor:
        return self.data[:, Datum.ORIGINAL_DEPTH_IDX]

    def get_original_normal_alt_counts(self) -> torch.Tensor:
        return self.data[:, Datum.ORIGINAL_NORMAL_ALT_COUNT_IDX]

    def get_original_normal_depths(self) -> torch.Tensor:
        return self.data[:, Datum.ORIGINAL_NORMAL_DEPTH_IDX]

    def get_info_2d(self) -> torch.Tensor:
        return self.data[:, self.info_start:self.info_end] / Datum.FLOAT_TO_LONG_MULTIPLIER

    # pin memory for all tensors that are sent to the GPU
    def pin_memory(self):
        self.data = self.data.pin_memory()
        return self

    def get_data_2d(self) -> np.ndarray:
        return self.data.numpy()

    def size(self) -> int:
        return self._size