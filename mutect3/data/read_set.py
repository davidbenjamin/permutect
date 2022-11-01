import torch
from typing import List

from mutect3 import utils


class ReadSet:
    # info tensor comes from GATK and does not include one-hot encoding of variant type
    def __init__(self, variant_type: utils.Variation, ref_tensor: torch.Tensor, alt_tensor: torch.Tensor,
                 info_tensor: torch.Tensor, label: utils.Label):
        self._ref_tensor = ref_tensor
        self._alt_tensor = alt_tensor
        self._variant_type = variant_type
        self._info_tensor = info_tensor
        self._label = label

    @classmethod
    def from_gatk(cls, variant_type: utils.Variation, ref_tensor: torch.Tensor, alt_tensor: torch.Tensor,
                 gatk_info_tensor: torch.Tensor, label: utils.Label):
        info_tensor = torch.cat((gatk_info_tensor, variant_type.one_hot_tensor()))
        return cls(variant_type, ref_tensor, alt_tensor, info_tensor, label)

    def variant_type(self) -> utils.Variation:
        return self._variant_type

    def ref_tensor(self) -> torch.Tensor:
        return self._ref_tensor

    def alt_tensor(self) -> torch.Tensor:
        return self._alt_tensor

    def info_tensor(self) -> torch.Tensor:
        return self._info_tensor

    def label(self) -> utils.Label:
        return self._label


def save_list_of_read_sets(read_sets: List[ReadSet], file):
    ref_tensors = [datum.ref_tensor() for datum in read_sets]
    alt_tensors = [datum.alt_tensor() for datum in read_sets]
    info_tensors = [datum.info_tensor() for datum in read_sets]
    labels = torch.IntTensor([datum.label().value for datum in read_sets])
    variant_types = torch.IntTensor([datum.variant_type().value for datum in read_sets])

    torch.save([ref_tensors, alt_tensors, info_tensors, labels, variant_types], file)


def load_list_of_read_sets(file) -> List[ReadSet]:
    ref_tensors, alt_tensors, info_tensors, labels, variant_types = torch.load(file)
    return [ReadSet(utils.Variation(var_type), ref, alt, info, utils.Label(label)) for ref, alt, info, label, var_type in
            zip(ref_tensors, alt_tensors, info_tensors, labels.tolist(), variant_types.tolist())]





