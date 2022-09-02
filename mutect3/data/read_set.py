import torch
from torch.distributions import Beta

from mutect3 import utils

NUM_GATK_INFO_FEATURES = 9  # size of each variant's info field tensor (3 components for HEC, one each for HAPDOM, HAPCOMP)
# and 5 for ref bases STR info
NUM_INFO_FEATURES = NUM_GATK_INFO_FEATURES + len(utils.Variation)


class ReadSet:
    # info tensor comes from GATK and does not include one-hot encoding of variant type
    def __init__(self, variant_type: utils.Variation, ref_tensor: torch.Tensor, alt_tensor: torch.Tensor,
                 gatk_info_tensor: torch.Tensor, label: utils.Label):
        self._ref_tensor = ref_tensor
        self._alt_tensor = alt_tensor
        self._variant_type = variant_type

        # self._info_tensor includes both the original variant info and the one-hot encoding of variant type
        self._gatk_info = gatk_info_tensor
        self._info_tensor = torch.cat((gatk_info_tensor, self._variant_type.one_hot_tensor()))
        self._label = label

    def variant_type(self) -> utils.Variation:
        return self._variant_type

    def ref_tensor(self) -> torch.Tensor:
        return self._ref_tensor

    def alt_tensor(self) -> torch.Tensor:
        return self._alt_tensor

    def gatk_info(self):
        return self._gatk_info

    def info_tensor(self) -> torch.Tensor:
        return self._info_tensor

    def label(self) -> utils.Label:
        return self._label


