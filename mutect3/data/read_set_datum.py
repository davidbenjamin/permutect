import torch
from torch.distributions import Beta

from mutect3 import utils

NUM_READ_FEATURES = 11  # size of each read's feature vector from M2 annotation

NUM_GATK_INFO_FEATURES = 9  # size of each variant's info field tensor (3 components for HEC, one each for HAPDOM, HAPCOMP)
# and 5 for ref bases STR info
NUM_INFO_FEATURES = NUM_GATK_INFO_FEATURES + len(utils.VariantType)


class ReadSetDatum:
    # info tensor comes from GATK and does not include one-hot encoding of variant type
    def __init__(self, contig: str, position: int, ref: str, alt: str, ref_tensor: torch.Tensor, alt_tensor: torch.Tensor,
                 gatk_info_tensor: torch.Tensor, label: str, tumor_depth: int, tumor_alt_count: int, normal_depth: int, normal_alt_count: int):
        self._contig = contig
        self._position = position
        self._ref = ref
        self._alt = alt
        self._ref_tensor = ref_tensor
        self._alt_tensor = alt_tensor

        diff = len(self.alt()) - len(self.ref())
        self._variant_type = utils.VariantType.SNV if diff == 0 else (
            utils.VariantType.INSERTION if diff > 0 else utils.VariantType.DELETION)

        # self._info_tensor includes both the original variant info and the one-hot encoding of variant type
        variant_one_hot = torch.Tensor([1 if self._variant_type.value == n else 0 for n in range(len(utils.VariantType))])
        self._gatk_info = gatk_info_tensor
        self._info_tensor = torch.cat((gatk_info_tensor, variant_one_hot))
        self._label = label

        # the following counts pertain to the data prior to any downsampling by the GATK Mutect3DatasetEngine or by
        # the unlabelled data downsampling consistency loss function
        self._tumor_depth = tumor_depth
        self._tumor_alt_count = tumor_alt_count
        self._normal_depth = normal_depth
        self._normal_alt_count = normal_alt_count

    def contig(self) -> str:
        return self._contig

    def position(self) -> int:
        return self._position

    def ref(self) -> str:
        return self._ref

    def alt(self) -> str:
        return self._alt

    def variant_type(self) -> utils.VariantType:
        return self._variant_type

    def ref_tensor(self) -> torch.Tensor:
        return self._ref_tensor

    def alt_tensor(self) -> torch.Tensor:
        return self._alt_tensor

    def gatk_info(self):
        return self._gatk_info

    def info_tensor(self) -> torch.Tensor:
        return self._info_tensor

    def label(self) -> str:
        return self._label

    def tumor_depth(self) -> int:
        return self._tumor_depth

    def tumor_alt_count(self) -> int:
        return self._tumor_alt_count

    def normal_depth(self) -> int:
        return self._normal_depth

    def normal_alt_count(self) -> int:
        return self._normal_alt_count

    def set_label(self, label):
        self._label = label
