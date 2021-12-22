import random
from typing import List

import pandas as pd
import torch
from torch.distributions.beta import Beta

from mutect3 import utils


class Datum:
    def __init__(self, contig: str, position: int, ref: str, alt: str, ref_tensor: torch.Tensor,
                 alt_tensor: torch.Tensor,
                 info_tensor: torch.Tensor, label: str, normal_depth: int, normal_alt_count: int):
        self._contig = contig
        self._position = position
        self._ref = ref
        self._alt = alt
        self._ref_tensor = ref_tensor
        self._alt_tensor = alt_tensor
        self._info_tensor = info_tensor
        self._label = label
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
        diff = len(self.alt()) - len(self.ref())
        return utils.VariantType.SNV if diff == 0 else (
            utils.VariantType.INSERTION if diff > 0 else utils.VariantType.DELETION)

    def ref_tensor(self) -> torch.Tensor:
        return self._ref_tensor

    def alt_tensor(self) -> torch.Tensor:
        return self._alt_tensor

    def info_tensor(self) -> torch.Tensor:
        return self._info_tensor

    def label(self) -> str:
        return self._label

    def normal_depth(self) -> int:
        return self._normal_depth

    def normal_alt_count(self) -> int:
        return self._normal_alt_count

    def set_label(self, label):
        self._label = label

    # beta is distribution of downsampling fractions
    def downsampled_copy(self, beta: Beta):
        ref_frac = beta.sample().item()
        alt_frac = beta.sample().item()

        ref_length = max(1, round(ref_frac * len(self._ref_tensor)))
        alt_length = max(1, round(alt_frac * len(self._alt_tensor)))
        return Datum(self._contig, self._position, self._ref, self._alt, downsample(self._ref_tensor, ref_length),
                     downsample(self._alt_tensor, alt_length), self._info_tensor, self._label, self._normal_depth,
                     self._normal_alt_count)


class NormalArtifactDatum:
    def __init__(self, normal_alt_count: int, normal_depth: int, tumor_alt_count: int, tumor_depth: int,
                 downsampling: float, variant_type: str):
        self._normal_alt_count = normal_alt_count
        self._normal_depth = normal_depth
        self._tumor_alt_count = tumor_alt_count
        self._tumor_depth = tumor_depth
        self._downsampling = downsampling
        self._variant_type = variant_type

    def normal_alt_count(self) -> int:
        return self._normal_alt_count

    def normal_depth(self) -> int:
        return self._normal_depth

    def tumor_alt_count(self) -> int:
        return self._tumor_alt_count

    def tumor_depth(self) -> int:
        return self._tumor_depth

    def downsampling(self) -> float:
        return self._downsampling

    def variant_type(self) -> str:
        return self._variant_type


def read_normal_artifact_data(table_file, shuffle=True) -> List[NormalArtifactDatum]:
    df = pd.read_table(table_file, header=0)
    df = df.astype({"normal_alt": int, "normal_dp": int, "tumor_alt": int, "tumor_dp": int, "downsampling": float,
                    "type": str})

    data = []
    for _, row in df.iterrows():
        data.append(NormalArtifactDatum(row['normal_alt'], row['normal_dp'], row['tumor_alt'], row['tumor_dp'],
                                        row['downsampling'], row['type']))

    if shuffle:
        random.shuffle(data)
    return data


def downsample(tensor: torch.Tensor, downsample_fraction) -> torch.Tensor:
    return tensor if (downsample_fraction is None or downsample_fraction >= len(tensor)) \
        else tensor[torch.randperm(len(tensor))[:downsample_fraction]]


NUM_READ_FEATURES = 11  # size of each read's feature vector from M2 annotation
NUM_INFO_FEATURES = 9  # size of each variant's info field tensor (3 components for HEC, one each for HAPDOM, HAPCOMP)
# and 5 for ref bases STR info
