import pickle
import random
from typing import Set, List

import torch
from torch.distributions.beta import Beta
from tqdm.autonotebook import tqdm

from mutect3 import utils

class Datum:
    #TODO: need ref, alt locus in constructor
    def __init__(self, ref_tensor: torch.Tensor, alt_tensor: torch.Tensor, info_tensor: torch.Tensor,
                 site_info: SiteInfo, label, normal_depth: int, normal_alt_count: int):
        self._site_info = site_info
        self._ref_tensor = ref_tensor
        self._alt_tensor = alt_tensor
        self._info_tensor = info_tensor
        self._label = label
        self._normal_depth = normal_depth
        self._normal_alt_count = normal_alt_count

    def locus(self):
        return self._locus

    def ref(self):
        return self._ref

    def alt(self):
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

    def label(self):
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
        ref = downsample(self._ref_tensor, ref_length)
        alt = downsample(self._alt_tensor, alt_length)
        return Datum(ref, alt, self.info_tensor(), self.label(),
                     self.normal_depth(), self.normal_alt_count())


# pickle and unpickle a Python list of Datum objects.  Convenient to have here because unpickling needs to have all
# the constituent classes of Datum explicitly imported.
def make_pickle(file, datum_list):
    with open(file, 'wb') as f:
        pickle.dump(datum_list, f)


def load_pickle(file):
    with open(file, 'rb') as f:
        return pickle.load(f)


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


class NormalArtifactTableReader:
    def __init__(self, header_tokens):
        self.normal_alt_idx = header_tokens.index("normal_alt")
        self.normal_dp_idx = header_tokens.index("normal_dp")
        self.tumor_alt_idx = header_tokens.index("tumor_alt")
        self.tumor_dp_idx = header_tokens.index("tumor_dp")
        self.downsampling_idx = header_tokens.index("downsampling")
        self.type_idx = header_tokens.index("type")

    def normal_alt_count(self, tokens):
        return int(tokens[self.normal_alt_idx])

    def normal_depth(self, tokens):
        return int(tokens[self.normal_dp_idx])

    def tumor_alt_count(self, tokens):
        return int(tokens[self.tumor_alt_idx])

    def tumor_depth(self, tokens):
        return int(tokens[self.tumor_dp_idx])

    def downsampling(self, tokens):
        return float(tokens[self.downsampling_idx])

    def variant_type(self, tokens):
        return tokens[self.type_idx]


def read_normal_artifact_data(table_file, shuffle=True) -> List[NormalArtifactDatum]:
    data = []

    with open(table_file) as fp:
        reader = NormalArtifactTableReader(fp.readline().split())

        pbar = tqdm(enumerate(fp))
        for n, line in pbar:
            tokens = line.split()

            normal_alt_count = reader.normal_alt_count(tokens)
            normal_depth = reader.normal_depth(tokens)
            tumor_alt_count = reader.tumor_alt_count(tokens)
            tumor_depth = reader.tumor_depth(tokens)
            downsampling = reader.downsampling(tokens)
            variant_type = reader.variant_type(tokens)

            data.append(NormalArtifactDatum(normal_alt_count, normal_depth, tumor_alt_count, tumor_depth, downsampling,
                                            variant_type))

    if shuffle:
        random.shuffle(data)
    print("Done")
    return data


def generate_normal_artifact_pickle(table_file, pickle_file):
    data = read_normal_artifact_data(table_file)
    make_pickle(pickle_file, data)


def downsample(tensor: torch.Tensor, downsample_fraction) -> torch.Tensor:
    if downsample_fraction is None or downsample_fraction >= len(tensor):
        return tensor
    else:
        return tensor[torch.randperm(len(tensor))[:downsample_fraction]]


NUM_READ_FEATURES = 11  # size of each read's feature vector from M2 annotation
NUM_INFO_FEATURES = 9  # size of each variant's info field tensor (3 components for HEC, one each for HAPDOM, HAPCOMP)
# and 5 for ref bases STR info