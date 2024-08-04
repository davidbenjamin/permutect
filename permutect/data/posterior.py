import random
import math
from typing import List, Iterable

import torch
from torch import IntTensor, vstack, from_numpy
from torch.utils.data import Dataset, DataLoader
from permutect.data.base_datum import Variant, CountsAndSeqLks

from permutect import utils
from permutect.utils import Label, Variation


class PosteriorDatum:
    CONTIG = 0
    POSITION = 1
    REF = 2
    ALT = 3
    VAR_TYPE = 4
    DEPTH = 5
    ALT_COUNT = 6
    NORMAL_DEPTH = 7
    NORMAL_ALT_COUNT = 8
    LABEL = 9

    SEQ_ERROR_LOG_LK = 0
    TLOD_FROM_M2 = 1
    NORMAL_SEQ_ERROR_LOG_LK = 2
    ALLELE_FREQUENCY = 3
    ARTIFACT_LOGIT = 4
    MAF = 5
    NORMAL_MAF = 6

    def __init__(self, variant: Variant, counts_and_seq_lks: CountsAndSeqLks, allele_frequency: float,
                 artifact_logit: float, embedding: torch.Tensor, label: Label, maf: float, normal_maf: float):
        self.embedding = embedding

        this_class = self.__class__
        self.int_array = torch.zeros(10, dtype=int)
        self.int_array[this_class.CONTIG] = variant.contig
        self.int_array[this_class.POSITION] = variant.position
        self.int_array[this_class.REF] = variant.get_ref_as_int()    # ref and alt are the base-5 encoding as integers
        self.int_array[this_class.ALT] = variant.get_alt_as_int()
        self.int_array[this_class.VAR_TYPE] = utils.Variation.get_type(self.ref, self.alt)  # Variation is IntEnum so this is int
        self.int_array[this_class.DEPTH] = counts_and_seq_lks.depth
        self.int_array[this_class.ALT_COUNT] = counts_and_seq_lks.alt_count
        self.int_array[this_class.NORMAL_DEPTH] = counts_and_seq_lks.normal_depth
        self.int_array[this_class.NORMAL_ALT_COUNT] = counts_and_seq_lks.normal_alt_count
        self.int_array[this_class.LABEL] = label

        self.float_array = torch.zeros(7)
        self.float_array[this_class.SEQ_ERROR_LOG_LK] = counts_and_seq_lks.seq_error_log_lk
        self.float_array[this_class.TLOD_FROM_M2] = -counts_and_seq_lks.seq_error_log_lk - math.log(self.depth + 1)
        self.float_array[this_class.NORMAL_SEQ_ERROR_LOG_LK] = counts_and_seq_lks.normal_seq_error_log_lk
        self.float_array[this_class.ALLELE_FREQUENCY] = allele_frequency
        self.float_array[this_class.ARTIFACT_LOGIT] = artifact_logit
        self.float_array[this_class.MAF] = maf
        self.float_array[this_class.NORMAL_MAF] = normal_maf


class PosteriorBatch:

    def __init__(self, data: List[PosteriorDatum]):
        self._original_list = data  # keep this for downsampling augmentation
        self.embeddings = torch.vstack([item.embedding for item in data])
        self.int_tensor = torch.vstack([item.int_array for item in data])
        self.float_tensor = torch.vstack([item.float_array for item in data])
        self._variant_type_one_hot = vstack([from_numpy(item.variant_type.one_hot_tensor()) for item in self._original_list])

        self._size = len(data)

    def get_alt_counts(self) -> torch.Tensor:
        return self.int_tensor[:, PosteriorDatum.ALT_COUNT]

    def get_depths(self) -> torch.Tensor:
        return self.int_tensor[:, PosteriorDatum.DEPTH]

    def get_normal_alt_counts(self) -> torch.Tensor:
        return self.int_tensor[:, PosteriorDatum.NORMAL_ALT_COUNT]

    def get_normal_depths(self) -> torch.Tensor:
        return self.int_tensor[:, PosteriorDatum.NORMAL_DEPTH]

    def get_tlods_from_m2(self) -> torch.Tensor:
        return self.float_tensor[:, PosteriorDatum.TLOD_FROM_M2]

    def get_allele_frequencies(self) -> torch.Tensor:
        return self.float_tensor[:, PosteriorDatum.ALLELE_FREQUENCY]

    def get_artifact_logits(self) -> torch.Tensor:
        return self.float_tensor[:, PosteriorDatum.ARTIFACT_LOGIT]

    def get_mafs(self) -> torch.Tensor:
        return self.float_tensor[:, PosteriorDatum.MAF]

    def get_normal_mafs(self) -> torch.Tensor:
        return self.float_tensor[:, PosteriorDatum.NORMAL_MAF]

    def original_list(self) -> List[PosteriorDatum]:
        return self._original_list

    def size(self) -> int:
        return self._size

    def get_normal_ref_counts(self) -> IntTensor:
        return self.get_normal_depths() - self.normal_alt_counts()

    def variant_type_one_hot(self):
        return self._variant_type_one_hot

    # return list of variant type integer indices
    def variant_types(self):
        one_hot = self.variant_type_one_hot()
        return [int(x) for x in sum([n * one_hot[:, n] for n in range(len(Variation))])]


class PosteriorDataset(Dataset):
    def __init__(self, data: Iterable[PosteriorDatum], shuffle: bool = True):
        self.data = data

        if shuffle:
            random.shuffle(self.data)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> PosteriorDatum:
        return self.data[index]

    def make_data_loader(self, batch_size: int):
        return DataLoader(dataset=self, batch_size=batch_size, collate_fn=PosteriorBatch)