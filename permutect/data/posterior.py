import random
import math
from typing import List, Iterable

from torch import Tensor, IntTensor, BoolTensor, vstack, from_numpy
from torch.utils.data import Dataset, DataLoader
from permutect.data.read_set import Variant, CountsAndSeqLks

from permutect import utils
from permutect.utils import Label


class PosteriorDatum:
    def __init__(self, variant: Variant, counts_and_seq_lks: CountsAndSeqLks,  index: int, allele_frequency: float,
                 artifact_logit: float, label: Label):

        self.contig = variant.contig
        self.position = variant.position
        self.ref = variant.ref
        self.alt = variant.alt
        self.variant_type = utils.Variation.get_type(self.ref, self.alt)

        self.depth = counts_and_seq_lks.depth
        self.alt_count = counts_and_seq_lks.alt_count
        self.normal_depth = counts_and_seq_lks.normal_depth
        self.normal_alt_count = counts_and_seq_lks.normal_alt_count
        self.index = index
        self.label = label

        self.seq_error_log_likelihood = counts_and_seq_lks.seq_error_log_lk
        self.tlod_from_m2 = -self.seq_error_log_likelihood - math.log(self.depth + 1)
        self.normal_seq_error_log_likelihood = counts_and_seq_lks.normal_seq_error_log_lk

        self.allele_frequency = allele_frequency
        self.artifact_logit = artifact_logit


class PosteriorBatch:

    def __init__(self, data: List[PosteriorDatum]):
        self._original_list = data  # keep this for downsampling augmentation
        self._size = len(data)

        self.depths = IntTensor([item.depth for item in data])
        self.alt_counts = IntTensor([item.alt_count for item in data])
        self.normal_depths = IntTensor([item.normal_depth for item in data])
        self.normal_alt_counts = IntTensor([item.normal_alt_count for item in data])

        self._variant_type_one_hot = vstack([from_numpy(item.variant_type.one_hot_tensor()) for item in self._original_list]).float()

        self.seq_error_log_likelihoods = Tensor([item.seq_error_log_likelihood for item in self._original_list])
        self.tlods_from_m2 = Tensor([item.tlod_from_m2 for item in self._original_list])
        self.normal_seq_error_log_likelihoods = Tensor([item.normal_seq_error_log_likelihood for item in self._original_list])
        self.allele_frequencies = Tensor([item.allele_frequency for item in self._original_list])
        self.artifact_logits = Tensor([item.artifact_logit for item in self._original_list])

    def original_list(self) -> List[PosteriorDatum]:
        return self._original_list

    def size(self) -> int:
        return self._size

    def ref_counts(self) -> IntTensor:
        return self.depths - self.alt_counts

    def normal_ref_counts(self) -> IntTensor:
        return self.normal_depths - self.normal_alt_counts

    def variant_type_one_hot(self):
        return self._variant_type_one_hot

    def variant_type_mask(self, variant_type):
        return BoolTensor([item.variant_type == variant_type for item in self._original_list])


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