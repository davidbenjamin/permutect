from __future__ import annotations

import copy
import random
import math
from typing import List, Iterable

import torch
from torch import IntTensor
from torch.utils.data import Dataset, DataLoader
from permutect.data.base_datum import Variant, bases5_as_base_string, ParentDatum


def variant_from_int_array(subarray) -> Variant:
    contig = subarray[0].item()
    position = subarray[1].item()
    ref = bases5_as_base_string(subarray[2].item())  # ref and alt are the base-5 encoding as integers
    alt = bases5_as_base_string(subarray[3].item())
    return Variant(contig, position, ref, alt)


class PosteriorDatum(ParentDatum):

    TLOD_FROM_M2 = 0
    ALLELE_FREQUENCY = 1
    ARTIFACT_LOGIT = 2
    MAF = 3
    NORMAL_MAF = 4

    def __init__(self, parent_datum_array, allele_frequency: float, artifact_logit: float, maf: float, normal_maf: float,
                 embedding: torch.Tensor):
        super().__init__(parent_datum_array)
        self.embedding = embedding

        self.float_array = torch.zeros(5, dtype=torch.float16)
        self.float_array[PosteriorDatum.TLOD_FROM_M2] = -self.get_seq_error_log_lk() - math.log(self.get_original_depth() + 1)
        self.float_array[PosteriorDatum.ALLELE_FREQUENCY] = allele_frequency
        self.float_array[PosteriorDatum.ARTIFACT_LOGIT] = artifact_logit
        self.float_array[PosteriorDatum.MAF] = maf
        self.float_array[PosteriorDatum.NORMAL_MAF] = normal_maf

    def get_artifact_logit(self) -> float:
        return self.float_array[self.__class__.ARTIFACT_LOGIT]


class PosteriorBatch:

    def __init__(self, data: List[PosteriorDatum]):
        self.embeddings = torch.vstack([item.embedding for item in data]).float()
        self.parent_data_2d = torch.vstack([item.get_array_1d() for item in data])
        self.float_tensor = torch.vstack([item.float_array for item in data]).float()

        self._size = len(data)

    def pin_memory(self):
        self.embeddings = self.embeddings.pin_memory()
        self.parent_data_2d = self.parent_data_2d.pin_memory()
        self.float_tensor = self.float_tensor.pin_memory()
        return self

    # dtype is just for floats!!! Better not convert the int tensor to a float accidentally!
    def copy_to(self, device, dtype):
        is_cuda = device.type == 'cuda'
        # For all non-tensor attributes, shallow copy is sufficient
        new_batch = copy.copy(self)

        new_batch.embeddings = self.embeddings.to(device=device, dtype=dtype, non_blocking=is_cuda)
        new_batch.parent_data_2d = self.parent_data_2d.to(device=device, non_blocking=is_cuda)
        new_batch.float_tensor = self.float_tensor.to(device=device, dtype=dtype, non_blocking=is_cuda)

        return new_batch

    def get_variant_types(self) -> torch.IntTensor:
        return self.parent_data_2d[:, ParentDatum.VARIANT_TYPE_IDX]

    def get_labels(self) -> torch.IntTensor:
        return self.parent_data_2d[:, ParentDatum.LABEL_IDX]

    def get_alt_counts(self) -> torch.IntTensor:
        return self.parent_data_2d[:, ParentDatum.ORIGINAL_ALT_COUNT_IDX]

    def get_depths(self) -> torch.IntTensor:
        return self.parent_data_2d[:, ParentDatum.ORIGINAL_DEPTH_IDX]

    def get_normal_alt_counts(self) -> torch.Tensor:
        return self.parent_data_2d[:, ParentDatum.ORIGINAL_NORMAL_ALT_COUNT_IDX]

    def get_normal_depths(self) -> torch.Tensor:
        return self.parent_data_2d[:, ParentDatum.ORIGINAL_NORMAL_DEPTH_IDX]

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

    def size(self) -> int:
        return self._size

    def get_normal_ref_counts(self) -> IntTensor:
        return self.get_normal_depths() - self.get_normal_alt_counts()


class PosteriorDataset(Dataset):
    def __init__(self, data: Iterable[PosteriorDatum], shuffle: bool = True):
        self.data = data

        if shuffle:
            random.shuffle(self.data)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> PosteriorDatum:
        return self.data[index]

    def make_data_loader(self, batch_size: int, pin_memory: bool = False, num_workers: int = 0):
        return DataLoader(dataset=self, batch_size=batch_size, pin_memory=pin_memory, num_workers=num_workers, collate_fn=PosteriorBatch)
