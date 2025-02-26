from __future__ import annotations

import copy
import random
import math
from typing import List, Iterable

import numpy as np
import torch
from torch import IntTensor, Tensor
from torch.utils.data import Dataset, DataLoader

from permutect.data.batch import Batch
from permutect.data.datum import Datum


class PosteriorDatum(Datum):

    TLOD_FROM_M2 = 0
    ALLELE_FREQUENCY = 1
    ARTIFACT_LOGIT = 2
    MAF = 3
    NORMAL_MAF = 4

    def __init__(self, datum_array, allele_frequency: float, artifact_logit: float, maf: float, normal_maf: float, embedding: Tensor):
        super().__init__(datum_array)
        self.embedding = embedding

        self.float_array = torch.zeros(5, dtype=torch.float16)
        self.float_array[PosteriorDatum.TLOD_FROM_M2] = -self.get_seq_error_log_lk() - math.log(self.get_original_depth() + 1)
        self.float_array[PosteriorDatum.ALLELE_FREQUENCY] = allele_frequency
        self.float_array[PosteriorDatum.ARTIFACT_LOGIT] = artifact_logit
        self.float_array[PosteriorDatum.MAF] = maf
        self.float_array[PosteriorDatum.NORMAL_MAF] = normal_maf

    def get_artifact_logit(self) -> float:
        return self.float_array[self.__class__.ARTIFACT_LOGIT]


class PosteriorBatch(Batch):

    def __init__(self, data: List[PosteriorDatum]):
        super().__init__(data)
        self.embeddings = torch.vstack([item.embedding for item in data]).float()
        self.float_tensor = torch.vstack([item.float_array for item in data]).float()

    def pin_memory(self):
        super().pin_memory()
        self.embeddings = self.embeddings.pin_memory()
        self.float_tensor = self.float_tensor.pin_memory()
        return self

    # dtype is just for floats!!! Better not convert the int tensor to a float accidentally!
    def copy_to(self, device, dtype):
        is_cuda = device.type == 'cuda'
        new_batch = copy.copy(self)
        new_batch.data = self.data.to(device, non_blocking=is_cuda)  # don't cast dtype -- needs to stay integral!
        new_batch.embeddings = self.embeddings.to(device=device, dtype=dtype, non_blocking=is_cuda)
        new_batch.float_tensor = self.float_tensor.to(device=device, dtype=dtype, non_blocking=is_cuda)
        return new_batch

    def get_tlods_from_m2(self) -> Tensor:
        return self.float_tensor[:, PosteriorDatum.TLOD_FROM_M2]

    def get_allele_frequencies(self) -> Tensor:
        return self.float_tensor[:, PosteriorDatum.ALLELE_FREQUENCY]

    def get_artifact_logits(self) -> Tensor:
        return self.float_tensor[:, PosteriorDatum.ARTIFACT_LOGIT]

    def get_mafs(self) -> Tensor:
        return self.float_tensor[:, PosteriorDatum.MAF]

    def get_normal_mafs(self) -> Tensor:
        return self.float_tensor[:, PosteriorDatum.NORMAL_MAF]

    def get_original_normal_ref_counts(self) -> IntTensor:
        return self.get_original_normal_depths() - self.get_original_normal_alt_counts()


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
