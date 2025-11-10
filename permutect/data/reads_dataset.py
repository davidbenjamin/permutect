import math
import psutil
import random
from typing import  List, Generator

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler

from permutect.data.memory_mapped_data import MemoryMappedData
from permutect.data.reads_datum import ReadsDatum
from permutect.data.reads_batch import ReadsBatch
from permutect.data.batch import BatchProperty, BatchIndexedTensor
from permutect.utils.enums import Variation, Label

WEIGHT_PSEUDOCOUNT = 10


# it is often convenient to arbitrarily use the last fold for validation
def last_fold_only(num_folds: int):
    return [num_folds - 1]  # use the last fold for validation

def all_but_the_last_fold(num_folds: int):
    return list(range(num_folds - 1))

def all_but_one_fold(num_folds: int, fold_to_exclude: int):
    return list(range(fold_to_exclude)) + list(range(fold_to_exclude + 1, num_folds))

def all_folds(num_folds: int):
    return list(range(num_folds))

def ratio_with_pseudocount(a, b):
    return (a + WEIGHT_PSEUDOCOUNT) / (b + WEIGHT_PSEUDOCOUNT)


class ReadsDataset(Dataset):
    def __init__(self, memory_mapped_data: MemoryMappedData, num_folds: int = 1, folds_to_use: List[int] = None):
        """
        :param num_folds:
        """
        super(ReadsDataset, self).__init__()
        self.num_folds = num_folds
        self.folds_to_use = folds_to_use
        self.totals_slvra = BatchIndexedTensor.make_zeros(num_sources=1, include_logits=False, device=torch.device('cpu'))
        self.memory_mapped_data = memory_mapped_data
        self._size = memory_mapped_data.num_data
        self._read_end_indices = memory_mapped_data.read_end_indices

        available_memory = psutil.virtual_memory().available
        fits_in_ram = memory_mapped_data.size_in_bytes() < 0.6 * available_memory
        self._memory_map_mode = fits_in_ram
        print(f"Data occupy {memory_mapped_data.size_in_bytes() // 1000000} Mb and the system has {available_memory // 1000000} Mb of RAM available.")

        # copy memory-mapped data to RAM if space allows, otherwise use the memory-mapped data
        self._stacked_reads_re = memory_mapped_data.reads_mmap.copy() if fits_in_ram else memory_mapped_data.reads_mmap
        self._stacked_data_ve = memory_mapped_data.data_mmap.copy() if fits_in_ram else memory_mapped_data.data_mmap

        # this is used in the batch sampler to make same-shape batches
        self.indices_by_fold = [[] for _ in range(num_folds)]

        for n, datum in enumerate(self):
            self.totals_slvra.record_datum(datum)
            fold = n % num_folds
            self.indices_by_fold[fold].append(n)

        first_datum: ReadsDatum = self[0]
        self.num_read_features = first_datum.num_read_features()
        self.num_info_features = len(first_datum.get_info_1d())
        self.haplotypes_length = len(first_datum.get_haplotypes_1d())

    def __len__(self):
        return self._size

    def __getitem__(self, index):
        read_start_index = 0 if index == 0 else self._read_end_indices[index - 1]
        read_end_index = self._read_end_indices[index]

        return ReadsDatum(datum_array=self._stacked_data_ve[index],
                              compressed_reads_re=self._stacked_reads_re[read_start_index:read_end_index])

    def num_sources(self) -> int:
        return self.totals_slvra.num_sources()

    def report_totals(self):
        totals_slv = self.totals_slvra.get_marginal((BatchProperty.SOURCE, BatchProperty.LABEL, BatchProperty.VARIANT_TYPE))
        for source in range(len(totals_slv)):
            print(f"Data counts for source {source}:")
            for var_type in Variation:
                print(f"Data counts for variant type {var_type.name}:")
                for label in Label:
                    print(f"{label.name}: {int(totals_slv[source, label, var_type].item())}")


    def validate_sources(self) -> int:
        num_sources = self.num_sources()
        totals_by_source_s = self.totals_slvra.get_marginal((BatchProperty.SOURCE,))
        if num_sources == 1:
            print("Data come from a single source")
        else:
            for source in range(self.num_sources()):
                assert totals_by_source_s[source].item() >= 1, f"No data for source {source}."
            print(f"Data come from multiple sources, with counts {totals_by_source_s.cpu().tolist()}.")
        return num_sources

    def make_data_loader(self, batch_size: int, pin_memory=False, num_workers: int = 0,
                         sources_to_use: List[int] = None, labeled_only: bool = False):
        sampler = SemiSupervisedBatchSampler(self, batch_size, folds_to_use, sources_to_use, labeled_only)
        return DataLoader(dataset=self, batch_sampler=sampler, collate_fn=ReadsBatch, pin_memory=pin_memory, num_workers=num_workers)


# from a generator that yields BaseDatum(s), create a generator that yields the two numpy arrays needed to reconstruct the datum
def make_flattened_tensor_generator(reads_data_generator: Generator[ReadsDatum, None, None]):
    for reads_datum in reads_data_generator:
        yield reads_datum.get_compressed_reads_re()
        yield reads_datum.get_array_1d()


# ex: chunk([a,b,c,d,e], 3) = [[a,b,c], [d,e]]
def chunk(lis, chunk_size):
    return [lis[i:i + chunk_size] for i in range(0, len(lis), chunk_size)]


# Labeled and unlabeled data are mixed.
# the artifact model handles weighting the losses to compensate for class imbalance between supervised and unsupervised
# thus the sampler is not responsible for balancing the data
class SemiSupervisedBatchSampler(Sampler):
    def __init__(self, dataset: ReadsDataset, batch_size: int, folds_to_use: List[int],
                 sources_to_use: List[int] = None, labeled_only: bool = False):
        # combine the index maps of all relevant folds
        self.indices_to_use = []
        source_set = None if sources_to_use is None else set(sources_to_use)
        for fold in folds_to_use:
            indices_in_fold = dataset.indices_by_fold[fold] if not labeled_only else \
                [idx for idx in dataset.indices_by_fold[fold] if dataset[idx].get_label() != Label.UNLABELED]
            if sources_to_use is None:
                source_indices_in_fold = indices_in_fold
            else:
                source_indices_in_fold = [idx for idx in indices_in_fold if dataset[idx].get_source() in source_set]

            self.indices_to_use.extend(source_indices_in_fold)

        self.batch_size = batch_size
        self.num_batches = math.ceil(len(self.indices_to_use) / self.batch_size)

    def __iter__(self):
        batches = []    # list of lists of indices -- each sublist is a batch
        random.shuffle(self.indices_to_use)
        batches.extend(chunk(self.indices_to_use, self.batch_size))
        random.shuffle(batches)

        return iter(batches)

    def __len__(self):
        return self.num_batches

