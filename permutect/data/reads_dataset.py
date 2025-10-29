import math
import os

import numpy as np
import psutil
import random
import tempfile
from typing import Iterable, List, Generator

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler

from permutect.data.datum import DATUM_ARRAY_DTYPE
from permutect.data.reads_datum import ReadsDatum, READS_ARRAY_DTYPE
from permutect.data.reads_batch import ReadsBatch
from permutect.data.batch import BatchProperty, BatchIndexedTensor
from permutect.utils.enums import Variation, Label

# dataset in memory takes a bit more space than on disk
RAM_TO_TARFILE_SPACE_RATIO = 1.25


WEIGHT_PSEUDOCOUNT = 10


def ratio_with_pseudocount(a, b):
    return (a + WEIGHT_PSEUDOCOUNT) / (b + WEIGHT_PSEUDOCOUNT)


class ReadsDataset(Dataset):
    # TODO: fix all uses of this that used to require a tarfile of saved chunks of data
    def __init__(self, data_in_ram: Iterable[ReadsDatum] = None, data_and_reads_mmaps = None, num_folds: int = 1):
        """

        :param data_in_ram a List or other Iterable of ReadsDatum -- only used for tests
        :param data_and_reads_mmaps: a tuple of (memory-mapped file of 1D data arrays, memory-mapped file of all reads)
        :param num_folds:
        """
        super(ReadsDataset, self).__init__()
        assert data_in_ram is not None or data_and_reads_mmaps is not None, "No data given"
        assert data_in_ram is None or data_and_reads_mmaps is None, "Data given from both RAM and memory maps"
        self.num_folds = num_folds
        self.totals_slvra = BatchIndexedTensor.make_zeros(num_sources=1, include_logits=False, device=torch.device('cpu'))

        # data in ram is really just a convenient way to write tests.  In actual deployment the data comes from a tarfile
        if data_in_ram is not None:
            self._stacked_reads_re = np.vstack([datum.compressed_reads_re for datum in data_in_ram])
            self._stacked_data_ve = np.vstack([datum.array for datum in data_in_ram])
            read_lengths = np.array([datum.get_ref_count() + datum.get_alt_count() for datum in data_in_ram], dtype=np.int32)
            self._read_end_indices = np.cumsum(read_lengths)
            self._size = len(data_in_ram)
            self._memory_map_mode = False
        else:
            tarfile_size = os.path.getsize(tarfile)    # in bytes
            total_num_data, total_num_reads, read_tensor_width, data_tensor_width = ReadsDatum.extract_counts_from_tarfile(tarfile)
            self._read_end_indices = np.zeros(total_num_data, dtype=np.uint64)  # nth element is the index where the nth datum's reads end (exclusive)

            self._size = total_num_data
            estimated_data_size_in_ram = tarfile_size * RAM_TO_TARFILE_SPACE_RATIO
            available_memory = psutil.virtual_memory().available
            fits_in_ram = estimated_data_size_in_ram < 0.6 * available_memory

            print(f"The tarfile size is {tarfile_size} bytes on disk for an estimated {estimated_data_size_in_ram} bytes in memory and the system has {available_memory} bytes of RAM available.")

            if fits_in_ram:
                # allocate the arrays of all data, then fill it from the tarfile
                self._stacked_reads_re = np.zeros((total_num_reads, read_tensor_width), dtype=READS_ARRAY_DTYPE)
                self._stacked_data_ve = np.zeros((total_num_data, data_tensor_width), dtype=DATUM_ARRAY_DTYPE)
                self._memory_map_mode = False
            else:
                stacked_reads_file = tempfile.NamedTemporaryFile()
                stacked_data_file = tempfile.NamedTemporaryFile()
                self._stacked_reads_re = np.memmap(stacked_reads_file.name, dtype=READS_ARRAY_DTYPE, mode='w+', shape=(total_num_reads, read_tensor_width))
                self._stacked_data_ve = np.memmap(stacked_data_file.name, dtype=DATUM_ARRAY_DTYPE, mode='w+', shape=(total_num_data, data_tensor_width))
                self._memory_map_mode = True

            print("loading the dataset from the tarfile...")

            # the following code should work for data in RAM or memmap
            read_start_idx, datum_start_idx = 0, 0
            for reads_re, data_ve, read_counts_v in ReadsDatum.generate_arrays_from_tarfile(tarfile):
                read_end_idx, datum_end_idx = read_start_idx + len(reads_re), datum_start_idx + len(data_ve)
                self._read_end_indices[datum_start_idx:datum_end_idx] = read_start_idx + np.cumsum(read_counts_v)
                self._stacked_reads_re[read_start_idx:read_end_idx] = reads_re
                self._stacked_data_ve[datum_start_idx:datum_end_idx] = data_ve
                read_start_idx, datum_start_idx = read_end_idx, datum_end_idx

            # set memory maps to read-only
            if self._memory_map_mode:
                self._stacked_reads_re.flags.writeable = False
                self._stacked_data_ve.flags.writeable = False

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

    # it is often convenient to arbitrarily use the last fold for validation
    def last_fold_only(self):
        return [self.num_folds - 1]  # use the last fold for validation

    def all_but_the_last_fold(self):
        return list(range(self.num_folds - 1))

    def all_but_one_fold(self, fold_to_exclude: int):
        return list(range(fold_to_exclude)) + list(range(fold_to_exclude + 1, self.num_folds))

    def all_folds(self):
        return list(range(self.num_folds))

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

    def make_data_loader(self, folds_to_use: List[int], batch_size: int, pin_memory=False, num_workers: int = 0,
                         sources_to_use: List[int] = None, labeled_only: bool = False):
        sampler = SemiSupervisedBatchSampler(self, batch_size, folds_to_use, sources_to_use, labeled_only)
        return DataLoader(dataset=self, batch_sampler=sampler, collate_fn=ReadsBatch, pin_memory=pin_memory, num_workers=num_workers)

    def make_train_and_valid_loaders(self, validation_fold: int, batch_size: int, is_cuda: bool, num_workers: int, sources_to_use: List[int] = None):
        train_loader = self.make_data_loader(self.all_but_one_fold(validation_fold), batch_size, is_cuda, num_workers, sources_to_use)
        valid_loader = self.make_data_loader([validation_fold], batch_size, is_cuda, num_workers, sources_to_use)
        return train_loader, valid_loader


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

