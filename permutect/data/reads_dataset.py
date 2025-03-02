import math
import os
import psutil
import random
import tarfile
import tempfile
from typing import Iterable, List

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler

from mmap_ninja.ragged import RaggedMmap
from permutect.data.count_binning import cap_ref_count, cap_alt_count
from permutect.data.reads_datum import ReadsDatum
from permutect.data.reads_batch import ReadsBatch
from permutect.data.batch import BatchProperty, BatchIndexedTensor
from permutect.utils.enums import Variation, Label

TENSORS_PER_BASE_DATUM = 2  # 1) 2D reads (ref and alt), 1) 1D concatenated stuff

# tarfiles on disk take up about 4x as much as the dataset on RAM
TARFILE_TO_RAM_RATIO = 4


WEIGHT_PSEUDOCOUNT = 10


def ratio_with_pseudocount(a, b):
    return (a + WEIGHT_PSEUDOCOUNT) / (b + WEIGHT_PSEUDOCOUNT)


class ReadsDataset(Dataset):
    def __init__(self, data_in_ram: Iterable[ReadsDatum] = None, data_tarfile=None, num_folds: int = 1):
        super(ReadsDataset, self).__init__()
        assert data_in_ram is not None or data_tarfile is not None, "No data given"
        assert data_in_ram is None or data_tarfile is None, "Data given from both RAM and tarfile"
        self.num_folds = num_folds
        self.totals_slvra = BatchIndexedTensor(num_sources=1, device=torch.device('cpu'), include_logits=False)   # on CPU

        if data_in_ram is not None:
            self._data = data_in_ram
            self._memory_map_mode = False
        else:
            tarfile_size = os.path.getsize(data_tarfile)    # in bytes
            estimated_data_size_in_ram = tarfile_size // TARFILE_TO_RAM_RATIO
            available_memory = psutil.virtual_memory().available
            fits_in_ram = estimated_data_size_in_ram < 0.8 * available_memory

            print(f"The tarfile size is {tarfile_size} bytes on disk for an estimated {estimated_data_size_in_ram} bytes in memory and the system has {available_memory} bytes of RAM available.")
            if fits_in_ram:
                print("loading the dataset from the tarfile into RAM:")
                self._data = list(make_base_data_generator_from_tarfile(data_tarfile))
                self._memory_map_mode = False
            else:
                print("loading the dataset into a memory-mapped file:")
                self._memory_map_dir = tempfile.TemporaryDirectory()

                RaggedMmap.from_generator(out_dir=self._memory_map_dir.name,
                                          sample_generator=make_flattened_tensor_generator(
                                              make_base_data_generator_from_tarfile(data_tarfile)),
                                          batch_size=10000, verbose=False)
                self._data = RaggedMmap(self._memory_map_dir.name)
                self._memory_map_mode = True

        # this is used in the batch sampler to make same-shape batches
        self.indices_by_fold = [[] for _ in range(num_folds)]

        for n, datum in enumerate(self):
            self.totals_slvra.record_datum(datum)
            fold = n % num_folds
            self.indices_by_fold[fold].append(n)

        self.num_read_features = self[0].get_reads_re().shape[1]
        self.num_info_features = len(self[0].get_info_1d())
        self.haplotypes_length = len(self[0].get_haplotypes_1d())

    def __len__(self):
        return len(self._data) // TENSORS_PER_BASE_DATUM if self._memory_map_mode else len(self._data)

    def __getitem__(self, index):
        if self._memory_map_mode:
            bottom_index = index * TENSORS_PER_BASE_DATUM
            return ReadsDatum(datum_array=self._data[bottom_index + 1], reads_re=self._data[bottom_index])
        else:
            return self._data[index]

    def num_sources(self) -> int:
        return self.totals_slvra.num_sources

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
def make_flattened_tensor_generator(reads_data_generator):
    for reads_datum in reads_data_generator:
        yield reads_datum.get_reads_re()
        yield reads_datum.get_array_1d()


def make_base_data_generator_from_tarfile(data_tarfile):
    # extract the tarfile to a temporary directory that will be cleaned up when the program ends
    temp_dir = tempfile.TemporaryDirectory()
    tar = tarfile.open(data_tarfile)
    tar.extractall(temp_dir.name)
    tar.close()
    data_files = [os.path.abspath(os.path.join(temp_dir.name, p)) for p in os.listdir(temp_dir.name)]

    for file in data_files:
        for datum in ReadsDatum.load_list(file):
            ref_count = cap_ref_count(datum.get_ref_count())
            alt_count = cap_alt_count(datum.get_alt_count())
            yield datum.copy_with_downsampled_reads(ref_count, alt_count)


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

