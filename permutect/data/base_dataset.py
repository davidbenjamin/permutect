import math
import os
import psutil
import random
import tarfile
import tempfile
from collections import defaultdict
from itertools import chain
from typing import Iterable, List

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler

from mmap_ninja.ragged import RaggedMmap
from permutect import utils
from permutect.data.base_datum import BaseDatum, BaseBatch, load_list_of_base_data, OneDimensionalData
from permutect.utils import Label, MutableInt, Variation

TENSORS_PER_BASE_DATUM = 2  # 1) 2D reads (ref and alt), 1) 1D concatenated stuff

# tarfiles on disk take up about 4x as much as the dataset on RAM
TARFILE_TO_RAM_RATIO = 4

ALL_COUNTS_INDEX = 0

WEIGHT_PSEUDOCOUNT = 10


def ratio_with_pseudocount(a, b):
    return (a + WEIGHT_PSEUDOCOUNT) / (b + WEIGHT_PSEUDOCOUNT)


class BaseDataset(Dataset):
    def __init__(self, data_in_ram: Iterable[BaseDatum] = None, data_tarfile=None, num_folds: int = 1):
        super(BaseDataset, self).__init__()
        assert data_in_ram is not None or data_tarfile is not None, "No data given"
        assert data_in_ram is None or data_tarfile is None, "Data given from both RAM and tarfile"
        self.num_folds = num_folds

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

        # determine the maximum count and source in order to allocate arrays
        max_count = 0
        self.max_source = 0
        datum: BaseDatum
        for datum in self:
            max_count = max(datum.alt_count, max_count)
            self.max_source = max(datum.source, self.max_source)

        # totals by source, count, label, variant type
        # we use a sentinel count value of 0 to denote aggregation over all counts
        self.totals_sclt = np.zeros((self.max_source + 1, max_count + 1, len(Label), len(Variation)))

        self.counts_by_source = defaultdict(lambda: MutableInt()) # amount of data for each source (which is an integer key)

        for n, datum in enumerate(self):
            source = datum.source
            self.counts_by_source[source].increment()

            fold = n % num_folds
            self.indices_by_fold[fold].append(n)

            variant_type_idx = datum.get_variant_type()
            self.totals_sclt[source][ALL_COUNTS_INDEX][datum.label][variant_type_idx] += 1
            self.totals_sclt[source][datum.alt_count][datum.label][variant_type_idx] += 1

        # weights that balance artifact and non-artifact loss for each source, count variation type independently
        # note: count == 0 (which never occurs as a real alt count) means aggregation over all alt counts
        self.label_balancing_weights_sclt = np.zeros((self.max_source + 1, max_count + 1, len(Label), len(Variation)))

        # balance the data between ARTIFACT, VARIANT, and UNLABELED labels within each source/count/variant type, separately
        art_to_nonart_ratios_sct = ratio_with_pseudocount(self.totals_sclt[:, :, Label.ARTIFACT, :],
                                                      self.totals_sclt[:, :, Label.VARIANT, :])
        # eg: if there are 1000 artifact and 10 non-artifact SNVs, the ratio is 100, and artifacts get a weight of 1/sqrt(100) = 1/10
        # while non-artifacts get a weight of 10 -- hence the effective count of each is 1000/10 = 10*10 = 100
        self.label_balancing_weights_sclt[:, :, Label.VARIANT, :] = np.sqrt(art_to_nonart_ratios_sct)
        self.label_balancing_weights_sclt[:, :, Label.ARTIFACT, :] = 1 / np.sqrt(art_to_nonart_ratios_sct)

        # unlabeled data are weighted down to have at most the same total weight as labeled data
        # example, 1000 unlabeled SNVs and 100 labeled SNVs -- unlabeled weight is 100/1000 = 1/10
        # example, 10 unlabeled and 100 labeled -- unlabeled weight is 1
        effective_labeled_counts_sct = self.totals_sclt[:, :, Label.ARTIFACT, :] * self.label_balancing_weights_sclt[:, :, Label.ARTIFACT, :] + \
                                   self.totals_sclt[:, :, Label.VARIANT, :] * self.label_balancing_weights_sclt[:, :, Label.VARIANT, :]
        self.label_balancing_weights_sclt[:, :, Label.UNLABELED, :] = np.clip(
            ratio_with_pseudocount(effective_labeled_counts_sct, self.totals_sclt[:, :, Label.UNLABELED, :]), 0, 1)

        # weights for adversarial source prediction task.  Balance over sources for each count and variant type
        totals_sct = np.sum(self.totals_sclt, axis=2)    # sum over label
        totals_ct = np.sum(totals_sct, axis=0)  # sum over source

        # contribution of each source to total data for each source, count, variant type
        source_ratios = ratio_with_pseudocount(totals_ct[None, :, :], totals_sct)
        self.source_balancing_weights_sct = np.sqrt(source_ratios)

        # finally, normalize source prediction weights to have same total effective count.  Note that this is modulated
        # downstream by set_alpha on the gradient reversal layer applied before source prediction
        effective_label_counts = np.sum(self.label_balancing_weights_sclt * self.totals_sclt)
        effective_source_counts = np.sum(self.source_balancing_weights_sct * totals_sct)
        self.source_balancing_weights_sct = self.source_balancing_weights_sct * (effective_label_counts / effective_source_counts)

        self.label_balancing_weights_sclt = torch.from_numpy(self.label_balancing_weights_sclt)
        self.source_balancing_weights_sct = torch.from_numpy(self.source_balancing_weights_sct)
        self.num_read_features = self[0].get_reads_2d().shape[1]
        self.num_info_features = len(self[0].get_info_tensor_1d())
        self.ref_sequence_length = len(self[0].get_ref_sequence_1d())

    def __len__(self):
        return len(self._data) // TENSORS_PER_BASE_DATUM if self._memory_map_mode else len(self._data)

    def __getitem__(self, index):
        if self._memory_map_mode:
            bottom_index = index * TENSORS_PER_BASE_DATUM
            other_stuff = OneDimensionalData.from_np_array(self._data[bottom_index + 1])

            return BaseDatum(reads_2d=self._data[bottom_index], ref_sequence_1d=None, alt_count=None, info_array_1d=None,
                             variant_type=None, label=None, source=None, variant=None, counts_and_seq_lks=None,
                             one_dimensional_data_override=other_stuff)
        else:
            return self._data[index]

    # it is often convenient to arbitrarily use the last fold for validation
    def last_fold_only(self):
        return [self.num_folds - 1]  # use the last fold for validation

    def all_but_the_last_fold(self):
        return list(range(self.num_folds - 1))

    def all_but_one_fold(self, fold_to_exclude: int):
        return list(range(fold_to_exclude)) + list(range(fold_to_exclude + 1, self.num_folds))

    def all_folds(self):
        return list(range(self.num_folds))

    def make_data_loader(self, folds_to_use: List[int], batch_size: int, pin_memory=False, num_workers: int = 0):
        sampler = SemiSupervisedBatchSampler(self, batch_size, folds_to_use)
        return DataLoader(dataset=self, batch_sampler=sampler, collate_fn=BaseBatch, pin_memory=pin_memory, num_workers=num_workers)


# from a generator that yields BaseDatum(s), create a generator that yields the two numpy arrays needed to reconstruct the datum
def make_flattened_tensor_generator(base_data_generator):
    for base_datum in base_data_generator:
        yield base_datum.get_reads_2d()
        yield base_datum.get_1d_data().to_np_array()


def make_base_data_generator_from_tarfile(data_tarfile):
    # extract the tarfile to a temporary directory that will be cleaned up when the program ends
    temp_dir = tempfile.TemporaryDirectory()
    tar = tarfile.open(data_tarfile)
    tar.extractall(temp_dir.name)
    tar.close()
    data_files = [os.path.abspath(os.path.join(temp_dir.name, p)) for p in os.listdir(temp_dir.name)]

    for file in data_files:
        for datum in load_list_of_base_data(file):
            yield datum


# ex: chunk([a,b,c,d,e], 3) = [[a,b,c], [d,e]]
def chunk(lis, chunk_size):
    return [lis[i:i + chunk_size] for i in range(0, len(lis), chunk_size)]


# Labeled and unlabeled data are mixed.
# the artifact model handles weighting the losses to compensate for class imbalance between supervised and unsupervised
# thus the sampler is not responsible for balancing the data
class SemiSupervisedBatchSampler(Sampler):
    def __init__(self, dataset: BaseDataset, batch_size, folds_to_use: List[int], sources_to_use: List[int] = None):
        # combine the index maps of all relevant folds
        self.indices_to_use = []
        source_set = None if sources_to_use is None else set(sources_to_use)
        for fold in folds_to_use:
            indices_in_fold = dataset.indices_by_fold[fold]
            if sources_to_use is None:
                source_indices_in_fold = indices_in_fold
            else:
                source_indices_in_fold = [idx for idx in indices_in_fold if dataset[idx].source in source_set]

            self.indices_to_use.extend(source_indices_in_fold)

        self.batch_size = batch_size
        self.num_batches = math.ceil(len(self.indices_to_use) // self.batch_size)

    def __iter__(self):
        batches = []    # list of lists of indices -- each sublist is a batch
        random.shuffle(self.indices_to_use)
        batches.extend(chunk(self.indices_to_use, self.batch_size))
        random.shuffle(batches)

        return iter(batches)

    def __len__(self):
        return self.num_batches

