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
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler

from mmap_ninja.ragged import RaggedMmap
from permutect import utils
from permutect.data.base_datum import BaseDatum, BaseBatch, load_list_of_base_data, BaseDatum1DStuff
from permutect.utils import Label, MutableInt

TENSORS_PER_BASE_DATUM = 2  # 1) 2D reads (ref and alt), 1) 1D concatenated stuff

# tarfiles on disk take up about 4x as much as the dataset on RAM
TARFILE_TO_RAM_RATIO = 4

ALL_COUNTS_SENTINEL = -1

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

        # totals by count, then by label -- ARTIFACT, VARIANT, UNLABELED, then by variant type
        # variant type is done as a 1D np array parallel to the one-hot encoding of variant type
        # we use a sentinel count value of -1 to denote aggregation over all counts
        # eg totals[4][Label.ARTIFACT] = [2,4,6,8,10] means there are 2 artifact SNVs with alt count 4
        self.totals = defaultdict(lambda: {label: np.zeros(len(utils.Variation)) for label in Label})

        # totals by count, then by source (integer) then by variant type
        # basically same as above but with source instead of label.  Since we don't know a priori how
        # many sources there are, we use a default dict
        # outer default dict is count, inner is source
        self.source_totals = defaultdict(lambda: defaultdict(lambda: np.zeros(len(utils.Variation))))

        self.counts_by_source = defaultdict(lambda: MutableInt()) # amount of data for each source (which is an integer key)

        for n, datum in enumerate(self):
            self.counts_by_source[datum.source].increment()

            fold = n % num_folds
            counts = (len(datum.reads_2d) - datum.alt_count, datum.alt_count)
            self.indices_by_fold[fold].append(n)

            one_hot = datum.variant_type_one_hot()
            self.totals[ALL_COUNTS_SENTINEL][datum.label] += one_hot
            self.totals[datum.alt_count][datum.label] += one_hot
            self.source_totals[ALL_COUNTS_SENTINEL][datum.source] += one_hot
            self.source_totals[datum.alt_count][datum.source] += one_hot

        # compute weights to balance loss even for unbalanced data
        # recall count == -1 is a sentinel value for aggregated totals over all alt counts
        self.weights = defaultdict(lambda: {label: np.zeros(len(utils.Variation)) for label in Label})

        # similar but indexed by count, then source, then variant type
        self.source_weights = defaultdict(lambda: defaultdict(lambda: np.zeros(len(utils.Variation))))

        sources = self.source_totals[-1].keys()
        for count in self.totals.keys():
            # eg: if there are 1000 artifact and 10 non-artifact SNVs, the ratio is 100, and artifacts get a weight of 1/sqrt(100) = 1/10
            # while non-artifacts get a weight of 10 -- hence the effective count of each is 1000/10 = 10*10 = 100
            art_to_nonart_ratios = ratio_with_pseudocount(self.totals[count][Label.ARTIFACT], self.totals[count][Label.VARIANT])
            self.weights[count][Label.VARIANT] = np.sqrt(art_to_nonart_ratios)
            self.weights[count][Label.ARTIFACT] = 1 / np.sqrt(art_to_nonart_ratios)

            effective_labeled_counts = self.totals[count][Label.ARTIFACT] * self.weights[count][Label.ARTIFACT] + \
                                       self.totals[count][Label.VARIANT] * self.weights[count][Label.VARIANT]

            # unlabeled data are weighted down to have at most the same total weight as labeled data
            # example, 1000 unlabeled SNVs and 100 labeled SNVs -- unlabeled weight is 100/1000 = 1/10
            # example, 10 unlabeled and 100 labeled -- unlabeled weight is 1
            self.weights[count][Label.UNLABELED] = np.clip(ratio_with_pseudocount(effective_labeled_counts, self.totals[count][Label.UNLABELED]), 0,1)

            # by variant type, for this count
            totals_over_sources = np.sum([self.source_totals[count][source] for source in sources])
            for source in sources:
                np.sum([self.source_weights[count][source] for source in sources])
                self.source_weights[count][source] = np.sqrt(ratio_with_pseudocount(totals_over_sources, self.source_weights[count][source]))

            # normalize source prediction weights to have same total effective count.  Note that this is modulated
            # downstream by set_alpha on the gradient reversal layer applied before source prediction
            effective_source_counts = np.sum([self.source_totals[count][source] * self.source_weights[count][source] for source in sources])
            source_weight_normalization = effective_labeled_counts / effective_source_counts
            for source in sources:
                self.source_weights[count][source] = self.source_weights[count][source] * source_weight_normalization

        self.num_read_features = self[0].get_reads_2d().shape[1]
        self.num_info_features = len(self[0].get_info_tensor_1d())
        self.ref_sequence_length = len(self[0].get_ref_sequence_1d())

    def __len__(self):
        return len(self._data) // TENSORS_PER_BASE_DATUM if self._memory_map_mode else len(self._data)

    def __getitem__(self, index):
        if self._memory_map_mode:
            bottom_index = index * TENSORS_PER_BASE_DATUM
            other_stuff = BaseDatum1DStuff.from_np_array(self._data[bottom_index+1])

            return BaseDatum(reads_2d=self._data[bottom_index], ref_sequence_1d=None, alt_count=None, info_array_1d=None, label=None,
                              source=None, variant=None, counts_and_seq_lks=None, other_stuff_override=other_stuff)
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
        yield base_datum.get_other_stuff_1d().to_np_array()


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
    def __init__(self, dataset: BaseDataset, batch_size, folds_to_use: List[int]):
        # combine the index maps of all relevant folds
        self.indices_to_use = []

        for fold in folds_to_use:
            self.indices_to_use.extend(dataset.indices_by_fold[fold])

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

