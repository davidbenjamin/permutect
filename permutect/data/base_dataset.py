import math
import os
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
from permutect.data.base_datum import BaseDatum, BaseBatch, load_list_of_base_data, CountsAndSeqLks, Variant, \
    BaseDatum1DStuff
from permutect.utils import Label

TENSORS_PER_BASE_DATUM = 2  # 1) 2D reads (ref and alt), 1) 1D concatenated stuff


class BaseDataset(Dataset):
    def __init__(self, data_in_ram: Iterable[BaseDatum] = None, data_tarfile=None, num_folds: int = 1):
        super(BaseDataset, self).__init__()
        assert data_in_ram is not None or data_tarfile is not None, "No data given"
        assert data_in_ram is None or data_tarfile is None, "Data given from both RAM and tarfile"
        self._memory_map_mode = data_tarfile is not None
        self.num_folds = num_folds

        if self._memory_map_mode:
            self._memory_map_dir = tempfile.TemporaryDirectory()

            RaggedMmap.from_generator(out_dir=self._memory_map_dir.name,
                sample_generator=make_flattened_tensor_generator(make_base_data_generator_from_tarfile(data_tarfile)),
                batch_size=10000, verbose=False)
            self._data = RaggedMmap(self._memory_map_dir.name)
        else:
            self._data = data_in_ram

        # keys = (ref read count, alt read count) tuples; values = list of indices
        # this is used in the batch sampler to make same-shape batches
        self.labeled_indices_by_count = [defaultdict(list) for _ in range(num_folds)]
        self.unlabeled_indices_by_count = [defaultdict(list) for _ in range(num_folds)]
        self.artifact_totals = np.zeros(len(utils.Variation))  # 1D tensor
        self.non_artifact_totals = np.zeros(len(utils.Variation))  # 1D tensor

        for n, datum in enumerate(self):
            fold = n % num_folds
            counts = (len(datum.reads_2d) - datum.alt_count, datum.alt_count)
            (self.unlabeled_indices_by_count if datum.label == Label.UNLABELED else self.labeled_indices_by_count)[fold][counts].append(n)

            if datum.label == Label.ARTIFACT:
                self.artifact_totals += datum.variant_type_one_hot()
            elif datum.label != Label.UNLABELED:
                self.non_artifact_totals += datum.variant_type_one_hot()

        self.num_read_features = self[0].reads_2d.shape[1]
        self.num_info_features = len(self[0].info_array_1d)
        self.ref_sequence_length = len(self[0].ref_sequence_1d)

    def __len__(self):
        return len(self._data) // TENSORS_PER_BASE_DATUM if self._memory_map_mode else len(self.data)

    def __getitem__(self, index):
        if self._memory_map_mode:
            bottom_index = index * TENSORS_PER_BASE_DATUM
            other_stuff = BaseDatum1DStuff.from_np_array(reads_2d=self._data[bottom_index]+1)

            return BaseDatum(reads_2d=self._data[bottom_index], ref_sequence_1d=None, alt_count=None, info_array_1d=None, label=None,
                              variant=None, counts_and_seq_lks=None, other_stuff_override=other_stuff)
        else:
            return self._data[index]

    def artifact_to_non_artifact_ratios(self):
        return self.artifact_totals / self.non_artifact_totals

    def total_labeled_and_unlabeled(self):
        total_labeled = np.sum(self.artifact_totals + self.non_artifact_totals)
        return total_labeled, len(self) - total_labeled

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


# make batches that are all supervised or all unsupervised, and have a single value for ref, alt counts within  batches
# the artifact model handles weighting the losses to compensate for class imbalance between supervised and unsupervised
# thus the sampler is not responsible for balancing the data
class SemiSupervisedBatchSampler(Sampler):
    def __init__(self, dataset: BaseDataset, batch_size, folds_to_use: List[int]):
        # combine the index maps of all relevant folds
        self.labeled_indices_by_count = defaultdict(list)
        self.unlabeled_indices_by_count = defaultdict(list)
        for fold in folds_to_use:
            new_labeled = dataset.labeled_indices_by_count[fold]
            new_unlabeled = dataset.unlabeled_indices_by_count[fold]
            for count, indices in new_labeled.items():
                self.labeled_indices_by_count[count].extend(indices)
            for count, indices in new_unlabeled.items():
                self.unlabeled_indices_by_count[count].extend(indices)

        self.batch_size = batch_size
        self.num_batches = sum(math.ceil(len(indices) // self.batch_size) for indices in
                            chain(self.labeled_indices_by_count.values(), self.unlabeled_indices_by_count.values()))

    def __iter__(self):
        batches = []    # list of lists of indices -- each sublist is a batch
        for index_list in chain(self.labeled_indices_by_count.values(), self.unlabeled_indices_by_count.values()):
            random.shuffle(index_list)
            batches.extend(chunk(index_list, self.batch_size))
        random.shuffle(batches)

        return iter(batches)

    def __len__(self):
        return self.num_batches

