import math
import os
import random
import tarfile
import tempfile
from collections import defaultdict
from itertools import chain
from typing import Iterable

import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler

from mmap_ninja.ragged import RaggedMmap
from mutect3 import utils
from mutect3.data.read_set import ReadSet, ReadSetBatch, load_list_of_read_sets
from mutect3.utils import Label

TENSORS_PER_READ_SET = 5


class ReadSetDataset(Dataset):
    def __init__(self, data_in_ram: Iterable[ReadSet] = None, data_tarfile=None, validation_fraction: float = 0.0):
        super(ReadSetDataset, self).__init__()
        assert data_in_ram is not None or data_tarfile is not None, "No data given"
        assert data_in_ram is None or data_tarfile is None, "Data given from both RAM and tarfile"
        self._memory_map_mode = data_tarfile is not None

        if self._memory_map_mode:
            self._memory_map_dir = tempfile.TemporaryDirectory()

            RaggedMmap.from_generator(out_dir=self._memory_map_dir.name,
                                      sample_generator=make_flattened_tensor_generator(
                                          make_read_set_generator_from_tarfile(data_tarfile)),
                                      batch_size=10000,
                                      verbose=True)

            self._data = RaggedMmap(self._memory_map_dir.name)
        else:
            self._data = data_in_ram

        # keys = (ref read count, alt read count) tuples; values = list of indices
        # this is used in the batch sampler to make same-shape batches
        self.labeled_indices_by_count = {utils.Epoch.TRAIN: defaultdict(list), utils.Epoch.VALID: defaultdict(list)}
        self.unlabeled_indices_by_count = {utils.Epoch.TRAIN: defaultdict(list), utils.Epoch.VALID: defaultdict(list)}
        self.artifact_totals = np.zeros(len(utils.Variation))  # 1D tensor
        self.non_artifact_totals = np.zeros(len(utils.Variation))  # 1D tensor

        for n, datum in enumerate(self):
            train_or_valid = utils.Epoch.VALID if random.random() < validation_fraction else utils.Epoch.TRAIN
            counts = (len(datum.ref_reads_2d) if datum.ref_reads_2d is not None else 0, len(datum.alt_reads_2d))
            (self.unlabeled_indices_by_count if datum.label == Label.UNLABELED else self.labeled_indices_by_count)[train_or_valid][counts].append(n)

            if datum.label == Label.ARTIFACT:
                self.artifact_totals += datum.variant_type_one_hot()
            elif datum.label != Label.UNLABELED:
                self.non_artifact_totals += datum.variant_type_one_hot()

        self.num_read_features = self[0].alt_reads_2d.shape[1]
        self.num_info_features = len(self[0].info_array_1d)
        self.ref_sequence_length = self[0].ref_sequence_2d.shape[-1]

    def __len__(self):
        return len(self._data) // TENSORS_PER_READ_SET if self._memory_map_mode else len(self.data)

    def __getitem__(self, index):
        if self._memory_map_mode:
            bottom_index = index * TENSORS_PER_READ_SET

            possible_ref = self._data[bottom_index]

            # The order here corresponds to the order of yield statements within make_flattened_tensor_generator()
            return ReadSet(ref_sequence_2d=self._data[bottom_index + 2],
                           ref_reads_2d=possible_ref if len(possible_ref) > 0 else None,
                           alt_reads_2d=self._data[bottom_index + 1],
                           info_array_1d=self._data[bottom_index + 3],
                           label=utils.Label(self._data[bottom_index + 4][0]))
        else:
            return self._data[index]

    def artifact_to_non_artifact_ratios(self):
        return self.artifact_totals / self.non_artifact_totals

    def total_labeled_and_unlabeled(self):
        total_labeled = np.sum(self.artifact_totals + self.non_artifact_totals)
        return total_labeled, len(self) - total_labeled


# from a generator that yields read sets, create a generator that yields
# ref tensor, alt tensor, ref sequence tensor, info tensor, label tensor, ref tensor alt tensor. . .
def make_flattened_tensor_generator(read_set_generator):
    for read_set in read_set_generator:
        yield read_set.ref_reads_2d if read_set.ref_reads_2d is not None else np.empty((0, 0))
        yield read_set.alt_reads_2d
        yield read_set.ref_sequence_2d
        yield read_set.info_array_1d
        yield np.array([read_set.label.value])  # single-element tensor of the Label enum


def make_read_set_generator_from_tarfile(data_tarfile):
    # extract the tarfile to a temporary directory that will be cleaned up when the program ends
    temp_dir = tempfile.TemporaryDirectory()
    tar = tarfile.open(data_tarfile)
    tar.extractall(temp_dir.name)
    tar.close()
    data_files = [os.path.abspath(os.path.join(temp_dir.name, p)) for p in os.listdir(temp_dir.name)]

    # recall each data file saves a list of ReadSets via ReadSet.save_list_of_read_sets
    # we reverse it with ReadSet.load_list_of_read_sets
    for file in data_files:
        for datum in load_list_of_read_sets(file):
            yield datum


def make_data_loader(dataset: ReadSetDataset, train_or_valid: utils.Epoch, batch_size: int, pin_memory=False, num_workers: int = 0):
    sampler = SemiSupervisedBatchSampler(dataset, batch_size, train_or_valid)
    return DataLoader(dataset=dataset, batch_sampler=sampler, collate_fn=ReadSetBatch, pin_memory=pin_memory, num_workers=num_workers)


# ex: chunk([a,b,c,d,e], 3) = [[a,b,c], [d,e]]
def chunk(lis, chunk_size):
    return [lis[i:i + chunk_size] for i in range(0, len(lis), chunk_size)]


# make batches that are all supervised or all unsupervised, and have a single value for ref, alt counts within  batches
# the artifact model handles weighting the losses to compensate for class imbalance between supervised and unsupervised
# thus the sampler is not responsible for balancing the data
class SemiSupervisedBatchSampler(Sampler):
    def __init__(self, dataset: ReadSetDataset, batch_size, train_or_valid: utils.Epoch):
        self.labeled_indices_by_count = dataset.labeled_indices_by_count[train_or_valid]
        self.unlabeled_indices_by_count = dataset.unlabeled_indices_by_count[train_or_valid]
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

