import math
import os
import random
import tarfile
import tempfile
from collections import defaultdict
from itertools import chain
from typing import Iterable

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler

from mutect3 import utils
from mutect3.data.read_set import ReadSet
from mutect3.data.read_set_batch import ReadSetBatch
from mutect3.utils import Label


class ReadSetDataset(Dataset):
    def __init__(self, data: Iterable[ReadSet], shuffle: bool = True):
        self.data = data

        if shuffle:
            random.shuffle(self.data)

        # keys = (ref read count, alt read count) tuples; values = list of indices
        # this is used in the batch sampler to make same-shape batches
        self.labeled_indices_by_count = defaultdict(list)
        self.unlabeled_indices_by_count = defaultdict(list)

        for n, datum in enumerate(self.data):
            counts = (len(datum.ref_tensor), len(datum.alt_tensor))
            (self.unlabeled_indices_by_count if datum.label() == Label.UNLABELED else self.labeled_indices_by_count)[counts].append(n)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def num_read_features(self) -> int:
        return self.data[0].alt_tensor().size()[1]  # number of columns in (arbitrarily) the first alt read tensor of the dataset

    def num_info_features(self) -> int:
        return len(self.data[0].info_tensor()) # number of columns in (arbitrarily) the first alt read tensor of the dataset

    def ref_sequence_length(self) -> int:
        return self.data[0].ref_sequence_tensor.shape[-1]


# this is used for training and validation but not deployment / testing
def make_semisupervised_data_loader(dataset: ReadSetDataset, batch_size: int, pin_memory=False, num_workers: int=0):
    sampler = SemiSupervisedBatchSampler(dataset, batch_size)
    return DataLoader(dataset=dataset, batch_sampler=sampler, collate_fn=ReadSetBatch, pin_memory=pin_memory, num_workers=num_workers)


def make_test_data_loader(dataset: ReadSetDataset, batch_size: int):
    return DataLoader(dataset=dataset, batch_size=batch_size, collate_fn=ReadSetBatch)


# TODO: this might belong somewhere else
def count_data(dataset_file):
    n = 0
    with open(dataset_file) as file:
        for line in file:
            if Label.is_label(line.strip()):
                n += 1
    return n


class BigReadSetDataset:


            # this parallels the order given in the save method below
            self.num_read_features, self.num_info_features, self.ref_sequence_length, self.num_training_data,\
                self.training_artifact_totals, self.training_non_artifact_totals = torch.load(metadata_file)

            # these will be cleaned up when the program ends, or maybe when the object goes out of scope
            self.train_temp_dir = tempfile.TemporaryDirectory()
            tar = tarfile.open(train_tar)
            tar.extractall(self.train_temp_dir.name)
            tar.close()
            self.train_data_files = [os.path.abspath(os.path.join(self.train_temp_dir.name, p)) for p in os.listdir(self.train_temp_dir.name)]

            train, valid = utils.split_dataset_into_train_and_valid(dataset, 0.9)


# ex: chunk([a,b,c,d,e], 3) = [[a,b,c], [d,e]]
def chunk(lis, chunk_size):
    return [lis[i:i + chunk_size] for i in range(0, len(lis), chunk_size)]


# make batches that are all supervised or all unsupervised
# the artifact model handles weights the losses to compensate for class imbalance between supervised and unsupervised
# thus the sampler is not responsible for balancing the data
class SemiSupervisedBatchSampler(Sampler):
    def __init__(self, dataset: ReadSetDataset, batch_size):
        self.labeled_indices_by_count = dataset.labeled_indices_by_count
        self.unlabeled_indices_by_count = dataset.unlabeled_indices_by_count
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

