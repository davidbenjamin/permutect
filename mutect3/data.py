import torch
import random
import pickle
import numpy as np

from torch.utils.data import Dataset, DataLoader, random_split

from mutect3 import tensors

# Read sets have different sizes so we can't form a batch by naively stacking tensors.  We need a custom collate
# function for our data loader, which our model must understand.

# input batch is a list of Datum objects

class Batch:
    def __init__(self, ref, alt, ref_counts, alt_counts, info, metadata, mutect2_data, labels):
        self._ref = ref
        self._alt = alt
        self._ref_counts = ref_counts
        self._alt_counts = alt_counts
        self._info = info
        self._metadata = metadata
        self._mutect2_data = mutect2_data
        self._labels = labels
        self._size = len(labels)

    def size(self):
        return self._size

    def ref(self):
        return self._ref

    def alt(self):
        return self._alt

    def ref_counts(self):
        return self._ref_counts

    def alt_counts(self):
        return self._alt_counts

    def info(self):
        return self._info

    def metadata(self):
        return self._metadata

    def mutect_info(self):
        return self._mutect2_data

    def labels(self):
        return self._labels


# collated batch contains:
# 2D tensors of ALL ref (alt) reads, not separated by set.
# number of reads in ref (alt) read sets, in same order as read tensors
# info: 2D tensor of info fields, one row per variant
# labels: 1D tensor of 0 if non-artifact, 1 if artifact
# lists of original mutect2_data and metadata
# Example: if we have two input data, one with alt reads [[0,1,2], [3,4,5] and the other with
# alt reads [[6,7,8], [9,10,11], [12,13,14] then the output alt reads tensor is
# [[0,1,2], [3,4,5], [6,7,8], [9,10,11], [12,13,14]] and the output counts are [2,3]
# inside the model, the counts will be used to separate the reads into sets
def collate_read_sets(batch):
    ref_counts = torch.IntTensor([len(item.ref_tensor()) for item in batch])
    alt_counts = torch.IntTensor([len(item.alt_tensor()) for item in batch])
    ref = torch.cat([item.ref_tensor() for item in batch], dim=0)
    alt = torch.cat([item.alt_tensor() for item in batch], dim=0)
    info = torch.stack([item.info_tensor() for item in batch], dim=0)
    labels = torch.FloatTensor([item.artifact_label() for item in batch])
    metadata = [item.metadata() for item in batch]
    mutect2_data = [item.mutect_info() for item in batch]
    return Batch(ref, alt, ref_counts, alt_counts, info, metadata, mutect2_data, labels)

EPSILON = 0.00001
DATA_COUNT_FOR_QUANTILES = 10000

#TODO bring this into the class
def medians_and_iqrs(tensor_2d):
    # column medians etc
    medians = torch.quantile(tensor_2d, 0.5, dim=0, keepdim=False)
    vals = [0.05, 0.01, 0.0]
    iqrs = [torch.quantile(tensor_2d, 1 - x, dim=0, keepdim=False) - torch.quantile(tensor_2d, x, dim=0, keepdim=False) for x in vals]

    # for each element, try first the IQR, but if it's zero try successively larger ranges
    adjusted_iqrs = []
    for n in range(len(medians)):
        # if all zero, add 1 for no scaling
        value_to_append = 1.0
        for iqr in iqrs:
            # add the first non-zero scale
            if iqr[n] > EPSILON:
                value_to_append = iqr[n]
                break
        adjusted_iqrs.append(value_to_append)
    return medians, torch.FloatTensor(adjusted_iqrs)

class Mutect3Dataset(Dataset):
    def __init__(self, pickled_files):
        self.data = []
        for pickled_file in pickled_files:
            self.data.extend(tensors.load_pickle(pickled_file))
        random.shuffle(self.data)

        # concatenate a bunch of ref tensors and take element-by-element quantiles
        ref = torch.cat([datum.ref_tensor() for datum in self.data[:DATA_COUNT_FOR_QUANTILES]], dim=0)
        info = torch.stack([datum.info_tensor() for datum in self.data[:DATA_COUNT_FOR_QUANTILES]], dim=0)

        self.read_medians, self.read_iqrs = medians_and_iqrs(ref)
        self.info_medians, self.info_iqrs = medians_and_iqrs(info)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        raw = self.data[index]
        ref = (raw.ref_tensor() - self.read_medians) / self.read_iqrs
        alt = (raw.alt_tensor() - self.read_medians) / self.read_iqrs
        info = (raw.info_tensor() - self.info_medians) / self.info_iqrs
        return tensors.Datum(ref, alt, info, raw.metadata(), raw.mutect_info(), raw.artifact_label())

def make_datasets(training_pickles, test_pickle):
    # make our training, validation, and testing data
    train_and_valid = Mutect3Dataset(training_pickles)
    train_len = int(0.9 * len(train_and_valid))
    valid_len = len(train_and_valid) - train_len
    train, valid = random_split(train_and_valid, lengths=[train_len, valid_len])
    test = Mutect3Dataset([test_pickle])

    print("Dataset sizes -- training: " + str(len(train)) + ", validation: " + str(len(valid)) + ", test: " + str(
        len(test)))
    return train, valid, test

BATCH_SIZE = 64
def make_data_loaders(train, valid, test):
    train_labels = [datum.artifact_label() for datum in train]
    valid_labels = [datum.artifact_label() for datum in valid]
    class_counts = torch.FloatTensor(np.bincount(train_labels).tolist())
    class_weights = 1.0 / class_counts

    # epoch should roughly go over every artifact O(1) times, but more than once because we want to squeeze more out of the non-artifact
    samples_per_epoch = 20 * int(class_counts[1])

    train_sampler = torch.utils.data.WeightedRandomSampler(weights=class_weights[train_labels],
                                                           num_samples=samples_per_epoch)
    valid_sampler = torch.utils.data.WeightedRandomSampler(weights=class_weights[valid_labels],
                                                           num_samples=2 * len(valid_labels))

    train_loader = DataLoader(dataset=train, batch_size=BATCH_SIZE, sampler=train_sampler,
                              collate_fn=collate_read_sets, drop_last=True)
    valid_loader = DataLoader(dataset=valid, batch_size=BATCH_SIZE, sampler=valid_sampler,
                              collate_fn=collate_read_sets, drop_last=True)
    test_loader = DataLoader(dataset=test, batch_size=BATCH_SIZE, collate_fn=collate_read_sets, drop_last=True)
    return train_loader, valid_loader, test_loader
