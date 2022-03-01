import random
from typing import Iterable

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler

from mutect3.data.read_set_datum import ReadSetDatum
from mutect3.data.read_set_batch import ReadSetBatch

MIN_REF = 5
EPSILON = 0.00001
DATA_COUNT_FOR_QUANTILES = 10000


class ReadSetDataset(Dataset):
    def __init__(self, files=[], data: Iterable[ReadSetDatum] = []):
        self.data = []
        for table_file in files:
            self.data.extend(read_data(table_file))
        self.data.extend(data)
        random.shuffle(self.data)

        # concatenate a bunch of ref tensors and take element-by-element quantiles
        ref = torch.cat([datum.ref_tensor() for datum in self.data[:DATA_COUNT_FOR_QUANTILES]], dim=0)
        info = torch.stack([datum.info_tensor() for datum in self.data[:DATA_COUNT_FOR_QUANTILES]], dim=0)

        self.read_medians, self.read_iqrs = medians_and_iqrs(ref)
        self.info_medians, self.info_iqrs = medians_and_iqrs(info)

    def __len__(self):
        return len(self.data)

    # TODO: try normalization in constructor instead of on the fly which has to be repeated every epoch of training
    def __getitem__(self, index):
        raw = self.data[index]
        normalized_ref = (raw.ref_tensor() - self.read_medians) / self.read_iqrs
        normalized_alt = (raw.alt_tensor() - self.read_medians) / self.read_iqrs
        normalized_info = (raw.info_tensor() - self.info_medians) / self.info_iqrs

        return ReadSetDatum(raw.contig(), raw.position(), raw.ref(), raw.alt(), normalized_ref, normalized_alt, normalized_info,
                            raw.label(), raw.tumor_depth(), raw.tumor_alt_count(), raw.normal_depth(), raw.normal_alt_count())

    # this is used for training and validation but not deployment / testing
    def make_semisupervised_data_loader(self, batch_size):
        sampler = SemiSupervisedBatchSampler(self, batch_size)
        return DataLoader(dataset=self, batch_sampler=sampler, collate_fn=ReadSetBatch)

    def make_test_data_loader(self, batch_size):
        return DataLoader(dataset=self, batch_size=batch_size, collate_fn=ReadSetBatch)


def read_data(dataset_file):
    data = []
    with open(dataset_file) as file:
        while True:
            # get label
            first_line = file.readline()
            if not first_line:
                break
            label = first_line.strip()

            # contig:position,ref->alt
            locus, mutation = file.readline().strip().split(",")
            contig, position = locus.split(":")
            position = int(position)
            ref, alt = mutation.split("->")

            ref_bases = file.readline().strip()  # not currently used

            info_tensor = line_to_tensor(file.readline())

            # tumor ref count, tumor alt count, normal ref count, normal alt count -- single-spaced
            tumor_ref_count, tumor_alt_count, normal_ref_count, normal_alt_count = map(int, file.readline().strip().split())

            ref_tensor = read_2d_tensor(file, tumor_ref_count)
            alt_tensor = read_2d_tensor(file, tumor_alt_count)
            # normal_ref_tensor = read_2d_tensor(file, normal_ref_count)  # not currently used
            # normal_alt_tensor = read_2d_tensor(file, normal_alt_count)  # not currently used

            # pre-downsampling (pd) counts
            pd_tumor_depth, pd_tumor_alt, pd_normal_depth, pd_normal_alt = read_integers(file.readline())

            datum = ReadSetDatum(contig, position, ref, alt, ref_tensor, alt_tensor, info_tensor, label, pd_tumor_depth, pd_tumor_alt, pd_normal_depth, pd_normal_alt)

            if tumor_ref_count >= MIN_REF and tumor_alt_count > 0:
                data.append(datum)

    return data


def medians_and_iqrs(tensor_2d: torch.Tensor):
    # column medians etc
    medians = torch.quantile(tensor_2d, 0.5, dim=0, keepdim=False)
    vals = [0.05, 0.01, 0.0]
    iqrs = [torch.quantile(tensor_2d, 1 - x, dim=0, keepdim=False) - torch.quantile(tensor_2d, x, dim=0, keepdim=False)
            for x in vals]

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


def line_to_tensor(line: str) -> torch.Tensor:
    tokens = line.strip().split()
    floats = [float(token) for token in tokens]
    return torch.FloatTensor(floats)


def read_2d_tensor(file, num_lines: int) -> torch.Tensor:
    if num_lines == 0:
        return None
    lines = [file.readline() for _ in range(num_lines)]
    tensors_1d = [line_to_tensor(line) for line in lines]
    return torch.vstack(tensors_1d)


def read_integers(line: str):
    return map(int, line.strip().split())


def chunk(indices, chunk_size):
    return torch.split(torch.tensor(indices), chunk_size)


# make batches that are all supervised or all unsupervised
# the model handles balancing the losses between supervised and unsupervised in training, so we don't need to worry
# it's convenient to have equal numbers of labeled and unlabeled batches, so we adjust the unlabeled batch size
class SemiSupervisedBatchSampler(Sampler):
    def __init__(self, dataset: ReadSetDataset, batch_size):
        self.artifact_indices = [n for n in range(len(dataset)) if dataset[n].label() == "ARTIFACT"]
        self.non_artifact_indices = [n for n in range(len(dataset)) if dataset[n].label() == "VARIANT"]
        self.unlabeled_indices = [n for n in range(len(dataset)) if dataset[n].label() == "UNLABELED"]
        self.batch_size = batch_size

    # randomly sample non-artifact indices to get a balanced training set
    def __iter__(self):
        random.shuffle(self.artifact_indices)
        random.shuffle(self.non_artifact_indices)
        random.shuffle(self.unlabeled_indices)
        artifact_count = min(len(self.artifact_indices), len(self.non_artifact_indices))

        # balanced dataset in each epoch -- labeled vs unlabeled and artifact vs non-artifact
        labeled_indices = self.artifact_indices[:artifact_count] + self.non_artifact_indices[:artifact_count]
        random.shuffle(labeled_indices)

        unlabeled_batch_size = round((len(labeled_indices) / len(self.unlabeled_indices)) * self.batch_size)

        labeled_batches = chunk(labeled_indices, unlabeled_batch_size)
        unlabeled_batches = chunk(self.unlabeled_indices, self.batch_size)
        combined = [batch.tolist() for batch in list(labeled_batches + unlabeled_batches)]
        random.shuffle(combined)
        return iter(combined)

    def __len__(self):
        return len(self.artifact_indices) * 2 // self.batch_size + len(self.artifact_indices) // self.batch_size
