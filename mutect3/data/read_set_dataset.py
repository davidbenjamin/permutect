import random
import math
import time
from typing import Iterable
import psutil
import os
import pickle
import tempfile

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler

from mutect3.data.read_set import ReadSet
from mutect3.data.posterior_datum import PosteriorDatum
from mutect3.data.read_set_batch import ReadSetBatch
from mutect3.utils import Variation, Label
from mutect3 import utils

MIN_REF = 5
EPSILON = 0.00001
QUANTILE_DATA_COUNT = 10000

# TODO: in order to handle very large fragments (and chimeric reads), we may eventually prefer to log-scale fragment sizes
MAX_VALUE = 10000  # clamp inputs to this range

PICKLE_EXTENSION = ".pickle"


class ReadSetDataset(Dataset):
    def __init__(self, files=[], data: Iterable[ReadSet] = [], shuffle: bool = True, normalize: bool = True):
        self.data = []
        for table_file in files:
            self.data.extend(read_data(table_file))
        self.data.extend(data)
        if shuffle:
            random.shuffle(self.data)

        if normalize:
            # concatenate a bunch of ref tensors and take element-by-element quantiles
            # for simplicity we do sampling with replacement
            N = len(self.data)
            random_indices = range(N) if N <= QUANTILE_DATA_COUNT else torch.randint(0, N,
                                                                                     (QUANTILE_DATA_COUNT,)).tolist()

            ref = torch.cat([self.data[n].ref_tensor() for n in random_indices], dim=0)
            gatk_info = torch.stack([self.data[n].gatk_info() for n in random_indices], dim=0)

            read_medians, read_iqrs = medians_and_iqrs(ref)
            gatk_info_medians, gatk_info_iqrs = medians_and_iqrs(gatk_info)

            for n in range(len(self.data)):
                raw = self.data[n]
                normalized_ref = (raw.ref_tensor() - read_medians) / read_iqrs
                normalized_alt = (raw.alt_tensor() - read_medians) / read_iqrs
                normalized_gatk_info = (raw.gatk_info() - gatk_info_medians) / gatk_info_iqrs
                self.data[n] = ReadSet(raw.variant_type(), normalized_ref, normalized_alt, normalized_gatk_info, raw.label())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def num_read_features(self) -> int:
        return self.data[0].alt_tensor().size()[1]  # number of columns in (arbitrarily) the first alt read tensor of the dataset


# this is used for training and validation but not deployment / testing
def make_semisupervised_data_loader(dataset: ReadSetDataset, batch_size: int, pin_memory=False):
    sampler = SemiSupervisedBatchSampler(dataset, batch_size)
    return DataLoader(dataset=dataset, batch_sampler=sampler, collate_fn=ReadSetBatch, pin_memory=pin_memory)


def make_test_data_loader(dataset: ReadSetDataset, batch_size: int):
    return DataLoader(dataset=dataset, batch_size=batch_size, collate_fn=ReadSetBatch)


def count_data(dataset_file):
    n = 0
    with open(dataset_file) as file:
        for line in file:
            if Label.is_label(line.strip()):
                n += 1
    return n


# generator that reads a plain text dataset file and yields data
# in posterior model, yield a tuple of ReadSet and PosteriorDatum
def read_data(dataset_file, posterior: bool = False, yield_nones: bool = False):
    with open(dataset_file) as file:
        n = 0
        while label_str := file.readline().strip():
            label = Label.get_label(label_str)
            n += 1

            # contig:position,ref->alt
            locus, mutation = file.readline().strip().split(",")
            contig, position = locus.split(":")
            position = int(position)
            if n % 100000 == 0:
                print(contig + ":" + str(position))
            ref, alt = mutation.strip().split("->")

            # ref base string
            ref_bases = file.readline().strip()  # not currently used

            gatk_info_tensor = line_to_tensor(file.readline())

            # tumor ref count, tumor alt count, normal ref count, normal alt count -- single-spaced
            # these are the read counts of our possibly-downsampled tensors, not the original sequencing data
            ref_tensor_size, alt_tensor_size, normal_ref_tensor_size, normal_alt_tensor_size = map(int, file.readline().strip().split())

            ref_tensor = read_2d_tensor(file, ref_tensor_size)
            alt_tensor = read_2d_tensor(file, alt_tensor_size)

            # normal_ref_tensor = read_2d_tensor(file, normal_ref_tensor_size)  # not currently used
            # normal_alt_tensor = read_2d_tensor(file, normal_alt_tensor_size)  # not currently used

            depth, alt_count, normal_depth, normal_alt_count = read_integers(file.readline())

            seq_error_log_likelihood = read_float(file.readline())
            normal_seq_error_log_likelihood = read_float(file.readline())

            assert ref_tensor is None or not torch.sum(ref_tensor).isnan().item(), contig + ":" + str(position)
            assert alt_tensor is None or not torch.sum(alt_tensor).isnan().item(), contig + ":" + str(position)
            assert not torch.sum(gatk_info_tensor).isnan().item(), contig + ":" + str(position)

            datum = ReadSet(Variation.get_type(ref, alt), ref_tensor, alt_tensor, gatk_info_tensor, label)

            if ref_tensor_size >= MIN_REF and alt_tensor_size > 0:
                if posterior:
                    posterior_datum = PosteriorDatum(contig, position, ref, alt, depth,
                                alt_count, normal_depth, normal_alt_count, seq_error_log_likelihood, normal_seq_error_log_likelihood)
                    yield datum, posterior_datum
                else:
                    yield datum
            elif yield_nones:
                if posterior:
                    yield None, None
                else:
                    yield None


# TODO: there is some code duplication between this and filter_variants.py
# TODO: also, the approach here is cleaner
def generate_datasets(dataset_file, chunk_size: int):
    num_data = count_data(dataset_file)
    num_chunks = math.ceil(num_data / chunk_size)
    actual_chunk_size = num_data // num_chunks

    buffer = []
    data_count = 0
    for read_set in read_data(dataset_file, posterior=False, yield_nones=True):
        data_count += 1
        if read_set is not None:
            buffer.append(read_set)
        if data_count == actual_chunk_size:
            print("memory usage percent: " + str(psutil.virtual_memory().percent))
            yield ReadSetDataset(data=buffer, shuffle=True, normalize=True)
            buffer = []
            data_count = 0


class BigReadSetDataset:

    # TODO: we need to record number of read features in the constructor
    # TODO: probably also record the labeled to unlabeled ratio

    def __init__(self, batch_size: int = 64, chunk_size: int = 1000000, dataset: ReadSetDataset = None, dataset_files=None):
        assert dataset is None or dataset_files is None, "Initialize either with dataset or files, not both"
        assert dataset is not None or dataset_files is not None, "Must initialize with a dataset or files, not nothing"
        self.use_gpu = torch.cuda.is_available()
        self.batch_size = batch_size

        # TODO: make this class autocloseable and clean up the temp dirs when closing
        self.train_pickles = []
        self.valid_pickles = []
        self.num_read_features = None

        if dataset is not None:
            self.train_fits_in_ram = True
            self.valid_fits_in_ram = True
            train, valid = utils.split_dataset_into_train_and_valid(dataset, 0.9)

            self.train_batches = [batch for batch in make_semisupervised_data_loader(train, batch_size, pin_memory=self.use_gpu)]
            self.valid_batches = [batch for batch in make_semisupervised_data_loader(valid, batch_size, pin_memory=self.use_gpu)]
            self.num_read_features = dataset.num_read_features()

        else:
            validation_batches = []
            for dataset_file in dataset_files:
                for dataset_from_file in generate_datasets(dataset_file, chunk_size):
                    if self.num_read_features is None:
                        self.num_read_features = dataset_from_file.num_read_features()
                    else:
                        assert self.num_read_features == dataset_from_file.num_read_features(), "inconsistent number of read features between files"
                    train, valid = utils.split_dataset_into_train_and_valid(dataset_from_file, 0.9)
                    train_loader = make_semisupervised_data_loader(train, self.batch_size, pin_memory=self.use_gpu)
                    valid_loader = make_semisupervised_data_loader(valid, self.batch_size, pin_memory=self.use_gpu)

                    with tempfile.NamedTemporaryFile(delete=False) as train_pickle:
                        pickle.dump([batch for batch in train_loader], train_pickle, protocol=pickle.HIGHEST_PROTOCOL)
                        self.train_pickles.append(train_pickle.name)
                        train_pickle.flush()    # is this really necessary?

                    # extend the validation data and pickle if it has gotten big enough
                    validation_batches.extend(valid_loader)
                    if batch_size * len(validation_batches) > chunk_size:
                        with tempfile.NamedTemporaryFile(delete=False) as valid_pickle:
                            pickle.dump(validation_batches, valid_pickle, protocol=pickle.HIGHEST_PROTOCOL)
                            self.valid_pickles.append(valid_pickle.name)
                            valid_pickle.flush()  # is this really necessary?
                        validation_batches = []

            # pickle any remaining validation data
            if validation_batches:
                with tempfile.NamedTemporaryFile(delete=False) as valid_pickle:
                    pickle.dump(validation_batches, valid_pickle, protocol=pickle.HIGHEST_PROTOCOL)
                    self.valid_pickles.append(valid_pickle.name)
                    valid_pickle.flush()  # is this really necessary?

            # check if there is only one pickled dataset, in which case we can fit it in RAM
            # note that we already normalized and shuffled
            if len(self.train_pickles) == 1:
                self.train_fits_in_ram = True
                with open(self.train_pickles[0], 'rb') as train_pickle:
                    self.train_batches = pickle.load(train_pickle)
            else:
                self.train_fits_in_ram = False
                self.train_batches = None

            if len(self.valid_pickles) == 1:
                self.valid_fits_in_ram = True
                with open(self.valid_pickles[0], 'rb') as valid_pickle:
                    self.valid_batches = pickle.load(valid_pickle)
            else:
                self.valid_fits_in_ram = False
                self.valid_batches = None

        assert self.train_fits_in_ram == (self.train_batches is not None)
        assert self.valid_fits_in_ram == (self.valid_batches is not None)
        assert self.train_fits_in_ram == (len(self.train_pickles) <= 1)
        assert self.valid_fits_in_ram == (len(self.valid_pickles) <= 1)
        assert self.num_read_features is not None

    def generate_batches(self, epoch_type: utils.Epoch):
        assert epoch_type != utils.Epoch.TEST, "test epochs not supported yet"

        if epoch_type == utils.Epoch.TRAIN and self.train_fits_in_ram:
            for batch in self.train_batches:
                yield batch
        elif epoch_type == utils.Epoch.VALID and self.valid_fits_in_ram:
            for batch in self.valid_batches:
                yield batch
        else:
            pickles = self.train_pickles if epoch_type == utils.Epoch.TRAIN else self.valid_pickles
            for file in pickles:
                with open(file, 'rb') as pickle_file:
                    start = time.time()
                    batches = pickle.load(pickle_file)
                    end = time.time()
                    print("{} batches loaded from disk in {} seconds.".format(len(batches), end - start))
                    for batch in batches:
                        yield batch


def medians_and_iqrs(tensor_2d: torch.Tensor):
    # column medians etc
    medians = torch.quantile(tensor_2d.float(), 0.5, dim=0, keepdim=False)
    vals = [0.05, 0.01, 0.0]
    iqrs = [torch.quantile(tensor_2d.float(), 1 - x, dim=0, keepdim=False) - torch.quantile(tensor_2d.float(), x, dim=0, keepdim=False)
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
    return torch.clamp(torch.FloatTensor(floats), -MAX_VALUE, MAX_VALUE).half()


def read_2d_tensor(file, num_lines: int) -> torch.Tensor:
    if num_lines == 0:
        return None
    lines = [file.readline() for _ in range(num_lines)]
    tensors_1d = [line_to_tensor(line) for line in lines]
    return torch.vstack(tensors_1d)


def read_integers(line: str):
    return map(int, line.strip().split())


def read_float(line: str):
    return float(line.strip().split()[0])


def chunk(indices, chunk_size):
    return torch.split(torch.tensor(indices), chunk_size)


# make batches that are all supervised or all unsupervised
# the model handles balancing the losses between supervised and unsupervised in training, so we don't need to worry
# it's convenient to have equal numbers of labeled and unlabeled batches, so we adjust the unlabeled batch size
class SemiSupervisedBatchSampler(Sampler):
    def __init__(self, dataset: ReadSetDataset, batch_size):
        self.artifact_indices = [n for n in range(len(dataset)) if dataset[n].label() == Label.ARTIFACT]
        self.non_artifact_indices = [n for n in range(len(dataset)) if dataset[n].label() == Label.VARIANT]
        self.unlabeled_indices = [n for n in range(len(dataset)) if dataset[n].label() == Label.UNLABELED]
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

        all_labeled = len(self.unlabeled_indices) == 0
        unlabeled_batch_size = None if all_labeled else \
            round((len(labeled_indices) / len(self.unlabeled_indices)) * self.batch_size)

        labeled_batches = chunk(labeled_indices, self.batch_size)
        unlabeled_batches = None if all_labeled else chunk(self.unlabeled_indices, unlabeled_batch_size)
        combined = [batch.tolist() for batch in list(labeled_batches if all_labeled else (labeled_batches + unlabeled_batches))]
        random.shuffle(combined)
        return iter(combined)

    def __len__(self):
        return len(self.artifact_indices) * 2 // self.batch_size + len(self.artifact_indices) // self.batch_size
