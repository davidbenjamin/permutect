import random
import math
import time
from typing import Iterable
import psutil
import os
import tempfile
import tarfile
from threading import Thread
from collections import defaultdict
from itertools import chain

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler

from mutect3.data.read_set import ReadSet, save_list_of_read_sets, load_list_of_read_sets
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


# ex: [0, 1, 3, 6, 7] -> [0, 1, 1, 3, 3, 3, 6, 7] -- 2 is rounded down to the nearest cutoff, 1 and likewise 4 and 5 are rounded to 3
# assumes that cutoffs include 0 and are ascending
def make_round_down_table(cutoffs):
    table = []
    n = 0
    for idx, cutoff in enumerate(cutoffs[:-1]):
        while n < cutoffs[idx + 1]:
            table.append(cutoff)
            n += 1
    table.append(cutoffs[-1])
    return table


# arrays where array[n] is what n reads are rounded down to
# this is important because we batch by equal ref and alt counts and we would like to reduce
# the number of combinations in order to have big batches
# ALT_ROUNDING = make_round_down_table([0, 1, 2, 3, 4, 5, 7, 10, 13, 16, 20])
REF_ROUNDING = make_round_down_table([0, 1, 5, 10])

ALT_ROUNDING = make_round_down_table(list(range(21)))
# REF_ROUNDING = make_round_down_table(list(range(21)))


def round_down_ref(n: int):
    return n
    #return REF_ROUNDING[min(len(REF_ROUNDING) - 1, n)]


def round_down_alt(n: int):
    return ALT_ROUNDING[min(len(ALT_ROUNDING) - 1, n)]


# check whether all elements are 0 or 1
def is_binary(column_tensor_1d: torch.Tensor):
    assert len(column_tensor_1d.shape) == 1
    return all(el.item() == 0 or el.item() == 1 for el in column_tensor_1d)


def binary_column_indices(tensor_2d: torch.Tensor):
    assert len(tensor_2d.shape) == 2
    return [n for n in range(tensor_2d.shape[1]) if is_binary(tensor_2d[:, n])]


# copy the unnormalized values of binary features (columns)
def restore_binary_columns(normalized, original, binary_columns):
    for idx in binary_columns:
        normalized[:, idx] = original[:, idx]


class ReadSetDataset(Dataset):
    def __init__(self, files=[], data: Iterable[ReadSet] = [], shuffle: bool = True, normalize: bool = True):
        self.data = []
        for table_file in files:
            self.data.extend(read_data(table_file))
        self.data.extend(data)
        if shuffle:
            random.shuffle(self.data)
        data_count = len(self.data)

        # keys = (ref read count, alt read count) tuples; values = list of indices
        # this is used in the batch sampler to make same-shape batches
        self.labeled_indices_by_count = defaultdict(list)
        self.unlabeled_indices_by_count = defaultdict(list)

        for n, datum in enumerate(self.data):
            counts = (len(datum.ref_tensor()), len(datum.alt_tensor()))
            (self.unlabeled_indices_by_count if datum.label() == Label.UNLABELED else self.labeled_indices_by_count)[counts].append(n)

        if normalize:
            # concatenate a bunch of ref tensors and take element-by-element quantiles
            # for simplicity we do sampling with replacement
            random_indices = range(data_count) if data_count <= QUANTILE_DATA_COUNT else torch.randint(0, data_count,
                                                                                     (QUANTILE_DATA_COUNT,)).tolist()

            ref = torch.cat([self.data[n].ref_tensor() for n in random_indices], dim=0)

            # since info is just a 1D tensor, use all of it
            info = torch.stack([datum.info_tensor() for datum in self.data], dim=0)

            read_medians, read_iqrs = medians_and_iqrs(ref)
            info_medians, info_iqrs = medians_and_iqrs(info)

            normalized_info = (info - info_medians) / info_iqrs
            restore_binary_columns(normalized=normalized_info, original=info, binary_columns=binary_column_indices(info))
            binary_read_columns = binary_column_indices(ref)

            # assert that the last columns of the info tensor are the one-hot encoding of variant type, hence binary
            for n in range(len(utils.Variation)):
                assert is_binary(normalized_info[:, -(n + 1)])

            for n in range(len(self.data)):
                raw = self.data[n]
                normalized_ref = (raw.ref_tensor() - read_medians) / read_iqrs
                normalized_alt = (raw.alt_tensor() - read_medians) / read_iqrs
                restore_binary_columns(normalized=normalized_ref, original=raw.ref_tensor(), binary_columns=binary_read_columns)
                restore_binary_columns(normalized=normalized_alt, original=raw.alt_tensor(), binary_columns=binary_read_columns)
                self.data[n] = ReadSet(raw.ref_sequence_tensor, normalized_ref, normalized_alt, normalized_info[n], raw.label())

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


def count_data(dataset_file):
    n = 0
    with open(dataset_file) as file:
        for line in file:
            if Label.is_label(line.strip()):
                n += 1
    return n


# generator that reads a plain text dataset file and yields data
# in posterior model, yield a tuple of ReadSet and PosteriorDatum
def read_data(dataset_file, posterior: bool = False, yield_nones: bool = False, round_down: bool = True):
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
            ref_sequence_string = file.readline().strip()

            gatk_info_tensor = line_to_tensor(file.readline())

            # tumor ref count, tumor alt count, normal ref count, normal alt count -- single-spaced
            # these are the read counts of our possibly-downsampled tensors, not the original sequencing data
            ref_tensor_size, alt_tensor_size, normal_ref_tensor_size, normal_alt_tensor_size = map(int, file.readline().strip().split())

            ref_tensor = read_2d_tensor(file, ref_tensor_size)
            alt_tensor = read_2d_tensor(file, alt_tensor_size)

            if round_down:
                ref_tensor = utils.downsample_tensor(ref_tensor, 0 if ref_tensor is None else round_down_ref(len(ref_tensor)))
                alt_tensor = utils.downsample_tensor(alt_tensor, 0 if alt_tensor is None else round_down_alt(len(alt_tensor)))

            # normal_ref_tensor = read_2d_tensor(file, normal_ref_tensor_size)  # not currently used
            # normal_alt_tensor = read_2d_tensor(file, normal_alt_tensor_size)  # not currently used
            # round down normal tensors as well

            depth, alt_count, normal_depth, normal_alt_count = read_integers(file.readline())

            seq_error_log_likelihood = read_float(file.readline())
            normal_seq_error_log_likelihood = read_float(file.readline())

            assert ref_tensor is None or not torch.sum(ref_tensor).isnan().item(), contig + ":" + str(position)
            assert alt_tensor is None or not torch.sum(alt_tensor).isnan().item(), contig + ":" + str(position)
            assert not torch.sum(gatk_info_tensor).isnan().item(), contig + ":" + str(position)

            datum = ReadSet.from_gatk(ref_sequence_string, Variation.get_type(ref, alt), ref_tensor, alt_tensor, gatk_info_tensor, label)

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
def generate_datasets(dataset_files, max_bytes_per_chunk: int):
    buffers = [[], []]  # a previous and a current buffer
    buffer_idx = 0  # first fill the "previous" buffer
    bytes_in_buffer = 0

    for dataset_file in dataset_files:
        for read_set in read_data(dataset_file, posterior=False, yield_nones=True):
            if read_set is not None:
                buffers[buffer_idx].append(read_set)
                bytes_in_buffer += read_set.size_in_bytes()
            if bytes_in_buffer > max_bytes_per_chunk:
                print("memory usage percent: " + str(psutil.virtual_memory().percent))
                print("bytes in chunk: " + str(bytes_in_buffer))
                bytes_in_buffer = 0

                if buffer_idx == 0:
                    # start filling the next buffer but don't yield anything yet
                    buffer_idx = 1
                else:
                    # we have now filled the second buffer, so we can yield the first
                    yield ReadSetDataset(data=buffers[0], shuffle=True, normalize=True), False
                    buffers[0] = buffers[1]
                    buffers[1] = []

    assert buffers[0], "There seems to have been no data"

    # Possibility 1: buffers[0] is full and buffers[1] is partially full: split buffers[0] and buffers[1] into two equal dataset
    if buffers[1]:
        num_elements = len(buffers[0]) + len(buffers[1])
        split_el = min(num_elements//2, len(buffers[0]))
        yield ReadSetDataset(data=buffers[0][:split_el], shuffle=True, normalize=True), False
        yield ReadSetDataset(data=buffers[0][split_el:] + buffers[1], shuffle=True, normalize=True), True
    # Possibility 2: buffers[1] is empty: yield just buffers[0]
    else:
        yield ReadSetDataset(data=buffers[0], shuffle=True, normalize=True), True








def make_loader_from_file(dataset_file, batch_size, use_gpu: bool, num_workers: int = 0):
    with open(dataset_file, 'rb') as file:
        data = load_list_of_read_sets(file)
        dataset = ReadSetDataset(data=data, shuffle=False, normalize=False)  # data has already been normalized
        return make_semisupervised_data_loader(dataset, batch_size, pin_memory=use_gpu, num_workers=num_workers)


class DataFetchingThread(Thread):
    def __init__(self, dataset_file, batch_size, use_gpu, num_workers: int=0):
        # execute the base constructor
        Thread.__init__(self)
        # set a default value
        self.fetched_data_loader = None
        self.dataset_file = dataset_file
        self.batch_size = batch_size
        self.use_gpu = use_gpu
        self.num_workers = num_workers

    def run(self):
        self.fetched_data_loader = make_loader_from_file(self.dataset_file, self.batch_size, self.use_gpu, num_workers=self.num_workers)

    def get_loader(self):
        return self.fetched_data_loader


class BigReadSetDataset:

    def __init__(self, batch_size: int = 64, max_bytes_per_chunk: int = int(2e9), dataset: ReadSetDataset = None, dataset_files=None,
                 train_valid_metadata_tar_tuple=None, num_workers: int = 0):

        self.use_gpu = torch.cuda.is_available()
        self.batch_size = batch_size
        self.num_workers = num_workers

        # TODO: make this class autocloseable and clean up the temp dirs when closing
        self.train_data_files, self.valid_data_files = [], []
        self.num_read_features, self.num_info_features = None, None

        # compute total counts of labeled/unlabeled, artifact/non-artifact, different variant types
        # during initialization.  These are used for balancing training weights
        self.num_training_data = 0
        self.training_artifact_totals = torch.zeros(len(utils.Variation))  # 1D tensor
        self.training_non_artifact_totals = torch.zeros(len(utils.Variation))  # 1D tensor

        # load from previously processed and saved tar files
        if train_valid_metadata_tar_tuple is not None:
            train_tar, valid_tar, metadata_file = train_valid_metadata_tar_tuple
            assert dataset is None, "can't specify a dataset when loading from tar"
            assert dataset_files is None, "can't specify text dataset files when loading from tar"

            # this parallels the order given in the save method below
            self.num_read_features, self.num_info_features, self.ref_sequence_length, self.num_training_data,\
                self.training_artifact_totals, self.training_non_artifact_totals = torch.load(metadata_file)

            # these will be cleaned up when the program ends, or maybe when the object goes out of scope
            self.train_temp_dir = tempfile.TemporaryDirectory()
            tar = tarfile.open(train_tar)
            tar.extractall(self.train_temp_dir)
            tar.close()
            self.train_data_files = os.listdir(self.train_temp_dir)

            self.valid_temp_dir = tempfile.TemporaryDirectory()
            tar = tarfile.open(valid_tar)
            tar.extractall(self.valid_temp_dir)
            tar.close()
            self.valid_data_files = os.listdir(self.valid_temp_dir)

            # almost done with the tar files case except we need to pre-load in RAM if there is only one train or valid file
        elif dataset is not None:
            assert dataset_files is None, "can't specify text dataset file when initializing from dataset"
            self.train_fits_in_ram = True
            self.valid_fits_in_ram = True
            train, valid = utils.split_dataset_into_train_and_valid(dataset, 0.9)
            train = ReadSetDataset(data=train, shuffle=False, normalize=False)
            valid = ReadSetDataset(data=valid, shuffle=False, normalize=False)

            self.train_loader = make_semisupervised_data_loader(train, batch_size, pin_memory=self.use_gpu, num_workers=num_workers)
            self.valid_loader = make_semisupervised_data_loader(valid, batch_size, pin_memory=self.use_gpu, num_workers=num_workers)
            self.num_read_features = dataset.num_read_features()
            self.num_info_features = dataset.num_info_features()
            self.ref_sequence_length = dataset.ref_sequence_length()
            self.accumulate_totals(self.train_loader)

        elif dataset_files is not None:
            validation_data = []
            validation_size_in_bytes = 0
            for dataset_from_files, is_last_chunk in generate_datasets(dataset_files, max_bytes_per_chunk):
                if self.num_read_features is None:
                    self.num_read_features = dataset_from_files.num_read_features()
                    self.num_info_features = dataset_from_files.num_info_features()
                    self.ref_sequence_length = dataset_from_files.ref_sequence_length()
                else:
                    assert self.num_read_features == dataset_from_files.num_read_features(), "inconsistent number of read features between files"
                    assert self.num_info_features == dataset_from_files.num_info_features(), "inconsistent number of info features between files"
                    assert self.ref_sequence_length == dataset_from_files.ref_sequence_length(), "inconsistent ref sequence lengths between files"
                train, valid = utils.split_dataset_into_train_and_valid(dataset_from_files, 0.9)
                train = ReadSetDataset(data=train, shuffle=False, normalize=False)
                valid = ReadSetDataset(data=valid, shuffle=False, normalize=False)
                self.accumulate_totals(make_semisupervised_data_loader(train, batch_size, pin_memory=self.use_gpu,
                                                                       num_workers=num_workers))

                with tempfile.NamedTemporaryFile(delete=False) as train_data_file:
                    save_list_of_read_sets([datum for datum in train], train_data_file)
                    self.train_data_files.append(train_data_file.name)

                # extend the validation data and save to disk if it has gotten big enough or if there is no more data
                validation_data.extend((datum for datum in valid))
                validation_size_in_bytes += sum((datum.size_in_bytes() for datum in valid))
                if is_last_chunk or validation_size_in_bytes > max_bytes_per_chunk:
                    with tempfile.NamedTemporaryFile(delete=False) as valid_data_file:
                        save_list_of_read_sets(validation_data, valid_data_file)
                        self.valid_data_files.append(valid_data_file.name)
                    validation_data = []
                    validation_size_in_bytes = 0

            # ht to be no validation data leftover because we wrote to disk on last chunk
            assert len(validation_data) == 0
        else:
            raise Exception("must input either a dataset, a dataset text file, or a dataset tarfile")

        # when initializing from files, it's possible we only got one datset's worth of data, in which case it all fits in RAM
        if dataset is None:
            # check if there is only one pickled dataset, in which case we can fit it in RAM
            # note that we already normalized and shuffled
            if len(self.train_data_files) == 1:
                self.train_fits_in_ram = True
                self.train_loader = make_loader_from_file(self.train_data_files[0], batch_size, self.use_gpu,
                                                          num_workers=self.num_workers)
            else:
                self.train_fits_in_ram = False
                self.train_loader = None

            if len(self.valid_data_files) == 1:
                self.valid_fits_in_ram = True
                self.valid_loader = make_loader_from_file(self.valid_data_files[0], batch_size, self.use_gpu,
                                                          num_workers=self.num_workers)
            else:
                self.valid_fits_in_ram = False
                self.valid_loader = None

        assert self.train_fits_in_ram == (self.train_loader is not None)
        assert self.valid_fits_in_ram == (self.valid_loader is not None)
        assert self.train_fits_in_ram == (len(self.train_data_files) <= 1)
        assert self.valid_fits_in_ram == (len(self.valid_data_files) <= 1)
        assert self.num_read_features is not None

    # used only in initialization
    def accumulate_totals(self, training_loader):
        for batch in training_loader:
            self.num_training_data += batch.size()
            if batch.is_labeled():
                labels = batch.labels()  # 1D tensor.  recall that 1 = artifact, 0 = non-artifact
                variant_type_one_hot = batch.variant_type_one_hot()  # 2D tensor; rows are batch index, columns are variant type

                self.training_artifact_totals += torch.sum(labels.unsqueeze(dim=1) * variant_type_one_hot,
                                             dim=0)  # yields 1D tensor of artifact counts for each type
                self.training_non_artifact_totals += torch.sum((1 - labels).unsqueeze(dim=1) * variant_type_one_hot, dim=0)

    def generate_batches(self, epoch_type: utils.Epoch):
        assert epoch_type != utils.Epoch.TEST, "test epochs not supported yet"

        if epoch_type == utils.Epoch.TRAIN and self.train_fits_in_ram:
            for batch in self.train_loader:
                yield batch
        elif epoch_type == utils.Epoch.VALID and self.valid_fits_in_ram:
            for batch in self.valid_loader:
                yield batch
        else:
            data_files = self.train_data_files if epoch_type == utils.Epoch.TRAIN else self.valid_data_files

            data_fetching_thread = DataFetchingThread(data_files[0], self.batch_size, self.use_gpu, num_workers=self.num_workers)
            data_fetching_thread.start()

            for n in range(len(data_files)):
                start = time.time()
                data_fetching_thread.join()
                loader = data_fetching_thread.get_loader()
                wait_time = time.time() - start

                print("Data fetched in background, with {} seconds of additional CPU wait time.".format(wait_time))

                # start new thread to fetch next file in background before yielding from the current file
                if n < len(data_files) - 1:
                    data_fetching_thread = DataFetchingThread(data_files[n+1], self.batch_size, self.use_gpu, num_workers=self.num_workers)
                    data_fetching_thread.start()

                for batch in loader:
                    yield batch

    def save_data(self, train_tar_file, valid_tar_file, metadata_file):
        assert self.train_data_files and self.valid_data_files, "we only save when data was loaded from text file"

        with tarfile.open(train_tar_file, "w") as train_tar:
            for train_file in self.train_data_files:
                train_tar.add(train_file)

        with tarfile.open(valid_tar_file, "w") as valid_tar:
            for valid_file in self.valid_data_files:
                valid_tar.add(valid_file)

        torch.save([self.num_read_features, self.num_info_features, self.ref_sequence_length, self.num_training_data,
                    self.training_artifact_totals, self.training_non_artifact_totals], metadata_file)


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


# ex: chunk([a,b,c,d,e], 3) = [[a,b,c], [d,e]]
def chunk(lis, chunk_size):
    return [lis[i:i + chunk_size] for i in range(0, len(lis), chunk_size)]


# make batches that are all supervised or all unsupervised
# the artifact model handles weights the losses to compensate for class imbalance between supervised and unsupervised
# thus the sampler is not responsible for balancing the data
class SemiSupervisedBatchSampler(Sampler):
    def __init__(self, dataset: ReadSetDataset, batch_size):
        self.labeled_indices_by_count = dataset.labeled_indices_by_count
        self.unlabeled_indices_by_count = dataset.labeled_indices_by_count
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

