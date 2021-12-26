import random
from typing import List
import pandas as pd
from mutect3 import utils

import torch
from torch.distributions.beta import Beta
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler

NUM_READ_FEATURES = 11  # size of each read's feature vector from M2 annotation
NUM_INFO_FEATURES = 9  # size of each variant's info field tensor (3 components for HEC, one each for HAPDOM, HAPCOMP)
# and 5 for ref bases STR info

MIN_REF = 5

EPSILON = 0.00001
DATA_COUNT_FOR_QUANTILES = 10000


class Datum:
    def __init__(self, contig: str, position: int, ref: str, alt: str, ref_tensor: torch.Tensor, alt_tensor: torch.Tensor,
                 info_tensor: torch.Tensor, label: str, normal_depth: int, normal_alt_count: int):
        self._contig = contig
        self._position = position
        self._ref = ref
        self._alt = alt
        self._ref_tensor = ref_tensor
        self._alt_tensor = alt_tensor
        self._info_tensor = info_tensor
        self._label = label
        self._normal_depth = normal_depth
        self._normal_alt_count = normal_alt_count

        diff = len(self.alt()) - len(self.ref())
        self._variant_type = utils.VariantType.SNV if diff == 0 else (
            utils.VariantType.INSERTION if diff > 0 else utils.VariantType.DELETION)

    def contig(self) -> str:
        return self._contig

    def position(self) -> int:
        return self._position

    def ref(self) -> str:
        return self._ref

    def alt(self) -> str:
        return self._alt

    def variant_type(self) -> utils.VariantType:
        return self._variant_type

    def ref_tensor(self) -> torch.Tensor:
        return self._ref_tensor

    def alt_tensor(self) -> torch.Tensor:
        return self._alt_tensor

    def info_tensor(self) -> torch.Tensor:
        return self._info_tensor

    def label(self) -> str:
        return self._label

    def normal_depth(self) -> int:
        return self._normal_depth

    def normal_alt_count(self) -> int:
        return self._normal_alt_count

    def set_label(self, label):
        self._label = label

    # beta is distribution of downsampling fractions
    def downsampled_copy(self, beta: Beta):
        ref_frac = beta.sample().item()
        alt_frac = beta.sample().item()

        ref_length = max(1, round(ref_frac * len(self._ref_tensor)))
        alt_length = max(1, round(alt_frac * len(self._alt_tensor)))
        return Datum(self._contig, self._position, self._ref, self._alt, downsample(self._ref_tensor, ref_length),
                     downsample(self._alt_tensor, alt_length), self._info_tensor, self._label, self._normal_depth,
                     self._normal_alt_count)


class NormalArtifactDatum:
    def __init__(self, normal_alt_count: int, normal_depth: int, tumor_alt_count: int, tumor_depth: int,
                 downsampling: float, variant_type: str):
        self._normal_alt_count = normal_alt_count
        self._normal_depth = normal_depth
        self._tumor_alt_count = tumor_alt_count
        self._tumor_depth = tumor_depth
        self._downsampling = downsampling
        # TODO: use the variant type class
        self._variant_type = variant_type

    def normal_alt_count(self) -> int:
        return self._normal_alt_count

    def normal_depth(self) -> int:
        return self._normal_depth

    def tumor_alt_count(self) -> int:
        return self._tumor_alt_count

    def tumor_depth(self) -> int:
        return self._tumor_depth

    def downsampling(self) -> float:
        return self._downsampling

    def variant_type(self) -> str:
        return self._variant_type


def read_normal_artifact_data(table_file, shuffle=True) -> List[NormalArtifactDatum]:
    df = pd.read_table(table_file, header=0)
    df = df.astype({"normal_alt": int, "normal_dp": int, "tumor_alt": int, "tumor_dp": int, "downsampling": float,
                    "type": str})

    data = []
    for _, row in df.iterrows():
        data.append(NormalArtifactDatum(row['normal_alt'], row['normal_dp'], row['tumor_alt'], row['tumor_dp'],
                                        row['downsampling'], row['type']))

    if shuffle:
        random.shuffle(data)
    return data


def downsample(tensor: torch.Tensor, downsample_fraction) -> torch.Tensor:
    return tensor if (downsample_fraction is None or downsample_fraction >= len(tensor)) \
        else tensor[torch.randperm(len(tensor))[:downsample_fraction]]


class NormalArtifactBatch:

    def __init__(self, data: List[NormalArtifactDatum]):
        self._normal_alt = torch.IntTensor([datum.normal_alt_count() for datum in data])
        self._normal_depth = torch.IntTensor([datum.normal_depth() for datum in data])
        self._tumor_alt = torch.IntTensor([datum.tumor_alt_count() for datum in data])
        self._tumor_depth = torch.IntTensor([datum.tumor_depth() for datum in data])
        self._downsampling = torch.FloatTensor([datum.downsampling() for datum in data])
        self._variant_type = [datum.variant_type() for datum in data]
        self._size = len(data)

    def size(self) -> int:
        return self._size

    def normal_alt(self) -> torch.IntTensor:
        return self._normal_alt

    def normal_depth(self) -> torch.IntTensor:
        return self._normal_depth

    def tumor_alt(self) -> torch.IntTensor:
        return self._tumor_alt

    def tumor_depth(self) -> torch.IntTensor:
        return self._tumor_depth

    def downsampling(self) -> torch.FloatTensor:
        return self._downsampling

    def variant_type(self) -> List[str]:
        return self._variant_type


# Read sets have different sizes so we can't form a batch by naively stacking tensors.  We need a custom way
# to collate a list of Datum into a Batch

# collated batch contains:
# 2D tensors of ALL ref (alt) reads, not separated by set.
# number of reads in ref (alt) read sets, in same order as read tensors
# info: 2D tensor of info fields, one row per variant
# labels: 1D tensor of 0 if non-artifact, 1 if artifact
# lists of original mutect2_data and site info
# Example: if we have two input data, one with alt reads [[0,1,2], [3,4,5] and the other with
# alt reads [[6,7,8], [9,10,11], [12,13,14] then the output alt reads tensor is
# [[0,1,2], [3,4,5], [6,7,8], [9,10,11], [12,13,14]] and the output counts are [2,3]
# inside the model, the counts will be used to separate the reads into sets
class Batch:

    # given list of slice sizes, produce a list of index slice objects
    # eg input = [2,3,1] --> [slice(0,2), slice(2,5), slice(5,6)]
    def make_slices(sizes, offset=0):
        slice_ends = offset + torch.cumsum(sizes, dim=0)
        return [slice(offset if n == 0 else slice_ends[n - 1], slice_ends[n]) for n in range(len(sizes))]

    def __init__(self, data: List[Datum]):
        self._original_list = data  # keep this for downsampling augmentation
        self.labeled = data[0].label() != "UNLABELED"
        for datum in data:
            if (datum.label() != "UNLABELED") != self.labeled:
                raise Exception("Batch may not mix labeled and unlabeled")

        self._ref_counts = torch.IntTensor([len(item.ref_tensor()) for item in data])
        self._alt_counts = torch.IntTensor([len(item.alt_tensor()) for item in data])
        self._ref_slices = Batch.make_slices(self._ref_counts)
        self._alt_slices = Batch.make_slices(self._alt_counts, torch.sum(self._ref_counts))
        self._reads = torch.cat([item.ref_tensor() for item in data] + [item.alt_tensor() for item in data], dim=0)
        self._info = torch.stack([item.info_tensor() for item in data], dim=0)
        self._labels = torch.FloatTensor([1.0 if item.label() == "ARTIFACT" else 0.0 for item in data]) if self.labeled else None
        self._ref = [item.ref() for item in data]
        self._alt = [item.alt() for item in data]
        self._variant_type = [item.variant_type() for item in data]
        self._size = len(data)

        # TODO: variant type needs to go in constructor -- and maybe it should be utils.VariantType, not str
        # TODO: we might need to change the counts in this constructor
        normal_artifact_data = [NormalArtifactDatum(item.normal_alt_count(), item.normal_depth(),
                                                            len(item.alt_tensor()),
                                                            len(item.alt_tensor()) + len(item.ref_tensor()),
                                                            1.0, item.variant_type) for item in data]
        self._normal_artifact_batch = NormalArtifactBatch(normal_artifact_data)

    def augmented_copy(self, beta):
        return Batch([datum.downsampled_copy(beta) for datum in self._original_list])

    def is_labeled(self) -> bool:
        return self.labeled

    def size(self) -> int:
        return self._size

    def reads(self) -> torch.Tensor:
        return self._reads

    def ref_slices(self) -> List[slice]:
        return self._ref_slices

    def alt_slices(self) -> List[slice]:
        return self._alt_slices

    def ref_counts(self) -> torch.IntTensor:
        return self._ref_counts

    def alt_counts(self) -> torch.IntTensor:
        return self._alt_counts

    def info(self) -> torch.Tensor:
        return self._info

    def labels(self) -> torch.Tensor:
        return self._labels

    def variant_type(self) -> List[utils.VariantType]:
        return self._variant_type

    def normal_artifact_batch(self) -> NormalArtifactBatch:
        return self._normal_artifact_batch


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


class Mutect3Dataset(Dataset):
    def __init__(self, table_files):
        self.data = []
        for table_file in table_files:
            self.data.extend(read_data(table_file))
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

        return Datum(raw.contig(), raw.position(), raw.ref(), raw.alt(), normalized_ref, normalized_alt, normalized_info,
              raw.label(), raw.normal_depth(), raw.normal_alt_count())


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


def read_data(dataset_file):
    data = []
    with open(dataset_file) as file:
        while True:
            # get label
            first_line = file.readline()
            if not first_line:
                break
            label = first_line.strip()

            #contig:position,ref->alt
            locus, mutation = file.readline().strip().split(",")
            contig, position = locus.split(":")
            position = int(position)
            ref, alt = mutation.split("->")

            ref_bases = file.readline().strip() # not currently used

            info_tensor = line_to_tensor(file.readline())

            # tumor ref count, tumor alt count, normal ref count, normal alt count -- single-spaced
            tumor_ref_count, tumor_alt_count, normal_ref_count, normal_alt_count = map(int, file.readline().strip().split())

            ref_tensor = read_2d_tensor(file, tumor_ref_count)
            alt_tensor = read_2d_tensor(file, tumor_alt_count)
            #normal_tensor = read_2d_tensor(file, normal_ref_count)  # not currently used
            #normal_tensor = read_2d_tensor(file, normal_alt_count)  # not currently used


            # pre-downsampling (pd) counts
            pd_tumor_depth, pd_tumor_alt, pd_normal_depth, pd_normal_alt = read_integers(file.readline())

            datum = Datum(contig, position, ref, alt, ref_tensor, alt_tensor, info_tensor, label, pd_normal_depth, pd_normal_alt)

            if tumor_ref_count >= MIN_REF and tumor_alt_count > 0:
                data.append(datum)

    return data


def chunk(indices, chunk_size):
    return torch.split(torch.tensor(indices), chunk_size)


# make batches that are all supervised or all unsupervised
# the model handles balancing the losses between supervised and unsupervised in training, so we don't need to worry
# it's convenient to have equal numbers of labeled and unlabeled batches, so we adjust the unlabeled batch size
class SemiSupervisedBatchSampler(Sampler):
    def __init__(self, dataset: Mutect3Dataset, batch_size):
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


# this is used for training and validation but not deployment / testing
def make_semisupervised_data_loader(training_dataset, batch_size):
    sampler = SemiSupervisedBatchSampler(training_dataset, batch_size)
    return DataLoader(dataset=training_dataset, batch_sampler=sampler, collate_fn=Batch)


def make_test_data_loader(test_dataset, batch_size):
    return DataLoader(dataset=test_dataset, batch_size=batch_size, collate_fn=Batch)


class NormalArtifactDataset(Dataset):
    def __init__(self, table_files):
        self.data = []
        for table_file in table_files:
            self.data.extend(read_normal_artifact_data(table_file))
        random.shuffle(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


def make_normal_artifact_data_loader(dataset: NormalArtifactDataset, batch_size):
    return DataLoader(dataset=dataset, batch_size=batch_size, collate_fn=NormalArtifactBatch)
