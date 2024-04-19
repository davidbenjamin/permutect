import numpy as np
import torch
from torch import Tensor, IntTensor, FloatTensor

from typing import List
import sys

from permutect import utils
from permutect.utils import Variation, Label, MutableInt


def make_sequence_tensor(sequence_string: str) -> np.ndarray:
    """
    convert string of form ACCGTA into 4-channel one-hot tensor
    [ [1, 0, 0, 0, 0, 1],   # A channel
      [0, 1, 1, 0, 0, 0],   # C channel
      [0, 0, 0, 1, 0, 0],   # G channel
      [0, 0, 0, 0, 1, 0] ]  # T channel
    """
    result = np.zeros([4, len(sequence_string)])
    for n, char in enumerate(sequence_string):
        channel = 0 if char == 'A' else (1 if char == 'C' else (2 if char == 'G' else 3))
        result[channel, n] = 1
    return result


class Variant:
    def __init__(self, contig: str, position: int, ref: str, alt: str):
        self.contig = contig
        self.position = position
        self.ref = ref
        self.alt = alt


class CountsAndSeqLks:
    def __init__(self, depth: int, alt_count: int, normal_depth: int, normal_alt_count: int,
                 seq_error_log_lk: float, normal_seq_error_log_lk: float):
        self.depth = depth
        self.alt_count = alt_count
        self.normal_depth = normal_depth
        self.normal_alt_count = normal_alt_count
        self.seq_error_log_lk = seq_error_log_lk
        self.normal_seq_error_log_lk = normal_seq_error_log_lk

    def to_np_array(self):
        return np.ndarray([self.depth, self.alt_count, self.normal_depth, self.normal_alt_count, self.seq_error_log_lk, self.normal_seq_error_log_lk])

    @classmethod
    def from_np_array(cls, np_array: np.ndarray):
        return cls(round(np_array[0]), round(np_array[1]), round(np_array[2]), round(np_array[3]), np_array[4], np_array[5])


class ReadSet:
    """
    :param ref_sequence_2d  2D tensor with 4 rows, one for each "channel" A,C, G, T, with each column a position, centered
                                at the alignment start of the variant
    :param ref_reads_2d   2D tensor, each row corresponding to one read supporting the reference allele
    :param alt_reads_2d   2D tensor, each row corresponding to one read supporting the alternate allele
    :param info_array_1d  1D tensor of information about the variant as a whole
    :param label        an object of the Label enum artifact, non-artifact, unlabeled
    """
    def __init__(self, ref_sequence_2d: np.ndarray, ref_reads_2d: np.ndarray, alt_reads_2d: np.ndarray, info_array_1d: np.ndarray, label: utils.Label, index: int,
                 variant: Variant = None, counts_and_seq_lks: CountsAndSeqLks = None):
        # Note: if changing any of the data fields below, make sure to modify the size_in_bytes() method below accordingly!
        self.ref_sequence_2d = ref_sequence_2d
        self.ref_reads_2d = ref_reads_2d
        self.alt_reads_2d = alt_reads_2d
        self.info_array_1d = info_array_1d
        self.label = label
        self.index = index
        self.variant = variant
        self.counts_and_seq_lks = counts_and_seq_lks

    # gatk_info tensor comes from GATK and does not include one-hot encoding of variant type
    @classmethod
    def from_gatk(cls, ref_sequence_string: str, variant_type: utils.Variation, ref_tensor: np.ndarray, alt_tensor: np.ndarray,
                 gatk_info_tensor: np.ndarray, label: utils.Label, index: int, variant: Variant = None, counts_and_seq_lks: CountsAndSeqLks = None):
        info_tensor = np.hstack([gatk_info_tensor, variant_type.one_hot_tensor()])
        return cls(make_sequence_tensor(ref_sequence_string), ref_tensor, alt_tensor, info_tensor, label, index, variant, counts_and_seq_lks)

    def size_in_bytes(self):
        return (self.ref_reads_2d.nbytes if self.ref_reads_2d is not None else 0) + self.alt_reads_2d.nbytes + self.info_array_1d.nbytes + sys.getsizeof(self.label)

    def variant_type_one_hot(self):
        return self.info_array_1d[-len(Variation):]


def save_list_of_read_sets(read_sets: List[ReadSet], file, datum_index: MutableInt, indices_file=None):
    """
    note that torch.save works fine with numpy data
    :param read_sets:
    :param file:
    :return:
    """
    start_index = datum_index.get()
    num_data = len(read_sets)

    ref_sequence_tensors = [datum.ref_sequence_2d for datum in read_sets]
    ref_tensors = [datum.ref_reads_2d for datum in read_sets]
    alt_tensors = [datum.alt_reads_2d for datum in read_sets]
    info_tensors = [datum.info_array_1d for datum in read_sets]
    labels = IntTensor([datum.label.value for datum in read_sets])
    indices = start_index + torch.arange(num_data).int()

    counts_and_lks = [datum.counts_and_seq_lks.to_np_array() for datum in read_sets]

    torch.save([ref_sequence_tensors, ref_tensors, alt_tensors, info_tensors, labels, indices, counts_and_lks], file)
    datum_index.increment(num_data)

    if indices_file is not None:
        for index, datum in zip(indices.tolist(), read_sets):
            assert datum.variant is not None
            indices_file.write("\t".join([str(index), datum.variant.contig, str(datum.variant.position), datum.variant.ref, datum.variant.alt]) + '\n')


def load_list_of_read_sets(file) -> List[ReadSet]:
    """
    file is torch, output is converted back to numpy
    :param file:
    :return:
    """
    ref_sequence_tensors, ref_tensors, alt_tensors, info_tensors, labels, indices, counts_and_lks = torch.load(file)
    return [ReadSet(ref_sequence_tensor, ref, alt, info, utils.Label(label), index, None, CountsAndSeqLks.from_np_array(cnts_lks)) for ref_sequence_tensor, ref, alt, info, label, index, cnts_lks in
            zip(ref_sequence_tensors, ref_tensors, alt_tensors, info_tensors, labels.tolist(), indices.tolist(), counts_and_lks)]


class ReadSetBatch:
    """
    Read sets have different sizes so we can't form a batch by naively stacking tensors.  We need a custom way
    to collate a list of Datum into a Batch

    collated batch contains:
    2D tensors of ALL ref (alt) reads, not separated by set.
    number of reads in ref (alt) read sets, in same order as read tensors
    info: 2D tensor of info fields, one row per variant
    labels: 1D tensor of 0 if non-artifact, 1 if artifact
    lists of original mutect2_data and site info

    Example: if we have two input data, one with alt reads [[0,1,2], [3,4,5] and the other with
    alt reads [[6,7,8], [9,10,11], [12,13,14] then the output alt reads tensor is
    [[0,1,2], [3,4,5], [6,7,8], [9,10,11], [12,13,14]] and the output counts are [2,3]
    inside the model, the counts will be used to separate the reads into sets
    """

    def __init__(self, data: List[ReadSet]):
        self.labeled = data[0].label != Label.UNLABELED
        self.ref_count = len(data[0].ref_reads_2d) if data[0].ref_reads_2d is not None else 0
        self.alt_count = len(data[0].alt_reads_2d)

        # for datum in data:
        #    assert (datum.label() != Label.UNLABELED) == self.labeled, "Batch may not mix labeled and unlabeled"
        #    assert len(datum.ref_tensor) == self.ref_count, "batch may not mix different ref counts"
        #    assert len(datum.alt_tensor) == self.alt_count, "batch may not mix different alt counts"

        self.ref_sequences_2d = torch.from_numpy(np.stack([item.ref_sequence_2d for item in data])).float()
        list_of_ref_tensors = [item.ref_reads_2d for item in data] if self.ref_count > 0 else []
        self.reads_2d = torch.from_numpy(np.vstack(list_of_ref_tensors + [item.alt_reads_2d for item in data])).float()
        self.info_2d = torch.from_numpy(np.vstack([item.info_array_1d for item in data])).float()
        self.labels = FloatTensor([1.0 if item.label == Label.ARTIFACT else 0.0 for item in data]) if self.labeled else None
        self.indices = IntTensor([item.index for item in data])
        self._size = len(data)

        self.counts_and_likelihoods = [datum.counts_and_seq_lks for datum in data]
        self.variants = None if data[0].variant is None else [datum.variant for datum in data]

    # pin memory for all tensors that are sent to the GPU
    def pin_memory(self):
        self.ref_sequences_2d = self.ref_sequences_2d.pin_memory()
        self.reads_2d = self.reads_2d.pin_memory()
        self.info_2d = self.info_2d.pin_memory()
        self.labels = self.labels.pin_memory()
        return self

    def get_reads_2d(self) -> Tensor:
        return self.reads_2d

    def get_info_2d(self) -> Tensor:
        return self.info_2d

    def get_ref_sequences_2d(self) -> Tensor:
        return self.ref_sequences_2d

    def is_labeled(self) -> bool:
        return self.labeled

    def size(self) -> int:
        return self._size

    def variant_type_one_hot(self) -> Tensor:
        return self.info_2d[:, -len(Variation):]

    def variant_type_mask(self, variant_type: Variation) -> Tensor:
        return self.info_2d[:, -len(Variation) + variant_type.value] == 1

    # return list of variant type integer indices
    def variant_types(self):
        one_hot = self.variant_type_one_hot()
        return [int(x) for x in sum([n*one_hot[:, n] for n in range(len(Variation))])]
