import numpy as np
import torch
from torch import Tensor, IntTensor, FloatTensor

from typing import List
import sys

from mutect3 import utils
from mutect3.data.read_set_dataset import EXTRA_READ_TENSOR_LENGTH
from mutect3.utils import Variation, Label

ENCODING_NAMES = {'M', 'I', 'D', 'E', 'Q', 'X'}


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


# quick and dirty tensor representation:
# first dimension is indel: 1 for insertion, -1 for deletion, 0 for match/substitution
# second dimension is quality: 0 for high-qual, 0.5 for medium low, 1 for very low
# third and fourth dimensions are substituted/inserted base: (0,0) for match, (1,0) for A (-1,0) for T, (0,1) for C, (0,-1) for G
# fifth dimension is end of read / past end of read: 1 if no read at this position, 0 otherwise
def make_tensor_from_read_string(read_string: str, expected_size) -> np.ndarray:
    """
    convert string of form ACCGTA into 4-channel one-hot tensor
    [ [1, 0, 0, 0, 0, 1],   # A channel
      [0, 1, 1, 0, 0, 0],   # C channel
      [0, 0, 0, 1, 0, 0],   # G channel
      [0, 0, 0, 0, 1, 0] ]  # T channel
    """
    result = np.zeros([5, expected_size])
    position = 0
    token_start = 0
    while token_start < len(read_string):
        encoding_type = read_string[token_start]

        token_end = token_start + 1
        while token_end < len(read_string) and not read_string[token_end] in ENCODING_NAMES:
            token_end += 1

        rest_of_token = read_string[token_start + 1:token_end]

        if encoding_type == 'M':    # high-quality matches -- format is eg M10 for 10 consecutive high-qual matching bases
            run_length = int(rest_of_token)
            position += run_length  # do nothing; for match the tensor remains all zeros as initialized
        elif encoding_type == 'E':  # before/after end of read -- format is eg E5
            run_length = int(rest_of_token)
            result[4, position:position+run_length] = 1 # set last channel to 1 for end of read
            position += run_length
        elif encoding_type == 'D':  # deletion -- format is eg D2
            run_length = int(rest_of_token)
            result[0, position:position+run_length] = -1 # set first channel to -1 for deletion
            position += run_length
        else:  # insertion format is eg IAcT, low-qual format is eg Qg, mismatch format is eg XTT
            run_length = len(rest_of_token)
            if encoding_type == 'I':
                result[0, position:position + run_length] = 1  # set first channel to +1 for insertion
            for m in range(run_length):
                base = rest_of_token[m]
                if position >= expected_size:   # TODO: delete this debug
                    print(read_string)
                    print(result)
                    print(encoding_type)
                    print(run_length)
                    print(m)
                    print(rest_of_token)
                    print(len(rest_of_token))
                result[1, position] = 1 if base == 'N' else (0 if base.isupper() else 0.5)  # encode the qual
                if not encoding_type == 'Q':    # the base encoding does not apply for low-qual matches
                    upper_base = base.upper()
                    if upper_base == 'A':
                        result[2,position] = 1
                    elif upper_base == 'T':
                        result[2,position] = -1
                    elif upper_base == 'C':
                        result[3, position] = 1
                    elif upper_base == 'G':
                        result[3, position] = -1
                position += 1
        token_start = token_end # get ready for next token
    # done parsing the string.  We should have parsed exactly the right amount
    assert token_end == len(read_string)

    return result


class ReadSet:
    """
    :param ref_sequence_2d  2D tensor with 4 rows, one for each "channel" A,C, G, T, with each column a position, centered
                                at the alignment start of the variant
    :param ref_reads_2d   2D tensor, each row corresponding to one read supporting the reference allele
    :param alt_reads_2d   2D tensor, each row corresponding to one read supporting the alternate allele
    :param info_array_1d  1D tensor of information about the variant as a whole
    :param label        an object of the Label enum artifact, non-artifact, unlabeled
    """
    def __init__(self, ref_sequence_2d: np.ndarray, ref_reads_2d: np.ndarray,  alt_reads_2d: np.ndarray,
                 info_array_1d: np.ndarray, label: utils.Label, extra_tensor_3d: np.ndarray, variant_string: str = None):
        # Note: if changing any of the data fields below, make sure to modify the size_in_bytes() method below accordingly!
        self.ref_sequence_2d = ref_sequence_2d
        self.ref_reads_2d = ref_reads_2d
        self.alt_reads_2d = alt_reads_2d
        self.info_array_1d = info_array_1d
        self.label = label
        self.extra_tensor_3d = extra_tensor_3d
        self.variant_string = variant_string

        self.nbytes = (self.ref_reads_2d.nbytes if self.ref_reads_2d is not None else 0) + self.alt_reads_2d.nbytes + \
                      self.info_array_1d.nbytes + sys.getsizeof(self.label) + self.extra_tensor_3d.nbytes

    # gatk_info tensor comes from GATK and does not include one-hot encoding of variant type
    @classmethod
    def from_gatk(cls, ref_sequence_string: str, variant_type: utils.Variation, ref_tensor: np.ndarray, alt_tensor: np.ndarray,
                 gatk_info_tensor: np.ndarray, label: utils.Label, read_strings: List[str], variant_string: str = None):
        info_tensor = np.hstack([gatk_info_tensor, variant_type.one_hot_tensor()])

        # make 3D tensors of num ref/alt reads x 5 (channels) x 2*padding + 1
        expected_size = len(ref_sequence_string)
        extra_tensor_3d = np.stack([make_tensor_from_read_string(rs, expected_size) for rs in read_strings], axis=0)
        return cls(make_sequence_tensor(ref_sequence_string), ref_tensor, alt_tensor, info_tensor, label, extra_tensor_3d, variant_string)

    def size_in_bytes(self):
        return self.nbytes

    def variant_type_one_hot(self):
        return self.info_array_1d[-len(Variation):]


# TODO: sparsify the extra 3d tensors by saving their indices, values, shapes
def save_list_of_read_sets(read_sets: List[ReadSet], file):
    """
    note that torch.save works fine with numpy data
    :param read_sets:
    :param file:
    :return:
    """
    ref_sequence_tensors = [datum.ref_sequence_2d for datum in read_sets]
    ref_tensors = [datum.ref_reads_2d for datum in read_sets]
    alt_tensors = [datum.alt_reads_2d for datum in read_sets]
    info_tensors = [datum.info_array_1d for datum in read_sets]
    labels = IntTensor([datum.label.value for datum in read_sets])

    flat_extra_tensors = [datum.extra_tensor_3d.flatten() for datum in read_sets]
    sparse_indices = [tens.nonzero()[0] for tens in flat_extra_tensors] # note that np.nonzero() return a tuple, the 0th element of which are the indices
    sparse_values = [tens[idx] for tens, idx in zip(flat_extra_tensors, sparse_indices)]

    torch.save([ref_sequence_tensors, ref_tensors, alt_tensors, info_tensors, labels, sparse_indices, sparse_values], file)


def reconstitute_extra_tensor(ref, alt, sparse_indices, sparse_values):
    num_reads = (0 if ref is None else len(ref)) + len(alt)
    result = np.zeros(num_reads * EXTRA_READ_TENSOR_LENGTH * 5)
    result[sparse_indices] = sparse_values
    result.resize((num_reads, 5, EXTRA_READ_TENSOR_LENGTH))     #in-place
    return result


# TODO: adjust based on the sparsifying above
def load_list_of_read_sets(file) -> List[ReadSet]:
    """
    file is torch, output is converted back to numpy
    :param file:
    :return:
    """
    ref_sequence_tensors, ref_tensors, alt_tensors, info_tensors, labels, sparse_indices, sparse_values = torch.load(file)
    return [ReadSet(ref_sequence_tensor, ref, alt, info, utils.Label(label), reconstitute_extra_tensor(ref, alt, sparse_idx, sparse_val)) for ref_sequence_tensor, ref, alt, info, label, sparse_idx, sparse_val in
            zip(ref_sequence_tensors, ref_tensors, alt_tensors, info_tensors, labels.tolist(), sparse_indices, sparse_values)]


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

        #list_of_ref_extra_tensors = [item.ref_extra_tensor_3d for item in data] if self.ref_count > 0 else []
        #list_of_alt_extra_tensors = [item.alt_extra_tensor_3d for item in data]
        #self.extra_reads_3d = torch.vstack(list_of_ref_extra_tensors + list_of_alt_extra_tensors).float()

        self.info_2d = torch.from_numpy(np.vstack([item.info_array_1d for item in data])).float()
        self.labels = FloatTensor([1.0 if item.label == Label.ARTIFACT else 0.0 for item in data]) if self.labeled else None
        self._size = len(data)

        self.variant_strings = None if data[0].variant_string is None else [datum.variant_string for datum in data]

    # pin memory for all tensors that are sent to the GPU
    def pin_memory(self):
        self.ref_sequences_2d = self.ref_sequences_2d.pin_memory()
        self.reads_2d = self.reads_2d.pin_memory()
        #self.extra_reads_3d = self.extra_reads_3d.pin_memory()
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
