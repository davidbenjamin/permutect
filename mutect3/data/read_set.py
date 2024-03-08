import numpy as np
import torch
from torch import Tensor, IntTensor, FloatTensor

from typing import List
import sys

from mutect3 import utils
from mutect3.utils import Variation, Label


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
def make_tensor_from_read_string(read_string: str, k: int) -> np.ndarray:
    """
    convert string of form ACCGTA into 4-channel one-hot tensor
    [ [1, 0, 0, 0, 0, 1],   # A channel
      [0, 1, 1, 0, 0, 0],   # C channel
      [0, 0, 0, 1, 0, 0],   # G channel
      [0, 0, 0, 0, 1, 0] ]  # T channel
    """
    result = np.zeros([5, 2*k + 1])
    before, after = tuple(read_string.split())  # before and after/including variant start separated by a space

    token_starts = []
    position = k
    for idx, char in enumerate(after):
        if char == 'M' or char == 'D' or char == 'I' or char == 'Q' or char == 'X':
            token_starts.append(idx)
    token_starts.append(len(after))
    for token_idx in range(len(token_starts) - 1):
        if position > 2*k:
            break

        token = after[token_starts[token_idx]:token_starts[token_idx+1]]

        if token.startswith('M'):
            consecutive_high_qual_match = int(token[1:])    # after the 'M' is the number of high-qual match bases
            position += consecutive_high_qual_match
            # note: the encoding for high-qual matches is all zero, so there's nothing to do!!!
        elif token.startswith('D'): # a single character for one deleted base.  We don't encode deletion length
            result[0, position] = -1
            position += 1
        elif token.startswith('I'):
            inserted_bases = token[1:]
            for base in inserted_bases:
                if position > 2*k:
                    break
                result[0, position] = 1
                # TODO: put in an inserted base and handle qual

                position += 1
        elif token.startswith('Q'):
            low_qual_base = token[1]
            result[1, position] = 1 if low_qual_base == 'N' else 0.5
            position += 1
        elif token.startswith('X'):
            mismatch_base = token[1]
            # TODO: put in an inserted base and handle qual
            position += 1

    # TODO: same for the before variant part

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
    def __init__(self, ref_sequence_2d: np.ndarray, ref_reads_2d: np.ndarray, ref_read_strings: List[str], alt_reads_2d: np.ndarray,
                 alt_read_strings: List[str], info_array_1d: np.ndarray, label: utils.Label, variant_string: str = None):
        # Note: if changing any of the data fields below, make sure to modify the size_in_bytes() method below accordingly!
        self.ref_sequence_2d = ref_sequence_2d
        self.ref_reads_2d = ref_reads_2d
        self.ref_read_strings = ref_read_strings
        self.alt_reads_2d = alt_reads_2d
        self.alt_read_strings = alt_read_strings
        self.info_array_1d = info_array_1d
        self.label = label
        self.variant_string = variant_string

    # gatk_info tensor comes from GATK and does not include one-hot encoding of variant type
    @classmethod
    def from_gatk(cls, ref_sequence_string: str, variant_type: utils.Variation, ref_tensor: np.ndarray, ref_read_strings: List[str], alt_tensor: np.ndarray,
                 alt_read_strings: List[str], gatk_info_tensor: np.ndarray, label: utils.Label, variant_string: str = None):
        info_tensor = np.hstack([gatk_info_tensor, variant_type.one_hot_tensor()])
        return cls(make_sequence_tensor(ref_sequence_string), ref_tensor, ref_read_strings, alt_tensor, alt_read_strings, info_tensor, label, variant_string)

    def size_in_bytes(self):
        return (self.ref_reads_2d.nbytes if self.ref_reads_2d is not None else 0) + self.alt_reads_2d.nbytes + self.info_array_1d.nbytes + sys.getsizeof(self.label)

    def variant_type_one_hot(self):
        return self.info_array_1d[-len(Variation):]


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

    torch.save([ref_sequence_tensors, ref_tensors, alt_tensors, info_tensors, labels], file)


def load_list_of_read_sets(file) -> List[ReadSet]:
    """
    file is torch, output is converted back to numpy
    :param file:
    :return:
    """
    ref_sequence_tensors, ref_tensors, alt_tensors, info_tensors, labels = torch.load(file)
    return [ReadSet(ref_sequence_tensor, ref, alt, info, utils.Label(label)) for ref_sequence_tensor, ref, alt, info, label in
            zip(ref_sequence_tensors, ref_tensors, alt_tensors, info_tensors, labels.tolist())]


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
        self._size = len(data)

        self.variant_strings = None if data[0].variant_string is None else [datum.variant_string for datum in data]

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
