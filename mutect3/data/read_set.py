import numpy as np
import torch
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


class ReadSet:
    """
    :param ref_sequence_tensor  2D tensor with 4 rows, one for each "channel" A,C, G, T, with each column a position, centered
                                at the alignment start of the variant
    :param ref_tensor   2D tensor, each row corresponding to one read supporting the reference allele
    :param alt_tensor   2D tensor, each row corresponding to one read supporting the alternate allele
    :param info_tensor  1D tensor of information about the variant as a whole
    :param label        an object of the Label enum artifact, non-artifact, unlabeled
    """
    def __init__(self, ref_sequence_tensor: np.ndarray, ref_tensor: np.ndarray, alt_tensor: np.ndarray, info_tensor: np.ndarray, label: utils.Label,
                 variant_string: str = None):
        # Note: if changing any of the data fields below, make sure to modify the size_in_bytes() method below accordingly!
        self.ref_sequence_tensor = ref_sequence_tensor
        self.ref_tensor = ref_tensor
        self.alt_tensor = alt_tensor
        self.info_tensor = info_tensor
        self.label = label
        self.variant_string = variant_string

    # gatk_info tensor comes from GATK and does not include one-hot encoding of variant type
    @classmethod
    def from_gatk(cls, ref_sequence_string: str, variant_type: utils.Variation, ref_tensor: np.ndarray, alt_tensor: np.ndarray,
                 gatk_info_tensor: np.ndarray, label: utils.Label, variant_string: str = None):
        info_tensor = np.hstack([gatk_info_tensor, variant_type.one_hot_tensor()])
        return cls(make_sequence_tensor(ref_sequence_string), ref_tensor, alt_tensor, info_tensor, label, variant_string)

    def size_in_bytes(self):
        return self.ref_tensor.nbytes + self.alt_tensor.nbytes + self.info_tensor.nbytes + sys.getsizeof(self.label)

    def variant_type_one_hot(self):
        return self.info_tensor[-len(Variation):]


def save_list_of_read_sets(read_sets: List[ReadSet], file):
    """
    note that torch.save works fine with numpy data
    :param read_sets:
    :param file:
    :return:
    """
    ref_sequence_tensors = [datum.ref_sequence_tensor for datum in read_sets]
    ref_tensors = [datum.ref_tensor for datum in read_sets]
    alt_tensors = [datum.alt_tensor for datum in read_sets]
    info_tensors = [datum.info_tensor for datum in read_sets]
    labels = torch.IntTensor([datum.label.value for datum in read_sets])

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
        self.ref_count = len(data[0].ref_tensor)
        self.alt_count = len(data[0].alt_tensor)

        # for datum in data:
        #    assert (datum.label() != Label.UNLABELED) == self.labeled, "Batch may not mix labeled and unlabeled"
        #    assert len(datum.ref_tensor) == self.ref_count, "batch may not mix different ref counts"
        #    assert len(datum.alt_tensor) == self.alt_count, "batch may not mix different alt counts"

        self.ref_sequences = torch.from_numpy(np.stack([item.ref_sequence_tensor for item in data])).float()
        self.reads = torch.from_numpy(np.vstack([item.ref_tensor for item in data] + [item.alt_tensor for item in data])).float()
        self.info = torch.from_numpy(np.vstack([item.info_tensor for item in data])).float()
        self.labels = torch.FloatTensor([1.0 if item.label == Label.ARTIFACT else 0.0 for item in data]) if self.labeled else None
        self._size = len(data)

        self.variant_strings = None if data[0].variant_string is None else [datum.variant_string for datum in data]

    # pin memory for all tensors that are sent to the GPU
    def pin_memory(self):
        self.ref_sequences = self.ref_sequences.pin_memory()
        self.reads = self.reads.pin_memory()
        self.info = self.info.pin_memory()
        self.labels = self.labels.pin_memory()
        return self

    def is_labeled(self) -> bool:
        return self.labeled

    def size(self) -> int:
        return self._size

    def variant_type_one_hot(self):
        return self.info[:, -len(Variation):]

    def variant_type_mask(self, variant_type: Variation):
        return self.info[:, -len(Variation) + variant_type.value] == 1

    # return list of variant type integer indices
    def variant_types(self):
        one_hot = self.variant_type_one_hot()
        return [int(x) for x in sum([n*one_hot[:, n] for n in range(len(Variation))])]
