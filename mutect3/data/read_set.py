import numpy as np
import torch
from typing import List
import sys

from mutect3 import utils


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
    def __init__(self, ref_sequence_tensor: np.ndarray, ref_tensor: np.ndarray, alt_tensor: np.ndarray, info_tensor: np.ndarray, label: utils.Label):
        # Note: if changing any of the data fields below, make sure to modify the size_in_bytes() method below accordingly!
        self.ref_sequence_tensor = ref_sequence_tensor
        self.ref_tensor = ref_tensor
        self.alt_tensor = alt_tensor
        self.info_tensor = info_tensor
        self.label = label

    # gatk_info tensor comes from GATK and does not include one-hot encoding of variant type
    @classmethod
    def from_gatk(cls, ref_sequence_string: str, variant_type: utils.Variation, ref_tensor: torch.Tensor, alt_tensor: torch.Tensor,
                 gatk_info_tensor: torch.Tensor, label: utils.Label):
        info_tensor = torch.cat((gatk_info_tensor, variant_type.one_hot_tensor()))
        return cls(make_sequence_tensor(ref_sequence_string), ref_tensor, alt_tensor, info_tensor, label)

    def size_in_bytes(self):
        return sys.getsizeof(self.ref_tensor.storage()) + sys.getsizeof(self.alt_tensor.storage()) + \
               sys.getsizeof(self.info_tensor.storage()) + sys.getsizeof(self.label)


def save_list_of_read_sets(read_sets: List[ReadSet], file):
    # convert to torch because the save method is more flexible
    ref_sequence_tensors = [torch.from_numpy(datum.ref_sequence_tensor) for datum in read_sets]
    ref_tensors = [torch.from_numpy(datum.ref_tensor) for datum in read_sets]
    alt_tensors = [torch.from_numpy(datum.alt_tensor) for datum in read_sets]
    info_tensors = [torch.from_numpy(datum.info_tensor) for datum in read_sets]
    labels = torch.IntTensor([datum.label.value for datum in read_sets])

    torch.save([ref_sequence_tensors, ref_tensors, alt_tensors, info_tensors, labels], file)


def load_list_of_read_sets(file) -> List[ReadSet]:
    # convert back to numpy
    ref_sequence_tensors, ref_tensors, alt_tensors, info_tensors, labels = torch.load(file)
    return [ReadSet(ref_sequence_tensor.numpy(), ref.numpy(), alt.numpy(), info.numpy(), utils.Label(label)) for ref_sequence_tensor, ref, alt, info, label in
            zip(ref_sequence_tensors, ref_tensors, alt_tensors, info_tensors, labels.tolist())]





