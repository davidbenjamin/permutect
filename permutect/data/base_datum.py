import numpy as np
import torch
from torch import Tensor, IntTensor, FloatTensor

from typing import List
import sys

from permutect.utils import Variation, Label

DEFAULT_NUMPY_FLOAT = np.float16
DEFAULT_GPU_FLOAT = torch.float16
DEFAULT_CPU_FLOAT = torch.float32


def make_1d_sequence_tensor(sequence_string: str) -> np.ndarray:
    """
    convert string of form ACCGTA into tensor [ 0, 1, 1, 2, 3, 0]
    """
    result = np.zeros(len(sequence_string), dtype=np.uint8)
    for n, char in enumerate(sequence_string):
        integer = 0 if char == 'A' else (1 if char == 'C' else (2 if char == 'G' else 3))
        result[n] = integer
    return result


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


# here we just butcher variants longer than 13 bases and chop!!!
def bases_as_base5_int(bases: str) -> int:
    power_of_5 = 1
    bases_to_use = bases if len(bases) < 14 else bases[:13]
    result = 0
    for nuc in bases_to_use:
        coeff = 1 if nuc == 'A' else (2 if nuc == 'C' else (3 if nuc == 'G' else 4))
        result += power_of_5 * coeff
        power_of_5 *= 5
    return result


def bases5_as_base_string(base5: int) -> str:
    result = ""
    remaining = base5
    while remaining > 0:
        digit = remaining % 5
        nuc = 'A' if digit == 1 else ('C' if digit == 2 else ('G' if digit == 3 else 'T'))
        result += nuc
        remaining = (remaining - digit) // 5
    return result


class Variant:
    LENGTH = 4

    def __init__(self, contig: int, position: int, ref: str, alt: str):
        self.contig = contig
        self.position = position
        self.ref = ref
        self.alt = alt

    # note: if base strings are treated as numbers in base 5, uint32 can hold up to 13 bases
    def to_np_array(self):
        return np.array([self.contig, self.position, bases_as_base5_int(self.ref), bases_as_base5_int(self.alt)], dtype=np.uint32)

    # do we need to specify that it's a uint32 array?
    @classmethod
    def from_np_array(cls, np_array: np.ndarray):
        assert len(np_array) == cls.LENGTH
        return cls(round(np_array[0]), round(np_array[1]), bases5_as_base_string(round(np_array[2])), bases5_as_base_string(round(np_array[3])))


class CountsAndSeqLks:
    LENGTH = 6

    def __init__(self, depth: int, alt_count: int, normal_depth: int, normal_alt_count: int,
                 seq_error_log_lk: float, normal_seq_error_log_lk: float):
        self.depth = depth
        self.alt_count = alt_count
        self.normal_depth = normal_depth
        self.normal_alt_count = normal_alt_count
        self.seq_error_log_lk = seq_error_log_lk
        self.normal_seq_error_log_lk = normal_seq_error_log_lk

    def to_np_array(self):
        return np.array([self.depth, self.alt_count, self.normal_depth, self.normal_alt_count, self.seq_error_log_lk, self.normal_seq_error_log_lk])

    @classmethod
    def from_np_array(cls, np_array: np.ndarray):
        assert len(np_array) == cls.LENGTH
        return cls(round(np_array[0]), round(np_array[1]), round(np_array[2]), round(np_array[3]), np_array[4], np_array[5])


class TensorSizes:
    LENGTH = 4

    def __init__(self, ref_count: int, alt_count: int, ref_sequence_length: int, info_tensor_length: int):
        self.ref_count = ref_count
        self.alt_count = alt_count
        self.ref_sequence_length = ref_sequence_length
        self.info_tensor_length = info_tensor_length

    def to_np_array(self):
        return np.array([self.ref_count, self.alt_count, self.ref_sequence_length, self.info_tensor_length])

    @classmethod
    def from_np_array(cls, np_array: np.ndarray):
        assert len(np_array) == cls.LENGTH
        return cls(round(np_array[0]), round(np_array[1]), round(np_array[2]), round(np_array[3]))


class BaseDatum1DStuff:
    NUM_ELEMENTS_AFTER_INFO = 1 + Variant.LENGTH + CountsAndSeqLks.LENGTH   # 1 for the label

    def __init__(self, tensor_sizes: TensorSizes, ref_sequence_1d: np.ndarray, info_array_1d: np.ndarray, label: Label,
                 variant: Variant, counts_and_seq_lks: CountsAndSeqLks, array_override: np.ndarray = None):
        if array_override is None:
            # note: Label is an IntEnum so we can treat label as an integer
            self.array = np.hstack((tensor_sizes.to_np_array(), ref_sequence_1d, info_array_1d, np.array([label]),
                                variant.to_np_array(), counts_and_seq_lks.to_np_array()))
        else:
            self.array = array_override

    def get_alt_count(self):
        return round(self.array[1])

    def get_ref_seq_1d(self):
        ref_seq_length = round(self.array[2])
        return self.array[4:4 + ref_seq_length]

    def get_info_1d(self):
        ref_seq_length = round(self.array[2])
        info_length = round(self.array[3])
        return self.array[4 + ref_seq_length:4 + ref_seq_length + info_length]

    # note: this potentially resizes the array and requires the leading info tensor size element to be modified
    # we do this in preprocessing when adding extra info to the info from GATK.
    # this method should not otherwise be used!!!
    def set_info_1d(self, new_info: np.ndarray):
        ref_seq_length = round(self.array[2])
        old_info_length = round(self.array[3])

        before_info = self.array[:4 + ref_seq_length]
        after_info = self.array[4 + ref_seq_length + old_info_length:]

        self.array[3] = len(new_info)   # update the info tensor size
        self.array = np.hstack((before_info, new_info, after_info))

    def get_label(self):
        return self.array[-self.cls.NUM_ELEMENTS_AFTER_INFO]

    def get_variant(self):
        return Variant.from_np_array(self.array[-self.cls.NUM_ELEMENTS_AFTER_INFO + 1:-CountsAndSeqLks.LENGTH])

    def get_counts_and_seq_lks(self):
        return CountsAndSeqLks.from_np_array(self.array[-CountsAndSeqLks.LENGTH:])

    def to_np_array(self):
        return self.array

    @classmethod
    def from_np_array(cls, np_array: np.ndarray):
        return cls(tensor_sizes=None, ref_sequence_1d=None, info_array_1d=None, label=None,
                   variant=None, counts_and_seq_lks=None, array_override=np_array)


class BaseDatum:
    """
    :param ref_sequence_1d  1D uint8 tensor of bases centered at the alignment start of the variant in form eg ACTG -> [0,1,3,2]
    :param reads_2d   2D tensor, each row corresponding to one read; first all the ref reads, then all the alt reads
    :param info_array_1d  1D tensor of information about the variant as a whole
    :param label        an object of the Label enum artifact, non-artifact, unlabeled
    """
    def __init__(self, reads_2d: np.ndarray, ref_sequence_1d: np.ndarray, alt_count: int,  info_array_1d: np.ndarray, label: Label,
                 variant: Variant, counts_and_seq_lks: CountsAndSeqLks, other_stuff_override: BaseDatum1DStuff = None):
        # Note: if changing any of the data fields below, make sure to modify the size_in_bytes() method below accordingly!

        self.reads_2d = reads_2d

        if other_stuff_override is None:
            self.alt_count = alt_count
            tensor_sizes = TensorSizes(ref_count=len(reads_2d) - alt_count, alt_count=alt_count,
                                       ref_sequence_length=len(ref_sequence_1d), info_tensor_length=len(info_array_1d))
            self.other_stuff = BaseDatum1DStuff(tensor_sizes, ref_sequence_1d, info_array_1d, label, variant, counts_and_seq_lks)
        else:
            self.other_stuff = other_stuff_override
            self.alt_count = other_stuff_override.get_alt_count()

        self.ref_sequence_1d = ref_sequence_1d

        self.label = label
        self.variant = variant
        self.counts_and_seq_lks = counts_and_seq_lks

    # gatk_info tensor comes from GATK and does not include one-hot encoding of variant type
    @classmethod
    def from_gatk(cls, ref_sequence_string: str, variant_type: Variation, ref_tensor: np.ndarray, alt_tensor: np.ndarray,
                 gatk_info_tensor: np.ndarray, label: Label, variant: Variant = None, counts_and_seq_lks: CountsAndSeqLks = None):
        read_tensor = np.vstack([ref_tensor, alt_tensor]) if ref_tensor is not None else alt_tensor
        alt_count = len(alt_tensor)
        info_tensor = np.hstack([gatk_info_tensor, variant_type.one_hot_tensor().astype(DEFAULT_NUMPY_FLOAT)])
        return cls(make_1d_sequence_tensor(ref_sequence_string), alt_count, read_tensor, info_tensor, label, variant, counts_and_seq_lks)

    def size_in_bytes(self):
        return self.reads_2d.nbytes + self.other_stuff.nbytes

    def get_reads_2d(self):
        return self.reads_2d

    def get_other_stuff_1d(self) -> BaseDatum1DStuff:
        return self.other_stuff

    def variant_type_one_hot(self):
        return self.other_stuff.get_info_1d()[-len(Variation):]

    def get_variant_type(self):
        return Variation.get_type(self.variant.ref, str, self.variant.alt)

    def get_ref_reads_2d(self) -> np.ndarray:
        return self.reads_2d[:-self.alt_count]

    def get_alt_reads_2d(self) -> np.ndarray:
        return self.reads_2d[-self.alt_count:]

    def get_info_tensor_1d(self) -> np.ndarray:
        return self.other_stuff.get_info_1d()

    def set_info_tensor_1d(self, new_info: np.ndarray) -> np.ndarray:
        return self.other_stuff.set_info_1d(new_info)

    def get_ref_sequence_1d(self) -> np.ndarray:
        return self.other_stuff.get_ref_seq_1d()


def save_list_base_data(base_data: List[BaseDatum], file):
    """
    note that torch.save works fine with numpy data
    :param base_data:
    :param file:
    :return:
    """
    # TODO: should I combine stack these into big arrays rather than leaving them as lists of arrays?
    read_tensors = [datum.get_reads_2d() for datum in base_data]
    other_stuff = [datum.get_other_stuff_1d().to_np_array() for datum in base_data]
    torch.save([read_tensors, other_stuff], file)


def load_list_of_base_data(file) -> List[BaseDatum]:
    """
    file is torch, output is converted back to numpy
    :param file:
    :return:
    """
    read_tensors, other_stuffs = torch.load(file)
    return [BaseDatum(reads_2d=reads, ref_sequence_1d=None, alt_count=None, info_array_1d=None, label=None,
                      variant=None, counts_and_seq_lks=None, other_stuff_override=BaseDatum1DStuff.from_np_array(other_stuff)) for reads, other_stuff in
            zip(read_tensors, other_stuffs)]


class BaseBatch:
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

    def __init__(self, data: List[BaseDatum]):
        self._original_list = data
        self.labeled = data[0].label != Label.UNLABELED
        self.ref_count = len(data[0].reads_2d) - data[0].alt_count
        self.alt_count = data[0].alt_count

        # for datum in data:
        #    assert (datum.label() != Label.UNLABELED) == self.labeled, "Batch may not mix labeled and unlabeled"
        #    assert len(datum.ref_tensor) == self.ref_count, "batch may not mix different ref counts"
        #    assert len(datum.alt_tensor) == self.alt_count, "batch may not mix different alt counts"

        # TODO: fix this
        self.ref_sequences_2d = torch.permute(torch.nn.functional.one_hot(torch.from_numpy(np.stack([item.ref_sequence_1d for item in data])).long(), num_classes=4), (0, 2, 1))

        list_of_ref_tensors = [item.get_ref_reads_2d() for item in data]
        list_of_alt_tensors = [item.get_alt_reads_2d() for item in data]
        self.reads_2d = torch.from_numpy(np.vstack(list_of_ref_tensors + list_of_alt_tensors))
        self.info_2d = torch.from_numpy(np.vstack([base_datum.get_info_tensor_1d() for base_datum in data]))

        # TODO: get this from the other_stuff_1d tensor
        self.labels = FloatTensor([1.0 if item.label == Label.ARTIFACT else 0.0 for item in data]) if self.labeled else None
        self._size = len(data)

        # TODO: these are unnecessary -- they can be obtained lazily from the original_list
        self.counts_and_likelihoods = [datum.counts_and_seq_lks for datum in data]
        self.variants = None if data[0].variant is None else [datum.variant for datum in data]

    # pin memory for all tensors that are sent to the GPU
    def pin_memory(self):
        self.ref_sequences_2d = self.ref_sequences_2d.pin_memory()
        self.reads_2d = self.reads_2d.pin_memory()
        self.info_2d = self.info_2d.pin_memory()
        self.labels = self.labels.pin_memory()
        return self

    def original_list(self):
        return self._original_list

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


class ArtifactDatum:
    """
    """
    def __init__(self, base_datum: BaseDatum, representation: Tensor):
        # Note: if changing any of the data fields below, make sure to modify the size_in_bytes() method below accordingly!
        assert representation.dim() == 1
        self.representation = representation
        self.label = base_datum.label
        self.variant = base_datum.variant
        self.counts_and_seq_lks = base_datum.counts_and_seq_lks
        self.ref_count = len(base_datum.reads_2d) - base_datum.alt_count
        self.alt_count = base_datum.alt_count
        self.variant_type_one_hot = base_datum.variant_type_one_hot()

    def size_in_bytes(self):
        pass

    def get_variant_type(self):
        for n, var_type in enumerate(Variation):
            if self.variant_type_one_hot[n] > 0:
                return var_type

    def is_labeled(self):
        return self.label != Label.UNLABELED


class ArtifactBatch:
    def __init__(self, data: List[ArtifactDatum]):
        self.labeled = data[0].label != Label.UNLABELED

        self.representations_2d = torch.vstack([item.representation for item in data])
        self.labels = FloatTensor([1.0 if item.label == Label.ARTIFACT else 0.0 for item in data]) if self.labeled else None
        self.ref_counts = IntTensor([int(item.ref_count) for item in data])
        self.alt_counts = IntTensor([int(item.alt_count) for item in data])
        self._size = len(data)
        self.counts_and_likelihoods = [item.counts_and_seq_lks for item in data]
        self.variants = None if data[0].variant is None else [datum.variant for datum in data]

        self._variant_type_one_hot = torch.from_numpy(np.vstack([item.variant_type_one_hot for item in data]))

    # pin memory for all tensors that are sent to the GPU
    def pin_memory(self):
        self.representations_2d = self.representations_2d.pin_memory()
        self.labels = self.labels.pin_memory()
        return self

    def get_representations_2d(self) -> Tensor:
        return self.representations_2d

    def is_labeled(self) -> bool:
        return self.labeled

    def size(self) -> int:
        return self._size

    def variant_type_one_hot(self) -> Tensor:
        return self._variant_type_one_hot

    def variant_type_mask(self, variant_type: Variation) -> Tensor:
        pass

    # return list of variant type integer indices
    def variant_types(self):
        one_hot = self.variant_type_one_hot()
        return [int(x) for x in sum([n * one_hot[:, n] for n in range(len(Variation))])]
