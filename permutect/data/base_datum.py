import copy
import math

import numpy as np
import torch
from torch import Tensor, IntTensor, FloatTensor

from typing import List

from permutect.utils import Variation, Label, trim_alleles_on_right

DEFAULT_NUMPY_FLOAT = np.float16
DEFAULT_GPU_FLOAT = torch.float32
DEFAULT_CPU_FLOAT = torch.float32

# base strings longer than this when encoding data
MAX_NUM_BASES_FOR_ENCODING = 13


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


def truncate_bases_if_necessary(bases: str):
    return bases if len(bases) <= MAX_NUM_BASES_FOR_ENCODING else bases[:MAX_NUM_BASES_FOR_ENCODING]


# here we just butcher variants longer than 13 bases and chop!!!
def bases_as_base5_int(bases: str) -> int:
    power_of_5 = 1
    bases_to_use = truncate_bases_if_necessary(bases)
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


def convert_to_three_ints(n: int, base: int):
    r3 = n % base
    m3 = (n - r3) // base
    r2 = m3 % base
    m2 = (m3 - r2) // base
    r1 = m2 % base
    return r1, r2, r3


def from_three_ints(r1, r2, r3, base):
    return r3 + base*r2 + base*base*r1


class Variant:
    LENGTH = 10  # in order to compress to float16 we need three numbers for the large position integer and the alt, ref encodings
    FLOAT_16_LIMIT = 2048  # float16 can *represent* bigger integers, but this is the limit of being reconstructed correctly
    # if we get this wrong, the position encoding is wrong and the posterior data don't "line up" with the VCF data,
    # causing very little filtering to actually occur

    def __init__(self, contig: int, position: int, ref: str, alt: str):
        self.contig = contig
        self.position = position
        # note: it is very important to trim here, as early as possible, because truncating to 13 or fewer bases
        # does not commute with trimming!!!  If we are not consistent about trimming first, dataset variants and
        # VCF variants might get inconsistent encodings!!!
        self.ref, self.alt = trim_alleles_on_right(ref, alt)

    # note: if base strings are treated as numbers in base 5, uint32 (equivalent to two uint16's) can hold up to 13 bases
    def to_np_array(self):
        base = self.__class__.FLOAT_16_LIMIT
        el1, el2, el3 = convert_to_three_ints(self.position, base)
        el4, el5, el6 = convert_to_three_ints(bases_as_base5_int(self.ref), base)
        el7, el8, el9 = convert_to_three_ints(bases_as_base5_int(self.alt), base)
        return np.array([self.contig, el1, el2, el3, el4, el5, el6, el7, el8, el9], dtype=np.uint16)

    # do we need to specify that it's a uint32 array?
    @classmethod
    def from_np_array(cls, np_array: np.ndarray):
        assert len(np_array) == cls.LENGTH
        base = cls.FLOAT_16_LIMIT
        position = from_three_ints(round(np_array[1]), round(np_array[2]), round(np_array[3]), base)
        ref = bases5_as_base_string(from_three_ints(round(np_array[4]), round(np_array[5]), round(np_array[6]), base))
        alt = bases5_as_base_string(from_three_ints(round(np_array[7]), round(np_array[8]), round(np_array[9]), base))
        return cls(round(np_array[0]), position, ref, alt)

    def get_ref_as_int(self):
        return bases_as_base5_int(self.ref)

    def get_alt_as_int(self):
        return bases_as_base5_int(self.alt)


# count how many times a unit string is repeated at the beginning of a larger string
# eg 'ATATGGG', 'AT' -> 1; 'AGGGGG', 'G' -> 0; 'TTATTATTAGTTA', 'TTA' -> 3
def count_leading_repeats(sequence: str, unit: str):
    result = 0
    idx = 0
    unit_length = len(unit)
    while (idx + unit_length - 1 < len(sequence)) and sequence[idx:idx + unit_length] == unit:
        result += 1
        idx += unit_length
    return result


# same, but at the end of a sequence
# eg 'ATATGGG', 'G' -> 3; 'AGGGGG', 'G' -> 5; 'TTATTATTAGTTA', 'TTA' -> 1
def count_trailing_repeats(sequence: str, unit: str):
    result = 0
    unit_length = len(unit)
    idx = len(sequence) - unit_length   # index at beginning of comparison eg 'GGATC', 'TC' starts at index 5 - 2 = 3, the 'T'
    while idx >= 0 and sequence[idx:idx + unit_length] == unit:
        result += 1
        idx -= unit_length
    return result


def find_factors(n: int):
    result = []
    for m in range(1, int(math.sqrt(n)) + 1):
        if n % m == 0:
            result.append(m)
            if (n // m) > m:
                result.append(n // m)
    result.sort()
    return result


# eg ACGACGACG, ACG -> True; TTATTA, TA -> False
def is_repeat(bases: str, unit: str):
    unit_length = len(unit)
    if len(bases) % unit_length == 0:
        num_repeats = len(bases) // len(unit)
        for repeat_idx in range(num_repeats):
            start = repeat_idx * unit_length
            if bases[start: start + unit_length] != unit:
                return False
        return True
    else:
        return False


# decompose an indel into its most basic repeated unit
# examples: "ATATAT" -> ("AT", 3); "AAAAA" -> ("A", 5); "TTGTTG" -> ("TTG", 2); "ATGTG" -> "ATGTG", 1
def decompose_str_unit(indel_bases: str):
    for unit_length in find_factors(len(indel_bases)):  # note: these are sorted ascending
        unit = indel_bases[:unit_length]
        if is_repeat(indel_bases, unit):
            return unit, (len(indel_bases) // unit_length)
    return indel_bases, 1


def get_str_info_array(ref_sequence_string: str, variant: Variant):
    assert len(ref_sequence_string) % 2 == 1, "must be odd length to have well-defined middle"
    middle_idx = (len(ref_sequence_string) - 1) // 2

    ref, alt = variant.ref, variant.alt

    insertion_length = max(len(alt) - len(ref), 0)
    deletion_length = max(len(ref) - len(alt), 0)

    if len(ref) == len(alt):
        unit, num_units = alt, 1
        repeats_after = count_leading_repeats(ref_sequence_string[middle_idx + len(ref):], unit)
        repeats_before = count_trailing_repeats(ref_sequence_string[:middle_idx], unit)
    elif insertion_length > 0:
        unit, num_units = decompose_str_unit(alt[1:])  # the inserted sequence is everything after the anchor base that matches ref
        repeats_after = count_leading_repeats(ref_sequence_string[middle_idx + len(ref):], unit)
        repeats_before = count_trailing_repeats(ref_sequence_string[:middle_idx+1], unit)   # +1 accounts for the anchor base
    else:
        unit, num_units = decompose_str_unit(ref[1:])  # the deleted sequence is everything after the anchor base
        # it's pretty arbitrary whether we include the deleted bases themselves as 'after' or not
        repeats_after = count_leading_repeats(ref_sequence_string[middle_idx + len(alt):], unit)
        repeats_before = count_trailing_repeats(ref_sequence_string[:middle_idx+1], unit)   # likewise, account for the anchor base
    # note that if indels are left-aligned (as they should be from the GATK) repeats_before really ought to be zero!!
    return np.array([insertion_length, deletion_length, len(unit), num_units, repeats_before, repeats_after])


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
        return cls(round(np_array[0]), round(np_array[1]), round(np_array[2]), round(np_array[3]), float(np_array[4]), float(np_array[5]))


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
    NUM_ELEMENTS_AFTER_INFO = 2 + Variant.LENGTH + CountsAndSeqLks.LENGTH   # 1 for the label, 1 for the source (integer)

    # 1st four elements are tensor sizes: ref count, alt count, ref seq length, info length
    # next is ref sequence as 1D array
    # next is info 1D array
    # label, source (each a single int)
    # Variant (Variant.LENGTH elements)
    # CountsAndSeqLks (CountsAndSeqLks.LENGTH elements)
    def __init__(self, tensor_sizes: TensorSizes, ref_sequence_1d: np.ndarray, info_array_1d: np.ndarray, label: Label, source: int,
                 variant: Variant, counts_and_seq_lks: CountsAndSeqLks, array_override: np.ndarray = None):
        if array_override is None:
            # note: Label is an IntEnum so we can treat label as an integer
            self.array = np.hstack((tensor_sizes.to_np_array(), ref_sequence_1d, info_array_1d, np.array([label, source]),
                                variant.to_np_array(), counts_and_seq_lks.to_np_array()))
        else:
            self.array = array_override

    def get_nbytes(self):
        return self.array.nbytes

    def set_dtype(self, dtype):
        self.array = self.array.astype(dtype)

    def get_alt_count(self) -> int:
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
        return self.array[-self.__class__.NUM_ELEMENTS_AFTER_INFO]

    def set_label(self, label: Label):
        self.array[-self.__class__.NUM_ELEMENTS_AFTER_INFO] = label

    def get_source(self) -> int:
        return round(self.array[-self.__class__.NUM_ELEMENTS_AFTER_INFO + 1])

    def set_source(self, source: int):
        self.array[-self.__class__.NUM_ELEMENTS_AFTER_INFO + 1] = source

    def variant_type_one_hot(self):
        return self.get_info_1d()[-len(Variation):]

    def get_variant(self):
        return Variant.from_np_array(self.array[-self.__class__.NUM_ELEMENTS_AFTER_INFO + 2:-CountsAndSeqLks.LENGTH])

    def get_variant_array(self):
        return self.array[-self.__class__.NUM_ELEMENTS_AFTER_INFO + 2:-CountsAndSeqLks.LENGTH]

    def get_counts_and_seq_lks(self):
        return CountsAndSeqLks.from_np_array(self.array[-CountsAndSeqLks.LENGTH:])

    def get_counts_and_seq_lks_array(self):
        return self.array[-CountsAndSeqLks.LENGTH:]

    def to_np_array(self):
        return self.array

    @classmethod
    def from_np_array(cls, np_array: np.ndarray):
        return cls(tensor_sizes=None, ref_sequence_1d=None, info_array_1d=None, label=None, source=None,
                   variant=None, counts_and_seq_lks=None, array_override=np_array)


class ArtifactDatum1DStuff:
    # 4 for ref count, alt count, label, source; len(Variation) for one-hot variant type
    NUM_ELEMENTS = 4 + Variant.LENGTH + CountsAndSeqLks.LENGTH + len(Variation)
    VARIANT_END_POS = 4 + Variant.LENGTH
    COUNTS_AND_SEQ_LKS_END_POS = VARIANT_END_POS + CountsAndSeqLks.LENGTH

    def __init__(self, base_datum_1d_stuff: BaseDatum1DStuff, array_override: np.ndarray = None):
        if array_override is None:
            # we need ref count, alt count, label, variant, countsandseqlks, variant type one-hot
            # note: Label is an IntEnum so we can treat label as an integer
            self.array = np.zeros(self.__class__.NUM_ELEMENTS)
            self.array[0] = base_datum_1d_stuff.array[0]    # ref count
            self.array[1] = base_datum_1d_stuff.array[1]    # alt count
            self.array[2] = base_datum_1d_stuff.get_label()
            self.array[3] = base_datum_1d_stuff.get_source()
            self.array[4:self.__class__.VARIANT_END_POS] = base_datum_1d_stuff.get_variant_array()
            self.array[self.__class__.VARIANT_END_POS:self.__class__.COUNTS_AND_SEQ_LKS_END_POS] = base_datum_1d_stuff.get_counts_and_seq_lks_array()
            self.array[self.__class__.COUNTS_AND_SEQ_LKS_END_POS:] = base_datum_1d_stuff.variant_type_one_hot()
        else:
            self.array = array_override

    def set_dtype(self, dtype):
        self.array = self.array.astype(dtype)

    def get_ref_count(self):
        return round(self.array[0])

    def get_alt_count(self):
        return round(self.array[1])

    def get_label(self):
        return self.array[2]

    def get_source(self):
        return self.array[3]

    def get_variant(self) -> Variant:
        return Variant.from_np_array(self.array[4:self.__class__.VARIANT_END_POS])

    def get_counts_and_seq_lks(self) -> CountsAndSeqLks:
        return CountsAndSeqLks.from_np_array(self.array[self.__class__.VARIANT_END_POS:self.__class__.COUNTS_AND_SEQ_LKS_END_POS])

    def variant_type_one_hot(self):
        return self.array[self.__class__.COUNTS_AND_SEQ_LKS_END_POS:]

    def to_np_array(self):
        return self.array

    @classmethod
    def from_np_array(cls, np_array: np.ndarray):
        return cls(base_datum_1d_stuff=None, array_override=np_array)


class BaseDatum:
    """
    :param ref_sequence_1d  1D uint8 tensor of bases centered at the alignment start of the variant in form eg ACTG -> [0,1,3,2]
    :param reads_2d   2D tensor, each row corresponding to one read; first all the ref reads, then all the alt reads
    :param info_array_1d  1D tensor of information about the variant as a whole
    :param label        an object of the Label enum artifact, non-artifact, unlabeled
    """
    def __init__(self, reads_2d: np.ndarray, ref_sequence_1d: np.ndarray, alt_count: int,  info_array_1d: np.ndarray, label: Label,
                 source: int, variant: Variant, counts_and_seq_lks: CountsAndSeqLks, other_stuff_override: BaseDatum1DStuff = None):
        # Note: if changing any of the data fields below, make sure to modify the size_in_bytes() method below accordingly!

        self.reads_2d = reads_2d

        if other_stuff_override is None:
            self.alt_count = alt_count
            self.label = label
            self.source = source
            tensor_sizes = TensorSizes(ref_count=len(reads_2d) - alt_count, alt_count=alt_count,
                                       ref_sequence_length=len(ref_sequence_1d), info_tensor_length=len(info_array_1d))
            self.other_stuff = BaseDatum1DStuff(tensor_sizes, ref_sequence_1d, info_array_1d, label, source, variant, counts_and_seq_lks)
        else:
            self.other_stuff = other_stuff_override
            self.alt_count = other_stuff_override.get_alt_count()
            self.label = other_stuff_override.get_label()
            self.source = other_stuff_override.get_source()
        self.set_dtype(np.float16)

    def set_dtype(self, dtype):
        self.other_stuff.set_dtype(dtype)
        self.reads_2d = self.reads_2d.astype(dtype)

    # gatk_info tensor comes from GATK and does not include one-hot encoding of variant type
    @classmethod
    def from_gatk(cls, ref_sequence_string: str, variant_type: Variation, ref_tensor: np.ndarray, alt_tensor: np.ndarray,
                 gatk_info_tensor: np.ndarray, label: Label, source: int, variant: Variant, counts_and_seq_lks: CountsAndSeqLks = None):
        read_tensor = np.vstack([ref_tensor, alt_tensor]) if ref_tensor is not None else alt_tensor
        alt_count = len(alt_tensor)
        str_info = get_str_info_array(ref_sequence_string, variant)
        info_tensor = np.hstack([gatk_info_tensor, str_info, variant_type.one_hot_tensor().astype(DEFAULT_NUMPY_FLOAT)])
        result = cls(read_tensor, make_1d_sequence_tensor(ref_sequence_string), alt_count, info_tensor, label, source, variant, counts_and_seq_lks)
        result.set_dtype(np.float16)
        return result

    def size_in_bytes(self):
        return self.reads_2d.nbytes + self.other_stuff.get_nbytes()

    def get_reads_2d(self):
        return self.reads_2d

    def get_other_stuff_1d(self) -> BaseDatum1DStuff:
        return self.other_stuff

    def variant_type_one_hot(self):
        return self.other_stuff.variant_type_one_hot()

    def set_label(self, label: Label):
        self.label = label
        self.other_stuff.set_label(label)

    def set_source(self, source: int):
        self.source = source
        self.other_stuff.set_source(source)

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

    # returns two length-L 1D arrays of ref stacked on top of alt, with '4' in alt(ref) for deletions(insertions)
    def get_ref_and_alt_sequences(self):
        original_ref_array = self.get_ref_sequence_1d() # gives an array eg ATTTCGG -> [0,3,3,3,1,2,2]
        assert len(original_ref_array) % 2 == 1, "ref sequence length should be odd"
        middle_idx = (len(original_ref_array) - 1) // 2
        max_allele_length = middle_idx  # just kind of a coincidence
        variant = self.other_stuff.get_variant()
        ref, alt = variant.ref[:max_allele_length], variant.alt[:max_allele_length] # these are strings, not integers

        if len(ref) >= len(alt):    # substitution or deletion
            ref_array = original_ref_array
            alt_array = np.copy(ref_array)
            deletion_length = len(ref) - len(alt)
            # add the deletion value '4' to make the alt allele array as long as the ref allele
            alt_allele_array = make_1d_sequence_tensor(alt) if deletion_length == 0 else np.hstack((make_1d_sequence_tensor(alt), np.full(shape=deletion_length, fill_value=4)))
            alt_array[middle_idx: middle_idx + len(alt_allele_array)] = alt_allele_array
        else:   # insertion
            insertion_length = len(alt) - len(ref)
            before = original_ref_array[:middle_idx]
            after = original_ref_array[middle_idx + len(ref):-insertion_length]

            alt_allele_array = make_1d_sequence_tensor(alt)
            ref_allele_array = np.hstack((make_1d_sequence_tensor(ref), np.full(shape=insertion_length, fill_value=4)))

            ref_array = np.hstack((before, ref_allele_array, after))
            alt_array = np.hstack((before, alt_allele_array, after))

        assert len(ref_array) == len(alt_array)
        if len(ref) == len(alt): # SNV -- ref and alt ought to be different
            assert alt_array[middle_idx] != ref_array[middle_idx]
        else:   # indel -- ref and alt are the same at the anchor base, then are different
            assert alt_array[middle_idx + 1] != ref_array[middle_idx + 1]
        return ref_array[:len(original_ref_array)], alt_array[:len(original_ref_array)] # this clipping may be redundant


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
    return [BaseDatum(reads_2d=reads, ref_sequence_1d=None, alt_count=None, info_array_1d=None, label=None, source=None,
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
        self.ref_count = len(data[0].reads_2d) - data[0].alt_count
        self.alt_count = data[0].alt_count
        self.alt_counts = IntTensor([data[0].alt_count for _ in data])

        # for datum in data:
        #    assert (datum.label() != Label.UNLABELED) == self.labeled, "Batch may not mix labeled and unlabeled"
        #    assert len(datum.ref_tensor) == self.ref_count, "batch may not mix different ref counts"
        #    assert len(datum.alt_tensor) == self.alt_count, "batch may not mix different alt counts"

        # num_classes = 5 for A, C, G, T, and deletion / insertion
        ref_alt = [torch.flatten(torch.permute(torch.nn.functional.one_hot(torch.from_numpy(np.vstack(item.get_ref_and_alt_sequences())).long(), num_classes=5), (0,2,1)), 0, 1) for item in data]    # list of 2D (2x5)xL
        # this is indexed by batch, length, channel (aka one-hot base encoding)
        ref_alt_bcl = torch.stack(ref_alt)

        self.ref_sequences_2d = ref_alt_bcl

        list_of_ref_tensors = [item.get_ref_reads_2d() for item in data]
        list_of_alt_tensors = [item.get_alt_reads_2d() for item in data]
        self.reads_2d = torch.from_numpy(np.vstack(list_of_ref_tensors + list_of_alt_tensors))
        self.info_2d = torch.from_numpy(np.vstack([base_datum.get_info_tensor_1d() for base_datum in data]))

        # TODO: get this from the other_stuff_1d tensor
        # labels are 1 for artifact, 0 for non-artifact
        # also 0 for UNLABELED but that value should NEVER be used if we apply the is_labeled_mask properly
        self.labels = FloatTensor([1.0 if item.label == Label.ARTIFACT else 0.0 for item in data])
        self.is_labeled_mask = FloatTensor([0.0 if item.label == Label.UNLABELED else 1.0 for item in data])

        self.sources = IntTensor([item.source for item in data])
        self._size = len(data)

    # pin memory for all tensors that are sent to the GPU
    def pin_memory(self):
        self.ref_sequences_2d = self.ref_sequences_2d.pin_memory()
        self.reads_2d = self.reads_2d.pin_memory()
        self.info_2d = self.info_2d.pin_memory()
        self.labels = self.labels.pin_memory()
        self.is_labeled_mask = self.is_labeled_mask.pin_memory()
        self.sources = self.sources.pin_memory()
        return self

    def copy_to(self, device, non_blocking):
        # For all non-tensor attributes, shallow copy is sufficient
        new_batch = copy.copy(self)
        new_batch.ref_sequences_2d = self.ref_sequences_2d.to(device, non_blocking=non_blocking)
        new_batch.reads_2d = self.reads_2d.to(device, non_blocking=non_blocking)
        new_batch.info_2d = self.info_2d.to(device, non_blocking=non_blocking)
        new_batch.labels = self.labels.to(device, non_blocking=non_blocking)
        new_batch.is_labeled_mask = self.is_labeled_mask.to(device, non_blocking=non_blocking)
        new_batch.sources = self.sources.to(device, non_blocking=non_blocking)
        new_batch.alt_counts = self.alt_counts.to(device, non_blocking=non_blocking)
        return new_batch

    def original_list(self):
        return self._original_list

    def get_reads_2d(self) -> Tensor:
        return self.reads_2d

    def get_alt_counts(self) -> IntTensor:
        return self.alt_counts

    def get_is_labeled_mask(self) -> IntTensor:
        return self.is_labeled_mask

    def get_info_2d(self) -> Tensor:
        return self.info_2d

    def get_ref_sequences_2d(self) -> Tensor:
        return self.ref_sequences_2d

    def size(self) -> int:
        return self._size

    def variant_type_one_hot(self) -> Tensor:
        return self.info_2d[:, -len(Variation):]

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
        self.other_stuff = ArtifactDatum1DStuff(base_datum.get_other_stuff_1d())
        self.set_dtype(np.float16)

    def set_dtype(self, dtype):
        self.representation = self.representation.to(torch.float16)
        self.other_stuff.set_dtype(dtype)

    def get_ref_count(self) -> int:
        return self.other_stuff.get_ref_count()

    def get_alt_count(self) -> int:
        return self.other_stuff.get_alt_count()

    def get_label(self):
        return self.other_stuff.get_label()

    def get_source(self) -> int:
        return round(self.other_stuff.get_source())

    def size_in_bytes(self):
        return self.representation.nbytes + self.other_stuff.nbytes

    def get_other_stuff_1d(self) -> ArtifactDatum1DStuff:
        return self.other_stuff

    def variant_type_one_hot(self):
        return self.other_stuff.variant_type_one_hot()

    def is_labeled(self):
        return self.get_label() != Label.UNLABELED


class ArtifactBatch:
    def __init__(self, data: List[ArtifactDatum]):

        self.original_variants = [d.get_other_stuff_1d().get_variant() for d in data]
        self.original_counts_and_seq_lks = [d.get_other_stuff_1d().get_counts_and_seq_lks() for d in data]

        self.representations_2d = torch.vstack([item.representation for item in data])
        self.labels = FloatTensor([1.0 if item.get_label() == Label.ARTIFACT else 0.0 for item in data])

        sources = IntTensor([item.get_source() for item in data])
        ref_counts = IntTensor([int(item.get_ref_count()) for item in data])
        alt_counts = IntTensor([int(item.get_alt_count()) for item in data])
        is_labeled_mask = FloatTensor([0.0 if item.get_label() == Label.UNLABELED else 1.0 for item in data])
        self.int_tensor = torch.vstack((sources, ref_counts, alt_counts, is_labeled_mask))

        self._size = len(data)

        self._variant_type_one_hot = torch.from_numpy(np.vstack([item.variant_type_one_hot() for item in data]))

    def get_sources(self) -> IntTensor:
        return self.int_tensor[0]

    def get_ref_counts(self) -> IntTensor:
        return self.int_tensor[1]

    def get_alt_counts(self) -> IntTensor:
        return self.int_tensor[2]

    def get_is_labeled_mask(self) -> IntTensor:
        return self.int_tensor[3]

    # pin memory for all tensors that are sent to the GPU
    def pin_memory(self):
        self.representations_2d = self.representations_2d.pin_memory()
        self.labels = self.labels.pin_memory()
        self.int_tensor = self.int_tensor.pin_memory()
        self._variant_type_one_hot = self._variant_type_one_hot.pin_memory()
        return self

    def copy_to(self, device, dtype, non_blocking):
        # For all non-tensor attributes, shallow copy is sufficient
        new_batch = copy.copy(self)

        new_batch.representations_2d = self.representations_2d.to(device=device, dtype=dtype, non_blocking=non_blocking)
        new_batch.labels = self.labels.to(device, dtype=dtype, non_blocking=non_blocking)
        new_batch.int_tensor = self.int_tensor.to(device, dtype=dtype, non_blocking=non_blocking)
        new_batch._variant_type_one_hot = self._variant_type_one_hot.to(device, dtype=dtype, non_blocking=non_blocking)

        return new_batch

    def get_representations_2d(self) -> Tensor:
        return self.representations_2d

    def size(self) -> int:
        return self._size

    def variant_type_one_hot(self) -> Tensor:
        return self._variant_type_one_hot

    # return list of variant type integer indices
    def variant_types(self):
        one_hot = self.variant_type_one_hot()
        return [int(x) for x in sum([n * one_hot[:, n] for n in range(len(Variation))])]
