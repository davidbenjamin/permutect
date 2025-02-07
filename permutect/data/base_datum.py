from __future__ import annotations
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

MAX_FLOAT_16 = torch.finfo(torch.float16).max
MIN_FLOAT_16 = torch.finfo(torch.float16).min


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


class ParentDatum:
    """
    contains data that apply to a candidate mutation as a whole i.e. not the read sets.  These are organized into a single
    LongTensor, containing some quantities that are inherently integral and some that are cast as longs by multiplying
    with a large number and rounding.
    """
    FLOAT_TO_LONG_MULTIPLIER = 100000

    # indices of inherently integral quantities
    REF_COUNT_IDX = 0               # potentially downsampled -- the actual size of the ref reads tensor
    ALT_COUNT_IDX = 1               # potentially downsampled -- the actual size of the alt reads tensor
    REF_SEQ_LENGTH_IDX = 2          # length of the float sub-array encoding the reference sequence
    INFO_LENGTH_IDX = 3             # length of the float sub-array encoding the info vector
    LABEL_IDX = 4                   # the IntEnum label
    VARIANT_TYPE_IDX = 5            # the IntEnum variant type
    SOURCE_IDX = 6                  # the integer encoding the source

    ORIGINAL_DEPTH_IDX = 7          # the original depth of the sequencing data before downsampling
    ORIGINAL_ALT_COUNT_IDX = 8      # the original alt count of the sequencing data before downsampling
    ORIGINAL_NORMAL_DEPTH_IDX = 9   # the original matched normal sample depth of the sequencing data before downsampling
    ORIGINAL_ALT_COUNT_IDX = 10     # the original matched normal sample alt count of the sequencing data before downsampling

    CONTIG_IDX = 11                 # the index of the contig/chromosome
    POSITION_IDX = 12               # the position of the variant start within the contig
    REF_ALLELE_AS_BASE_5_IDX = 13   # the reference allele encoded as a single base 5 integer
    ALT_ALLELE_AS_BASE_5_IDX = 14   # the reference allele encoded as a single base 5 integer

    # FloatTensor indices
    SEQ_ERROR_LOG_LK_IDX = 15
    NORMAL_SEQ_ERROR_LOG_LK_IDX = 16

    NUM_SCALAR_ELEMENTS = NORMAL_SEQ_ERROR_LOG_LK_IDX + 1
    REF_SEQ_START_IDX = NUM_SCALAR_ELEMENTS

    # after these come the variable-length sub-arrays (not within a single dataset, but in principle variable length for
    # different versions of Permutect or different sequencing) for the reference sequence context and the info tensor

    def __init__(self, array: np.ndarray):
        assert array.ndim == 1 and len(array) >= ParentDatum.NUM_SCALAR_ELEMENTS
        self.array: np.ndarray = np.int64(array)

    @classmethod
    def make_datum_without_reads(cls, label: Label, variant_type: Variation, source: int,
        original_depth: int, original_alt_count: int, original_normal_depth: int, original_normal_alt_count: int,
        contig: int, position: int, ref_allele: str, alt_allele: str,
        seq_error_log_lk: float, normal_seq_error_log_lk: float, ref_seq_array: np.ndarray, info_array: np.ndarray):
        """
        We are careful about our float to long conversions here and in the getters!
        """
        ref_seq_length, info_length = len(ref_seq_array), len(info_array)
        result = cls(np.zeros(ParentDatum.NUM_SCALAR_ELEMENTS + ref_seq_length + info_length, dtype=np.int64))
        # ref count and alt count remain zero
        result.array[ParentDatum.REF_SEQ_LENGTH_IDX] = ref_seq_length
        result.array[ParentDatum.INFO_LENGTH_IDX] = info_length

        result.array[ParentDatum.LABEL_IDX] = label
        result.array[ParentDatum.VARIANT_TYPE_IDX] = variant_type
        result.array[ParentDatum.SOURCE_IDX] = source

        result.array[ParentDatum.ORIGINAL_DEPTH_IDX] = original_depth
        result.array[ParentDatum.ORIGINAL_ALT_COUNT_IDX] = original_alt_count
        result.array[ParentDatum.ORIGINAL_NORMAL_DEPTH_IDX] = original_normal_depth
        result.array[ParentDatum.ORIGINAL_NORMAL_ALT_COUNT_IDX] = original_normal_alt_count

        result.array[ParentDatum.CONTIG_IDX] = contig
        result.array[ParentDatum.POSITION_IDX] = position
        result.array[ParentDatum.REF_ALLELE_AS_BASE_5_IDX] = bases_as_base5_int(ref_allele)
        result.array[ParentDatum.ALT_ALLELE_AS_BASE_5_IDX] = bases_as_base5_int(alt_allele)

        result.array[ParentDatum.SEQ_ERROR_LOG_LK_IDX] = round(seq_error_log_lk * ParentDatum.FLOAT_TO_LONG_MULTIPLIER)
        result.array[ParentDatum.NORMAL_SEQ_ERROR_LOG_LK_IDX] = round(normal_seq_error_log_lk * ParentDatum.FLOAT_TO_LONG_MULTIPLIER)

        ref_seq_start = ParentDatum.REF_SEQ_START_IDX
        ref_seq_end = ref_seq_start + ref_seq_length
        info_end = ref_seq_end + info_length
        result.array[ref_seq_start:ref_seq_end] = np.int64(ref_seq_array * ParentDatum.FLOAT_TO_LONG_MULTIPLIER)
        result.array[ref_seq_end:info_end] = np.int64(info_array * ParentDatum.FLOAT_TO_LONG_MULTIPLIER)

    def get_ref_count(self) -> int:
        return self.array[ParentDatum.REF_COUNT_IDX]

    def get_alt_count(self) -> int:
        return self.array[ParentDatum.ALT_COUNT_IDX]

    def get_ref_seq_array_length(self) -> int:
        return self.array[ParentDatum.REF_SEQ_LENGTH_IDX]

    def get_info_array_length(self) -> int:
        return self.array[ParentDatum.INFO_LENGTH_IDX]

    def get_label(self) -> int:
        return self.array[ParentDatum.LABEL_IDX]

    def set_label(self, label: Label):
        self.array[ParentDatum.LABEL_IDX] = label

    def get_variant_type(self) -> int:
        return self.array[ParentDatum.VARIANT_TYPE_IDX]

    def get_source(self) -> int:
        return self.array[ParentDatum.SOURCE_IDX]

    def set_source(self, source: int):
        self.array[ParentDatum.SOURCE_IDX] = source

    def get_original_depth(self) -> int:
        return self.array[ParentDatum.ORIGINAL_DEPTH_IDX]

    def get_original_alt_count(self) -> int:
        return self.array[ParentDatum.ORIGINAL_ALT_COUNT_IDX]

    def get_original_normal_depth(self) -> int:
        return self.array[ParentDatum.ORIGINAL_NORMAL_DEPTH_IDX]

    def get_original_normal_alt_count(self) -> int:
        return self.array[ParentDatum.ORIGINAL_NORMAL_ALT_COUNT_IDX]

    def get_contig(self) -> int:
        return self.array[ParentDatum.CONTIG_IDX]

    def get_position(self) -> int:
        return self.array[ParentDatum.POSITION_IDX]

    def get_ref_allele(self) -> str:
        return bases5_as_base_string(self.array[ParentDatum.REF_ALLELE_AS_BASE_5_IDX])

    def get_alt_allele(self) -> str:
        return bases5_as_base_string(self.array[ParentDatum.ALT_ALLELE_AS_BASE_5_IDX])

    def get_seq_error_log_lk(self) -> float:
        return self.array[ParentDatum.SEQ_ERROR_LOG_LK_IDX] / ParentDatum.FLOAT_TO_LONG_MULTIPLIER

    def get_normal_seq_error_log_lk(self) -> float:
        return self.array[ParentDatum.NORMAL_SEQ_ERROR_LOG_LK_IDX] / ParentDatum.FLOAT_TO_LONG_MULTIPLIER

    def get_ref_seq_1d(self) -> np.ndarray:
        start = ParentDatum.REF_SEQ_START_IDX
        ref_seq_length = self.array[ParentDatum.REF_SEQ_LENGTH_IDX]
        assert ref_seq_length > 0, "trying to get ref seq array when none exists"
        return self.array[start:start + ref_seq_length] / ParentDatum.FLOAT_TO_LONG_MULTIPLIER

    def get_info_1d(self) -> np.ndarray:
        start = ParentDatum.REF_SEQ_START_IDX + self.array[ParentDatum.REF_SEQ_LENGTH_IDX]
        info_length = self.array[ParentDatum.INFO_LENGTH_IDX]
        assert info_length > 0, "trying to get info array when none exists"
        return self.array[start:start + info_length] / ParentDatum.FLOAT_TO_LONG_MULTIPLIER

    # note: this potentially resizes the array and requires the leading info tensor size element to be modified
    # we do this in preprocessing when adding extra info to the info from GATK.
    # this method should not otherwise be used!!!
    def set_info_1d(self, new_info: np.ndarray):
        old_info_start = ParentDatum.REF_SEQ_START_IDX + self.array[ParentDatum.REF_SEQ_LENGTH_IDX]
        self.array = np.hstack((self.array[:old_info_start], new_info))
        self.array[ParentDatum.INFO_LENGTH_IDX] = len(new_info)

    def get_array_1d(self) -> np.ndarray:
        return self.array

    def get_nbytes(self) -> int:
        return self.array.nbytes

    def copy_without_ref_seq_and_info(self) -> ParentDatum:
        result = ParentDatum(self.array[:ParentDatum.NUM_SCALAR_ELEMENTS].copy())
        result.array[ParentDatum.REF_SEQ_LENGTH_IDX] = 0
        result.array[ParentDatum.INFO_LENGTH_IDX] = 0
        return result


class BaseDatum(ParentDatum):
    """
    :param ref_sequence_1d  1D uint8 tensor of bases centered at the alignment start of the variant in form eg ACTG -> [0,1,3,2]
    :param reads_2d   2D tensor, each row corresponding to one read; first all the ref reads, then all the alt reads
    :param info_array_1d  1D tensor of information about the variant as a whole
    :param label        an object of the Label enum artifact, non-artifact, unlabeled
    """
    def __init__(self, parent_datum_array: np.ndarray, reads_2d: np.ndarray):
        super().__init__(parent_datum_array)
        self.reads_2d = reads_2d
        # Note: if changing any of the data fields below, make sure to modify the size_in_bytes() method below accordingly!

        self.alt_count = self.get_alt_count()
        self.label = self.get_label()
        self.source = self.get_source()

        self.set_reads_dtype(np.float16)

    # gatk_info tensor comes from GATK and does not include one-hot encoding of variant type
    @classmethod
    def from_gatk(cls, ref_sequence_string: str, variant_type: Variation, ref_tensor: np.ndarray, alt_tensor: np.ndarray,
                 gatk_info_tensor: np.ndarray, label: Label, source: int, variant: Variant, counts_and_seq_lks: CountsAndSeqLks = None):
        read_tensor = np.vstack([ref_tensor, alt_tensor]) if ref_tensor is not None else alt_tensor
        alt_count = len(alt_tensor)
        str_info = get_str_info_array(ref_sequence_string, variant)
        info_tensor = np.hstack([gatk_info_tensor, str_info])
        result = cls(read_tensor, make_1d_sequence_tensor(ref_sequence_string), alt_count, info_tensor, variant_type, label, source, variant, counts_and_seq_lks)
        result.set_reads_dtype(np.float16)
        return result

    def set_reads_dtype(self, dtype):
        self.reads_2d = self.reads_2d.astype(dtype)

    def size_in_bytes(self):
        return self.reads_2d.nbytes + self.get_nbytes()

    def get_reads_2d(self) -> np.ndarray:
        return self.reads_2d

    def get_ref_reads_2d(self) -> np.ndarray:
        return self.reads_2d[:-self.get_alt_count()]

    def get_alt_reads_2d(self) -> np.ndarray:
        return self.reads_2d[-self.get_alt_count():]

    # returns two length-L 1D arrays of ref stacked on top of alt, with '4' in alt(ref) for deletions(insertions)
    def get_ref_and_alt_sequences(self):
        original_ref_array = self.get_ref_seq_1d() # gives an array eg ATTTCGG -> [0,3,3,3,1,2,2]
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
    read_tensors = np.vstack([datum.get_reads_2d() for datum in base_data])
    other_stuff = np.vstack([datum.get_array_1d() for datum in base_data])
    torch.save([read_tensors, other_stuff], file)


def load_list_of_base_data(file) -> List[BaseDatum]:
    """
    file is torch, output is converted back to numpy
    :param file:
    :return:
    """
    # these are vstacked -- see save method above
    read_tensors, other_stuffs = torch.load(file)

    result = []
    read_start_row = 0
    for parent_datum_array in other_stuffs:
        parent_datum = ParentDatum(parent_datum_array)
        read_count = parent_datum.get_ref_count() + parent_datum.get_alt_count()
        read_end_row = read_start_row+read_count

        base_datum = BaseDatum(parent_datum_array=parent_datum_array, reads_2d=read_tensors[read_start_row:read_end_row])
        read_start_row = read_end_row
        result.append(base_datum)

    return result


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
        # TODO: can we get rid of this potential bottleneck (might interact really badly with multiple workers)?
        self._original_list = data

        # num_classes = 5 for A, C, G, T, and deletion / insertion
        ref_alt = [torch.flatten(torch.permute(torch.nn.functional.one_hot(torch.from_numpy(np.vstack(item.get_ref_and_alt_sequences())).long(), num_classes=5), (0,2,1)), 0, 1) for item in data]    # list of 2D (2x5)xL
        # this is indexed by batch, length, channel (aka one-hot base encoding)
        ref_alt_bcl = torch.stack(ref_alt)

        self.ref_sequences_2d = ref_alt_bcl

        list_of_ref_tensors = [item.get_ref_reads_2d() for item in data]
        list_of_alt_tensors = [item.get_alt_reads_2d() for item in data]
        self.reads_2d = torch.from_numpy(np.vstack(list_of_ref_tensors + list_of_alt_tensors))
        self.info_2d = torch.from_numpy(np.vstack([base_datum.get_info_1d() for base_datum in data]))

        ref_counts = IntTensor([len(datum.reads_2d) - datum.alt_count for datum in data])
        alt_counts = IntTensor([datum.alt_count for datum in data])
        labels = IntTensor([1 if item.label == Label.ARTIFACT else 0 for item in data])
        is_labeled_mask = IntTensor([0 if item.label == Label.UNLABELED else 1 for item in data])
        sources = IntTensor([item.source for item in data])
        variant_types = IntTensor([datum.get_variant_type() for datum in data])
        self.int_tensor = torch.vstack((ref_counts, alt_counts, labels, is_labeled_mask, sources, variant_types))

        self._size = len(data)

    # pin memory for all tensors that are sent to the GPU
    def pin_memory(self):
        self.ref_sequences_2d = self.ref_sequences_2d.pin_memory()
        self.reads_2d = self.reads_2d.pin_memory()
        self.info_2d = self.info_2d.pin_memory()
        self.int_tensor = self.int_tensor.pin_memory()

        return self

    def copy_to(self, device, dtype):
        is_cuda = device.type == 'cuda'
        # For all non-tensor attributes, shallow copy is sufficient
        new_batch = copy.copy(self)
        new_batch.ref_sequences_2d = self.ref_sequences_2d.to(device=device, dtype=dtype, non_blocking=is_cuda)
        new_batch.reads_2d = self.reads_2d.to(device=device, dtype=dtype, non_blocking=is_cuda)
        new_batch.info_2d = self.info_2d.to(device=device, dtype=dtype, non_blocking=is_cuda)
        new_batch.int_tensor = self.int_tensor.to(device=device, non_blocking=is_cuda)
        return new_batch

    def original_list(self):
        return self._original_list

    def get_reads_2d(self) -> Tensor:
        return self.reads_2d

    def get_ref_counts(self) -> IntTensor:
        return self.int_tensor[0, :]

    def get_alt_counts(self) -> IntTensor:
        return self.int_tensor[1, :]

    # the original IntEnum format
    def get_labels(self):
        return self.int_tensor[2, :]

    def get_training_labels(self):
        int_enum_labels = self.get_labels()
        return 1.0 * (int_enum_labels == Label.ARTIFACT) + 0.5 * (int_enum_labels == Label.UNLABELED)

    def get_is_labeled_mask(self) -> IntTensor:
        return self.int_tensor[3, :]

    def get_sources(self) -> IntTensor:
        return self.int_tensor[4, :]

    def get_variant_types(self) -> IntTensor:
        return self.int_tensor[5, :]

    def get_info_2d(self) -> Tensor:
        return self.info_2d

    def get_ref_sequences_2d(self) -> Tensor:
        return self.ref_sequences_2d

    def size(self) -> int:
        return self._size


class ArtifactDatum:
    """
    """
    def __init__(self, base_datum: BaseDatum, representation: Tensor):
        # Note: if changing any of the data fields below, make sure to modify the size_in_bytes() method below accordingly!
        assert representation.dim() == 1
        self.representation = torch.clamp(representation, MIN_FLOAT_16, MAX_FLOAT_16)
        self.one_dimensional_data = base_datum.copy_without_ref_seq_and_info()
        self.set_dtype(np.float16)

    def set_dtype(self, dtype):
        self.representation = self.representation.to(torch.float16)
        self.one_dimensional_data.set_reads_dtype(dtype)

    def get_ref_count(self) -> int:
        return self.one_dimensional_data.get_ref_count()

    def get_alt_count(self) -> int:
        return self.one_dimensional_data.get_alt_count()

    def get_depth(self) -> int:
        return self.one_dimensional_data.get_counts_and_seq_lks().depth

    def get_variant_type(self) -> int:
        return self.one_dimensional_data.get_variant_type()

    def get_label(self) -> int:
        return self.one_dimensional_data.get_label()

    def get_source(self) -> int:
        return self.one_dimensional_data.get_source()

    def size_in_bytes(self):
        return self.representation.nbytes + self.one_dimensional_data.get_nbytes()

    def get_1d_data(self) -> OneDimensionalData:
        return self.one_dimensional_data

    def is_labeled(self):
        return self.get_label() != Label.UNLABELED


class ArtifactBatch:
    def __init__(self, data: List[ArtifactDatum]):
        self.representations_2d = torch.vstack([item.representation for item in data])
        self.other_stuff_array = torch.from_numpy(np.vstack([d.get_1d_data().to_np_array() for d in data]))
        self._size = len(data)

    def get_variants(self) -> List[Variant]:
        relevant_cols = self.other_stuff_array[:, OneDimensionalData.VARIANT_START_IDX:OneDimensionalData.VARIANT_END_IDX].numpy()
        return [Variant.from_np_array(var_array_1d) for var_array_1d in relevant_cols]

    def get_counts_and_seq_lks(self) -> List[CountsAndSeqLks]:
        relevant_cols = self.other_stuff_array[:, OneDimensionalData.COUNTS_AND_SEQ_LKS_START_IDX:].numpy()
        return [CountsAndSeqLks.from_np_array(var_array_1d) for var_array_1d in relevant_cols]

    # get the original IntEnum format (VARIANT = 0, ARTIFACT = 1, UNLABELED = 2) labels
    def get_labels(self) -> IntTensor:
        return self.other_stuff_array[:, OneDimensionalData.LABEL_IDX].int()

    # TODO: left off here
    # TODO: put in some breakpoints to double-check that this works
    # convert to the training format of 0.0 / 0.5 / 1.0 for variant / unlabeled / artifact
    # the 0.5 for unlabeled data is reasonable but should never actually be used due to the is_labeled mask
    def get_training_labels(self) -> FloatTensor:
        int_enum_labels = self.get_labels()
        return 1.0 * (int_enum_labels == Label.ARTIFACT) + 0.5 * (int_enum_labels == Label.UNLABELED)

    # TODO: put in some breakpoints to double-check that this works
    def get_is_labeled_mask(self):
        int_enum_labels = self.get_labels()
        return (int_enum_labels != Label.UNLABELED).int()

    def get_sources(self) -> IntTensor:
        return self.other_stuff_array[:, OneDimensionalData.SOURCE_IDX].int()

    def get_variant_types(self) -> IntTensor:
        result = self.other_stuff_array[:, OneDimensionalData.VAR_TYPE_IDX].int()
        return result

    def get_ref_counts(self) -> IntTensor:
        return self.other_stuff_array[:, OneDimensionalData.REF_COUNT_IDX].int()

    def get_alt_counts(self) -> IntTensor:
        return self.other_stuff_array[:, OneDimensionalData.ALT_COUNT_IDX].int()

    # pin memory for all tensors that are sent to the GPU
    def pin_memory(self):
        self.representations_2d = self.representations_2d.pin_memory()
        self.other_stuff_array = self.other_stuff_array.pin_memory()
        return self

    def copy_to(self, device, dtype):
        is_cuda = device.type == 'cuda'
        # For all non-tensor attributes, shallow copy is sufficient
        # note that variants_array and counts_and_seq_lks_array are not used in training and are never sent to GPU
        new_batch = copy.copy(self)
        new_batch.representations_2d = self.representations_2d.to(device=device, dtype=dtype, non_blocking=is_cuda)
        new_batch.other_stuff_array = self.other_stuff_array.to(device, dtype=dtype, non_blocking=is_cuda)

        return new_batch

    def get_representations_2d(self) -> Tensor:
        return self.representations_2d

    def size(self) -> int:
        return self._size

