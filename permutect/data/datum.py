from __future__ import annotations

import numpy as np
import torch

from permutect.utils.allele_utils import bases_as_base5_int, bases5_as_base_string, get_ref_and_alt_sequences
from permutect.utils.enums import Label, Variation

# the range is -32,768 to 32,767
# this is sufficient for the count, length, and enum variables, as well as the floats (multiplied by something like 100
# and rounded to the nearest integer)
# haplotypes are represented as A = 0, C = 1, G = 2, T = 3 so 16 bits are easily enough (and we could compress further)
# the position needs 32 bits (to get up to 2 billion or so) so we give it two int16s
# the ref and alt alleles also need 32 bits to handle up to 13 bases
DATUM_ARRAY_DTYPE = np.int16
BIGGEST_UINT16 = 65535
BIGGEST_INT16 = 32767
FLOAT_TO_LONG_MULTIPLIER = 30

MAX_FLOAT = BIGGEST_INT16 / FLOAT_TO_LONG_MULTIPLIER


def float_to_clipped_int16(float_number: float) -> int:
    unbounded_int = round(float_number * FLOAT_TO_LONG_MULTIPLIER)
    return max(min(unbounded_int, BIGGEST_INT16), -BIGGEST_INT16)


def int16_to_float(int16_number: int) -> float:
    return int16_number / FLOAT_TO_LONG_MULTIPLIER


def uint32_to_two_int16s(num: int):
    uint16_1, uint16_2 = num // BIGGEST_UINT16, num % BIGGEST_UINT16
    return uint16_1 - (BIGGEST_INT16 + 1), uint16_2 - (BIGGEST_INT16 + 1)


def uint32_from_two_int16s(int16_1: int, int16_2: int) -> int:
    shifted1, shifted2 = int16_1 + (BIGGEST_INT16 + 1), int16_2 + (BIGGEST_INT16 + 1)
    return BIGGEST_UINT16 * shifted1 + shifted2


class Datum:
    """
    contains data that apply to a candidate mutation as a whole i.e. not the read sets.  These are organized into a single
    LongTensor, containing some quantities that are inherently integral and some that are cast as longs by multiplying
    with a large number and rounding.
    """

    # indices of inherently integral quantities
    REF_COUNT_IDX = 0               # potentially downsampled -- the actual size of the ref reads tensor
    ALT_COUNT_IDX = 1               # potentially downsampled -- the actual size of the alt reads tensor
    HAPLOTYPES_LENGTH_IDX = 2       # length of the sub-array encoding the reference and alt haplotype sequences
    INFO_LENGTH_IDX = 3             # length of the sub-array encoding the info vector
    LABEL_IDX = 4                   # the IntEnum label
    VARIANT_TYPE_IDX = 5            # the IntEnum variant type
    SOURCE_IDX = 6                  # the integer encoding the source

    ORIGINAL_DEPTH_IDX = 7          # the original depth of the sequencing data before downsampling
    ORIGINAL_ALT_COUNT_IDX = 8      # the original alt count of the sequencing data before downsampling
    ORIGINAL_NORMAL_DEPTH_IDX = 9   # the original matched normal sample depth of the sequencing data before downsampling
    ORIGINAL_NORMAL_ALT_COUNT_IDX = 10     # the original matched normal sample alt count of the sequencing data before downsampling

    CONTIG_IDX = 11                 # the index of the contig/chromosome

    # NOTE: the next three elements all require TWO int16s i.e. 32 bits to represent!!!!
    POSITION_IDX = 12               # the position of the variant start within the contig
    REF_ALLELE_AS_BASE_5_IDX = 14   # the reference allele encoded as a single base 5 integer
    ALT_ALLELE_AS_BASE_5_IDX = 16   # the reference allele encoded as a single base 5 integer

    # FloatTensor indices
    SEQ_ERROR_LOG_LK_IDX = 18
    NORMAL_SEQ_ERROR_LOG_LK_IDX = 19

    NUM_SCALAR_ELEMENTS = NORMAL_SEQ_ERROR_LOG_LK_IDX + 1
    HAPLOTYPES_START_IDX = 20

    # after these come the variable-length sub-arrays (not within a single dataset, but in principle variable length for
    # different versions of Permutect or different sequencing) for the reference sequence context and the info tensor

    def __init__(self, array: np.ndarray):
        # note: this constructor does no checking eg of whether the arrays are consistent with their purported lengths
        # or of whether ref, alt alleles have been trimmed
        assert array.ndim == 1 and len(array) >= Datum.NUM_SCALAR_ELEMENTS
        self.array: np.ndarray = np.ndarray.astype(array, DATUM_ARRAY_DTYPE)

    @classmethod
    def make_datum_without_reads(cls, label: Label, variant_type: Variation, source: int,
        original_depth: int, original_alt_count: int, original_normal_depth: int, original_normal_alt_count: int,
        contig: int, position: int, ref_allele: str, alt_allele: str,
        seq_error_log_lk: float, normal_seq_error_log_lk: float, ref_seq_array: np.ndarray, info_array: np.ndarray) -> Datum:
        """
        We are careful about our float to long conversions here and in the getters!
        """
        ref_hap, alt_hap = get_ref_and_alt_sequences(ref_seq_array, ref_allele, alt_allele)
        assert len(ref_hap) == len(ref_seq_array) and len(alt_hap) == len(ref_seq_array)
        haplotypes = np.hstack((ref_hap, alt_hap))

        haplotypes_length, info_length = len(haplotypes), len(info_array)
        result = cls(np.zeros(Datum.NUM_SCALAR_ELEMENTS + haplotypes_length + info_length, dtype=DATUM_ARRAY_DTYPE))
        # ref count and alt count remain zero
        result.array[Datum.HAPLOTYPES_LENGTH_IDX] = haplotypes_length
        result.array[Datum.INFO_LENGTH_IDX] = info_length

        result.array[Datum.LABEL_IDX] = label
        result.array[Datum.VARIANT_TYPE_IDX] = variant_type
        result.array[Datum.SOURCE_IDX] = source

        result.array[Datum.ORIGINAL_DEPTH_IDX] = original_depth
        result.array[Datum.ORIGINAL_ALT_COUNT_IDX] = original_alt_count
        result.array[Datum.ORIGINAL_NORMAL_DEPTH_IDX] = original_normal_depth
        result.array[Datum.ORIGINAL_NORMAL_ALT_COUNT_IDX] = original_normal_alt_count

        result.array[Datum.CONTIG_IDX] = contig

        result.store_uint32_as_two_int16s(position, Datum.POSITION_IDX)
        result.store_uint32_as_two_int16s(bases_as_base5_int(ref_allele), Datum.REF_ALLELE_AS_BASE_5_IDX)
        result.store_uint32_as_two_int16s(bases_as_base5_int(alt_allele), Datum.ALT_ALLELE_AS_BASE_5_IDX)

        result.store_float_as_int16(seq_error_log_lk, Datum.SEQ_ERROR_LOG_LK_IDX)
        result.store_float_as_int16(normal_seq_error_log_lk, Datum.NORMAL_SEQ_ERROR_LOG_LK_IDX)

        haplotypes_start = Datum.HAPLOTYPES_START_IDX
        haplotypes_end = haplotypes_start + haplotypes_length
        info_end = haplotypes_end + info_length
        result.array[haplotypes_start:haplotypes_end] = haplotypes  # haplotypes array is uint8
        result.array[haplotypes_end:info_end] = np.ndarray.astype(info_array * FLOAT_TO_LONG_MULTIPLIER, DATUM_ARRAY_DTYPE)

        return result

    def store_uint32_as_two_int16s(self, uint32_number, start_index):
        int16_1, int16_2 = uint32_to_two_int16s(uint32_number)
        self.array[start_index] = int16_1
        self.array[start_index + 1] = int16_2

    def get_uint32_from_two_int16s(self, start_index):
        return uint32_from_two_int16s(self.array[start_index], self.array[start_index + 1])

    def store_float_as_int16(self, float_number, index):
        self.array[index] = float_to_clipped_int16(float_number)

    def get_float_from_int16(self, index):
        return int16_to_float(self.array[index])

    def get_ref_count(self) -> int:
        return self.array[Datum.REF_COUNT_IDX]

    def get_alt_count(self) -> int:
        return self.array[Datum.ALT_COUNT_IDX]

    def get_haplotypes_array_length(self) -> int:
        return self.array[Datum.HAPLOTYPES_LENGTH_IDX]

    def get_info_array_length(self) -> int:
        return self.array[Datum.INFO_LENGTH_IDX]

    def get_label(self) -> int:
        return self.array[Datum.LABEL_IDX]

    def is_labeled(self):
        return self.get_label() != Label.UNLABELED

    def set_label(self, label: Label):
        self.array[Datum.LABEL_IDX] = label

    def get_variant_type(self) -> int:
        return self.array[Datum.VARIANT_TYPE_IDX]

    def get_source(self) -> int:
        return self.array[Datum.SOURCE_IDX]

    def set_source(self, source: int):
        self.array[Datum.SOURCE_IDX] = source

    def get_original_depth(self) -> int:
        return self.array[Datum.ORIGINAL_DEPTH_IDX]

    def get_original_alt_count(self) -> int:
        return self.array[Datum.ORIGINAL_ALT_COUNT_IDX]

    def get_original_normal_depth(self) -> int:
        return self.array[Datum.ORIGINAL_NORMAL_DEPTH_IDX]

    def get_original_normal_alt_count(self) -> int:
        return self.array[Datum.ORIGINAL_NORMAL_ALT_COUNT_IDX]

    def get_contig(self) -> int:
        return self.array[Datum.CONTIG_IDX]

    def get_position(self) -> int:
        return self.get_uint32_from_two_int16s(Datum.POSITION_IDX)

    def get_ref_allele(self) -> str:
        return bases5_as_base_string(self.get_uint32_from_two_int16s(Datum.REF_ALLELE_AS_BASE_5_IDX))

    def get_alt_allele(self) -> str:
        return bases5_as_base_string(self.get_uint32_from_two_int16s(Datum.ALT_ALLELE_AS_BASE_5_IDX))

    def get_seq_error_log_lk(self) -> float:
        return self.get_float_from_int16(Datum.SEQ_ERROR_LOG_LK_IDX)

    def get_normal_seq_error_log_lk(self) -> float:
        return self.get_float_from_int16(Datum.NORMAL_SEQ_ERROR_LOG_LK_IDX)

    def get_haplotypes_1d(self) -> np.ndarray:
        # 1D array of integer array reference and alt haplotypes concatenated -- A, C, G, T, deletion = 0, 1, 2, 3, 4
        start = Datum.HAPLOTYPES_START_IDX
        haplotypes_length = self.array[Datum.HAPLOTYPES_LENGTH_IDX]
        assert haplotypes_length > 0, "trying to get ref seq array when none exists"
        return self.array[start:start + haplotypes_length]

    def get_info_1d(self) -> np.ndarray:
        start = Datum.HAPLOTYPES_START_IDX + self.array[Datum.HAPLOTYPES_LENGTH_IDX]
        info_length = self.array[Datum.INFO_LENGTH_IDX]
        assert info_length > 0, "trying to get info array when none exists"
        return self.array[start:start + info_length] / FLOAT_TO_LONG_MULTIPLIER

    # note: this potentially resizes the array and requires the leading info tensor size element to be modified
    # we do this in preprocessing when adding extra info to the info from GATK.
    # this method should not otherwise be used!!!
    def set_info_1d(self, new_info: np.ndarray):
        new_info_as_long = np.ndarray.astype(new_info * FLOAT_TO_LONG_MULTIPLIER, DATUM_ARRAY_DTYPE)
        old_info_start = Datum.HAPLOTYPES_START_IDX + self.array[Datum.HAPLOTYPES_LENGTH_IDX]
        self.array = np.hstack((self.array[:old_info_start], new_info_as_long))
        self.array[Datum.INFO_LENGTH_IDX] = len(new_info)

    def get_array_1d(self) -> np.ndarray:
        return self.array

    def get_nbytes(self) -> int:
        return self.array.nbytes

    @classmethod
    def copy_data_without_haplotypes_and_info(cls, data_array: np.ndarray) -> np.ndarray:
        result = data_array[:Datum.NUM_SCALAR_ELEMENTS].copy()
        result[Datum.HAPLOTYPES_LENGTH_IDX] = 0
        result[Datum.INFO_LENGTH_IDX] = 0
        return result


DEFAULT_NUMPY_FLOAT = np.float16
DEFAULT_GPU_FLOAT = torch.float32
DEFAULT_CPU_FLOAT = torch.float32
MAX_FLOAT_16 = torch.finfo(torch.float16).max
MIN_FLOAT_16 = torch.finfo(torch.float16).min