from __future__ import annotations

import numpy as np
import torch

from permutect.utils.allele_utils import bases_as_base5_int, bases5_as_base_string
from permutect.utils.enums import Label, Variation


class Datum:
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
    ORIGINAL_NORMAL_ALT_COUNT_IDX = 10     # the original matched normal sample alt count of the sequencing data before downsampling

    CONTIG_IDX = 11                 # the index of the contig/chromosome
    POSITION_IDX = 12               # the position of the variant start within the contig
    REF_ALLELE_AS_BASE_5_IDX = 13   # the reference allele encoded as a single base 5 integer
    ALT_ALLELE_AS_BASE_5_IDX = 14   # the reference allele encoded as a single base 5 integer

    # FloatTensor indices
    SEQ_ERROR_LOG_LK_IDX = 15
    NORMAL_SEQ_ERROR_LOG_LK_IDX = 16

    NUM_SCALAR_ELEMENTS = NORMAL_SEQ_ERROR_LOG_LK_IDX + 1
    REF_SEQ_START_IDX = 17

    # after these come the variable-length sub-arrays (not within a single dataset, but in principle variable length for
    # different versions of Permutect or different sequencing) for the reference sequence context and the info tensor

    def __init__(self, array: np.ndarray):
        # note: this constructor does no checking eg of whether the arrays are consistent with their purported lengths
        # or of whether ref, alt alleles have been trimmed
        assert array.ndim == 1 and len(array) >= Datum.NUM_SCALAR_ELEMENTS
        self.array: np.ndarray = np.int64(array)

    @classmethod
    def make_datum_without_reads(cls, label: Label, variant_type: Variation, source: int,
        original_depth: int, original_alt_count: int, original_normal_depth: int, original_normal_alt_count: int,
        contig: int, position: int, ref_allele: str, alt_allele: str,
        seq_error_log_lk: float, normal_seq_error_log_lk: float, ref_seq_array: np.ndarray, info_array: np.ndarray) -> Datum:
        """
        We are careful about our float to long conversions here and in the getters!
        """
        ref_seq_length, info_length = len(ref_seq_array), len(info_array)
        result = cls(np.zeros(Datum.NUM_SCALAR_ELEMENTS + ref_seq_length + info_length, dtype=np.int64))
        # ref count and alt count remain zero
        result.array[Datum.REF_SEQ_LENGTH_IDX] = ref_seq_length
        result.array[Datum.INFO_LENGTH_IDX] = info_length

        result.array[Datum.LABEL_IDX] = label
        result.array[Datum.VARIANT_TYPE_IDX] = variant_type
        result.array[Datum.SOURCE_IDX] = source

        result.array[Datum.ORIGINAL_DEPTH_IDX] = original_depth
        result.array[Datum.ORIGINAL_ALT_COUNT_IDX] = original_alt_count
        result.array[Datum.ORIGINAL_NORMAL_DEPTH_IDX] = original_normal_depth
        result.array[Datum.ORIGINAL_NORMAL_ALT_COUNT_IDX] = original_normal_alt_count

        result.array[Datum.CONTIG_IDX] = contig
        result.array[Datum.POSITION_IDX] = position
        result.array[Datum.REF_ALLELE_AS_BASE_5_IDX] = bases_as_base5_int(ref_allele)
        result.array[Datum.ALT_ALLELE_AS_BASE_5_IDX] = bases_as_base5_int(alt_allele)

        result.array[Datum.SEQ_ERROR_LOG_LK_IDX] = round(seq_error_log_lk * Datum.FLOAT_TO_LONG_MULTIPLIER)
        result.array[Datum.NORMAL_SEQ_ERROR_LOG_LK_IDX] = round(normal_seq_error_log_lk * Datum.FLOAT_TO_LONG_MULTIPLIER)

        ref_seq_start = Datum.REF_SEQ_START_IDX
        ref_seq_end = ref_seq_start + ref_seq_length
        info_end = ref_seq_end + info_length
        result.array[ref_seq_start:ref_seq_end] = ref_seq_array # ref seq array is uint8
        result.array[ref_seq_end:info_end] = np.int64(info_array * Datum.FLOAT_TO_LONG_MULTIPLIER)

        return result

    def get_ref_count(self) -> int:
        return self.array[Datum.REF_COUNT_IDX]

    def get_alt_count(self) -> int:
        return self.array[Datum.ALT_COUNT_IDX]

    def get_ref_seq_array_length(self) -> int:
        return self.array[Datum.REF_SEQ_LENGTH_IDX]

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
        return self.array[Datum.POSITION_IDX]

    def get_ref_allele(self) -> str:
        return bases5_as_base_string(self.array[Datum.REF_ALLELE_AS_BASE_5_IDX])

    def get_alt_allele(self) -> str:
        return bases5_as_base_string(self.array[Datum.ALT_ALLELE_AS_BASE_5_IDX])

    def get_seq_error_log_lk(self) -> float:
        return self.array[Datum.SEQ_ERROR_LOG_LK_IDX] / Datum.FLOAT_TO_LONG_MULTIPLIER

    def get_normal_seq_error_log_lk(self) -> float:
        return self.array[Datum.NORMAL_SEQ_ERROR_LOG_LK_IDX] / Datum.FLOAT_TO_LONG_MULTIPLIER

    def get_ref_seq_1d(self) -> np.ndarray:
        start = Datum.REF_SEQ_START_IDX
        ref_seq_length = self.array[Datum.REF_SEQ_LENGTH_IDX]
        assert ref_seq_length > 0, "trying to get ref seq array when none exists"
        return self.array[start:start + ref_seq_length]

    def get_info_1d(self) -> np.ndarray:
        start = Datum.REF_SEQ_START_IDX + self.array[Datum.REF_SEQ_LENGTH_IDX]
        info_length = self.array[Datum.INFO_LENGTH_IDX]
        assert info_length > 0, "trying to get info array when none exists"
        return self.array[start:start + info_length] / Datum.FLOAT_TO_LONG_MULTIPLIER

    # note: this potentially resizes the array and requires the leading info tensor size element to be modified
    # we do this in preprocessing when adding extra info to the info from GATK.
    # this method should not otherwise be used!!!
    def set_info_1d(self, new_info: np.ndarray):
        new_info_as_long = np.int64(new_info * Datum.FLOAT_TO_LONG_MULTIPLIER)
        old_info_start = Datum.REF_SEQ_START_IDX + self.array[Datum.REF_SEQ_LENGTH_IDX]
        self.array = np.hstack((self.array[:old_info_start], new_info_as_long))
        self.array[Datum.INFO_LENGTH_IDX] = len(new_info)

    def get_array_1d(self) -> np.ndarray:
        return self.array

    def get_nbytes(self) -> int:
        return self.array.nbytes

    def copy_without_ref_seq_and_info(self) -> Datum:
        result = Datum(self.array[:Datum.NUM_SCALAR_ELEMENTS].copy())
        result.array[Datum.REF_SEQ_LENGTH_IDX] = 0
        result.array[Datum.INFO_LENGTH_IDX] = 0
        return result


DEFAULT_NUMPY_FLOAT = np.float16
DEFAULT_GPU_FLOAT = torch.float32
DEFAULT_CPU_FLOAT = torch.float32
MAX_FLOAT_16 = torch.finfo(torch.float16).max
MIN_FLOAT_16 = torch.finfo(torch.float16).min