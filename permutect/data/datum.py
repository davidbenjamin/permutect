from __future__ import annotations

import enum

import numpy as np
import torch

from permutect.utils.allele_utils import bases5_as_base_string
from permutect.utils.allele_utils import bases_as_base5_int
from permutect.utils.allele_utils import get_ref_and_alt_sequences
from permutect.utils.allele_utils import get_str_info_array
from permutect.utils.allele_utils import make_1d_sequence_tensor
from permutect.utils.allele_utils import trim_alleles_on_right
from permutect.utils.enums import Label
from permutect.utils.enums import Variation

# the range is -32,768 to 32,767
# this is sufficient for the count, length, and enum variables, as well as the floats (multiplied by something like 100
# and rounded to the nearest integer)
# haplotypes are represented as A = 0, C = 1, G = 2, T = 3 so 16 bits are easily enough (and we could compress further)
# the position needs 32 bits (to get up to 2 billion or so) so we give it two int16s
# the ref and alt alleles also need 32 bits to handle up to 13 bases
INTEGER_DTYPE = np.int16
LARGE_INTEGER_DTYPE = np.uint32
FLOAT_DTYPE = np.float16
RAW_READS_ARRAY_DTYPE = np.float16
COMPRESSED_READS_ARRAY_DTYPE = np.uint8
BIGGEST_UINT16 = 65535
BIGGEST_INT16 = 32767

# on disk and in datasets before creating batches, read tensors are stored with dtype np.uint8
# the leading columns are from binary quantities (either inherently binary or one-hot-encoded categorical) with np.packbits
# applied (this converts bool tensors into byte tensors, each byte holding eight bools)
# if the number of boolean bits is not a multiple of 8, it gets padded with zeros.  This isn't a problem.
NUMBER_OF_BYTES_IN_PACKED_READ = 7

DEFAULT_GPU_FLOAT = torch.float32
DEFAULT_CPU_FLOAT = torch.float32


def uint32_to_two_int16s(num: int):
    uint16_1, uint16_2 = num // BIGGEST_UINT16, num % BIGGEST_UINT16
    return uint16_1 - (BIGGEST_INT16 + 1), uint16_2 - (BIGGEST_INT16 + 1)


def uint32_from_two_int16s(int16_1, int16_2):
    shifted1, shifted2 = int(int16_1) + (BIGGEST_INT16 + 1), int(int16_2) + (BIGGEST_INT16 + 1)
    return BIGGEST_UINT16 * shifted1 + shifted2


class Data(enum.Enum):
    # int array elements
    REF_COUNT = (INTEGER_DTYPE, 0)
    ALT_COUNT = (INTEGER_DTYPE, 1)
    LABEL = (INTEGER_DTYPE, 2)
    VARIANT_TYPE = (INTEGER_DTYPE, 3)
    SOURCE = (INTEGER_DTYPE, 4)
    ORIGINAL_DEPTH = (INTEGER_DTYPE, 5)
    ORIGINAL_ALT_COUNT = (INTEGER_DTYPE, 6)
    ORIGINAL_NORMAL_DEPTH = (INTEGER_DTYPE, 7)
    ORIGINAL_NORMAL_ALT_COUNT = (INTEGER_DTYPE, 8)
    CONTIG = (INTEGER_DTYPE, 9)
    POSITION = (LARGE_INTEGER_DTYPE, 10)  # NOTE: uint32 takes TWO uint16s!
    REF_ALLELE_AS_BASE_5 = (LARGE_INTEGER_DTYPE, 12)  # NOTE: uint32 takes TWO uint16s!
    ALT_ALLELE_AS_BASE_5 = (LARGE_INTEGER_DTYPE, 14)  # NOTE: uint32 takes TWO uint16s!
    # after this, at the end of the int16 array comes the sub-array containing the ref sequence haplotype

    # float array elements
    # allele frequency, maf, normal maf, and cached artifact logit are not used until the posterior model
    # are are set to
    SEQ_ERROR_LOG_LK = (FLOAT_DTYPE, 0)
    NORMAL_SEQ_ERROR_LOG_LK = (FLOAT_DTYPE, 1)
    ALLELE_FREQUENCY = (FLOAT_DTYPE, 2)
    MAF = (FLOAT_DTYPE, 3)
    NORMAL_MAF = (FLOAT_DTYPE, 4)
    CACHED_ARTIFACT_LOGIT = (FLOAT_DTYPE, 5)  # +inf marks a known artifact, eg from a trusted upstream filter
    # TODO: left off here -- I added these constants to the enum but haven't yet initialized them in the constructor
    # TODO: nor have I cleaned up how they interact with the posterior datum
    # after this, at the end of the float16 array comes the sub-array containing the INFO vector

    def __init__(self, dtype: np.dtype, idx: int):
        self.dtype = dtype
        self.idx = idx


Data.NUM_SCALAR_INT_ELEMENTS = 16  # in Python 3.11+ can use enum.nonmember
Data.HAPLOTYPES_START_IDX = 16  # in Python 3.11+ can use enum.nonmember
Data.NUM_SCALAR_FLOAT_ELEMENTS = 6  # in Python 3.11+ can use enum.nonmember
Data.INFO_START_IDX = 6  # in Python 3.11+ can use enum.nonmember


class Datum:
    """
    TODO: need documentation
    """

    def __init__(
        self,
        int_array: np.ndarray,
        float_array: np.ndarray,
        reads_re: np.ndarray = None,
        compressed: bool = False,
    ):
        # note: this constructor does no checking eg of whether the arrays are consistent with their purported lengths
        # or of whether ref, alt alleles have been trimmed
        assert int_array.ndim == 1 and len(int_array) >= Data.NUM_SCALAR_INT_ELEMENTS
        self.int_array: np.ndarray = np.ndarray.astype(int_array, np.int16)
        assert float_array.ndim == 1 and len(float_array) >= Data.NUM_SCALAR_FLOAT_ELEMENTS
        self.float_array: np.ndarray = np.ndarray.astype(float_array, FLOAT_DTYPE)

        self.reads_re: np.ndarray = np.zeros((0, 0), dtype=RAW_READS_ARRAY_DTYPE) if reads_re is None else reads_re
        assert self.reads_re.dtype == (COMPRESSED_READS_ARRAY_DTYPE if compressed else RAW_READS_ARRAY_DTYPE)

    # this is what we get from GATK plain text data.  It must be normalized and processed before becoming the
    # data used by Permutect
    # gatk_info tensor comes from GATK and does not include one-hot encoding of variant type
    @classmethod
    def from_gatk(
        cls,
        label: Label,
        variant_type: Variation,
        source: int,
        original_depth: int,
        original_alt_count: int,
        original_normal_depth: int,
        original_normal_alt_count: int,
        contig: int,
        position: int,
        ref_allele: str,
        alt_allele: str,
        seq_error_log_lk: float,
        normal_seq_error_log_lk: float,
        ref_sequence_string: str,
        gatk_info_array: np.ndarray,
        ref_reads_array_re: np.ndarray,
        alt_reads_array_re: np.ndarray,
    ) -> Datum:
        # note: it is very important to trim here, as early as possible, because truncating to 13 or fewer bases
        # does not commute with trimming!!!  If we are not consistent about trimming first, dataset variants and
        # VCF variants might get inconsistent encodings!!!
        trimmed_ref, trimmed_alt = trim_alleles_on_right(ref_allele, alt_allele)
        str_info = get_str_info_array(ref_sequence_string, trimmed_ref, trimmed_alt)
        info_array = np.hstack([gatk_info_array, str_info])
        ref_seq_array = make_1d_sequence_tensor(ref_sequence_string)

        ref_hap, alt_hap = get_ref_and_alt_sequences(ref_seq_array, trimmed_ref, trimmed_alt)
        assert len(ref_hap) == len(ref_seq_array) and len(alt_hap) == len(ref_seq_array)
        haplotypes = np.hstack((ref_hap, alt_hap))

        haplotypes_length, info_length = len(haplotypes), len(info_array)
        zeroed_int_array = np.zeros(Data.NUM_SCALAR_INT_ELEMENTS + haplotypes_length, dtype=INTEGER_DTYPE)
        zeroed_float_array = np.zeros(Data.NUM_SCALAR_FLOAT_ELEMENTS + info_length, dtype=FLOAT_DTYPE)
        reads_array_re = (
            np.vstack([ref_reads_array_re, alt_reads_array_re])
            if ref_reads_array_re is not None
            else alt_reads_array_re
        )

        result = cls(int_array=zeroed_int_array, float_array=zeroed_float_array, reads_re=reads_array_re)
        result.set(Data.REF_COUNT, 0 if ref_reads_array_re is None else len(ref_reads_array_re))
        result.set(Data.ALT_COUNT, 0 if alt_reads_array_re is None else len(alt_reads_array_re))
        result.set(Data.LABEL, label)
        result.set(Data.VARIANT_TYPE, variant_type)
        result.set(Data.SOURCE, source)
        result.set(Data.ORIGINAL_DEPTH, original_depth)
        result.set(Data.ORIGINAL_ALT_COUNT, original_alt_count)
        result.set(Data.ORIGINAL_NORMAL_DEPTH, original_normal_depth)
        result.set(Data.ORIGINAL_NORMAL_ALT_COUNT, original_normal_alt_count)
        result.set(Data.CONTIG, contig)
        result.set(Data.POSITION, position)
        result.set(Data.REF_ALLELE_AS_BASE_5, bases_as_base5_int(trimmed_ref))
        result.set(Data.ALT_ALLELE_AS_BASE_5, bases_as_base5_int(trimmed_alt))
        result.int_array[Data.HAPLOTYPES_START_IDX :] = haplotypes

        result.set(Data.SEQ_ERROR_LOG_LK, seq_error_log_lk)  # this is -log10ToLog(TLOD) - log(tumorDepth + 1)
        result.set(
            Data.NORMAL_SEQ_ERROR_LOG_LK, normal_seq_error_log_lk
        )  # this is -log10ToLog(NALOD) - log(normalDepth + 1)

        result.set(Data.ALLELE_FREQUENCY, np.nan)
        result.set(Data.MAF, np.nan)
        result.set(Data.NORMAL_MAF, np.nan)
        result.set(Data.CACHED_ARTIFACT_LOGIT, np.nan)

        result.float_array[Data.INFO_START_IDX :] = info_array
        return result

    def get(self, data_field: Data):
        index = data_field.idx
        if data_field.dtype == INTEGER_DTYPE:
            return self.int_array[index]
        elif data_field.dtype == LARGE_INTEGER_DTYPE:
            return uint32_from_two_int16s(self.int_array[index], self.int_array[index + 1])
        elif data_field.dtype == FLOAT_DTYPE:
            return self.float_array[index]
        else:
            assert False, "Unsupported data type"

    def set(self, data_field: Data, value):
        index = data_field.idx
        if data_field.dtype == INTEGER_DTYPE:
            self.int_array[index] = value
        elif data_field.dtype == LARGE_INTEGER_DTYPE:
            int_1, int_2 = uint32_to_two_int16s(value)
            self.int_array[index] = int_1
            self.int_array[index + 1] = int_2
        elif data_field.dtype == FLOAT_DTYPE:
            self.float_array[index] = value
        else:
            assert False, "Unsupported data type"

    def get_read_count(self) -> int:
        return self.get(Data.ALT_COUNT) + self.get(Data.REF_COUNT)

    def is_labeled(self):
        return self.get(Data.LABEL) != Label.UNLABELED

    def get_ref_allele(self) -> str:
        return bases5_as_base_string(self.get(Data.REF_ALLELE_AS_BASE_5))

    def get_alt_allele(self) -> str:
        return bases5_as_base_string(self.get(Data.ALT_ALLELE_AS_BASE_5))

    def get_haplotypes_array_size(self):
        return len(self.int_array) - Data.NUM_SCALAR_INT_ELEMENTS

    def get_haplotypes_1d(self) -> np.ndarray:
        # 1D array of integer array reference and alt haplotypes concatenated -- A, C, G, T, deletion = 0, 1, 2, 3, 4
        assert len(self.int_array) > Data.NUM_SCALAR_INT_ELEMENTS, "trying to get ref seq array when none exists"
        return self.int_array[Data.HAPLOTYPES_START_IDX :]

    def get_info_1d(self) -> np.ndarray:
        assert len(self.float_array) > Data.NUM_SCALAR_FLOAT_ELEMENTS, "trying to get info array when none exists"
        return self.float_array[Data.INFO_START_IDX :]

    # note: this potentially resizes the array
    # we do this in preprocessing when adding extra info to the info from GATK.
    # this method should not otherwise be used!!!
    def set_info_1d(self, new_info: np.ndarray):
        self.float_array = np.hstack((self.float_array[: Data.INFO_START_IDX], new_info))

    def set_haplotypes_1d(self, new_haplotypes: np.ndarray):
        self.int_array = np.hstack((self.int_array[: Data.HAPLOTYPES_START_IDX], new_haplotypes))

    def get_int_array(self) -> np.ndarray:
        return self.int_array

    def get_float_array(self) -> np.ndarray:
        return self.float_array

    def get_reads_array_re(self) -> np.ndarray:
        return self.reads_re

    def get_ref_reads_re(self) -> np.ndarray:
        return self.reads_re[: -self.get(Data.ALT_COUNT)]

    def get_alt_reads_re(self) -> np.ndarray:
        return self.reads_re[-self.get(Data.ALT_COUNT) :]

    # only applies after normalization and compression
    def get_compressed_ref_reads_re(self) -> np.ndarray:
        assert self.reads_re.dtype == COMPRESSED_READS_ARRAY_DTYPE
        return self.reads_re[: -self.get(Data.ALT_COUNT)]

    # only applies after normalization and compression
    def get_compressed_alt_reads_re(self) -> np.ndarray:
        assert self.reads_re.dtype == COMPRESSED_READS_ARRAY_DTYPE
        return self.reads_re[-self.get(Data.ALT_COUNT) :]

    def get_compressed_reads_re(self) -> np.ndarray:
        assert self.reads_re.dtype == COMPRESSED_READS_ARRAY_DTYPE
        return self.reads_re

    # this is the number of read features in a PyTorch float tensor, after unpacking the compressed binaries
    def num_read_features(self) -> int:
        if self.reads_re.dtype == RAW_READS_ARRAY_DTYPE or self.reads_re.shape[1] == 0:
            return self.reads_re.shape[1]
        elif self.reads_re.dtype == COMPRESSED_READS_ARRAY_DTYPE:
            num_read_uint8s = self.reads_re.shape[1]
            num_nonbinary_features = num_read_uint8s - NUMBER_OF_BYTES_IN_PACKED_READ
            # 8 bits per byte.  In general the last few features will be extraneous padded zeros, but that's okay.
            # the nonbinary features are converted to float, but it's one-to-one so no multiplicative factor
            return 8 * NUMBER_OF_BYTES_IN_PACKED_READ + num_nonbinary_features
        else:
            assert False, "Unsupported data type"

    def get_nbytes(self) -> int:
        return self.int_array.nbytes + self.float_array.nbytes + self.reads_re.nbytes

    def copy_with_downsampled_reads(self, ref_downsample: int, alt_downsample: int) -> Datum:
        old_alt_count = self.get(Data.ALT_COUNT)
        old_ref_count = len(self.reads_re) - old_alt_count
        new_ref_count = min(old_ref_count, ref_downsample)
        new_alt_count = min(self.get(Data.ALT_COUNT), alt_downsample)

        if new_ref_count == old_ref_count and new_alt_count == old_alt_count:
            return self
        else:
            # new reads are random selection of ref reads vstacked on top of all the alts
            random_ref_read_indices = torch.randperm(old_ref_count)[:new_ref_count]
            random_alt_read_indices = old_ref_count + torch.randperm(old_alt_count)[:new_alt_count]
            new_reads = np.vstack((self.reads_re[random_ref_read_indices], self.reads_re[random_alt_read_indices]))
            result = Datum(
                int_array=self.int_array.copy(),
                float_array=self.float_array.copy(),
                reads_re=new_reads,
            )
            result.set(Data.REF_COUNT, new_ref_count)
            result.set(Data.ALT_COUNT, new_alt_count)
            return result
