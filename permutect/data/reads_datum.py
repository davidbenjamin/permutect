from __future__ import annotations
from typing import List

import numpy as np
import torch

from permutect.data.datum import Datum
from permutect.utils.allele_utils import trim_alleles_on_right, get_str_info_array, make_1d_sequence_tensor
from permutect.utils.enums import Variation, Label


# base strings longer than this when encoding data


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


# this is what we get from GATK plain tect data.  It must be normalized and processed before becoming the
# data used by Permutect
class RawUnnormalizedReadsDatum(Datum):
    def __init__(self, datum_array: np.ndarray, reads_re: np.ndarray):
        super().__init__(datum_array)
        self.reads_re = reads_re
        self.set_reads_dtype(np.float16)

    # gatk_info tensor comes from GATK and does not include one-hot encoding of variant type
    @classmethod
    def from_gatk(cls, label: Label, variant_type: Variation, source: int,
            original_depth: int, original_alt_count: int, original_normal_depth: int, original_normal_alt_count: int,
            contig: int, position: int, ref_allele: str, alt_allele: str,
            seq_error_log_lk: float, normal_seq_error_log_lk: float,
            ref_sequence_string: str, gatk_info_array: np.ndarray,
            ref_tensor: np.ndarray, alt_tensor: np.ndarray) -> RawUnnormalizedReadsDatum:
        # note: it is very important to trim here, as early as possible, because truncating to 13 or fewer bases
        # does not commute with trimming!!!  If we are not consistent about trimming first, dataset variants and
        # VCF variants might get inconsistent encodings!!!
        trimmed_ref, trimmed_alt = trim_alleles_on_right(ref_allele, alt_allele)
        str_info = get_str_info_array(ref_sequence_string, trimmed_ref, trimmed_alt)
        info_array = np.hstack([gatk_info_array, str_info])
        ref_seq_array = make_1d_sequence_tensor(ref_sequence_string)
        read_tensor = np.vstack([ref_tensor, alt_tensor]) if ref_tensor is not None else alt_tensor

        datum = Datum.make_datum_without_reads(label=label, variant_type=variant_type, source=source,
            original_depth=original_depth, original_alt_count=original_alt_count, original_normal_depth=original_normal_depth,
            original_normal_alt_count=original_normal_alt_count, contig=contig, position=position,
            ref_allele=trimmed_ref, alt_allele=trimmed_alt, seq_error_log_lk=seq_error_log_lk,
            normal_seq_error_log_lk=normal_seq_error_log_lk, ref_seq_array=ref_seq_array, info_array=info_array)
        # ref and alt counts need to be set manually.  Everything else is handled in the ParentDatum constructor
        datum.array[Datum.REF_COUNT_IDX] = 0 if ref_tensor is None else len(ref_tensor)
        datum.array[Datum.ALT_COUNT_IDX] = 0 if alt_tensor is None else len(alt_tensor)

        result = cls(datum_array=datum.get_array_1d(), reads_re=read_tensor)
        result.set_reads_dtype(np.float16)
        return result

    def copy_with_downsampled_reads(self, ref_downsample: int, alt_downsample: int) -> ReadsDatum:
        old_ref_count, old_alt_count = len(self.reads_re) - self.get_alt_count(), self.get_alt_count()
        new_ref_count = min(old_ref_count, ref_downsample)
        new_alt_count = min(self.get_alt_count(), alt_downsample)

        if new_ref_count == old_ref_count and new_alt_count == old_alt_count:
            return self
        else:
            new_data_array = self.array.copy()
            new_data_array[Datum.REF_COUNT_IDX] = new_ref_count
            new_data_array[Datum.ALT_COUNT_IDX] = new_alt_count

            # new reads are random selection of ref reads vstacked on top of all the alts
            random_ref_read_indices = torch.randperm(old_ref_count)[:new_ref_count]
            random_alt_read_indices = old_ref_count + torch.randperm(old_alt_count)[:new_alt_count]
            new_reads = np.vstack((self.reads_re[random_ref_read_indices], self.reads_re[random_alt_read_indices]))
            return ReadsDatum(new_data_array, new_reads)

    # TODO: used in both, but won't be needed for the compressed constructor
    def set_reads_dtype(self, dtype):
        self.reads_re = self.reads_re.astype(dtype)

    # TODO: used before normalization, and after normalization for pruning and editing datasets
    def size_in_bytes(self):
        return self.reads_re.nbytes + self.get_nbytes()

    # TODO: used only after normalization, but will have to be modified
    def get_reads_re(self) -> np.ndarray:
        return self.reads_re

    # TODO: used both before and after normalization (when making batches)
    def get_ref_reads_re(self) -> np.ndarray:
        return self.reads_re[:-self.get_alt_count()]

    # TODO: used both before and after normalization (when making batches)
    def get_alt_reads_re(self) -> np.ndarray:
        return self.reads_re[-self.get_alt_count():]

    # TODO: used only after normalization
    @classmethod
    def save_list(cls, base_data: List[ReadsDatum], file):
        read_tensors = np.vstack([datum.get_reads_re() for datum in base_data])
        other_stuff = np.vstack([datum.get_array_1d() for datum in base_data])
        torch.save([read_tensors, other_stuff], file, pickle_protocol=4)

    # TODO: used only after normalization
    @classmethod
    def load_list(cls, file) -> List[ReadsDatum]:
        # these are vstacked -- see save method above
        read_tensors, datum_arrays = torch.load(file)

        result = []
        read_start_row = 0
        for datum_array in datum_arrays:
            datum = Datum(datum_array)
            read_count = datum.get_ref_count() + datum.get_alt_count()
            read_end_row = read_start_row + read_count

            reads_datum = ReadsDatum(datum_array=datum_array, reads_re=read_tensors[read_start_row:read_end_row])
            read_start_row = read_end_row
            result.append(reads_datum)

        return result


class ReadsDatum(Datum):
    def __init__(self, datum_array: np.ndarray, reads_re: np.ndarray):
        super().__init__(datum_array)
        self.reads_re = reads_re
        self.set_reads_dtype(np.float16)

    # TODO: used in both, but won't be needed for the compressed constructor
    def set_reads_dtype(self, dtype):
        self.reads_re = self.reads_re.astype(dtype)

    # TODO: used before normalization, and after normalization for pruning and editing datasets
    def size_in_bytes(self):
        return self.reads_re.nbytes + self.get_nbytes()

    # TODO: used only after normalization, but will have to be modified
    def get_reads_re(self) -> np.ndarray:
        return self.reads_re

    # TODO: used both before and after normalization (when making batches)
    def get_ref_reads_re(self) -> np.ndarray:
        return self.reads_re[:-self.get_alt_count()]

    # TODO: used both before and after normalization (when making batches)
    def get_alt_reads_re(self) -> np.ndarray:
        return self.reads_re[-self.get_alt_count():]

    # TODO: used only after normalization
    @classmethod
    def save_list(cls, base_data: List[ReadsDatum], file):
        read_tensors = np.vstack([datum.get_reads_re() for datum in base_data])
        other_stuff = np.vstack([datum.get_array_1d() for datum in base_data])
        torch.save([read_tensors, other_stuff], file, pickle_protocol=4)

    # TODO: used only after normalization
    @classmethod
    def load_list(cls, file) -> List[ReadsDatum]:
        # these are vstacked -- see save method above
        read_tensors, datum_arrays = torch.load(file)

        result = []
        read_start_row = 0
        for datum_array in datum_arrays:
            datum = Datum(datum_array)
            read_count = datum.get_ref_count() + datum.get_alt_count()
            read_end_row = read_start_row + read_count

            reads_datum = ReadsDatum(datum_array=datum_array, reads_re=read_tensors[read_start_row:read_end_row])
            read_start_row = read_end_row
            result.append(reads_datum)

        return result


