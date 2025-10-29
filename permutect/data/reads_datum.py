from __future__ import annotations

import os
import tarfile
import tempfile
from typing import List, Generator

import numpy as np
import torch

from permutect.data.datum import Datum, DEFAULT_NUMPY_FLOAT
from permutect.misc_utils import ConsistentValue, report_memory_usage
from permutect.utils.allele_utils import trim_alleles_on_right, get_str_info_array, make_1d_sequence_tensor
from permutect.utils.enums import Variation, Label


# base strings longer than this when encoding data


# on disk and in datasets before creating batches, read tensors are stored with dtype np.uint8
# the leading columns are from binary quantities (either inherently binary or one-hot-encoded categorical) with np.packbits
# applied (this converts bool tensors into byte tensors, each byte holding eight bools)
# if the number of boolean bits is not a multiple of 8, it gets padded with zeros.  This isn't a problem.
NUMBER_OF_BYTES_IN_PACKED_READ = 7
READS_ARRAY_DTYPE = np.uint8

SUFFIX_FOR_DATA_FILES_IN_TAR = ".data"
SUFFIX_FOR_DATA_COUNT_FILE_IN_TAR = ".counts"

SUFFIX_FOR_DATA_MMAP_IN_TAR = ".data_mmap"
SUFFIX_FOR_READS_MMAP_IN_TAR = ".reads_mmap"


# quantile-normalized data is generally some small number of standard deviations from 0.  We can store as uint8 by
# mapping x --> 32x + 128 and restricting to the range [0,255], which maps -4 and +4 standard deviations to the limits
# of uint8
def convert_quantile_normalized_to_uint8(data: np.ndarray):
    return np.ndarray.astype(np.clip(data * 32 + 128, 0, 255), np.uint8)


# the inverse of the above
def convert_uint8_to_quantile_normalized(data: np.ndarray):
    return np.ndarray.astype((data - 128) / 32, DEFAULT_NUMPY_FLOAT)


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


# this is what we get from GATK plain text data.  It must be normalized and processed before becoming the
# data used by Permutect
class RawUnnormalizedReadsDatum(Datum):
    def __init__(self, datum_array: np.ndarray, reads_re: np.ndarray):
        super().__init__(datum_array)
        self.reads_re = reads_re

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
        return result

    def copy_with_downsampled_reads(self, ref_downsample: int, alt_downsample: int) -> RawUnnormalizedReadsDatum:
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
            return RawUnnormalizedReadsDatum(new_data_array, new_reads)

    def size_in_bytes(self):
        return self.reads_re.nbytes + self.get_nbytes()

    def get_reads_array_re(self) -> np.ndarray:
        return self.reads_re

    def get_ref_reads_re(self) -> np.ndarray:
        return self.reads_re[:-self.get_alt_count()]

    def get_alt_reads_re(self) -> np.ndarray:
        return self.reads_re[-self.get_alt_count():]


class ReadsDatum(Datum):
    # TODO: this comes up in 1) pruning, 2) memory-mapped dataset 3) loading, 4) normalizing plain-text data
    # TODO: 1 still needs to be fixed
    def __init__(self, datum_array: np.ndarray, compressed_reads_re: np.ndarray):
        super().__init__(datum_array)
        assert compressed_reads_re.dtype == READS_ARRAY_DTYPE

        # Reads are in a compressed, unusable form.  Binary columns must be unpacked and float
        # columns must be transformed back from uint8
        self.compressed_reads_re = compressed_reads_re

    def size_in_bytes(self):
        return self.compressed_reads_re.nbytes + self.get_nbytes()

    def get_reads_array_re(self) -> np.ndarray:
        return self.compressed_reads_re

    def get_compressed_reads_re(self) -> np.ndarray:
        return self.compressed_reads_re

    # this is the number of read features in a PyTorch float tensor, after unpacking the compressed binaries
    def num_read_features(self) -> int:
        num_read_uint8s = self.compressed_reads_re.shape[1]
        num_nonbinary_features = num_read_uint8s - NUMBER_OF_BYTES_IN_PACKED_READ
        # 8 bits per byte.  In general the last few features will be extraneous padded zeros, but that's okay.
        # the nonbinary features are converted to float, but it's one-to-one so no multiplicative factor
        return 8 * NUMBER_OF_BYTES_IN_PACKED_READ + num_nonbinary_features

    def get_compressed_ref_reads_re(self) -> np.ndarray:
        return self.compressed_reads_re[:-self.get_alt_count()]

    def get_compressed_alt_reads_re(self) -> np.ndarray:
        return self.compressed_reads_re[-self.get_alt_count():]

    @classmethod
    def save_list(cls, data: List[ReadsDatum], file):
        stacked_compressed_reads_re = np.vstack([datum.get_compressed_reads_re() for datum in data])
        stacked_data_array_ve = np.vstack([datum.get_array_1d() for datum in data])

        # if tensors are allowed to have more than 127 total reads (after downsampling) change this to int16!!!!
        read_counts_v = np.array([datum.get_ref_count() + datum.get_alt_count() for datum in data], dtype=np.int16)
        to_save = {'compressed_reads': stacked_compressed_reads_re,
                   'data_array': stacked_data_array_ve,
                   'read_counts': read_counts_v}
        #[stacked_compressed_reads_re, stacked_data_array_ve, read_counts_v]
        torch.save(to_save, file, pickle_protocol=4)

    # TODO: maybe the stacked arrays could be loaded directly as a dataset as opposed to the current approach of
    # TODO: unstacking them into Lists of ReadsDatum, then storing the data list in the datasetx
    @classmethod
    def load_list(cls, file) -> List[ReadsDatum]:
        assert file.endswith(SUFFIX_FOR_DATA_FILES_IN_TAR)
        loaded_data = torch.load(file)
        # these are vstacked -- see save method above
        stacked_compressed_reads_re, stacked_data_array_ve = loaded_data['compressed_reads'], loaded_data['data_array']

        result = []
        read_start_row = 0
        for datum_array in stacked_data_array_ve:
            datum = Datum(datum_array)
            read_count = datum.get_ref_count() + datum.get_alt_count()
            read_end_row = read_start_row + read_count

            reads_datum = ReadsDatum(datum_array=datum_array, compressed_reads_re=stacked_compressed_reads_re[read_start_row:read_end_row])
            read_start_row = read_end_row
            result.append(reads_datum)

        return result

    # takes a ReadsDatum generator and generates List[ReadsDatum]s of fixed size (except for a smaller final chunk)
    @classmethod
    def generate_data_lists(cls, data_generator: Generator[ReadsDatum], max_bytes_per_chunk: int) -> Generator[List[ReadsDatum]]:
        buffer, bytes_in_buffer = [], 0
        for datum in data_generator:

            buffer.append(datum)
            bytes_in_buffer += datum.size_in_bytes()
            if bytes_in_buffer > max_bytes_per_chunk:
                report_memory_usage(f"{bytes_in_buffer} bytes in chunk.")
                yield buffer
                buffer, bytes_in_buffer = [], 0

        # There will be some data left over, in general.
        if buffer:
            yield buffer

    @classmethod
    def save_lists_into_tarfile(cls, data_list_generator: Generator[List[ReadsDatum]], output_tarfile):
        data_files = []
        num_read_features, num_info_features, haplotypes_length = ConsistentValue(), ConsistentValue(), ConsistentValue()
        read_tensor_width, data_tensor_width = ConsistentValue(), ConsistentValue()

        # save all the lists of read sets to tempfiles. . .
        total_num_reads, total_num_data = 0, 0
        reads_datum_list: List[ReadsDatum]
        for reads_datum_list in data_list_generator:
            first_datum: ReadsDatum = reads_datum_list[0]
            num_read_features.check(first_datum.num_read_features())
            num_info_features.check(first_datum.get_info_1d().shape[0])
            haplotypes_length.check(first_datum.get_haplotypes_1d().shape[0])

            read_tensor_width.check(first_datum.compressed_reads_re.shape[-1])
            data_tensor_width.check(first_datum.array.shape[-1])

            with tempfile.NamedTemporaryFile(suffix=SUFFIX_FOR_DATA_FILES_IN_TAR, delete=False) as train_data_file:
                ReadsDatum.save_list(reads_datum_list, train_data_file)
                data_files.append(train_data_file.name)

            total_num_data += len(reads_datum_list)
            for datum in reads_datum_list:
                total_num_reads += (datum.get_ref_count() + datum.get_alt_count())

        # for pre-allocating tensor of all data in the tarfile, we store 1) the total data count 2) the total read count
        # 3) the tensor size (in memory, not after expanding binaries) of read data 4) the tensor size of 1D data
        with tempfile.NamedTemporaryFile(suffix=SUFFIX_FOR_DATA_COUNT_FILE_IN_TAR, delete=False) as counts_file:
            torch.save(np.array([total_num_data, total_num_reads, read_tensor_width.value, data_tensor_width.value]),
                       counts_file, pickle_protocol=4)
            data_files.append(counts_file.name)

        # . . . and bundle them in a tarfile
        with tarfile.open(output_tarfile, "w") as train_tar:
            for train_file in data_files:
                train_tar.add(train_file, arcname=os.path.basename(train_file))

    @classmethod
    def save_data_in_tarfile(cls, data_generator: Generator[ReadsDatum], max_bytes_in_chunk: int, output_tarfile):
        list_generator = ReadsDatum.generate_data_lists(data_generator=data_generator, max_bytes_per_chunk=max_bytes_in_chunk)
        ReadsDatum.save_lists_into_tarfile(data_list_generator=list_generator, output_tarfile=output_tarfile)

    @classmethod
    def extract_counts_from_tarfile(cls, data_tarfile):
        temp_dir = tempfile.TemporaryDirectory()

        with tarfile.open(data_tarfile, 'r') as tar:
            for member in tar.getmembers():
                if member.isfile() and member.name.endswith(SUFFIX_FOR_DATA_COUNT_FILE_IN_TAR):
                    tar.extract(member, path=temp_dir.name)

        count_files = [os.path.abspath(os.path.join(temp_dir.name, p)) for p in os.listdir(temp_dir.name) if
                      p.endswith(SUFFIX_FOR_DATA_COUNT_FILE_IN_TAR)]

        assert len(count_files) == 1

        count_array = torch.load(count_files[0])
        # counts array contains: total_num_data, total_num_reads, read_tensor_width, data_tensor_width
        return count_array[0], count_array[1], count_array[2], count_array[3]

    @classmethod
    def generate_reads_data_from_tarfile(cls, data_tarfile) -> Generator[ReadsDatum, None, None]:
        # extract the tarfile to a temporary directory that will be cleaned up when the program ends
        temp_dir = tempfile.TemporaryDirectory()
        tar = tarfile.open(data_tarfile)
        tar.extractall(temp_dir.name)
        tar.close()
        data_files = [os.path.abspath(os.path.join(temp_dir.name, p)) for p in os.listdir(temp_dir.name) if
                      p.endswith(SUFFIX_FOR_DATA_FILES_IN_TAR)]

        for file in data_files:
            for datum in ReadsDatum.load_list(file):
                yield datum

    @classmethod
    def generate_arrays_from_tarfile(cls, data_tarfile):
        # extract the tarfile to a temporary directory that will be cleaned up when the program ends
        temp_dir = tempfile.TemporaryDirectory()
        tar = tarfile.open(data_tarfile)
        tar.extractall(temp_dir.name)
        tar.close()
        data_files = [os.path.abspath(os.path.join(temp_dir.name, p)) for p in os.listdir(temp_dir.name) if
                          p.endswith(SUFFIX_FOR_DATA_FILES_IN_TAR)]

        for file in data_files:
            loaded_data = torch.load(file)
            stacked_compressed_reads_re, stacked_data_array_ve, read_counts_v = loaded_data['compressed_reads'], \
                loaded_data['data_array'], loaded_data['read_counts']
            yield stacked_compressed_reads_re, stacked_data_array_ve, read_counts_v




