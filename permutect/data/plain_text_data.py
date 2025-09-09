"""
Functions for reading from plain text dataset files of the format

UNLABELED                                       # label
1:13118,A->G                                    # locus and mutation
GAGGAAAGTGAGGTTGCCTGC                           # reference context
0.12 0.09 2.00 0.57 1.00 1.00 1.00 1.00 1.00    # variant-level info vector
5 4 0 0                                         # ref count, alt count, matched normal ref count, matched normal alt count
27 30 0 1 11 29 333 321 12 0 0                  # one ref read vector per line
50 22 1 0 30 19 342 70 272 0 0
27 31 0 1 32 17 236 203 33 0 0
27 20 0 1 32 17 141 72 69 1 0
21 28 1 0 49 0 232 49 183 1 0
23 29 1 1 40 9 335 294 41 0 0                   # one alt read vector per line
24 29 0 1 38 11 354 315 39 0 0
24 30 0 1 36 13 351 314 37 0 0
23 30 1 1 42 7 341 298 43 0 0
51 13 0 0                                       # original ref, alt, normal ref, normal alt counts before downsampling
-108.131                                        # sequencing error log likelihood
-0.000                                          # matched normal sequencing error log likelihood
VARIANT
1:13302,C->T
GTCCTGGACACGCTGTTGGCC
0.00 0.00 0.00 1.00 1.00 2.00 1.00 1.00 1.00
2 1 0 0
24 29 0 0 11 21 338 11 327 0 0
50 25 0 1 49 -8 355 303 52 0 0
23 33 1 0 13 21 312 87 225 0 0
69 4 0 0
-11.327
-0.000
"""
from typing import List, Generator
import sys

import numpy as np
from sklearn.preprocessing import QuantileTransformer

from permutect.data.count_binning import cap_ref_count, cap_alt_count
from permutect.data.reads_datum import ReadsDatum, RawUnnormalizedReadsDatum, NUMBER_OF_BYTES_IN_PACKED_READ, \
    convert_quantile_normalized_to_uint8
from permutect.data.datum import DEFAULT_NUMPY_FLOAT

from permutect.misc_utils import report_memory_usage
from permutect.utils.enums import Variation, Label

MAX_VALUE = 10000
EPSILON = 0.00001
QUANTILE_DATA_COUNT = 10000


def read_data(dataset_file, only_artifacts: bool = False, source: int=0) -> Generator[RawUnnormalizedReadsDatum, None, None]:
    """
    generator that yields data from a plain text dataset file.
    """
    with open(dataset_file) as file:
        n = 0
        while label_str := file.readline().strip():
            label = Label.get_label(label_str)
            passes_label_filter = (label == Label.ARTIFACT or not only_artifacts)
            n += 1

            # contig:position,ref->alt
            variant_line = file.readline().strip()
            locus, mutation = variant_line.split(",")
            contig, position = map(int, locus.split(":"))   # contig is an integer *index* from a sequence dictionary
            # TODO: replace with tqdm progress bar by counting file in initial pass.  It can't be that expensive.
            if n % 100000 == 0:
                print(f"{contig}:{position}")
            ref_allele, alt_allele = mutation.strip().split("->")

            ref_sequence_string = file.readline().strip()
            gatk_info_array = line_to_tensor(file.readline())
            ref_tensor_size, alt_tensor_size, normal_ref_tensor_size, normal_alt_tensor_size = map(int, file.readline().strip().split())

            # the first column is read group index, which we currently discard
            # later we're going to want to use this
            ref_tensor = read_2d_tensor(file, ref_tensor_size)[:,1:] if ref_tensor_size > 0 else None
            alt_tensor = read_2d_tensor(file, alt_tensor_size)[:,1:] if alt_tensor_size > 0 else None

            # normal_ref_tensor = read_2d_tensor(file, normal_ref_tensor_size)  # not currently used
            # normal_alt_tensor = read_2d_tensor(file, normal_alt_tensor_size)  # not currently used
            # round down normal tensors as well

            original_depth, original_alt_count, original_normal_depth, original_normal_alt_count = read_integers(file.readline())
            # this is -log10ToLog(tlod) - log(tumorDepth + 1);
            seq_error_log_lk = read_float(file.readline())
            # this is -log10ToLog(nalod) - log(normalDepth + 1)
            normal_seq_error_log_lk = read_float(file.readline())

            if alt_tensor_size > 0 and passes_label_filter:
                datum = RawUnnormalizedReadsDatum.from_gatk(label=label, variant_type=Variation.get_type(ref_allele, alt_allele), source=source,
                                           original_depth=original_depth, original_alt_count=original_alt_count,
                                           original_normal_depth=original_normal_depth, original_normal_alt_count=original_normal_alt_count,
                                           contig=contig, position=position, ref_allele=ref_allele, alt_allele=alt_allele,
                                           seq_error_log_lk=seq_error_log_lk, normal_seq_error_log_lk=normal_seq_error_log_lk,
                                           ref_sequence_string=ref_sequence_string, gatk_info_array=gatk_info_array,
                                           ref_tensor=ref_tensor, alt_tensor=alt_tensor)

                ref_count = cap_ref_count(datum.get_ref_count())
                alt_count = cap_alt_count(datum.get_alt_count())
                yield datum.copy_with_downsampled_reads(ref_count, alt_count)


# if sources is None, source is set to zero
# if List is length-1, that's the source for all files
# otherwise each file has its own source int
def generate_normalized_data(dataset_files, max_bytes_per_chunk: int, sources: List[int]=None) -> Generator[List[ReadsDatum], None, None]:
    """
    given text dataset files, generate normalized lists of read sets that fit in memory

    In addition to quantile-normalizing read tensors it also enlarges the info tensors
    :param dataset_files:
    :param max_bytes_per_chunk:
    :return:
    """
    for n, dataset_file in enumerate(dataset_files):
        buffer: List[RawUnnormalizedReadsDatum]
        buffer, bytes_in_buffer = [], 0
        read_quantile_transform = QuantileTransformer(n_quantiles=100, output_distribution='normal')
        info_quantile_transform = QuantileTransformer(n_quantiles=100, output_distribution='normal')

        num_buffers_filled = 0
        source = 0 if sources is None else (sources[0] if len(sources) == 1 else sources[n])
        reads_datum: RawUnnormalizedReadsDatum
        for reads_datum in read_data(dataset_file, source=source):
            buffer.append(reads_datum)
            bytes_in_buffer += reads_datum.size_in_bytes()
            if bytes_in_buffer > max_bytes_per_chunk:
                report_memory_usage()
                print(f"{bytes_in_buffer} bytes in chunk")

                yield normalize_buffer(buffer, read_quantile_transform, info_quantile_transform)
                num_buffers_filled += 1
                buffer, bytes_in_buffer = [], 0
        # There will be some data left over, in general.  Since it's small, use the last buffer's
        # quantile transforms for better statistical power if it's from the same text file
        if buffer:
            yield normalize_buffer(buffer, read_quantile_transform, info_quantile_transform, refit_transforms=(num_buffers_filled==0))


# this normalizes the buffer and also prepends new features to the info tensor
def normalize_buffer(buffer: List[RawUnnormalizedReadsDatum], read_quantile_transform,
                     info_quantile_transform, refit_transforms=True) -> List[ReadsDatum]:
    # 2D array.  Rows are ref/alt reads, columns are read features
    all_ref_re = np.vstack([datum.get_ref_reads_re() for datum in buffer])
    all_reads_re = np.vstack([datum.reads_re for datum in buffer])

    # 2D array.  Rows are read sets, columns are info features
    all_info_ve = np.vstack([datum.get_info_1d() for datum in buffer])
    binary_info_columns = binary_column_indices(all_info_ve)

    distance_columns_re = all_reads_re[:, 4:9]
    ref_distance_columns_re = all_ref_re[:, 4:9]
    if refit_transforms:    # fit quantiles column by column (aka feature by feature)
        read_quantile_transform.fit(ref_distance_columns_re)   # only these columns are quantile-transformed
        info_quantile_transform.fit(all_info_ve)

    distance_columns_transformed_re = read_quantile_transform.transform(distance_columns_re)

    all_info_transformed_ve = transform_except_for_binary_columns(all_info_ve, info_quantile_transform, binary_info_columns)

    # columns of raw read data are
    # 0 map qual -> 4 categorical columns
    # 1 base qual -> 4 categorical columns
    # 2,3 strand and orientation (binary) -> remain binary
    # 4,5,6,7,8 distance stuff
    # 9 and higher -- SNV/indel error and low BQ counts

    read_counts = np.array([len(datum.reads_re) for datum in buffer])
    read_index_ranges = np.cumsum(read_counts)

    map_qual_column = all_reads_re[:, 0]
    map_qual_boolean = np.full((len(all_reads_re), 4), True, dtype=bool)
    map_qual_boolean[:, 0] = (map_qual_column == 60)
    map_qual_boolean[:, 1] = (map_qual_column < 60) & (map_qual_column >= 40)
    map_qual_boolean[:, 2] = (map_qual_column < 40) & (map_qual_column >= 20)
    map_qual_boolean[:, 3] = (map_qual_column < 20)

    base_qual_column = all_reads_re[:, 1]
    base_qual_boolean = np.full((len(all_reads_re), 4), True, dtype=bool)
    base_qual_boolean[:, 0] = (base_qual_column >= 30)
    base_qual_boolean[:, 1] = (base_qual_column < 30) & (base_qual_column >= 20)
    base_qual_boolean[:, 2] = (base_qual_column < 20) & (base_qual_column >= 10)
    base_qual_boolean[:, 3] = (base_qual_column < 10)

    strand_and_orientation_boolean = all_reads_re[:, 2:4] < 0.5
    error_counts_boolean_1 = all_reads_re[:, 9:] < 0.5
    error_counts_boolean_2 = (all_reads_re[:, 9:] > 0.5) & (all_reads_re[:, 9:] < 1.5)
    error_counts_boolean_3 = (all_reads_re[:, 9:] > 1.5)

    boolean_output_array = np.hstack((map_qual_boolean, base_qual_boolean, strand_and_orientation_boolean,
               error_counts_boolean_1, error_counts_boolean_2, error_counts_boolean_3))

    # axis = 1 is essential so that each row (read) of the packed data corresponds to a row of the unpacked data
    packed_output_array = np.packbits(boolean_output_array, axis=1)

    assert packed_output_array.dtype == np.uint8
    assert packed_output_array.shape[1] == NUMBER_OF_BYTES_IN_PACKED_READ, f"boolean array shape {boolean_output_array.shape}, packed shape {packed_output_array.shape}"

    distance_columns_output = convert_quantile_normalized_to_uint8(distance_columns_transformed_re)
    assert packed_output_array.dtype == distance_columns_output.dtype
    output_uint8_reads_array = np.hstack((packed_output_array, distance_columns_output))

    # TODO: a thought -- reads dataset doesn't really need to store a list of reads datum.  It could JUST store the numpy arrays,
    # TODO: and furthermore, they could be concatenated to reduce the memory cost of many different objects

    normalized_result = []
    raw_datum: RawUnnormalizedReadsDatum
    for n, raw_datum in enumerate(buffer):
        output_reads_re = output_uint8_reads_array[0 if n == 0 else read_index_ranges[n - 1]:read_index_ranges[n]]
        output_datum: ReadsDatum = ReadsDatum(datum_array=raw_datum.array, compressed_reads_re=output_reads_re)

        # TODO: we used to put alt means and medians as extra info here.  If that's helpful to the model maybe restore
        # TODO: it not when storing a dataset but when creating batches?

        # TODO: info/base datum array should be uint8, not int64
        output_datum.set_info_1d(all_info_transformed_ve[n])
        normalized_result.append(output_datum)

    return normalized_result


def line_to_tensor(line: str) -> np.ndarray:
    tokens = line.strip().split()
    floats = [max(min(MAX_VALUE, float(token)), -MAX_VALUE) for token in tokens]
    return np.array(floats, dtype=DEFAULT_NUMPY_FLOAT)


def read_2d_tensor(file, num_lines: int) -> np.ndarray:
    if num_lines == 0:
        return None
    lines = [file.readline() for _ in range(num_lines)]
    return np.vstack([line_to_tensor(line) for line in lines])


def read_integers(line: str):
    return map(int, line.strip().split())


def read_float(line: str):
    return float(line.strip().split()[0])


def is_binary(column_tensor_1d: np.ndarray):
    assert len(column_tensor_1d.shape) == 1
    return all(el.item() == 0 or el.item() == 1 for el in column_tensor_1d)


def binary_column_indices(tensor_2d: np.ndarray):
    assert len(tensor_2d.shape) == 2
    return [n for n in range(tensor_2d.shape[1]) if is_binary(tensor_2d[:, n])]


def non_binary_column_indices(tensor_2d: np.ndarray):
    assert len(tensor_2d.shape) == 2
    return [n for n in range(tensor_2d.shape[1]) if not is_binary(tensor_2d[:, n])]


# copy the unnormalized values of binary features (columns)
# we modify the normalized values in-place
def restore_binary_columns(normalized, original, binary_columns):
    result = normalized
    if len(normalized.shape) == 2:
        result[:, binary_columns] = original[:, binary_columns]
    elif len(normalized.shape) == 1:
        result[binary_columns] = original[binary_columns]
    else:
        raise Exception("This is only for 1D or 2D tensors")
    return result


def transform_except_for_binary_columns(tensor_2d, quantile_transform: QuantileTransformer, binary_column_indices):
    return restore_binary_columns(quantile_transform.transform(tensor_2d), tensor_2d, binary_column_indices)