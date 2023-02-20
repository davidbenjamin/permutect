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
import numpy as np
import psutil

from mutect3 import utils
from mutect3.data.posterior import PosteriorDatum
from mutect3.data.read_set import ReadSet

from mutect3.utils import Label, Variation, encode

MAX_VALUE = 10000
EPSILON = 0.00001
QUANTILE_DATA_COUNT = 10000


# ex: [0, 1, 3, 6, 7] -> [0, 1, 1, 3, 3, 3, 6, 7] -- 2 is rounded down to the nearest cutoff, 1 and likewise 4 and 5 are rounded to 3
# assumes that cutoffs include 0 and are ascending
def make_round_down_table(cutoffs):
    table = []
    n = 0
    for idx, cutoff in enumerate(cutoffs[:-1]):
        while n < cutoffs[idx + 1]:
            table.append(cutoff)
            n += 1
    table.append(cutoffs[-1])
    return table

# arrays where array[n] is what n reads are rounded down to
# this is important because we batch by equal ref and alt counts and we would like to reduce
# the number of combinations in order to have big batches
# ALT_ROUNDING = make_round_down_table([0, 1, 2, 3, 4, 5, 7, 10, 13, 16, 20])
REF_ROUNDING = make_round_down_table([0, 1, 5, 10])

ALT_ROUNDING = make_round_down_table(list(range(21)))
# REF_ROUNDING = make_round_down_table(list(range(21)))


def round_down_ref(n: int):
    return n
    #return REF_ROUNDING[min(len(REF_ROUNDING) - 1, n)]


def round_down_alt(n: int):
    return ALT_ROUNDING[min(len(ALT_ROUNDING) - 1, n)]


def read_data(dataset_file, posterior: bool, round_down: bool = True, include_variant_string: bool = False):
    """
    generator that yields data from a plain text dataset file. In posterior mode, yield a tuple of ReadSet and PosteriorDatum
    """
    with open(dataset_file) as file:
        n = 0
        while label_str := file.readline().strip():
            label = Label.get_label(label_str)
            n += 1

            # contig:position,ref->alt
            variant_line = file.readline().strip()
            locus, mutation = variant_line.split(",")
            contig, position = locus.split(":")
            position = int(position)
            # TODO: replace with tqdm progress bar by counting file in initial pass.  It can't be that expensive.
            if n % 100000 == 0:
                print(contig + ":" + str(position))
            ref_allele, alt_allele = mutation.strip().split("->")

            if posterior:
                skip_ref_sequence_line = file.readline()
                skip_info_tensor_line = file.readline()
                ref_tensor_size, alt_tensor_size, normal_ref_tensor_size, normal_alt_tensor_size = map(int, file.readline().strip().split())

                tensor_lines_to_skip = ref_tensor_size + alt_tensor_size
                for _ in range(tensor_lines_to_skip):
                    file.readline()
                depth, alt_count, normal_depth, normal_alt_count = read_integers(file.readline())
                seq_error_log_likelihood = read_float(file.readline())
                normal_seq_error_log_likelihood = -read_float(file.readline())  # the GATK emits the negative of what should really be output

                if alt_tensor_size > 0:
                    yield PosteriorDatum(contig, position, ref_allele, alt_allele, depth, alt_count, normal_depth,
                                         normal_alt_count, seq_error_log_likelihood, normal_seq_error_log_likelihood)
            else:
                # ref base string
                ref_sequence_string = file.readline().strip()
                gatk_info_tensor = line_to_tensor(file.readline())
                ref_tensor_size, alt_tensor_size, normal_ref_tensor_size, normal_alt_tensor_size = map(int, file.readline().strip().split())

                ref_tensor = read_2d_tensor(file, ref_tensor_size) if ref_tensor_size > 0 else None
                alt_tensor = read_2d_tensor(file, alt_tensor_size)

                if round_down:
                    ref_tensor = utils.downsample_tensor(ref_tensor, 0 if ref_tensor is None else round_down_ref(len(ref_tensor)))
                    alt_tensor = utils.downsample_tensor(alt_tensor, 0 if alt_tensor is None else round_down_alt(len(alt_tensor)))

                # normal_ref_tensor = read_2d_tensor(file, normal_ref_tensor_size)  # not currently used
                # normal_alt_tensor = read_2d_tensor(file, normal_alt_tensor_size)  # not currently used
                # round down normal tensors as well

                skip_depths = file.readline()
                skip_seq_error = file.readline()
                skip_normal_seq_error = file.readline()

                if alt_tensor_size > 0:
                    yield ReadSet.from_gatk(ref_sequence_string, Variation.get_type(ref_allele, alt_allele), ref_tensor,
                                          alt_tensor, gatk_info_tensor, label, encode(contig, position, alt_allele) if include_variant_string else None)


def generate_normalized_data(dataset_files, max_bytes_per_chunk: int, include_variant_string: bool = False):
    """
    given text dataset files, generate normalized lists of read sets that fit in memory
    :param dataset_files:
    :param max_bytes_per_chunk:
    :param include_variant_string: include the variant string in generated ReadSet data
    :return:
    """
    for dataset_file in dataset_files:
        buffer, bytes_in_buffer = [], 0
        read_medians, read_iqrs, info_medians, info_iqrs = None, None, None, None

        for read_set in read_data(dataset_file, posterior=False, include_variant_string=include_variant_string):
            buffer.append(read_set)
            bytes_in_buffer += read_set.size_in_bytes()
            if bytes_in_buffer > max_bytes_per_chunk:
                print("memory usage percent: " + str(psutil.virtual_memory().percent))
                print("bytes in chunk: " + str(bytes_in_buffer))

                normalize_buffer(buffer)
                yield buffer
                buffer, bytes_in_buffer = [], 0
        # There will be some data left over, in general.  Since it's small, use the last buffer's
        # medians and IQRs if available for better statistical power if it's from the same text file
        if buffer:
            normalize_buffer(buffer, read_medians_override=read_medians, read_iqrs_override=read_iqrs,
                         info_medians_override=info_medians, info_iqrs_override=info_iqrs)
            yield buffer


def normalize_buffer(buffer, read_medians_override=None, read_iqrs_override=None, info_medians_override=None, info_iqrs_override=None):
    all_ref = np.vstack([datum.ref_tensor for datum in buffer if datum.ref_tensor is not None])
    all_info = np.vstack([datum.info_tensor for datum in buffer])

    read_medians, read_iqrs = medians_and_iqrs(all_ref) if read_medians_override is None else (read_medians_override, read_iqrs_override)
    info_medians, info_iqrs = medians_and_iqrs(all_info) if info_medians_override is None else (info_medians_override, info_iqrs_override)

    binary_read_columns = binary_column_indices(all_ref)
    binary_info_columns = binary_column_indices(all_info)

    # normalize by subtracting medium and dividing by quantile range
    for datum in buffer:
        datum.ref_tensor = None if datum.ref_tensor is None else restore_binary_columns(normalized=(datum.ref_tensor - read_medians) / read_iqrs,
            original=datum.ref_tensor, binary_columns=binary_read_columns)
        datum.alt_tensor = restore_binary_columns(normalized=(datum.alt_tensor - read_medians) / read_iqrs,
            original=datum.alt_tensor, binary_columns=binary_read_columns)
        datum.info_tensor = restore_binary_columns(normalized=(datum.info_tensor - info_medians) / info_iqrs,
            original=datum.info_tensor, binary_columns=binary_info_columns)


# check whether all elements are 0 or 1
def is_binary(column_tensor_1d: np.ndarray):
    assert len(column_tensor_1d.shape) == 1
    return all(el.item() == 0 or el.item() == 1 for el in column_tensor_1d)


def binary_column_indices(tensor_2d: np.ndarray):
    assert len(tensor_2d.shape) == 2
    return [n for n in range(tensor_2d.shape[1]) if is_binary(tensor_2d[:, n])]


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


def medians_and_iqrs(tensor_2d: np.ndarray):
    # column medians etc
    medians = np.quantile(tensor_2d, q=0.5, axis=0)
    val = 0.05
    iqrs = (np.quantile(tensor_2d, 1 - val, axis=0) - np.quantile(tensor_2d, val, axis=0)) + EPSILON

    return medians, iqrs


def line_to_tensor(line: str) -> np.ndarray:
    tokens = line.strip().split()
    floats = [float(token) for token in tokens]
    return np.clip(np.array(floats), -MAX_VALUE, MAX_VALUE)


def read_2d_tensor(file, num_lines: int) -> np.ndarray:
    if num_lines == 0:
        return None
    lines = [file.readline() for _ in range(num_lines)]
    return np.vstack([line_to_tensor(line) for line in lines])


def read_integers(line: str):
    return map(int, line.strip().split())


def read_float(line: str):
    return float(line.strip().split()[0])