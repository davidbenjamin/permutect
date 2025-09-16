import argparse
import os
import tarfile
import tempfile
from typing import List

import numpy as np
import torch

from permutect import constants
from permutect.data.reads_datum import ReadsDatum
from permutect.data.plain_text_data import generate_normalized_data
from permutect.misc_utils import ConsistentValue

SUFFIX_FOR_DATA_FILES_IN_TAR = ".data"
SUFFIX_FOR_DATA_COUNT_FILE_IN_TAR = ".counts"

"""
This tool takes as input a list of text file Mutect3 training datasets, reads them in chunks that fit in memory,
normalizes each chunk, outputs each chunk as a binary PyTorch file, and bundles the output as a tarfile.
"""


def parse_arguments():
    parser = argparse.ArgumentParser(description='preprocess plain text training dataset into tarfile of nprmalized binary data')
    parser.add_argument('--' + constants.TRAINING_DATASETS_NAME, nargs='+', type=str, required=True,
                        help='list of plain text data files')
    parser.add_argument('--' + constants.CHUNK_SIZE_NAME, type=int, default=int(2e9), required=False,
                        help='size in bytes of output binary data files')
    parser.add_argument('--' + constants.SOURCES_NAME, nargs='+', type=int, required=False,
                        help='integer sources corresponding to plain text data files for distinguishing different sequencing conditions')
    parser.add_argument('--' + constants.OUTPUT_NAME, type=str, default=None, required=True,
                        help='path to output tarfile of training data')
    return parser.parse_args()


def do_work(training_datasets, training_output_file, chunk_size, sources: List[int]):
    data_files = []
    num_read_features, num_info_features, haplotypes_length = ConsistentValue(), ConsistentValue(), ConsistentValue()
    read_tensor_width, data_tensor_width = ConsistentValue(), ConsistentValue()

    # TODO: extract a save reads in tarfiles method and deal with code duplication in prune and edit
    # save all the lists of read sets to tempfiles. . .
    total_num_reads, total_num_data = 0, 0
    reads_datum_list: List[ReadsDatum]
    for reads_datum_list in generate_normalized_data(training_datasets, max_bytes_per_chunk=chunk_size, sources=sources):
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
        ReadsDatum.save_list(reads_datum_list, train_data_file)
        torch.save([np.array([total_num_data, total_num_reads, read_tensor_width.value, data_tensor_width.value])], counts_file, pickle_protocol=4)
        data_files.append(counts_file.name)

    # . . . and bundle them in a tarfile
    with tarfile.open(training_output_file, "w") as train_tar:
        for train_file in data_files:
            train_tar.add(train_file, arcname=os.path.basename(train_file))


def main_without_parsing(args):
    chunk_size = getattr(args, constants.CHUNK_SIZE_NAME)
    training_datasets = getattr(args, constants.TRAINING_DATASETS_NAME)
    output_file = getattr(args, constants.OUTPUT_NAME)
    sources = getattr(args, constants.SOURCES_NAME)

    do_work(training_datasets, output_file, chunk_size, sources)


def main():
    args = parse_arguments()
    main_without_parsing(args)


if __name__ == '__main__':
    main()
