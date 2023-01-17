import argparse
import os
import tarfile
import tempfile

from mutect3 import constants, utils
from mutect3.utils import ConsistentValue
from mutect3.data import read_set
from mutect3.data.plain_text_data import generate_normalized_data

"""
This tool takes as input a list of text file Mutect3 training datasets, reads them in chunks that fit in memory,
normalizes each chunk, outputs each chunk as a binary PyTorch file, and bundles the output as a tarfile.
"""


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--' + constants.TRAINING_DATASETS_NAME, nargs='+', type=str, required=True)
    parser.add_argument('--' + constants.CHUNK_SIZE_NAME, type=int, default=int(2e9), required=False)
    parser.add_argument('--' + constants.OUTPUT_NAME, type=str, default=None, required=False)
    return parser.parse_args()


def do_work(training_datasets, output_file, chunk_size):
    data_files = []
    num_read_features, num_info_features, ref_sequence_length = ConsistentValue(), ConsistentValue(), ConsistentValue()

    # save all the lists of read sets to tempfiles. . .
    for read_set_list in generate_normalized_data(training_datasets, max_bytes_per_chunk=chunk_size):
        num_read_features.check(read_set_list[0].ref_tensor.shape[1])
        num_info_features.check(read_set_list[0].info_tensor.shape[0])
        ref_sequence_length.check(read_set_list[0].ref_sequence_tensor.shape[-1])

        with tempfile.NamedTemporaryFile(delete=False) as train_data_file:
            read_set.save_list_of_read_sets(read_set_list, train_data_file)
            data_files.append(train_data_file.name)

        tempfile.NamedTemporaryFile()

    # . . . and bundle them in a tarfile
    with tarfile.open(output_file, "w") as train_tar:
        for train_file in data_files:
            train_tar.add(train_file, arcname=os.path.basename(train_file))


def main(args):
    chunk_size = getattr(args, constants.CHUNK_SIZE_NAME)
    training_datasets = getattr(args, constants.TRAINING_DATASETS_NAME)
    output_file = getattr(args, constants.OUTPUT_NAME)

    do_work(training_datasets, output_file, chunk_size)


if __name__ == '__main__':
    main(parse_arguments())