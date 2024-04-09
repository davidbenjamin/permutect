import argparse
import os
import tarfile
import tempfile
import pickle

from mutect3 import constants
from mutect3.data import read_set
from mutect3.data.plain_text_data import generate_normalized_data, generate_artifact_posterior_data
from mutect3.utils import ConsistentValue, MutableInt

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
    parser.add_argument('--' + constants.OUTPUT_NAME, type=str, default=None, required=True,
                        help='path to output tarfile of training data')
    parser.add_argument('--' + constants.ARTIFACT_POSTERIOR_OUTPUT_NAME, type=str, default=None, required=True,
                        help='path to output tarfile of artifact posteriors data')
    return parser.parse_args()


def do_work(training_datasets, training_output_file, chunk_size, artifact_posterior_output_file):
    data_files = []
    num_read_features, num_info_features, ref_sequence_length = ConsistentValue(), ConsistentValue(), ConsistentValue()

    # save all the lists of read sets to tempfiles. . .
    datum_index = MutableInt()
    indices_file = open("indices.txt", 'w')
    # we include the variant string in order to record it in the indices file
    # even though we throw it out in save_list_of_read_sets
    for read_set_list in generate_normalized_data(training_datasets, max_bytes_per_chunk=chunk_size, include_variant_string=True):
        num_read_features.check(read_set_list[0].alt_reads_2d.shape[1])
        num_info_features.check(read_set_list[0].info_array_1d.shape[0])
        ref_sequence_length.check(read_set_list[0].ref_sequence_2d.shape[-1])

        with tempfile.NamedTemporaryFile(delete=False) as train_data_file:
            read_set.save_list_of_read_sets(read_set_list, train_data_file, datum_index, indices_file)
            data_files.append(train_data_file.name)

    indices_file.close()

    # . . . and bundle them in a tarfile
    with tarfile.open(training_output_file, "w") as train_tar:
        for train_file in data_files:
            train_tar.add(train_file, arcname=os.path.basename(train_file))

    artifact_posterior_files = []
    # TODO: hard-coded num posteriors per chunk!
    for list_of_posterior_data in generate_artifact_posterior_data(training_datasets, num_data_per_chunk=1000000):
        with tempfile.NamedTemporaryFile(delete=False) as artifact_posterior_file:
            pickle.dump(list_of_posterior_data, artifact_posterior_file)
            artifact_posterior_files.append(artifact_posterior_file.name)

    with tarfile.open(artifact_posterior_output_file, "w") as artifact_tar:
        for artifact_file in artifact_posterior_files:
            artifact_tar.add(artifact_file, arcname=os.path.basename(artifact_file))


def main_without_parsing(args):
    chunk_size = getattr(args, constants.CHUNK_SIZE_NAME)
    training_datasets = getattr(args, constants.TRAINING_DATASETS_NAME)
    output_file = getattr(args, constants.OUTPUT_NAME)
    artifact_posterior_output_file = getattr(args, constants.ARTIFACT_POSTERIOR_OUTPUT_NAME)

    do_work(training_datasets, output_file, chunk_size, artifact_posterior_output_file)


def main():
    args = parse_arguments()
    main_without_parsing(args)


if __name__ == '__main__':
    main()
