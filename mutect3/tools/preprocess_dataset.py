import argparse
import os
import tempfile
import tarfile

from mutect3 import constants
from mutect3.data import read_set
from mutect3.data.plain_text_data import generate_normalized_data

"""
This tool takes as input a list of text file Mutect3 training datasets, reads them in chunks that fit in memory,
normalizes each chunk, outputs each chunk as a binary PyTorch file, and bundles the output as a tarfile.
"""


TRAIN_TAR_NAME = "train.tar"
VALID_TAR_NAME = "valid.tar"
METADATA_NAME = "metadata.pt"


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--' + constants.TRAINING_DATASETS_NAME, nargs='+', type=str, required=True)
    parser.add_argument('--' + constants.CHUNK_SIZE_NAME, type=int, default=int(2e9), required=False)
    parser.add_argument('--' + constants.OUTPUT_DIR_NAME, type=str, default=None, required=False)
    return parser.parse_args()


def do_work(training_datasets, output_dir, chunk_size):
    data_files = []
    # TODO: need to accumulate the metadata here

    # save all the lists of read sets to tempfiles
    for read_set_list in generate_normalized_data(training_datasets, max_bytes_per_chunk=chunk_size):
        with tempfile.NamedTemporaryFile(delete=False) as train_data_file:
            read_set.save_list_of_read_sets(read_set_list, train_data_file)
            data_files.append(train_data_file.name)

        tempfile.NamedTemporaryFile()

    # and bundle them in a tarfile
    train_tar = output_dir + '/' + TRAIN_TAR_NAME
    with tarfile.open(train_tar, "w") as train_tar:
        for train_file in data_files:
            train_tar.add(train_file, arcname=os.path.basename(train_file))



    metadata = output_dir + '/' + METADATA_NAME

    torch.save([self.num_read_features, self.num_info_features, self.ref_sequence_length, self.num_training_data,
                self.training_artifact_totals, self.training_non_artifact_totals], metadata_file)



def main():
    args = parse_arguments()
    chunk_size = getattr(args, constants.CHUNK_SIZE_NAME)
    training_datasets = getattr(args, constants.TRAINING_DATASETS_NAME)
    output_dir = getattr(args, constants.OUTPUT_DIR_NAME)
    output_dir = output_dir if output_dir is not None else os.getcwd()

    do_work(training_datasets, output_dir, chunk_size)





if __name__ == '__main__':
    main()