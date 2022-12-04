import argparse
import os

from mutect3 import constants
from mutect3.data import read_set_dataset


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
    big_dataset = read_set_dataset.BigReadSetDataset(max_bytes_per_chunk=chunk_size, dataset_files=training_datasets)

    train_tar = output_dir + '/' + TRAIN_TAR_NAME
    valid_tar = output_dir + '/' + VALID_TAR_NAME
    metadata = output_dir + '/' + METADATA_NAME

    big_dataset.save_data(train_tar_file=train_tar, valid_tar_file=valid_tar, metadata_file=metadata)


def main():
    args = parse_arguments()
    chunk_size = getattr(args, constants.CHUNK_SIZE_NAME)
    training_datasets = getattr(args, constants.TRAINING_DATASETS_NAME)
    output_dir = getattr(args, constants.OUTPUT_DIR_NAME)
    output_dir = output_dir if output_dir is not None else os.getcwd()

    do_work(training_datasets, output_dir, chunk_size)




if __name__ == '__main__':
    main()