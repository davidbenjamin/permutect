import argparse
import os
import tarfile
import tempfile
from typing import List

import numpy as np
import torch

from permutect import constants
from permutect.data.reads_datum import ReadsDatum
from permutect.data.plain_text_data import make_normalized_mmap_data
from permutect.misc_utils import ConsistentValue



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


def do_work(plain_text_dataset_files, output_tarfile, sources: List[int]):
    memory_mapped_data = make_normalized_mmap_data(plain_text_dataset_files, sources=sources)
    memory_mapped_data.save_to_tarfile(output_tarfile)


def main_without_parsing(args):
    chunk_size = getattr(args, constants.CHUNK_SIZE_NAME)
    training_datasets = getattr(args, constants.TRAINING_DATASETS_NAME)
    output_file = getattr(args, constants.OUTPUT_NAME)
    sources = getattr(args, constants.SOURCES_NAME)

    do_work(training_datasets, output_file, sources)


def main():
    args = parse_arguments()
    main_without_parsing(args)


if __name__ == '__main__':
    main()
