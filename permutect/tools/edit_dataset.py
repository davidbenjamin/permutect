import argparse
import os
import tarfile
import tempfile

import psutil

from permutect.data import base_datum
from tqdm.autonotebook import tqdm

from permutect import constants
from permutect.data.base_dataset import BaseDataset


# generates BaseDatum(s) from the original dataset that *pass* the pruning thresholds
def generate_edited_data(base_dataset: BaseDataset):
    print("pruning the dataset")
    pbar = tqdm(enumerate(filter(lambda bat: bat.is_labeled(), base_dataset)), mininterval=60)
    for n, base_datum in pbar:
        if:
            pass
        else:
            yield edited_datum # this is a ReadSet


# takes a ReadSet generator and organizes into buffers.
# TODO: probably code duplication since the generator is already pruned
def generate_output_data_buffers(output_data_generator, max_bytes_per_chunk: int):
    buffer, bytes_in_buffer = [], 0
    for datum in output_data_generator:

        buffer.append(datum)
        bytes_in_buffer += datum.size_in_bytes()
        if bytes_in_buffer > max_bytes_per_chunk:
            print(f"Memory usage percent: {psutil.virtual_memory().percent:.1f}")
            print(f"{bytes_in_buffer} bytes in chunk")
            yield buffer
            buffer, bytes_in_buffer = [], 0

    # There will be some data left over, in general.
    if buffer:
        yield buffer


def make_output_training_dataset(pruned_data_buffer_generator, output_tarfile):
    pruned_data_files = []
    for base_data_list in pruned_data_buffer_generator:
        with tempfile.NamedTemporaryFile(delete=False) as train_data_file:
            base_datum.save_list_base_data(base_data_list, train_data_file)
            pruned_data_files.append(train_data_file.name)

    # bundle them in a tarfile
    with tarfile.open(output_tarfile, "w") as train_tar:
        for train_file in pruned_data_files:
            train_tar.add(train_file, arcname=os.path.basename(train_file))


def parse_arguments():
    parser = argparse.ArgumentParser(description='train the Mutect3 artifact model')
    parser.add_argument('--' + constants.CHUNK_SIZE_NAME, type=int, default=int(2e9), required=False,
                        help='size in bytes of output binary data files')

    # input / output
    parser.add_argument('--' + constants.TRAIN_TAR_NAME, type=str, required=True,
                        help='tarfile of training/validation datasets produced by preprocess_dataset.py')
    parser.add_argument('--' + constants.OUTPUT_NAME, type=str, required=True, help='path to pruned dataset file')

    return parser.parse_args()


def main_without_parsing(args):
    original_tarfile = getattr(args, constants.TRAIN_TAR_NAME)
    output_tarfile = getattr(args, constants.OUTPUT_NAME)
    chunk_size = getattr(args, constants.CHUNK_SIZE_NAME)
    base_dataset = BaseDataset(data_tarfile=original_tarfile)

    # generate ReadSets
    output_data_generator = generate_edited_data(base_dataset)

    # generate List[ReadSet]s
    output_data_buffer_generator = generate_output_data_buffers(output_data_generator, chunk_size)

    make_output_training_dataset(output_data_buffer_generator, output_tarfile=output_tarfile)


def main():
    args = parse_arguments()
    main_without_parsing(args)


if __name__ == '__main__':
    main()
