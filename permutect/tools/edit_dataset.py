import argparse
import os
import tarfile
import tempfile
from enum import Enum

import torch.utils.data

from tqdm.autonotebook import tqdm

from permutect import constants
from permutect.data.memory_mapped_data import MemoryMappedData
from permutect.data.reads_dataset import ReadsDataset
from permutect.data.reads_datum import ReadsDatum
from permutect.misc_utils import report_memory_usage
from permutect.utils.enums import Label


class EditType(Enum):
    UNLABEL_ARTIFACTS = "unlabel_artifacts"
    UNLABEL_VARIANTS = "unlabel_variants"
    UNLABEL_EVERYTHING = "unlabel_everything"
    REMOVE_ARTIFACTS = "remove_artifacts"
    REMOVE_VARIANTS = "remove_variants"
    KEEP_EVERYTHING = "keep_everything"


# generates BaseDatum(s) from the original dataset that *pass* the pruning thresholds
def generate_edited_data(memory_mapped_datas, edit_type: str, source: int):
    for memory_mapped_data in memory_mapped_datas:
        for reads_datum in memory_mapped_data.generate_reads_data():
            if source is not None:
                reads_datum.set_source(source)

            if edit_type == EditType.UNLABEL_ARTIFACTS.value:
                if reads_datum.get_label() == Label.ARTIFACT:
                    reads_datum.set_label(Label.UNLABELED)
                yield reads_datum
            elif edit_type == EditType.UNLABEL_VARIANTS.value:
                if reads_datum.get_label() == Label.VARIANT:
                    reads_datum.set_label(Label.UNLABELED)
                yield reads_datum
            elif edit_type == EditType.UNLABEL_EVERYTHING.value:
                reads_datum.set_label(Label.UNLABELED)
                yield reads_datum
            elif edit_type == EditType.REMOVE_ARTIFACTS.value:
                if reads_datum.get_label() != Label.ARTIFACT:
                    yield reads_datum
            elif edit_type == EditType.REMOVE_VARIANTS.value:
                if reads_datum.get_label() != Label.VARIANT:
                    yield reads_datum
            elif edit_type == EditType.KEEP_EVERYTHING.value:
                yield reads_datum
            else:
                raise Exception(f"edit type {edit_type} not implemented yet")


def parse_arguments():
    parser = argparse.ArgumentParser(description='train the Mutect3 artifact model')
    parser.add_argument('--' + constants.DATASET_EDIT_TYPE_NAME, type=str, required=True,
                        help='how to modify the dataset')
    parser.add_argument('--' + constants.SOURCE_NAME, type=int, required=False, help='new source integer to apply')

    # input / output
    parser.add_argument('--' + constants.TRAIN_TAR_NAME, nargs='+', type=str, required=True,
                        help='tarfile(s) of training/validation datasets produced by preprocess_dataset.py')
    parser.add_argument('--' + constants.OUTPUT_NAME, type=str, required=True, help='path to pruned dataset file')

    return parser.parse_args()


def main_without_parsing(args):
    original_tarfiles = getattr(args, constants.TRAIN_TAR_NAME) # list of files
    output_tarfile = getattr(args, constants.OUTPUT_NAME)
    edit_type = getattr(args, constants.DATASET_EDIT_TYPE_NAME)
    new_source = getattr(args, constants.SOURCE_NAME)
    memory_mapped_datas = map(lambda input_tarfile: MemoryMappedData.load_from_tarfile(data_tarfile=input_tarfile), original_tarfiles)

    for mmd in memory_mapped_datas:
        print(f"Input dataset with {mmd.num_data} data and {mmd.num_reads} reads.")

    output_data_generator = generate_edited_data(memory_mapped_datas, edit_type, new_source)
    total_num_data = sum([mmd.num_data for mmd in memory_mapped_datas])
    total_num_reads = sum([mmd.num_reads for mmd in memory_mapped_datas])
    MemoryMappedData.from_generator(reads_datum_source=output_data_generator, estimated_num_data=total_num_data,
                                    estimated_num_reads=total_num_reads).save_to_tarfile(output_tarfile=output_tarfile)


def main():
    args = parse_arguments()
    main_without_parsing(args)


if __name__ == '__main__':
    main()
