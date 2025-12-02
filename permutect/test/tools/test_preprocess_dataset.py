from argparse import Namespace
import tempfile

from permutect.data.memory_mapped_data import MemoryMappedData
from permutect.data.reads_dataset import ReadsDataset
from permutect.tools import preprocess_dataset
from permutect import constants
from permutect.test.test_file_names import *

OVERWRITE_SAVED_TARFILE = False

def test_on_10_megabases_singular():
    training_datasets = [SMALL_PLAIN_TEXT_DATA]
    training_data_tarfile = tempfile.NamedTemporaryFile() if not OVERWRITE_SAVED_TARFILE else PREPROCESSED_DATA

    tarfile_name = training_data_tarfile.name if not OVERWRITE_SAVED_TARFILE else training_data_tarfile

    preprocess_args = Namespace()
    setattr(preprocess_args, constants.TRAINING_DATASETS_NAME, training_datasets)
    setattr(preprocess_args, constants.OUTPUT_NAME, tarfile_name)
    setattr(preprocess_args, constants.SOURCES_NAME, [0])
    preprocess_dataset.main_without_parsing(preprocess_args)

    memory_mapped_data = MemoryMappedData.load_from_tarfile(tarfile_name)
    dataset = ReadsDataset(memory_mapped_data=memory_mapped_data, num_folds=10)
    print("success")

test_on_10_megabases_singular()