from argparse import Namespace
import tempfile

from permutect.data.reads_dataset import ReadsDataset
from permutect.data.reads_datum import ReadsDatum
from permutect.tools import preprocess_dataset
from permutect import constants
from permutect.misc_utils import extract_to_temp_dir


def test_on_10_megabases_singular():
    training_datasets = ["/Users/davidben/mutect3/permutect/integration-tests/singular-10-Mb/training-dataset.txt"]
    training_data_tarfile = tempfile.NamedTemporaryFile()

    preprocess_args = Namespace()
    setattr(preprocess_args, constants.CHUNK_SIZE_NAME, 1e6)
    setattr(preprocess_args, constants.TRAINING_DATASETS_NAME, training_datasets)
    setattr(preprocess_args, constants.OUTPUT_NAME, training_data_tarfile.name)
    setattr(preprocess_args, constants.SOURCES_NAME, [0])
    preprocess_dataset.main_without_parsing(preprocess_args)

    with tempfile.TemporaryDirectory() as train_temp_dir:
        training_files = extract_to_temp_dir(training_data_tarfile.name, train_temp_dir)
        for training_file in training_files:
            base_data_list = ReadsDatum.load_list(training_file)

    dataset = ReadsDataset(data_tarfile=training_data_tarfile.name, num_folds=10)