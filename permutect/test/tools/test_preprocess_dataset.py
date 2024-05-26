from argparse import Namespace
import tempfile
import pickle

from permutect.data.read_set_dataset import ReadSetDataset
from permutect.tools import preprocess_dataset
from permutect import constants
from permutect.utils import extract_to_temp_dir


def test_on_10_megabases_singular():
    # Input Files
    training_datasets = ["/Users/davidben/mutect3/permutect/integration-tests/singular-10-Mb/training-dataset.txt"]

    # Intermediate and Output Files
    training_data_tarfile = tempfile.NamedTemporaryFile()

    # STEP 1: preprocess the plain text training dataset yielding a training tarfile
    preprocess_args = Namespace()
    setattr(preprocess_args, constants.CHUNK_SIZE_NAME, 1e6)
    setattr(preprocess_args, constants.TRAINING_DATASETS_NAME, training_datasets)
    setattr(preprocess_args, constants.OUTPUT_NAME, training_data_tarfile.name)
    preprocess_dataset.main_without_parsing(preprocess_args)

    with tempfile.TemporaryDirectory() as train_temp_dir:
        training_files = extract_to_temp_dir(training_data_tarfile.name, train_temp_dir)

    dataset = ReadSetDataset(data_tarfile=training_data_tarfile.name, num_folds=10)