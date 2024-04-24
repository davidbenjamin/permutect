from argparse import Namespace
import tempfile
import pickle

from permutect.data.read_set_dataset import ReadSetDataset
from permutect.tools import preprocess_dataset, train_model, filter_variants
from permutect import constants
from permutect.utils import extract_to_temp_dir


def test_on_dream1():
    # Input Files
    training_datasets = ["/Users/davidben/permutect/just-dream-1/dream1-normal-small-training.dataset"]

    # Intermediate and Output Files
    training_data_tarfile = tempfile.NamedTemporaryFile()
    artifact_posterior_tarfile = tempfile.NamedTemporaryFile()

    # STEP 1: preprocess the plain text training dataset yielding a training tarfile
    preprocess_args = Namespace()
    setattr(preprocess_args, constants.CHUNK_SIZE_NAME, 1e6)
    setattr(preprocess_args, constants.TRAINING_DATASETS_NAME, training_datasets)
    setattr(preprocess_args, constants.OUTPUT_NAME, training_data_tarfile.name)
    preprocess_dataset.main_without_parsing(preprocess_args)

    with tempfile.TemporaryDirectory() as train_temp_dir, tempfile.TemporaryDirectory() as artifact_temp_dir:
        training_files = extract_to_temp_dir(training_data_tarfile.name, train_temp_dir)
        artifact_files = extract_to_temp_dir(artifact_posterior_tarfile.name, artifact_temp_dir)

        depth = 0
        for file in artifact_files:
            for artifact_posterior in pickle.load(open(file, 'rb')):
                depth += artifact_posterior.depth

    dataset = ReadSetDataset(data_tarfile=training_data_tarfile.name, num_folds=10)