from argparse import Namespace
import tempfile

from permutect.data.reads_dataset import ReadsDataset
from permutect.data.reads_datum import ReadsDatum, SUFFIX_FOR_DATA_FILES_IN_TAR
from permutect.tools import preprocess_dataset
from permutect import constants
from permutect.misc_utils import extract_to_temp_dir

OVERWRITE_SAVED_TARFILE = True


def test_on_10_megabases_singular():
    training_datasets = ["/Users/davidben/mutect3/permutect/integration-tests/hiseqx-NA12878.dataset"]
    training_data_tarfile = tempfile.NamedTemporaryFile() if not OVERWRITE_SAVED_TARFILE else \
        "/Users/davidben/mutect3/permutect/integration-tests/preprocessed-dataset.tar"

    tarfile_name = training_data_tarfile.name if not OVERWRITE_SAVED_TARFILE else training_data_tarfile

    preprocess_args = Namespace()
    setattr(preprocess_args, constants.TRAINING_DATASETS_NAME, training_datasets)
    setattr(preprocess_args, constants.OUTPUT_NAME, tarfile_name)
    setattr(preprocess_args, constants.SOURCES_NAME, [0])
    preprocess_dataset.main_without_parsing(preprocess_args)

    with tempfile.TemporaryDirectory() as train_temp_dir:
        training_files = extract_to_temp_dir(tarfile_name, train_temp_dir)

    dataset = ReadsDataset(tarfile=tarfile_name, num_folds=10)