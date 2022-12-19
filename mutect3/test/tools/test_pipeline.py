from argparse import Namespace
import tempfile

from mutect3.tools import preprocess_dataset
from mutect3 import constants


def test_on_dream1():
    training_datasets = ["/Users/davidben/mutect3/just-dream-1/dream1-normal-small-training.dataset"]
    output_dir = tempfile.TemporaryDirectory()
    chunk_size = int(1e6)

    args = Namespace()
    setattr(args, constants.CHUNK_SIZE_NAME, chunk_size)
    setattr(args, constants.TRAINING_DATASETS_NAME, training_datasets)
    setattr(args, constants.OUTPUT_DIR_NAME, output_dir.name)
    preprocess_dataset.main(args)