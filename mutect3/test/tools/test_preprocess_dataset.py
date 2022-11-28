from mutect3.tools import preprocess_dataset
import tempfile


def test_on_dream1():
    training_datasets = ["/Users/davidben/mutect3/just-dream-1/dream1-normal-medium-training.dataset"]
    output_dir = tempfile.TemporaryDirectory()
    chunk_size = 100000
    preprocess_dataset.do_work(training_datasets, output_dir.name, chunk_size)

    k = 90
