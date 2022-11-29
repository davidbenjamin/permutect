from mutect3.tools import preprocess_dataset
import tempfile
from mutect3.data.read_set_dataset import BigReadSetDataset
from mutect3.test.architecture.test_artifact_model import SMALL_MODEL_PARAMS, TRAINING_PARAMS
from torch.utils.tensorboard import SummaryWriter
from mutect3.architecture.artifact_model import ArtifactModel


def test_on_dream1():
    training_datasets = ["/Users/davidben/mutect3/just-dream-1/dream1-normal-small-training.dataset"]
    output_dir = tempfile.TemporaryDirectory()
    chunk_size = int(1e6)
    preprocess_dataset.do_work(training_datasets, output_dir.name, chunk_size)

    train_tar = output_dir.name + '/' + "train.tar"
    valid_tar = output_dir.name + '/' + "valid.tar"
    metadata = output_dir.name + '/' + "metadata.pt"

    big_dataset = BigReadSetDataset(batch_size=64, train_valid_meta_tuple=(train_tar, valid_tar, metadata))
    params = SMALL_MODEL_PARAMS
    training_params = TRAINING_PARAMS

    with tempfile.TemporaryDirectory() as tensorboard_dir:
        summary_writer = SummaryWriter(tensorboard_dir)
        model = ArtifactModel(params=params, num_read_features=big_dataset.num_read_features,
                              num_info_features=big_dataset.num_info_features,
                              ref_sequence_length=big_dataset.ref_sequence_length).float()
        model.train_model(big_dataset, training_params.num_epochs, summary_writer=summary_writer,
                          reweighting_range=training_params.reweighting_range, m3_params=params)



    k = 90
