from mutect3.tools import train_model
from mutect3.architecture import artifact_model

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import tempfile
from torch.utils.tensorboard import SummaryWriter


def test_training(dataset):
    m3_params = artifact_model.ArtifactModelParameters(read_layers=[20, 20, 20], info_layers=[20, 20], aggregation_layers=[20, 20],
                                                       dropout_p=0.0, batch_normalize=False, learning_rate=0.001)
    training_params = train_model.TrainingParameters(batch_size=64, num_epochs=5, reweighting_range=0.3)

    with tempfile.TemporaryDirectory() as tensorboard_dir:
        train_model.train_artifact_model(m3_params, [dataset], training_params, tensorboard_dir)
        events = EventAccumulator(tensorboard_dir)
        events.Reload()
        h = 99


def test_on_dream1():
    test_training("/Users/davidben/mutect3/all_dream/small.dataset")
