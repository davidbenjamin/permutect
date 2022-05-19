from mutect3.test.test_utils import artificial_data
from mutect3.data.read_set_dataset import ReadSetDataset, make_semisupervised_data_loader, make_test_data_loader
from mutect3.data.read_set_datum import ReadSetDatum
from typing import Iterable
from mutect3.architecture.normal_artifact_model import NormalArtifactModel
from mutect3.architecture.read_set_classifier import ReadSetClassifier, Mutect3Parameters
from mutect3 import utils
from mutect3.tools.train_model import TrainingParameters

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import tempfile
from torch.utils.tensorboard import SummaryWriter

BATCH_SIZE = 64
NUM_EPOCHS = 100
NUM_SPECTRUM_ITERATIONS=10
TRAINING_PARAMS = TrainingParameters(batch_size=BATCH_SIZE, num_epochs=NUM_EPOCHS, reweighting_range=0.3)

SMALL_MODEL_PARAMS = Mutect3Parameters(read_layers=[5, 5], info_layers=[5, 5], aggregation_layers=[5, 5, 5, 5],
                                       dropout_p=0.2, batch_normalize=False, learning_rate=0.001)


# Note that the test methods in this class also cover batching, samplers, datasets, and data loaders
def train_model_and_write_summary(m3_params: Mutect3Parameters, training_params: TrainingParameters,
                                  data: Iterable[ReadSetDatum], summary_writer: SummaryWriter = None):
    dataset = ReadSetDataset(data=data)
    training, valid = utils.split_dataset_into_train_and_valid(dataset, 0.9)
    na_model = NormalArtifactModel([10, 10, 10])

    train_loader = make_semisupervised_data_loader(training, training_params.batch_size)
    valid_loader = make_semisupervised_data_loader(valid, training_params.batch_size)
    model = ReadSetClassifier(m3_params=m3_params, na_model=na_model).float()

    model.train_model(train_loader, valid_loader, training_params.num_epochs, summary_writer=summary_writer,
                      reweighting_range=training_params.reweighting_range, m3_params=m3_params)
    model.learn_calibration(valid_loader, num_epochs=50)
    model.evaluate_model_after_training(train_loader, summary_writer)
    return model


def test_separate_gaussian_data():
    data = artificial_data.make_two_gaussian_data(10000)
    params = SMALL_MODEL_PARAMS
    training_params = TRAINING_PARAMS

    with tempfile.TemporaryDirectory() as tensorboard_dir:
        summary_writer = SummaryWriter(tensorboard_dir)
        model = train_model_and_write_summary(m3_params=params, training_params=training_params, data=data, summary_writer=summary_writer)

        test_data = artificial_data.make_two_gaussian_data(1000, is_training_data=False, vaf=0.5, unlabeled_fraction=0.0)
        test_dataset = ReadSetDataset(data=test_data)
        test_loader = make_test_data_loader(test_dataset, BATCH_SIZE)
        model.learn_spectra(test_loader, NUM_SPECTRUM_ITERATIONS, summary_writer=summary_writer)

        events = EventAccumulator(tensorboard_dir)
        events.Reload()

        last = training_params.num_epochs - 1
        assert events.Scalars('Variant Sensitivity')[0].value > 0.98
        assert events.Scalars('Artifact Sensitivity')[0].value > 0.98


def test_wide_and_narrow_gaussian_data():
    data = artificial_data.make_wide_and_narrow_gaussian_data(10000)
    params = SMALL_MODEL_PARAMS
    training_params = TRAINING_PARAMS

    with tempfile.TemporaryDirectory() as tensorboard_dir:
        summary_writer = SummaryWriter(tensorboard_dir)
        model = train_model_and_write_summary(m3_params=params, training_params=training_params, data=data, summary_writer=summary_writer)

        test_data = artificial_data.make_two_gaussian_data(1000, is_training_data=False, vaf=0.25, unlabeled_fraction=0.0)
        test_dataset = ReadSetDataset(data=test_data)
        test_loader = make_test_data_loader(test_dataset, BATCH_SIZE)
        model.learn_spectra(test_loader, NUM_SPECTRUM_ITERATIONS, summary_writer=summary_writer)

        events = EventAccumulator(tensorboard_dir)
        events.Reload()

        last = training_params.num_epochs - 1
        assert events.Scalars('Variant Sensitivity')[0].value > 0.90
        assert events.Scalars('Artifact Sensitivity')[0].value > 0.90
