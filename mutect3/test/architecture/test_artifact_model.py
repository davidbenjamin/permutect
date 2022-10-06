from mutect3.test.test_utils import artificial_data
from mutect3.data.read_set_dataset import ReadSetDataset, BigReadSetDataset, make_semisupervised_data_loader, make_test_data_loader
from mutect3.data.read_set import ReadSet
from typing import Iterable
from mutect3.architecture.artifact_model import ArtifactModel, ArtifactModelParameters
from mutect3 import utils
from mutect3.tools.train_model import TrainingParameters

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import tempfile
from torch.utils.tensorboard import SummaryWriter

BATCH_SIZE = 64
NUM_EPOCHS = 100
NUM_SPECTRUM_ITERATIONS = 100
TRAINING_PARAMS = TrainingParameters(batch_size=BATCH_SIZE, num_epochs=NUM_EPOCHS, reweighting_range=0.3)

SMALL_MODEL_PARAMS = ArtifactModelParameters(read_layers=[5, 5], info_layers=[5, 5], aggregation_layers=[5, 5, 5, 5],
                                             dropout_p=0.2, batch_normalize=False, learning_rate=0.001)


# Note that the test methods in this class also cover batching, samplers, datasets, and data loaders
def train_model_and_write_summary(m3_params: ArtifactModelParameters, training_params: TrainingParameters,
                                  data: Iterable[ReadSet], summary_writer: SummaryWriter = None):
    dataset = ReadSetDataset(data=data)
    big_dataset = BigReadSetDataset(batch_size=training_params.batch_size, dataset=dataset)
    model = ArtifactModel(params=m3_params, num_read_features=dataset.num_read_features()).float()

    model.train_model(big_dataset, training_params.num_epochs, summary_writer=summary_writer,
                      reweighting_range=training_params.reweighting_range, m3_params=m3_params)
    model.learn_calibration(big_dataset.generate_batches(utils.Epoch.VALID), num_epochs=50)
    model.evaluate_model_after_training({"training": big_dataset.generate_batches(utils.Epoch.TRAIN)}, summary_writer, "training data: ")
    return model


def test_separate_gaussian_data():
    # in the test for alt count agnostic, we make training data where variant alt counts are much larger than artifact
    # alt counts and test data with a low alt allele fraction
    for test_alt_fraction_agnostic in (False, True):
        data = artificial_data.make_two_gaussian_data(1000) if not test_alt_fraction_agnostic else \
            artificial_data.make_two_gaussian_data(1000, vaf=0.5, downsample_variants_to_match_artifacts=False, alt_downsampling=20)
        params = SMALL_MODEL_PARAMS
        training_params = TRAINING_PARAMS

        with tempfile.TemporaryDirectory() as tensorboard_dir:
            summary_writer = SummaryWriter(tensorboard_dir)
            model = train_model_and_write_summary(m3_params=params, training_params=training_params, data=data, summary_writer=summary_writer)

            # TODO: migrate this old stuff to test for PosteriorModel
            # test_vaf = 0.05 if test_alt_fraction_agnostic else 0.5
            # test_data = artificial_data.make_two_gaussian_data(1000, is_training_data=False, vaf=test_vaf, unlabeled_fraction=0.0)
            # test_dataset = ReadSetDataset(data=test_data)
            # test_loader = make_test_data_loader(test_dataset, BATCH_SIZE)
            # model.learn_spectra(test_loader, NUM_SPECTRUM_ITERATIONS, summary_writer=summary_writer)

            events = EventAccumulator(tensorboard_dir)
            events.Reload()

            # TODO: these have been replaced with images, so it's not so simple to check the output quality from the tensorboard
            # TODO: for now I can put in a breakpoint and manually run tensorboard --logdir <tensorboard temp directory>
            # TODO: to spot check the figures
            # assert events.Scalars('Variant Sensitivity')[0].value > 0.98
            # assert events.Scalars('Artifact Sensitivity')[0].value > 0.98


def test_wide_and_narrow_gaussian_data():
    data = artificial_data.make_wide_and_narrow_gaussian_data(10000)
    params = SMALL_MODEL_PARAMS
    training_params = TRAINING_PARAMS

    with tempfile.TemporaryDirectory() as tensorboard_dir:
        summary_writer = SummaryWriter(tensorboard_dir)
        model = train_model_and_write_summary(m3_params=params, training_params=training_params, data=data, summary_writer=summary_writer)

        events = EventAccumulator(tensorboard_dir)
        events.Reload()


# TODO: this test currently fails -- almost everything is considered an artifact
# TODO: I must investigate
def test_strand_bias_data():
    data = artificial_data.make_random_strand_bias_data(1000, is_training_data=True)
    params = SMALL_MODEL_PARAMS # TODO: change!!!!!!!
    training_params = TRAINING_PARAMS

    with tempfile.TemporaryDirectory() as tensorboard_dir:
        summary_writer = SummaryWriter(tensorboard_dir)
        model = train_model_and_write_summary(m3_params=params, training_params=training_params, data=data, summary_writer=summary_writer)

        test_data = artificial_data.make_random_strand_bias_data(1000, is_training_data=False, vaf=0.25, unlabeled_fraction=0.0)
        test_dataset = ReadSetDataset(data=test_data)
        test_loader = make_test_data_loader(test_dataset, BATCH_SIZE)
        model.learn_spectra(test_loader, NUM_SPECTRUM_ITERATIONS, summary_writer=summary_writer)

        events = EventAccumulator(tensorboard_dir)
        events.Reload()

        assert events.Scalars('Variant Sensitivity')[0].value > 0.90
        assert events.Scalars('Artifact Sensitivity')[0].value > 0.90
