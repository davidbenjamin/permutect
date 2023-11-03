import tempfile
from argparse import Namespace
import torch

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch.utils.tensorboard import SummaryWriter

from mutect3 import constants
from mutect3.architecture import artifact_model
from mutect3.tools import train_model
from mutect3.tools.filter_variants import load_artifact_model


def test_train_model():
    # Inputs
    training_data_tarfile = "/Users/davidben/mutect3/just-dream-1/dream1-normal-small-train.tar"
    artifact_posterior_tarfile = "/Users/davidben/mutect3/just-dream-1/dream1-normal-small-artifact.tar"

    # Outputs
    saved_artifact_model = tempfile.NamedTemporaryFile()
    training_tensorboard_dir = tempfile.TemporaryDirectory()

    # STEP 2: train a model
    train_model_args = Namespace()
    setattr(train_model_args, constants.READ_LAYERS_NAME, [20, 20])
    setattr(train_model_args, constants.INFO_LAYERS_NAME, [20, 20])
    setattr(train_model_args, constants.AGGREGATION_LAYERS_NAME, [20, 20, 20])
    cnn_layer_strings = ['convolution/kernel_size=3/out_channels=64',
                     'pool/kernel_size=2',
                     'leaky_relu',
                     'flatten',
                     'linear/out_features=10']
    setattr(train_model_args, constants.REF_SEQ_LAYER_STRINGS_NAME, cnn_layer_strings)
    setattr(train_model_args, constants.DROPOUT_P_NAME, 0.0)
    setattr(train_model_args, constants.LEARNING_RATE_NAME, 0.001)
    setattr(train_model_args, constants.BATCH_NORMALIZE_NAME, False)
    setattr(train_model_args, constants.LEARN_ARTIFACT_SPECTRA_NAME, True)  # could go either way
    setattr(train_model_args, constants.GENOMIC_SPAN_NAME, 100000)

    # Training data inputs
    setattr(train_model_args, constants.TRAIN_TAR_NAME, training_data_tarfile)
    setattr(train_model_args, constants.ARTIFACT_TAR_NAME, artifact_posterior_tarfile)

    # training hyperparameters
    setattr(train_model_args, constants.REWEIGHTING_RANGE_NAME, 0.3)
    setattr(train_model_args, constants.BATCH_SIZE_NAME, 64)
    setattr(train_model_args, constants.NUM_WORKERS_NAME, 2)
    setattr(train_model_args, constants.NUM_EPOCHS_NAME, 2)

    # path to saved model
    setattr(train_model_args, constants.OUTPUT_NAME, saved_artifact_model.name)
    setattr(train_model_args, constants.TENSORBOARD_DIR_NAME, training_tensorboard_dir.name)

    train_model.main_without_parsing(train_model_args)

    events = EventAccumulator(training_tensorboard_dir.name)
    events.Reload()

    loaded_artifact_model, artifact_log_priors, artifact_spectra_state_dict = load_artifact_model(saved_artifact_model)
    assert artifact_log_priors is not None
    assert artifact_spectra_state_dict is not None

    saved = torch.load(saved_artifact_model)
    assert constants.ARTIFACT_LOG_PRIORS_NAME in saved
    assert constants.ARTIFACT_SPECTRA_STATE_DICT_NAME in saved

    print(artifact_log_priors)
    h = 99


















def test_training(dataset):
    m3_params = artifact_model.ArtifactModelParameters(read_layers=[20, 20, 20], info_layers=[20, 20], aggregation_layers=[20, 20],
                                                       dropout_p=0.0, batch_normalize=False, learning_rate=0.001)
    training_params = train_model.TrainingParameters(batch_size=64, num_epochs=5, reweighting_range=0.3)

    with tempfile.TemporaryDirectory() as tensorboard_dir:
        summary_writer = SummaryWriter(tensorboard_dir)
        train_model.train_artifact_model(m3_params=m3_params, training_datasets=[dataset], params=training_params, summary_writer=summary_writer)
        summary_writer.close()
        events = EventAccumulator(tensorboard_dir)
        events.Reload()
        h = 99


def test_on_dream1():
    test_training("/Users/davidben/mutect3/just-dream-1/dream1-normal-medium-training.dataset")
