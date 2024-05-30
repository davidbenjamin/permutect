import tempfile
from argparse import Namespace
import torch

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch.utils.tensorboard import SummaryWriter

import permutect.parameters
from permutect import constants
from permutect.architecture import artifact_model
from permutect.tools import train_model
from permutect.architecture.artifact_model import load_artifact_model


def test_train_model():
    # Inputs
    training_data_tarfile = "/Users/davidben/permutect/just-dream-1/dream1-normal-small-train.tar"

    # Outputs
    saved_artifact_model = tempfile.NamedTemporaryFile()
    training_tensorboard_dir = tempfile.TemporaryDirectory()

    # STEP 2: train a model
    train_model_args = Namespace()
    setattr(train_model_args, constants.READ_EMBEDDING_DIMENSION_NAME, 18)
    setattr(train_model_args, constants.NUM_TRANSFORMER_HEADS_NAME, 3)
    setattr(train_model_args, constants.TRANSFORMER_HIDDEN_DIMENSION_NAME, 20)
    setattr(train_model_args, constants.NUM_TRANSFORMER_LAYERS_NAME, 2)
    setattr(train_model_args, constants.INFO_LAYERS_NAME, [20, 20])
    setattr(train_model_args, constants.AGGREGATION_LAYERS_NAME, [20, 20, 20])
    setattr(train_model_args, constants.CALIBRATION_LAYERS_NAME, [6,6])
    cnn_layer_strings = ['convolution/kernel_size=3/out_channels=64',
                     'pool/kernel_size=2',
                     'leaky_relu',
                     'flatten',
                     'linear/out_features=10']
    setattr(train_model_args, constants.REF_SEQ_LAYER_STRINGS_NAME, cnn_layer_strings)
    setattr(train_model_args, constants.DROPOUT_P_NAME, 0.0)
    setattr(train_model_args, constants.ALT_DOWNSAMPLE_NAME, 20)
    setattr(train_model_args, constants.LEARNING_RATE_NAME, 0.001)
    setattr(train_model_args, constants.WEIGHT_DECAY_NAME, 0.01)
    setattr(train_model_args, constants.BATCH_NORMALIZE_NAME, False)
    setattr(train_model_args, constants.LEARN_ARTIFACT_SPECTRA_NAME, True)  # could go either way
    setattr(train_model_args, constants.GENOMIC_SPAN_NAME, 100000)

    # Training data inputs
    setattr(train_model_args, constants.TRAIN_TAR_NAME, training_data_tarfile)

    # training hyperparameters
    setattr(train_model_args, constants.REWEIGHTING_RANGE_NAME, 0.3)
    setattr(train_model_args, constants.BATCH_SIZE_NAME, 64)
    setattr(train_model_args, constants.NUM_WORKERS_NAME, 2)
    setattr(train_model_args, constants.NUM_EPOCHS_NAME, 2)
    setattr(train_model_args, constants.NUM_CALIBRATION_EPOCHS_NAME, 1)

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
    hyperparams = permutect.parameters.ArtifactModelParameters(read_embedding_dimension=12, num_transformer_heads=3,
                                                               transformer_hidden_dimension=20, num_transformer_layers=2,
                                                               info_layers=[20, 20], aggregation_layers=[20, 20], calibration_layers=[6],
                                                               dropout_p=0.0, batch_normalize=False, learning_rate=0.001, weight_decay=0.01, alt_downsample=20)
    training_params = train_model.TrainingParameters(batch_size=64, num_epochs=5, num_calibration_epochs=2, reweighting_range=0.3)

    with tempfile.TemporaryDirectory() as tensorboard_dir:
        summary_writer = SummaryWriter(tensorboard_dir)
        train_model.train_artifact_model(hyperparams=hyperparams, training_datasets=[dataset], training_params=training_params, summary_writer=summary_writer)
        summary_writer.close()
        events = EventAccumulator(tensorboard_dir)
        events.Reload()
        h = 99


def test_on_dream1():
    test_training("/Users/davidben/permutect/just-dream-1/dream1-normal-medium-training.dataset")
