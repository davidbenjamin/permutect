import tempfile
from argparse import Namespace

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from permutect import constants
from permutect.architecture.artifact_model import load_model
from permutect.test.test_file_names import *
from permutect.tools import refine_artifact_model


def test_refine_artifact_model():
    # NOTE: EXPERIMENTAL_MODEL does not currently exist in the repo, so this
    # test will fail with FileNotFoundError until a compatible model is committed.
    saved_model = tempfile.NamedTemporaryFile()
    training_tensorboard_dir = tempfile.TemporaryDirectory()

    train_model_args = Namespace()
    setattr(train_model_args, constants.CALIBRATION_SOURCES_NAME, None)
    setattr(train_model_args, constants.LEARN_ARTIFACT_SPECTRA_NAME, False)
    setattr(train_model_args, constants.GENOMIC_SPAN_NAME, 100000)
    setattr(train_model_args, constants.TRAIN_TAR_NAME, PREPROCESSED_DATA)
    setattr(train_model_args, constants.PRETRAINED_ARTIFACT_MODEL_NAME, EXPERIMENTAL_MODEL)
    setattr(train_model_args, constants.TRAINABLE_PARAMETERS_NAME, ["WHOLE_MODEL"])
    setattr(train_model_args, constants.BATCH_SIZE_NAME, 64)
    setattr(train_model_args, constants.INFERENCE_BATCH_SIZE_NAME, 64)
    setattr(train_model_args, constants.NUM_WORKERS_NAME, 0)
    setattr(train_model_args, constants.NUM_EPOCHS_NAME, 2)
    setattr(train_model_args, constants.NUM_CALIBRATION_EPOCHS_NAME, 1)
    setattr(train_model_args, constants.LEARNING_RATE_NAME, 0.001)
    setattr(train_model_args, constants.WEIGHT_DECAY_NAME, 0.01)
    setattr(train_model_args, constants.OUTPUT_NAME, saved_model.name)
    setattr(train_model_args, constants.TENSORBOARD_DIR_NAME, training_tensorboard_dir.name)

    refine_artifact_model.main_without_parsing(train_model_args)

    events = EventAccumulator(training_tensorboard_dir.name)
    events.Reload()

    load_model(saved_model)
