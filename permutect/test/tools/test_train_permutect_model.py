import tempfile
from argparse import Namespace

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from permutect import constants
from permutect.architecture.artifact_model import load_model
from permutect.tools import train_artifact_model
from permutect.test.test_file_names import *

MAKE_NEW_ARTIFACT_MODEL = False


def test_train_artifact_model():
    training_data_tarfile = PREPROCESSED_DATA
    #training_data_tarfile = "/Users/davidben/permutect/integration-tests/preprocessed-dataset.tar"
    saved_model = tempfile.NamedTemporaryFile() if not MAKE_NEW_ARTIFACT_MODEL else SMALL_ARTIFACT_MODEL
    training_tensorboard_dir = tempfile.TemporaryDirectory()

    train_model_args = Namespace()
    setattr(train_model_args, constants.READ_LAYERS_NAME, [10, 10, 10])
    setattr(train_model_args, constants.SELF_ATTENTION_HIDDEN_DIMENSION_NAME, 20)
    setattr(train_model_args, constants.NUM_SELF_ATTENTION_LAYERS_NAME, 2)
    setattr(train_model_args, constants.INFO_LAYERS_NAME, [10, 10])
    setattr(train_model_args, constants.AGGREGATION_LAYERS_NAME, [20, 20, 20])
    setattr(train_model_args, constants.NUM_ARTIFACT_CLUSTERS_NAME, 4)
    setattr(train_model_args, constants.CALIBRATION_LAYERS_NAME, [10, 10, 10])
    cnn_layer_strings = ['convolution/kernel_size=3/out_channels=64',
                     'pool/kernel_size=2',
                     'leaky_relu',
                     'flatten',
                     'linear/out_features=10']
    setattr(train_model_args, constants.REF_SEQ_LAYER_STRINGS_NAME, cnn_layer_strings)
    setattr(train_model_args, constants.DROPOUT_P_NAME, 0.0)
    setattr(train_model_args, constants.BATCH_NORMALIZE_NAME, False)

    # Training data inputs
    setattr(train_model_args, constants.TRAIN_TAR_NAME, training_data_tarfile)
    setattr(train_model_args, constants.PRETRAINED_ARTIFACT_MODEL_NAME, None)

    # training hyperparameters
    setattr(train_model_args, constants.REWEIGHTING_RANGE_NAME, 0.3)
    setattr(train_model_args, constants.BATCH_SIZE_NAME, 64)
    setattr(train_model_args, constants.INFERENCE_BATCH_SIZE_NAME, 64)
    setattr(train_model_args, constants.NUM_WORKERS_NAME, 2)
    setattr(train_model_args, constants.NUM_EPOCHS_NAME, 2)
    setattr(train_model_args, constants.NUM_CALIBRATION_EPOCHS_NAME, 0)
    setattr(train_model_args, constants.LEARNING_RATE_NAME, 0.001)
    setattr(train_model_args, constants.WEIGHT_DECAY_NAME, 0.01)

    # path to saved model
    setattr(train_model_args, constants.OUTPUT_NAME, saved_model if MAKE_NEW_ARTIFACT_MODEL else saved_model.name)
    setattr(train_model_args, constants.TENSORBOARD_DIR_NAME, training_tensorboard_dir.name)

    train_artifact_model.main_without_parsing(train_model_args)

    events = EventAccumulator(training_tensorboard_dir.name)
    events.Reload()

    loaded_model, _, _ = load_model(saved_model)

def main():
    test_train_artifact_model()

# this is necessary; otherwise running with multiple workers will create a weird multiprocessing error
if __name__ == '__main__':
    main()
