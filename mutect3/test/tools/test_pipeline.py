from argparse import Namespace
import tempfile

from mutect3.tools import preprocess_dataset, train_model
from mutect3 import constants


def test_on_dream1():

    # STEP 1: preprocess the plain text training dataset yielding a training tarfile
    training_datasets = ["/Users/davidben/mutect3/just-dream-1/dream1-normal-small-training.dataset"]
    training_data_tarfile = tempfile.NamedTemporaryFile()
    saved_artifact_model = tempfile.NamedTemporaryFile()
    training_tensorboard_dir = tempfile.TemporaryDirectory()

    preprocess_args = Namespace()
    setattr(preprocess_args, constants.CHUNK_SIZE_NAME, 1e6)
    setattr(preprocess_args, constants.TRAINING_DATASETS_NAME, training_datasets)
    setattr(preprocess_args, constants.OUTPUT_NAME, training_data_tarfile.name)
    preprocess_dataset.main(preprocess_args)

    # STEP 2: train a model
    train_model_args = Namespace()
    setattr(train_model_args, constants.READ_LAYERS_NAME, [30, 30, 30])
    setattr(train_model_args, constants.INFO_LAYERS_NAME, [30, 30, 30])
    setattr(train_model_args, constants.AGGREGATION_LAYERS_NAME, [30, 30, 30, 30])
    cnn_layer_strings = ['convolution/kernel_size=3/out_channels=64',
                     'pool/kernel_size=2',
                     'leaky_relu',
                     'convolution/kernel_size=3/dilation=2/out_channels=5',
                     'leaky_relu',
                     'flatten',
                     'linear/out_features=10']
    setattr(train_model_args, constants.REF_SEQ_LAYER_STRINGS_NAME, cnn_layer_strings)
    setattr(train_model_args, constants.DROPOUT_P_NAME, 0.0)
    setattr(train_model_args, constants.LEARNING_RATE_NAME, 0.001)
    setattr(train_model_args, constants.BATCH_NORMALIZE_NAME, False)

    # Training data inputs
    setattr(train_model_args, constants.TRAIN_TAR_NAME, training_data_tarfile.name)

    # training hyperparameters
    setattr(train_model_args, constants.REWEIGHTING_RANGE_NAME, 0.3)
    setattr(train_model_args, constants.BATCH_SIZE_NAME, 64)
    setattr(train_model_args, constants.NUM_WORKERS_NAME, 2)
    setattr(train_model_args, constants.NUM_EPOCHS_NAME, 10)

    # path to saved model
    setattr(train_model_args, constants.OUTPUT_NAME, saved_artifact_model.name)
    setattr(train_model_args, constants.TENSORBOARD_DIR_NAME, training_tensorboard_dir.name)

    train_model.main(train_model_args)
    k = 9