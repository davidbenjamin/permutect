import argparse

import torch
from matplotlib.backends.backend_pdf import PdfPages
from torch.distributions.beta import Beta

from mutect3.architecture.read_set_classifier import Mutect3Parameters, ReadSetClassifier
from mutect3 import utils, constants
from mutect3.data import read_set_dataset


class TrainingParameters:
    def __init__(self, batch_size, num_epochs, beta1: Beta, beta2: Beta):
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.beta1 = beta1
        self.beta2 = beta2


def train_m3_model(m3_params: Mutect3Parameters, training_datasets, params: TrainingParameters, tensorboard_dir):
    print("Loading datasets")
    train_and_valid = read_set_dataset.ReadSetDataset(files=training_datasets)
    training, valid = utils.split_dataset_into_train_and_valid(train_and_valid, 0.9)

    unlabeled_count = sum([1 for datum in train_and_valid if datum.label() == "UNLABELED"])
    print("Unlabeled data: " + str(unlabeled_count) + ", labeled data: " + str(len(train_and_valid) - unlabeled_count))
    print("Dataset sizes -- training: " + str(len(training)) + ", validation: " + str(len(valid)))

    train_loader = read_set_dataset.make_semisupervised_data_loader(training, params.batch_size)
    valid_loader = read_set_dataset.make_semisupervised_data_loader(valid, params.batch_size)
    model = ReadSetClassifier(m3_params=m3_params, na_model=None).float()

    print("Training model")
    training_metrics = model.train_model(train_loader, valid_loader, params.num_epochs, params.beta1, params.beta2)
    calibration_metrics = model.learn_calibration(valid_loader, num_epochs=50)

    # TODO: tensorboard dir is not pdf!!!!!
        #with PdfPages(tensorboard_dir) as pdf:
        #    for fig, curve in training_metrics.plot_curves():
        #        pdf.savefig(fig)
        #    for fig, curve in calibration_metrics.plot_curves():
        #        pdf.savefig(fig)
    # TODO: done with TODO

    return model


def save_m3_model(model, m3_params, path):
    torch.save({
        constants.STATE_DICT_NAME: model.state_dict(),
        constants.M3_PARAMS_NAME: m3_params
    }, path)


def parse_training_params(args) -> TrainingParameters:
    beta1 = Beta(getattr(args, constants.ALPHA1_NAME), getattr(args, constants.BETA1_NAME))
    beta2 = Beta(getattr(args, constants.ALPHA2_NAME), getattr(args, constants.BETA2_NAME))
    batch_size = getattr(args, constants.BATCH_SIZE_NAME)
    num_epochs = getattr(args, constants.NUM_EPOCHS_NAME)
    return TrainingParameters(batch_size, num_epochs, beta1, beta2)


def parse_mutect3_params(args) -> Mutect3Parameters:
    hidden_read_layers = getattr(args, constants.HIDDEN_READ_LAYERS_NAME)
    hidden_info_layers = getattr(args, constants.HIDDEN_INFO_LAYERS_NAME)
    aggregation_layers = getattr(args, constants.AGGREGATION_LAYERS_NAME)
    output_layers = getattr(args, constants.OUTPUT_LAYERS_NAME)
    dropout_p = getattr(args, constants.DROPOUT_P_NAME)
    return Mutect3Parameters(hidden_read_layers, hidden_info_layers, aggregation_layers, output_layers, dropout_p)


def parse_arguments():
    parser = argparse.ArgumentParser()

    # architecture hyperparameters
    parser.add_argument('--' + constants.HIDDEN_READ_LAYERS_NAME, nargs='+', type=int, required=True)
    parser.add_argument('--' + constants.HIDDEN_INFO_LAYERS_NAME, nargs='+', type=int, required=True)
    parser.add_argument('--' + constants.AGGREGATION_LAYERS_NAME, nargs='+', type=int, required=True)
    parser.add_argument('--' + constants.OUTPUT_LAYERS_NAME, nargs='+', type=int, required=True)
    parser.add_argument('--' + constants.DROPOUT_P_NAME, type=float, default=0.0, required=False)

    # Training data inputs
    parser.add_argument('--' + constants.TRAINING_DATASETS_NAME, nargs='+', type=str, required=True)

    # training hyperparameters
    parser.add_argument('--' + constants.ALPHA1_NAME, type=float, default=5.0, required=False)
    parser.add_argument('--' + constants.BETA1_NAME, type=float, default=1.0, required=False)
    parser.add_argument('--' + constants.ALPHA2_NAME, type=float, default=5.0, required=False)
    parser.add_argument('--' + constants.BETA2_NAME, type=float, default=1.0, required=False)
    parser.add_argument('--' + constants.BATCH_SIZE_NAME, type=int, default=64, required=False)
    parser.add_argument('--' + constants.NUM_EPOCHS_NAME, type=int, required=True)

    # path to saved model
    parser.add_argument('--' + constants.OUTPUT_NAME, type=str, required=True)
    parser.add_argument('--' + constants.TENSORBOARD_DIR_NAME, type=str, default='tensorboard', required=False)

    return parser.parse_args()


def main():
    args = parse_arguments()
    m3_params = parse_mutect3_params(args)
    training_params = parse_training_params(args)
    model = train_m3_model(m3_params, getattr(args, constants.TRAINING_DATASETS_NAME), training_params,
                           getattr(args, constants.TENSORBOARD_DIR_NAME))
    save_m3_model(model, m3_params, getattr(args, constants.OUTPUT_NAME))


if __name__ == '__main__':
    main()
