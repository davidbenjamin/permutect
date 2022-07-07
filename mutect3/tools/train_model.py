import argparse

import torch
from torch.utils.tensorboard import SummaryWriter

from mutect3.architecture.artifact_model import ArtifactModelParameters, ArtifactModel
from mutect3 import utils, constants
from mutect3.data import read_set_dataset


class TrainingParameters:
    def __init__(self, batch_size, num_epochs, reweighting_range: float):
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.reweighting_range = reweighting_range


def train_artifact_model(m3_params: ArtifactModelParameters, training_datasets, params: TrainingParameters, tensorboard_dir):
    print("Loading datasets")
    train_and_valid = read_set_dataset.ReadSetDataset(files=training_datasets)
    training, valid = utils.split_dataset_into_train_and_valid(train_and_valid, 0.9)

    unlabeled_count = sum([1 for datum in train_and_valid if datum.label() == "UNLABELED"])
    print("Unlabeled data: " + str(unlabeled_count) + ", labeled data: " + str(len(train_and_valid) - unlabeled_count))
    print("Dataset sizes -- training: " + str(len(training)) + ", validation: " + str(len(valid)))

    use_gpu = torch.cuda.is_available()
    device = torch.device('cuda' if use_gpu else 'cpu')

    train_loader = read_set_dataset.make_semisupervised_data_loader(training, params.batch_size, pin_memory=use_gpu)
    valid_loader = read_set_dataset.make_semisupervised_data_loader(valid, params.batch_size, pin_memory=use_gpu)

    model = ArtifactModel(params=m3_params, device=device).float()

    print("Training model. . .")
    summary_writer = SummaryWriter(tensorboard_dir)
    model.train_model(train_loader, valid_loader, params.num_epochs, summary_writer=summary_writer,
                      reweighting_range=params.reweighting_range, m3_params=m3_params)
    print("Training complete.  Calibrating. . .")
    model.learn_calibration(valid_loader, num_epochs=50)
    print("Calibration complete.  Evaluating trained model. . .")
    model.evaluate_model_after_training(train_loader, summary_writer, "training data: ")
    model.evaluate_model_after_training(valid_loader, summary_writer, "validation data: ")
    summary_writer.close()

    return model


def save_artifact_model(model, m3_params, path):
    torch.save({
        constants.STATE_DICT_NAME: model.state_dict(),
        constants.M3_PARAMS_NAME: m3_params
    }, path)


def parse_training_params(args) -> TrainingParameters:
    reweighting_range = getattr(args, constants.REWEIGHTING_RANGE_NAME)
    batch_size = getattr(args, constants.BATCH_SIZE_NAME)
    num_epochs = getattr(args, constants.NUM_EPOCHS_NAME)
    return TrainingParameters(batch_size, num_epochs, reweighting_range)


def parse_mutect3_params(args) -> ArtifactModelParameters:
    read_layers = getattr(args, constants.READ_LAYERS_NAME)
    info_layers = getattr(args, constants.INFO_LAYERS_NAME)
    aggregation_layers = getattr(args, constants.AGGREGATION_LAYERS_NAME)
    dropout_p = getattr(args, constants.DROPOUT_P_NAME)
    batch_normalize = getattr(args, constants.BATCH_NORMALIZE_NAME)
    learning_rate = getattr(args, constants.LEARNING_RATE_NAME)
    return ArtifactModelParameters(read_layers, info_layers, aggregation_layers, dropout_p, batch_normalize, learning_rate)


def parse_arguments():
    parser = argparse.ArgumentParser()

    # architecture hyperparameters
    parser.add_argument('--' + constants.READ_LAYERS_NAME, nargs='+', type=int, required=True)
    parser.add_argument('--' + constants.INFO_LAYERS_NAME, nargs='+', type=int, required=True)
    parser.add_argument('--' + constants.AGGREGATION_LAYERS_NAME, nargs='+', type=int, required=True)
    parser.add_argument('--' + constants.DROPOUT_P_NAME, type=float, default=0.0, required=False)
    parser.add_argument('--' + constants.LEARNING_RATE_NAME, type=float, default=0.001, required=False)
    parser.add_argument('--' + constants.BATCH_NORMALIZE_NAME, action='store_true')

    # Training data inputs
    parser.add_argument('--' + constants.TRAINING_DATASETS_NAME, nargs='+', type=str, required=True)

    # training hyperparameters
    parser.add_argument('--' + constants.REWEIGHTING_RANGE_NAME, type=float, default=0.3, required=False)
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
    model = train_artifact_model(m3_params, getattr(args, constants.TRAINING_DATASETS_NAME), training_params,
                                 getattr(args, constants.TENSORBOARD_DIR_NAME))
    save_artifact_model(model, m3_params, getattr(args, constants.OUTPUT_NAME))


if __name__ == '__main__':
    main()
