import argparse
import tempfile

import torch
from torch.utils.tensorboard import SummaryWriter

from mutect3.architecture.artifact_model import ArtifactModelParameters, ArtifactModel
from mutect3 import utils, constants
from mutect3.data import read_set_dataset
from mutect3.utils import Label


class TrainingParameters:
    def __init__(self, batch_size, num_epochs, reweighting_range: float):
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.reweighting_range = reweighting_range


def train_artifact_model(m3_params: ArtifactModelParameters, training_datasets, params: TrainingParameters, tensorboard_dir):
    use_gpu = torch.cuda.is_available()
    device = torch.device('cuda' if use_gpu else 'cpu')
    big_dataset = read_set_dataset.BigReadSetDataset(batch_size=params.batch_size, dataset_files=training_datasets)
    model = ArtifactModel(params=m3_params, num_read_features=big_dataset.num_read_features, device=device).float()

    print("Training. . .")
    summary_writer = SummaryWriter(tensorboard_dir)
    model.train_model(big_dataset, params.num_epochs, summary_writer=summary_writer, reweighting_range=params.reweighting_range, m3_params=m3_params)

    print("Calibrating. . .")
    model.learn_calibration(big_dataset.generate_batches(utils.Epoch.VALID), num_epochs=50)

    print("Evaluating trained model. . .")
    model.evaluate_model_after_training({"training": big_dataset.generate_batches(utils.Epoch.TRAIN), "validation": big_dataset.generate_batches(utils.Epoch.VALID)}, summary_writer)
    # model.evaluate_model_after_training(valid_loader, summary_writer, "validation data for (1,25) and (50,50): ",
    #                                    artifact_beta_shape=torch.Tensor((1, 25)), variant_beta_shape=torch.Tensor((50, 50)))
    # model.evaluate_model_after_training(valid_loader, summary_writer, "validation data for (1,25) and (25,25): ",
    #                                    artifact_beta_shape=torch.Tensor((1, 25)), variant_beta_shape=torch.Tensor((25, 25)))
    # model.evaluate_model_after_training(valid_loader, summary_writer, "validation data for (1,50) and (50,50): ",
    #                                    artifact_beta_shape=torch.Tensor((1, 50)), variant_beta_shape=torch.Tensor((50, 50)))
    summary_writer.close()

    return model


def save_artifact_model(model, m3_params, path):
    torch.save({
        constants.STATE_DICT_NAME: model.state_dict(),
        constants.M3_PARAMS_NAME: m3_params,
        constants.NUM_READ_FEATURES_NAME: model.num_read_features()
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
