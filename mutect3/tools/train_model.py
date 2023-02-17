import argparse

import torch
from torch.utils.tensorboard import SummaryWriter

from mutect3.architecture.artifact_model import ArtifactModelParameters, ArtifactModel
from mutect3 import constants
from mutect3.data.read_set_dataset import ReadSetDataset


class TrainingParameters:
    def __init__(self, batch_size, num_epochs, num_refless_epochs, reweighting_range: float, num_workers: int=0):
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.num_refless_epochs = num_refless_epochs
        self.reweighting_range = reweighting_range
        self.num_workers = num_workers


def train_artifact_model(m3_params: ArtifactModelParameters, params: TrainingParameters, tensorboard_dir, data_tarfile):
    use_gpu = torch.cuda.is_available()
    device = torch.device('cuda' if use_gpu else 'cpu')

    dataset = ReadSetDataset(data_tarfile=data_tarfile, validation_fraction=0.1)

    model = ArtifactModel(params=m3_params, num_read_features=dataset.num_read_features,
                          num_info_features=dataset.num_info_features, ref_sequence_length=dataset.ref_sequence_length, device=device).float()

    print("Training. . .")
    summary_writer = SummaryWriter(tensorboard_dir)
    model.train_model(dataset, params.num_epochs, params.batch_size, params.num_workers, summary_writer=summary_writer, reweighting_range=params.reweighting_range, m3_params=m3_params)

    model.train_model(dataset, params.num_refless_epochs, params.batch_size, params.num_workers, summary_writer=summary_writer,
                      reweighting_range=params.reweighting_range, m3_params=m3_params, use_ref_reads=False)

    print("Calibrating. . .")
    temp_fig_before, temp_curve_before = model.calibration.plot_temperature("Count-Dependent Calibration Before")
    model.learn_calibration(dataset, num_epochs=50, batch_size=params.batch_size, num_workers=params.num_workers)
    temp_fig_after, temp_curve_after = model.calibration.plot_temperature("Count-Dependent Calibration After")
    summary_writer.add_figure("calibration before", temp_fig_before)
    summary_writer.add_figure("calibration after", temp_fig_after)

    print("Evaluating trained model. . .")
    model.evaluate_model_after_training(dataset, params.batch_size, params.num_workers, summary_writer)

    summary_writer.close()

    return model


def save_artifact_model(model, m3_params, path):
    torch.save({
        constants.STATE_DICT_NAME: model.state_dict(),
        constants.M3_PARAMS_NAME: m3_params,
        constants.NUM_READ_FEATURES_NAME: model.num_read_features(),
        constants.NUM_INFO_FEATURES_NAME: model.num_info_features(),
        constants.REF_SEQUENCE_LENGTH_NAME: model.ref_sequence_length()
    }, path)


def parse_training_params(args) -> TrainingParameters:
    reweighting_range = getattr(args, constants.REWEIGHTING_RANGE_NAME)
    batch_size = getattr(args, constants.BATCH_SIZE_NAME)
    num_epochs = getattr(args, constants.NUM_EPOCHS_NAME)
    num_refless_epochs = getattr(args, constants.NUM_REFLESS_EPOCHS_NAME)
    num_workers = getattr(args, constants.NUM_WORKERS_NAME)
    return TrainingParameters(batch_size, num_epochs, num_refless_epochs, reweighting_range, num_workers=num_workers)


def parse_mutect3_params(args) -> ArtifactModelParameters:
    read_layers = getattr(args, constants.READ_LAYERS_NAME)
    info_layers = getattr(args, constants.INFO_LAYERS_NAME)
    aggregation_layers = getattr(args, constants.AGGREGATION_LAYERS_NAME)
    ref_seq_layer_strings = getattr(args, constants.REF_SEQ_LAYER_STRINGS_NAME)
    dropout_p = getattr(args, constants.DROPOUT_P_NAME)
    batch_normalize = getattr(args, constants.BATCH_NORMALIZE_NAME)
    learning_rate = getattr(args, constants.LEARNING_RATE_NAME)
    return ArtifactModelParameters(read_layers, info_layers, aggregation_layers, ref_seq_layer_strings, dropout_p, batch_normalize, learning_rate)


def parse_arguments():
    parser = argparse.ArgumentParser()

    # architecture hyperparameters
    parser.add_argument('--' + constants.READ_LAYERS_NAME, nargs='+', type=int, required=True)
    parser.add_argument('--' + constants.INFO_LAYERS_NAME, nargs='+', type=int, required=True)
    parser.add_argument('--' + constants.AGGREGATION_LAYERS_NAME, nargs='+', type=int, required=True)
    parser.add_argument('--' + constants.REF_SEQ_LAYER_STRINGS_NAME, nargs='+', type=str, required=True)
    parser.add_argument('--' + constants.DROPOUT_P_NAME, type=float, default=0.0, required=False)
    parser.add_argument('--' + constants.LEARNING_RATE_NAME, type=float, default=0.001, required=False)
    parser.add_argument('--' + constants.BATCH_NORMALIZE_NAME, action='store_true')

    # Training data inputs
    parser.add_argument('--' + constants.TRAIN_TAR_NAME, type=str, required=True)

    # training hyperparameters
    parser.add_argument('--' + constants.REWEIGHTING_RANGE_NAME, type=float, default=0.3, required=False)
    parser.add_argument('--' + constants.BATCH_SIZE_NAME, type=int, default=64, required=False)
    parser.add_argument('--' + constants.NUM_WORKERS_NAME, type=int, default=0, required=False)
    parser.add_argument('--' + constants.NUM_EPOCHS_NAME, type=int, required=True)
    parser.add_argument('--' + constants.NUM_REFLESS_EPOCHS_NAME, type=int, required=True)

    # path to saved model
    parser.add_argument('--' + constants.OUTPUT_NAME, type=str, required=True)
    parser.add_argument('--' + constants.TENSORBOARD_DIR_NAME, type=str, default='tensorboard', required=False)

    return parser.parse_args()


def main_without_parsing(args):
    m3_params = parse_mutect3_params(args)
    training_params = parse_training_params(args)

    tarfile_data = getattr(args, constants.TRAIN_TAR_NAME)
    model = train_artifact_model(m3_params=m3_params, data_tarfile=tarfile_data,
                                 params=training_params, tensorboard_dir=getattr(args, constants.TENSORBOARD_DIR_NAME))
    save_artifact_model(model, m3_params, getattr(args, constants.OUTPUT_NAME))


def main():
    args = parse_arguments()
    main_without_parsing(args)


if __name__ == '__main__':
    main()
