import argparse

import torch
from torch.utils.tensorboard import SummaryWriter

from permutect import constants
from permutect.architecture.representation_model import RepresentationModel, LearningMethod, load_representation_model
from permutect.parameters import RepresentationModelParameters, TrainingParameters, parse_training_params, \
    parse_representation_model_params, add_representation_model_params_to_parser, add_training_params_to_parser
from permutect.data.read_set_dataset import ReadSetDataset


def train_representation_model(params: RepresentationModelParameters, training_params: TrainingParameters, summary_writer: SummaryWriter,
                               dataset: ReadSetDataset, pretrained_model: RepresentationModel = None) -> RepresentationModel:
    use_gpu = torch.cuda.is_available()
    device = torch.device('cuda' if use_gpu else 'cpu')

    use_pretrained = (pretrained_model is not None)
    model = pretrained_model if use_pretrained else \
        RepresentationModel(params=params, num_read_features=dataset.num_read_features, num_info_features=dataset.num_info_features,
                            ref_sequence_length=dataset.ref_sequence_length, device=device).float()

    print("Training. . .")
    model.learn(dataset, LearningMethod.SEMISUPERVISED, training_params, summary_writer=summary_writer)

    return model


def main_without_parsing(args):
    hyperparams = parse_representation_model_params(args)
    training_params = parse_training_params(args)

    tarfile_data = getattr(args, constants.TRAIN_TAR_NAME)
    pretrained_model_path = getattr(args, constants.PRETRAINED_MODEL_NAME)
    pretrained_model = None if pretrained_model_path is None else load_representation_model(pretrained_model_path)
    tensorboard_dir = getattr(args, constants.TENSORBOARD_DIR_NAME)
    summary_writer = SummaryWriter(tensorboard_dir)
    dataset = ReadSetDataset(data_tarfile=tarfile_data, num_folds=10)
    model = train_representation_model(params=hyperparams, dataset=dataset, training_params=training_params,
                                       summary_writer=summary_writer, pretrained_model=pretrained_model)

    summary_writer.close()
    model.save(getattr(args, constants.OUTPUT_NAME))


def main():
    args = parse_arguments()
    main_without_parsing(args)


if __name__ == '__main__':
    main()


def parse_arguments():
    parser = argparse.ArgumentParser(description='train the Permutect read set representation model')
    add_representation_model_params_to_parser(parser)
    add_training_params_to_parser(parser)

    parser.add_argument('--' + constants.TRAIN_TAR_NAME, type=str, required=True,
                        help='tarfile of training/validation datasets produced by preprocess_dataset.py')
    parser.add_argument('--' + constants.OUTPUT_NAME, type=str, required=True, help='output saved model file')
    parser.add_argument('--' + constants.TENSORBOARD_DIR_NAME, type=str, default='tensorboard', required=False,
                        help='output tensorboard directory')

    return parser.parse_args()