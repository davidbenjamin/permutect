import argparse

import torch
from torch.utils.tensorboard import SummaryWriter

from permutect import constants
from permutect.architecture.base_model import BaseModel, LearningMethod, load_base_model
from permutect.parameters import BaseModelParameters, TrainingParameters, parse_training_params, \
    parse_base_model_params, add_base_model_params_to_parser, add_training_params_to_parser
from permutect.data.base_dataset import BaseDataset


def train_base_model(params: BaseModelParameters, training_params: TrainingParameters, summary_writer: SummaryWriter,
                     dataset: BaseDataset, pretrained_model: BaseModel = None) -> BaseModel:
    use_gpu = torch.cuda.is_available()
    device = torch.device('cuda' if use_gpu else 'cpu')

    use_pretrained = (pretrained_model is not None)
    model = pretrained_model if use_pretrained else \
        BaseModel(params=params, num_read_features=dataset.num_read_features, num_info_features=dataset.num_info_features,
                  ref_sequence_length=dataset.ref_sequence_length, device=device).float()

    print("Training. . .")
    model.learn(dataset, LearningMethod.SEMISUPERVISED, training_params, summary_writer=summary_writer)

    return model


def main_without_parsing(args):
    hyperparams = parse_base_model_params(args)
    training_params = parse_training_params(args)

    tarfile_data = getattr(args, constants.TRAIN_TAR_NAME)
    pretrained_model_path = getattr(args, constants.PRETRAINED_MODEL_NAME)
    pretrained_model = None if pretrained_model_path is None else load_base_model(pretrained_model_path)
    tensorboard_dir = getattr(args, constants.TENSORBOARD_DIR_NAME)
    summary_writer = SummaryWriter(tensorboard_dir)
    dataset = BaseDataset(data_tarfile=tarfile_data, num_folds=10)
    model = train_base_model(params=hyperparams, dataset=dataset, training_params=training_params,
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
    add_base_model_params_to_parser(parser)
    add_training_params_to_parser(parser)

    parser.add_argument('--' + constants.TRAIN_TAR_NAME, type=str, required=True,
                        help='tarfile of training/validation datasets produced by preprocess_dataset.py')
    parser.add_argument('--' + constants.OUTPUT_NAME, type=str, required=True, help='output saved model file')
    parser.add_argument('--' + constants.TENSORBOARD_DIR_NAME, type=str, default='tensorboard', required=False,
                        help='output tensorboard directory')

    return parser.parse_args()