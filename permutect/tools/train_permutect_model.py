import argparse

from torch.utils.tensorboard import SummaryWriter

from permutect import constants, misc_utils
from permutect.architecture.permutect_model import PermutectModel, load_model
from permutect.architecture.model_training import train_permutect_model
from permutect.misc_utils import gpu_if_available
from permutect.parameters import parse_training_params, parse_model_params, add_model_params_to_parser, add_training_params_to_parser
from permutect.data.base_dataset import BaseDataset


def main_without_parsing(args):
    params = parse_model_params(args)
    training_params = parse_training_params(args)

    tarfile_data = getattr(args, constants.TRAIN_TAR_NAME)
    saved_model_path = getattr(args, constants.SAVED_MODEL_NAME)    # optional pretrained model to use as initialization

    saved_model, _, _ = (None, None, None) if saved_model_path is None else load_model(saved_model_path)

    tensorboard_dir = getattr(args, constants.TENSORBOARD_DIR_NAME)
    summary_writer = SummaryWriter(tensorboard_dir)
    dataset = BaseDataset(data_tarfile=tarfile_data, num_folds=10)

    model = saved_model if (saved_model is not None) else \
            PermutectModel(params=params, num_read_features=dataset.num_read_features, num_info_features=dataset.num_info_features,
                           ref_sequence_length=dataset.ref_sequence_length, device=gpu_if_available())

    train_permutect_model(model, dataset, training_params, summary_writer=summary_writer)
    summary_writer.close()

    # TODO: this is currently wrong because we are using the separate artifact model, not the full model
    model.save_model(path=getattr(args, constants.OUTPUT_NAME))


def parse_arguments():
    parser = argparse.ArgumentParser(description='train the Permutect read set representation model')
    add_model_params_to_parser(parser)
    add_training_params_to_parser(parser)

    parser.add_argument('--' + constants.TRAIN_TAR_NAME, type=str, required=True,
                        help='tarfile of training/validation datasets produced by preprocess_dataset.py')
    parser.add_argument('--' + constants.OUTPUT_NAME, type=str, required=True, help='output saved model file')
    parser.add_argument('--' + constants.TENSORBOARD_DIR_NAME, type=str, default='tensorboard', required=False,
                        help='output tensorboard directory')

    return parser.parse_args()

def main():
    args = parse_arguments()
    main_without_parsing(args)


if __name__ == '__main__':
    main()