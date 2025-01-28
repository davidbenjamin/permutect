import argparse

from torch.utils.tensorboard import SummaryWriter

from permutect import constants, utils
from permutect.architecture.artifact_model import ArtifactModel
from permutect.architecture.base_model import BaseModel
from permutect.architecture.model_training import learn_base_model
from permutect.architecture.model_io import load_models, save
from permutect.parameters import parse_training_params, parse_model_params, add_model_params_to_parser, add_training_params_to_parser
from permutect.data.base_dataset import BaseDataset


def main_without_parsing(args):
    params = parse_model_params(args)
    training_params = parse_training_params(args)

    tarfile_data = getattr(args, constants.TRAIN_TAR_NAME)
    saved_model_path = getattr(args, constants.SAVED_MODEL_NAME)    # optional pretrained model to use as initialization

    saved_base_model, saved_artifact_model, _, _ = (None, None, None, None) if saved_model_path is None else \
        load_models(saved_model_path)

    tensorboard_dir = getattr(args, constants.TENSORBOARD_DIR_NAME)
    summary_writer = SummaryWriter(tensorboard_dir)
    dataset = BaseDataset(data_tarfile=tarfile_data, num_folds=10)

    base_model = saved_base_model if (saved_base_model is not None) else \
            BaseModel(params=params, num_read_features=dataset.num_read_features, num_info_features=dataset.num_info_features,
                      ref_sequence_length=dataset.ref_sequence_length, device=utils.gpu_if_available())
    artifact_model = saved_artifact_model if (saved_artifact_model is not None) else \
        ArtifactModel(params=params, num_base_features=base_model.output_dimension(), device=utils.gpu_if_available())

    learn_base_model(base_model, artifact_model, dataset, training_params, summary_writer=summary_writer)
    summary_writer.close()
    save(path=getattr(args, constants.OUTPUT_NAME), base_model=base_model, artifact_model=artifact_model)


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