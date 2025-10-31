import argparse

from torch.utils.tensorboard import SummaryWriter

from permutect import constants
from permutect.architecture.artifact_model import ArtifactModel, load_model
from permutect.data.memory_mapped_data import MemoryMappedData
from permutect.training.model_training import train_artifact_model
from permutect.misc_utils import gpu_if_available, report_memory_usage
from permutect.parameters import parse_training_params, parse_model_params, add_model_params_to_parser, add_training_params_to_parser
from permutect.data.reads_dataset import ReadsDataset


def main_without_parsing(args):
    params = parse_model_params(args)
    training_params = parse_training_params(args)
    pretrained_model_path = getattr(args, constants.PRETRAINED_ARTIFACT_MODEL_NAME)    # optional pretrained model to use as initialization

    pretrained_model, _, _ = (None, None, None) if pretrained_model_path is None else load_model(pretrained_model_path)

    tensorboard_dir = getattr(args, constants.TENSORBOARD_DIR_NAME)
    summary_writer = SummaryWriter(tensorboard_dir)
    report_memory_usage("Training data about to be loaded from tarfile.")
    memory_mapped_data = MemoryMappedData.load_from_tarfile(getattr(args, constants.TRAIN_TAR_NAME))
    dataset = ReadsDataset(memory_mapped_data=memory_mapped_data, num_folds=10)

    model = pretrained_model if (pretrained_model is not None) else \
            ArtifactModel(params=params, num_read_features=dataset.num_read_features, num_info_features=dataset.num_info_features,
                          haplotypes_length=dataset.haplotypes_length, device=gpu_if_available())

    train_artifact_model(model, dataset, training_params, summary_writer=summary_writer, epochs_per_evaluation=10)
    summary_writer.close()

    # TODO: this is currently wrong because we are using the separate artifact model, not the full model
    model.save_model(path=getattr(args, constants.OUTPUT_NAME))


def parse_arguments():
    parser = argparse.ArgumentParser(description='train the Permutect artifact model')
    add_model_params_to_parser(parser)
    add_training_params_to_parser(parser)

    parser.add_argument('--' + constants.TRAIN_TAR_NAME, type=str, required=True,
                        help='training dataset .tar.gz file produced by preprocess_dataset.py')
    parser.add_argument('--' + constants.OUTPUT_NAME, type=str, required=True, help='output artifact model file')
    parser.add_argument('--' + constants.TENSORBOARD_DIR_NAME, type=str, default='tensorboard', required=False,
                        help='output tensorboard directory')

    return parser.parse_args()

def main():
    args = parse_arguments()
    main_without_parsing(args)


if __name__ == '__main__':
    main()