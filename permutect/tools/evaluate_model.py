import argparse
from torch.utils.tensorboard import SummaryWriter
from permutect import constants
from permutect.architecture.model_training import evaluate_model_after_training
from permutect.architecture.permutect_model import load_model
from permutect.data.base_dataset import BaseDataset


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--' + constants.EVALUATION_TAR_NAME, type=str, required=True,
                        help='tarfile of evaluation datasets produced by preprocess_dataset.py')
    parser.add_argument('--' + constants.SAVED_MODEL_NAME, required=True, help='trained Permutect model from train_artifact_model.py')
    parser.add_argument('--' + constants.TENSORBOARD_DIR_NAME, type=str, default='tensorboard', required=False, help='path to output tensorboard')
    parser.add_argument('--' + constants.BATCH_SIZE_NAME, type=int, default=64, required=False, help='batch size')
    parser.add_argument('--' + constants.NUM_WORKERS_NAME, type=int, default=0, required=False,
                        help='number of subprocesses devoted to data loading, which includes reading from memory map, '
                             'collating batches, and transferring to GPU.')

    return parser.parse_args()


def main_without_parsing(args):
    data_tarfile = getattr(args, constants.EVALUATION_TAR_NAME)
    saved_model = getattr(args, constants.SAVED_MODEL_NAME)
    tensorboard_dir = getattr(args, constants.TENSORBOARD_DIR_NAME)
    batch_size = getattr(args, constants.BATCH_SIZE_NAME)
    num_workers = getattr(args, constants.NUM_WORKERS_NAME)

    dataset = BaseDataset(data_tarfile=data_tarfile, num_folds=10)
    model, _, _ = load_model(saved_model)

    summary_writer = SummaryWriter(tensorboard_dir)
    # TODO: this has been broken for a long time, since it expects an artifact dataset, not a base dataset!!!!
    evaluate_model_after_training(model, dataset, batch_size, num_workers, summary_writer)


def main():
    args = parse_arguments()
    main_without_parsing(args)


if __name__ == '__main__':
    main()