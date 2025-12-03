import argparse

import torch
from torch.utils.tensorboard import SummaryWriter

from permutect import constants
from permutect.architecture.artifact_model import ArtifactModel, load_model
from permutect.data.memory_mapped_data import MemoryMappedData
from permutect.training.model_training import train_artifact_model
from permutect.misc_utils import gpu_if_available, report_memory_usage, Timer
from permutect.parameters import parse_training_params, parse_model_params, add_model_params_to_parser, add_training_params_to_parser
from permutect.data.reads_dataset import ReadsDataset, all_but_the_last_fold, last_fold_only


MAX_EMBEDDINGS_TO_RECORD_PER_LABEL = 10000

def main_without_parsing(args):
    params = parse_model_params(args)
    training_params = parse_training_params(args)
    pretrained_model_path = getattr(args, constants.PRETRAINED_ARTIFACT_MODEL_NAME)    # optional pretrained model to use as initialization

    pretrained_model, _, _ = (None, None, None) if pretrained_model_path is None else load_model(pretrained_model_path)

    tensorboard_dir = getattr(args, constants.TENSORBOARD_DIR_NAME)
    summary_writer = SummaryWriter(tensorboard_dir)
    report_memory_usage("Training data about to be loaded from tarfile.")
    memory_mapped_data = MemoryMappedData.load_from_tarfile(getattr(args, constants.TRAIN_TAR_NAME))
    num_folds=10
    subset_timer = Timer("Creating training and validation datasets")
    train_dataset = ReadsDataset(memory_mapped_data=memory_mapped_data, num_folds=num_folds, folds_to_use=all_but_the_last_fold(num_folds))
    valid_dataset = ReadsDataset(memory_mapped_data=memory_mapped_data, num_folds=num_folds, folds_to_use=last_fold_only(num_folds))
    keep_probs_by_label_l = torch.clamp(MAX_EMBEDDINGS_TO_RECORD_PER_LABEL / (train_dataset.totals_by_label() + 1), min=0, max=1)
    embeddings_dataset = ReadsDataset(memory_mapped_data=memory_mapped_data, num_folds=num_folds,
                                 folds_to_use=all_but_the_last_fold(num_folds), keep_probs_by_label_l=keep_probs_by_label_l)
    subset_timer.report("Time to create training and validation datasets")

    model = pretrained_model if (pretrained_model is not None) else \
            ArtifactModel(params=params, num_read_features=train_dataset.num_read_features(), num_info_features=train_dataset.num_info_features(),
                          haplotypes_length=train_dataset.haplotypes_length(), device=gpu_if_available())

    train_artifact_model(model, train_dataset, valid_dataset, training_params, summary_writer=summary_writer, epochs_per_evaluation=10)

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