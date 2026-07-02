import argparse

from torch.utils.tensorboard import SummaryWriter

from permutect import constants
from permutect.architecture.artifact_model import ArtifactModel
from permutect.architecture.artifact_model import load_model
from permutect.data.memory_mapped_data import MemoryMappedData
from permutect.data.reads_dataset import ReadsDataset
from permutect.data.reads_dataset import all_but_last_fold
from permutect.data.reads_dataset import last_fold_only
from permutect.misc_utils import Timer
from permutect.misc_utils import gpu_if_available
from permutect.misc_utils import report_memory_usage
from permutect.parameters import add_model_params_to_parser
from permutect.parameters import add_training_params_to_parser
from permutect.parameters import parse_model_params
from permutect.parameters import parse_training_params
from permutect.training.model_training import train_artifact_model


def main_without_parsing(args):
    params = parse_model_params(args)
    training_params = parse_training_params(args)
    # optional pretrained model to use as initialization
    pretrained_model_path = getattr(args, constants.PRETRAINED_ARTIFACT_MODEL_NAME)

    pretrained_model, _, _ = (None, None, None) if pretrained_model_path is None else load_model(pretrained_model_path)

    tensorboard_dir = getattr(args, constants.TENSORBOARD_DIR_NAME)
    summary_writer = SummaryWriter(tensorboard_dir)
    report_memory_usage("Training data about to be loaded from tarfile.")
    memory_mapped_data = MemoryMappedData.load_from_tarfile(getattr(args, constants.TRAIN_TAR_NAME))
    num_folds = 10
    subset_timer = Timer("Creating training and validation datasets")
    train_dataset = ReadsDataset(memory_mapped_data, num_folds=num_folds, folds_to_use=all_but_last_fold(num_folds))
    valid_dataset = ReadsDataset(memory_mapped_data, num_folds=num_folds, folds_to_use=last_fold_only(num_folds))
    subset_timer.report("Time to create training and validation datasets")

    model = (
        pretrained_model
        if (pretrained_model is not None)
        else ArtifactModel(
            params=params,
            num_read_features=train_dataset.num_read_features(),
            num_info_features=train_dataset.num_info_features(),
            haplotypes_length=train_dataset.haplotypes_length(),
            device=gpu_if_available(),
        )
    )

    train_artifact_model(
        model,
        train_dataset,
        valid_dataset,
        training_params,
        summary_writer=summary_writer,
        epochs_per_evaluation=10,
        trainable_params=training_params.trainable_parameter_sets,
    )

    summary_writer.close()

    model.save_model(path=getattr(args, constants.OUTPUT_NAME))


def parse_arguments():
    parser = argparse.ArgumentParser(description="train the Permutect artifact model")
    add_model_params_to_parser(parser)
    add_training_params_to_parser(parser)

    tar_kwargs = {"type": str, "required": True, "help": "dataset .tar.gz file produced by preprocess_dataset.py"}
    parser.add_argument("--" + constants.TRAIN_TAR_NAME, **tar_kwargs)


    parser.add_argument("--" + constants.OUTPUT_NAME, type=str, required=True, help="output artifact model file")

    tensorboard_kwargs = {"type": str, "default": "tensorboard", "required": False, "help": "tensorboard directory"}
    parser.add_argument("--" + constants.TENSORBOARD_DIR_NAME, **tensorboard_kwargs)

    return parser.parse_args()


def main():
    args = parse_arguments()
    main_without_parsing(args)


if __name__ == "__main__":
    main()
