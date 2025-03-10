import argparse

import torch
from torch.utils.tensorboard import SummaryWriter

from permutect import constants
from permutect.architecture.spectra.artifact_spectra import ArtifactSpectra
from permutect.architecture.model_training import train_permutect_model
from permutect.architecture.permutect_model import load_model
from permutect.architecture.posterior_model import plot_artifact_spectra
from permutect.data.reads_dataset import ReadsDataset
from permutect.data.reads_datum import ReadsDatum
from permutect.parameters import add_training_params_to_parser, parse_training_params
from permutect.misc_utils import report_memory_usage
from permutect.utils.enums import Variation, Label


def learn_artifact_priors_and_spectra(dataset: ReadsDataset, genomic_span_of_data: int):
    artifact_counts = torch.zeros(len(Variation))
    types_list, depths_list, alt_counts_list = [], [], []

    datum: ReadsDatum
    for datum in dataset:
        if datum.get_label() != Label.ARTIFACT:
            continue
        variant_type = datum.get_variant_type()
        artifact_counts[variant_type] += 1
        types_list.append(variant_type)
        depths_list.append(datum.get_original_depth())
        alt_counts_list.append(datum.get_original_alt_count())

    # turn the lists into tensors
    types_tensor = torch.LongTensor(types_list)
    depths_tensor = torch.tensor(depths_list).float()
    alt_counts_tensor = torch.tensor(alt_counts_list).float()

    log_artifact_priors = torch.log(artifact_counts / genomic_span_of_data)
    artifact_spectra = ArtifactSpectra(num_components=2)

    # TODO: hard-coded num epochs!!!
    artifact_spectra.fit(num_epochs=10, types_b=types_tensor, depths_b=depths_tensor,
                         alt_counts_b=alt_counts_tensor, batch_size=64)

    return log_artifact_priors, artifact_spectra


def parse_arguments():
    parser = argparse.ArgumentParser(description='train the Permutect artifact model')

    add_training_params_to_parser(parser)

    parser.add_argument('--' + constants.CALIBRATION_SOURCES_NAME, nargs='+', default=None, type=int, required=False,
                        help='which sources to use in calibration.  Default: use all sources.')
    parser.add_argument('--' + constants.LEARN_ARTIFACT_SPECTRA_NAME, action='store_true',
                        help='flag to include artifact priors and allele fraction spectra in saved output.  '
                             'This is worth doing if labeled training data is available but might work poorly '
                             'when Mutect3 generates weak labels based on allele fractions.')
    parser.add_argument('--' + constants.GENOMIC_SPAN_NAME, type=float, required=False,
                        help='Total number of sites considered by Mutect2 in all training data, including those lacking variation or artifacts, hence absent from input datasets.  '
                             'Necessary for learning priors since otherwise rates of artifacts and variants would be overinflated. '
                             'Only required if learning artifact log priors')

    # inputs and outputs
    parser.add_argument('--' + constants.TRAIN_TAR_NAME, type=str, required=True,
                        help='tarfile of training/validation datasets produced by preprocess_dataset.py')
    parser.add_argument('--' + constants.SAVED_MODEL_NAME, type=str, help='Base model from train_permutect_model.py')
    parser.add_argument('--' + constants.OUTPUT_NAME, type=str, required=True, help='path to output saved model file')
    parser.add_argument('--' + constants.TENSORBOARD_DIR_NAME, type=str, default='tensorboard', required=False,
                        help='path to output tensorboard directory')

    return parser.parse_args()


def main_without_parsing(args):
    training_params = parse_training_params(args)
    learn_artifact_spectra = getattr(args, constants.LEARN_ARTIFACT_SPECTRA_NAME)
    calibration_sources = getattr(args, constants.CALIBRATION_SOURCES_NAME)
    genomic_span = getattr(args, constants.GENOMIC_SPAN_NAME)

    tensorboard_dir = getattr(args, constants.TENSORBOARD_DIR_NAME)
    summary_writer = SummaryWriter(tensorboard_dir)

    # base and artifact models have already been trained.  We're just refining it here.
    model, _, _ = load_model(getattr(args, constants.SAVED_MODEL_NAME))
    report_memory_usage("Creating ReadsDataset.")
    dataset = ReadsDataset(data_tarfile=getattr(args, constants.TRAIN_TAR_NAME), num_folds=10)

    train_permutect_model(model, dataset, training_params, summary_writer, epochs_per_evaluation=10, calibration_sources=calibration_sources)

    for var_type in Variation:
        cal_fig, cal_axes = model.calibration.plot_calibration_module(var_type=var_type, device=model._device, dtype=model._dtype)
        summary_writer.add_figure("calibration by count for " + var_type.name, cal_fig)

    report_memory_usage("Finished training.")

    artifact_log_priors, artifact_spectra = learn_artifact_priors_and_spectra(dataset, genomic_span) if learn_artifact_spectra else (None, None)
    if artifact_spectra is not None:
        art_spectra_fig, art_spectra_axs = plot_artifact_spectra(artifact_spectra, depth=50)
        summary_writer.add_figure("Artifact AF Spectra", art_spectra_fig)

    summary_writer.close()

    # TODO: this will only be correct once we use the full base model, not the separate artifact model
    model.save_model(path=getattr(args, constants.OUTPUT_NAME), artifact_log_priors=artifact_log_priors, artifact_spectra=artifact_spectra)


def main():
    args = parse_arguments()
    main_without_parsing(args)


if __name__ == '__main__':
    main()
