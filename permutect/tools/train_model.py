import argparse

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from permutect import constants, utils
from permutect.architecture.artifact_model import ArtifactModel
from permutect.architecture.posterior_model import initialize_artifact_spectra, plot_artifact_spectra
from permutect.architecture.base_model import load_base_model
from permutect.data.read_set_dataset import ReadSetDataset
from permutect.data.representation_dataset import RepresentationDataset
from permutect.parameters import TrainingParameters, add_training_params_to_parser, parse_training_params, \
    ArtifactModelParameters, parse_artifact_model_params, add_artifact_model_params_to_parser
from permutect.utils import Variation, Label


def train_artifact_model(hyperparams: ArtifactModelParameters, training_params: TrainingParameters, summary_writer: SummaryWriter, dataset: RepresentationDataset):
    use_gpu = torch.cuda.is_available()
    device = torch.device('cuda' if use_gpu else 'cpu')

    model = ArtifactModel(params=hyperparams, num_representation_features=dataset.num_representation_features, device=device).float()

    print("Training. . .")
    model.learn(dataset, training_params, summary_writer=summary_writer)

    for n, var_type in enumerate(Variation):
        cal_fig, cal_axes = model.calibration[n].plot_calibration()
        summary_writer.add_figure("calibration for " + var_type.name, cal_fig)

    print("Evaluating trained model. . .")
    model.evaluate_model_after_training(dataset, training_params.batch_size, training_params.num_workers, summary_writer)

    return model


def learn_artifact_priors_and_spectra(dataset: RepresentationDataset, genomic_span_of_data: int):
    artifact_counts = torch.zeros(len(utils.Variation))
    types_list, depths_list, alt_counts_list = [], [], []

    for read_set in dataset:
        if read_set.label != Label.ARTIFACT:
            continue
        variant_type = read_set.get_variant_type()
        artifact_counts[variant_type] += 1
        types_list.append(variant_type)
        depths_list.append(read_set.counts_and_seq_lks.depth)
        alt_counts_list.append(read_set.counts_and_seq_lks.alt_count)

    # turn the lists into tensors
    types_one_hot_tensor = torch.from_numpy(np.vstack([var_type.one_hot_tensor() for var_type in types_list])).float()
    depths_tensor = torch.Tensor(depths_list).float()
    alt_counts_tensor = torch.Tensor(alt_counts_list).float()

    log_artifact_priors = torch.log(artifact_counts / genomic_span_of_data)
    artifact_spectra = initialize_artifact_spectra()

    # TODO: hard-coded num epochs!!!
    artifact_spectra.fit(num_epochs=10, inputs_2d_tensor=types_one_hot_tensor, depths_1d_tensor=depths_tensor,
                         alt_counts_1d_tensor=alt_counts_tensor, batch_size=64)

    return log_artifact_priors, artifact_spectra


def parse_arguments():
    parser = argparse.ArgumentParser(description='train the Permutect artifact model')

    add_artifact_model_params_to_parser(parser)
    add_training_params_to_parser(parser)

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
    parser.add_argument('--' + constants.BASE_MODEL_NAME, type=str, help='Base model from train_base_model.py')
    parser.add_argument('--' + constants.OUTPUT_NAME, type=str, required=True, help='path to output saved model file')
    parser.add_argument('--' + constants.TENSORBOARD_DIR_NAME, type=str, default='tensorboard', required=False,
                        help='path to output tensorboard directory')

    return parser.parse_args()


def main_without_parsing(args):
    params = parse_artifact_model_params(args)
    training_params = parse_training_params(args)
    learn_artifact_spectra = getattr(args, constants.LEARN_ARTIFACT_SPECTRA_NAME)
    genomic_span = getattr(args, constants.GENOMIC_SPAN_NAME)

    tensorboard_dir = getattr(args, constants.TENSORBOARD_DIR_NAME)
    summary_writer = SummaryWriter(tensorboard_dir)

    base_model = load_base_model(getattr(args, constants.BASE_MODEL_NAME))
    read_set_dataset = ReadSetDataset(data_tarfile=getattr(args, constants.TRAIN_TAR_NAME), num_folds=10)
    representation_dataset = RepresentationDataset(read_set_dataset, base_model)

    model = train_artifact_model(hyperparams=params, training_params=training_params,
                                 summary_writer=summary_writer, dataset=representation_dataset)

    artifact_log_priors, artifact_spectra = learn_artifact_priors_and_spectra(representation_dataset, genomic_span) if learn_artifact_spectra else (None, None)
    if artifact_spectra is not None:
        art_spectra_fig, art_spectra_axs = plot_artifact_spectra(artifact_spectra)
        summary_writer.add_figure("Artifact AF Spectra", art_spectra_fig)

    summary_writer.close()
    model.save(getattr(args, constants.OUTPUT_NAME), artifact_log_priors, artifact_spectra)


def main():
    args = parse_arguments()
    main_without_parsing(args)


if __name__ == '__main__':
    main()
