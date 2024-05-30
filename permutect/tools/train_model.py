import argparse

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from permutect import constants, utils
from permutect.architecture.artifact_model import ArtifactModelParameters, ArtifactModel
from permutect.architecture.posterior_model import initialize_artifact_spectra, plot_artifact_spectra
from permutect.data.read_set_dataset import ReadSetDataset, RepresentationDataset
from permutect.parameters import TrainingParameters, add_training_params_to_parser, parse_training_params
from permutect.tools.filter_variants import load_artifact_model
from permutect.utils import Variation, Label


def train_artifact_model(hyperparams: ArtifactModelParameters, training_params: TrainingParameters, summary_writer: SummaryWriter, dataset: RepresentationDataset):
    use_gpu = torch.cuda.is_available()
    device = torch.device('cuda' if use_gpu else 'cpu')

    model = ArtifactModel(params=hyperparams, device=device).float()

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


def save_artifact_model(model, hyperparams, path, artifact_log_priors, artifact_spectra):
    torch.save({
        constants.STATE_DICT_NAME: model.state_dict(),
        constants.HYPERPARAMS_NAME: hyperparams,
        constants.NUM_READ_FEATURES_NAME: model.num_read_features(),
        constants.NUM_INFO_FEATURES_NAME: model.num_info_features(),
        constants.REF_SEQUENCE_LENGTH_NAME: model.ref_sequence_length(),
        constants.ARTIFACT_LOG_PRIORS_NAME: artifact_log_priors,
        constants.ARTIFACT_SPECTRA_STATE_DICT_NAME: artifact_spectra.state_dict()
    }, path)


def parse_hyperparams(args) -> ArtifactModelParameters:
    read_embedding_dimension = getattr(args, constants.READ_EMBEDDING_DIMENSION_NAME)
    num_transformer_heads = getattr(args, constants.NUM_TRANSFORMER_HEADS_NAME)
    transformer_hidden_dimension = getattr(args, constants.TRANSFORMER_HIDDEN_DIMENSION_NAME)
    num_transformer_layers = getattr(args, constants.NUM_TRANSFORMER_LAYERS_NAME)

    info_layers = getattr(args, constants.INFO_LAYERS_NAME)
    aggregation_layers = getattr(args, constants.AGGREGATION_LAYERS_NAME)
    calibration_layers = getattr(args, constants.CALIBRATION_LAYERS_NAME)
    ref_seq_layer_strings = getattr(args, constants.REF_SEQ_LAYER_STRINGS_NAME)
    dropout_p = getattr(args, constants.DROPOUT_P_NAME)
    batch_normalize = getattr(args, constants.BATCH_NORMALIZE_NAME)
    learning_rate = getattr(args, constants.LEARNING_RATE_NAME)
    weight_decay = getattr(args, constants.WEIGHT_DECAY_NAME)
    alt_downsample = getattr(args, constants.ALT_DOWNSAMPLE_NAME)
    return ArtifactModelParameters(read_embedding_dimension, num_transformer_heads, transformer_hidden_dimension,
                 num_transformer_layers, info_layers, aggregation_layers, calibration_layers, ref_seq_layer_strings, dropout_p,
        batch_normalize, learning_rate, weight_decay, alt_downsample)


def parse_arguments():
    parser = argparse.ArgumentParser(description='train the Permutect artifact model')

    add_artifact_model_hyperparameters_to_parser(parser)
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
    parser.add_argument('--' + constants.PRETRAINED_MODEL_NAME, type=str,
                        help='Pre-trained representation model from train_representation_model.py')
    parser.add_argument('--' + constants.OUTPUT_NAME, type=str, required=True,
                        help='path to output saved model file')
    parser.add_argument('--' + constants.TENSORBOARD_DIR_NAME, type=str, default='tensorboard', required=False,
                        help='path to output tensorboard directory')

    return parser.parse_args()


def add_artifact_model_hyperparameters_to_parser(parser):
    parser.add_argument('--' + constants.AGGREGATION_LAYERS_NAME, nargs='+', type=int, required=True,
                        help='dimensions of hidden layers in the aggregation subnetwork, excluding the dimension of input from lower subnetworks '
                             'and the dimension (1) of the output logit.  Negative values indicate residual skip connections')
    parser.add_argument('--' + constants.CALIBRATION_LAYERS_NAME, nargs='+', type=int, required=True,
                        help='dimensions of hidden layers in the calibration subnetwork, excluding the dimension (1) of input logit and) '
                             'and the dimension (also 1) of the output logit.')
    parser.add_argument('--' + constants.DROPOUT_P_NAME, type=float, default=0.0, required=False,
                        help='dropout probability')
    parser.add_argument('--' + constants.BATCH_NORMALIZE_NAME, action='store_true',
                        help='flag to turn on batch normalization')



def main_without_parsing(args):
    hyperparams = parse_hyperparams(args)
    training_params = parse_training_params(args)
    learn_artifact_spectra = getattr(args, constants.LEARN_ARTIFACT_SPECTRA_NAME)
    genomic_span = getattr(args, constants.GENOMIC_SPAN_NAME)

    tarfile_data = getattr(args, constants.TRAIN_TAR_NAME)
    pretrained_model = getattr(args, constants.PRETRAINED_MODEL_NAME)
    tensorboard_dir = getattr(args, constants.TENSORBOARD_DIR_NAME)
    summary_writer = SummaryWriter(tensorboard_dir)
    # TODO: load the representation model
    # TODO: construct the representation dataset
    dataset = ReadSetDataset(data_tarfile=tarfile_data, num_folds=10)
    model = train_artifact_model(hyperparams=hyperparams, dataset=dataset, training_params=training_params,
                                 summary_writer=summary_writer, pretrained_model=pretrained_model)

    artifact_log_priors, artifact_spectra = learn_artifact_priors_and_spectra(dataset, genomic_span) if learn_artifact_spectra else (None, None)
    if artifact_spectra is not None:
        art_spectra_fig, art_spectra_axs = plot_artifact_spectra(artifact_spectra)
        summary_writer.add_figure("Artifact AF Spectra", art_spectra_fig)

    summary_writer.close()
    save_artifact_model(model, hyperparams, getattr(args, constants.OUTPUT_NAME), artifact_log_priors, artifact_spectra)


def main():
    args = parse_arguments()
    main_without_parsing(args)


if __name__ == '__main__':
    main()
