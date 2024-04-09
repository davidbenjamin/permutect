import argparse
import tempfile
import tarfile
import os
import pickle

import torch
from torch.utils.tensorboard import SummaryWriter

from mutect3 import constants, utils
from mutect3.architecture.artifact_model import ArtifactModelParameters, ArtifactModel
from mutect3.architecture.posterior_model import initialize_artifact_spectra, plot_artifact_spectra
from mutect3.data.read_set_dataset import ReadSetDataset
from mutect3.utils import Variation


class TrainingParameters:
    def __init__(self, batch_size, num_epochs, num_calibration_epochs, reweighting_range: float, num_workers: int = 0):
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.num_calibration_epochs = num_calibration_epochs
        self.reweighting_range = reweighting_range
        self.num_workers = num_workers


def train_artifact_model(m3_params: ArtifactModelParameters, params: TrainingParameters, summary_writer: SummaryWriter, data_tarfile):
    use_gpu = torch.cuda.is_available()
    device = torch.device('cuda' if use_gpu else 'cpu')

    dataset = ReadSetDataset(data_tarfile=data_tarfile, num_folds=10)

    model = ArtifactModel(params=m3_params, num_read_features=dataset.num_read_features,
                          num_info_features=dataset.num_info_features, ref_sequence_length=dataset.ref_sequence_length,
                          device=device).float()

    print("Training. . .")
    model.train_model(dataset, params.num_epochs, params.num_calibration_epochs, params.batch_size, params.num_workers, summary_writer=summary_writer,
                      reweighting_range=params.reweighting_range, m3_params=m3_params)

    for n, var_type in enumerate(Variation):
        cal_fig, cal_axes = model.calibration[n].plot_calibration()
        summary_writer.add_figure("calibration for " + var_type.name, cal_fig)

    print("Evaluating trained model. . .")
    model.evaluate_model_after_training(dataset, params.batch_size, params.num_workers, summary_writer)

    return model


def learn_artifact_priors_and_spectra(artifact_tarfile, genomic_span_of_data: int):
    with tempfile.TemporaryDirectory() as temp_dir:
        data_files = utils.extract_to_temp_dir(artifact_tarfile, temp_dir)

        artifact_counts = torch.zeros(len(utils.Variation))
        types_one_hot_buffer, depths_buffer, alt_counts_buffer = [], [], []   # list of tensors as they accumulate
        types_one_hot_tensors, depths_tensors, alt_counts_tensors = [], [], []
        for file in data_files:
            for artifact_posterior in pickle.load(open(file, 'rb')):
                variant_type = utils.Variation.get_type(artifact_posterior.ref, artifact_posterior.alt)
                artifact_counts[variant_type] += 1
                types_one_hot_buffer.append(torch.from_numpy(variant_type.one_hot_tensor()))
                depths_buffer.append(artifact_posterior.depth)
                alt_counts_buffer.append(artifact_posterior.alt_count)

            # after each file, turn the buffers into tensors
            types_one_hot_tensors.append(torch.vstack(types_one_hot_buffer))
            depths_tensors.append(torch.Tensor(depths_buffer))
            alt_counts_tensors.append(torch.Tensor(alt_counts_buffer))
            types_one_hot_buffer, depths_buffer, alt_counts_buffer = [], [], []

    log_artifact_priors = torch.log(artifact_counts / genomic_span_of_data)

    artifact_spectra = initialize_artifact_spectra()

    # TODO: hard-coded num epochs!!!
    artifact_spectra.fit(num_epochs=10, inputs_2d_tensor=torch.vstack(types_one_hot_tensors).float(),
                         depths_1d_tensor=torch.hstack(depths_tensors).float(), alt_counts_1d_tensor=torch.hstack(alt_counts_tensors).float(),
                         batch_size=64)

    return log_artifact_priors, artifact_spectra


def save_artifact_model(model, m3_params, path, artifact_log_priors, artifact_spectra):
    torch.save({
        constants.STATE_DICT_NAME: model.state_dict(),
        constants.M3_PARAMS_NAME: m3_params,
        constants.NUM_READ_FEATURES_NAME: model.num_read_features(),
        constants.NUM_INFO_FEATURES_NAME: model.num_info_features(),
        constants.REF_SEQUENCE_LENGTH_NAME: model.ref_sequence_length(),
        constants.ARTIFACT_LOG_PRIORS_NAME: artifact_log_priors,
        constants.ARTIFACT_SPECTRA_STATE_DICT_NAME: artifact_spectra.state_dict()
    }, path)


def parse_training_params(args) -> TrainingParameters:
    reweighting_range = getattr(args, constants.REWEIGHTING_RANGE_NAME)
    batch_size = getattr(args, constants.BATCH_SIZE_NAME)
    num_epochs = getattr(args, constants.NUM_EPOCHS_NAME)
    num_calibration_epochs = getattr(args, constants.NUM_CALIBRATION_EPOCHS_NAME)
    num_workers = getattr(args, constants.NUM_WORKERS_NAME)
    return TrainingParameters(batch_size, num_epochs, num_calibration_epochs, reweighting_range, num_workers=num_workers)


def parse_mutect3_params(args) -> ArtifactModelParameters:
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
    alt_downsample = getattr(args, constants.ALT_DOWNSAMPLE_NAME)
    return ArtifactModelParameters(read_embedding_dimension, num_transformer_heads, transformer_hidden_dimension,
                 num_transformer_layers, info_layers, aggregation_layers, calibration_layers, ref_seq_layer_strings, dropout_p,
        batch_normalize, learning_rate, alt_downsample)


def parse_arguments():
    parser = argparse.ArgumentParser(description='train the Mutect3 artifact model')

    # architecture hyperparameters
    parser.add_argument('--' + constants.READ_EMBEDDING_DIMENSION_NAME, type=int, required=True,
                        help='dimension of read embedding output by the transformer')
    parser.add_argument('--' + constants.NUM_TRANSFORMER_HEADS_NAME, type=int, required=True,
                        help='number of transformer self-attention heads')
    parser.add_argument('--' + constants.TRANSFORMER_HIDDEN_DIMENSION_NAME, type=int, required=True,
                        help='hidden dimension of transformer keys and values')
    parser.add_argument('--' + constants.NUM_TRANSFORMER_LAYERS_NAME, type=int, required=True,
                        help='number of transformer layers')
    parser.add_argument('--' + constants.INFO_LAYERS_NAME, nargs='+', type=int, required=True,
                        help='dimensions of hidden layers in the info embedding subnetwork, including the dimension of the embedding itself.  '
                             'Negative values indicate residual skip connections')
    parser.add_argument('--' + constants.AGGREGATION_LAYERS_NAME, nargs='+', type=int, required=True,
                        help='dimensions of hidden layers in the aggregation subnetwork, excluding the dimension of input from lower subnetworks '
                             'and the dimension (1) of the output logit.  Negative values indicate residual skip connections')
    parser.add_argument('--' + constants.CALIBRATION_LAYERS_NAME, nargs='+', type=int, required=True,
                        help='dimensions of hidden layers in the calibration subnetwork, excluding the dimension (1) of input logit and) '
                             'and the dimension (also 1) of the output logit.')
    parser.add_argument('--' + constants.REF_SEQ_LAYER_STRINGS_NAME, nargs='+', type=str, required=True,
                        help='list of strings specifying convolution layers of the reference sequence embedding.  For example '
                             'convolution/kernel_size=3/out_channels=64 pool/kernel_size=2 leaky_relu '
                             'convolution/kernel_size=3/dilation=2/out_channels=5 leaky_relu flatten linear/out_features=10')
    parser.add_argument('--' + constants.DROPOUT_P_NAME, type=float, default=0.0, required=False,
                        help='dropout probability')
    parser.add_argument('--' + constants.LEARNING_RATE_NAME, type=float, default=0.001, required=False,
                        help='learning rate')
    parser.add_argument('--' + constants.ALT_DOWNSAMPLE_NAME, type=int, default=100, required=False,
                        help='max number of alt reads to downsample to inside the model')
    parser.add_argument('--' + constants.BATCH_NORMALIZE_NAME, action='store_true',
                        help='flag to turn on batch normalization')

    # Training data inputs
    parser.add_argument('--' + constants.TRAIN_TAR_NAME, type=str, required=True,
                        help='tarfile of training/validation datasets produced by preprocess_dataset.py')
    parser.add_argument('--' + constants.ARTIFACT_TAR_NAME, type=str, required=True,
                        help='tarfile of artifact posterior data produced by preprocess_dataset.py')

    # training hyperparameters
    parser.add_argument('--' + constants.REWEIGHTING_RANGE_NAME, type=float, default=0.3, required=False,
                        help='magnitude of data augmentation by randomly weighted average of read embeddings.  '
                             'a value of x yields random weights between 1 - x and 1 + x')
    parser.add_argument('--' + constants.BATCH_SIZE_NAME, type=int, default=64, required=False,
                        help='batch size')
    parser.add_argument('--' + constants.NUM_WORKERS_NAME, type=int, default=0, required=False,
                        help='number of subprocesses devoted to data loading, which includes reading from memory map, '
                             'collating batches, and transferring to GPU.')
    parser.add_argument('--' + constants.NUM_EPOCHS_NAME, type=int, required=True,
                        help='number of epochs for primary training loop')
    parser.add_argument('--' + constants.NUM_CALIBRATION_EPOCHS_NAME, type=int, required=True,
                        help='number of calibration epochs following primary training loop')

    parser.add_argument('--' + constants.LEARN_ARTIFACT_SPECTRA_NAME, action='store_true',
                        help='flag to include artifact priors and allele fraction spectra in saved output.  '
                             'This is worth doing if labeled training data is available but might work poorly '
                             'when Mutect3 generates weak labels based on allele fractions.')
    parser.add_argument('--' + constants.GENOMIC_SPAN_NAME, type=float, required=False,
                        help='Total number of sites considered by Mutect2 in all training data, including those lacking variation or artifacts, hence absent from input datasets.  '
                             'Necessary for learning priors since otherwise rates of artifacts and variants would be overinflated. '
                             'Only required if learning artifact log priors')

    # path to saved model
    parser.add_argument('--' + constants.OUTPUT_NAME, type=str, required=True,
                        help='path to output saved model file')
    parser.add_argument('--' + constants.TENSORBOARD_DIR_NAME, type=str, default='tensorboard', required=False,
                        help='path to output tensorboard directory')

    return parser.parse_args()


def main_without_parsing(args):
    m3_params = parse_mutect3_params(args)
    training_params = parse_training_params(args)
    learn_artifact_spectra = getattr(args, constants.LEARN_ARTIFACT_SPECTRA_NAME)
    genomic_span = getattr(args, constants.GENOMIC_SPAN_NAME)

    tarfile_data = getattr(args, constants.TRAIN_TAR_NAME)
    tensorboard_dir = getattr(args, constants.TENSORBOARD_DIR_NAME)
    summary_writer = SummaryWriter(tensorboard_dir)
    model = train_artifact_model(m3_params=m3_params, data_tarfile=tarfile_data, params=training_params, summary_writer=summary_writer)

    artifact_tarfile_data = getattr(args, constants.ARTIFACT_TAR_NAME)
    artifact_log_priors, artifact_spectra = learn_artifact_priors_and_spectra(artifact_tarfile_data, genomic_span) if learn_artifact_spectra else (None, None)
    if artifact_spectra is not None:
        art_spectra_fig, art_spectra_axs = plot_artifact_spectra(artifact_spectra)
        summary_writer.add_figure("Artifact AF Spectra", art_spectra_fig)

    summary_writer.close()
    save_artifact_model(model, m3_params, getattr(args, constants.OUTPUT_NAME), artifact_log_priors, artifact_spectra)


def main():
    args = parse_arguments()
    main_without_parsing(args)


if __name__ == '__main__':
    main()
