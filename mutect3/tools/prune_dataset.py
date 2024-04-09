import argparse
import tempfile
import pickle
from tqdm.autonotebook import tqdm

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from mutect3 import constants, utils
from mutect3.architecture.artifact_model import ArtifactModelParameters, ArtifactModel
from mutect3.architecture.posterior_model import initialize_artifact_spectra, plot_artifact_spectra
from mutect3.data.read_set_dataset import ReadSetDataset, make_data_loader
from mutect3.tools.train_model import TrainingParameters, parse_mutect3_params, parse_training_params

NUM_FOLDS = 3


# TODO: still need the summary writer??
def prune_training_data(m3_params: ArtifactModelParameters, params: TrainingParameters, summary_writer: SummaryWriter, data_tarfile):
    use_gpu = torch.cuda.is_available()
    device = torch.device('cuda' if use_gpu else 'cpu')

    dataset = ReadSetDataset(data_tarfile=data_tarfile, num_folds=NUM_FOLDS)

    model = ArtifactModel(params=m3_params, num_read_features=dataset.num_read_features,
                          num_info_features=dataset.num_info_features, ref_sequence_length=dataset.ref_sequence_length,
                          device=device).float()

    pruned_indices = []
    for fold in range(NUM_FOLDS):
        print("Training model on fold " + str(fold) + " of " + str(NUM_FOLDS))
        # note: not training from scratch.  I assume that there are enough epochs to forget any overfitting from
        # previous folds
        model.train_model(dataset, params.num_epochs, params.num_calibration_epochs, params.batch_size, params.num_workers, summary_writer=summary_writer,
                          reweighting_range=params.reweighting_range, m3_params=m3_params, validation_fold=fold)

        average_artifact_confidence, average_nonartifact_confidence = utils.StreamingAverage(), utils.StreamingAverage()

        # now we go over all the labeled data in the validation set -- that is, the current fold -- and perform rank pruning
        valid_loader = make_data_loader(dataset, [fold], params.batch_size, use_gpu, params.num_workers)

        # TODO: eventually this should all be segregated by variant type

        # TODO: may also want to divide by alt count

        # the 0th/1st element is a list of predicted probabilities that data labeled as non-artifact/artifact are actually non-artifact/artifact
        probs_of_agreeing_with_label = [[],[]]
        print("calculating average confidence and gathering predicted probabilities")
        pbar = tqdm(enumerate(filter(lambda bat: bat.is_labeled(), valid_loader)), mininterval=60)
        for n, batch in pbar:
            # TODO: should we use likelihoods as in evaluation or posteriors as in training???
            # TODO: does it even matter??
            art_probs = torch.sigmoid(model.forward(batch).detach())

            art_label_mask = (batch.labels > 0.5)
            average_artifact_confidence.record_with_mask(art_probs, art_label_mask)
            average_nonartifact_confidence.record_with_mask(1 - art_probs, 1 - art_label_mask)

            for art_prob, labeled_as_art in zip(art_probs.tolist(), art_label_mask.tolist()):
                agreement_prob = art_prob if labeled_as_art else (1 - art_prob)
                probs_of_agreeing_with_label[1 if labeled_as_art else 0].append(agreement_prob)

        # TODO: it is wasteful to run forward passes on all the data again when we can just record indices and logits
        print("estimating error rates")
        # The i,j element is the count of data labeled as i that pass the confidence threshold for j
        # here 0 means non-artifact and 1 means artifact
        confusion = [[0, 0], [0, 0]]
        art_conf_threshold = average_artifact_confidence.get()
        nonart_conf_threshold = average_nonartifact_confidence.get()
        pbar = tqdm(enumerate(filter(lambda bat: bat.is_labeled(), valid_loader)), mininterval=60)
        for n, batch in pbar:
            predicted_artifact_probs = torch.sigmoid(model.forward(batch).detach())

            conf_art_mask = predicted_artifact_probs >= art_conf_threshold
            conf_nonart_mask = (1 - predicted_artifact_probs) >= nonart_conf_threshold
            art_label_mask = (batch.labels > 0.5)

            for conf_artifact, conf_nonartifact, artifact_label in zip(conf_art_mask.tolist(), conf_nonart_mask.tolist(), art_label_mask.tolist()):
                row = 1 if artifact_label else 0
                if conf_artifact:
                    confusion[row][1] += 1
                if conf_nonartifact:
                    confusion[row][0] += 1

        # these are the probabilities of a true (hidden label) artifact/non-artifact being mislabeled as non-artifact/artifact
        art_error_rate = confusion[0][1] / (confusion[0][1] + confusion[1][1])
        nonart_error_rate = confusion[1][0] / (confusion[0][0] + confusion[1][0])

        # fraction of labeled data that are labeled as artifact
        label_art_frac = np.sum(dataset.artifact_totals) / (np.sum(dataset.artifact_totals + dataset.non_artifact_totals))
        label_nonart_frac = 1 - label_art_frac

        # these are the inverse probabilities that something labeled as artifact/non-artifact was actually a mislabeled nonartifact/artifact
        inv_art_error_rate = (nonart_error_rate / label_art_frac) * (label_nonart_frac - art_error_rate) / (1 - art_error_rate - nonart_error_rate)
        inv_nonart_error_rate = (art_error_rate / label_nonart_frac) * (label_art_frac - nonart_error_rate) / (1 - art_error_rate - nonart_error_rate)

        print("Estimated error rates: ")
        print("artifact mislabeled as nonartifact " + str(art_error_rate))
        print("non-artifact mislabeled as artifact " + str(nonart_error_rate))

        print("Estimated inverse error rates: ")
        print("Labeled artifact was actually nonartifact " + str(inv_art_error_rate))
        print("Labeled non-artifact was actually artifact " + str(inv_nonart_error_rate))

        print("calculating rank pruning thresholds")
        nonart_threshold = torch.quantile(torch.Tensor(probs_of_agreeing_with_label[0]), inv_nonart_error_rate).item()
        art_threshold = torch.quantile(torch.Tensor(probs_of_agreeing_with_label[1]), inv_art_error_rate).item()

        print("Rank pruning thresholds: ")
        print("Labeled artifacts are pruned if predicted artifact probability is less than " + str(art_threshold))
        print("Labeled non-artifacts are pruned if predicted non-artifact probability is less than " + str(nonart_threshold))

        print("pruning the dataset")
        pbar = tqdm(enumerate(filter(lambda bat: bat.is_labeled(), valid_loader)), mininterval=60)
        for n, batch in pbar:
            art_probs = torch.sigmoid(model.forward(batch).detach())
            art_label_mask = (batch.labels > 0.5)

            for art_prob, labeled_as_art, index in zip(art_probs.tolist(), art_label_mask.tolist(), batch.indices.tolist()):
                if (labeled_as_art and art_prob < art_threshold) or ((not labeled_as_art) and (1-art_prob) < nonart_threshold):
                    pruned_indices.append(index)



    # done with this particular fold

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
