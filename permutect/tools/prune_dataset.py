import argparse
import os
import tarfile
import tempfile
import pickle

import psutil
from permutect.data import read_set
from tqdm.autonotebook import tqdm

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from permutect import constants, utils
from permutect.architecture.artifact_model import ArtifactModelParameters, ArtifactModel
from permutect.data.read_set_dataset import ReadSetDataset, make_data_loader
from permutect.tools.train_model import TrainingParameters, parse_hyperparams, parse_training_params, \
    add_artifact_model_hyperparameters_to_parser, add_artifact_model_training_hyperparameters_to_parser
from permutect.utils import MutableInt

NUM_FOLDS = 3


def prune_training_data(hyperparams: ArtifactModelParameters, params: TrainingParameters, tensorboard_dir, data_tarfile, pruned_tarfile, chunk_size):
    use_gpu = torch.cuda.is_available()
    device = torch.device('cuda' if use_gpu else 'cpu')

    dataset = ReadSetDataset(data_tarfile=data_tarfile, num_folds=NUM_FOLDS)

    model = ArtifactModel(params=hyperparams, num_read_features=dataset.num_read_features,
                          num_info_features=dataset.num_info_features, ref_sequence_length=dataset.ref_sequence_length,
                          device=device).float()

    pruned_indices = []
    for fold in range(NUM_FOLDS):
        summary_writer = SummaryWriter(tensorboard_dir + "/fold_" + str(fold))
        print("Training model on fold " + str(fold) + " of " + str(NUM_FOLDS))
        # note: not training from scratch.  I assume that there are enough epochs to forget any overfitting from
        # previous folds
        model.train_model(dataset, params.num_epochs, params.num_calibration_epochs, params.batch_size, params.num_workers, summary_writer=summary_writer,
                          reweighting_range=params.reweighting_range, hyperparams=hyperparams, validation_fold=fold)

        average_artifact_confidence, average_nonartifact_confidence = utils.StreamingAverage(), utils.StreamingAverage()

        # now we go over all the labeled data in the validation set -- that is, the current fold -- and perform rank pruning
        valid_loader = make_data_loader(dataset, [fold], params.batch_size, use_gpu, params.num_workers)

        # TODO: eventually this should all be segregated by variant type

        # TODO: may also want to segregate by alt count

        # the 0th/1st element is a list of predicted probabilities that data labeled as non-artifact/artifact are actually non-artifact/artifact
        probs_of_agreeing_with_label = [[],[]]
        print("calculating average confidence and gathering predicted probabilities")
        pbar = tqdm(enumerate(filter(lambda bat: bat.is_labeled(), valid_loader)), mininterval=60)
        for n, batch in pbar:
            # TODO: should we use likelihoods as in evaluation or posteriors as in training???
            # TODO: does it even matter??
            art_probs = torch.sigmoid(model.forward(batch).detach())

            art_label_mask = (batch.labels > 0.5)
            nonart_label_mask = (batch.labels < 0.5)
            average_artifact_confidence.record_with_mask(art_probs, art_label_mask)
            average_nonartifact_confidence.record_with_mask(1 - art_probs, nonart_label_mask)

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
    # TODO: Maybe also save a dataset of discarded values?
    pruned_data_files = []
    pruned_datum_index = MutableInt()
    pruned_indices_file = open("pruned_indices.txt", 'w')
    for read_set_list in generate_pruned_data(dataset=dataset, max_bytes_per_chunk=chunk_size, pruned_indices=pruned_indices):
        with tempfile.NamedTemporaryFile(delete=False) as train_data_file:
            read_set.save_list_of_read_sets(read_set_list, train_data_file, pruned_datum_index, pruned_indices_file)
            pruned_data_files.append(train_data_file.name)

    pruned_indices_file.close()
    # bundle them in a tarfile
    with tarfile.open(pruned_tarfile, "w") as train_tar:
        for train_file in pruned_data_files:
            train_tar.add(train_file, arcname=os.path.basename(train_file))


def generate_pruned_data(dataset: ReadSetDataset, max_bytes_per_chunk: int, pruned_indices):
    buffer, bytes_in_buffer = [], 0
    for datum in dataset:
        if datum.index in pruned_indices:   # exclude pruned data from output
            continue

        buffer.append(datum)
        bytes_in_buffer += datum.size_in_bytes()
        if bytes_in_buffer > max_bytes_per_chunk:
            print("memory usage percent: " + str(psutil.virtual_memory().percent))
            print("bytes in chunk: " + str(bytes_in_buffer))
            yield buffer
            buffer, bytes_in_buffer = [], 0

    # There will be some data left over, in general.
    if buffer:
        yield buffer


def parse_arguments():
    parser = argparse.ArgumentParser(description='train the Mutect3 artifact model')

    add_artifact_model_hyperparameters_to_parser(parser)
    add_artifact_model_training_hyperparameters_to_parser(parser)

    parser.add_argument('--' + constants.CHUNK_SIZE_NAME, type=int, default=int(2e9), required=False,
                        help='size in bytes of output binary data files')

    # input / output
    parser.add_argument('--' + constants.TRAIN_TAR_NAME, type=str, required=True,
                        help='tarfile of training/validation datasets produced by preprocess_dataset.py')
    parser.add_argument('--' + constants.OUTPUT_NAME, type=str, required=True,
                        help='path to output saved model file')

    parser.add_argument('--' + constants.TENSORBOARD_DIR_NAME, type=str, default='tensorboard', required=False,
                        help='path to output tensorboard directory')

    return parser.parse_args()


def main_without_parsing(args):
    hyperparams = parse_hyperparams(args)
    training_params = parse_training_params(args)

    tarfile_data = getattr(args, constants.TRAIN_TAR_NAME)
    tensorboard_dir = getattr(args, constants.TENSORBOARD_DIR_NAME)
    pruned_tarfile = getattr(args, constants.OUTPUT_NAME)
    chunk_size = getattr(args, constants.CHUNK_SIZE_NAME)

    # this writes the new pruned data tarfile and the new post-pruning indices files
    prune_training_data(hyperparams=hyperparams, data_tarfile=tarfile_data, params=training_params, tensorboard_dir=tensorboard_dir,
                        pruned_tarfile=pruned_tarfile, chunk_size=chunk_size)


def main():
    args = parse_arguments()
    main_without_parsing(args)


if __name__ == '__main__':
    main()
