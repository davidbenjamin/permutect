import argparse
import os
import tarfile
import tempfile
from typing import List

from permutect.architecture.model_training import train_permutect_model
from permutect.architecture.permutect_model import PermutectModel, load_model
from tqdm.autonotebook import tqdm

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from permutect import constants
from permutect.data.reads_batch import ReadsBatch
from permutect.data.reads_datum import ReadsDatum
from permutect.data.prefetch_generator import prefetch_generator
from permutect.metrics.loss_metrics import BatchProperty
from permutect.parameters import add_training_params_to_parser, TrainingParameters
from permutect.data.reads_dataset import ReadsDataset
from permutect.tools.refine_permutect_model import parse_training_params
from permutect.misc_utils import report_memory_usage, StreamingAverage
from permutect.utils.enums import Label

NUM_FOLDS = 3


# labeled only pruning loader must be constructed with options to emit batches of all-labeled data
def calculate_pruning_thresholds(labeled_only_pruning_loader, model: PermutectModel, label_art_frac: float, training_params: TrainingParameters) -> List[int]:
    for fold in range(NUM_FOLDS):
        average_artifact_confidence, average_nonartifact_confidence = StreamingAverage(), StreamingAverage()
        # TODO: eventually this should all be segregated by variant type and maybe also alt count

        # the 0th/1st element is a list of predicted probabilities that data labeled as non-artifact/artifact are actually non-artifact/artifact
        probs_of_agreeing_with_label = [[],[]]
        print("calculating average confidence and gathering predicted probabilities")
        batch: ReadsBatch
        for batch in tqdm(prefetch_generator(labeled_only_pruning_loader), mininterval=60, total=len(labeled_only_pruning_loader)):
            # TODO: should we use likelihoods as in evaluation or posteriors as in training???
            # TODO: does it even matter??
            representations, _ = model.calculate_representations(batch, weight_range=model._params.reweighting_range)
            art_logits, _ = model.logits_from_reads_batch(representations, batch)
            art_probs = torch.sigmoid(art_logits.detach())

            labels = batch.get_training_labels()
            art_label_mask = (labels > 0.5)
            nonart_label_mask = (labels < 0.5)
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
        for batch in tqdm(prefetch_generator(labeled_only_pruning_loader), mininterval=60, total=len(labeled_only_pruning_loader)):
            representations, _ = model.calculate_representations(batch, weight_range=model._params.reweighting_range)
            predicted_artifact_logits, _ = model.logits_from_reads_batch(representations, batch)
            predicted_artifact_probs = torch.sigmoid(predicted_artifact_logits.detach())

            conf_art_mask = predicted_artifact_probs >= art_conf_threshold
            conf_nonart_mask = (1 - predicted_artifact_probs) >= nonart_conf_threshold
            art_label_mask = (batch.get_training_labels() > 0.5)

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
        label_nonart_frac = 1 - label_art_frac

        # these are the inverse probabilities that something labeled as artifact/non-artifact was actually a mislabeled nonartifact/artifact
        inv_art_error_rate = (nonart_error_rate / label_art_frac) * (label_nonart_frac - art_error_rate) / (1 - art_error_rate - nonart_error_rate)
        inv_nonart_error_rate = (art_error_rate / label_nonart_frac) * (label_art_frac - nonart_error_rate) / (1 - art_error_rate - nonart_error_rate)

        print("Estimated error rates: ")
        print(f"artifact mislabeled as non-artifact: {art_error_rate:.3f}")
        print(f"non-artifact mislabeled as artifact: {nonart_error_rate:.3f}")

        print("Estimated inverse error rates: ")
        print(f"Labeled artifact was actually non-artifact: {inv_art_error_rate:.3f}")
        print(f"Labeled non-artifact was actually artifact: {inv_nonart_error_rate:.3f}")

        print("calculating rank pruning thresholds")
        nonart_threshold = torch.quantile(torch.Tensor(probs_of_agreeing_with_label[0]), inv_nonart_error_rate).item()
        art_threshold = torch.quantile(torch.Tensor(probs_of_agreeing_with_label[1]), inv_art_error_rate).item()

        print("Rank pruning thresholds: ")
        print(f"Labeled artifacts are pruned if predicted artifact probability is less than {art_threshold:.3f}")
        print(f"Labeled non-artifacts are pruned if predicted non-artifact probability is less than {nonart_threshold:.3f}")

        return art_threshold, nonart_threshold


# generates BaseDatum(s) from the original dataset that *pass* the pruning thresholds
def generated_pruned_data_for_fold(art_threshold: float, nonart_threshold: float, pruning_base_data_loader, model: PermutectModel) -> List[int]:
    print("pruning the dataset")
    reads_batch: ReadsBatch
    for reads_batch in tqdm(prefetch_generator(pruning_base_data_loader), mininterval=60, total=len(pruning_base_data_loader)):
        # apply the representation model AND the artifact model to go from the original read set to artifact logits
        representation, _ = model.calculate_representations(reads_batch)

        art_logits, _ = model.logits_from_reads_batch(representation, reads_batch)
        art_probs = torch.sigmoid(art_logits.detach())
        art_label_mask = (reads_batch.get_training_labels() > 0.5)
        is_labeled_mask = (reads_batch.get_is_labeled_mask() > 0.5)

        for art_prob, labeled_as_art, data_1d, reads_2d, is_labeled in zip(art_probs.tolist(), art_label_mask.tolist(),
                reads_batch.get_data_2d(), reads_batch.get_list_of_reads_2d(), is_labeled_mask.tolist()):
            datum = ReadsDatum(data_1d, reads_2d)
            if not is_labeled:
                yield datum
            elif (labeled_as_art and art_prob < art_threshold) or ((not labeled_as_art) and (1-art_prob) < nonart_threshold):
                 # TODO: process failing data, perhaps add option to output a pruned dataset? or flip labels?
                pass
            else:
                yield datum # this is a ReadSet


def generate_pruned_data_for_all_folds(dataset: ReadsDataset, model: PermutectModel, training_params: TrainingParameters, tensorboard_dir):
    # for each fold in turn, train an artifact model on all other folds and prune the chosen fold
    use_gpu = torch.cuda.is_available()

    for pruning_fold in range(NUM_FOLDS):
        summary_writer = SummaryWriter(tensorboard_dir + "/fold_" + str(pruning_fold))
        report_memory_usage(f"Pruning data from fold {pruning_fold} of {NUM_FOLDS}.")

        totals_l = dataset.totals.get_marginal((BatchProperty.LABEL, )) # totals by label
        label_art_frac = totals_l[Label.ARTIFACT].item() / (totals_l[Label.ARTIFACT].item() + totals_l[Label.VARIANT].item())
        train_permutect_model(model, dataset, training_params, summary_writer=summary_writer, training_folds=[pruning_fold])

        # TODO: maybe this should be done by variant type and/or count
        # learn pruning thresholds on the held-out data
        labeled_only_pruning_loader = dataset.make_data_loader([pruning_fold], training_params.batch_size, use_gpu,
                                                               training_params.num_workers, labeled_only=True)
        art_threshold, nonart_threshold = calculate_pruning_thresholds(labeled_only_pruning_loader, model, label_art_frac, training_params)

        # unlike when learning thresholds, we load labeled and unlabeled data here
        pruning_base_data_loader = dataset.make_data_loader([pruning_fold], training_params.batch_size, use_gpu, training_params.num_epochs)
        for passing_reads_datum in generated_pruned_data_for_fold(art_threshold, nonart_threshold, pruning_base_data_loader, model):
            yield passing_reads_datum


# takes a ReadSet generator and organies into buffers.
# TODO: probably code duplication since the generator is already pruned
def generate_pruned_data_buffers(pruned_data_generator, max_bytes_per_chunk: int):
    buffer, bytes_in_buffer = [], 0
    for datum in pruned_data_generator:

        buffer.append(datum)
        bytes_in_buffer += datum.size_in_bytes()
        if bytes_in_buffer > max_bytes_per_chunk:
            report_memory_usage(f"{bytes_in_buffer} bytes in chunk.")
            yield buffer
            buffer, bytes_in_buffer = [], 0

    # There will be some data left over, in general.
    if buffer:
        yield buffer


def make_pruned_training_dataset(pruned_data_buffer_generator, pruned_tarfile):
    pruned_data_files = []
    for base_data_list in pruned_data_buffer_generator:
        with tempfile.NamedTemporaryFile(delete=False) as train_data_file:
            ReadsDatum.save_list(base_data_list, train_data_file)
            pruned_data_files.append(train_data_file.name)

    # bundle them in a tarfile
    with tarfile.open(pruned_tarfile, "w") as train_tar:
        for train_file in pruned_data_files:
            train_tar.add(train_file, arcname=os.path.basename(train_file))


def parse_arguments():
    parser = argparse.ArgumentParser(description='train the Mutect3 artifact model')

    add_training_params_to_parser(parser)

    parser.add_argument('--' + constants.CHUNK_SIZE_NAME, type=int, default=int(2e9), required=False,
                        help='size in bytes of output binary data files')

    # input / output
    parser.add_argument('--' + constants.TRAIN_TAR_NAME, type=str, required=True,
                        help='tarfile of training/validation datasets produced by preprocess_dataset.py')
    parser.add_argument('--' + constants.SAVED_MODEL_NAME, type=str, help='Base model from train_permutect_model.py')
    parser.add_argument('--' + constants.OUTPUT_NAME, type=str, required=True, help='path to pruned dataset file')
    parser.add_argument('--' + constants.TENSORBOARD_DIR_NAME, type=str, default='tensorboard', required=False,
                        help='path to output tensorboard directory')

    return parser.parse_args()


def main_without_parsing(args):
    training_params = parse_training_params(args)

    tensorboard_dir = getattr(args, constants.TENSORBOARD_DIR_NAME)
    pruned_tarfile = getattr(args, constants.OUTPUT_NAME)
    chunk_size = getattr(args, constants.CHUNK_SIZE_NAME)
    original_tarfile = getattr(args, constants.TRAIN_TAR_NAME)

    model,  _, _ = load_model(getattr(args, constants.SAVED_MODEL_NAME))

    base_dataset = ReadsDataset(data_tarfile=original_tarfile, num_folds=NUM_FOLDS)

    # generate ReadSets passing pruning
    pruned_data_generator = generate_pruned_data_for_all_folds(base_dataset, model, training_params, tensorboard_dir)

    # generate List[ReadSet]s passing pruning
    pruned_data_buffer_generator = generate_pruned_data_buffers(pruned_data_generator, chunk_size)

    # save as a tarfile dataset
    make_pruned_training_dataset(pruned_data_buffer_generator, pruned_tarfile=pruned_tarfile)


def main():
    args = parse_arguments()
    main_without_parsing(args)


if __name__ == '__main__':
    main()
