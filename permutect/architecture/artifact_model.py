# bug before PyTorch 1.7.1 that warns when constructing ParameterList
import warnings
from collections import defaultdict
from typing import List

import psutil
import torch
from torch import nn, Tensor
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from queue import PriorityQueue


from tqdm.autonotebook import trange, tqdm
from itertools import chain
from matplotlib import pyplot as plt

from permutect.architecture.mlp import MLP
from permutect.architecture.monotonic import MonoDense
from permutect.data.base_datum import ArtifactBatch, DEFAULT_GPU_FLOAT, DEFAULT_CPU_FLOAT
from permutect.data.artifact_dataset import ArtifactDataset
from permutect import utils, constants
from permutect.metrics.evaluation_metrics import LossMetrics, EvaluationMetrics, MAX_COUNT, round_up_to_nearest_three, \
    EmbeddingMetrics, multiple_of_three_bin_index_to_count, multiple_of_three_bin_index
from permutect.parameters import TrainingParameters, ArtifactModelParameters
from permutect.utils import Variation, Epoch, Label
from permutect.metrics import plotting

warnings.filterwarnings("ignore", message="Setting attributes on ParameterList is not supported.")


WORST_OFFENDERS_QUEUE_SIZE = 100


def effective_count(weights: Tensor):
    return (torch.square(torch.sum(weights)) / torch.sum(torch.square(weights))).item()


# group rows into consecutive chunks to yield a 3D tensor, average over dim=1 to get
# 2D tensor of sums within each chunk
def sums_over_chunks(tensor2d: Tensor, chunk_size: int):
    assert len(tensor2d) % chunk_size == 0
    return torch.sum(tensor2d.reshape([len(tensor2d) // chunk_size, chunk_size, -1]), dim=1)


class Calibration(nn.Module):

    def __init__(self, hidden_layer_sizes: List[int]):
        super(Calibration, self).__init__()

        # calibration takes [logit, ref count, alt count] as input and maps it to [calibrated logit]
        # it is monotonically increasing in logit, unconstrained in ref and alt count
        # we initialize it to calibrated logit = input logit

        # likewise, we cap the effective alt and ref counts and input logits to avoid arbitrarily large confidence
        self.max_alt = nn.Parameter(torch.tensor(20.0))
        self.max_ref = nn.Parameter(torch.tensor(20.0))
        self.max_input_logit = nn.Parameter(torch.tensor(20.0))

        center_spacing = 1
        ref_center_spacing = 5

        # centers of Gaussian comb featurizations
        self.alt_centers = torch.arange(start=1, end=20, step=center_spacing)
        self.ref_centers = torch.arange(start=1, end=20, step=ref_center_spacing)

        # increasing in the 1st feature, logits
        # logit is one feature, then the Gaussian comb for alt and ref counts is the other
        self.monotonic = MonoDense(1 + len(self.ref_centers) + len(self.alt_centers), hidden_layer_sizes + [1], 1, 0)

        self.is_turned_on = True

        self.max_alt_count_for_adjustment = 20
        # after training we comopute one final calibration adjustment, which depends on alt count
        # the nth element is the adjustment for alt count n
        # note that this is NOT a parameter!!!! It is *set* but not learned!!
        self.final_adjustments = torch.zeros(self.max_alt_count_for_adjustment + 1)

    def set_adjustments(self, adjustments):
        self.final_adjustments = adjustments
        self.max_alt_count_for_adjustment = len(adjustments) - 1

    def calibrated_logits(self, logits_b: Tensor, ref_counts_b: Tensor, alt_counts_b: Tensor):
        if self.is_turned_on:
            logits_bc = torch.tanh(logits_b / self.max_input_logit)[:, None]

            ref_comb_bc = torch.softmax(-torch.square(ref_counts_b[:, None] - self.ref_centers[None, :]).float(), dim=1)
            alt_comb_bc = torch.softmax(-torch.square(alt_counts_b[:, None] - self.alt_centers[None, :]).float(), dim=1)
            input_2d = torch.hstack([logits_bc, ref_comb_bc, alt_comb_bc])
            calibrated_b = self.monotonic.forward(input_2d).squeeze()

            counts_for_adjustment = torch.clamp(alt_counts_b, max=self.max_alt_count_for_adjustment).long()
            adjustments = self.final_adjustments[counts_for_adjustment]

            return calibrated_b + adjustments
        else:   # should never happen
            return logits_b

    def forward(self, logits, ref_counts: Tensor, alt_counts: Tensor):
        return self.calibrated_logits(logits, ref_counts, alt_counts)

    def plot_calibration(self):
        alt_counts = [1, 3, 5, 10, 15, 20]
        ref_counts = [1, 3, 5, 10, 15, 20]
        logits = torch.range(-10, 10, 0.1)
        cal_fig,cal_axes = plt.subplots(len(alt_counts), len(ref_counts), sharex='all', sharey='all',
                                        squeeze=False, figsize=(10, 6), dpi=100)

        for row_idx, alt_count in enumerate(alt_counts):
            for col_idx, ref_count in enumerate(ref_counts):
                calibrated = self.forward(logits, ref_count * torch.ones_like(logits), alt_count * torch.ones_like(logits))
                plotting.simple_plot_on_axis(cal_axes[row_idx, col_idx], [(logits.detach(), calibrated.detach(), "")], None, None)

        plotting.tidy_subplots(cal_fig, cal_axes, x_label="alt count", y_label="ref count",
                               row_labels=[str(n) for n in ref_counts], column_labels=[str(n) for n in alt_counts])

        return cal_fig, cal_axes


class ArtifactModel(nn.Module):
    """
    aggregation_layers: dimensions of layers for aggregation, excluding its input which is determined by the
    representation model.

    output_layers: dimensions of layers after aggregation, excluding the output dimension,
    which is 1 for a single logit representing artifact/non-artifact.  This is not part of the aggregation layers
    because we have different output layers for each variant type.
    """

    def __init__(self, params: ArtifactModelParameters, num_base_features: int, num_ref_alt_features: int, device=utils.gpu_if_available()):
        super(ArtifactModel, self).__init__()

        self._device = device
        self._dtype = DEFAULT_GPU_FLOAT if device != torch.device("cpu") else DEFAULT_CPU_FLOAT
        self.num_base_features = num_base_features
        self.num_ref_alt_features = num_ref_alt_features
        self.params = params

        # The [1] is for the output logit
        self.aggregation = MLP([num_base_features] + params.aggregation_layers + [1], batch_normalize=params.batch_normalize, dropout_p=params.dropout_p)

        # one Calibration module for each variant type; that is, calibration depends on both count and type
        self.calibration = nn.ModuleList([Calibration(params.calibration_layers) for variant_type in Variation])

        self.to(device=self._device, dtype=self._dtype)

    def training_parameters(self):
        return chain(self.aggregation.parameters(), self.calibration.parameters())

    def calibration_parameters(self):
        return self.calibration.parameters()

    def freeze_all(self):
        utils.freeze(self.parameters())

    def set_epoch_type(self, epoch_type: utils.Epoch):
        if epoch_type == utils.Epoch.TRAIN:
            self.train(True)
            utils.freeze(self.parameters())
            utils.unfreeze(self.training_parameters())
        else:
            self.freeze_all()

    # returns 1D tensor of length batch_size of log odds ratio (logits) between artifact and non-artifact
    def forward(self, batch: ArtifactBatch):
        precalibrated_logits = self.aggregation.forward(batch.get_representations_2d().to(device=self._device, dtype=self._dtype)).reshape(batch.size())
        calibrated_logits = torch.zeros_like(precalibrated_logits)
        one_hot_types_2d = batch.variant_type_one_hot().to(device=self._device, dtype=self._dtype)
        for n, _ in enumerate(Variation):
            mask = one_hot_types_2d[:, n]
            calibrated_logits += mask * self.calibration[n].forward(precalibrated_logits, batch.ref_counts, batch.alt_counts)
        return calibrated_logits, precalibrated_logits

    def learn(self, dataset: ArtifactDataset, training_params: TrainingParameters, summary_writer: SummaryWriter, validation_fold: int = None, epochs_per_evaluation: int = None):
        bce = nn.BCEWithLogitsLoss(reduction='none')  # no reduction because we may want to first multiply by weights for unbalanced data
        train_optimizer = torch.optim.AdamW(self.training_parameters(), lr=training_params.learning_rate, weight_decay=training_params.weight_decay)
        train_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            train_optimizer, factor=0.2, patience=3, threshold=0.001, min_lr=(training_params.learning_rate / 100),verbose=True)

        artifact_to_non_artifact_ratios = torch.from_numpy(dataset.artifact_to_non_artifact_ratios()).to(self._device)

        # balance training by weighting the loss function
        # if total unlabeled is less than total labeled, we do not compensate, since labeled data are more informative
        total_labeled, total_unlabeled = dataset.total_labeled_and_unlabeled()
        labeled_to_unlabeled_ratio = 1 if total_unlabeled < total_labeled else total_labeled / total_unlabeled

        print(f"Training data contains {total_labeled:.0f} labeled examples and {total_unlabeled:.0f} unlabeled examples")
        for variation_type in utils.Variation:
            idx = variation_type.value
            print(f"For variation type {variation_type.name} there are {int(dataset.artifact_totals[idx].item())} labeled artifact examples and {int(dataset.non_artifact_totals[idx].item())} labeled non-artifact examples")

        validation_fold_to_use = (dataset.num_folds - 1) if validation_fold is None else validation_fold
        train_loader = dataset.make_data_loader(dataset.all_but_one_fold(validation_fold_to_use), training_params.batch_size, self._device.type == 'cuda', training_params.num_workers)
        print(f"Train loader created, memory usage percent: {psutil.virtual_memory().percent:.1f}")
        valid_loader = dataset.make_data_loader([validation_fold_to_use], training_params.batch_size, self._device.type == 'cuda', training_params.num_workers)
        print(f"Validation loader created, memory usage percent: {psutil.virtual_memory().percent:.1f}")

        first_epoch, last_epoch = 1, training_params.num_epochs + training_params.num_calibration_epochs
        for epoch in trange(1, last_epoch + 1, desc="Epoch"):
            print(f"Epoch {epoch}, memory usage percent: {psutil.virtual_memory().percent:.1f}")
            is_calibration_epoch = epoch > training_params.num_epochs

            for epoch_type in [utils.Epoch.TRAIN, utils.Epoch.VALID]:
                self.set_epoch_type(epoch_type)
                # in calibration epoch, freeze the model except for calibration
                if is_calibration_epoch and epoch_type == utils.Epoch.TRAIN:
                    utils.freeze(self.parameters())
                    utils.unfreeze(self.calibration_parameters())  # unfreeze calibration but everything else stays frozen

                loss_metrics = LossMetrics(self._device)    # based on calibrated logits
                uncalibrated_loss_metrics = LossMetrics(self._device)  # based on uncalibrated logits

                loader = train_loader if epoch_type == utils.Epoch.TRAIN else valid_loader
                pbar = tqdm(enumerate(loader), mininterval=60)
                for n, batch in pbar:
                    logits, precalibrated_logits = self.forward(batch)
                    types_one_hot = batch.variant_type_one_hot()

                    # if it's a calibration epoch, get count- and variant type-dependent artifact ratio; otherwise it depends only on type
                    if is_calibration_epoch:
                        ratios = [dataset.artifact_to_non_artifact_ratios_by_count(alt_count)[var_type] for alt_count, var_type in zip(batch.alt_counts, batch.variant_types())]
                        non_artifact_weights = torch.FloatTensor(ratios)
                    else:
                        non_artifact_weights = torch.sum(artifact_to_non_artifact_ratios * types_one_hot, dim=1)

                    # TODO: labeled_to_unlabeled ratio should be recalculated based on the weighting of labeled data
                    # maintain the interpretation of the logits as a likelihood ratio by weighting to effectively
                    # achieve a balanced data set eg equal prior between artifact and non-artifact
                    # for artifacts, weight is 1; for non-artifacts it's artifact to nonartifact ratio
                    # for unlabeled data, weight is labeled_to_unlabeled_ratio
                    labeled_weights = batch.labels + (1 - batch.labels) * non_artifact_weights
                    weights = (batch.is_labeled_mask * labeled_weights) + (1 - batch.is_labeled_mask) * labeled_to_unlabeled_ratio

                    uncalibrated_cross_entropies = bce(precalibrated_logits, batch.labels)
                    calibrated_cross_entropies = bce(logits, batch.labels)
                    labeled_losses = batch.is_labeled_mask * (uncalibrated_cross_entropies + calibrated_cross_entropies) / 2

                    # unlabeled loss: entropy regularization e use the uncalibrated logits because otherwise entropy
                    # regularization simply biases calibration to be overconfident.
                    probabilities = torch.sigmoid(precalibrated_logits)
                    entropies = torch.nn.functional.binary_cross_entropy_with_logits(precalibrated_logits, probabilities, reduction='none')
                    unlabeled_losses = (1 - batch.is_labeled_mask) * entropies

                    # these losses include weights and take labeled vs unlabeled into account
                    losses = (labeled_losses + unlabeled_losses) * weights
                    loss = torch.sum(losses)

                    # TODO: we need a parameter to control the relative weight of unlabeled loss to labeled loss
                    # TODO: this interacts with the artifact / non-artifact weighting of labeled data!!!

                    loss_metrics.record_losses(calibrated_cross_entropies.detach(), batch, weights * batch.is_labeled_mask)
                    uncalibrated_loss_metrics.record_losses(uncalibrated_cross_entropies.detach(), batch, weights * batch.is_labeled_mask)
                    uncalibrated_loss_metrics.record_losses(entropies.detach(), batch, weights * (1 - batch.is_labeled_mask))

                    # calibration epochs freeze the model up to calibration, so the unlabeled loss is irrelevant
                    if epoch_type == utils.Epoch.TRAIN and not (is_calibration_epoch and not batch.is_labeled()):
                        utils.backpropagate(train_optimizer, loss)

                # done with one epoch type -- training or validation -- for this epoch
                loss_metrics.write_to_summary_writer(epoch_type, epoch, summary_writer)
                uncalibrated_loss_metrics.write_to_summary_writer(epoch_type, epoch, summary_writer, prefix="uncalibrated")
                if epoch_type == utils.Epoch.TRAIN:
                    train_scheduler.step(loss_metrics.get_labeled_loss())

                print(f"Labeled loss for {epoch_type.name} epoch {epoch}: {loss_metrics.get_labeled_loss():.3f}")
            # done with training and validation for this epoch
            is_last = (epoch == last_epoch)
            if (epochs_per_evaluation is not None and epoch % epochs_per_evaluation == 0) or is_last:
                self.evaluate_model(epoch, dataset, train_loader, valid_loader, summary_writer, collect_embeddings=False, report_worst=False)
            if is_last:
                # collect data in order to do final calibration
                print("collecting data for final calibration")
                evaluation_metrics, _ = self.collect_evaluation_data(dataset, train_loader, valid_loader, report_worst=False)

                logit_adjustments_by_var_type_and_count_bin = evaluation_metrics.metrics[Epoch.VALID].calculate_logit_adjustments(use_harmonic_mean=False)
                print("here are the logit adjustments:")
                for var_type_idx, var_type in enumerate(Variation):
                    adjustments_by_count_bin = logit_adjustments_by_var_type_and_count_bin[var_type]
                    max_bin_idx = len(adjustments_by_count_bin) - 1
                    max_count = multiple_of_three_bin_index_to_count(max_bin_idx)
                    adjustments_by_count = torch.zeros(max_count + 1)
                    for count in range(max_count + 1):
                        bin_idx = multiple_of_three_bin_index(count)
                        # negative sign because these are subtractive adjustments
                        adjustments_by_count[count] = -adjustments_by_count_bin[bin_idx]
                    print(f"for variant type {var_type.name} the adjustments are ")
                    print(adjustments_by_count.tolist())
                    self.calibration[var_type_idx].set_adjustments(adjustments_by_count)

                # consider this an extra post-postprocessing/final calibration epoch, hence epoch+1
                print("doing one final evaluation after the last logit adjustment")
                self.evaluate_model(epoch + 1, dataset, train_loader, valid_loader, summary_writer, collect_embeddings=True, report_worst=True)

            # note that we have not learned the AF spectrum yet
        # done with training

    def evaluate_model_after_training(self, dataset: ArtifactDataset, batch_size, num_workers, summary_writer: SummaryWriter):
        train_loader = dataset.make_data_loader(dataset.all_but_the_last_fold(), batch_size, self._device.type == 'cuda', num_workers)
        valid_loader = dataset.make_data_loader(dataset.last_fold_only(), batch_size, self._device.type == 'cuda', num_workers)
        self.evaluate_model(None, dataset, train_loader, valid_loader, summary_writer, collect_embeddings=True, report_worst=True)

    def collect_evaluation_data(self, dataset: ArtifactDataset, train_loader, valid_loader, report_worst: bool):
        # the keys are tuples of (true label -- 1 for variant, 0 for artifact; rounded alt count)
        worst_offenders_by_truth_and_alt_count = defaultdict(lambda: PriorityQueue(WORST_OFFENDERS_QUEUE_SIZE))

        artifact_to_non_artifact_ratios = torch.from_numpy(dataset.artifact_to_non_artifact_ratios())
        evaluation_metrics = EvaluationMetrics()
        epoch_types = [Epoch.TRAIN, Epoch.VALID]
        for epoch_type in epoch_types:
            assert epoch_type == Epoch.TRAIN or epoch_type == Epoch.VALID  # not doing TEST here
            loader = train_loader if epoch_type == Epoch.TRAIN else valid_loader
            pbar = tqdm(enumerate(filter(lambda bat: bat.is_labeled(), loader)), mininterval=60)
            for n, batch in pbar:

                # these are the same weights used in training to effectively balance the data between artifact and
                # non-artifact for each variant type
                types_one_hot = batch.variant_type_one_hot()
                non_artifact_weights = torch.sum(artifact_to_non_artifact_ratios * types_one_hot, dim=1)
                weights = batch.labels + (1 - batch.labels) * non_artifact_weights

                logits, _ = self.forward(batch)
                pred = logits.detach()
                correct = ((pred > 0) == (batch.labels > 0.5)).tolist()

                for variant_type, predicted_logit, label, correct_call, alt_count, datum, weight in zip(
                        batch.variant_types(), pred.tolist(), batch.labels.tolist(), correct,
                        batch.alt_counts, batch.original_data, weights.tolist()):
                    evaluation_metrics.record_call(epoch_type, variant_type, predicted_logit, label, correct_call, alt_count, weight)
                    if report_worst and not correct_call:
                        rounded_count = round_up_to_nearest_three(alt_count)
                        label_name = Label.ARTIFACT.name if label > 0.5 else Label.VARIANT.name
                        confidence = abs(predicted_logit)

                        # the 0th aka highest priority element in the queue is the one with the lowest confidence
                        pqueue = worst_offenders_by_truth_and_alt_count[(label_name, rounded_count)]

                        # clear space if this confidence is more egregious
                        if pqueue.full() and pqueue.queue[0][0] < confidence:
                            pqueue.get()  # discards the least confident bad call

                        if not pqueue.full():  # if space was cleared or if it wasn't full already
                            variant = datum.get_other_stuff_1d().get_variant()
                            pqueue.put((confidence, str(variant.contig) + ":" + str(
                                variant.position) + ':' + variant.ref + "->" + variant.alt))
            # done with this epoch type
        # done collecting data
        return evaluation_metrics, worst_offenders_by_truth_and_alt_count

    def evaluate_model(self, epoch: int, dataset: ArtifactDataset, train_loader, valid_loader, summary_writer: SummaryWriter,
                                      collect_embeddings: bool = False, report_worst: bool = False):

        # self.freeze_all()
        evaluation_metrics, worst_offenders_by_truth_and_alt_count = self.collect_evaluation_data(dataset, train_loader, valid_loader, report_worst)
        evaluation_metrics.make_plots(summary_writer, epoch=epoch)

        if report_worst:
            for (true_label, rounded_count), pqueue in worst_offenders_by_truth_and_alt_count.items():
                tag = "True label: " + true_label + ", rounded alt count: " + str(rounded_count)

                lines = []
                while not pqueue.empty():   # this goes from least to most egregious, FYI
                    confidence, var_string = pqueue.get()
                    lines.append(f"{var_string} ({confidence:.2f})")

                summary_writer.add_text(tag, "\n".join(lines), global_step=epoch)

        if collect_embeddings:
            embedding_metrics = EmbeddingMetrics()
            ref_alt_seq_metrics = EmbeddingMetrics()

            # now go over just the validation data and generate feature vectors / metadata for tensorboard projectors (UMAP)
            pbar = tqdm(enumerate(filter(lambda bat: bat.is_labeled(), valid_loader)), mininterval=60)

            for n, batch in pbar:
                logits, _ = self.forward(batch)
                pred = logits.detach()
                correct = ((pred > 0) == (batch.labels > 0.5)).tolist()

                for (metrics, embedding) in [(embedding_metrics, batch.get_representations_2d().detach()),
                                              (ref_alt_seq_metrics, batch.get_ref_alt_seq_embeddings_2d().detach())]:
                    metrics.label_metadata.extend(["artifact" if x > 0.5 else "non-artifact" for x in batch.labels.tolist()])
                    metrics.correct_metadata.extend([str(val) for val in correct])
                    metrics.type_metadata.extend([Variation(idx).name for idx in batch.variant_types()])
                    metrics.truncated_count_metadata.extend([str(round_up_to_nearest_three(min(MAX_COUNT, alt_count))) for alt_count in batch.alt_counts])
                    metrics.representations.append(embedding)
            embedding_metrics.output_to_summary_writer(summary_writer, epoch=epoch)
            ref_alt_seq_metrics.output_to_summary_writer(summary_writer, prefix="ref alt seq ", epoch=epoch)

        # done collecting data

    def save(self, path, artifact_log_priors, artifact_spectra):
        torch.save({
            constants.STATE_DICT_NAME: self.state_dict(),
            constants.NUM_BASE_FEATURES_NAME: self.num_base_features,
            constants.NUM_REF_ALT_FEATURES_NAME: self.num_ref_alt_features,
            constants.HYPERPARAMS_NAME: self.params,
            constants.ARTIFACT_LOG_PRIORS_NAME: artifact_log_priors,
            constants.ARTIFACT_SPECTRA_STATE_DICT_NAME: artifact_spectra.state_dict()
        }, path)


# log artifact priors and artifact spectra may be None
def load_artifact_model(path) -> ArtifactModel:
    saved = torch.load(path)
    model_params = saved[constants.HYPERPARAMS_NAME]
    num_base_features = saved[constants.NUM_BASE_FEATURES_NAME]
    num_ref_alt_features = saved[constants.NUM_REF_ALT_FEATURES_NAME]
    model = ArtifactModel(model_params, num_base_features, num_ref_alt_features)
    model.load_state_dict(saved[constants.STATE_DICT_NAME])

    artifact_log_priors = saved[constants.ARTIFACT_LOG_PRIORS_NAME]     # possibly None
    artifact_spectra_state_dict = saved[constants.ARTIFACT_SPECTRA_STATE_DICT_NAME]     #possibly None
    return model, artifact_log_priors, artifact_spectra_state_dict

