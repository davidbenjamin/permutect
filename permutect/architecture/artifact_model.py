# bug before PyTorch 1.7.1 that warns when constructing ParameterList
import warnings
from typing import List

import torch
from torch import nn, Tensor
import numpy as np
from torch.utils.tensorboard import SummaryWriter


from tqdm.autonotebook import trange, tqdm
from itertools import chain
from matplotlib import pyplot as plt

from permutect.architecture.mlp import MLP
from permutect.architecture.monotonic import MonoDense
from permutect.architecture.overdispersed_binomial_mixture import OverdispersedBinomialMixture
from permutect.data.base_datum import ArtifactBatch, DEFAULT_GPU_FLOAT, DEFAULT_CPU_FLOAT
from permutect.data.artifact_dataset import ArtifactDataset
from permutect import utils, constants
from permutect.metrics.evaluation_metrics import LossMetrics, EvaluationMetrics, NUM_COUNT_BINS, \
    multiple_of_three_bin_index_to_count, MAX_COUNT, round_up_to_nearest_three, EmbeddingMetrics
from permutect.parameters import TrainingParameters, ArtifactModelParameters
from permutect.utils import Variation, Epoch
from permutect.metrics import plotting

warnings.filterwarnings("ignore", message="Setting attributes on ParameterList is not supported.")


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
        # it is monotonically increasing in each input
        # we initialize it to calibrated logit = input logit

        # likewise, we cap the effective alt and ref counts to avoid arbitrarily large confidence
        self.max_alt = nn.Parameter(torch.tensor(20.0))
        self.max_ref = nn.Parameter(torch.tensor(20.0))

        # because calibration is monotonic in the magnitude of the logit, not the logit itself i.e. more reads pushes
        # the logit away from zero, not simply up, we have two monotonic networks, one for positive and one for negative

        # for positive input logit the calibrated logit grows more positive with the input and the read support
        self.monotonic_positive = MonoDense(3, hidden_layer_sizes + [1], 3, 0)

        # for negative input logit the calibrated logit grows more negative with the read support
        self.monotonic_negative = MonoDense(3, hidden_layer_sizes + [1], 1, 2)  # monotonically increasing in each input

    def calibrated_logits(self, logits: Tensor, ref_counts: Tensor, alt_counts: Tensor):

        # scale counts and make everything batch size x 1 column tensors
        ref_eff = torch.tanh(ref_counts / self.max_ref).reshape(-1, 1)
        alt_eff = torch.tanh(alt_counts / self.max_alt).reshape(-1, 1)
        logits_2d = logits.reshape(-1, 1)
        input_2d = torch.hstack([logits_2d, ref_eff, alt_eff])

        is_positive = torch.where(logits > 0, 1.0, 0.0)
        return self.monotonic_positive.forward(input_2d).squeeze() * is_positive + self.monotonic_negative.forward(input_2d).squeeze() * (1 - is_positive)

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

    def __init__(self, params: ArtifactModelParameters, num_base_features: int, device=utils.gpu_if_available()):
        super(ArtifactModel, self).__init__()

        self._device = device
        self._dtype = DEFAULT_GPU_FLOAT if device != torch.device("cpu") else DEFAULT_CPU_FLOAT
        self.num_base_features = num_base_features
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
        logits = self.aggregation.forward(batch.get_representations_2d().to(device=self._device, dtype=self._dtype)).reshape(batch.size())

        calibrated = torch.zeros_like(logits)
        one_hot_types_2d = batch.variant_type_one_hot().to(device=self._device, dtype=self._dtype)
        for n, _ in enumerate(Variation):
            mask = one_hot_types_2d[:, n]
            calibrated += mask * self.calibration[n].forward(logits, batch.ref_counts, batch.alt_counts)
        return calibrated

    def learn(self, dataset: ArtifactDataset, training_params: TrainingParameters, summary_writer: SummaryWriter, validation_fold: int = None):
        bce = nn.BCEWithLogitsLoss(reduction='none')  # no reduction because we may want to first multiply by weights for unbalanced data
        train_optimizer = torch.optim.AdamW(self.training_parameters(), lr=training_params.learning_rate, weight_decay=training_params.weight_decay)
        calibration_optimizer = torch.optim.AdamW(self.calibration_parameters(), lr=training_params.learning_rate, weight_decay=training_params.weight_decay)

        artifact_to_non_artifact_ratios = torch.from_numpy(dataset.artifact_to_non_artifact_ratios()).to(self._device)
        artifact_to_non_artifact_log_prior_ratios = torch.log(artifact_to_non_artifact_ratios)

        # balance training by weighting the loss function
        # if total unlabeled is less than total labeled, we do not compensate, since labeled data are more informative
        total_labeled, total_unlabeled = dataset.total_labeled_and_unlabeled()
        labeled_to_unlabeled_ratio = 1 if total_unlabeled < total_labeled else total_labeled / total_unlabeled

        print("Training data contains {} labeled examples and {} unlabeled examples".format(total_labeled, total_unlabeled))
        for variation_type in utils.Variation:
            idx = variation_type.value
            print("For variation type {}, there are {} labeled artifact examples and {} labeled non-artifact examples"
                  .format(variation_type.name, dataset.artifact_totals[idx].item(), dataset.non_artifact_totals[idx].item()))

        validation_fold_to_use = (dataset.num_folds - 1) if validation_fold is None else validation_fold
        train_loader = dataset.make_data_loader(dataset.all_but_one_fold(validation_fold_to_use), training_params.batch_size, self._device.type == 'cuda', training_params.num_workers)
        valid_loader = dataset.make_data_loader([validation_fold_to_use], training_params.batch_size, self._device.type == 'cuda', training_params.num_workers)

        for epoch in trange(1, training_params.num_epochs + 1 + training_params.num_calibration_epochs, desc="Epoch"):
            is_calibration_epoch = epoch > training_params.num_epochs
            for epoch_type in ([utils.Epoch.VALID] if is_calibration_epoch else [utils.Epoch.TRAIN, utils.Epoch.VALID]):
                self.set_epoch_type(epoch_type)
                if is_calibration_epoch:
                    utils.unfreeze(self.calibration_parameters())  # unfreeze calibration but everything else stays frozen

                loss_metrics = LossMetrics(self._device)

                loader = train_loader if epoch_type == utils.Epoch.TRAIN else valid_loader
                pbar = tqdm(enumerate(loader), mininterval=60)
                for n, batch in pbar:
                    logits = self.forward(batch)
                    types_one_hot = batch.variant_type_one_hot()
                    log_prior_ratios = torch.sum(artifact_to_non_artifact_log_prior_ratios * types_one_hot, dim=1)
                    posterior_logits = logits + log_prior_ratios

                    if batch.is_labeled():
                        separate_losses = bce(posterior_logits, batch.labels)
                        loss = torch.sum(separate_losses)

                        loss_metrics.record_total_batch_loss(loss.detach(), batch)
                        loss_metrics.record_losses_by_type_and_count(separate_losses, batch)
                    else:
                        # unlabeled loss: entropy regularization
                        posterior_probabilities = torch.sigmoid(posterior_logits)
                        entropies = torch.nn.functional.binary_cross_entropy_with_logits(posterior_logits, posterior_probabilities, reduction='none')

                        # TODO: we need a parameter to control the relative weight of unlabeled loss to labeled loss
                        loss = torch.sum(entropies) * labeled_to_unlabeled_ratio
                        loss_metrics.record_total_batch_loss(loss.detach(), batch)

                    if epoch_type == utils.Epoch.TRAIN:
                        utils.backpropagate(train_optimizer, loss)
                    if is_calibration_epoch:
                        utils.backpropagate(calibration_optimizer, loss)

                # done with one epoch type -- training or validation -- for this epoch
                loss_metrics.write_to_summary_writer(epoch_type, epoch, summary_writer)

                print("Labeled loss for epoch " + str(epoch) + " of " + epoch_type.name + ": " + str(loss_metrics.get_labeled_loss()))
            # done with training and validation for this epoch
            # note that we have not learned the AF spectrum yet
        # done with training

    def evaluate_model_after_training(self, dataset: ArtifactDataset, batch_size, num_workers, summary_writer: SummaryWriter):
        train_loader = dataset.make_data_loader(dataset.all_but_the_last_fold(), batch_size, self._device.type == 'cuda', num_workers)
        valid_loader = dataset.make_data_loader(dataset.last_fold_only(), batch_size, self._device.type == 'cuda', num_workers)
        epoch_types = [Epoch.TRAIN, Epoch.VALID]
        self.freeze_all()
        self.cpu()
        self._device = "cpu"

        log_artifact_to_non_artifact_ratios = torch.from_numpy(np.log(dataset.artifact_to_non_artifact_ratios()))
        evaluation_metrics = EvaluationMetrics()
        for epoch_type in epoch_types:
            assert epoch_type == Epoch.TRAIN or epoch_type == Epoch.VALID   # not doing TEST here
            loader = train_loader if epoch_type == Epoch.TRAIN else valid_loader
            pbar = tqdm(enumerate(filter(lambda bat: bat.is_labeled(), loader)), mininterval=60)
            for n, batch in pbar:
                # In training we minimize the cross entropy loss wrt the posterior probability, accounting for priors;
                # Here we are considering artifacts and non-artifacts separately and use the uncorrected likelihood logits
                pred = self.forward(batch)
                correct = ((pred > 0) == (batch.labels > 0.5)).tolist()

                for variant_type, predicted_logit, label, correct_call, alt_count in zip(batch.variant_types(), pred.tolist(), batch.labels.tolist(), correct, batch.alt_counts):
                    evaluation_metrics.record_call(epoch_type, variant_type, predicted_logit, label, correct_call, alt_count)
            # done with this epoch type
        # done collecting data

        evaluation_metrics.make_plots(summary_writer)

        embedding_metrics = EmbeddingMetrics()

        # now go over just the validation data and generate feature vectors / metadata for tensorboard projectors (UMAP)
        pbar = tqdm(enumerate(filter(lambda bat: bat.is_labeled(), valid_loader)), mininterval=60)

        for n, batch in pbar:
            types_one_hot = batch.variant_type_one_hot()
            log_prior_odds = torch.sum(log_artifact_to_non_artifact_ratios * types_one_hot, dim=1)

            pred = self.forward(batch)
            posterior_pred = pred + log_prior_odds
            correct = ((posterior_pred > 0) == (batch.labels > 0.5)).tolist()

            embedding_metrics.label_metadata.extend(["artifact" if x > 0.5 else "non-artifact" for x in batch.labels.tolist()])
            embedding_metrics.correct_metadata.extend([str(val) for val in correct])
            embedding_metrics.type_metadata.extend([Variation(idx).name for idx in batch.variant_types()])
            embedding_metrics.truncated_count_metadata.extend([str(round_up_to_nearest_three(min(MAX_COUNT, alt_count))) for alt_count in batch.alt_counts])
            embedding_metrics.representations.append(batch.get_representations_2d())
        embedding_metrics.output_to_summary_writer(summary_writer)

        # done collecting data

    def save(self, path, artifact_log_priors, artifact_spectra):
        torch.save({
            constants.STATE_DICT_NAME: self.state_dict(),
            constants.NUM_BASE_FEATURES_NAME: self.num_base_features,
            constants.HYPERPARAMS_NAME: self.params,
            constants.ARTIFACT_LOG_PRIORS_NAME: artifact_log_priors,
            constants.ARTIFACT_SPECTRA_STATE_DICT_NAME: artifact_spectra.state_dict()
        }, path)


# log artifact priors and artifact spectra may be None
def load_artifact_model(path) -> ArtifactModel:
    saved = torch.load(path)
    model_params = saved[constants.HYPERPARAMS_NAME]
    num_base_features = saved[constants.NUM_BASE_FEATURES_NAME]
    model = ArtifactModel(model_params, num_base_features)
    model.load_state_dict(saved[constants.STATE_DICT_NAME])

    artifact_log_priors = saved[constants.ARTIFACT_LOG_PRIORS_NAME]     # possibly None
    artifact_spectra_state_dict = saved[constants.ARTIFACT_SPECTRA_STATE_DICT_NAME]     #possibly None
    return model, artifact_log_priors, artifact_spectra_state_dict

