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
from permutect.architecture.dna_sequence_convolution import DNASequenceConvolution
from permutect.data.read_set import ReadSetBatch
from permutect.data.read_set_dataset import ReadSetDataset
from permutect import utils
from permutect.metrics.evaluation_metrics import LossMetrics, EvaluationMetrics, NUM_COUNT_BINS, \
    multiple_of_three_bin_index_to_count, multiple_of_three_bin_index, MAX_COUNT, round_up_to_nearest_three
from permutect.utils import Variation, Epoch
from permutect.metrics import plotting

warnings.filterwarnings("ignore", message="Setting attributes on ParameterList is not supported.")

NUM_DATA_FOR_TENSORBOARD_PROJECTION = 10000


def effective_count(weights: Tensor):
    return (torch.square(torch.sum(weights)) / torch.sum(torch.square(weights))).item()


# group rows into consecutive chunks to yield a 3D tensor, average over dim=1 to get
# 2D tensor of sums within each chunk
def sums_over_chunks(tensor2d: Tensor, chunk_size: int):
    assert len(tensor2d) % chunk_size == 0
    return torch.sum(tensor2d.reshape([len(tensor2d) // chunk_size, chunk_size, -1]), dim=1)


class ArtifactModelParameters:
    def __init__(self, representation_dimension: int, aggregation_layers: List[int], calibration_layers: List[int],
                 dropout_p: float = 0.0, batch_normalize: bool = False, learning_rate: float = 0.001, weight_decay: float = 0.01):
        self.representation_dimension = representation_dimension
        self.aggregation_layers = aggregation_layers
        self.calibration_layers = calibration_layers
        self.dropout_p = dropout_p
        self.batch_normalize = batch_normalize
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay


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

    def __init__(self, params: ArtifactModelParameters, num_read_features: int, num_info_features: int, ref_sequence_length: int, device=torch.device("cpu")):
        super(ArtifactModel, self).__init__()

        self._device = device
        self._representation_dimension = params.representation_dimension

        # we have a different aggregation subnetwork for each variant type.  The [1] is for the output logit
        self.refined_dimension = params.aggregation_layers[-1]
        self.rho = nn.ModuleList([MLP([params.representation_dimension] + params.aggregation_layers + [1], batch_normalize=params.batch_normalize,
                       dropout_p=params.dropout_p) for variant_type in Variation])
        self.rho.to(self._device)

        # one Calibration module for each variant type; that is, calibration depends on both count and type
        self.calibration = nn.ModuleList([Calibration(params.calibration_layers) for variant_type in Variation])
        self.calibration.to(self._device)

    def num_read_features(self) -> int:
        return self._num_read_features

    def num_info_features(self) -> int:
        return self._num_info_features

    def ref_sequence_length(self) -> int:
        return self._ref_sequence_length

    def training_parameters(self):
        return chain(self.rho.parameters(), self.calibration.parameters())

    def training_parameters_if_using_pretrained_model(self):
        return chain(self.rho.parameters(), self.final_logit.parameters(), self.calibration.parameters())

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
    def forward(self, batch: ReadSetBatch):
        transformed_reads = self.apply_transformer_to_reads(batch)
        return self.forward_from_transformed_reads(transformed_reads=transformed_reads, batch=batch)


    # input: reads that have passed through the alt/ref transformers
    # output: reads that have been refined through the rho subnetwork that sees the transformed alts along with ref, alt, and info
    def forward_from_transformed_reads_to_refined_reads(self, transformed_reads: Tensor, batch: ReadSetBatch, weight_range: float = 0, extra_output=False):
        weights = torch.ones(len(transformed_reads), 1, device=self._device) if weight_range == 0 else (1 + weight_range * (1 - 2 * torch.rand(len(transformed_reads), 1, device=self._device)))

        ref_count, alt_count = batch.ref_count, min(batch.alt_count, self.alt_downsample)
        total_ref, total_alt = ref_count * batch.size(), alt_count * batch.size()

        alt_wts = weights[total_ref:]
        alt_wt_sums = sums_over_chunks(alt_wts, alt_count)
        alt_wt_sq_sums = sums_over_chunks(torch.square(alt_wts), alt_count)

        # mean embedding of every read, alt and ref, at each datum
        all_read_means = ((0 if ref_count == 0 else sums_over_chunks(transformed_reads[:total_ref], ref_count)) + sums_over_chunks(transformed_reads[total_ref:], alt_count)) / (alt_count + ref_count)
        omega_info = self.omega(batch.get_info_2d().to(self._device))
        ref_seq_embedding = self.ref_seq_cnn(batch.get_ref_sequences_2d())

        # dimension is batch_size x ref transformer output size + omega output size + ref seq embedding size
        extra_tensor_2d = torch.hstack([all_read_means, omega_info, ref_seq_embedding])
        extra_tensor_2d = torch.repeat_interleave(extra_tensor_2d, repeats=alt_count, dim=0)

        # the alt reads have not been averaged yet to we need to copy each row of the extra tensor batch.alt_count times
        padded_transformed_alt_reads = torch.hstack([transformed_reads[total_ref:], extra_tensor_2d])

        # now we refine the alt reads -- no need to separate between read sets yet as this broadcasts over every read
        refined_alt = torch.zeros(total_alt, self.refined_dimension)
        one_hot_types_2d = batch.variant_type_one_hot()
        for n, _ in enumerate(Variation):
            # multiply the result of this variant type's aggregation layers by its
            # one-hot mask.  Yes, this is wasteful because we apply every aggregation to every datum.
            # TODO: make this more efficient.
            mask = torch.repeat_interleave(one_hot_types_2d[:, n], repeats=alt_count).reshape(total_alt, 1)  # 2D column tensor
            refined_alt += mask * self.rho[n].forward(padded_transformed_alt_reads)

        weighted_refined = alt_wts * refined_alt
        # weighted mean is sum of reads in a chunk divided by sum of weights in same chunk
        alt_means = sums_over_chunks(weighted_refined, alt_count) / alt_wt_sums

        # these are fed to the calibration, since reweighting effectively reduces the read counts
        effective_alt_counts = torch.square(alt_wt_sums) / alt_wt_sq_sums

        # 3D tensors -- batch size x alt count x transformed / refined dimension
        # TODO: maybe I should output these regardless -- they probably get garbage-collected all the same
        # TODO: with no hit in performance
        transformed_alt_output = transformed_reads[total_ref:].reshape(batch.size(), alt_count, -1) if extra_output else None
        refined_alt_output = refined_alt.reshape(batch.size(), alt_count, -1) if extra_output else None

        # note all_read_means pertains to transformer; alt_means pertains to refined
        return all_read_means, alt_means, omega_info, ref_seq_embedding, effective_alt_counts, transformed_alt_output, refined_alt_output

    def forward_from_transformed_reads_to_calibration(self, phi_reads: Tensor, batch: ReadSetBatch, weight_range: float = 0):
        all_read_means, alt_means, omega_info, ref_seq_embedding, effective_alt_counts, _, _ = self.forward_from_transformed_reads_to_refined_reads(phi_reads, batch, weight_range)

        logits = self.final_logit(alt_means).reshape(batch.size())

        return logits, batch.ref_count * torch.ones_like(effective_alt_counts), effective_alt_counts

    def forward_from_transformed_reads(self, transformed_reads: Tensor, batch: ReadSetBatch, weight_range: float = 0):
        logits, ref_counts, effective_alt_counts = self.forward_from_transformed_reads_to_calibration(transformed_reads, batch, weight_range)

        result = torch.zeros_like(logits)

        one_hot_types_2d = batch.variant_type_one_hot()
        for n, _ in enumerate(Variation):
            mask = one_hot_types_2d[:, n]
            result += mask * self.calibration[n].forward(logits, ref_counts, effective_alt_counts)

        return result

    def train_model(self, dataset: ReadSetDataset, num_epochs, num_calibration_epochs, batch_size, num_workers,
                    summary_writer: SummaryWriter, reweighting_range: float, hyperparams: ArtifactModelParameters,
                    validation_fold: int = None, freeze_lower_layers: bool = False):
        bce = nn.BCEWithLogitsLoss(reduction='none')  # no reduction because we may want to first multiply by weights for unbalanced data
        train_optimizer = torch.optim.AdamW(self.training_parameters_if_using_pretrained_model() if freeze_lower_layers else self.training_parameters(), lr=hyperparams.learning_rate, weight_decay=hyperparams.weight_decay)
        calibration_optimizer = torch.optim.AdamW(self.calibration_parameters(), lr=hyperparams.learning_rate, weight_decay=hyperparams.weight_decay)

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
        train_loader = dataset.make_data_loader(dataset.all_but_one_fold(validation_fold_to_use), batch_size, self._device.type == 'cuda', num_workers)
        valid_loader = dataset.make_data_loader([validation_fold_to_use], batch_size, self._device.type == 'cuda', num_workers)

        for epoch in trange(1, num_epochs + 1 + num_calibration_epochs, desc="Epoch"):
            is_calibration_epoch = epoch > num_epochs
            for epoch_type in ([utils.Epoch.VALID] if is_calibration_epoch else [utils.Epoch.TRAIN, utils.Epoch.VALID]):
                self.set_epoch_type(epoch_type)
                if is_calibration_epoch:
                    utils.unfreeze(self.calibration_parameters())  # unfreeze calibration but everything else stays frozen

                loss_metrics = LossMetrics(self._device)

                loader = train_loader if epoch_type == utils.Epoch.TRAIN else valid_loader
                pbar = tqdm(enumerate(loader), mininterval=60)
                for n, batch in pbar:
                    transformed_reads = self.apply_transformer_to_reads(batch)

                    logits = self.forward_from_transformed_reads(transformed_reads, batch, weight_range=reweighting_range)
                    types_one_hot = batch.variant_type_one_hot()
                    log_prior_ratios = torch.sum(artifact_to_non_artifact_log_prior_ratios * types_one_hot, dim=1)
                    posterior_logits = logits + log_prior_ratios

                    if batch.is_labeled():
                        separate_losses = bce(posterior_logits, batch.labels)
                        loss = torch.sum(separate_losses)

                        loss_metrics.record_total_batch_loss(loss.detach(), batch)
                        loss_metrics.record_separate_losses(separate_losses, batch)
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

    # generators by name is eg {"train": train_generator, "valid": valid_generator}
    def evaluate_model_after_training(self, dataset, batch_size, num_workers, summary_writer: SummaryWriter):
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

                for variant_type, predicted_logit, label, correct_call in zip(batch.variant_types(), pred.tolist(), batch.labels.tolist(), correct):
                    evaluation_metrics.record_call(epoch_type, variant_type, predicted_logit, label, correct_call, batch.alt_count)
            # done with this epoch type
        # done collecting data

        evaluation_metrics.make_plots(summary_writer)

        # now go over just the validation data and generate feature vectors / metadata for tensorboard projectors (UMAP)
        # also generate histograms of magnitudes of average read embeddings
        pbar = tqdm(enumerate(filter(lambda bat: bat.is_labeled(), valid_loader)), mininterval=60)

        # things we will collect for the projections
        label_metadata = []     # list (extended by each batch) 1 if artifact, 0 if not
        correct_metadata = []   # list (extended by each batch), 1 if correct prediction, 0 if not
        type_metadata = []      # list of lists, strings of variant type
        truncated_count_metadata = []   # list of lists
        average_read_embedding_features = []    # list of 2D tensors (to be stacked into a single 2D tensor), average read embedding over batches
        info_embedding_features = []
        ref_seq_embedding_features = []

        # we output array figures of histograms.  The rows of each histogram are (rounded) alt counts, the columns
        # are variant types, and each subplot has overlapping histograms for non-artifacts and artifacts.

        # There are several figures of this type, which for now we enocde as tuples.  The first element is 0, 1, 2 for L0, L1, or L2 norm
        # the second is a boolean norm first for whether we take the norm then the means (if true) or vice versa (if false)
        # the third is a boolean for transformed (True) or refined (False)
        norm_types = [1, 2, 100]
        norm_first_values = [True, False]
        trans_or_refined_values = [True, False]
        histogram_tuples = [(x, y, z) for x in norm_types for y in norm_first_values for z in trans_or_refined_values]

        hists_by_type_cnt = [{var_type: [([], []) for _ in range(NUM_COUNT_BINS)] for var_type in Variation} for tup in histogram_tuples]

        for n, batch in pbar:
            types_one_hot = batch.variant_type_one_hot()
            log_prior_odds = torch.sum(log_artifact_to_non_artifact_ratios * types_one_hot, dim=1)

            pred = self.forward(batch)
            posterior_pred = pred + log_prior_odds
            correct = ((posterior_pred > 0) == (batch.labels > 0.5)).tolist()

            label_metadata.extend(["artifact" if x > 0.5 else "non-artifact" for x in batch.labels.tolist()])
            correct_metadata.extend([str(val) for val in correct])
            type_metadata.extend([Variation(idx).name for idx in batch.variant_types()])
            truncated_count_metadata.extend(batch.size() * [str(round_up_to_nearest_three(min(MAX_COUNT, batch.alt_count)))])

            read_embeddings = self.apply_transformer_to_reads(batch)

            all_read_means, alt_means, omega_info, ref_seq_embedding, effective_alt_counts, transformed_alt_3d, refined_alt_3d = \
                self.forward_from_transformed_reads_to_refined_reads(read_embeddings, batch, extra_output=True)
            average_read_embedding_features.append(alt_means)
            info_embedding_features.append(omega_info)
            ref_seq_embedding_features.append(ref_seq_embedding)

            # code for the norm histograms
            for n, (norm_type, norm_first, transformed_vs_refined) in enumerate(histogram_tuples):
                tensor_3d = transformed_alt_3d if transformed_vs_refined else refined_alt_3d

                # norm over representation dimension (2), then mean over read index (1) if norm_first
                # else mean over read index (1) then norm over the new representation dimension (1)
                output_1d = (tensor_3d.norm(dim=2, p=norm_type).mean(dim=1) if norm_first else \
                    tensor_3d.mean(dim=1).norm(dim=1, p=norm_type)) if norm_type < 100 else \
                    (tensor_3d.max(dim=2).values.mean(dim=1) if norm_first else tensor_3d.mean(dim=1).max(dim=1).values)

                for variant_type,  label, output_value, in zip(batch.variant_types(), batch.labels.tolist(), output_1d):
                    bin_idx = multiple_of_three_bin_index(min(MAX_COUNT, batch.alt_count))
                    is_artifact_index = 0 if label < 0.5 else 1
                    hists_by_type_cnt[n][variant_type][bin_idx][is_artifact_index].append(output_value.item())

        # done collecting data

        for n, (norm_type, norm_first, transformed_vs_refined) in enumerate(histogram_tuples):
            hist_fig, hist_axes = plt.subplots(NUM_COUNT_BINS, len(Variation), sharex='none', sharey='none',
                                                   squeeze=False, figsize=(15, 15), dpi=100)

            name1 = "L" + str(norm_type) + " norm"
            name2 = ("mean of " + name1) if norm_first else (name1 + " of mean")
            name3 = name2 + " transformed" if transformed_vs_refined else " refined"

            for col_idx, var_type in enumerate(Variation):
                for row_idx in range(NUM_COUNT_BINS):
                    # data for one particular subplot (row = count bin, column = variant type)
                    plotting.simple_histograms_on_axis(hist_axes[row_idx, col_idx], hists_by_type_cnt[n][var_type][row_idx], ["good", "bad"], 25)
            variation_types = [var_type.name for var_type in Variation]
            plotting.tidy_subplots(hist_fig, hist_axes, x_label=name3, y_label="",
                                   row_labels=["alt count " + str(multiple_of_three_bin_index_to_count(idx)) for idx in range(NUM_COUNT_BINS)],
                                   column_labels=variation_types, keep_axes_tick_labels=True)
            summary_writer.add_figure(name3, hist_fig)

        # downsample to a reasonable amount of UMAP data
        all_metadata=list(zip(label_metadata, correct_metadata, type_metadata, truncated_count_metadata))
        idx = np.random.choice(len(all_metadata), size=min(NUM_DATA_FOR_TENSORBOARD_PROJECTION, len(all_metadata)), replace=False)

        summary_writer.add_embedding(torch.vstack(average_read_embedding_features)[idx],
                                     metadata=[all_metadata[n] for n in idx],
                                     metadata_header=["Labels", "Correctness", "Types", "Counts"],
                                     tag="mean read embedding")

        summary_writer.add_embedding(torch.vstack(info_embedding_features)[idx],
                                     metadata=[all_metadata[n] for n in idx],
                                     metadata_header=["Labels", "Correctness", "Types", "Counts"],
                                     tag="info embedding")

        summary_writer.add_embedding(torch.vstack(ref_seq_embedding_features)[idx],
                                     metadata=[all_metadata[n] for n in idx],
                                     metadata_header=["Labels", "Correctness", "Types", "Counts"],
                                     tag="ref seq embedding")

        # read average embeddings stratified by variant type
        for variant_type in Variation:
            variant_name = variant_type.name
            indices = [n for n, type_name in enumerate(type_metadata) if type_name == variant_name]
            indices = sample_indices_for_tensorboard(indices)
            summary_writer.add_embedding(torch.vstack(average_read_embedding_features)[indices],
                                         metadata=[all_metadata[n] for n in indices],
                                         metadata_header=["Labels", "Correctness", "Types", "Counts"],
                                         tag="mean read embedding for variant type " + variant_name)

        # read average embeddings stratified by alt count
        for row_idx in range(NUM_COUNT_BINS):
            count = multiple_of_three_bin_index_to_count(row_idx)
            indices = [n for n, alt_count in enumerate(truncated_count_metadata) if alt_count == str(count)]
            indices = sample_indices_for_tensorboard(indices)
            if len(indices) > 0:
                summary_writer.add_embedding(torch.vstack(average_read_embedding_features)[indices],
                                        metadata=[all_metadata[n] for n in indices],
                                        metadata_header=["Labels", "Correctness", "Types", "Counts"],
                                        tag="mean read embedding for alt count " + str(count))


def sample_indices_for_tensorboard(indices: List[int]):
    indices_np = np.array(indices)

    if len(indices_np) <= NUM_DATA_FOR_TENSORBOARD_PROJECTION:
        return indices_np

    idx = np.random.choice(len(indices_np), size=NUM_DATA_FOR_TENSORBOARD_PROJECTION, replace=False)
    return indices_np[idx]

