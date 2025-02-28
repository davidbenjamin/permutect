import math
from collections import defaultdict
from itertools import chain
from typing import List

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter

from permutect.data.batch import Batch
from permutect.data.batch_indexing import BatchIndices
from permutect.metrics import plotting
from permutect.metrics.loss_metrics import AccuracyMetrics
from permutect.data.count_binning import NUM_ALT_COUNT_BINS, count_from_ref_bin_index, NUM_REF_COUNT_BINS, \
    ref_count_bin_name, count_from_alt_bin_index, alt_count_bin_name
from permutect.metrics.posterior_result import PosteriorResult
from permutect.misc_utils import gpu_if_available
from permutect.utils.enums import Variation, Call, Epoch, Label

NUM_DATA_FOR_TENSORBOARD_PROJECTION = 10000


class EvaluationMetrics:
    def __init__(self, num_sources, device=gpu_if_available()):
        # we will have a map from epoch type to EvaluationMetricsForOneEpochType
        self.metrics = defaultdict(lambda: AccuracyMetrics(num_sources, device))

        # list of (PosteriorResult, Call) tuples
        self.mistakes = []
        self.has_been_sent_to_cpu = False

    def put_on_cpu(self):
        """
        Do this at the end of an epoch so that the whole tensor is on CPU in one operation rather than computing various
        marginals etc on GPU and sending them each to CPU for plotting etc.
        :return:
        """
        for metric in self.metrics.values():
            metric.put_on_cpu()
        self.has_been_sent_to_cpu = True
        return self

    # TODO: currently doesn't record unlabeled data at all
    def record_batch(self, epoch_type: Epoch, batch_indices: BatchIndices, weights: Tensor = None):
        assert not self.has_been_sent_to_cpu, "Can't record after already sending to CPU"
        is_labeled = batch_indices.labels != Label.UNLABELED
        weights_with_labeled_mask = is_labeled * (weights if weights is not None else torch.ones_like(batch_indices.sources))
        self.metrics[epoch_type].record(batch_indices, weights_with_labeled_mask)

    # track bad calls when filtering is given an optional evaluation truth VCF
    def record_mistake(self, posterior_result: PosteriorResult, call: Call):
        self.mistakes.append((posterior_result, call))

    def make_mistake_histograms(self, summary_writer: SummaryWriter):
        assert self.has_been_sent_to_cpu, "Can't make plots before sending to CPU"
        # indexed by call then var_type, inner is a list of posterior results with that call and var type
        posterior_result_mistakes_by_call_and_var_type = defaultdict(lambda: defaultdict(list))
        for posterior_result, call in self.mistakes:
            posterior_result_mistakes_by_call_and_var_type[call][posterior_result.variant_type].append(posterior_result)

        mistake_calls = posterior_result_mistakes_by_call_and_var_type.keys()
        num_rows = len(mistake_calls)

        af_fig, af_axes = plt.subplots(num_rows, len(Variation), sharex='all', sharey='none', squeeze=False)
        logit_fig, logit_axes = plt.subplots(num_rows, len(Variation), sharex='all', sharey='none', squeeze=False)
        ac_fig, ac_axes = plt.subplots(num_rows, len(Variation), sharex='all', sharey='none', squeeze=False)
        prob_fig, prob_axes = plt.subplots(num_rows, len(Variation), sharex='all', sharey='none', squeeze=False)

        for row_idx, mistake_call in enumerate(mistake_calls):
            for var_type in Variation:
                posterior_results = posterior_result_mistakes_by_call_and_var_type[mistake_call][var_type]

                af_data = [pr.alt_count / pr.depth for pr in posterior_results]
                plotting.simple_histograms_on_axis(af_axes[row_idx, var_type], [af_data], [""], 20)

                ac_data = [pr.alt_count for pr in posterior_results]
                plotting.simple_histograms_on_axis(ac_axes[row_idx, var_type], [ac_data], [""], 20)

                logit_data = [pr.artifact_logit for pr in posterior_results]
                plotting.simple_histograms_on_axis(logit_axes[row_idx, var_type], [logit_data], [""], 20)

                # posterior probability assigned to this incorrect call
                prob_data = [pr.posterior_probabilities[mistake_call] for pr in posterior_results]
                plotting.simple_histograms_on_axis(prob_axes[row_idx, var_type], [prob_data], [""], 20)

        variation_types = [var_type.name for var_type in Variation]
        row_names = [mistake.name for mistake in mistake_calls]

        plotting.tidy_subplots(af_fig, af_axes, x_label="alt allele fraction", y_label="", row_labels=row_names, column_labels=variation_types)
        plotting.tidy_subplots(ac_fig, ac_axes, x_label="alt count", y_label="", row_labels=row_names,
                               column_labels=variation_types)
        plotting.tidy_subplots(logit_fig, logit_axes, x_label="artifact logit", y_label="", row_labels=row_names,
                               column_labels=variation_types)
        plotting.tidy_subplots(prob_fig, prob_axes, x_label="mistake call probability", y_label="", row_labels=row_names,
                               column_labels=variation_types)

        summary_writer.add_figure("mistake allele fractions", af_fig)
        summary_writer.add_figure("mistake alt counts", ac_fig)
        summary_writer.add_figure("mistake artifact logits", logit_fig)
        summary_writer.add_figure("probability assigned to mistake calls", prob_fig)

    def make_plots(self, summary_writer: SummaryWriter, given_thresholds=None, sens_prec: bool = False, epoch: int = None):
        assert self.has_been_sent_to_cpu, "Can't make plots before sending to CPU"
        # given_thresholds is a dict from Variation to float (logit-scaled) used in the ROC curves
        num_sources = next(iter(self.metrics.values())).num_sources
        ref_count_bins = list(range(NUM_REF_COUNT_BINS)) + [None]
        alt_count_bins = list(range(NUM_ALT_COUNT_BINS)) + [None]
        ref_count_names = [ref_count_bin_name(bin_idx) for bin_idx in range(NUM_REF_COUNT_BINS)] + ["ALL"]
        alt_count_names = [alt_count_bin_name(bin_idx) for bin_idx in range(NUM_ALT_COUNT_BINS)] + ["ALL"]

        for epoch_type, metric in self.metrics.items():
            for source in chain(range(num_sources), [None]):
                acc_fig, acc_axes = plt.subplots(2, len(Variation), sharex='all', sharey='all', squeeze=False, figsize=(2.5 * len(Variation), 2.5 * 2))
                cal_fig, cal_axes = plt.subplots(len(ref_count_bins), len(alt_count_bins), sharex='all', sharey='all', squeeze=False)
                roc_fig, roc_axes = plt.subplots(len(ref_count_bins), len(alt_count_bins), sharex='all', sharey='all', squeeze=False, dpi=200)

                # make accuracy plots: overall figure is rows = label, columns = variant
                # each subplot is color map of accuracy where x is ref count, y is alt count
                accuracy_rows = [Label.VARIANT, Label.ARTIFACT]
                accuracy_row_names = [label.name for label in accuracy_rows]
                common_colormesh = None
                for row, label in enumerate(accuracy_rows):
                    for col, var_type in enumerate(Variation):
                        common_colormesh = metric.plot_accuracy(label, var_type, acc_axes[row, col], source)
                acc_fig.colorbar(common_colormesh)

                for row, ref_count_bin in enumerate(ref_count_bins):
                    for col, alt_count_bin in enumerate(alt_count_bins):
                        metric.plot_calibration(cal_axes[row, col], ref_count_bin, alt_count_bin, source)
                        metric.plot_roc(roc_axes[row, col], ref_count_bin, alt_count_bin, source, given_thresholds, sens_prec)

                nonart_label = "sensitivity" if sens_prec else "non-artifact accuracy"
                art_label = "precision" if sens_prec else "artifact accuracy"

                variation_types = [var_type.name for var_type in Variation]
                plotting.tidy_subplots(acc_fig, acc_axes, x_label="alt count", y_label="ref count", row_labels=accuracy_row_names, column_labels=variation_types)
                plotting.tidy_subplots(roc_fig, roc_axes, x_label=nonart_label, y_label=art_label, row_labels=ref_count_names, column_labels=alt_count_names)
                plotting.tidy_subplots(cal_fig, cal_axes, x_label="logit", y_label="accuracy", row_labels=ref_count_names, column_labels=alt_count_names)

                name_suffix = epoch_type.name + "" if num_sources == 1 else (", all sources" if source is None else f", source {source}")

                summary_writer.add_figure("accuracy by alt count " + name_suffix, acc_fig, global_step=epoch)
                summary_writer.add_figure(" accuracy by logit output by count" + name_suffix, cal_fig, global_step=epoch)
                summary_writer.add_figure(("sensitivity vs precision" if sens_prec else "variant accuracy vs artifact accuracy") + name_suffix, roc_fig, global_step=epoch)

            # One more plot.  In each figure the grid of subplots is by variant type and count.  Within each subplot we have
            # overlapping density plots of artifact logit predictions for all combinations of Label and source
            hist_fig, hist_ax = metric.make_logit_histograms()
            summary_writer.add_figure(f"logit histograms ({epoch_type.name})", hist_fig, global_step=epoch)


def sample_indices_for_tensorboard(indices: List[int]):
    indices_np = np.array(indices)

    if len(indices_np) <= NUM_DATA_FOR_TENSORBOARD_PROJECTION:
        return indices_np

    idx = np.random.choice(len(indices_np), size=NUM_DATA_FOR_TENSORBOARD_PROJECTION, replace=False)
    return indices_np[idx]


class EmbeddingMetrics:
    TRUE_POSITIVE = "true-positive"
    FALSE_POSITIVE = "false-positive"
    TRUE_NEGATIVE_ARTIFACT = "true-negative-artifact"   # distinguish these because artifact and eg germline should embed differently
    TRUE_NEGATIVE_NONARTIFACT = "true-negative-nonartifact"
    TRUE_NEGATIVE = "true-negative"
    FALSE_NEGATIVE_ARTIFACT = "false-negative-artifact"
    TRUE_NEGATIVE_SEQ_ERROR = "true-negative-seq-error"

    def __init__(self):
        # things we will collect for the projections
        self.label_metadata = []  # list (extended by each batch) 1 if artifact, 0 if not
        self.correct_metadata = []  # list (extended by each batch), 1 if correct prediction, 0 if not
        self.type_metadata = []  # list of lists, strings of variant type
        self.truncated_count_metadata = []  # list of lists
        self.features = []  # list of 2D tensors (to be stacked into a single 2D tensor), features over batches

    def output_to_summary_writer(self, summary_writer: SummaryWriter, prefix: str = "", is_filter_variants: bool = False, epoch: int = None):
        # downsample to a reasonable amount of UMAP data
        all_metadata = list(zip(self.label_metadata, self.correct_metadata, self.type_metadata, self.truncated_count_metadata))

        indices_by_correct_status = defaultdict(list)

        for n, correct_status in enumerate(self.correct_metadata):
            indices_by_correct_status[correct_status].append(n)

        # note that if we don't have labeled truth, everything is boring
        all_indices = set(range(len(all_metadata)))
        interesting_indices = set(indices_by_correct_status[EmbeddingMetrics.TRUE_POSITIVE] +
                                       indices_by_correct_status[EmbeddingMetrics.FALSE_POSITIVE] +
                                       indices_by_correct_status[EmbeddingMetrics.FALSE_NEGATIVE_ARTIFACT])
        boring_indices = all_indices - interesting_indices

        '''if is_filter_variants:
            boring_indices = np.array(indices_by_correct_status["unknown"] + indices_by_correct_status[EmbeddingMetrics.TRUE_NEGATIVE_ARTIFACT])

            # if we have labeled truth, keep a few "boring" true negatives around; otherwise we only have "unknown"s
            boring_count = len(interesting_indices) // 3 if len(interesting_indices) > 0 else len(boring_indices)
            boring_to_keep = boring_indices[np.random.choice(len(boring_indices), size=boring_count, replace=False)]
            idx = np.hstack((boring_to_keep, interesting_indices))

        idx = np.random.choice(len(all_metadata), size=min(NUM_DATA_FOR_TENSORBOARD_PROJECTION, len(all_metadata)), replace=False)
'''

        stacked_features = torch.vstack(self.features)

        # read average embeddings stratified by variant type
        for variant_type in Variation:
            variant_name = variant_type.name
            indices = set([n for n, type_name in enumerate(self.type_metadata) if type_name == variant_name])

            interesting = interesting_indices & indices
            boring = boring_indices & indices
            boring_count = max(len(interesting) // 3, 100) if is_filter_variants else len(boring)
            boring_to_keep = np.array([int(n) for n in boring])[np.random.choice(len(boring), size=boring_count, replace=False)]
            idx = sample_indices_for_tensorboard(np.hstack((boring_to_keep, np.array([int(n) for n in interesting]))))

            summary_writer.add_embedding(stacked_features[idx],
                                         metadata=[all_metadata[round(n)] for n in idx.tolist()],
                                         metadata_header=["Labels", "Correctness", "Types", "Counts"],
                                         tag=prefix+"embedding for variant type " + variant_name, global_step=epoch)

        # read average embeddings stratified by alt count
        for count_bin in range(NUM_ALT_COUNT_BINS):
            count = count_from_alt_bin_index(count_bin)
            indices = set([n for n, alt_count in enumerate(self.truncated_count_metadata) if alt_count == str(count)])
            interesting = interesting_indices & indices
            boring = boring_indices & indices
            boring_count = max(len(interesting) // 3, 100) if is_filter_variants else len(boring)
            boring_to_keep = np.array([int(n) for n in boring])[np.random.choice(len(boring), size=boring_count, replace=False)]
            idx = sample_indices_for_tensorboard(np.hstack((boring_to_keep, np.array([int(n) for n in interesting]))))

            if len(idx) > 0:
                summary_writer.add_embedding(stacked_features[idx],
                                        metadata=[all_metadata[round(n)] for n in idx.tolist()],
                                        metadata_header=["Labels", "Correctness", "Types", "Counts"],
                                        tag=prefix+"embedding for alt count " + str(count), global_step=epoch)



