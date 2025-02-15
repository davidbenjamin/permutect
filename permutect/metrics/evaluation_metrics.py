import math
from collections import defaultdict
from typing import List

import numpy as np
import torch
from matplotlib import pyplot as plt
import seaborn as sns
from torch.utils.tensorboard import SummaryWriter

from permutect.metrics import plotting
from permutect.metrics.posterior_result import PosteriorResult
from permutect.misc_utils import StreamingAverage
from permutect.utils.enums import Variation, Call, Epoch, Label

MAX_COUNT = 18  # counts above this will be truncated
MAX_LOGIT = 15
NUM_DATA_FOR_TENSORBOARD_PROJECTION = 10000


def round_up_to_nearest_three(x: int):
    return math.ceil(x / 3) * 3


def multiple_of_three_bin_index(x: int):
    return (round_up_to_nearest_three(x)//3) - 1    # -1 because zero is not a bin


MAX_BIN = multiple_of_three_bin_index(MAX_COUNT)

def multiple_of_three_bin_indices(counts: torch.Tensor):
    return (torch.ceil(counts/3) - 1).int()


def multiple_of_three_bin_index_to_count(idx: int):
    return 3 * (idx + 1)


# round logit to nearest int, truncate to range, ending up with bins 0. . . 2*max_logit
def logit_to_bin(logit):
    return min(max(round(logit), -MAX_LOGIT), MAX_LOGIT) + MAX_LOGIT


def bin_center(bin_idx):
    return bin_idx - MAX_LOGIT


NUM_COUNT_BINS = round_up_to_nearest_three(MAX_COUNT) // 3    # zero is not a bin


def make_count_bin_mask(bin_index: int, counts: torch.Tensor):
    assert bin_index < NUM_COUNT_BINS
    count_bin_bottom = 3*bin_index + 1
    count_bin_top = 3*bin_index + 3
    return (count_bin_bottom <= counts) * (counts <= count_bin_top)


# predictions_and_labels is list of (predicted logit, actual label) tuples
# adjustment is the logit threshold that maximizes accuracy -- basically we're trying to find the shift such that
# a logit of 0 expresses 50/50 confidence
# output is the amount to be SUBTRACTED from logit to get a final adjusted logit
def calculate_logit_adjustment(predictions_and_labels, use_harmonic_mean: bool = False):
    _, adjustment = plotting.get_roc_data(predictions_and_labels, given_threshold=None, sens_prec=False, use_harmonic_mean=use_harmonic_mean)
    return adjustment


ALL_SOURCES = -1


class EvaluationMetricsForOneEpochType:
    def __init__(self):
        # indexed by source, then variant type, then count bin, then logit bin
        self.acc_vs_logit = defaultdict(lambda: {
            var_type: [[StreamingAverage() for _ in range(2 * MAX_LOGIT + 1)] for _ in range(NUM_COUNT_BINS)] for
            var_type in Variation})

        # indexed by source, then variant type
        # TODO: unite this with the above using an ALL_COUNTS = -1
        self.acc_vs_logit_all_counts = defaultdict(lambda: {
            var_type: [StreamingAverage() for _ in range(2 * MAX_LOGIT + 1)] for var_type in Variation})

        # indexed by source, then variant type, then Label (artifact vs variant), then count bin
        self.acc_vs_cnt = defaultdict(lambda: {var_type: defaultdict(lambda: [StreamingAverage() for _ in range(NUM_COUNT_BINS)]) for
                      var_type in Variation})

        # source -> variant type -> (predicted logit, actual label)
        self.roc_data = defaultdict(lambda: {var_type: [] for var_type in Variation})

        # source -> variant type, count -> (predicted logit, actual label)
        self.roc_data_by_cnt = defaultdict(lambda: {var_type: [[] for _ in range(NUM_COUNT_BINS)] for var_type in Variation})

        # list of logits for histograms, by variant type, count, label, source
        self.logit_histogram_data_vcls = {var_type: [defaultdict(lambda: defaultdict(list)) for _ in range(NUM_COUNT_BINS)] for var_type in Variation}

        self.all_sources = set()

    # correct_call is boolean -- was the prediction correct?
    # the predicted logit is the logit corresponding to the predicted probability that call in question is an artifact / error
    def record_call(self, variant_type: Variation, predicted_logit: float, label: Label, correct_call, alt_count: int, weight: float = 1.0, source: int = 0):
        count_bin_index = multiple_of_three_bin_index(min(MAX_COUNT, alt_count))
        self.all_sources.add(source)
        self.logit_histogram_data_vcls[variant_type][count_bin_index][label][source].append(predicted_logit)

        if label != Label.UNLABELED:
            self.acc_vs_cnt[source][variant_type][label][count_bin_index].record(correct_call, weight)
            self.acc_vs_cnt[ALL_SOURCES][variant_type][label][count_bin_index].record(correct_call, weight)
            self.acc_vs_logit[source][variant_type][count_bin_index][logit_to_bin(predicted_logit)].record(correct_call, weight)
            self.acc_vs_logit[ALL_SOURCES][variant_type][count_bin_index][logit_to_bin(predicted_logit)].record(correct_call,
                                                                                                           weight)
            self.acc_vs_logit_all_counts[source][variant_type][logit_to_bin(predicted_logit)].record(correct_call, weight)
            self.acc_vs_logit_all_counts[ALL_SOURCES][variant_type][logit_to_bin(predicted_logit)].record(correct_call,
                                                                                                     weight)

            float_label = (1.0 if label == Label.ARTIFACT else 0.0)
            self.roc_data[source][variant_type].append((predicted_logit, float_label))
            self.roc_data[ALL_SOURCES][variant_type].append((predicted_logit, float_label))
            self.roc_data_by_cnt[source][variant_type][count_bin_index].append((predicted_logit, float_label))
            self.roc_data_by_cnt[ALL_SOURCES][variant_type][count_bin_index].append((predicted_logit, float_label))

    # return a list of tuples.  This outer list is over the two labels, Call.SOMATIC and Call.ARTIFACT.  Each tuple consists of
    # (list of alt counts (x axis), list of accuracies (y axis), the label)
    def make_data_for_accuracy_plot(self, var_type: Variation, source: int = ALL_SOURCES):
        non_empty_count_bins_by_label = {
            label: [idx for idx in range(NUM_COUNT_BINS) if not self.acc_vs_cnt[source][var_type][label][idx].is_empty()]
            for label in self.acc_vs_cnt[source][var_type].keys()}

        return [([multiple_of_three_bin_index_to_count(idx) for idx in non_empty_count_bins_by_label[label]],
                    [self.acc_vs_cnt[source][var_type][label][idx].get() for idx in non_empty_count_bins_by_label[label]],
                    label.name) for label in self.acc_vs_cnt[source][var_type].keys()]

    # similar tuple format but now it's (list of logits, list of accuracies, count)
    def make_data_for_calibration_plot(self, var_type: Variation, source: int = ALL_SOURCES):
        non_empty_logit_bins = [
            [idx for idx in range(2 * MAX_LOGIT + 1) if not self.acc_vs_logit[source][var_type][count_idx][idx].is_empty()]
            for count_idx in range(NUM_COUNT_BINS)]
        return [([bin_center(idx) for idx in non_empty_logit_bins[count_idx]],
                                        [self.acc_vs_logit[source][var_type][count_idx][idx].get() for idx in
                                         non_empty_logit_bins[count_idx]],
                                        str(multiple_of_three_bin_index_to_count(count_idx))) for count_idx in
                                       range(NUM_COUNT_BINS)]

    def make_logit_histograms(self):
        fig, axes = plt.subplots(len(Variation), NUM_COUNT_BINS, sharex='all', sharey='all', squeeze=False,
                                 figsize=(2.5 * NUM_COUNT_BINS, 2.5 * len(Variation)), dpi=200)

        multiple_sources = len(self.all_sources) > 1
        source_zero_line_colors = {Label.VARIANT: 'red', Label.ARTIFACT: 'magenta', Label.UNLABELED: 'limegreen'}
        other_source_line_colors = {Label.VARIANT: 'darkred', Label.ARTIFACT: 'darkmagenta', Label.UNLABELED: 'darkgreen'}
        for row, variation_type in enumerate(Variation):
            for count_bin in range(NUM_COUNT_BINS): # this is also the column index
                plot_data = self.logit_histogram_data_vcls[variation_type][count_bin]
                different_labels = plot_data.keys()

                # overlapping density plots for all source / label combinations
                # source 0 is filled; others are not
                ax = axes[row, count_bin]
                for source in self.all_sources:
                    for label in different_labels:
                        line_label = f"{label.name} ({source})" if multiple_sources else label.name
                        color = source_zero_line_colors[label] if source == 0 else other_source_line_colors[label]

                        sns.kdeplot(data=np.clip(np.array(plot_data[label][source]), -10, 10), fill=(source == 0),
                                    color=color, ax=ax, label=line_label, clip=(-10, 10))
                ax.set_ylim(0, 0.5)     # don't go all the way to 1 because
                ax.legend()

        column_names = [str(multiple_of_three_bin_index_to_count(count_idx)) for count_idx in range(NUM_COUNT_BINS)]
        row_names = [var_type.name for var_type in Variation]
        plotting.tidy_subplots(fig, axes, x_label="predicted logit", y_label="frequency", row_labels=row_names, column_labels=column_names)
        return fig, axes

    # now it's (list of logits, list of accuracies)
    def make_data_for_calibration_plot_all_counts(self, var_type: Variation, source: int = ALL_SOURCES):
        non_empty_logit_bins = [idx for idx in range(2 * MAX_LOGIT + 1) if not self.acc_vs_logit_all_counts[source][var_type][idx].is_empty()]
        return ([bin_center(idx) for idx in non_empty_logit_bins],
                    [self.acc_vs_logit_all_counts[source][var_type][idx].get() for idx in non_empty_logit_bins])

    def plot_accuracy(self, var_type: Variation, axis, source: int = ALL_SOURCES):
        acc_vs_cnt_x_y_lab_tuples = self.make_data_for_accuracy_plot(var_type, source)
        plotting.simple_plot_on_axis(axis, acc_vs_cnt_x_y_lab_tuples, None, None)

    def plot_calibration(self, var_type: Variation, axis, source: int = ALL_SOURCES):
        acc_vs_logit_x_y_lab_tuples = self.make_data_for_calibration_plot(var_type, source)
        plotting.simple_plot_on_axis(axis, acc_vs_logit_x_y_lab_tuples, None, None)

    def plot_calibration_all_counts(self, var_type: Variation, axis, source: int = ALL_SOURCES):
        logits_list, accuracies_list = self.make_data_for_calibration_plot_all_counts(var_type, source)
        plotting.simple_plot_on_axis(axis, [(logits_list, accuracies_list, "calibration")], None, None)

    def plot_roc_curve(self, var_type: Variation, axis, given_threshold: float = None, sens_prec: bool = False, source: int = ALL_SOURCES):
        plotting.plot_accuracy_vs_accuracy_roc_on_axis([self.roc_data[source][var_type]], [None], axis, given_threshold, sens_prec)

    def plot_roc_curves_by_count(self, var_type: Variation, axis, given_threshold: float = None, sens_prec: bool = False, source: int = ALL_SOURCES):
        plotting.plot_accuracy_vs_accuracy_roc_on_axis(self.roc_data_by_cnt[source][var_type],
                                                       [str(multiple_of_three_bin_index_to_count(idx)) for idx in
                                                        range(NUM_COUNT_BINS)], axis, given_threshold, sens_prec)

    # return variant type, count bin -> logit adjustment to be subtracted (so that maximum accuracy is at threshold of logit = 0)
    def calculate_logit_adjustments(self, use_harmonic_mean: bool = False, source: int = ALL_SOURCES):
        result = {var_type: [0.0 for _ in range(NUM_COUNT_BINS)] for var_type in Variation}
        for var_type in Variation:
            for cbin in range(NUM_COUNT_BINS):
                data = self.roc_data_by_cnt[source][var_type][cbin]
                if data:    # leave adjustment at 0 if no data
                    result[var_type][cbin] = calculate_logit_adjustment(data, use_harmonic_mean)

        return result


class EvaluationMetrics:
    def __init__(self):
        # we will have a map from epoch type to EvaluationMetricsForOneEpochType
        self.metrics = defaultdict(EvaluationMetricsForOneEpochType)

        # list of (PosteriorResult, Call) tuples
        self.mistakes = []

    # Variation is an IntEnum, so variant_type can also be integer
    # correct_call is boolean -- was the prediction correct?
    # the predicted logit is the logit corresponding to the predicted probability that call in question is an artifact / error
    def record_call(self, epoch_type: Epoch, variant_type: Variation, predicted_logit: float, label: Label, correct_call, alt_count: int, weight: float = 1.0, source: int = 0):
        self.metrics[epoch_type].record_call(variant_type, predicted_logit, label, correct_call, alt_count, weight, source=source)

    # track bad calls when filtering is given an optional evaluation truth VCF
    def record_mistake(self, posterior_result: PosteriorResult, call: Call):
        self.mistakes.append((posterior_result, call))

    def make_mistake_histograms(self, summary_writer: SummaryWriter):
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
        # given_thresholds is a dict from Variation to float (logit-scaled) used in the ROC curves
        epoch_keys = self.metrics.keys()
        sources = next(iter(self.metrics.values())).all_sources
        source_keys = ([ALL_SOURCES] + list(sources)) if (len(sources) > 0) else sources

        num_rows = len(epoch_keys)

        for source in source_keys:
            # grid of figures -- rows are epoch types, columns are variant types
            # each subplot has two line graphs of accuracy vs alt count, one each for artifact, non-artifact
            acc_vs_cnt_fig, acc_vs_cnt_axes = plt.subplots(num_rows, len(Variation), sharex='all', sharey='all', squeeze=False)
            roc_fig, roc_axes = plt.subplots(num_rows, len(Variation), sharex='all', sharey='all', squeeze=False, figsize=(2.5 * len(Variation), 2.5 * len(epoch_keys)), dpi=200)
            cal_fig, cal_axes = plt.subplots(num_rows, len(Variation), sharex='all', sharey='all', squeeze=False)
            cal_fig_all_counts, cal_axes_all_counts = plt.subplots(num_rows, len(Variation), sharex='all', sharey='all', squeeze=False)
            roc_by_cnt_fig, roc_by_cnt_axes = plt.subplots(num_rows, len(Variation), sharex='all', sharey='all', squeeze=False, figsize=(2.5 * len(Variation), 2.5 * len(epoch_keys)), dpi=200)

            for row_idx, key in enumerate(epoch_keys):
                metric = self.metrics[key]
                for var_type in Variation:
                    given_threshold = None if given_thresholds is None else given_thresholds[var_type]
                    metric.plot_accuracy(var_type, acc_vs_cnt_axes[row_idx, var_type], source)
                    metric.plot_calibration(var_type, cal_axes[row_idx, var_type], source)
                    metric.plot_calibration_all_counts(var_type, cal_axes_all_counts[row_idx, var_type], source)
                    metric.plot_roc_curve(var_type, roc_axes[row_idx, var_type], given_threshold, sens_prec, source)
                    metric.plot_roc_curves_by_count(var_type, roc_by_cnt_axes[row_idx, var_type], given_threshold, sens_prec, source)
            # done collecting stats for all loaders and filling in subplots

            nonart_label = "sensitivity" if sens_prec else "non-artifact accuracy"
            art_label = "precision" if sens_prec else "artifact accuracy"

            variation_types = [var_type.name for var_type in Variation]
            row_names = [epoch_type.name for epoch_type in self.metrics.keys()]
            plotting.tidy_subplots(acc_vs_cnt_fig, acc_vs_cnt_axes, x_label="alt count", y_label="accuracy", row_labels=row_names, column_labels=variation_types)
            plotting.tidy_subplots(roc_fig, roc_axes, x_label=nonart_label, y_label=art_label, row_labels=row_names, column_labels=variation_types)
            plotting.tidy_subplots(roc_by_cnt_fig, roc_by_cnt_axes, x_label=nonart_label, y_label=art_label, row_labels=row_names, column_labels=variation_types)
            plotting.tidy_subplots(cal_fig, cal_axes, x_label="predicted logit", y_label="accuracy", row_labels=row_names, column_labels=variation_types)
            plotting.tidy_subplots(cal_fig_all_counts, cal_axes_all_counts, x_label="predicted logit", y_label="accuracy", row_labels=row_names, column_labels=variation_types)

            name_suffix = "" if len(sources) == 1 else (", all sources" if source == ALL_SOURCES else f", source {source}")

            summary_writer.add_figure("accuracy by alt count" + name_suffix, acc_vs_cnt_fig, global_step=epoch)
            summary_writer.add_figure(" accuracy by logit output by count" + name_suffix, cal_fig, global_step=epoch)
            summary_writer.add_figure(" accuracy by logit output" + name_suffix, cal_fig_all_counts, global_step=epoch)
            summary_writer.add_figure(("sensitivity vs precision" if sens_prec else "variant accuracy vs artifact accuracy") + name_suffix, roc_fig, global_step=epoch)
            summary_writer.add_figure(("sensitivity vs precision by alt count" if sens_prec else "variant accuracy vs artifact accuracy by alt count") + name_suffix, roc_by_cnt_fig, global_step=epoch)

        # one more plot, different from the rest.  Here each epoch is its own figure, and within each figure the grid of subplots
        # is by variant type and count.  Within each subplot we have overlapping density plots of artifact logit predictions for all
        # combinations of Label and source
        for key in epoch_keys:
            metric = self.metrics[key]
            hist_fig, hist_ax = metric.make_logit_histograms()
            summary_writer.add_figure(f"logit histograms ({Epoch(key).name})", hist_fig, global_step=epoch)


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
        self.representations = []  # list of 2D tensors (to be stacked into a single 2D tensor), representations over batches

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

        stacked_representations = torch.vstack(self.representations)

        # read average embeddings stratified by variant type
        for variant_type in Variation:
            variant_name = variant_type.name
            indices = set([n for n, type_name in enumerate(self.type_metadata) if type_name == variant_name])

            interesting = interesting_indices & indices
            boring = boring_indices & indices
            boring_count = max(len(interesting) // 3, 100) if is_filter_variants else len(boring)
            boring_to_keep = np.array([int(n) for n in boring])[np.random.choice(len(boring), size=boring_count, replace=False)]
            idx = sample_indices_for_tensorboard(np.hstack((boring_to_keep, np.array([int(n) for n in interesting]))))

            summary_writer.add_embedding(stacked_representations[idx],
                                         metadata=[all_metadata[round(n)] for n in idx.tolist()],
                                         metadata_header=["Labels", "Correctness", "Types", "Counts"],
                                         tag=prefix+"embedding for variant type " + variant_name, global_step=epoch)

        # read average embeddings stratified by alt count
        for count_bin in range(NUM_COUNT_BINS):
            count = multiple_of_three_bin_index_to_count(count_bin)
            indices = set([n for n, alt_count in enumerate(self.truncated_count_metadata) if alt_count == str(count)])
            interesting = interesting_indices & indices
            boring = boring_indices & indices
            boring_count = max(len(interesting) // 3, 100) if is_filter_variants else len(boring)
            boring_to_keep = np.array([int(n) for n in boring])[np.random.choice(len(boring), size=boring_count, replace=False)]
            idx = sample_indices_for_tensorboard(np.hstack((boring_to_keep, np.array([int(n) for n in interesting]))))

            if len(idx) > 0:
                summary_writer.add_embedding(stacked_representations[idx],
                                        metadata=[all_metadata[round(n)] for n in idx.tolist()],
                                        metadata_header=["Labels", "Correctness", "Types", "Counts"],
                                        tag=prefix+"embedding for alt count " + str(count), global_step=epoch)



