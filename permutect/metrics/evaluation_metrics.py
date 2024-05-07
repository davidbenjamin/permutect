import math
from collections import defaultdict

import torch
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter

from permutect import utils
from permutect.data.read_set import ReadSetBatch, Variant
from permutect.metrics import plotting
from permutect.utils import Variation, Call, Epoch

MAX_COUNT = 18  # counts above this will be truncated
MAX_LOGIT = 6


def round_up_to_nearest_three(x: int):
    return math.ceil(x / 3) * 3


def multiple_of_three_bin_index(x: int):
    return (round_up_to_nearest_three(x)//3) - 1    # -1 because zero is not a bin


def multiple_of_three_bin_index_to_count(idx: int):
    return 3 * (idx + 1)


# round logit to nearest int, truncate to range, ending up with bins 0. . . 2*max_logit
def logit_to_bin(logit):
    return min(max(round(logit), -MAX_LOGIT), MAX_LOGIT) + MAX_LOGIT


def bin_center(bin_idx):
    return bin_idx - MAX_LOGIT

NUM_COUNT_BINS = round_up_to_nearest_three(MAX_COUNT) // 3    # zero is not a bin




# keep track of losses during training of artifact model
class LossMetrics:
    def __init__(self, device):
        self.device = device

        self.labeled_loss = utils.StreamingAverage(device=self._device)
        self.unlabeled_loss = utils.StreamingAverage(device=self._device)

        self.labeled_loss_by_type = {variant_type: utils.StreamingAverage(device=self._device) for variant_type in Variation}
        self.labeled_loss_by_count = {bin_idx: utils.StreamingAverage(device=self._device) for bin_idx in range(NUM_COUNT_BINS)}

    def get_labeled_loss(self):
        return self.labeled_loss.get()

    def get_unlabeled_loss(self):
        return self.unlabeled_loss.get()

    def write_to_summary_writer(self, epoch_type: utils.Epoch, epoch: int, summary_writer: SummaryWriter):
        summary_writer.add_scalar(epoch_type.name + "/Labeled Loss", self.labeled_loss.get(), epoch)
        summary_writer.add_scalar(epoch_type.name + "/Unlabeled Loss", self.unlabeled_loss.get(), epoch)

        for bin_idx, loss in self.labeled_loss_by_count.items():
            summary_writer.add_scalar(
                epoch_type.name + "/Labeled Loss/By Count/" + str(multiple_of_three_bin_index_to_count(bin_idx)), loss.get(), epoch)

        for var_type, loss in self.labeled_loss_by_type.items():
            summary_writer.add_scalar(epoch_type.name + "/Labeled Loss/By Type/" + var_type.name, loss.get(), epoch)

    def record_total_batch_loss(self, total_loss: float, batch: ReadSetBatch):
        if batch.is_labeled():
            self.labeled_loss.record_sum(total_loss, batch.size())

            if batch.alt_count <= MAX_COUNT:
                self.labeled_loss_by_count[multiple_of_three_bin_index(batch.alt_count)].record_sum(total_loss, batch.size())
        else:
            self.unlabeled_loss.record_sum(total_loss, batch.size())


    def record_separate_losses(self, losses: torch.Tensor, batch: ReadSetBatch):
        if batch.is_labeled():
            types_one_hot = batch.variant_type_one_hot()
            losses_masked_by_type = losses.reshape(batch.size(), 1) * types_one_hot
            counts_by_type = torch.sum(types_one_hot, dim=0)
            total_loss_by_type = torch.sum(losses_masked_by_type, dim=0)
            variant_types = list(Variation)
            for variant_type_idx in range(len(Variation)):
                count_for_type = int(counts_by_type[variant_type_idx].item())
                loss_for_type = total_loss_by_type[variant_type_idx].item()
                self.labeled_loss_by_type[variant_types[variant_type_idx]].record_sum(loss_for_type, count_for_type)


class EvaluationMetricsForOneEpochType:
    def __init__(self):
        # indexed by variant type, then count bin, then logit bin
        self.acc_vs_logit = {
            var_type: [[utils.StreamingAverage() for _ in range(2 * MAX_LOGIT + 1)] for _ in range(NUM_COUNT_BINS)] for
            var_type in Variation}

        # indexed by variant type, then call type (artifact vs variant), then count bin
        self.acc_vs_cnt = {var_type: defaultdict(lambda: [utils.StreamingAverage() for _ in range(NUM_COUNT_BINS)]) for
                      var_type in Variation}

        # variant type -> (predicted logit, actual label)
        self.roc_data = {var_type: [] for var_type in Variation}

        # variant type, count -> (predicted logit, actual label)
        self.roc_data_by_cnt = {var_type: [[] for _ in range(NUM_COUNT_BINS)] for var_type in Variation}

    # Variant is an IntEnum, so variant_type can also be integer
    # label is 1 for artifact / error; 0 for non-artifact / true variant
    # correct_call is boolean -- was the prediction correct?
    # the predicted logit is the logit corresponding to the predicted probability that call in question is an artifact / error
    def record_call(self, variant_type: Variation, predicted_logit: float, label: float, correct_call, alt_count: int):
        count_bin_index = multiple_of_three_bin_index(min(MAX_COUNT, alt_count))
        self.acc_vs_cnt[variant_type][Call.SOMATIC if label < 0.5 else Call.ARTIFACT][count_bin_index].record(correct_call)
        self.acc_vs_logit[variant_type][count_bin_index][logit_to_bin(predicted_logit)].record(correct_call)

        self.roc_data[variant_type].append((predicted_logit, label))
        self.roc_data_by_cnt[variant_type][count_bin_index].append((predicted_logit, label))

    # return a list of tuples.  This outer list is over the two labels, Call.SOMATIC and Call.ARTIFACT.  Each tuple consists of
    # (list of alt counts (x axis), list of accuracies (y axis), the label)
    def make_data_for_accuracy_plot(self, var_type: Variation):
        non_empty_count_bins_by_label = {
            label: [idx for idx in range(NUM_COUNT_BINS) if not self.acc_vs_cnt[var_type][label][idx].is_empty()]
            for label in self.acc_vs_cnt[var_type].keys()}

        return [([multiple_of_three_bin_index_to_count(idx) for idx in non_empty_count_bins_by_label[label]],
                    [self.acc_vs_cnt[var_type][label][idx].get() for idx in non_empty_count_bins_by_label[label]],
                    label.name) for label in self.acc_vs_cnt[var_type].keys()]

    # similar tuple format but now it's (list of logits, list of accuracies, count)
    def make_data_for_calibration_plot(self, var_type: Variation):
        non_empty_logit_bins = [
            [idx for idx in range(2 * MAX_LOGIT + 1) if not self.acc_vs_logit[var_type][count_idx][idx].is_empty()]
            for count_idx in range(NUM_COUNT_BINS)]
        return [([bin_center(idx) for idx in non_empty_logit_bins[count_idx]],
                                        [self.acc_vs_logit[var_type][count_idx][idx].get() for idx in
                                         non_empty_logit_bins[count_idx]],
                                        str(multiple_of_three_bin_index_to_count(count_idx))) for count_idx in
                                       range(NUM_COUNT_BINS)]

    def plot_accuracy(self, var_type: Variation, axis):
        acc_vs_cnt_x_y_lab_tuples = self.make_data_for_accuracy_plot(var_type)
        plotting.simple_plot_on_axis(axis, acc_vs_cnt_x_y_lab_tuples, None, None)

    def plot_calibration(self, var_type: Variation, axis):
        acc_vs_logit_x_y_lab_tuples = self.make_data_for_calibration_plot(var_type)
        plotting.simple_plot_on_axis(axis, acc_vs_logit_x_y_lab_tuples, None, None)

    def plot_roc_curve(self, var_type: Variation, axis):
        plotting.plot_accuracy_vs_accuracy_roc_on_axis([self.roc_data[var_type]], [None], axis)

    def plot_roc_curves_by_count(self, var_type: Variation, axis):
        plotting.plot_accuracy_vs_accuracy_roc_on_axis(self.roc_data_by_cnt[var_type],
                                                       [str(multiple_of_three_bin_index_to_count(idx)) for idx in
                                                        range(NUM_COUNT_BINS)], axis)


class EvaluationMetrics:
    def __init__(self):
        # we will have a map from epoch type to EvaluationMetricsForOneEpochType
        self.metrics = defaultdict(EvaluationMetricsForOneEpochType)

    # Variant is an IntEnum, so variant_type can also be integer
    # label is 1 for artifact / error; 0 for non-artifact / true variant
    # correct_call is boolean -- was the prediction correct?
    # the predicted logit is the logit corresponding to the predicted probability that call in question is an artifact / error
    def record_call(self, epoch_type: Epoch, variant_type: Variation, predicted_logit: float, label: float, correct_call, alt_count: int):
        self.metrics[epoch_type].record_call(variant_type, predicted_logit, label, correct_call, alt_count)

    def make_plots(self, summary_writer: SummaryWriter):
        keys = self.metrics.keys()
        num_rows = len(keys)
        # grid of figures -- rows are epoch types, columns are variant types
        # each subplot has two line graphs of accuracy vs alt count, one each for artifact, non-artifact
        acc_vs_cnt_fig, acc_vs_cnt_axes = plt.subplots(num_rows, len(Variation), sharex='all', sharey='all', squeeze=False)
        roc_fig, roc_axes = plt.subplots(num_rows, len(Variation), sharex='all', sharey='all', squeeze=False)
        cal_fig, cal_axes = plt.subplots(num_rows, len(Variation), sharex='all', sharey='all', squeeze=False)
        roc_by_cnt_fig, roc_by_cnt_axes = plt.subplots(num_rows, len(Variation), sharex='all', sharey='all', squeeze=False, figsize=(4 * len(Variation), 4 * len(keys)), dpi=200)

        for row_idx, key in enumerate(keys):
            metric = self.metrics[key]
            for var_type in Variation:
                metric.plot_accuracy(var_type, acc_vs_cnt_axes[row_idx, var_type])
                metric.plot_calibration(var_type, cal_axes[row_idx, var_type])
                metric.plot_roc_curve(var_type, roc_axes[row_idx, var_type])
                metric.plot_roc_curves_by_count(var_type, roc_by_cnt_axes[row_idx, var_type])
        # done collecting stats for all loaders and filling in subplots

        variation_types = [var_type.name for var_type in Variation]
        row_names = [epoch_type.name for epoch_type in self.metrics.keys()]
        plotting.tidy_subplots(acc_vs_cnt_fig, acc_vs_cnt_axes, x_label="alt count", y_label="accuracy", row_labels=row_names, column_labels=variation_types)
        plotting.tidy_subplots(roc_fig, roc_axes, x_label="non-artifact accuracy", y_label="artifact accuracy", row_labels=row_names, column_labels=variation_types)
        plotting.tidy_subplots(roc_by_cnt_fig, roc_by_cnt_axes, x_label="non-artifact accuracy", y_label="artifact accuracy", row_labels=row_names, column_labels=variation_types)
        plotting.tidy_subplots(cal_fig, cal_axes, x_label="predicted logit", y_label="accuracy", row_labels=row_names, column_labels=variation_types)

        summary_writer.add_figure("accuracy by alt count", acc_vs_cnt_fig)
        summary_writer.add_figure(" accuracy by logit output", cal_fig)
        summary_writer.add_figure(" variant accuracy vs artifact accuracy curve", roc_fig)
        summary_writer.add_figure(" variant accuracy vs artifact accuracy curves by alt count", roc_by_cnt_fig)



