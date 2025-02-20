from enum import IntEnum
from typing import Tuple

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter

from permutect.data.batch import Batch
from permutect.data.datum import Datum
from permutect.data.count_binning import NUM_REF_COUNT_BINS, NUM_ALT_COUNT_BINS, \
    NUM_LOGIT_BINS, logit_bin_indices, top_of_logit_bin, logits_from_bin_indices, logit_bin_name, count_bin_indices, \
    count_bin_index, counts_from_bin_indices, count_bin_name, ALT_COUNT_BIN_BOUNDS, REF_COUNT_BIN_BOUNDS
from permutect.metrics import plotting
from permutect.misc_utils import gpu_if_available
from permutect.utils.array_utils import add_to_6d_array, select_and_sum
from permutect.utils.enums import Variation, Epoch, Label


class BatchProperty(IntEnum):
    SOURCE = (0, None)
    LABEL = (1, [label.name for label in Label])
    VARIANT_TYPE = (2, [var_type.name for var_type in Variation])
    REF_COUNT_BIN = (3, [count_bin_name(idx) for idx in range(NUM_REF_COUNT_BINS)])
    ALT_COUNT_BIN = (4, [count_bin_name(idx) for idx in range(NUM_ALT_COUNT_BINS)])
    LOGIT_BIN = (5, [logit_bin_name(idx) for idx in range(NUM_LOGIT_BINS)])

    def __new__(cls, value, names_list):
        member = int.__new__(cls, value)
        member._value_ = value
        member.names_list = names_list
        return member

    def get_name(self, n: int):
        return str(n) if self.names_list is None else self.names_list[n]


class BatchIndexedTotals:
    NUM_DIMS = 6
    """
    stores sums, indexed by batch properties source (s), label (l), variant type (v), ref count (r), alt count (a), logit (g)
    """
    def __init__(self, num_sources: int, device=gpu_if_available(), include_logits: bool = False):
        self.totals_slvrag = torch.zeros((num_sources, len(Label), len(Variation), NUM_REF_COUNT_BINS,
                                          NUM_ALT_COUNT_BINS, NUM_LOGIT_BINS), device=device)
        assert self.totals_slvrag.dim() == len(BatchProperty)
        self.include_logits = include_logits
        self.device = device
        self.has_been_sent_to_cpu = False
        self.num_sources = num_sources

    def put_on_cpu(self):
        """
        Do this at the end of an epoch so that the whole tensor is on CPU in one operation rather than computing various
        marginals etc on GPU and sending them each to CPU for plotting etc.
        :return:
        """
        self.totals_slvrag = self.totals_slvrag.cpu()
        self.device = torch.device('cpu')
        self.has_been_sent_to_cpu = True
        return self

    def resize_sources(self, new_num_sources):
        old_num_sources = self.num_sources
        if new_num_sources < old_num_sources:
            self.totals_slvrag = self.totals_slvrag[:new_num_sources]
        elif new_num_sources > old_num_sources:
            new_totals = torch.zeros((new_num_sources, len(Label), len(Variation), NUM_REF_COUNT_BINS,
                                      NUM_ALT_COUNT_BINS, NUM_LOGIT_BINS), device=self.totals_slvrag.device)
            new_totals[:old_num_sources] = self.totals_slvrag
            self.totals_slvrag = new_totals
        self.num_sources = new_num_sources

    def record_datum(self, datum: Datum, value: float = 1.0, grow_source_if_necessary: bool = True):
        source = datum.get_source()
        if source >= self.num_sources:
            if grow_source_if_necessary:
                self.resize_sources(source + 1)
            else:
                raise Exception("Datum source doesn't fit.")
        # no logits here
        ref_idx, alt_idx = count_bin_index(datum.get_ref_count()), count_bin_index(datum.get_alt_count())
        self.totals_slvrag[source, datum.get_label(), datum.get_variant_type(), ref_idx, alt_idx, 0] += value

    def record(self, batch: Batch, logits: torch.Tensor, values: torch.Tensor):
        # values is a 1D tensor
        assert batch.size() == len(values)
        assert not self.has_been_sent_to_cpu, "Can't record after already sending to CPU"
        sources = batch.get_sources()
        logit_indices = logit_bin_indices(logits) if self.include_logits else torch.zeros_like(sources)

        add_to_6d_array(self.totals_slvrag, sources, batch.get_labels(), batch.get_variant_types(),
                        count_bin_indices(batch.get_ref_counts()), count_bin_indices(batch.get_alt_counts()), logit_indices, values)

    def get_totals(self) -> torch.Tensor:
        return self.totals_slvrag

    def get_marginal(self, *properties: Tuple[BatchProperty, ...]) -> torch.Tensor:
        """
        sum over all but one or more batch properties.
        For example self.get_marginal(BatchProperty.SOURCE, BatchProperty.LABEL) yields a (num sources x len(Label)) output
        """
        property_set = set(*properties)
        other_dims = tuple(n for n in range(BatchIndexedTotals.NUM_DIMS) if n not in property_set)
        return torch.sum(self.totals_slvrag, dim=other_dims)

    def make_logit_histograms(self):
        assert self.has_been_sent_to_cpu, "Can't make plots before sending to CPU"
        fig, axes = plt.subplots(len(Variation), NUM_ALT_COUNT_BINS, sharex='all', sharey='all', squeeze=False,
                                 figsize=(2.5 * NUM_ALT_COUNT_BINS, 2.5 * len(Variation)), dpi=200)
        x_axis_logits = logits_from_bin_indices(torch.tensor(range(NUM_LOGIT_BINS)))

        num_sources = self.totals_slvrag.shape[BatchProperty.SOURCE]
        multiple_sources = num_sources > 1
        source_zero_line_colors = {Label.VARIANT: 'red', Label.ARTIFACT: 'magenta', Label.UNLABELED: 'limegreen'}
        other_source_line_colors = {Label.VARIANT: 'darkred', Label.ARTIFACT: 'darkmagenta', Label.UNLABELED: 'darkgreen'}
        for row, variation_type in enumerate(Variation):
            for count_bin in range(NUM_ALT_COUNT_BINS): # this is also the column index
                selection={BatchProperty.VARIANT_TYPE: variation_type, BatchProperty.ALT_COUNT_BIN: count_bin}
                totals_slg = select_and_sum(self.totals_slvrag, select=selection, sum=(BatchProperty.REF_COUNT_BIN,))

                # overlapping line plots for all source / label combinations
                # source 0 is filled; others are not
                ax = axes[row, count_bin]
                x_y_label_tuples = []
                for source in range(num_sources):
                    for label in Label:
                        totals_g = totals_slg[source, label, :]
                        if torch.sum(totals_g).item() >= 1:
                            line_label = f"{label.name} ({source})" if multiple_sources else label.name
                            # TODO: rearrange this old code to get colors right?
                            # TODO: or maybe automatic colors are fine
                            #color = source_zero_line_colors[label] if source == 0 else other_source_line_colors[label]
                            x_y_label_tuples.append((x_axis_logits.cpu().numpy(), totals_g.cpu().numpy(), line_label))
                plotting.simple_plot_on_axis(ax, x_y_label_tuples, None, None)
                ax.legend()

        column_names = [count_bin_name(count_idx) for count_idx in range(NUM_ALT_COUNT_BINS)]
        row_names = [var_type.name for var_type in Variation]
        plotting.tidy_subplots(fig, axes, x_label="predicted logit", y_label="frequency", row_labels=row_names, column_labels=column_names)
        return fig, axes


class BatchIndexedAverages:
    def __init__(self, num_sources: int, device=gpu_if_available(), include_logits: bool = False):
        self.totals = BatchIndexedTotals(num_sources, device, include_logits)
        self.counts = BatchIndexedTotals(num_sources, device, include_logits)
        self.include_logits = include_logits
        self.num_sources = num_sources
        self.has_been_sent_to_cpu = False

    def put_on_cpu(self):
        """
        Do this at the end of an epoch so that the whole tensor is on CPU in one operation rather than computing various
        marginals etc on GPU and sending them each to CPU for plotting etc.
        :return:
        """
        self.totals.put_on_cpu()
        self.counts.put_on_cpu()
        self.has_been_sent_to_cpu = True
        return self

    def record(self, batch: Batch, logits: torch.Tensor, values: torch.Tensor, weights: torch.Tensor=None):
        assert not self.has_been_sent_to_cpu, "Can't record after already sending to CPU"
        weights_to_use = torch.ones_like(values) if weights is None else weights
        self.totals.record(batch, logits, (values*weights_to_use).detach())
        self.counts.record(batch, logits, weights_to_use.detach())

    def get_averages(self) -> torch.Tensor:
        return self.totals.get_totals() / (0.001 + self.counts.get_totals())

    def get_marginal(self, *properties: Tuple[BatchProperty, ...]) -> torch.Tensor:
        return self.totals.get_marginal(properties) / self.counts.get_marginal(properties)

    def report_marginals(self, message: str):
        assert self.has_been_sent_to_cpu, "Can't report marginals before sending to CPU"
        print(message)
        batch_property: BatchProperty
        for batch_property in BatchProperty:
            if batch_property == BatchProperty.LOGIT_BIN and not self.include_logits:
                continue
            values = self.get_marginal(batch_property).tolist()
            print(f"Marginalizing by {batch_property.name}")
            for n, ave in enumerate(values):
                print(f"{batch_property.get_name(n)}: {ave:.3f}")

    def write_to_summary_writer(self, epoch_type: Epoch, epoch: int, summary_writer: SummaryWriter, prefix: str):
        """
        write marginals for every batch property
        :return:
        """
        assert self.has_been_sent_to_cpu, "Can't write to sumamry writer before sending to CPU"
        batch_property: BatchProperty
        for batch_property in BatchProperty:
            if batch_property == BatchProperty.LOGIT_BIN and not self.include_logits:
                continue
            marginals = self.get_marginal(batch_property)
            for n, average in enumerate(marginals.tolist()):
                heading = f"{prefix}/{epoch_type.name}/{batch_property.name}/{batch_property.get_name(n)}"
                summary_writer.add_scalar(heading, average, epoch)


def make_true_and_false_masks_lg():
    # note that this is on the CPU because it's for evaluation and plotting
    all_logits_g = logits_from_bin_indices(torch.tensor(range(NUM_LOGIT_BINS)))

    # labeled as non-artifact, called as non-artifact
    true_positive_lg = torch.zeros(len(Label), NUM_LOGIT_BINS)
    true_positive_lg[Label.VARIANT] = all_logits_g < 0

    # labeled as artifact, called as artifact
    true_negative_lg = torch.zeros(len(Label), NUM_LOGIT_BINS)
    true_negative_lg[Label.ARTIFACT] = all_logits_g > 0

    # labeled as artifact, called as non-artifact
    false_positive_lg = torch.zeros(len(Label), NUM_LOGIT_BINS)
    false_positive_lg[Label.ARTIFACT] = all_logits_g < 0

    # labeled as non-artifact, called as artifact
    false_negative_lg = torch.zeros(len(Label), NUM_LOGIT_BINS)
    false_negative_lg[Label.VARIANT] = all_logits_g > 0

    true_lg = true_positive_lg + true_negative_lg
    false_lg = false_positive_lg + false_negative_lg
    return true_lg, false_lg


class AccuracyMetrics(BatchIndexedTotals):
    TRUE_LG, FALSE_LG = make_true_and_false_masks_lg()

    """
    Record should be called with values=tensor of 1 if correct, 0 if incorrect.  Accuracies are the averages of the correctness

    ROC curves can also be generated by calculating with labels and logit cumulative sums

    calibration can be done with accuracy vs logit
    """
    def __init__(self, num_sources: int, device=gpu_if_available()):
        super().__init__(num_sources, device, include_logits=True)
        self.num_sources = num_sources

    # TODO: quite a bit of duplication with the accuracy plot
    # TODO: make a parent function called accuracy plot by count or something where the variant type
    # TODO: yeah, really it's just one is accuracy vs count and one is accuracy vs logit
    def make_data_for_calibration_plot(self, var_type: Variation, alt_count_bin: int, source: int = None, ):
        """
        if combine_alt_counts is False
            return a list of tuples.  This outer list is over alt counts (not bins).  Each tuple consists of
            (list of logits (x axis), list of accuracies (y axis), the alt count)
            sum over source by default, select over source if given
        if combine_alt_counts is True
            return a Tuple (list of logits (x axis), list of accuracies (y axis))
        """
        assert self.has_been_sent_to_cpu, "Can't make plots before sending to CPU"
        sum_dims = ((BatchProperty.SOURCE,) if source is None else ()) + \
                   ((BatchProperty.ALT_COUNT_BIN,) if alt_count_bin is None else ()) + \
                   (BatchProperty.REF_COUNT_BIN, )
        selection = ({} if source is None else {BatchProperty.SOURCE: source}) | \
                    ({} if alt_count_bin is None else {BatchProperty.ALT_COUNT_BIN: alt_count_bin}) | \
                    {BatchProperty.VARIANT_TYPE: var_type}

        totals_lg = select_and_sum(self.totals_slvrag, select=selection, sum=sum_dims)
        true_g = torch.sum(AccuracyMetrics.TRUE_LG * totals_lg, dim=0)
        false_g = torch.sum(AccuracyMetrics.FALSE_LG * totals_lg, dim=0)
        total_g = true_g + false_g
        non_empty_bin_indices = torch.argwhere(total_g >= 1)
        logits = logits_from_bin_indices(non_empty_bin_indices)
        accuracies = true_g[non_empty_bin_indices] / total_g[non_empty_bin_indices]
        return logits.tolist(), accuracies.tolist(), ("calibration" if alt_count_bin is None else count_bin_name(alt_count_bin))

    def make_roc_data(self, var_type: Variation, source: int, alt_count_bin: int, sens_prec: bool):
        """
        :param var_type:
        :param sens_prec:   is it sensitivity and precision, or sensitivity and accuracy on artifacts
        :return: list of (threshold, nonartifact metric, artifact metric

        source = None means sum over all sources
        alt_count_bin = None means sum over all alt counts
        """
        assert self.has_been_sent_to_cpu, "Can't make plots before sending to CPU"
        sum_dims = ((BatchProperty.SOURCE,) if source is None else ()) + \
                   ((BatchProperty.ALT_COUNT_BIN,) if alt_count_bin is None else ()) + \
                    (BatchProperty.REF_COUNT_BIN,)
        selection = ({} if source is None else {BatchProperty.SOURCE: source}) | \
                    ({} if alt_count_bin is None else {BatchProperty.ALT_COUNT_BIN: alt_count_bin}) | \
                    {BatchProperty.VARIANT_TYPE: var_type}

        totals_lg = select_and_sum(self.totals_slvrag, select=selection, sum=sum_dims)
        # starting point is threshold below bottom bin, hence everything is considered artifact, hence 1) everything labeled
        # non-artifact is a false negative, 2) there are no true positives, 3) there are no false positives
        true_positive, false_positive = 0, 0
        true_negative, false_negative = torch.sum(totals_lg[Label.ARTIFACT]).item(), torch.sum(totals_lg[Label.VARIANT]).item()
        # last bin is clipped, so the top of the last bin isn't meaningful; likewise for the bottom of the first bin
        result = []
        #TODO: replace some of this logic with cumulative sums?
        for logit_bin in range(NUM_LOGIT_BINS - 1):
            # logit threshold below which everything is considered non-artifact and above which everything is considered artifact
            threshold = top_of_logit_bin(logit_bin)

            # the artifacts in this logit bin go from true negatives to false positives, while the non-artifacts go
            # from false negatives to true positives
            true_positive += totals_lg[Label.VARIANT, logit_bin].item()
            false_negative -= totals_lg[Label.VARIANT, logit_bin].item()
            false_positive += totals_lg[Label.ARTIFACT, logit_bin].item()
            true_negative -= totals_lg[Label.ARTIFACT, logit_bin].item()

            if (true_positive + false_negative) > 0 and (true_positive + false_positive) > 0 and (true_negative + false_positive) > 0:
                nonartifact_metric = true_positive / (true_positive + false_negative)
                artifact_metric = (true_positive / (true_positive + false_positive)) if sens_prec else (true_negative / (true_negative + false_positive))
                result.append((threshold, nonartifact_metric, artifact_metric))
        return result

    def plot_roc_curve(self, var_type: Variation, axis, given_threshold: float = None, sens_prec: bool = False, source: int = None):
        assert self.has_been_sent_to_cpu, "Can't make plots before sending to CPU"
        thresh_nonart_art_tuples = [self.make_roc_data(var_type, source=source, alt_count_bin=None, sens_prec=sens_prec)]
        plotting.plot_roc_on_axis(thresh_nonart_art_tuples, [None], axis, sens_prec, given_threshold)

    def plot_roc_curves_by_count(self, var_type: Variation, axis, given_threshold: float = None, sens_prec: bool = False, source: int = None):
        assert self.has_been_sent_to_cpu, "Can't make plots before sending to CPU"
        thresh_nonart_art_tuples = [self.make_roc_data(var_type, source, alt_count_bin, sens_prec) for alt_count_bin in range(
            NUM_ALT_COUNT_BINS)]
        curve_labels = [count_bin_name(bin_idx) for bin_idx in range(NUM_ALT_COUNT_BINS)]
        plotting.plot_roc_on_axis(thresh_nonart_art_tuples, curve_labels, axis, sens_prec, given_threshold)

    def plot_accuracy(self, label: Label, var_type: Variation, axis, source: int = None):
        """
        for given Label and Variation, plot color map of accuracy vs ref (x axis) and alt (y axis) counts
        :return:
        """
        assert self.has_been_sent_to_cpu, "Can't make plots before sending to CPU"
        sum_dims = ((BatchProperty.SOURCE,) if source is None else ())
        selection = ({} if source is None else {BatchProperty.SOURCE: source}) | {BatchProperty.VARIANT_TYPE: var_type}

        # don't select label yet because we have to multiply by the truth mask
        totals_lrag = select_and_sum(self.totals_slvrag, select=selection, sum=sum_dims)
        true_lrag = AccuracyMetrics.TRUE_LG.view(len(Label), 1, 1, NUM_LOGIT_BINS) * totals_lrag
        false_lrag = AccuracyMetrics.FALSE_LG.view(len(Label), 1, 1, NUM_LOGIT_BINS) * totals_lrag
        true_ra, false_ra = torch.sum(true_lrag[label], dim=-1), torch.sum(false_lrag[label], dim=-1)
        acc_ra = true_ra / (true_ra + false_ra + 0.001)
        return plotting.color_plot_2d_on_axis(axis, np.array(ALT_COUNT_BIN_BOUNDS), np.array(REF_COUNT_BIN_BOUNDS), acc_ra, None, None,
                                       vmin=0, vmax=1)

    def plot_calibration_by_count(self, var_type: Variation, axis, source: int = None):
        assert self.has_been_sent_to_cpu, "Can't make plots before sending to CPU"
        acc_vs_logit_x_y_lab_tuples = [self.make_data_for_calibration_plot(var_type, alt_count_bin, source) for alt_count_bin in range(
            NUM_ALT_COUNT_BINS)]
        plotting.simple_plot_on_axis(axis, acc_vs_logit_x_y_lab_tuples, None, None)

    def plot_calibration_all_counts(self, var_type: Variation, axis, source: int = None):
        assert self.has_been_sent_to_cpu, "Can't make plots before sending to CPU"
        acc_vs_logit_x_y_lab_tuples = [self.make_data_for_calibration_plot(var_type, None, source)]
        plotting.simple_plot_on_axis(axis, acc_vs_logit_x_y_lab_tuples, None, None)


