from __future__ import annotations

from enum import IntEnum
from typing import Tuple, List

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter

from permutect.data.batch import Batch
from permutect.data.datum import Datum
from permutect.data.count_binning import NUM_REF_COUNT_BINS, NUM_ALT_COUNT_BINS, \
    NUM_LOGIT_BINS, logit_bin_indices, top_of_logit_bin, logits_from_bin_indices, logit_bin_name, ref_count_bin_indices, \
    ref_count_bin_index, ref_count_bin_name, ALT_COUNT_BIN_BOUNDS, REF_COUNT_BIN_BOUNDS, alt_count_bin_index, \
    alt_count_bin_indices, alt_count_bin_name
from permutect.metrics import plotting
from permutect.misc_utils import gpu_if_available
from permutect.utils.array_utils import add_to_6d_array, select_and_sum
from permutect.utils.enums import Variation, Epoch, Label


class BatchProperty(IntEnum):
    SOURCE = (0, None)
    LABEL = (1, [label.name for label in Label])
    VARIANT_TYPE = (2, [var_type.name for var_type in Variation])
    REF_COUNT_BIN = (3, [ref_count_bin_name(idx) for idx in range(NUM_REF_COUNT_BINS)])
    ALT_COUNT_BIN = (4, [alt_count_bin_name(idx) for idx in range(NUM_ALT_COUNT_BINS)])
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

    def split_over_sources(self) -> List[BatchIndexedTotals]:
        # split into single-source BatchIndexedTotals
        result = []
        for source in range(self.num_sources):
            element = BatchIndexedTotals(num_sources=1, device=self.device, include_logits=self.include_logits)
            element.totals_slvrag[0].copy_(self.totals_slvrag[source])
            result.append(element)
        return result

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
        ref_idx, alt_idx = ref_count_bin_index(datum.get_ref_count()), alt_count_bin_index(datum.get_alt_count())
        self.totals_slvrag[source, datum.get_label(), datum.get_variant_type(), ref_idx, alt_idx, 0] += value

    def record(self, batch: Batch, logits: torch.Tensor, values: torch.Tensor):
        # values is a 1D tensor
        assert batch.size() == len(values)
        assert not self.has_been_sent_to_cpu, "Can't record after already sending to CPU"
        sources = batch.get_sources()
        logit_indices = logit_bin_indices(logits) if self.include_logits else torch.zeros_like(sources)

        add_to_6d_array(self.totals_slvrag, sources, batch.get_labels(), batch.get_variant_types(),
                        ref_count_bin_indices(batch.get_ref_counts()), alt_count_bin_indices(batch.get_alt_counts()), logit_indices, values)

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
        for row, variation_type in enumerate(Variation):
            for count_bin in range(NUM_ALT_COUNT_BINS): # this is also the column index
                selection={BatchProperty.VARIANT_TYPE: variation_type, BatchProperty.ALT_COUNT_BIN: count_bin}
                totals_slg = select_and_sum(self.totals_slvrag, select=selection, sum=(BatchProperty.REF_COUNT_BIN,))

                # The normalizing factor for each source, label is the sum over all logits for that source and label
                # This renders a histogram into a sort of probability density plot for each source, label
                normalization_slg = torch.sum(totals_slg, dim=-1, keepdim=True)
                normalized_totals_slg = (totals_slg + 0.000001) / normalization_slg

                # overlapping line plots for all source / label combinations
                # source 0 is filled; others are not
                ax = axes[row, count_bin]
                x_y_label_tuples = []
                for source in range(num_sources):
                    for label in Label:
                        if normalization_slg[source, label, 0].item() >= 1:
                            line_label = f"{label.name} ({source})" if multiple_sources else label.name
                            x_y_label_tuples.append((x_axis_logits.cpu().numpy(), normalized_totals_slg[source, label].cpu().numpy(), line_label))
                plotting.simple_plot_on_axis(ax, x_y_label_tuples, None, None)
                ax.legend()

        column_names = [alt_count_bin_name(count_idx) for count_idx in range(NUM_ALT_COUNT_BINS)]
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

    def plot_losses(self, label: Label, var_type: Variation, axis, source: int = None):
        """
        for given Label and Variation, plot color map of accuracy vs ref (x axis) and alt (y axis) counts
        :return:
        """
        assert self.has_been_sent_to_cpu, "Can't make plots before sending to CPU"
        sum_dims = ((BatchProperty.SOURCE,) if source is None else ()) + (BatchProperty.LOGIT_BIN,)
        selection = ({} if source is None else {BatchProperty.SOURCE: source}) | \
                    {BatchProperty.VARIANT_TYPE: var_type, BatchProperty.LABEL: label}

        totals_ra = select_and_sum(self.totals.totals_slvrag, select=selection, sum=sum_dims)
        counts_ra = select_and_sum(self.counts.totals_slvrag, select=selection, sum=sum_dims)
        average_ra = totals_ra / (counts_ra + 0.001)
        transformed_ra = torch.pow(average_ra, 1/3)     # cube root is a good scaling
        return plotting.color_plot_2d_on_axis(axis, np.array(ALT_COUNT_BIN_BOUNDS), np.array(REF_COUNT_BIN_BOUNDS), transformed_ra, None, None,
                                       vmin=0, vmax=1)

    # TODO: maybe move this to BatchIndexedTotals
    def plot_counts(self, label: Label, var_type: Variation, axis, source: int = None):
        """
        for given Label and Variation, plot color map of (effective) data counts vs ref (x axis) and alt (y axis) counts
        :return:
        """
        assert self.has_been_sent_to_cpu, "Can't make plots before sending to CPU"
        max_for_this_source_and_variant_type = torch.max(self.counts.totals_slvrag[source, :, var_type])
        sum_dims = ((BatchProperty.SOURCE,) if source is None else ()) + (BatchProperty.LOGIT_BIN,)
        selection = ({} if source is None else {BatchProperty.SOURCE: source}) | \
                    {BatchProperty.VARIANT_TYPE: var_type, BatchProperty.LABEL: label}

        counts_ra = select_and_sum(self.counts.totals_slvrag, select=selection, sum=sum_dims)
        normalized_counts_ra = counts_ra / max_for_this_source_and_variant_type
        return plotting.color_plot_2d_on_axis(axis, np.array(ALT_COUNT_BIN_BOUNDS), np.array(REF_COUNT_BIN_BOUNDS), normalized_counts_ra, None, None,
                                       vmin=0, vmax=1)

    def make_plots(self, summary_writer: SummaryWriter, prefix: str, epoch_type: Epoch, epoch: int = None, type_of_plot: str = "loss"):
        assert self.has_been_sent_to_cpu, "Can't make plots before sending to CPU"

        for source in range(self.num_sources):
            fig, axes = plt.subplots(len(Label), len(Variation), sharex='all', sharey='all', squeeze=False, figsize=(2.5 * len(Variation), 2.5 * len(Label)))
            row_names = [label.name for label in Label]
            variation_types = [var_type.name for var_type in Variation]
            common_colormesh = None
            for label in Label:
                for var_type in Variation:
                    if type_of_plot == "loss":
                        common_colormesh = self.plot_losses(label, var_type, axes[label, var_type], source)
                    elif type_of_plot == "counts":
                        common_colormesh = self.plot_counts(label, var_type, axes[label, var_type], source)
            fig.colorbar(common_colormesh)
            plotting.tidy_subplots(fig, axes, x_label="alt count", y_label="ref count", row_labels=row_names, column_labels=variation_types)
            name_suffix = epoch_type.name + "" if self.num_sources == 1 else (", all sources" if source is None else f", source {source}")
            summary_writer.add_figure(prefix + name_suffix, fig, global_step=epoch)


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

    def make_roc_data(self, ref_count_bin: int, alt_count_bin: int, source: int, variant_type: Variation, sens_prec: bool):
        assert self.has_been_sent_to_cpu, "Can't make plots before sending to CPU"
        sum_dims = ((BatchProperty.SOURCE,) if source is None else ()) + \
                   ((BatchProperty.REF_COUNT_BIN,) if ref_count_bin is None else ()) + \
                    ((BatchProperty.ALT_COUNT_BIN,) if alt_count_bin is None else ())
        selection = ({} if source is None else {BatchProperty.SOURCE: source}) | \
                    ({} if ref_count_bin is None else {BatchProperty.REF_COUNT_BIN: ref_count_bin}) | \
                    ({} if alt_count_bin is None else {BatchProperty.ALT_COUNT_BIN: alt_count_bin}) | \
                    {BatchProperty.VARIANT_TYPE: variant_type}

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

    def plot_roc(self, axis, ref_count_bin: int, alt_count_bin: int, source: int, given_thresholds, sens_prec: bool = False):
        assert self.has_been_sent_to_cpu, "Can't make plots before sending to CPU"
        thresh_nonart_art_tuples = [self.make_roc_data(ref_count_bin, alt_count_bin, source, var_type, sens_prec) for var_type in Variation]
        curve_labels = [var_type.name for var_type in Variation]
        thresholds = ([None]*len(Variation)) if given_thresholds is None else [given_thresholds[var_type] for var_type in Variation]
        plotting.plot_roc_on_axis(thresh_nonart_art_tuples, curve_labels, axis, sens_prec, thresholds)

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

    def plot_calibration(self, axis, ref_count_bin: int, alt_count_bin: int, source: int):
        assert self.has_been_sent_to_cpu, "Can't make plots before sending to CPU"
        assert self.has_been_sent_to_cpu, "Can't make plots before sending to CPU"
        sum_dims = ((BatchProperty.SOURCE,) if source is None else ()) + \
                   ((BatchProperty.ALT_COUNT_BIN,) if alt_count_bin is None else ()) + \
                   ((BatchProperty.REF_COUNT_BIN,) if ref_count_bin is None else ())
        selection = ({} if source is None else {BatchProperty.SOURCE: source}) | \
                    ({} if alt_count_bin is None else {BatchProperty.ALT_COUNT_BIN: alt_count_bin}) | \
                    ({} if ref_count_bin is None else {BatchProperty.REF_COUNT_BIN: ref_count_bin})
        totals_lvg = select_and_sum(self.totals_slvrag, select=selection, sum=sum_dims)
        true_lvg = AccuracyMetrics.TRUE_LG.view(len(Label), 1, NUM_LOGIT_BINS) * totals_lvg
        false_lvg = AccuracyMetrics.FALSE_LG.view(len(Label), 1, NUM_LOGIT_BINS) * totals_lvg
        true_vg, false_vg = torch.sum(true_lvg, dim=0), torch.sum(false_lvg, dim=0)
        x_y_lab_tuples = []
        for var_type in Variation:
            true_g, false_g = true_vg[var_type], false_vg[var_type]
            total_g = true_g + false_g
            non_empty_bin_indices = torch.argwhere(total_g >= 0.0001)
            logits = logits_from_bin_indices(non_empty_bin_indices)
            accuracies = true_g[non_empty_bin_indices] / total_g[non_empty_bin_indices]
            x_y_lab_tuples.append((logits, accuracies, var_type.name))
        plotting.simple_plot_on_axis(axis, x_y_lab_tuples, None, None)


