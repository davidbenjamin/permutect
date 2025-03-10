from __future__ import annotations

from typing import Tuple, List

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import Tensor, IntTensor
from torch.utils.tensorboard import SummaryWriter

from permutect.data.batch import Batch, BatchProperty, BatchIndexedTensor
from permutect.data.count_binning import NUM_LOGIT_BINS, top_of_logit_bin, logits_from_bin_indices, \
    ALT_COUNT_BIN_BOUNDS, REF_COUNT_BIN_BOUNDS, NUM_ALT_COUNT_BINS, alt_count_bin_name, NUM_REF_COUNT_BINS
from permutect.metrics import plotting
from permutect.misc_utils import gpu_if_available
from permutect.utils.array_utils import select_and_sum
from permutect.utils.enums import Variation, Epoch, Label


class LossMetrics:
    def __init__(self, num_sources: int, device=gpu_if_available()):
        self.totals_slvra = BatchIndexedTensor.make_zeros(num_sources=num_sources, device=device)
        self.counts_slvra = BatchIndexedTensor.make_zeros(num_sources=num_sources, device=device)
        self.num_sources = num_sources
        self.has_been_sent_to_cpu = False

    def put_on_cpu(self):
        """
        Do this at the end of an epoch so that the whole tensor is on CPU in one operation rather than computing various
        marginals etc on GPU and sending them each to CPU for plotting etc.
        :return:
        """
        self.totals_slvra = self.totals_slvra.cpu()
        self.counts_slvra = self.counts_slvra.cpu()
        self.has_been_sent_to_cpu = True
        return self

    def record(self, batch: Batch, values: Tensor, weights: Tensor=None):
        assert not self.has_been_sent_to_cpu, "Can't record after already sending to CPU"
        weights_to_use = torch.ones_like(values) if weights is None else weights
        self.totals_slvra.record(batch, (values * weights_to_use).detach())
        self.counts_slvra.record(batch, weights_to_use.detach())

    def get_averages(self) -> BatchIndexedTensor:
        return self.totals_slvra / (0.001 + self.counts_slvra)

    def get_marginal(self, *properties: Tuple[BatchProperty, ...]) -> Tensor:
        return self.totals_slvra.get_marginal(properties) / self.counts_slvra.get_marginal(properties)

    def report_marginals(self, message: str):
        assert self.has_been_sent_to_cpu, "Can't report marginals before sending to CPU"
        print(message)
        batch_property: BatchProperty
        for batch_property in (b for b in BatchProperty if b != BatchProperty.LOGIT_BIN):
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
        for batch_property in (b for b in BatchProperty if b != BatchProperty.LOGIT_BIN):
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
        # TODO: only if include logits
        sum_dims = ((BatchProperty.SOURCE,) if source is None else ())
        selection = ({} if source is None else {BatchProperty.SOURCE: source}) | \
                    {BatchProperty.VARIANT_TYPE: var_type, BatchProperty.LABEL: label}

        totals_ra = select_and_sum(self.totals_slvra, select=selection, sum=sum_dims)
        counts_ra = select_and_sum(self.counts_slvra, select=selection, sum=sum_dims)
        average_ra = totals_ra / (counts_ra + 0.001)
        return plotting.color_plot_2d_on_axis(axis, np.array(ALT_COUNT_BIN_BOUNDS), np.array(REF_COUNT_BIN_BOUNDS), average_ra, None, None,
                                       vmin=0, vmax=1)

    # TODO: maybe move this to BatchIndexedTotals
    def plot_counts(self, label: Label, var_type: Variation, axis, source: int = None):
        """
        for given Label and Variation, plot color map of (effective) data counts vs ref (x axis) and alt (y axis) counts
        :return:
        """
        assert self.has_been_sent_to_cpu, "Can't make plots before sending to CPU"
        max_for_this_source_and_variant_type = torch.max(self.counts_slvra[source, :, var_type])
        # TODO: only if include logits
        sum_dims = ((BatchProperty.SOURCE,) if source is None else ())
        selection = ({} if source is None else {BatchProperty.SOURCE: source}) | \
                    {BatchProperty.VARIANT_TYPE: var_type, BatchProperty.LABEL: label}

        counts_ra = select_and_sum(self.counts_slvra, select=selection, sum=sum_dims)
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
            source_suffix = "" if self.num_sources == 1 else (", all sources" if source is None else f", source {source}")
            summary_writer.add_figure(f"{prefix} ({epoch_type.name})" + source_suffix, fig, global_step=epoch)


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


class AccuracyMetrics(BatchIndexedTensor):
    TRUE_LG, FALSE_LG = make_true_and_false_masks_lg()

    """
    Used for calibration, accuracy, and ROC curves
    """

    @staticmethod
    def __new__(cls, data: Tensor):
        return torch.Tensor._make_subclass(cls, data)

    # I think this needs to have the same signature as __new__?
    def __init__(self, data: Tensor):
        assert data.dim() == 6, "needs 6 dimensions"

    @classmethod
    def create(cls, num_sources: int, device=gpu_if_available()):
        shape = (num_sources, len(Label), len(Variation), NUM_REF_COUNT_BINS, NUM_ALT_COUNT_BINS, NUM_LOGIT_BINS)
        return cls(torch.zeros(shape, device=device))

    def record_with_sources_and_logits(self, batch: Batch, values: Tensor, sources_override: IntTensor, logits: Tensor):
        assert self.has_logits(), "Tensor lacks a logit dimension"
        batch.batch_indices().increment_tensor_with_sources_and_logits(self, values=values, sources_override=sources_override, logits=logits)

    def split_over_sources(self) -> List[AccuracyMetrics]:
        # split into single-source BatchIndexedTotals
        result = []
        for source in range(self.num_sources()):
            element = AccuracyMetrics.create(num_sources=1, device=self.device)
            element[0].copy_(self[source])
            result.append(element)
        return result

    def make_roc_data(self, ref_count_bin: int, alt_count_bin: int, source: int, variant_type: Variation, sens_prec: bool):
        sum_dims = ((BatchProperty.SOURCE,) if source is None else ()) + \
                   ((BatchProperty.REF_COUNT_BIN,) if ref_count_bin is None else ()) + \
                   ((BatchProperty.ALT_COUNT_BIN,) if alt_count_bin is None else ())
        selection = ({} if source is None else {BatchProperty.SOURCE: source}) | \
                    ({} if ref_count_bin is None else {BatchProperty.REF_COUNT_BIN: ref_count_bin}) | \
                    ({} if alt_count_bin is None else {BatchProperty.ALT_COUNT_BIN: alt_count_bin}) | \
                    {BatchProperty.VARIANT_TYPE: variant_type}

        totals_lg = select_and_sum(self, select=selection, sum=sum_dims)
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
        thresh_nonart_art_tuples = [self.make_roc_data(ref_count_bin, alt_count_bin, source, var_type, sens_prec) for var_type in Variation]
        curve_labels = [var_type.name for var_type in Variation]
        thresholds = ([None]*len(Variation)) if given_thresholds is None else [given_thresholds[var_type] for var_type in Variation]
        plotting.plot_roc_on_axis(thresh_nonart_art_tuples, curve_labels, axis, sens_prec, thresholds)

    def plot_accuracy(self, label: Label, var_type: Variation, axis, source: int = None):
        """
        for given Label and Variation, plot color map of accuracy vs ref (x axis) and alt (y axis) counts
        :return:
        """
        sum_dims = ((BatchProperty.SOURCE,) if source is None else ())
        selection = ({} if source is None else {BatchProperty.SOURCE: source}) | {BatchProperty.VARIANT_TYPE: var_type}

        # don't select label yet because we have to multiply by the truth mask
        totals_lrag = select_and_sum(self, select=selection, sum=sum_dims)
        true_lrag = AccuracyMetrics.TRUE_LG.view(len(Label), 1, 1, NUM_LOGIT_BINS) * totals_lrag
        false_lrag = AccuracyMetrics.FALSE_LG.view(len(Label), 1, 1, NUM_LOGIT_BINS) * totals_lrag
        true_ra, false_ra = torch.sum(true_lrag[label], dim=-1), torch.sum(false_lrag[label], dim=-1)
        acc_ra = true_ra / (true_ra + false_ra + 0.001)
        return plotting.color_plot_2d_on_axis(axis, np.array(ALT_COUNT_BIN_BOUNDS), np.array(REF_COUNT_BIN_BOUNDS), acc_ra, None, None,
                                       vmin=0, vmax=1)

    def plot_calibration(self, axis, ref_count_bin: int, alt_count_bin: int, source: int):
        sum_dims = ((BatchProperty.SOURCE,) if source is None else ()) + \
                   ((BatchProperty.ALT_COUNT_BIN,) if alt_count_bin is None else ()) + \
                   ((BatchProperty.REF_COUNT_BIN,) if ref_count_bin is None else ())
        selection = ({} if source is None else {BatchProperty.SOURCE: source}) | \
                    ({} if alt_count_bin is None else {BatchProperty.ALT_COUNT_BIN: alt_count_bin}) | \
                    ({} if ref_count_bin is None else {BatchProperty.REF_COUNT_BIN: ref_count_bin})
        totals_lvg = select_and_sum(self, select=selection, sum=sum_dims)
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

    def make_logit_histograms(self):
        fig, axes = plt.subplots(len(Variation), NUM_ALT_COUNT_BINS, sharex='all', sharey='all', squeeze=False,
                                 figsize=(2.5 * NUM_ALT_COUNT_BINS, 2.5 * len(Variation)), dpi=200)
        x_axis_logits = logits_from_bin_indices(torch.tensor(range(NUM_LOGIT_BINS)))

        num_sources = self.shape[BatchProperty.SOURCE]
        multiple_sources = num_sources > 1
        for row, variation_type in enumerate(Variation):
            for count_bin in range(NUM_ALT_COUNT_BINS): # this is also the column index
                selection={BatchProperty.VARIANT_TYPE: variation_type, BatchProperty.ALT_COUNT_BIN: count_bin}
                totals_slg = select_and_sum(self, select=selection, sum=(BatchProperty.REF_COUNT_BIN,))

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


