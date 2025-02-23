import math

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.nn import Module, Parameter
from torch.utils.tensorboard import SummaryWriter

from permutect.data.reads_batch import ReadsBatch
from permutect.data.count_binning import alt_count_bin_indices, NUM_ALT_COUNT_BINS, NUM_REF_COUNT_BINS, \
    ref_count_bin_indices, ALT_COUNT_BIN_BOUNDS, REF_COUNT_BIN_BOUNDS
from permutect.metrics import plotting
from permutect.utils.array_utils import index_5d_array, add_to_5d_array
from permutect.utils.enums import Label, Variation, Epoch


class Balancer(Module):
    ATTENUATION_PER_DATUM = 0.99999
    DATA_BEFORE_RECOMPUTE = 10000

    def __init__(self, num_sources: int, device):
        super(Balancer, self).__init__()
        self.device = device
        self.num_sources = num_sources
        self.count_since_last_recomputation = 0

        # not weighted, just the actual counts of data seen
        self.counts_slvra = Parameter(torch.zeros(num_sources, len(Label), len(Variation), NUM_REF_COUNT_BINS, NUM_ALT_COUNT_BINS, device=device), requires_grad=False)

        # initialize weights to be flat
        self.weights_slvra = Parameter(torch.ones(num_sources, len(Label), len(Variation), NUM_REF_COUNT_BINS, NUM_ALT_COUNT_BINS, device=device), requires_grad=False)

        # the overall weights for adversarial source predicition are the regular weights times the source weights
        self.source_weights_s = Parameter(torch.ones(num_sources, device=device), requires_grad=False)

    def process_batch_and_compute_weights(self, batch: ReadsBatch):
        # this updates the counts that are used to compute weights, recomputes the weights, and returns the weights
        # increment counts by 1
        sources, labels, var_types = batch.get_sources(), batch.get_labels(), batch.get_variant_types()
        ref_count_bins, alt_count_bins = ref_count_bin_indices(batch.get_ref_counts()), alt_count_bin_indices(batch.get_alt_counts())
        add_to_5d_array(self.counts_slvra, sources, labels, var_types, ref_count_bins, alt_count_bins, values=torch.ones(batch.size(), device=self.device))
        self.count_since_last_recomputation += batch.size()

        if self.count_since_last_recomputation > Balancer.DATA_BEFORE_RECOMPUTE:
            art_to_nonart_ratios_svra = (self.counts_slvra[:, Label.ARTIFACT] + 0.01) / (self.counts_slvra[:, Label.VARIANT] + 0.01)
            # TODO: perhaps don't recompute weights at every batch, as we do here
            new_weights_slvra = torch.zeros_like(self.weights_slvra)
            new_weights_slvra[:, Label.ARTIFACT] = torch.clip((1 + 1/art_to_nonart_ratios_svra)/2, min=0.01, max=100)
            new_weights_slvra[:, Label.VARIANT] = torch.clip((1 + art_to_nonart_ratios_svra) / 2, min=0.01, max=100)

            counts_slv = torch.sum(self.counts_slvra, dim=(-2,-1))
            unlabeled_weight_sv = torch.clip((counts_slv[:, Label.ARTIFACT] + counts_slv[:, Label.ARTIFACT])/counts_slv[:, Label.UNLABELED], 0, 1)
            new_weights_slvra[:, Label.UNLABELED] = unlabeled_weight_sv.view(self.num_sources, len(Variation), 1, 1)

            attenuation = math.pow(Balancer.ATTENUATION_PER_DATUM, self.count_since_last_recomputation)
            self.weights_slvra.copy_(attenuation * self.weights_slvra + (1-attenuation)*new_weights_slvra)

            counts_s = torch.sum(counts_slv, dim=(-2, -1))
            total_s = torch.sum(counts_s, dim=0, keepdim=True)
            new_source_weights_s = (total_s / counts_s) / self.num_sources
            self.source_weights_s.copy_(attenuation * self.source_weights_s + (1-attenuation)*new_source_weights_s)
            self.count_since_last_recomputation = 0
            # TODO: also attenuate counts -- multiply by an attenuation factor or something?
        batch_weights = index_5d_array(self.weights_slvra, sources, labels, var_types, ref_count_bins, alt_count_bins)
        source_weights = self.source_weights_s[sources]
        return batch_weights, source_weights

    # TODO: lots of code duplication with the plotting in loss_metrics.py
    def plot_weights(self, label: Label, var_type: Variation, axis, source: int):
        """
        for given Label and Variation, plot color map of (effective) data counts vs ref (x axis) and alt (y axis) counts
        :return:
        """
        weights_ra = self.weights_slvra[source, label, var_type].cpu()
        log_weights_ra = torch.clip(torch.log(weights_ra), -4, 4)
        return plotting.color_plot_2d_on_axis(axis, np.array(ALT_COUNT_BIN_BOUNDS), np.array(REF_COUNT_BIN_BOUNDS), log_weights_ra, None, None,
                                       vmin=-4, vmax=4)

    def plot_counts(self, label: Label, var_type: Variation, axis, source: int):
        """
        for given Label and Variation, plot color map of (effective) data counts vs ref (x axis) and alt (y axis) counts
        :return:
        """
        counts_lra = self.counts_slvra[source, :, var_type].cpu()
        max_count = torch.max(counts_lra)
        normalized_counts_ra = (counts_lra / max_count)[label] + 0.0001
        log_normalized_count_ra = torch.clip(torch.log(normalized_counts_ra), -10, 0)
        return plotting.color_plot_2d_on_axis(axis, np.array(ALT_COUNT_BIN_BOUNDS), np.array(REF_COUNT_BIN_BOUNDS), log_normalized_count_ra, None, None,
                                       vmin=-10, vmax=0)

    def make_plots(self, summary_writer: SummaryWriter, prefix: str, epoch_type: Epoch, epoch: int = None, type_of_plot: str = "weights"):
        for source in range(self.num_sources):
            fig, axes = plt.subplots(len(Label), len(Variation), sharex='all', sharey='all', squeeze=False, figsize=(2.5 * len(Variation), 2.5 * len(Label)))
            row_names = [label.name for label in Label]
            variation_types = [var_type.name for var_type in Variation]
            common_colormesh = None
            for label in Label:
                for var_type in Variation:
                    if type_of_plot == "weights":
                        common_colormesh = self.plot_weights(label, var_type, axes[label, var_type], source)
                    elif type_of_plot == "counts":
                        common_colormesh = self.plot_counts(label, var_type, axes[label, var_type], source)
                    else:
                        raise Exception("BAD")
            fig.colorbar(common_colormesh)
            plotting.tidy_subplots(fig, axes, x_label="alt count", y_label="ref count", row_labels=row_names, column_labels=variation_types)
            name_suffix = epoch_type.name + "" if self.num_sources == 1 else (", all sources" if source is None else f", source {source}")
            summary_writer.add_figure(prefix + name_suffix, fig, global_step=epoch)
