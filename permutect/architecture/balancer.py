import math

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import Tensor
from torch.distributions import Beta
from torch.nn import Module, Parameter
from torch.utils.tensorboard import SummaryWriter

from permutect.data.batch import Batch, make_batch_indexed_tensor
from permutect.data.reads_batch import ReadsBatch
from permutect.data.count_binning import ALT_COUNT_BIN_BOUNDS, REF_COUNT_BIN_BOUNDS
from permutect.metrics import plotting
from permutect.utils.array_utils import index_tensor
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
        self.counts_slvra = Parameter(make_batch_indexed_tensor(num_sources=num_sources, include_logits=False, value=0.0), requires_grad=False)

        # initialize weights to be flat
        self.weights_slvra = Parameter(make_batch_indexed_tensor(num_sources=num_sources, include_logits=False, value=1.0), requires_grad=False)

        # the overall weights for adversarial source prediction are the regular weights times the source weights
        self.source_weights_s = Parameter(torch.ones(num_sources), requires_grad=False)

        # Beta binomial shape parameters for ref and alt count balancing by downsampling
        # for data with ref count r and alt count a we have distributions Beta_ref = (alpha_ref[r,a], beta_ref[r,a]) and
        # Beta_alt = (alpha_alt[r,a], beta_alt[r,a]) from which we sample the downsampling ref and alt fractions (and then
        # in downsampling a binomial draw of reads to keep happens).
        self.alpha_ref_pre_exp_slvra = Parameter(make_batch_indexed_tensor(num_sources=num_sources, include_logits=False, value=0.0), requires_grad=False)
        self.beta_ref_pre_exp_slvra = Parameter(make_batch_indexed_tensor(num_sources=num_sources, include_logits=False, value=0.0), requires_grad=False)
        self.alpha_alt_pre_exp_slvra = Parameter(make_batch_indexed_tensor(num_sources=num_sources, include_logits=False, value=0.0), requires_grad=False)
        self.beta_alt_pre_exp_slvra = Parameter(make_batch_indexed_tensor(num_sources=num_sources, include_logits=False, value=0.0), requires_grad=False)
        self.to(device=device)

    # TODO: compute multiple sets of samplings
    def compute_downsampling_fractions(self, batch: Batch):
        batch_indices = batch.batch_indices()
        alpha_ref_b = batch_indices.index_into_tensor(torch.exp(self.alpha_ref_pre_exp_slvra))
        beta_ref_b = batch_indices.index_into_tensor(torch.exp(self.beta_ref_pre_exp_slvra))
        alpha_alt_b = batch_indices.index_into_tensor(torch.exp(self.alpha_alt_pre_exp_slvra))
        beta_alt_b = batch_indices.index_into_tensor(torch.exp(self.beta_alt_pre_exp_slvra))

        ref_fractions_b = Beta(alpha_ref_b, beta_ref_b).sample()
        alt_fractions_b = Beta(alpha_alt_b, beta_alt_b).sample()

        return ref_fractions_b, alt_fractions_b


    def fit_downsampling_parameters(self, undownsampled_counts_slvra: Tensor):
        pass
        # I could do this fully vectorized without a for loop, but in this case I think it's actually more readable
        #beta_binom_densities_slvrar
        #downsampled_counts_slvra =


    def process_batch_and_compute_weights(self, batch: ReadsBatch):
        # this updates the counts that are used to compute weights, recomputes the weights, and returns the weights
        # increment counts by 1
        batch.batch_indices().increment_tensor(self.counts_slvra, values=torch.ones(batch.size(), device=self.device))
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
        batch_weights = batch.batch_indices().index_into_tensor(self.weights_slvra)
        source_weights = self.source_weights_s[batch.batch_indices().sources]
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
