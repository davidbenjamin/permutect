import enum
import itertools
import math

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import Tensor
from torch.nn import Module
from torch.nn.parameter import Buffer
from torch.utils.tensorboard import SummaryWriter

from permutect.data.batch import Batch
from permutect.data.batch import BatchIndexedTensor
from permutect.data.count_binning import ALT_COUNT_BIN_BOUNDS
from permutect.data.count_binning import REF_COUNT_BIN_BOUNDS
from permutect.metrics import plotting
from permutect.utils.enums import Label
from permutect.utils.enums import Variation


class PlotType(enum.Enum):
    COUNTS = "counts"
    WEIGHTS = "weights"


class Balancer(Module):
    ATTENUATION_PER_DATUM = 0.99999
    DATA_BEFORE_RECOMPUTE = 10000

    def __init__(self, num_sources: int, device):
        super(Balancer, self).__init__()
        self.device = device
        self.num_sources = num_sources
        self.count_since_last_recomputation = 0

        # not weighted, just the actual counts of data seen
        self.counts_slvra = Buffer(BatchIndexedTensor.zeros(num_sources=num_sources))

        # same, but here unlabeled data are probabilistically assigned to artifact or non-artifact based on
        # the model output.  This lets us balance training when, for example, the preponderance of unlabeled
        # data are non-artifacts.  (Very often unlabeled data are sequencing errors, which are not artifacts.
        # also, in test-time adaptation of somatic calls most data are germline variants, which are not artifacts).
        self.pseudo_counts_slvra = Buffer(BatchIndexedTensor.zeros(num_sources=num_sources))

        # weights for labeled data, initialized flat
        self.weights_slvra = Buffer(BatchIndexedTensor.ones(num_sources=num_sources))

        # weights for unlabeled data, where index l is a guess for the correct label
        self.unlabeled_weights_slvra = Buffer(BatchIndexedTensor.ones(num_sources=num_sources))

        # the overall weights for adversarial source prediction are the regular weights times the source weights
        self.source_weights_s = Buffer(torch.ones(num_sources))

        self.to(device=device)

    def process_batch_and_compute_weights(self, batch: Batch, artifact_probs_b: Tensor):
        # this updates the counts that are used to compute weights, recomputes the weights, and returns the weights
        # increment counts by 1
        idx = batch.batch_indices()
        idx.increment_tensor(self.counts_slvra, values=torch.ones(batch.size(), device=self.device))

        # increment the pseudo-label counts for unlabeled data according to the amount of probability
        # assigned to artifact/nonartifact
        art_probs_b = artifact_probs_b.to(device=self.device)
        unlabeled_mask = 1 - batch.get_is_labeled_mask()
        artifact_labels = torch.tensor([Label.ARTIFACT], device=self.device).expand(batch.size())
        nonartifact_labels = torch.tensor([Label.VARIANT], device=self.device).expand(batch.size())
        idx.increment_tensor(self.pseudo_counts_slvra, labels=artifact_labels, values=unlabeled_mask * art_probs_b)
        idx.increment_tensor(
            self.pseudo_counts_slvra, labels=nonartifact_labels, values=unlabeled_mask * (1 - art_probs_b)
        )

        self.count_since_last_recomputation += batch.size()

        if self.count_since_last_recomputation > Balancer.DATA_BEFORE_RECOMPUTE:
            attenuation = math.pow(Balancer.ATTENUATION_PER_DATUM, self.count_since_last_recomputation)

            # update weights for both labeled and unlabeled data
            for is_labeled in (True, False):
                counts_slvra = self.counts_slvra if is_labeled else self.pseudo_counts_slvra
                art_counts_svra = counts_slvra[:, Label.ARTIFACT]
                nonart_counts_svra = counts_slvra[:, Label.VARIANT]
                ratio_svra = (art_counts_svra + 0.01) / (nonart_counts_svra + 0.01)

                new_weights_slvra = torch.zeros_like(counts_slvra)
                new_weights_slvra[:, Label.ARTIFACT] = torch.clip((1 + 1 / ratio_svra) / 2, min=0.01, max=100)
                new_weights_slvra[:, Label.VARIANT] = torch.clip((1 + ratio_svra) / 2, min=0.01, max=100)

                old_weights_slvra = self.weights_slvra if is_labeled else self.unlabeled_weights_slvra
                lin_comb_slvra = attenuation * old_weights_slvra + (1 - attenuation) * new_weights_slvra
                old_weights_slvra.copy_(lin_comb_slvra)

            counts_slv = torch.sum(self.counts_slvra, dim=(-2, -1))

            # TODO: here is old code for making total unlabeled weight at most equal to total labeled weight
            # TODO: can it be thrown out?  Wha tis the right thing to do?  Maybe nothing?
            # TODO: maybe it's the responsibility of the dataset?
            # total_labeled_sv = counts_slv[:, Label.ARTIFACT] + counts_slv[:, Label.VARIANT]
            # unlabeled_weight_sv = torch.clip(total_labeled_sv / counts_slv[:, Label.UNLABELED], 0, 1)
            # new_weights_slvra[:, Label.UNLABELED] = unlabeled_weight_sv.view(self.num_sources, len(Variation), 1, 1)

            counts_s = torch.sum(counts_slv, dim=(-2, -1))
            total_s = torch.sum(counts_s, dim=0, keepdim=True)
            new_source_weights_s = (total_s / counts_s) / self.num_sources
            self.source_weights_s.copy_(attenuation * self.source_weights_s + (1 - attenuation) * new_source_weights_s)
            self.count_since_last_recomputation = 0
            # TODO: also attenuate counts -- multiply by an attenuation factor or something?

        labeled_weights_b = idx.index_into_tensor(self.weights_slvra)
        pseudo_art_weights_b = idx.index_into_tensor(self.unlabeled_weights_slvra, labels=artifact_labels)
        pseudo_nonart_weights_b = idx.index_into_tensor(self.unlabeled_weights_slvra, labels=nonartifact_labels)
        unlabeled_weights_b = art_probs_b * pseudo_art_weights_b + (1 - art_probs_b) * pseudo_nonart_weights_b
        weights_b = unlabeled_mask * unlabeled_weights_b + (1 - unlabeled_mask) * labeled_weights_b

        source_weights = self.source_weights_s[idx.sources]
        return weights_b, source_weights

    # TODO: lots of code duplication with the plotting in loss_metrics.py
    def make_plot(self, label: Label, var_type: Variation, axis, source: int, plot_type: PlotType):
        if plot_type == PlotType.WEIGHTS:
            vmin, vmax = -4, 4
            weights_ra = self.weights_slvra[source, label, var_type].cpu()
            plot_data_ra = torch.clip(torch.log(weights_ra), min=vmin, max=vmax)
        elif plot_type == PlotType.COUNTS:
            vmin, vmax = -10, 0
            counts_lra = self.counts_slvra[source, :, var_type].cpu()
            max_count = torch.max(counts_lra)
            normalized_counts_ra = (counts_lra / max_count)[label] + 0.0001
            plot_data_ra = torch.clip(torch.log(normalized_counts_ra), -10, 0)
        else:
            raise ValueError(f"Unknown type_of_plot: {plot_type.name}.")
        xbounds, ybounds = np.array(ALT_COUNT_BIN_BOUNDS), np.array(REF_COUNT_BIN_BOUNDS)
        kwargs = {"x_label": None, "y_label": None, "vmin": vmin, "vmax": vmax}

        return plotting.color_plot_2d_on_axis(axis, xbounds, ybounds, plot_data_ra, **kwargs)

    def make_plots(self, summary_writer: SummaryWriter, prefix: str, epoch: int, plot_type: PlotType):
        for source in range(self.num_sources):
            figsize = (2.5 * len(Variation), 2.5 * len(Label))
            subplots_kwargs = {"sharex": "all", "sharey": "all", "squeeze": False, "figsize": figsize}
            fig, axes = plt.subplots(len(Label), len(Variation), **subplots_kwargs)

            row_names = [label.name for label in Label]
            col_names = [var_type.name for var_type in Variation]
            common_colormesh = None
            for label, var_type in itertools.product(Label, Variation):
                common_colormesh = self.make_plot(label, var_type, axes[label, var_type], source, plot_type)

            fig.colorbar(common_colormesh)
            tidy_kwargs = {"x_label": "N_alt", "y_label": "N_ref", "row_labels": row_names, "col_labels": col_names}
            plotting.tidy_subplots(fig, axes, **tidy_kwargs)
            multi_source_suffix = ", all sources" if source is None else f", source {source}"
            source_suffix = "" if self.num_sources == 1 else multi_source_suffix
            summary_writer.add_figure(prefix + source_suffix, fig, global_step=epoch)
