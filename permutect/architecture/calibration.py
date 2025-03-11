from typing import List

import torch
from matplotlib import pyplot as plt
from torch import nn, Tensor, IntTensor
from torch.nn import Parameter

from permutect.architecture.monotonic import MonoDense
from permutect.data.count_binning import MAX_REF_COUNT, MAX_ALT_COUNT, NUM_REF_COUNT_BINS, NUM_ALT_COUNT_BINS, \
    NUM_LOGIT_BINS, logits_from_bin_indices, counts_from_alt_bin_indices, counts_from_ref_bin_indices
from permutect.metrics import plotting
from permutect.metrics.loss_metrics import AccuracyMetrics
from permutect.misc_utils import backpropagate
from permutect.utils.enums import Variation, Label


class Calibration(nn.Module):
    VAR_TYPE_EMBEDDING_DIM = 10

    def __init__(self, hidden_layer_sizes: List[int]):
        super(Calibration, self).__init__()

        # calibration takes [logit, ref count, alt count] as input and maps it to [calibrated logit]
        # it maps 0 to 0 and is monotonic in uncalibrated logit.  This is achieved by having
        # calibration(x) = f(x) - f(0) where f is monotonic in logit.

        # the input features are logit, ref count, alt count, var_type embedding
        # the positive logit function is increasing in logits, increasing in counts
        # the negative logit function is increasing in logits, decreasing in counts
        self.mono_fxn = MonoDense(3 + Calibration.VAR_TYPE_EMBEDDING_DIM, hidden_layer_sizes + [1], 1, 0)
        self.var_type_embeddings_ve = Parameter(torch.rand(len(Variation), Calibration.VAR_TYPE_EMBEDDING_DIM))

    def calibrated_logits(self, logits_b: Tensor, ref_counts_b: Tensor, alt_counts_b: Tensor, var_types_b: IntTensor):
        # indices: 'b' for batch, 3 for logit, ref, alt
        ref_b1 = ref_counts_b.view(-1, 1) / MAX_REF_COUNT
        alt_b1 = alt_counts_b.view(-1, 1) / MAX_ALT_COUNT
        var_type_embeddings_ve = self.var_type_embeddings_ve[var_types_b]

        inputs_be = torch.hstack((logits_b.view(-1, 1), ref_b1, alt_b1, var_type_embeddings_ve))
        zero_inputs_be = torch.hstack((torch.zeros_like(logits_b).view(-1, 1), ref_b1, alt_b1, var_type_embeddings_ve))
        output_b1 = self.mono_fxn.forward(inputs_be) - self.mono_fxn.forward(zero_inputs_be)
        return output_b1.view(-1)

    def perform_m_step(self, counts_slvrag: AccuracyMetrics):
        counts_lvrag = torch.sum(counts_slvrag, dim=0)
        counts_vrag = counts_lvrag[Label.ARTIFACT] + counts_lvrag[Label.VARIANT]    # only labeled data is relevant here

        # the empirical artifact probability that we want the calibration function to match
        artifact_prob_vrag = counts_lvrag[Label.ARTIFACT] / (counts_vrag + 0.0001)
        artifact_prob_n = artifact_prob_vrag.view(-1).detach()   # flattened.  I doubt detach is necessary but just in case. . .

        # to compute the calibration function at every vrag bin we have to convert the vrag indices into flattened indices
        # and make a big batch.  That is, the nth element of the batch is the nth flattened vrag bin.
        # we have n = g + G * (a + A * (r + R * v)), where eg g is the logit bin index and G is the number of logit bins
        # thus g = n % G, a = (n - g)//G mod A etc
        num_vrag_bins = len(Variation) * NUM_REF_COUNT_BINS * NUM_ALT_COUNT_BINS * NUM_LOGIT_BINS
        flattened_n = torch.arange(num_vrag_bins)
        idx = flattened_n
        logit_indices_n = torch.remainder(idx, NUM_LOGIT_BINS)
        idx = torch.div(idx - logit_indices_n, NUM_LOGIT_BINS, rounding_mode='floor')
        alt_count_indices_n = torch.remainder(idx, NUM_ALT_COUNT_BINS)
        idx = torch.div(idx - alt_count_indices_n, NUM_ALT_COUNT_BINS, rounding_mode='floor')
        ref_count_indices_n = torch.remainder(idx, NUM_REF_COUNT_BINS)
        idx = torch.div(idx - ref_count_indices_n, NUM_REF_COUNT_BINS, rounding_mode='floor')
        var_type_indices_n = torch.remainder(idx, len(Variation))
        logits_n = logits_from_bin_indices(logit_indices_n)
        alt_counts_n = counts_from_alt_bin_indices(alt_count_indices_n)
        ref_counts_n = counts_from_ref_bin_indices(ref_count_indices_n)
        var_types_n = var_type_indices_n

        optimizer = torch.optim.AdamW(self.parameters())
        bce = nn.BCEWithLogitsLoss(reduction='none')
        # we want to weight each vra bin equally, but give more weight to logit bins that have more data
        # the weight for vrag is the proportion of vra that has the given g
        weights_vrag = counts_vrag / torch.sum(counts_vrag, dim=-1, keepdim=True)

        # TODO: magic constant!!!
        for epoch in range(1000):
            calibrated_logits_n = self.calibrated_logits(logits_n, ref_counts_n, alt_counts_n, var_types_n)
            calibrated_logits_vrag = calibrated_logits_n.view(len(Variation), NUM_REF_COUNT_BINS, NUM_ALT_COUNT_BINS, NUM_LOGIT_BINS)
            losses_vrag = bce(calibrated_logits_n, artifact_prob_n)
            loss = torch.sum(losses_vrag * weights_vrag)
            backpropagate(optimizer, loss)

    def forward(self, logits_b, ref_counts_b: Tensor, alt_counts_b: Tensor, var_types_b: IntTensor):
        return self.calibrated_logits(logits_b, ref_counts_b, alt_counts_b, var_types_b)

    def plot_calibration_module(self, var_type: Variation, device, dtype):
        alt_counts = [1, 3, 5, 10, 15, 20]
        ref_counts = [1, 3, 5, 10, 15, 20]
        logits = torch.arange(start=-10, end=10, step=0.1, device=device, dtype=dtype)
        cal_fig,cal_axes = plt.subplots(len(alt_counts), len(ref_counts), sharex='all', sharey='all',
                                        squeeze=False, figsize=(10, 6), dpi=100)

        var_types_b = var_type * torch.ones(len(logits), device=device, dtype=torch.long)
        for row_idx, alt_count in enumerate(alt_counts):
            alt_counts_b = alt_count * torch.ones_like(logits, device=device, dtype=dtype)
            for col_idx, ref_count in enumerate(ref_counts):
                ref_counts_b = ref_count * torch.ones_like(logits, device=device, dtype=dtype)
                calibrated = self.calibrated_logits(logits, ref_counts_b, alt_counts_b, var_types_b)
                plotting.simple_plot_on_axis(cal_axes[row_idx, col_idx], [(logits.detach().cpu(), calibrated.detach().cpu(), "")], None, None)

        plotting.tidy_subplots(cal_fig, cal_axes, x_label="alt count", y_label="ref count",
                               row_labels=[str(n) for n in ref_counts], column_labels=[str(n) for n in alt_counts])

        return cal_fig, cal_axes