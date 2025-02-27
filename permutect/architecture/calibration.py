from typing import List

import torch
from matplotlib import pyplot as plt
from torch import nn, Tensor, IntTensor
from torch.nn import Parameter

from permutect.architecture.monotonic import MonoDense
from permutect.data.count_binning import MAX_REF_COUNT, MAX_ALT_COUNT
from permutect.metrics import plotting
from permutect.utils.enums import Variation


class Calibration(nn.Module):
    VAR_TYPE_EMBEDDING_DIM = 10
    def __init__(self, hidden_layer_sizes: List[int]):
        super(Calibration, self).__init__()

        # calibration takes [logit, ref count, alt count] as input and maps it to [calibrated logit]
        # it is split into two functions, one for logit > 0 and one for logit < 0
        # Both are monotonically increasing in the input logit because the uncalibrated logit means something!
        # The logit > 0 function is increasing in both the ref and alt count -- more reads imply more confidence --
        # and the logit < 0 is decreasing in the counts for the same reason -- more negative is more confident.

        #Finally, to ensure that zero maps to zero (we don't want calibration to shift predicitons, just modify confidence)
        # we implement the logit > 0 and logit < 0 functions as f(logit, counts) = g(logit, count) - g(logit=0, counts),
        # where g has the desired monotonicity.

        # the input features are logit, ref count, alt count, var_type embedding
        # the positive logit function is increasing in logits, increasing in counts
        # the negative logit function is increasing in logits, decreasing in counts
        self.positive_fxn = MonoDense(3 + Calibration.VAR_TYPE_EMBEDDING_DIM, hidden_layer_sizes + [1], 3, 0)
        self.negative_fxn = MonoDense(3 + Calibration.VAR_TYPE_EMBEDDING_DIM, hidden_layer_sizes + [1], 1, 2)

        self.var_type_embeddings_ve = Parameter(torch.rand(len(Variation), Calibration.VAR_TYPE_EMBEDDING_DIM))

    def calibrated_logits(self, logits_b: Tensor, ref_counts_b: Tensor, alt_counts_b: Tensor, var_types_b: IntTensor):
        # indices: 'b' for batch, 3 for logit, ref, alt
        ref_b1 = ref_counts_b.view(-1, 1) / MAX_REF_COUNT
        alt_b1 = alt_counts_b.view(-1, 1) / MAX_ALT_COUNT
        var_type_embeddings_ve = self.var_type_embeddings_ve[var_types_b]

        monotonic_inputs_be = torch.hstack((logits_b.view(-1, 1), ref_b1, alt_b1, var_type_embeddings_ve))
        zero_inputs_be = torch.hstack((torch.zeros_like(logits_b).view(-1, 1), ref_b1, alt_b1, var_type_embeddings_ve))
        positive_output_b1 = self.positive_fxn.forward(monotonic_inputs_be) - self.positive_fxn.forward(zero_inputs_be)
        negative_output_b1 = self.negative_fxn.forward(monotonic_inputs_be) - self.negative_fxn.forward(zero_inputs_be)
        return torch.where(logits_b > 0, positive_output_b1.view(-1), negative_output_b1.view(-1))

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