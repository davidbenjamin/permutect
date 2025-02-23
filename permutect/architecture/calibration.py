from typing import List

import torch
from matplotlib import pyplot as plt
from torch import nn, Tensor

from permutect.architecture.monotonic import MonoDense
from permutect.data.count_binning import MAX_REF_COUNT, MAX_ALT_COUNT
from permutect.metrics import plotting


class Calibration(nn.Module):

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

        # the three input features are logit, ref count, alt count
        # the positive logit function is increasing in logits, increasing in counts
        # the negative logit function is increasing in logits, decreasing in counts
        self.positive_fxn = MonoDense(3, hidden_layer_sizes + [1], 3, 0)
        self.negative_fxn = MonoDense(3, hidden_layer_sizes + [1], 1, 2)


        # Final layer of calibration is a count-dependent linear shift.  This is particularly useful when calibrating only on
        # a subset of data sources
        # self.final_adjustments = nn.Parameter(torch.zeros(self.max_alt_count_for_adjustment + 1), requires_grad=True)

    def calibrated_logits(self, logits_b: Tensor, ref_counts_b: Tensor, alt_counts_b: Tensor):
        # indices: 'b' for batch, 3 for logit, ref, alt
        ref_b1 = ref_counts_b.view(-1, 1) / MAX_REF_COUNT
        alt_b1 = alt_counts_b.view(-1, 1) / MAX_ALT_COUNT
        monotonic_inputs_b3 = torch.hstack((logits_b.view(-1, 1), ref_b1, alt_b1))
        zero_inputs_b3 = torch.hstack((torch.zeros_like(logits_b).view(-1, 1), ref_b1, alt_b1))
        positive_output_b1 = self.positive_fxn.forward(monotonic_inputs_b3) - self.positive_fxn.forward(zero_inputs_b3)
        negative_output_b1 = self.negative_fxn.forward(monotonic_inputs_b3) - self.negative_fxn.forward(zero_inputs_b3)
        return torch.where(logits_b > 0, positive_output_b1.view(-1), negative_output_b1.view(-1))

    def forward(self, logits, ref_counts: Tensor, alt_counts: Tensor):
        return self.calibrated_logits(logits, ref_counts, alt_counts)

    def plot_calibration_module(self, device, dtype):
        alt_counts = [1, 3, 5, 10, 15, 20]
        ref_counts = [1, 3, 5, 10, 15, 20]
        logits = torch.arange(-10, 10, 0.1, device=device, dtype=dtype)
        cal_fig,cal_axes = plt.subplots(len(alt_counts), len(ref_counts), sharex='all', sharey='all',
                                        squeeze=False, figsize=(10, 6), dpi=100)

        for row_idx, alt_count in enumerate(alt_counts):
            for col_idx, ref_count in enumerate(ref_counts):
                calibrated = self.forward(logits, ref_count * torch.ones_like(logits, device=device, dtype=dtype), alt_count * torch.ones_like(logits, device=device, dtype=dtype))
                plotting.simple_plot_on_axis(cal_axes[row_idx, col_idx], [(logits.detach().cpu(), calibrated.detach().cpu(), "")], None, None)

        plotting.tidy_subplots(cal_fig, cal_axes, x_label="alt count", y_label="ref count",
                               row_labels=[str(n) for n in ref_counts], column_labels=[str(n) for n in alt_counts])

        return cal_fig, cal_axes