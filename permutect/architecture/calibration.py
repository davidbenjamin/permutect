from typing import List

import torch
from matplotlib import pyplot as plt
from torch import nn, Tensor

from permutect.architecture.monotonic import MonoDense
from permutect.metrics import plotting


class Calibration(nn.Module):

    def __init__(self, hidden_layer_sizes: List[int]):
        super(Calibration, self).__init__()

        # calibration takes [logit, ref count, alt count] as input and maps it to [calibrated logit]
        # it is monotonically increasing in logit, unconstrained in ref and alt count
        # we initialize it to calibrated logit = input logit

        # likewise, we cap the effective alt and ref counts and input logits to avoid arbitrarily large confidence
        self.max_alt = nn.Parameter(torch.tensor(20.0))
        self.max_ref = nn.Parameter(torch.tensor(20.0))
        self.max_input_logit = nn.Parameter(torch.tensor(20.0))

        center_spacing = 1
        ref_center_spacing = 5

        # centers of Gaussian comb featurizations
        # note: even though they aren't learned and requires_grad is False, we still wrap them in nn.Parameter
        # so that they can be sent to GPU recursively when the grandparent ArtifactModel is
        self.alt_centers = nn.Parameter(torch.arange(start=1, end=20, step=center_spacing), requires_grad=False)
        self.ref_centers = nn.Parameter(torch.arange(start=1, end=20, step=ref_center_spacing), requires_grad=False)

        # increasing in the 1st feature, logits
        # logit is one feature, then the Gaussian comb for alt and ref counts is the other
        self.monotonic = MonoDense(1 + len(self.ref_centers) + len(self.alt_centers), hidden_layer_sizes + [1], 1, 0)

        self.max_alt_count_for_adjustment = 20

        # Final layer of calibration is a count-dependent linear shift.  This is particularly useful when calibrating only on
        # a subset of data sources
        self.final_adjustments = nn.Parameter(torch.zeros(self.max_alt_count_for_adjustment + 1), requires_grad=True)

    def calibrated_logits(self, logits_b: Tensor, ref_counts_b: Tensor, alt_counts_b: Tensor):
        logits_bc = torch.tanh(logits_b / self.max_input_logit)[:, None]

        ref_comb_bc = torch.softmax(-torch.square(ref_counts_b[:, None] - self.ref_centers[None, :]).float(), dim=1)
        alt_comb_bc = torch.softmax(-torch.square(alt_counts_b[:, None] - self.alt_centers[None, :]).float(), dim=1)
        input_2d = torch.hstack([logits_bc, ref_comb_bc, alt_comb_bc])
        calibrated_b = self.monotonic.forward(input_2d).squeeze()

        counts_for_adjustment = torch.clamp(alt_counts_b, max=self.max_alt_count_for_adjustment).long()
        adjustments = self.final_adjustments[counts_for_adjustment]

        return calibrated_b + adjustments

    def forward(self, logits, ref_counts: Tensor, alt_counts: Tensor):
        return self.calibrated_logits(logits, ref_counts, alt_counts)

    def plot_calibration_module(self):
        device, dtype = self.final_adjustments.device, self.final_adjustments.dtype
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