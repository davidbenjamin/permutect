from torch import nn
import torch


class MonoDenseLayer(nn.Module):
    """
    MonoDenseLayer from Constrained Monotonic Neural Networks, Runje and Shankaranarayana, https://arxiv.org/abs/2205.11775

    It is a modification of a plain old linear layer.

    1) The output is constrained to be monotonically increasing, decreasing, or unconstrained with respect to each input

    2) Input vectors are assumed ordered with increasing features, then decreasing, then unconstrained
    """

    def __init__(self, input_dimension: int, output_dimension: int, num_increasing: int, num_decreasing):
        super(MonoDenseLayer, self).__init__()

        num_constrained = num_increasing + num_decreasing
        num_free = input_dimension - num_constrained
        assert num_constrained <= input_dimension
        assert num_constrained > 0

        # mask has -1's for decreasing features, otherwise 1's
        # in the forward pass we multiply by the mask for convenience so that monotonically increasing AND decreasing can both
        # be treated as increasing
        self.mask = torch.ones(input_dimension)
        self.mask[num_increasing: num_increasing + num_decreasing] = -self.mask[num_increasing: num_increasing + num_decreasing]

        # it's redundant that both weight matrices W are linear layers containing
        # TODO: we can't use a linear layer because we have to take the absolute value
        # TODO: instead we must go to a raw weight matrix and bias froms cratch
        self.monotonic_W = nn.Linear(in_features=(num_increasing+num_decreasing), out_features=output_dimension)
        self.free_W = nn.Linear(in_features=num_free, out_features=output_dimension) if num_free > 0 else None

        self.b = torch.

    def forward(self, x):
        flipped = x * self.mask