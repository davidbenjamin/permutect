from torch import nn
import torch.nn.functional as F
import torch
import math
from typing import List


class MonoDenseLayer(nn.Module):
    """
    MonoDenseLayer from Constrained Monotonic Neural Networks, Runje and Shankaranarayana, https://arxiv.org/abs/2205.11775

    It is a modification of a plain old linear layer.

    1) The output is constrained to be monotonically increasing, decreasing, or unconstrained with respect to each input

    2) Input vectors are assumed ordered with increasing features, then decreasing, then unconstrained
    """

    def __init__(self, input_dimension: int, output_dimension: int, num_increasing: int, num_decreasing):
        super(MonoDenseLayer, self).__init__()

        self.num_constrained = num_increasing + num_decreasing
        num_free = input_dimension - self.num_constrained
        assert self.num_constrained <= input_dimension
        assert self.num_constrained > 0

        self.input_dimension = input_dimension
        self.output_dimension = output_dimension

        # mask has -1's for decreasing features, otherwise 1's
        # in the forward pass we multiply by the mask for convenience so that monotonically increasing AND decreasing can both
        # be treated as increasing
        self.mask = torch.ones(input_dimension)
        self.mask[num_increasing: num_increasing + num_decreasing] = -self.mask[num_increasing: self.num_constrained]

        # it's redundant that both weight matrices W are linear layers containing
        # TODO: we can't use a linear layer because we have to take the absolute value
        # TODO: instead we must go to a raw weight matrix and bias froms cratch
        self.monotonic_W = nn.Parameter(torch.empty((output_dimension, input_dimension)))
        nn.init.kaiming_uniform_(self.monotonic_W, a=math.sqrt(5))

        self.free_W = nn.Parameter(torch.empty((output_dimension, input_dimension))) if num_free > 0 else None
        if self.free_W is not None:
            nn.init.kaiming_uniform_(self.free_W, a=math.sqrt(5))

        self.b = nn.Parameter(torch.empty(output_dimension))
        bound = 1 / math.sqrt(input_dimension)
        nn.init.uniform_(self.b, -bound, bound)

    def forward(self, x):
        flipped = x * self.mask

        # note that monotonicity is enforced by taking the absolute value of the monotonic weight matrix
        monotonic_contribution = F.linear(flipped[:, :self.num_constrained], torch.abs(self.monotonic_W))
        free_contribution = F.linear(flipped[:, self.num_constrained:], self.free_W)

        before_activation = monotonic_contribution + free_contribution + self.b

        # as in the paper, we apply three nonlinear activation functions: 1) an ordinary convex activation g(x) which
        # could be a ReLU, leaky ReLU, tanh etc; 2) the concave reflection -g(-x); 3) g(x+1)-g(1) (if x < 0) or g(1) - g(1-x) (if x > 0)

        features_per_activation = self.output_dimension // 3

        left = before_activation[:, :features_per_activation]
        middle = before_activation[:, features_per_activation:(2*features_per_activation)]
        right = before_activation[:, (2*features_per_activation):]

        output1 = self.convex_activation(left)
        output2 = -self.convex_activation(-middle)
        output3 = torch.sgn(right)*(self.convex_activation(torch.ones_like(right)) - self.convex_activation(1-torch.abs(right)))

        return torch.hstack([output1, output2, output3])


class MonoDense(nn.Module):
    """

    """

    def __init__(self, input_dimension: int, output_dimensions: List[int], num_increasing: int, num_decreasing):
        super(MonoDense, self).__init__()

        self.input_dimension = input_dimension
        self.layers = torch.nn.Sequential()

        for layer, dim in enumerate(output_dimensions):
            if layer == 0:
                self.layers.append(MonoDenseLayer(input_dimension, dim, num_increasing, num_decreasing))
            else:
                # note how layers after the first are pure monotonically increasing
                input_dim_to_layer = output_dimensions[layer-1]
                self.layers.append(MonoDenseLayer(input_dim_to_layer, dim, input_dim_to_layer, 0))

    def forward(self, x):
        return self.layers.forward(x)
