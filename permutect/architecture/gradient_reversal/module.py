from .functional import revgrad
from torch import nn


class GradientReversal(nn.Module):
    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        return revgrad(x, self.alpha)

    def set_alpha(self, alpha_new):
        self.alpha = alpha_new