import torch.nn
from permutect.architecture.gradient_reversal.module import GradientReversal


class Adversarial(torch.nn.Module):
    """
    Simple wrapper class that converts a Pytorch module into an adversarial task in which all parameters of the module
    have regular derivatives and try to minimize the loss associated with the wrapped module's output, while upstream parameters
    that precede the module in a larger network have a reversed derivative and hence try to maximize the loss.
    That is, the wrapped module tries to succeed and the network as a whole tries to feed it features that make the task
    more difficult.
    """
    def __init__(self, wrapped_module: torch.nn.Module, adversarial_strength: float = 1.0):
        super(Adversarial, self).__init__()
        self.wrapped_module = wrapped_module
        self.gradient_reversal = GradientReversal(alpha=adversarial_strength)

    def set_adversarial_strength(self, new_alpha):
        self.gradient_reversal.set_alpha(new_alpha)

    def forward(self, x):
        return self.wrapped_module.forward(self.gradient_reversal(x))

    def adversarial_forward(self, x):
        return self.forward(x)