from typing import Any

from torch.autograd import Function


class GradientReversal(Function):
    @staticmethod
    def jvp(ctx: Any, *grad_inputs: Any) -> Any:
        pass

    @staticmethod
    def forward(ctx, x, alpha: float = 1.0):
        ctx.save_for_backward(x)
        ctx.alpha = alpha
        return x

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None
        if ctx.needs_input_grad[0]:
            grad_input = - ctx.alpha * grad_output
        return grad_input, None


revgrad = GradientReversal.apply
