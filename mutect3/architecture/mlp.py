from torch import nn


class MLP(nn.Module):
    """
    A fully-connected network (multi-layer perceptron) that we need frequently
    as a sub-network.  It is parameterized by the dimensions of its layers, starting with
    the input layer and ending with the output.  Output is logits and as such no non-linearity
    is applied after the last linear transformation.
    """

    def __init__(self, layer_sizes, batch_normalize=False, dropout_p=None):
        super(MLP, self).__init__()

        self.layers = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.dropout = nn.ModuleList()
        for k in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[k], layer_sizes[k + 1]))

        if batch_normalize:
            for size in layer_sizes[1:]:
                self.bn.append(nn.BatchNorm1d(num_features=size))

        if dropout_p is not None:
            for _ in layer_sizes[1:]:
                self.dropout.append(nn.Dropout(p=dropout_p))

    def forward(self, x):
        for n, layer in enumerate(self.layers):
            x = layer(x)
            if self.bn:
                x = self.bn[n](x)
            if self.dropout:
                x = self.dropout[n](x)
            if n < len(self.layers) - 1:
                x = nn.functional.leaky_relu(x)
        return x