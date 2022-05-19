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

        layers = []
        for k in range(len(layer_sizes) - 1):
            input_dim, output_dim = layer_sizes[k], layer_sizes[k + 1]
            if batch_normalize:
                layers.append(nn.BatchNorm1d(num_features=input_dim))
            layers.append(nn.Linear(input_dim, output_dim))
            if dropout_p is not None and dropout_p > 0:
                layers.append(nn.Dropout(p=dropout_p))
            if k < len(layer_sizes) - 2:
                layers.append(nn.LeakyReLU())

        self._model = nn.Sequential(*layers)

    def forward(self, x):
        return self._model.forward(x)