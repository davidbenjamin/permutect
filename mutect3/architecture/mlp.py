from torch import nn


class DenseSkipBlock(nn.Module):
    """
    computes x + f(x) where f(x) has some given number of linear layers, each with input and output dimension equal
    to that of the input x.  As suggested in arxiv:1603.05027, nonlinearities come before each linear transformation
    """
    def __init__(self, input_size: int, num_layers: int, batch_normalize: bool = False, dropout_p=None):
        super(DenseSkipBlock, self).__init__()
        self.mlp = MLP((num_layers + 1)*[input_size], batch_normalize, dropout_p, prepend_relu=True)

    def forward(self, x):
        return x + self.mlp.forward(x)


class MLP(nn.Module):
    """
    A fully-connected network (multi-layer perceptron) that we need frequently
    as a sub-network.  It is parameterized by the dimensions of its layers, starting with
    the input layer and ending with the output.  Output is logits and as such no non-linearity
    is applied after the last linear transformation.
    """

    def __init__(self, layer_sizes, batch_normalize=False, dropout_p=None, prepend_relu=False):
        super(MLP, self).__init__()

        layers = [nn.LeakyReLU()] if prepend_relu else []
        input_dim = layer_sizes[0]
        for k, output_dim in enumerate(layer_sizes[1:]):
            # negative output dimension -d will denote a d-layer residual skip connection
            # the output dimension of which equals the current input dimension
            if output_dim < 0:
                layers.append(DenseSkipBlock(input_dim, -output_dim, batch_normalize, dropout_p))
                continue

            if batch_normalize:
                layers.append(nn.BatchNorm1d(num_features=input_dim))

            layers.append(nn.Linear(input_dim, output_dim))

            if dropout_p is not None and dropout_p > 0:
                layers.append(nn.Dropout(p=dropout_p))

            # k runs from 0 to len(layer_sizes) - 2.  Omit the nonlinearity after the last layer.
            if k < len(layer_sizes) - 2:
                layers.append(nn.LeakyReLU())

            input_dim = output_dim  # note that this does not happen for a residual skip connection

        self._model = nn.Sequential(*layers)

    def forward(self, x):
        return self._model.forward(x)