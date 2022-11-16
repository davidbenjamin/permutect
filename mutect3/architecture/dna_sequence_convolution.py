from torch import nn
from math import floor


def conv_output_length(input_length, kernel_size=1, stride=1, pad=0, dilation=1, **kwargs):
    """
    Output length of 1D convolution or pooling given input length and various options.  Copied from PyTorch docs
    """
    return floor( ((input_length + (2 * pad) - ( dilation * (kernel_size - 1) ) - 1 )/ stride) + 1)


INITIAL_NUM_CHANNELS = 4    # one channel for each DNA base
TOKEN_SEPARATOR = '/'
KEY_VALUE_SEPARATOR = '='


class DNASequenceConvolution(nn.Module):
    """
    A fully-connected network (multi-layer perceptron) that we need frequently
    as a sub-network.  It is parameterized by the dimensions of its layers, starting with
    the input layer and ending with the output.  Output is logits and as such no non-linearity
    is applied after the last linear transformation.
    """

    def __init__(self, layer_strings, sequence_length):
        super(DNASequenceConvolution, self).__init__()

        last_layer_shape = (INITIAL_NUM_CHANNELS, sequence_length)   # we exclude the batch dimension, which is the first

        layers = []
        for layer_string in layer_strings:
            tokens = layer_string.split(TOKEN_SEPARATOR)
            layer_type_token = tokens[0]

            kwargs = {}
            for key_value_token in tokens[1:]:
                key, value = tuple(key_value_token.split(KEY_VALUE_SEPARATOR))
                kwargs[key] = int(value)    # we're assuming all params are integers

            match layer_type_token:
                case "convolution":
                    kwargs["in_channels"] = last_layer_shape[0]
                    layers.append(nn.Conv1d(**kwargs))
                    last_layer_shape = (kwargs["out_channels"], conv_output_length(last_layer_shape[1], **kwargs))
                case "pool":
                    assert last_layer_shape[1] > 1, "You are trying to pool a length-1 sequence, which, while defined, is silly"
                    layers.append(nn.MaxPool1d(**kwargs))
                    last_layer_shape = (last_layer_shape[0], conv_output_length(last_layer_shape[1], **kwargs))
                case "leaky_relu":
                    layers.append(nn.LeakyReLU())
                case "flatten":
                    layers.append(nn.Flatten()) # by default, batch dimension is not flattened
                    last_layer_shape = (last_layer_shape[0] * last_layer_shape[1], 1)   # no position left, everything is a "channel"
                case "linear":
                    assert last_layer_shape[1] == 1, "Trying to use fully-connected layer before data have been flattened"
                    kwargs["in_features"] = last_layer_shape[0]
                    layers.append(nn.Linear(**kwargs))
                    last_layer_shape = (kwargs["out_features"], 1)

        self._model = nn.Sequential(*layers)

    def forward(self, x):
        """
        :param x: a batch of DNA sequences represented as a 3D tensor -- 1st index batch, 2nd index channel (A, C, G, T),
                    3rd index position in the sequence.
        :return:
        """
        return self._model.forward(x)