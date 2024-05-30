from typing import List

from permutect import constants


class RepresentationModelParameters:
    """
    note that read layers and info layers exclude the input dimension
    read_embedding_dimension: read tensors are linear-transformed to this dimension before
    input to the transformer.  This is also the output dimension of reads from the transformer
    num_transformer_heads: number of attention heads in the read transformer.  Must be a divisor
        of the read_embedding_dimension
    num_transformer_layers: number of layers of read transformer
    """
    def __init__(self, read_embedding_dimension: int, num_transformer_heads: int, transformer_hidden_dimension: int,
                 num_transformer_layers: int, info_layers: List[int], aggregation_layers: List[int],
                 ref_seq_layers_strings: List[str], dropout_p: float, batch_normalize: bool = False, alt_downsample: int = 100):

        assert read_embedding_dimension % num_transformer_heads == 0

        self.read_embedding_dimension = read_embedding_dimension
        self.num_transformer_heads = num_transformer_heads
        self.transformer_hidden_dimension = transformer_hidden_dimension
        self.num_transformer_layers = num_transformer_layers
        self.info_layers = info_layers
        self.aggregation_layers = aggregation_layers
        self.ref_seq_layer_strings = ref_seq_layers_strings
        self.dropout_p = dropout_p
        self.batch_normalize = batch_normalize
        self.alt_downsample = alt_downsample


def parse_representation_model_params(args) -> RepresentationModelParameters:
    read_embedding_dimension = getattr(args, constants.READ_EMBEDDING_DIMENSION_NAME)
    num_transformer_heads = getattr(args, constants.NUM_TRANSFORMER_HEADS_NAME)
    transformer_hidden_dimension = getattr(args, constants.TRANSFORMER_HIDDEN_DIMENSION_NAME)
    num_transformer_layers = getattr(args, constants.NUM_TRANSFORMER_LAYERS_NAME)

    info_layers = getattr(args, constants.INFO_LAYERS_NAME)
    aggregation_layers = getattr(args, constants.AGGREGATION_LAYERS_NAME)
    ref_seq_layer_strings = getattr(args, constants.REF_SEQ_LAYER_STRINGS_NAME)
    dropout_p = getattr(args, constants.DROPOUT_P_NAME)
    batch_normalize = getattr(args, constants.BATCH_NORMALIZE_NAME)
    alt_downsample = getattr(args, constants.ALT_DOWNSAMPLE_NAME)
    return RepresentationModelParameters(read_embedding_dimension, num_transformer_heads, transformer_hidden_dimension,
                                         num_transformer_layers, info_layers, aggregation_layers, ref_seq_layer_strings, dropout_p,
                                         batch_normalize, alt_downsample)


def add_representation_model_params_to_parser(parser):
    parser.add_argument('--' + constants.PRETRAINED_MODEL_NAME, required=False, type=str, help='optional pretrained Permutect embedding model')
    parser.add_argument('--' + constants.READ_EMBEDDING_DIMENSION_NAME, type=int, required=True,
                        help='dimension of read embedding output by the transformer')
    parser.add_argument('--' + constants.NUM_TRANSFORMER_HEADS_NAME, type=int, required=True,
                        help='number of transformer self-attention heads')
    parser.add_argument('--' + constants.TRANSFORMER_HIDDEN_DIMENSION_NAME, type=int, required=True,
                        help='hidden dimension of transformer keys and values')
    parser.add_argument('--' + constants.NUM_TRANSFORMER_LAYERS_NAME, type=int, required=True,
                        help='number of transformer layers')
    parser.add_argument('--' + constants.INFO_LAYERS_NAME, nargs='+', type=int, required=True,
                        help='dimensions of hidden layers in the info embedding subnetwork, including the dimension of the embedding itself.  '
                             'Negative values indicate residual skip connections')
    parser.add_argument('--' + constants.AGGREGATION_LAYERS_NAME, nargs='+', type=int, required=True,
                        help='dimensions of hidden layers in the aggregation subnetwork, excluding the dimension of input from lower subnetworks '
                             'and the dimension (1) of the output logit.  Negative values indicate residual skip connections')
    parser.add_argument('--' + constants.REF_SEQ_LAYER_STRINGS_NAME, nargs='+', type=str, required=True,
                        help='list of strings specifying convolution layers of the reference sequence embedding.  For example '
                             'convolution/kernel_size=3/out_channels=64 pool/kernel_size=2 leaky_relu '
                             'convolution/kernel_size=3/dilation=2/out_channels=5 leaky_relu flatten linear/out_features=10')
    parser.add_argument('--' + constants.DROPOUT_P_NAME, type=float, default=0.0, required=False,
                        help='dropout probability')
    parser.add_argument('--' + constants.ALT_DOWNSAMPLE_NAME, type=int, default=100, required=False,
                        help='max number of alt reads to downsample to inside the model')
    parser.add_argument('--' + constants.BATCH_NORMALIZE_NAME, action='store_true',
                        help='flag to turn on batch normalization')


# common parameters for training models
class TrainingParameters:
    def __init__(self, batch_size: int, num_epochs: int, reweighting_range: float, learning_rate: float = 0.001,
                 weight_decay: float = 0.01, num_workers: int = 0, num_calibration_epochs: int = 0):
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.reweighting_range = reweighting_range
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_workers = num_workers
        self.num_calibration_epochs = num_calibration_epochs


def parse_training_params(args) -> TrainingParameters:
    reweighting_range = getattr(args, constants.REWEIGHTING_RANGE_NAME)
    learning_rate = getattr(args, constants.LEARNING_RATE_NAME)
    weight_decay = getattr(args, constants.WEIGHT_DECAY_NAME)
    batch_size = getattr(args, constants.BATCH_SIZE_NAME)
    num_epochs = getattr(args, constants.NUM_EPOCHS_NAME)
    num_calibration_epochs = getattr(args, constants.NUM_CALIBRATION_EPOCHS_NAME)
    num_workers = getattr(args, constants.NUM_WORKERS_NAME)
    return TrainingParameters(batch_size, num_epochs, reweighting_range, learning_rate, weight_decay, num_workers, num_calibration_epochs)


def add_training_params_to_parser(parser):
    parser.add_argument('--' + constants.REWEIGHTING_RANGE_NAME, type=float, default=0.3, required=False,
                        help='magnitude of data augmentation by randomly weighted average of read embeddings.  '
                             'a value of x yields random weights between 1 - x and 1 + x')
    parser.add_argument('--' + constants.LEARNING_RATE_NAME, type=float, default=0.001, required=False,
                        help='learning rate')
    parser.add_argument('--' + constants.WEIGHT_DECAY_NAME, type=float, default=0.0, required=False,
                        help='learning rate')
    parser.add_argument('--' + constants.BATCH_SIZE_NAME, type=int, default=64, required=False,
                        help='batch size')
    parser.add_argument('--' + constants.NUM_WORKERS_NAME, type=int, default=0, required=False,
                        help='number of subprocesses devoted to data loading, which includes reading from memory map, '
                             'collating batches, and transferring to GPU.')
    parser.add_argument('--' + constants.NUM_EPOCHS_NAME, type=int, required=True,
                        help='number of epochs for primary training loop')
    parser.add_argument('--' + constants.NUM_CALIBRATION_EPOCHS_NAME, type=int, default=0, required=False,
                        help='number of calibration-only epochs')


class ArtifactModelParameters:
    def __init__(self, aggregation_layers: List[int], calibration_layers: List[int],
                 dropout_p: float = 0.0, batch_normalize: bool = False):
        self.aggregation_layers = aggregation_layers
        self.calibration_layers = calibration_layers
        self.dropout_p = dropout_p
        self.batch_normalize = batch_normalize


def parse_artifact_model_params(args) -> ArtifactModelParameters:
    aggregation_layers = getattr(args, constants.AGGREGATION_LAYERS_NAME)
    calibration_layers = getattr(args, constants.CALIBRATION_LAYERS_NAME)
    dropout_p = getattr(args, constants.DROPOUT_P_NAME)
    batch_normalize = getattr(args, constants.BATCH_NORMALIZE_NAME)
    return ArtifactModelParameters(aggregation_layers, calibration_layers, dropout_p, batch_normalize)


def add_artifact_model_params_to_parser(parser):
    parser.add_argument('--' + constants.AGGREGATION_LAYERS_NAME, nargs='+', type=int, required=True,
                        help='dimensions of hidden layers in the aggregation subnetwork, excluding the dimension of input from lower subnetworks '
                             'and the dimension (1) of the output logit.  Negative values indicate residual skip connections')
    parser.add_argument('--' + constants.CALIBRATION_LAYERS_NAME, nargs='+', type=int, required=True,
                        help='dimensions of hidden layers in the calibration subnetwork, excluding the dimension (1) of input logit and) '
                             'and the dimension (also 1) of the output logit.')
    parser.add_argument('--' + constants.DROPOUT_P_NAME, type=float, default=0.0, required=False,
                        help='dropout probability')
    parser.add_argument('--' + constants.BATCH_NORMALIZE_NAME, action='store_true',
                        help='flag to turn on batch normalization')