from typing import List

from permutect import constants
from permutect.utils.enums import ParameterSet

DEFAULT_READ_LAYERS = [30,-2, -2, -2]
DEFAULT_INFO_LAYERS = [20,-2,-2,-2]
DEFAULT_AGGREGATION_LAYERS = [-2,-2,10]
DEFAULT_SELF_ATTENTION_HIDDEN_DIMENSION = 20
DEFAULT_NUM_SELF_ATTENTION_LAYERS = 6
DEFAULT_NUM_ARTIFACT_CLUSTERS = 4
DEFAULT_REF_SEQ_LAYER_STRINGS = ["convolution/kernel_size=3/out_channels=32",
                                 "selu", "pool/kernel_size=2/stride=1",
                                 "convolution/kernel_size=3/out_channels=32",
                                 "selu", "pool/kernel_size=1",
                                 "convolution/kernel_size=5/out_channels=32",
                                 "selu", "pool/kernel_size=2",
                                 "convolution/kernel_size=5/out_channels=32",
                                 "selu",
                                 "pool/kernel_size=2",
                                 "flatten",
                                 "linear/out_features=10"]

DEFAULT_NUM_EPOCHS = 10
DEFAULT_DROPOUT = 0.0
DEFAULT_BATCH_SIZE = 1024
DEFAULT_NUM_WORKERS = 0
DEFAULT_WEIGHT_DECAY = 0.0
DEFAULT_LEARNING_RATE = 0.001

class ModelParameters:
    """
    note that read layers and info layers exclude the input dimension
    read_embedding_dimension: read tensors are linear-transformed to this dimension before
    input to the transformer.  This is also the output dimension of reads from the transformer
    num_transformer_heads: number of attention heads in the read transformer.  Must be a divisor
        of the read_embedding_dimension
    num_transformer_layers: number of layers of read transformer
    """

    def __init__(
        self,
        read_layers: List[int],
        self_attention_hidden_dimension: int,
        num_self_attention_layers: int,
        info_layers: List[int],
        aggregation_layers: List[int],
        num_artifact_clusters: int,
        ref_seq_layers_strings: List[str],
        dropout_p: float,
        batch_normalize: bool = False,
    ):
        self.read_layers = read_layers
        self.info_layers = info_layers
        self.ref_seq_layer_strings = ref_seq_layers_strings
        self.self_attention_hidden_dimension = self_attention_hidden_dimension
        self.num_self_attention_layers = num_self_attention_layers
        self.aggregation_layers = aggregation_layers
        self.num_artifact_clusters = num_artifact_clusters
        self.dropout_p = dropout_p
        self.batch_normalize = batch_normalize


def parse_model_params(args) -> ModelParameters:
    read_layers = getattr(args, constants.READ_LAYERS_NAME)
    info_layers = getattr(args, constants.INFO_LAYERS_NAME)
    ref_seq_layer_strings = getattr(args, constants.REF_SEQ_LAYER_STRINGS_NAME)
    self_attention_hidden_dimension = getattr(args, constants.SELF_ATTENTION_HIDDEN_DIMENSION_NAME)
    num_self_attention_layers = getattr(args, constants.NUM_SELF_ATTENTION_LAYERS_NAME)
    aggregation_layers = getattr(args, constants.AGGREGATION_LAYERS_NAME)
    num_artifact_clusters = getattr(args, constants.NUM_ARTIFACT_CLUSTERS_NAME)
    dropout_p = getattr(args, constants.DROPOUT_P_NAME)
    batch_normalize = getattr(args, constants.BATCH_NORMALIZE_NAME)
    return ModelParameters(
        read_layers,
        self_attention_hidden_dimension,
        num_self_attention_layers,
        info_layers,
        aggregation_layers,
        num_artifact_clusters,
        ref_seq_layer_strings,
        dropout_p,
        batch_normalize,
    )


def add_model_params_to_parser(parser):
    parser.add_argument(
        "--" + constants.PRETRAINED_ARTIFACT_MODEL_NAME,
        required=False,
        type=str,
        help="optional pretrained model to initialize training",
    )
    parser.add_argument(
        "--" + constants.READ_LAYERS_NAME,
        nargs="+",
        type=int,
        required=False,
        default=DEFAULT_READ_LAYERS,
        help="dimensions of hidden layers in the read embedding subnetwork, including the dimension of the embedding itself.  "
        "Negative values indicate residual skip connections",
    )
    parser.add_argument(
        "--" + constants.INFO_LAYERS_NAME,
        nargs="+",
        type=int,
        required=False,
        default=DEFAULT_INFO_LAYERS,
        help="dimensions of hidden layers in the info embedding subnetwork, including the dimension of the embedding itself.  "
             "Negative values indicate residual skip connections",
    )
    parser.add_argument(
        "--" + constants.AGGREGATION_LAYERS_NAME,
        nargs="+",
        type=int,
        required=False,
        default=DEFAULT_AGGREGATION_LAYERS,
        help="dimensions of hidden layers in the aggregation subnetwork, excluding the dimension of input from lower subnetworks "
        "and the dimension (1) of the output logit.  Negative values indicate residual skip connections",
    )
    parser.add_argument(
        "--" + constants.SELF_ATTENTION_HIDDEN_DIMENSION_NAME,
        type=int,
        required=False,
        default=DEFAULT_SELF_ATTENTION_HIDDEN_DIMENSION,
        help="hidden dimension of transformer keys and values",
    )
    parser.add_argument(
        "--" + constants.NUM_SELF_ATTENTION_LAYERS_NAME,
        type=int,
        required=False,
        default=DEFAULT_NUM_SELF_ATTENTION_LAYERS,
        help="number of symmetric gated MLP self-attention layers",
    )
    parser.add_argument(
        "--" + constants.NUM_ARTIFACT_CLUSTERS_NAME,
        type=int,
        default=DEFAULT_NUM_ARTIFACT_CLUSTERS,
        required=False,
        help="number of clusters for representing different types of artifact",
    )
    parser.add_argument(
        "--" + constants.REF_SEQ_LAYER_STRINGS_NAME,
        nargs="+",
        type=str,
        required=False,
        default=DEFAULT_REF_SEQ_LAYER_STRINGS,
        help="list of strings specifying convolution layers of the reference sequence embedding.  For example "
        "convolution/kernel_size=3/out_channels=64 pool/kernel_size=2 leaky_relu "
        "convolution/kernel_size=3/dilation=2/out_channels=5 leaky_relu flatten linear/out_features=10",
    )
    parser.add_argument(
        "--" + constants.DROPOUT_P_NAME,
        type=float,
        default=0.0,
        required=False,
        help="dropout probability",
    )
    parser.add_argument(
        "--" + constants.BATCH_NORMALIZE_NAME,
        action="store_true",
        help="flag to turn on batch normalization",
    )


# common parameters for training models
class TrainingParameters:
    def __init__(
        self,
        batch_size: int,
        num_epochs: int,
        learning_rate: float = 0.001,
        weight_decay: float = 0.01,
        num_workers: int = 0,
        trainable_parameter_sets: List[ParameterSet] = None,
    ):
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_workers = num_workers

        self.trainable_parameter_sets = [ParameterSet.WHOLE_MODEL] if trainable_parameter_sets is None else trainable_parameter_sets


def parse_training_params(args) -> TrainingParameters:
    learning_rate = getattr(args, constants.LEARNING_RATE_NAME)
    weight_decay = getattr(args, constants.WEIGHT_DECAY_NAME)
    batch_size = getattr(args, constants.BATCH_SIZE_NAME)
    num_epochs = getattr(args, constants.NUM_EPOCHS_NAME)
    num_workers = getattr(args, constants.NUM_WORKERS_NAME)

    trainable_parameter_strings = getattr(args, constants.TRAINABLE_PARAMETERS_NAME)
    trainable_parameter_sets = (
        [ParameterSet.WHOLE_MODEL]
        if trainable_parameter_strings is None
        else [ParameterSet.get_parameter_set(set_str) for set_str in trainable_parameter_strings]
    )

    return TrainingParameters(
        batch_size,
        num_epochs,
        learning_rate,
        weight_decay,
        num_workers,
        trainable_parameter_sets,
    )


def add_training_params_to_parser(parser):
    parser.add_argument(
        "--" + constants.TRAINABLE_PARAMETERS_NAME,
        nargs="*",
        type=str,
        required=False,
        help="zero or more parameter set types to be re-fit in test time domain adaptation",
    )

    parser.add_argument(
        "--" + constants.LEARNING_RATE_NAME,
        type=float,
        default=DEFAULT_LEARNING_RATE,
        help="learning rate",
    )
    parser.add_argument(
        "--" + constants.WEIGHT_DECAY_NAME,
        type=float,
        default=DEFAULT_WEIGHT_DECAY,
        help="weight decay",
    )
    parser.add_argument("--" + constants.BATCH_SIZE_NAME, type=int, default=64, required=False, help="batch size")
    parser.add_argument(
        "--" + constants.NUM_WORKERS_NAME,
        type=int,
        default=DEFAULT_NUM_WORKERS,
        help="number of subprocesses devoted to data loading, which includes reading from memory map, "
        "collating batches, and transferring to GPU.",
    )
    parser.add_argument("--" + constants.NUM_EPOCHS_NAME, type=int, default=DEFAULT_NUM_EPOCHS, help="training epochs")
