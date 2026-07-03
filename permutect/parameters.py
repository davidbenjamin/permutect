from typing import List

from permutect import constants
from permutect.utils.enums import ParameterSet

DEFAULT_READ_LAYERS = [30,-2, -2, -2]
DEFAULT_INFO_LAYERS = [20,-2,-2,-2]
DEFAULT_AGGREGATION_LAYERS = [-2,-2,10]
DEFAULT_GMLP_HIDDEN_DIM = 20
DEFAULT_NUM_GMLP_LAYERS = 6
DEFAULT_NUM_CLUSTERS = 4
DEFAULT_CNN_STRINGS = ["convolution/kernel_size=3/out_channels=32",
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
    pretrained_help = "optional pretrained model to initialize training"
    parser.add_argument("--" + constants.PRETRAINED_MODEL_NAME, required=False, type=str, help=pretrained_help)

    read_layers_help= "Read embedding hidden layer dimensions, including output. Negative values indicate residual skip connections"
    read_layers_kwargs = {"nargs": "+", "type": int, "default": DEFAULT_READ_LAYERS, "help": read_layers_help}
    parser.add_argument("--" + constants.READ_LAYERS_NAME, **read_layers_kwargs)

    info_layers_help = "Info embedding hidden layer dimensions, including output. Negative values indicate residual skip connections"
    info_layers_kwargs = {"nargs": "+", "type": int, "default": DEFAULT_INFO_LAYERS, "help": info_layers_help}
    parser.add_argument("--" + constants.INFO_LAYERS_NAME, **info_layers_kwargs)

    agg_layers_help = "Aggregation hidden layer dimensions, including output. Negative values indicate residual skip connections"
    agg_layers_kwargs = {"nargs": "+", "type": int, "default": DEFAULT_AGGREGATION_LAYERS, "help": agg_layers_help}
    parser.add_argument("--" + constants.AGGREGATION_LAYERS_NAME, **agg_layers_kwargs)


    hidden_dim_kwargs = {"type": int, "default": DEFAULT_GMLP_HIDDEN_DIM, "help": "self-attention hidden dimension"}
    parser.add_argument("--" + constants.SELF_ATTENTION_HIDDEN_DIMENSION_NAME, **hidden_dim_kwargs)

    gmlp_help = "Number of self-attention layers"
    gmlp_layers_kwargs = {"type": int, "default": DEFAULT_NUM_GMLP_LAYERS, "help": gmlp_help}
    parser.add_argument("--" + constants.NUM_SELF_ATTENTION_LAYERS_NAME, **gmlp_layers_kwargs)

    num_clusters_kwargs = {"type": int, "default": DEFAULT_NUM_CLUSTERS, "help": "number of artifact clusters"}
    parser.add_argument("--" + constants.NUM_ARTIFACT_CLUSTERS_NAME, **num_clusters_kwargs)

    cnn_help = "list of strings specifying convolution layers of the reference sequence embedding.  For example "
    "convolution/kernel_size=3/out_channels=64 pool/kernel_size=2 leaky_relu "
    "convolution/kernel_size=3/dilation=2/out_channels=5 leaky_relu flatten linear/out_features=10"
    cnn_kwargs = {"nargs": "+", "type": str, "default": DEFAULT_CNN_STRINGS, "help": cnn_help}
    parser.add_argument("--" + constants.REF_SEQ_LAYER_STRINGS_NAME, **cnn_kwargs)

    parser.add_argument("--" + constants.DROPOUT_P_NAME, type=float, default=DEFAULT_DROPOUT, help="dropout fraction")
    parser.add_argument("--" + constants.BATCH_NORMALIZE_NAME, action="store_true", help="enable batch normalization")

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
    tp_kwargs = {"nargs": "*", "type": str, "help": "parameter sets to be trained"}
    parser.add_argument("--" + constants.TRAINABLE_PARAMETERS_NAME, **tp_kwargs)

    lr_kwargs = {"type": float, "default": DEFAULT_LEARNING_RATE, "help": "learning rate"}
    parser.add_argument("--" + constants.LEARNING_RATE_NAME, **lr_kwargs)

    wd_kwargs = {"type": float, "default": DEFAULT_WEIGHT_DECAY, "help": "weight decay"}
    parser.add_argument("--" + constants.WEIGHT_DECAY_NAME, **wd_kwargs)

    parser.add_argument("--" + constants.BATCH_SIZE_NAME, type=int, default=DEFAULT_BATCH_SIZE, required=False, help="batch size")

    nw_kwargs = {"type": int, "default": DEFAULT_NUM_WORKERS, "help": "number of data loading subprocesses"}
    parser.add_argument("--" + constants.NUM_WORKERS_NAME, ** nw_kwargs)

    parser.add_argument("--" + constants.NUM_EPOCHS_NAME, type=int, default=DEFAULT_NUM_EPOCHS, help="training epochs")
