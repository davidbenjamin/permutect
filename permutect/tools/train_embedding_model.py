import argparse

import torch
from torch.utils.tensorboard import SummaryWriter

from permutect import constants
from permutect.architecture.read_set_embedding import ReadSetEmbeddingParameters, ReadSetEmbedding, LearningMethod, \
    EmbeddingTrainingParameters
from permutect.data.read_set_dataset import ReadSetDataset
from permutect.tools.filter_variants import load_artifact_model


def train_embedding_model(params: ReadSetEmbeddingParameters, training_params: EmbeddingTrainingParameters, summary_writer: SummaryWriter,
                          dataset: ReadSetDataset, pretrained_model: ReadSetEmbedding = None) -> ReadSetEmbedding:
    use_gpu = torch.cuda.is_available()
    device = torch.device('cuda' if use_gpu else 'cpu')

    use_pretrained = (pretrained_model is not None)
    model = load_artifact_model(pretrained_model)[0] if use_pretrained else \
        ReadSetEmbedding(params=params, num_read_features=dataset.num_read_features, num_info_features=dataset.num_info_features,
                      ref_sequence_length=dataset.ref_sequence_length, device=device).float()

    print("Training. . .")
    # TODO: fix params here
    model.train_model(dataset, LearningMethod.SEMISUPERVISED, training_params, summary_writer=summary_writer)

    return model


def save_embedding_model(model, hyperparams, path):
    torch.save({
        constants.STATE_DICT_NAME: model.state_dict(),
        constants.HYPERPARAMS_NAME: hyperparams,
        constants.NUM_READ_FEATURES_NAME: model.num_read_features(),
        constants.NUM_INFO_FEATURES_NAME: model.num_info_features(),
        constants.REF_SEQUENCE_LENGTH_NAME: model.ref_sequence_length()
    }, path)


def parse_training_params(args) -> EmbeddingTrainingParameters:
    reweighting_range = getattr(args, constants.REWEIGHTING_RANGE_NAME)
    learning_rate = getattr(args, constants.LEARNING_RATE_NAME)
    weight_decay = getattr(args, constants.WEIGHT_DECAY_NAME)
    batch_size = getattr(args, constants.BATCH_SIZE_NAME)
    num_epochs = getattr(args, constants.NUM_EPOCHS_NAME)
    num_workers = getattr(args, constants.NUM_WORKERS_NAME)
    return EmbeddingTrainingParameters(batch_size, num_epochs, reweighting_range, learning_rate, weight_decay, num_workers=num_workers)


def parse_model_params(args) -> ReadSetEmbeddingParameters:
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
    return ReadSetEmbeddingParameters(read_embedding_dimension, num_transformer_heads, transformer_hidden_dimension,
                 num_transformer_layers, info_layers, aggregation_layers, ref_seq_layer_strings, dropout_p,
        batch_normalize, alt_downsample)


def parse_arguments():
    parser = argparse.ArgumentParser(description='train the Mutect3 artifact model')
    add_embedding_model_params_to_parser(parser)
    add_embedding_training_params_to_parser(parser)
    parser.add_argument('--' + constants.TRAIN_TAR_NAME, type=str, required=True,
                        help='tarfile of training/validation datasets produced by preprocess_dataset.py')
    parser.add_argument('--' + constants.OUTPUT_NAME, type=str, required=True, help='output saved model file')
    parser.add_argument('--' + constants.TENSORBOARD_DIR_NAME, type=str, default='tensorboard', required=False,
                        help='output tensorboard directory')

    return parser.parse_args()


def add_embedding_training_params_to_parser(parser):
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


def add_embedding_model_params_to_parser(parser):
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


def main_without_parsing(args):
    hyperparams = parse_model_params(args)
    training_params = parse_training_params(args)

    tarfile_data = getattr(args, constants.TRAIN_TAR_NAME)
    pretrained_model = getattr(args, constants.PRETRAINED_MODEL_NAME)
    tensorboard_dir = getattr(args, constants.TENSORBOARD_DIR_NAME)
    summary_writer = SummaryWriter(tensorboard_dir)
    dataset = ReadSetDataset(data_tarfile=tarfile_data, num_folds=10)
    model = train_embedding_model(params=hyperparams, dataset=dataset, training_params=training_params,
                                  summary_writer=summary_writer, pretrained_model=pretrained_model)

    summary_writer.close()
    save_embedding_model(model, hyperparams, getattr(args, constants.OUTPUT_NAME))


def main():
    args = parse_arguments()
    main_without_parsing(args)


if __name__ == '__main__':
    main()
