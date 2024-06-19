from enum import Enum
from itertools import chain

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm.autonotebook import trange, tqdm

from permutect import utils, constants
from permutect.architecture.dna_sequence_convolution import DNASequenceConvolution
from permutect.architecture.mlp import MLP
from permutect.data.base_datum import BaseBatch, DEFAULT_GPU_FLOAT, DEFAULT_CPU_FLOAT
from permutect.data.base_dataset import BaseDataset
from permutect.metrics.evaluation_metrics import LossMetrics
from permutect.parameters import BaseModelParameters, TrainingParameters
from permutect.utils import Variation


# group rows into consecutive chunks to yield a 3D tensor, average over dim=1 to get
# 2D tensor of sums within each chunk
def sums_over_chunks(tensor2d: torch.Tensor, chunk_size: int):
    assert len(tensor2d) % chunk_size == 0
    return torch.sum(tensor2d.reshape([len(tensor2d) // chunk_size, chunk_size, -1]), dim=1)


class LearningMethod(Enum):
    # train the embedding by minimizing cross-entropy loss of binary predictor on labeled data
    SUPERVISED = "supervised"

    # same but use entropy regularization loss on unlabeled data
    SEMISUPERVISED = "semisupervised"

    # optimize a clustering model with center triplet loss
    SUPERVISED_CLUSTERING = "supervised_clustering"

    # modify data via a finite set of affine transformations and train the embedding to recognize which was applied
    AFFINE_TRANSFORMATION = "affine"


def make_transformer_encoder(input_dimension: int, params: BaseModelParameters):
    encoder_layer = torch.nn.TransformerEncoderLayer(d_model=input_dimension, nhead=params.num_transformer_heads,
                                                     batch_first=True, dim_feedforward=params.transformer_hidden_dimension, dropout=params.dropout_p)
    encoder_norm = torch.nn.LayerNorm(input_dimension)
    return torch.nn.TransformerEncoder(encoder_layer, num_layers=params.num_transformer_layers, norm=encoder_norm)


class BaseModel(torch.nn.Module):
    """
    DeepSets framework for reads and variant info.  We embed each read and concatenate the mean ref read
    embedding, mean alt read embedding, and variant info embedding, then apply an aggregation function to
    this concatenation to obtain an embedding / representation of the read set for downstream use such as
    variant filtering and clustering.

    hidden_read_layers: dimensions of layers for embedding reads, excluding input dimension, which is the
    size of each read's 1D tensor

    hidden_info_layers: dimensions of layers for embedding variant info, excluding input dimension, which is the
    size of variant info 1D tensor

    aggregation_layers: dimensions of layers for aggregation, excluding its input which is determined by the
    read and info embeddings.

    output_layers: dimensions of layers after aggregation, excluding the output dimension,
    which is 1 for a single logit representing artifact/non-artifact.  This is not part of the aggregation layers
    because we have different output layers for each variant type.
    """

    def __init__(self, params: BaseModelParameters, num_read_features: int, num_info_features: int, ref_sequence_length: int, device=utils.gpu_if_available()):
        super(BaseModel, self).__init__()

        self._device = device
        self._dtype = DEFAULT_GPU_FLOAT if device != torch.device("cpu") else DEFAULT_CPU_FLOAT
        self._ref_sequence_length = ref_sequence_length
        self._params = params
        self.alt_downsample = params.alt_downsample

        # embeddings of reads, info, and reference sequence prior to the transformer layers
        self.read_embedding = MLP([num_read_features] + params.read_layers, batch_normalize=params.batch_normalize, dropout_p=params.dropout_p)
        self.info_embedding = MLP([num_info_features] + params.info_layers, batch_normalize=params.batch_normalize, dropout_p=params.dropout_p)
        self.ref_seq_cnn = DNASequenceConvolution(params.ref_seq_layer_strings, ref_sequence_length)

        embedding_dim = self.read_embedding.output_dimension() + self.info_embedding.output_dimension() + self.ref_seq_cnn.output_dimension()
        assert embedding_dim % params.num_transformer_heads == 0

        self.alt_transformer_encoder = make_transformer_encoder(embedding_dim, params)
        self.ref_transformer_encoder = make_transformer_encoder(embedding_dim, params)

        # after passing alt and ref reads (along with info and ref seq embeddings) through transformers, concatenate and
        # pass through another MLP
        self.aggregation = MLP([2 * embedding_dim] + params.aggregation_layers, batch_normalize=params.batch_normalize, dropout_p=params.dropout_p)

        self.to(device=self._device, dtype=self._dtype)

    def output_dimension(self) -> int:
        return self.aggregation.output_dimension()

    def ref_sequence_length(self) -> int:
        return self._ref_sequence_length

    def set_epoch_type(self, epoch_type: utils.Epoch):
        if epoch_type == utils.Epoch.TRAIN:
            self.train(True)
            utils.unfreeze(self.parameters())
        else:
            self.train(False)
            utils.freeze(self.parameters())

    # I really don't like the forward method of torch.nn.Module with its implicit calling that PyCharm doesn't recognize
    def forward(self, batch: BaseBatch):
        pass

    # here 'v' means "variant index within a batch", 'r' means "read index within a variant or the batch", 'e' means "index within an embedding"
    # so, for example, "re" means a 2D tensor with all reads in the batch stacked and "vre" means a 3D tensor indexed
    # first by variant within the batch, then the read
    def calculate_representations(self, batch: BaseBatch, weight_range: float = 0) -> torch.Tensor:
        ref_count, alt_count = batch.ref_count, batch.alt_count
        total_ref, total_alt = ref_count * batch.size(), alt_count * batch.size()

        read_embeddings_re = self.read_embedding.forward(batch.get_reads_2d().to(device=self._device, dtype=self._dtype))
        info_embeddings_ve = self.info_embedding.forward(batch.get_info_2d().to(device=self._device, dtype=self._dtype))
        ref_seq_embeddings_ve = self.ref_seq_cnn(batch.get_ref_sequences_2d().to(device=self._device, dtype=self._dtype))
        info_and_seq_ve = torch.hstack((info_embeddings_ve, ref_seq_embeddings_ve))
        info_and_seq_re = torch.vstack((torch.repeat_interleave(info_and_seq_ve, ref_count, dim=0),
                                       torch.repeat_interleave(info_and_seq_ve, alt_count, dim=0)))
        reads_info_seq_re = torch.hstack((read_embeddings_re, info_and_seq_re))
        ref_reads_info_seq_vre = None if total_ref == 0 else reads_info_seq_re[:total_ref].reshape(batch.size(), ref_count, -1)
        alt_reads_info_seq_vre = reads_info_seq_re[total_ref:].reshape(batch.size(), alt_count, -1)

        if self.alt_downsample < alt_count:
            alt_read_indices = torch.randperm(alt_count)[:self.alt_downsample]
            alt_reads_info_seq_vre = alt_reads_info_seq_vre[:, alt_read_indices, :]   # downsample only along the middle (read) dimension
            alt_count = self.alt_downsample
            total_alt = batch.size() * self.alt_downsample

        # undo some of the above rearrangement
        transformed_alt_vre = self.alt_transformer_encoder(alt_reads_info_seq_vre)
        transformed_ref_vre = None if total_ref == 0 else self.ref_transformer_encoder(ref_reads_info_seq_vre)

        all_read_means_ve = ((0 if ref_count == 0 else torch.sum(transformed_ref_vre, dim=1)) + torch.sum(transformed_alt_vre, dim=1)) / (alt_count + ref_count)

        alt_weights_vr = 1 + weight_range * (1 - 2 * torch.rand(batch.size(), alt_count, device=self._device, dtype=self._dtype))
        alt_wt_sums = torch.sum(alt_weights_vr, dim=1, keepdim=True)
        # normalized so read weights within each variant sum to 1 and add dummy e dimension for broadcasting the multiply below
        normalized_alt_weights_vr1 = (alt_weights_vr / alt_wt_sums).reshape(batch.size(), alt_count, 1)
        alt_means_ve = torch.sum(transformed_alt_vre * normalized_alt_weights_vr1, dim=1)

        concat_ve = torch.hstack((alt_means_ve, all_read_means_ve))
        result_ve = self.aggregation.forward(concat_ve)

        return result_ve


    # TODO: we need to attach an MLP on top of the embedding model to output logits, then train this combination
    # TODO: actually, this can be the framework of a LOT of different ways to train.  There's a ton of overlap.  There's always going to
    # TODO: be running the model over epochs, loading the dataset, backpropagating the loss.
    # TODO: the differences will mainly be in auxiliary tasks attached to the embedding and different loss functions
    def learn(self, dataset: BaseDataset, learning_method: LearningMethod, training_params: TrainingParameters,
              summary_writer: SummaryWriter, validation_fold: int = None):
        bce = torch.nn.BCEWithLogitsLoss(reduction='none')  # no reduction because we may want to first multiply by weights for unbalanced data

        # TODO: use Python's match syntax, but this requires updating Python version in the docker
        if learning_method == LearningMethod.SUPERVISED or learning_method == LearningMethod.SEMISUPERVISED:
            top_layer = torch.nn.Linear(in_features=self.output_dimension(), out_features=1)
        else:
            raise Exception("not implemented yet")

        train_optimizer = torch.optim.AdamW(chain(self.parameters(), top_layer.parameters()),
                                            lr=training_params.learning_rate, weight_decay=training_params.weight_decay)

        artifact_to_non_artifact_ratios = torch.from_numpy(dataset.artifact_to_non_artifact_ratios()).to(device=self._device, dtype=self._dtype)
        artifact_to_non_artifact_log_prior_ratios = torch.log(artifact_to_non_artifact_ratios)

        # balance training by weighting the loss function
        # if total unlabeled is less than total labeled, we do not compensate, since labeled data are more informative
        total_labeled, total_unlabeled = dataset.total_labeled_and_unlabeled()
        labeled_to_unlabeled_ratio = 1 if total_unlabeled < total_labeled else total_labeled / total_unlabeled

        print("Training data contains {} labeled examples and {} unlabeled examples".format(total_labeled, total_unlabeled))
        for variation_type in utils.Variation:
            idx = variation_type.value
            print("For variation type {}, there are {} labeled artifact examples and {} labeled non-artifact examples"
                  .format(variation_type.name, dataset.artifact_totals[idx].item(), dataset.non_artifact_totals[idx].item()))

        validation_fold_to_use = (dataset.num_folds - 1) if validation_fold is None else validation_fold
        train_loader = dataset.make_data_loader(dataset.all_but_one_fold(validation_fold_to_use), training_params.batch_size, self._device.type == 'cuda', training_params.num_workers)
        valid_loader = dataset.make_data_loader([validation_fold_to_use], training_params.batch_size, self._device.type == 'cuda', training_params.num_workers)

        for epoch in trange(1, training_params.num_epochs + 1, desc="Epoch"):
            for epoch_type in (utils.Epoch.TRAIN, utils.Epoch.VALID):
                self.set_epoch_type(epoch_type)

                loss_metrics = LossMetrics(self._device)

                loader = train_loader if epoch_type == utils.Epoch.TRAIN else valid_loader
                pbar = tqdm(enumerate(loader), mininterval=60)
                for n, batch in pbar:
                    embeddings = self.calculate_representations(batch, self._params.reweighting_range)

                    # TODO: this is only handling the supervised/semi-supervised cases
                    logits = torch.squeeze(top_layer(embeddings), dim=1)

                    types_one_hot = batch.variant_type_one_hot()
                    log_prior_ratios = torch.sum(artifact_to_non_artifact_log_prior_ratios * types_one_hot, dim=1)
                    posterior_logits = logits + log_prior_ratios

                    if batch.is_labeled():
                        separate_losses = bce(posterior_logits, batch.labels)
                        loss = torch.sum(separate_losses)

                        loss_metrics.record_total_batch_loss(loss.detach(), batch)
                        loss_metrics.record_losses_by_type_and_count(separate_losses, batch)
                    else:
                        # unlabeled loss: entropy regularization
                        posterior_probabilities = torch.sigmoid(posterior_logits)
                        entropies = torch.nn.functional.binary_cross_entropy_with_logits(posterior_logits, posterior_probabilities, reduction='none')

                        # TODO: we need a parameter to control the relative weight of unlabeled loss to labeled loss
                        loss = torch.sum(entropies) * labeled_to_unlabeled_ratio
                        loss_metrics.record_total_batch_loss(loss.detach(), batch)

                    if epoch_type == utils.Epoch.TRAIN:
                        utils.backpropagate(train_optimizer, loss)

                # done with one epoch type -- training or validation -- for this epoch
                loss_metrics.write_to_summary_writer(epoch_type, epoch, summary_writer)

                print("Labeled loss for epoch " + str(epoch) + " of " + epoch_type.name + ": " + str(loss_metrics.get_labeled_loss()))
            # done with training and validation for this epoch
            # note that we have not learned the AF spectrum yet
        # done with training

    def save(self, path):
        torch.save({
            constants.STATE_DICT_NAME: self.state_dict(),
            constants.HYPERPARAMS_NAME: self._params,
            constants.NUM_READ_FEATURES_NAME: self.read_embedding.in_features,
            constants.NUM_INFO_FEATURES_NAME: self.info_embedding.input_dimension(),
            constants.REF_SEQUENCE_LENGTH_NAME: self.ref_sequence_length()
        }, path)


def load_base_model(path, device: torch.device = utils.gpu_if_available()) -> BaseModel:
    saved = torch.load(path)
    hyperparams = saved[constants.HYPERPARAMS_NAME]
    num_read_features = saved[constants.NUM_READ_FEATURES_NAME]
    num_info_features = saved[constants.NUM_INFO_FEATURES_NAME]
    ref_sequence_length = saved[constants.REF_SEQUENCE_LENGTH_NAME]

    model = BaseModel(hyperparams, num_read_features=num_read_features, num_info_features=num_info_features,
                      ref_sequence_length=ref_sequence_length, device=device)
    model.load_state_dict(saved[constants.STATE_DICT_NAME])

    # in case the state dict had the wrong dtype for the device we're on now eg base model was pretrained on GPU
    # and we're now on CPU
    model.to(model._dtype)

    return model
