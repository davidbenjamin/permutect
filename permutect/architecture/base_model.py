from abc import ABC, abstractmethod
from enum import Enum
from itertools import chain
from typing import List

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


# group rows into consecutive chunks to yield a 3D tensor, average over dim=1 to get
# 2D tensor of sums within each chunk
def sums_over_chunks(tensor2d: torch.Tensor, chunk_size: int):
    assert len(tensor2d) % chunk_size == 0
    return torch.sum(tensor2d.reshape([len(tensor2d) // chunk_size, chunk_size, -1]), dim=1)


class LearningMethod(Enum):
    # train the embedding by minimizing cross-entropy loss of binary predictor on labeled data
    SUPERVISED = "SUPERVISED"

    # same but use entropy regularization loss on unlabeled data
    SEMISUPERVISED = "SEMISUPERVISED"

    # TODO: IMPLEMENT THIS
    # optimize a clustering model with center triplet loss
    SUPERVISED_CLUSTERING = "SUPERVISED_CLUSTERING"

    # TODO: IMPLEMENT THIS
    # modify data via a finite set of affine transformations and train the embedding to recognize which was applied
    AFFINE_TRANSFORMATION = "AFFINE"

    # modify data via a finite set of affine transformations and train the embedding to recognize which was applied
    MASK_PREDICTION = "MASK_PREDICTION"

    AUTOENCODER = "AUTOENCODER"


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

    def save(self, path):
        torch.save({
            constants.STATE_DICT_NAME: self.state_dict(),
            constants.HYPERPARAMS_NAME: self._params,
            constants.NUM_READ_FEATURES_NAME: self.read_embedding.input_dimension(),
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


# outputs a 1D tensor of losses over the batch
class BaseModelLearningStrategy(ABC):
    @abstractmethod
    def loss_function(self, base_model: BaseModel, base_batch: BaseBatch):
        pass


class BaseModelSemiSupervisedLoss(torch.nn.Module, BaseModelLearningStrategy):
    def __init__(self, input_dim: int, hidden_top_layers: List[int], params: BaseModelParameters,
                 artifact_to_non_artifact_log_prior_ratios, labeled_to_unlabeled_ratio):
        super(BaseModelSemiSupervisedLoss, self).__init__()
        self.weight_range = params.reweighting_range
        self.artifact_to_non_artifact_log_prior_ratios = artifact_to_non_artifact_log_prior_ratios
        self.labeled_to_unlabeled_ratio = labeled_to_unlabeled_ratio

        self.bce = torch.nn.BCEWithLogitsLoss(reduction='none')  # no reduction because we may want to first multiply by weights for unbalanced data

        # go from base model output representation to artifact logit for supervised loss
        self.logit_predictor = MLP([input_dim] + hidden_top_layers + [1], batch_normalize=params.batch_normalize, dropout_p=params.dropout_p)

    def loss_function(self, base_model: BaseModel, base_batch: BaseBatch):
        representations = base_model.calculate_representations(base_batch, self.weight_range)


        logits = self.logit_predictor.forward(representations).reshape((base_batch.size()))

        types_one_hot = base_batch.variant_type_one_hot()
        log_prior_ratios = torch.sum(self.artifact_to_non_artifact_log_prior_ratios * types_one_hot, dim=1)
        posterior_logits = logits + log_prior_ratios

        if base_batch.is_labeled():
            return self.bce(posterior_logits, base_batch.labels)
        else:
            # unlabeled loss: entropy regularization
            posterior_probabilities = torch.sigmoid(posterior_logits)
            entropies = self.bce(posterior_logits, posterior_probabilities)

            # TODO: we need a parameter to control the relative weight of unlabeled loss to labeled loss
            return entropies * self.labeled_to_unlabeled_ratio

    # I don't like implicit forward!!
    def forward(self):
        pass


def permute_columns_independently(mat: torch.Tensor):
    assert mat.dim() == 2
    num_rows, num_cols = mat.size()
    weights = torch.ones(num_rows)

    result = torch.clone(mat)
    for col in range(num_cols):
        idx = torch.multinomial(weights, num_rows, replacement=True)
        result[:, col] = result[:, col][idx]
    return result


# randomly choose read features to "mask" -- where a masked feature is permuted randomly over all the reads in the batch.
# this essentially means drawing masked features from the empirical marginal distribution
# the pretext self-supervision task is, for each datum, to predict which features were masked
# note that this basically means destroy correlations for a random selection of features
class BaseModelMaskPredictionLoss(torch.nn.Module, BaseModelLearningStrategy):
    def __init__(self, num_read_features: int, base_model_output_dim: int, hidden_top_layers: List[int], params: BaseModelParameters):
        super(BaseModelMaskPredictionLoss, self).__init__()

        self.num_read_features = num_read_features

        self.bce = torch.nn.BCEWithLogitsLoss(reduction='none')  # no reduction because we may want to first multiply by weights for unbalanced data

        # go from base model output representation to artifact logit for supervised loss
        self.mask_predictor = MLP([base_model_output_dim] + hidden_top_layers + [num_read_features], batch_normalize=params.batch_normalize, dropout_p=params.dropout_p)

    def loss_function(self, base_model: BaseModel, base_batch: BaseBatch):
        ref_count, alt_count = base_batch.ref_count, base_batch.alt_count
        total_ref, total_alt = ref_count * base_batch.size(), alt_count * base_batch.size()

        alt_reads_2d = base_batch.get_reads_2d()[total_ref:]
        permuted_reads = permute_columns_independently(base_batch.get_reads_2d())
        permuted_alt_reads = permuted_reads[:total_alt]

        datum_mask = torch.bernoulli(0.1 * torch.ones(base_batch.size(), self.num_read_features))

        # each read within a datum gets the same mask
        reads_mask = torch.repeat_interleave(datum_mask, repeats=alt_count, dim=0)

        original_reads_2d = base_batch.reads_2d
        modified_alt_reads = alt_reads_2d * (1 - reads_mask) + permuted_alt_reads * reads_mask
        base_batch.reads_2d = torch.vstack((original_reads_2d[:total_ref], modified_alt_reads))
        representations = base_model.calculate_representations(base_batch)

        # TODO: is there any reason to fix the batch with base_batch.reads_2d = original_reads_2d?

        # shape is batch size x num read features, each entry being a logit for "was this feature masked in this datum?"
        mask_prediction_logits = self.mask_predictor.forward(representations)

        # by batch index and feature
        losses_bf = self.bce(mask_prediction_logits, datum_mask)
        return torch.mean(losses_bf, dim=1)   # average over read features

    # I don't like implicit forward!!
    def forward(self):
        pass

# chamfer distance between two 3D tensors B x N1 x E and B x N2 x E, where B is the batch size, N1/2 are the number
# of items in the two sets, and E is the dimensionality of each item
# returns a 1D tensor of length B
def chamfer_distance(set1_bne, set2_bne):
    diffs_bnne = torch.unsqueeze(set1_bne, dim=2) - torch.unsqueeze(set2_bne, dim=1)
    l1_dists_bnn = torch.sum(torch.abs(diffs_bnne), dim=-1)

    chamfer_dists12_bn = torch.min(l1_dists_bnn, dim=-2).values
    chamfer_dists21_bn = torch.min(l1_dists_bnn, dim=-1).values
    symmetric_chamfer_b = torch.sum(chamfer_dists12_bn, dim=-1) + torch.sum(chamfer_dists21_bn, dim=-1)
    return symmetric_chamfer_b


# self-supervision approach where we use the base model embedding to regenerate the set and use Chamfer distance as the
# reconstruction error.  We regenerate the set via the Transformer Set Prediction Network approach of Kosiorek et al -- seed a set
# of N reads by concatenated the embedding with N random vectors, then map it so the final reconstructed set with transformers.
class BaseModelAutoencoderLoss(torch.nn.Module, BaseModelLearningStrategy):
    def __init__(self, read_dim: int, hidden_top_layers: List[int], params: BaseModelParameters):
        super(BaseModelAutoencoderLoss, self).__init__()
        self.base_model_output_dimension = params.output_dimension()

        # TODO: explore making random seed dimension different from the base model embedding dimension
        self.random_seed_dimension = self.base_model_output_dimension
        self.transformer_dimension = self.base_model_output_dimension + self.random_seed_dimension

        # TODO: should these decoder params be the same as the base model encoder params?  It seems reasonable.
        self.alt_transformer_decoder = make_transformer_encoder(self.transformer_dimension, params)
        self.ref_transformer_decoder = make_transformer_encoder(self.transformer_dimension, params)

        self.mapping_back_to_reads = MLP([self.transformer_dimension] + hidden_top_layers + [read_dim])


    def loss_function(self, base_model: BaseModel, base_batch: BaseBatch):
        var_count, alt_count, ref_count = base_batch.size(), base_batch.alt_count, base_batch.ref_count

        total_ref, total_alt = ref_count * var_count, alt_count * var_count

        representations_ve = base_model.calculate_representations(base_batch, self.weight_range)
        random_alt_seeds_vre = torch.randn(var_count, alt_count, self.random_seed_dimension)
        random_ref_seeds_vre = torch.randn(var_count, ref_count, self.random_seed_dimension) if ref_count > 0 else None
        representations_vre = representations_ve.expand_as(random_alt_seeds_vre) # repeat over the dummy read index

        alt_vre = torch.cat((representations_vre, random_alt_seeds_vre), dim=-1)
        ref_vre = torch.cat((representations_vre, random_ref_seeds_vre), dim=-1) if ref_count > 0 else None

        decoded_alt_vre = self.alt_transformer_decoder(alt_vre)
        decoded_ref_vre = self.ref_transformer_decoder(ref_vre) if ref_count > 0 else None

        decoded_alt_re = torch.reshape(decoded_alt_vre, (var_count*alt_count, -1))
        decoded_ref_re = torch.reshape(decoded_ref_vre, (var_count * alt_count, -1)) if ref_count > 0 else None

        reconstructed_alt_vre = torch.reshape(self.mapping_back_to_reads(decoded_alt_re),(var_count, alt_count, -1))
        reconstructed_ref_vre = torch.reshape(self.mapping_back_to_reads(decoded_ref_re), (var_count, ref_count, -1)) if ref_count > 0 else None

        original_alt_vre = base_batch.get_reads_2d()[total_ref:].reshape(var_count, alt_count, -1)
        original_ref_vre = base_batch.get_reads_2d()[:total_ref].reshape(var_count, ref_count, -1) if ref_count > - else None


        alt_chamfer_dist = chamfer_distance(original_alt_vre, reconstructed_alt_vre)
        ref_chamfer_dist = chamfer_distance(original_ref_vre, reconstructed_ref_vre) if ref_count > 0 else 0
        return alt_chamfer_dist + ref_chamfer_dist

    # I don't like implicit forward!!
    def forward(self):
        pass


def learn_base_model(base_model: BaseModel, dataset: BaseDataset, learning_method: LearningMethod, training_params: TrainingParameters,
            summary_writer: SummaryWriter, validation_fold: int = None):
    # balance training by weighting the loss function
    # if total unlabeled is less than total labeled, we do not compensate, since labeled data are more informative
    total_labeled, total_unlabeled = dataset.total_labeled_and_unlabeled()
    labeled_to_unlabeled_ratio = 1 if total_unlabeled < total_labeled else total_labeled / total_unlabeled

    print("Training data contains {} labeled examples and {} unlabeled examples".format(total_labeled,
                                                                                            total_unlabeled))
    for variation_type in utils.Variation:
        idx = variation_type.value
        print("For variation type {}, there are {} labeled artifact examples and {} labeled non-artifact examples"
            .format(variation_type.name, dataset.artifact_totals[idx].item(),
                    dataset.non_artifact_totals[idx].item()))

    # TODO: use Python's match syntax, but this requires updating Python version in the docker
    # TODO: hidden_top_layers are hard-coded!
    if learning_method == LearningMethod.SUPERVISED or learning_method == LearningMethod.SEMISUPERVISED:
        artifact_to_non_artifact_ratios = torch.from_numpy(dataset.artifact_to_non_artifact_ratios()).to(
                device=base_model._device, dtype=base_model._dtype)
        artifact_to_non_artifact_log_prior_ratios = torch.log(artifact_to_non_artifact_ratios)

        learning_strategy = BaseModelSemiSupervisedLoss(input_dim=base_model.output_dimension(), hidden_top_layers=[10,10,10],
                                                            params=base_model._params, artifact_to_non_artifact_log_prior_ratios=artifact_to_non_artifact_log_prior_ratios,
                                                            labeled_to_unlabeled_ratio=labeled_to_unlabeled_ratio)
    elif learning_method == LearningMethod.MASK_PREDICTION:
        learning_strategy = BaseModelMaskPredictionLoss(num_read_features=dataset.num_read_features,
                                                        base_model_output_dim=base_model.output_dimension(), hidden_top_layers=[10,10,10], params=base_model._params)
    elif learning_method == LearningMethod.AUTOENCODER:
        learning_strategy = BaseModelAutoencoderLoss(read_dim=dataset.num_read_features, hidden_top_layers=[20,20,20], params=base_model._params)
    else:
        raise Exception("not implemented yet")

    train_optimizer = torch.optim.AdamW(chain(base_model.parameters(), learning_strategy.parameters()),
                                        lr=training_params.learning_rate, weight_decay=training_params.weight_decay)

    validation_fold_to_use = (dataset.num_folds - 1) if validation_fold is None else validation_fold
    train_loader = dataset.make_data_loader(dataset.all_but_one_fold(validation_fold_to_use), training_params.batch_size, base_model._device.type == 'cuda', training_params.num_workers)
    valid_loader = dataset.make_data_loader([validation_fold_to_use], training_params.batch_size, base_model._device.type == 'cuda', training_params.num_workers)

    for epoch in trange(1, training_params.num_epochs + 1, desc="Epoch"):
        for epoch_type in (utils.Epoch.TRAIN, utils.Epoch.VALID):
            base_model.set_epoch_type(epoch_type)

            loss_metrics = LossMetrics(base_model._device)

            loader = train_loader if epoch_type == utils.Epoch.TRAIN else valid_loader
            pbar = tqdm(enumerate(loader), mininterval=60)
            for n, batch in pbar:
                separate_losses = learning_strategy.loss_function(base_model, batch)
                loss = torch.sum(separate_losses)
                loss_metrics.record_total_batch_loss(loss.detach(), batch)

                if batch.is_labeled():
                    loss_metrics.record_losses_by_type_and_count(separate_losses, batch)

                if epoch_type == utils.Epoch.TRAIN:
                    utils.backpropagate(train_optimizer, loss)

            # done with one epoch type -- training or validation -- for this epoch
            loss_metrics.write_to_summary_writer(epoch_type, epoch, summary_writer)

            print("Labeled loss for epoch " + str(epoch) + " of " + epoch_type.name + ": " + str(loss_metrics.get_labeled_loss()))
        # done with training and validation for this epoch
        # note that we have not learned the AF spectrum yet
    # done with training


