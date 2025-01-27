import math
from abc import ABC, abstractmethod
from enum import Enum
from itertools import chain
import time
from typing import List

import psutil
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parameter import Parameter
from tqdm.autonotebook import trange, tqdm

from permutect import utils, constants
from permutect.architecture.dna_sequence_convolution import DNASequenceConvolution
from permutect.architecture.gated_mlp import GatedMLP, GatedRefAltMLP
from permutect.architecture.gradient_reversal.module import GradientReversal
from permutect.architecture.mlp import MLP
from permutect.architecture.set_pooling import SetPooling
from permutect.data.base_datum import BaseBatch, DEFAULT_GPU_FLOAT, DEFAULT_CPU_FLOAT
from permutect.data.base_dataset import BaseDataset, ALL_COUNTS_INDEX
from permutect.metrics.evaluation_metrics import LossMetrics, EmbeddingMetrics, round_up_to_nearest_three, MAX_COUNT
from permutect.parameters import BaseModelParameters, TrainingParameters


# group rows into consecutive chunks to yield a 3D tensor, average over dim=1 to get
# 2D tensor of sums within each chunk
from permutect.sets.ragged_sets import RaggedSets
from permutect.utils import Variation, Label


def sums_over_chunks(tensor2d: torch.Tensor, chunk_size: int):
    assert len(tensor2d) % chunk_size == 0
    return torch.sum(tensor2d.reshape([len(tensor2d) // chunk_size, chunk_size, -1]), dim=1)


# note: this works for both BaseBatch/BaseDataset AND ArtifactBatch/ArtifactDataset
# if by_count is True, each count is weighted separately for balanced loss within that count
def calculate_batch_weights(batch, dataset, by_count: bool):
    # TODO: we need a parameter to control the relative weight of unlabeled loss to labeled loss
    # For batch index n, we want weight[n] = dataset.weights[alt_counts[n], labels[n], variant_types[n]]
    sources = batch.get_sources()
    counts = batch.get_alt_counts() if by_count else torch.full(size=(len(sources), ), fill_value=ALL_COUNTS_INDEX, dtype=torch.int)
    labels = batch.get_labels()
    variant_types = batch.get_variant_types()

    return utils.index_4d_array(dataset.label_balancing_weights_sclt, sources, counts, labels, variant_types)


# note: this works for both BaseBatch/BaseDataset AND ArtifactBatch/ArtifactDataset
# if by_count is True, each count is weighted separately for balanced loss within that count
def calculate_batch_source_weights(batch, dataset, by_count: bool):
    # For batch index n, we want weight[n] = dataset.source_weights[sources[n], alt_counts[n], variant_types[n]]
    sources = batch.get_sources()
    counts = batch.get_alt_counts() if by_count else torch.full(size=(len(sources), ), fill_value=ALL_COUNTS_INDEX, dtype=torch.int)
    variant_types = batch.get_variant_types()

    return utils.index_3d_array(dataset.source_balancing_weights_sct, sources, counts, variant_types)


def make_gated_ref_alt_mlp_encoder(input_dimension: int, params: BaseModelParameters) -> GatedRefAltMLP:
    return GatedRefAltMLP(d_model=input_dimension, d_ffn=params.self_attention_hidden_dimension, num_blocks=params.num_self_attention_layers)


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

        # embeddings of reads, info, and reference sequence prior to the transformer layers
        self.read_embedding = MLP([num_read_features] + params.read_layers, batch_normalize=params.batch_normalize, dropout_p=params.dropout_p)
        self.info_embedding = MLP([num_info_features] + params.info_layers, batch_normalize=params.batch_normalize, dropout_p=params.dropout_p)
        self.ref_seq_cnn = DNASequenceConvolution(params.ref_seq_layer_strings, ref_sequence_length)

        embedding_dim = self.read_embedding.output_dimension() + self.info_embedding.output_dimension() + self.ref_seq_cnn.output_dimension()

        self.ref_alt_reads_encoder = make_gated_ref_alt_mlp_encoder(embedding_dim, params)

        # after encoding alt reads (along with info and ref seq embeddings and with self-attention to ref reads)
        # aggregate encoded sets in a permutation-invariant way
        # TODO: hard-coded magic constant!!!!!
        aggregation_hidden_layers = [-2, -2]
        self.aggregation = SetPooling(input_dim=embedding_dim, mlp_layers=aggregation_hidden_layers,
            final_mlp_layers=params.aggregation_layers, batch_normalize=params.batch_normalize, dropout_p=params.dropout_p)

        self.to(device=self._device, dtype=self._dtype)

    def output_dimension(self) -> int:
        return self.aggregation.output_dimension()

    def ref_alt_seq_embedding_dimension(self) -> int:
        return self.ref_seq_cnn.output_dimension()

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
        ref_counts, alt_counts = batch.get_ref_counts(), batch.get_alt_counts()
        total_ref, total_alt = torch.sum(ref_counts).item(), torch.sum(alt_counts).item()

        read_embeddings_re = self.read_embedding.forward(batch.get_reads_2d().to(dtype=self._dtype))
        info_embeddings_ve = self.info_embedding.forward(batch.get_info_2d().to(dtype=self._dtype))
        ref_seq_embeddings_ve = self.ref_seq_cnn(batch.get_ref_sequences_2d().to(dtype=self._dtype))
        info_and_seq_ve = torch.hstack((info_embeddings_ve, ref_seq_embeddings_ve))
        info_and_seq_re = torch.vstack((torch.repeat_interleave(info_and_seq_ve, repeats=ref_counts, dim=0),
                                       torch.repeat_interleave(info_and_seq_ve, repeats=alt_counts, dim=0)))
        reads_info_seq_re = torch.hstack((read_embeddings_re, info_and_seq_re))

        # TODO: might be a bug if every datum in batch has zero ref reads?
        ref_bre = RaggedSets.from_flattened_tensor_and_sizes(reads_info_seq_re[:total_ref], ref_counts)
        alt_bre = RaggedSets.from_flattened_tensor_and_sizes(reads_info_seq_re[total_ref:], alt_counts)
        _, transformed_alt_bre = self.ref_alt_reads_encoder.forward(ref_bre, alt_bre)

        # TODO: this old code has the random weighting logic which might still be valuable
        """
        transformed_alt_re = transformed_alt_bre.flattened_tensor_nf

        alt_weights_r = 1 + weight_range * (1 - 2 * torch.rand(total_alt, device=self._device, dtype=self._dtype))

        # normalize so read weights within each variant sum to 1
        alt_wt_sums_v = utils.sums_over_rows(alt_weights_r, alt_counts)
        normalized_alt_weights_r = alt_weights_r / torch.repeat_interleave(alt_wt_sums_v, repeats=alt_counts, dim=0)

        alt_means_ve = utils.sums_over_rows(transformed_alt_re * normalized_alt_weights_r[:,None], alt_counts)
        """
        result_be = self.aggregation.forward(transformed_alt_bre)

        return result_be, ref_seq_embeddings_ve # ref seq embeddings are useful later

    def make_dict_for_saving(self, prefix: str = ""):
        return {(prefix + constants.STATE_DICT_NAME): self.state_dict(),
                (prefix + constants.HYPERPARAMS_NAME): self._params,
                (prefix + constants.NUM_READ_FEATURES_NAME): self.read_embedding.input_dimension(),
                (prefix + constants.NUM_INFO_FEATURES_NAME): self.info_embedding.input_dimension(),
                (prefix + constants.REF_SEQUENCE_LENGTH_NAME): self.ref_sequence_length()}

    def save(self, path):
        torch.save(self.make_dict_for_saving(), path)


def base_model_from_saved_dict(saved, prefix: str = "", device: torch.device = utils.gpu_if_available()):
    hyperparams = saved[prefix + constants.HYPERPARAMS_NAME]
    num_read_features = saved[prefix + constants.NUM_READ_FEATURES_NAME]
    num_info_features = saved[prefix + constants.NUM_INFO_FEATURES_NAME]
    ref_sequence_length = saved[prefix + constants.REF_SEQUENCE_LENGTH_NAME]

    model = BaseModel(hyperparams, num_read_features=num_read_features, num_info_features=num_info_features,
                      ref_sequence_length=ref_sequence_length, device=device)
    model.load_state_dict(saved[prefix + constants.STATE_DICT_NAME])

    # in case the state dict had the wrong dtype for the device we're on now eg base model was pretrained on GPU
    # and we're now on CPU
    model.to(model._dtype)

    return model


def load_base_model(path, prefix: str = "", device: torch.device = utils.gpu_if_available()) -> BaseModel:
    saved = torch.load(path, map_location=device)
    return base_model_from_saved_dict(saved, prefix, device)


# outputs a 1D tensor of losses over the batch.  We assume it needs the representations of the batch data from the base
# model.  We nonetheless also use the model as an input because there are some learning strategies that involve
# computing representations of a modified batch.
class BaseModelLearningStrategy(ABC):
    @abstractmethod
    def loss_function(self, base_model: BaseModel, base_batch: BaseBatch, base_model_representations: torch.Tensor):
        pass


class BaseModelSemiSupervisedLoss(torch.nn.Module, BaseModelLearningStrategy):
    def __init__(self, input_dim: int, hidden_top_layers: List[int], params: BaseModelParameters):
        super(BaseModelSemiSupervisedLoss, self).__init__()

        self.bce = torch.nn.BCEWithLogitsLoss(reduction='none')  # no reduction because we may want to first multiply by weights for unbalanced data

        # go from base model output representation to artifact logit for supervised loss
        self.logit_predictor = MLP([input_dim] + hidden_top_layers + [1], batch_normalize=params.batch_normalize, dropout_p=params.dropout_p)

    def loss_function(self, base_model: BaseModel, base_batch: BaseBatch, base_model_representations: torch.Tensor):
        logits = self.logit_predictor.forward(base_model_representations).reshape((base_batch.size()))
        labels = base_batch.get_training_labels()

        # base batch always has labels, but for unlabeled elements these labels are meaningless and is_labeled_mask is zero
        cross_entropies = self.bce(logits, labels)
        probabilities = torch.sigmoid(logits)
        entropies = self.bce(logits, probabilities)

        return base_batch.get_is_labeled_mask() * cross_entropies + (1 - base_batch.get_is_labeled_mask()) * entropies

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


# artifact model parameters are for simultaneously training an artifact model on top of the base model
# to measure quality, especially in unsupervised training when the loss metric isn't directly related to accuracy or cross-entropy
def learn_base_model(base_model: BaseModel, dataset: BaseDataset, training_params: TrainingParameters,
                     summary_writer: SummaryWriter, validation_fold: int = None):
    print(f"Memory usage percent: {psutil.virtual_memory().percent:.1f}")
    is_cuda = base_model._device.type == 'cuda'
    print(f"Is CUDA available? {is_cuda}")

    for source in range(dataset.max_source + 1):
        print(f"Data counts for source {source}:")
        for var_type in utils.Variation:
            print(f"Data counts for variant type {var_type.name}:")
            for label in Label:
                print(f"{label.name}: {int(dataset.totals_sclt[source][ALL_COUNTS_INDEX][label][var_type].item())}")

    # TODO: hidden_top_layers are hard-coded!
    learning_strategy = BaseModelSemiSupervisedLoss(input_dim=base_model.output_dimension(), hidden_top_layers=[30,-1,-1,-1,10], params=base_model._params)

    learning_strategy.to(device=base_model._device, dtype=base_model._dtype)

    # adversarial loss to learn features that forget the alt count
    alt_count_gradient_reversal = GradientReversal(alpha=0.01)  #initialize as barely active
    alt_count_predictor = MLP([base_model.output_dimension()] + [30, -1, -1, -1, 1]).to(device=base_model._device, dtype=base_model._dtype)
    alt_count_loss_func = torch.nn.MSELoss(reduction='none')
    alt_count_adversarial_metrics = LossMetrics()

    # TODO: fused = is_cuda?
    train_optimizer = torch.optim.AdamW(chain(base_model.parameters(), learning_strategy.parameters(), alt_count_predictor.parameters()),
                                        lr=training_params.learning_rate, weight_decay=training_params.weight_decay)
    # train scheduler needs to be given the thing that's supposed to decrease at the end of each epoch
    train_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        train_optimizer, factor=0.2, patience=5, threshold=0.001, min_lr=(training_params.learning_rate/100), verbose=True)

    classifier_on_top = MLP([base_model.output_dimension()] + [30, -1, -1, -1, 10] + [1])\
        .to(device=base_model._device, dtype=base_model._dtype)
    classifier_bce = torch.nn.BCEWithLogitsLoss(reduction='none')

    classifier_optimizer = torch.optim.AdamW(classifier_on_top.parameters(),
                                             lr=training_params.learning_rate,
                                             weight_decay=training_params.weight_decay,
                                             fused=is_cuda)
    classifier_metrics = LossMetrics()

    validation_fold_to_use = (dataset.num_folds - 1) if validation_fold is None else validation_fold
    train_loader = dataset.make_data_loader(dataset.all_but_one_fold(validation_fold_to_use), training_params.batch_size, is_cuda, training_params.num_workers)
    valid_loader = dataset.make_data_loader([validation_fold_to_use], training_params.batch_size, is_cuda, training_params.num_workers)

    for epoch in trange(1, training_params.num_epochs + 1, desc="Epoch"):
        p = epoch - 1
        new_alpha = (2/(1 + math.exp(-0.1*p))) - 1
        alt_count_gradient_reversal.set_alpha(new_alpha) # alpha increases linearly
        start_epoch = time.time()
        print(f"Start of epoch {epoch}, memory usage percent: {psutil.virtual_memory().percent:.1f}")
        for epoch_type in (utils.Epoch.TRAIN, utils.Epoch.VALID):
            base_model.set_epoch_type(epoch_type)
            loss_metrics = LossMetrics()

            loader = train_loader if epoch_type == utils.Epoch.TRAIN else valid_loader
            loader_iter = iter(loader)

            next_batch_cpu = next(loader_iter)
            next_batch = next_batch_cpu.copy_to(base_model._device, non_blocking=is_cuda)

            pbar = tqdm(range(len(loader)), mininterval=60)
            for n in pbar:
                batch_cpu = next_batch_cpu
                batch = next_batch

                # Optimization: Asynchronously send the next batch to the device while the model does work
                next_batch_cpu = next(loader_iter)
                next_batch = next_batch_cpu.copy_to(base_model._device, non_blocking=is_cuda)

                # TODO: we need a parameter to control the relative weight of unlabeled loss to labeled loss
                weights = calculate_batch_weights(batch_cpu, dataset, by_count=True)
                weights = weights.to(device=base_model._device, dtype=base_model._dtype, non_blocking=True)

                # unused output is the embedding of ref and alt alleles with context
                representations, _ = base_model.calculate_representations(batch, weight_range=base_model._params.reweighting_range)
                losses = learning_strategy.loss_function(base_model, batch, representations)

                if losses is None:
                    continue

                loss_metrics.record_losses(losses.detach(), batch, weights)

                # gradient reversal means parameters before the representation try to maximize alt count prediction loss, i.e. features
                # try to forget alt count, while parameters after the representation try to minimize it, i.e. they try
                # to achieve the adversarial task
                alt_count_pred = torch.sigmoid(alt_count_predictor.forward(alt_count_gradient_reversal(representations)).squeeze())
                alt_count_target = batch.get_alt_counts().to(dtype=alt_count_pred.dtype)/20
                alt_count_losses = alt_count_loss_func(alt_count_pred, alt_count_target)

                alt_count_adversarial_metrics.record_losses(alt_count_losses.detach(), batch, weights=torch.ones_like(alt_count_losses))

                loss = torch.sum((weights * losses) + alt_count_losses)

                classification_logits = classifier_on_top.forward(representations.detach()).reshape(batch.size())
                classification_losses = classifier_bce(classification_logits, batch.get_training_labels())
                classification_loss = torch.sum(batch.get_is_labeled_mask() * weights * classification_losses)
                classifier_metrics.record_losses(classification_losses.detach(), batch, batch.get_is_labeled_mask() * weights)

                if epoch_type == utils.Epoch.TRAIN:
                    utils.backpropagate(train_optimizer, loss)
                    utils.backpropagate(classifier_optimizer, classification_loss)

            # done with one epoch type -- training or validation -- for this epoch
            loss_metrics.write_to_summary_writer(epoch_type, epoch, summary_writer)
            classifier_metrics.write_to_summary_writer(epoch_type, epoch, summary_writer, prefix="auxiliary-classifier-")
            alt_count_adversarial_metrics.write_to_summary_writer(epoch_type, epoch, summary_writer, prefix="alt-count-adversarial-predictor")

            if epoch_type == utils.Epoch.TRAIN:
                train_scheduler.step(loss_metrics.get_labeled_loss())

            print(f"Labeled base model loss for {epoch_type.name} epoch {epoch}: {loss_metrics.get_labeled_loss():.3f}")
            print(f"Labeled auxiliary classifier loss for {epoch_type.name} epoch {epoch}: {classifier_metrics.get_labeled_loss():.3f}")
            print(f"Alt count adversarial loss for {epoch_type.name} epoch {epoch}: {alt_count_adversarial_metrics.get_labeled_loss():.3f}")
        print(f"End of epoch {epoch}, memory usage percent: {psutil.virtual_memory().percent:.1f}, time elapsed(s): {time.time() - start_epoch:.2f}")
        # done with training and validation for this epoch
        # note that we have not learned the AF spectrum yet
    # done with training

    record_embeddings(base_model, train_loader, summary_writer)


# after training for visualizing clustering etc of base model embeddings
def record_embeddings(base_model: BaseModel, loader, summary_writer: SummaryWriter):
    # base_model.freeze_all() whoops -- it doesn't have freeze_all
    embedding_metrics = EmbeddingMetrics()
    ref_alt_seq_metrics = EmbeddingMetrics()

    pbar = tqdm(enumerate(loader), mininterval=60)
    for n, batch_cpu in pbar:
        batch = batch_cpu.copy_to(base_model._device, non_blocking=base_model._device.type=='cuda')
        representations, ref_alt_seq_embeddings = base_model.calculate_representations(batch, weight_range=base_model._params.reweighting_range)

        representations = representations.cpu()
        ref_alt_seq_embeddings = ref_alt_seq_embeddings.cpu()

        labels = [("artifact" if label > 0.5 else "non-artifact") if is_labeled > 0.5 else "unlabeled" for (label, is_labeled) in
                  zip(batch.get_training_labels().tolist(), batch.get_is_labeled_mask().tolist())]
        for (metrics, embeddings) in [(embedding_metrics, representations), (ref_alt_seq_metrics, ref_alt_seq_embeddings)]:
            metrics.label_metadata.extend(labels)
            metrics.correct_metadata.extend(["unknown"] * batch.size())
            metrics.type_metadata.extend([Variation(idx).name for idx in batch.get_variant_types().tolist()])
            alt_count_strings = [str(round_up_to_nearest_three(min(MAX_COUNT, ac))) for ac in batch.get_alt_counts().tolist()]
            metrics.truncated_count_metadata.extend(alt_count_strings)
            metrics.representations.append(embeddings)
    embedding_metrics.output_to_summary_writer(summary_writer)
    ref_alt_seq_metrics.output_to_summary_writer(summary_writer, prefix="ref and alt allele context")

