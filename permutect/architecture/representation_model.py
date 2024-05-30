from enum import Enum
from itertools import chain

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm.autonotebook import trange, tqdm

from permutect import utils, constants
from permutect.architecture.dna_sequence_convolution import DNASequenceConvolution
from permutect.architecture.mlp import MLP
from permutect.data.read_set import ReadSetBatch
from permutect.data.read_set_dataset import ReadSetDataset
from permutect.metrics.evaluation_metrics import LossMetrics
from permutect.parameters import RepresentationModelParameters, TrainingParameters
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


class RepresentationModel(torch.nn.Module):
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

    def __init__(self, params: RepresentationModelParameters, num_read_features: int, num_info_features: int, ref_sequence_length: int, device=torch.device("cpu")):
        super(RepresentationModel, self).__init__()

        self._device = device
        self._num_read_features = num_read_features
        self._num_info_features = num_info_features
        self._ref_sequence_length = ref_sequence_length
        self._params = params
        self.alt_downsample = params.alt_downsample

        # linear transformation to convert read tensors from their initial dimensionality
        # to the embedding dimension eg data of shape [num_batches -- optional] x num_reads x read dimension
        # maps to data of shape [num_batches] x num_reads x embedding dimension)
        self.initial_read_embedding = torch.nn.Linear(in_features=num_read_features, out_features=params.read_embedding_dimension)
        self.initial_read_embedding.to(self._device)

        self.read_embedding_dimension = params.read_embedding_dimension
        self.num_transformer_heads = params.num_transformer_heads
        self.transformer_hidden_dimension = params.transformer_hidden_dimension
        self.num_transformer_layers = params.num_transformer_layers

        alt_transformer_encoder_layer = torch.nn.TransformerEncoderLayer(d_model=params.read_embedding_dimension,
            nhead=params.num_transformer_heads, batch_first=True, dim_feedforward=params.transformer_hidden_dimension, dropout=params.dropout_p)
        alt_encoder_norm = torch.nn.LayerNorm(params.read_embedding_dimension)
        self.alt_transformer_encoder = torch.nn.TransformerEncoder(alt_transformer_encoder_layer, num_layers=params.num_transformer_layers, norm=alt_encoder_norm)
        self.alt_transformer_encoder.to(self._device)

        ref_transformer_encoder_layer = torch.nn.TransformerEncoderLayer(d_model=params.read_embedding_dimension,
             nhead=params.num_transformer_heads, batch_first=True, dim_feedforward=params.transformer_hidden_dimension, dropout=params.dropout_p)
        ref_encoder_norm = torch.nn.LayerNorm(params.read_embedding_dimension)
        self.ref_transformer_encoder = torch.nn.TransformerEncoder(ref_transformer_encoder_layer, num_layers=params.num_transformer_layers, norm=ref_encoder_norm)
        self.ref_transformer_encoder.to(self._device)

        # omega is the embedding of info field variant-level data
        info_layers = [self._num_info_features] + params.info_layers
        self.omega = MLP(info_layers, batch_normalize=params.batch_normalize, dropout_p=params.dropout_p)
        self.omega.to(self._device)

        self.ref_seq_cnn = DNASequenceConvolution(params.ref_seq_layer_strings, ref_sequence_length)
        self.ref_seq_cnn.to(self._device)

        # rho is the aggregation function that combines read, info, and sequence tensors and maps to the output representation
        ref_alt_info_ref_seq_embedding_dimension = 2 * self.read_embedding_dimension + self.omega.output_dimension() + self.ref_seq_cnn.output_dimension()

        # we have a different aggregation subnetwork for each variant type.  Everything below, in particular the read
        # transformers, is shared
        # The [1] is for the output of binary classification, represented as a single artifact/non-artifact logit
        self._output_dimension = params.aggregation_layers[-1]
        self.rho = torch.nn.ModuleList([MLP([ref_alt_info_ref_seq_embedding_dimension] + params.aggregation_layers, batch_normalize=params.batch_normalize,
                       dropout_p=params.dropout_p) for variant_type in Variation])
        self.rho.to(self._device)

    def num_read_features(self) -> int:
        return self._num_read_features

    def num_info_features(self) -> int:
        return self._num_info_features

    def output_dimension(self) -> int:
        return self._output_dimension

    def ref_sequence_length(self) -> int:
        return self._ref_sequence_length

    def set_epoch_type(self, epoch_type: utils.Epoch):
        if epoch_type == utils.Epoch.TRAIN:
            self.train(True)
            utils.unfreeze(self.parameters())
        else:
            self.train(False)
            utils.freeze(self.parameters())

    # return 2D tensor of shape (batch size x output dimension)
    def calculate_representations(self, batch: ReadSetBatch, weight_range: float = 0) -> torch.Tensor:
        transformed_reads = self.apply_transformer_to_reads(batch)
        return self.representations_from_transformed_reads(transformed_reads, batch, weight_range)

    # I really don't like the forward method of torch.nn.Module with its implicit calling that PyCharm doesn't recognize
    def forward(self, batch: ReadSetBatch):
        pass

    def apply_transformer_to_reads(self, batch: ReadSetBatch) -> torch.Tensor:
        initial_embedded_reads = self.initial_read_embedding(batch.get_reads_2d().to(self._device))

        # rearrange the 2D tensor where each row is a read  into two 3D tensors of shape
        # (batch_size x batch alt/ref count x read embedding dimension), so that teh transformer doesn't mix
        # between read sets or mix ref and alt.
        ref_count, alt_count = batch.ref_count, batch.alt_count
        total_ref, total_alt = ref_count * batch.size(), alt_count * batch.size()
        ref_reads_3d = None if total_ref == 0 else initial_embedded_reads[:total_ref].reshape(batch.size(), ref_count, self.read_embedding_dimension)
        alt_reads_3d = initial_embedded_reads[total_ref:].reshape(batch.size(), alt_count, self.read_embedding_dimension)

        if self.alt_downsample < alt_count:
            alt_read_indices = torch.randperm(alt_count)[:self.alt_downsample]
            alt_reads_3d = alt_reads_3d[:, alt_read_indices, :]   # downsample only along the middle (read) dimension
            total_alt = batch.size() * self.alt_downsample

        # undo some of the above rearrangement
        transformed_alt_reads_2d = self.alt_transformer_encoder(alt_reads_3d).reshape(total_alt, self.read_embedding_dimension)
        transformed_ref_reads_2d = None if total_ref == 0 else self.ref_transformer_encoder(ref_reads_3d).reshape(total_ref, self.read_embedding_dimension)

        transformed_reads_2d = transformed_alt_reads_2d if total_ref == 0 else \
            torch.vstack([transformed_ref_reads_2d, transformed_alt_reads_2d])

        return transformed_reads_2d

    # input: reads that have passed through the alt/ref transformers
    # output: reads that have been refined through the rho subnetwork that sees the transformed alts along with ref, alt, and info
    def representations_from_transformed_reads(self, transformed_reads: torch.Tensor, batch: ReadSetBatch, weight_range: float = 0):
        weights = torch.ones(len(transformed_reads), 1, device=self._device) if weight_range == 0 else (1 + weight_range * (1 - 2 * torch.rand(len(transformed_reads), 1, device=self._device)))

        ref_count, alt_count = batch.ref_count, min(batch.alt_count, self.alt_downsample)
        total_ref, total_alt = ref_count * batch.size(), alt_count * batch.size()

        # mean embedding of every read, alt and ref, at each datum
        all_read_means = ((0 if ref_count == 0 else sums_over_chunks(transformed_reads[:total_ref], ref_count)) + sums_over_chunks(transformed_reads[total_ref:], alt_count)) / (alt_count + ref_count)
        omega_info = self.omega(batch.get_info_2d().to(self._device))
        ref_seq_embedding = self.ref_seq_cnn(batch.get_ref_sequences_2d())

        # take the all-read-average / info embedding / ref seq embedding of each read set in the batch and
        # concatenate a copy next to each transformed alt in the set
        extra_tensor_2d = torch.hstack([all_read_means, omega_info, ref_seq_embedding])     # shape is (batch size, ___)
        extra_tensor_2d = torch.repeat_interleave(extra_tensor_2d, repeats=alt_count, dim=0)    # shape is (batch size * alt count, ___)

        # the alt reads have not been averaged yet to we need to copy each row of the extra tensor batch.alt_count times
        padded_transformed_alt_reads = torch.hstack([transformed_reads[total_ref:], extra_tensor_2d])

        # now we refine the alt reads -- no need to separate between read sets yet as this broadcasts over every read
        refined_alt = torch.zeros(total_alt, self._output_dimension)
        one_hot_types_2d = batch.variant_type_one_hot()
        for n, _ in enumerate(Variation):
            # multiply the result of this variant type's aggregation layers by its
            # one-hot mask.  Yes, this is wasteful because we apply every aggregation to every datum.
            # TODO: make this more efficient.
            mask = torch.repeat_interleave(one_hot_types_2d[:, n], repeats=alt_count).reshape(total_alt, 1)  # 2D column tensor
            refined_alt += mask * self.rho[n].forward(padded_transformed_alt_reads)

        # weighted mean is sum of reads in a chunk divided by sum of weights in same chunk
        alt_wts = weights[total_ref:]
        alt_wt_sums = sums_over_chunks(alt_wts, alt_count)
        return sums_over_chunks(alt_wts * refined_alt, alt_count) / alt_wt_sums     # shape is (batch size, output dimension)

    # TODO: we need to attach an MLP on top of the embedding model to output logits, then train this combination
    # TODO: actually, this can be the framework of a LOT of different ways to train.  There's a ton of overlap.  There's always going to
    # TODO: be running the model over epochs, loading the dataset, backpropagating the loss.
    # TODO: the differences will mainly be in auxiliary tasks attached to the embedding and different loss functions
    def learn(self, dataset: ReadSetDataset, learning_method: LearningMethod, training_params: TrainingParameters,
              summary_writer: SummaryWriter, validation_fold: int = None):
        bce = torch.nn.BCEWithLogitsLoss(reduction='none')  # no reduction because we may want to first multiply by weights for unbalanced data

        match learning_method:
            case LearningMethod.SUPERVISED | LearningMethod.SEMISUPERVISED:
                # top layer maps the embedding to a logit for binary classification
                # TODO: make this an MLP??
                top_layer = torch.nn.Linear(in_features=self._output_dimension, out_features=1)
            case _:
                raise Exception("not implemented yet")

        train_optimizer = torch.optim.AdamW(chain(self.parameters(), top_layer.parameters()),
                                            lr=training_params.learning_rate, weight_decay=training_params.weight_decay)

        artifact_to_non_artifact_ratios = torch.from_numpy(dataset.artifact_to_non_artifact_ratios()).to(self._device)
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
                    embeddings = self.calculate_representations(batch, training_params.reweighting_range)

                    # TODO: this is only handling the supervised/semi-supervised cases
                    logits = torch.squeeze(top_layer(embeddings), dim=1)

                    types_one_hot = batch.variant_type_one_hot()
                    log_prior_ratios = torch.sum(artifact_to_non_artifact_log_prior_ratios * types_one_hot, dim=1)
                    posterior_logits = logits + log_prior_ratios

                    if batch.is_labeled():
                        separate_losses = bce(posterior_logits, batch.labels)
                        loss = torch.sum(separate_losses)

                        loss_metrics.record_total_batch_loss(loss.detach(), batch)
                        loss_metrics.record_separate_losses(separate_losses, batch)
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
            constants.NUM_READ_FEATURES_NAME: self.num_read_features(),
            constants.NUM_INFO_FEATURES_NAME: self.num_info_features(),
            constants.REF_SEQUENCE_LENGTH_NAME: self.ref_sequence_length()
        }, path)


def load_representation_model(path) -> RepresentationModel:
    saved = torch.load(path)
    hyperparams = saved[constants.HYPERPARAMS_NAME]
    num_read_features = saved[constants.NUM_READ_FEATURES_NAME]
    num_info_features = saved[constants.NUM_INFO_FEATURES_NAME]
    ref_sequence_length = saved[constants.REF_SEQUENCE_LENGTH_NAME]

    model = RepresentationModel(hyperparams, num_read_features=num_read_features, num_info_features=num_info_features,
                                ref_sequence_length=ref_sequence_length)
    model.load_state_dict(saved[constants.STATE_DICT_NAME])

    return model
