# bug before PyTorch 1.7.1 that warns when constructing ParameterList
import warnings
from collections import defaultdict
import math
from typing import List

import torch
from torch import nn, Tensor
import numpy as np
from torch.utils.tensorboard import SummaryWriter


from tqdm.autonotebook import trange, tqdm
from itertools import chain
from matplotlib import pyplot as plt

from mutect3.architecture.mlp import MLP
from mutect3.architecture.dna_sequence_convolution import DNASequenceConvolution
from mutect3.data.read_set import ReadSetBatch
from mutect3.data.read_set_dataset import ReadSetDataset, make_data_loader
from mutect3 import utils
from mutect3.utils import Call, Variation
from mutect3.metrics import plotting

warnings.filterwarnings("ignore", message="Setting attributes on ParameterList is not supported.")

NUM_DATA_FOR_TENSORBOARD_PROJECTION = 10000


def round_up_to_nearest_three(x: int):
    return math.ceil(x / 3) * 3


def multiple_of_three_bin_index(x: int):
    return (round_up_to_nearest_three(x)//3) - 1    # -1 because zero is not a bin


def multiple_of_three_bin_index_to_count(idx: int):
    return 3 * (idx + 1)


MAX_COUNT = 18  # counts above this will be truncated
MAX_LOGIT = 6
NUM_COUNT_BINS = round_up_to_nearest_three(MAX_COUNT) // 3    # zero is not a bin


def effective_count(weights: Tensor):
    return (torch.square(torch.sum(weights)) / torch.sum(torch.square(weights))).item()


# group rows into consecutive chunks to yield a 3D tensor, average over dim=1 to get
# 2D tensor of sums within each chunk
def sums_over_chunks(tensor2d: Tensor, chunk_size: int):
    assert len(tensor2d) % chunk_size == 0
    return torch.sum(tensor2d.reshape([len(tensor2d) // chunk_size, chunk_size, -1]), dim=1)


# note that read layers and info layers exclude the input dimension
# read_embedding_dimension: read tensors are linear-transformed to this dimension before
#    input to the transformer.  This is also the output dimension of reads from the transformer
# num_transformer_heads: number of attention heads in the read transformer.  Must be a divisor
#    of the read_embedding_dimension
# num_transformer_layers: number of layers of read transformer
class ArtifactModelParameters:
    def __init__(self,
                 read_embedding_dimension, num_transformer_heads, transformer_hidden_dimension,
                 num_transformer_layers, info_layers, aggregation_layers,
                 ref_seq_layers_strings, dropout_p, batch_normalize, learning_rate,
                 alt_downsample):

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
        self.learning_rate = learning_rate
        self.alt_downsample = alt_downsample


class Calibration(nn.Module):

    def __init__(self):
        super(Calibration, self).__init__()

        # temperature is of form
        # a_11 f_1(alt) * f_1(ref) + a_12 f_1(alt)*f_2(ref) . . . + a_NN f_NN(alt) f_NN(ref)
        # where coefficients are all positive and
        # f_1(x) = 1
        # f_2(x) = x^(1/4)
        # f_3(x) = x^(1/3)
        # f_4(x) = x^(1/2)

        # now, remember that alt and ref counts are constant within a batch, so that means temperature is constant within a batch!
        # so we don't need to worry about vectorizing our operations or otherwise making them efficient
        # In matrix form this is, for a batch, with A the matrix of above coefficients

        self.coeffs = nn.Parameter(Tensor([[1.0, 0.001, 0.001, 0.001], [0.001, 0.001, 0.001, 0.001], [0.001, 0.001, 0.001, 0.001], [0.001, 0.001, 0.001, 0.001]]))

        # we apply as asymptotic threshold function logit --> M * tanh(logit/M) where M is the maximum absolute
        # value of the thresholded output.  For logits << M this is the identity, and approaching M the asymptote
        # gradually turns on.  This is a continuous way to truncate the model's confidence and is part of calibration.
        # We initialize it to something large.
        self.max_logit = nn.Parameter(torch.tensor(10.0))

        # likewise, we cap the effective alt and ref counts to avoid arbitrarily large confidence
        self.max_alt = nn.Parameter(torch.tensor(20.0))
        self.max_ref = nn.Parameter(torch.tensor(20.0))

    def temperature(self, ref_counts: Tensor, alt_counts: Tensor):
        ref_eff = torch.squeeze(self.max_ref * torch.tanh(ref_counts / self.max_ref)) + 0.0001
        alt_eff = torch.squeeze(self.max_alt * torch.tanh(alt_counts / self.max_alt))

        batch_size = len(ref_counts)
        basis_size = len(self.coeffs)

        F_alt = torch.vstack((torch.ones_like(alt_eff), alt_eff**(1/4), alt_eff**(1/3), alt_eff**(1/2)))
        F_ref = torch.vstack((torch.ones_like(ref_eff), ref_eff ** (1 / 4), ref_eff ** (1 / 3), ref_eff ** (1 / 2)))
        assert F_alt.shape == (basis_size, batch_size)
        assert F_ref.shape == (basis_size, batch_size)

        X = torch.matmul(self.coeffs, F_alt) * F_ref
        assert X.shape == (basis_size, batch_size)

        temperature = torch.sum(X, dim=0)

        assert temperature.shape == (batch_size, )

        return temperature

    def forward(self, logits, ref_counts: Tensor, alt_counts: Tensor):
        calibrated_logits = logits * self.temperature(ref_counts, alt_counts)
        return self.max_logit * torch.tanh(calibrated_logits / self.max_logit)

    def plot_temperature(self, title):
        x_y_lab_tuples = []
        alt_counts = torch.range(1, 50)
        for ref_count in [1, 5, 10, 25]:
            ref_counts = ref_count * torch.ones_like(alt_counts)
            temps = self.temperature(ref_counts, alt_counts).detach()
            x_y_lab_tuples.append((alt_counts.numpy(), temps.numpy(), str(ref_count) + " ref reads"))

        return plotting.simple_plot(x_y_lab_tuples, "alt count", "temperature", title)


class ArtifactModel(nn.Module):
    """
    DeepSets framework for reads and variant info.  We embed each read and concatenate the mean ref read
    embedding, mean alt read embedding, and variant info embedding, then apply an aggregation function to
    this concatenation.

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

    def __init__(self, params: ArtifactModelParameters, num_read_features: int, num_info_features: int, ref_sequence_length: int, device=torch.device("cpu")):
        super(ArtifactModel, self).__init__()

        self._device = device
        self._num_read_features = num_read_features
        self._num_info_features = num_info_features
        self._ref_sequence_length = ref_sequence_length
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

        # omega is the universal embedding of info field variant-level data
        info_layers = [self._num_info_features] + params.info_layers
        self.omega = MLP(info_layers, batch_normalize=params.batch_normalize, dropout_p=params.dropout_p)
        self.omega.to(self._device)

        self.ref_seq_cnn = DNASequenceConvolution(params.ref_seq_layer_strings, ref_sequence_length)
        self.ref_seq_cnn.to(self._device)

        # rho is the universal aggregation function
        ref_alt_info_ref_seq_embedding_dimension = 2 * self.read_embedding_dimension + self.omega.output_dimension() + self.ref_seq_cnn.output_dimension()

        # we have a different aggregation subnetwork for each variant type.  Everything below, in particular the read
        # transformers, is shared
        # The [1] is for the output of binary classification, represented as a single artifact/non-artifact logit
        self.refined_dimension = params.aggregation_layers[-1]
        self.rho = nn.ModuleList([MLP([ref_alt_info_ref_seq_embedding_dimension] + params.aggregation_layers, batch_normalize=params.batch_normalize,
                       dropout_p=params.dropout_p) for variant_type in Variation])
        self.rho.to(self._device)

        # after rho is applied to each alt read and averaged, pass the average to a final linear logit layer
        self.final_logit = nn.Linear(in_features=self.refined_dimension, out_features=1)
        self.final_logit.to(self._device)

        # one Calibration module for each variant type; that is, calibration depends on both count and type
        self.calibration = nn.ModuleList([Calibration() for variant_type in Variation])
        self.calibration.to(self._device)

    def num_read_features(self) -> int:
        return self._num_read_features

    def num_info_features(self) -> int:
        return self._num_info_features

    def ref_sequence_length(self) -> int:
        return self._ref_sequence_length

    def training_parameters(self):
        return chain(self.initial_read_embedding.parameters(), self.alt_transformer_encoder.parameters(), self.ref_transformer_encoder.parameters(),
                     self.omega.parameters(), self.rho.parameters(), self.final_logit.parameters(), self.calibration.parameters())

    def calibration_parameters(self):
        return self.calibration.parameters()

    def freeze_all(self):
        utils.freeze(self.parameters())

    def set_epoch_type(self, epoch_type: utils.Epoch):
        if epoch_type == utils.Epoch.TRAIN:
            self.train(True)
            utils.freeze(self.parameters())
            utils.unfreeze(self.training_parameters())
        else:
            self.freeze_all()

    # returns 1D tensor of length batch_size of log odds ratio (logits) between artifact and non-artifact
    def forward(self, batch: ReadSetBatch):
        transformed_reads = self.apply_transformer_to_reads(batch)
        return self.forward_from_transformed_reads(transformed_reads=transformed_reads, batch=batch)

    # for the sake of recycling the read embeddings when training with data augmentation, we split the forward pass
    # into 1) the expensive and recyclable embedding of every single read and 2) everything else
    # note that apply_phi_to_reads returns a 2D tensor of N x E, where E is the embedding dimensions and N is the total
    # number of reads in the whole batch.  Thus, we have to be careful to downsample within each datum.
    def apply_transformer_to_reads(self, batch: ReadSetBatch):
        initial_embedded_reads = self.initial_read_embedding(batch.get_reads_2d().to(self._device))

        # we have a 2D tensor where each row is a read, but we want to group them into read sets
        # since reads from different read sets should not see each other (also, refs and alts
        # shouldn't either)
        # thus we take the alt/ref reads and reshape into 3D tensors of shape
        # (batch_size x batch alt/ref count x read embedding dimension), then apply the
        # transformer

        ref_count, alt_count = batch.ref_count, batch.alt_count
        total_ref, total_alt = ref_count * batch.size(), alt_count * batch.size()
        ref_reads_3d = None if total_ref == 0 else initial_embedded_reads[:total_ref].reshape(batch.size(), ref_count, self.read_embedding_dimension)
        alt_reads_3d = initial_embedded_reads[total_ref:].reshape(batch.size(), alt_count, self.read_embedding_dimension)

        if self.alt_downsample < alt_count:
            alt_read_indices = torch.randperm(alt_count)[:self.alt_downsample]
            alt_reads_3d = alt_reads_3d[:, alt_read_indices, :]   # downsample only along the middle (read) dimension
            total_alt = batch.size() * self.alt_downsample

        transformed_alt_reads_2d = self.alt_transformer_encoder(alt_reads_3d).reshape(total_alt, self.read_embedding_dimension)
        transformed_ref_reads_2d = None if total_ref == 0 else self.ref_transformer_encoder(ref_reads_3d).reshape(total_ref, self.read_embedding_dimension)

        transformed_reads_2d = transformed_alt_reads_2d if total_ref == 0 else \
            torch.vstack([transformed_ref_reads_2d, transformed_alt_reads_2d])

        return transformed_reads_2d

    # input: reads that have passed through the alt/ref transformers
    # output: reads that have been refined through the rho subnetwork that sees the transformed alts along with ref, alt, and info
    def forward_from_transformed_reads_to_refined_reads(self, transformed_reads: Tensor, batch: ReadSetBatch, weight_range: float = 0, extra_output=False):
        weights = torch.ones(len(transformed_reads), 1, device=self._device) if weight_range == 0 else (1 + weight_range * (1 - 2 * torch.rand(len(transformed_reads), 1, device=self._device)))

        ref_count, alt_count = batch.ref_count, min(batch.alt_count, self.alt_downsample)
        total_ref, total_alt = ref_count * batch.size(), alt_count * batch.size()

        alt_wts = weights[total_ref:]
        alt_wt_sums = sums_over_chunks(alt_wts, alt_count)
        alt_wt_sq_sums = sums_over_chunks(torch.square(alt_wts), alt_count)

        # mean embedding of every read, alt and ref, at each datum
        all_read_means = ((0 if ref_count == 0 else sums_over_chunks(transformed_reads[:total_ref], ref_count)) + sums_over_chunks(transformed_reads[total_ref:], alt_count)) / (alt_count + ref_count)
        omega_info = self.omega(batch.get_info_2d().to(self._device))
        ref_seq_embedding = self.ref_seq_cnn(batch.get_ref_sequences_2d())

        # dimension is batch_size x ref transformer output size + omega output size + ref seq embedding size
        extra_tensor_2d = torch.hstack([all_read_means, omega_info, ref_seq_embedding])
        extra_tensor_2d = torch.repeat_interleave(extra_tensor_2d, repeats=alt_count, dim=0)

        # the alt reads have not been averaged yet to we need to copy each row of the extra tensor batch.alt_count times
        padded_transformed_alt_reads = torch.hstack([transformed_reads[total_ref:], extra_tensor_2d])

        # now we refine the alt reads -- no need to separate between read sets yet as this broadcasts over every read
        refined_alt = torch.zeros(total_alt, self.refined_dimension)
        one_hot_types_2d = batch.variant_type_one_hot()
        for n, _ in enumerate(Variation):
            # multiply the result of this variant type's aggregation layers by its
            # one-hot mask.  Yes, this is wasteful because we apply every aggregation to every datum.
            # TODO: make this more efficient.
            mask = torch.repeat_interleave(one_hot_types_2d[:, n], repeats=alt_count).reshape(total_alt, 1)  # 2D column tensor
            refined_alt += mask * self.rho[n].forward(padded_transformed_alt_reads)

        weighted_refined = alt_wts * refined_alt
        # weighted mean is sum of reads in a chunk divided by sum of weights in same chunk
        alt_means = sums_over_chunks(weighted_refined, alt_count) / alt_wt_sums

        # these are fed to the calibration, since reweighting effectively reduces the read counts
        effective_alt_counts = torch.square(alt_wt_sums) / alt_wt_sq_sums

        # 3D tensors -- batch size x alt count x transformed / refined dimension
        # TODO: maybe I should output these regardless -- they probably get garbage-collected all the same
        # TODO: with no hit in performance
        transformed_alt_output = transformed_reads[total_ref:].reshape(batch.size(), alt_count, -1) if extra_output else None
        refined_alt_output = refined_alt.reshape(batch.size(), alt_count, -1) if extra_output else None

        # note all_read_means pertains to transformer; alt_means pertains to refined
        return all_read_means, alt_means, omega_info, ref_seq_embedding, effective_alt_counts, transformed_alt_output, refined_alt_output

    def forward_from_transformed_reads_to_calibration(self, phi_reads: Tensor, batch: ReadSetBatch, weight_range: float = 0):
        all_read_means, alt_means, omega_info, ref_seq_embedding, effective_alt_counts, _, _ = self.forward_from_transformed_reads_to_refined_reads(phi_reads, batch, weight_range)

        logits = self.final_logit(alt_means).reshape(batch.size())

        return logits, batch.ref_count * torch.ones_like(effective_alt_counts), effective_alt_counts

    def forward_from_transformed_reads(self, transformed_reads: Tensor, batch: ReadSetBatch, weight_range: float = 0):
        logits, ref_counts, effective_alt_counts = self.forward_from_transformed_reads_to_calibration(transformed_reads, batch, weight_range)

        result = torch.zeros_like(logits)

        one_hot_types_2d = batch.variant_type_one_hot()
        for n, _ in enumerate(Variation):
            mask = one_hot_types_2d[:, n]
            result += mask * self.calibration[n].forward(logits, ref_counts, effective_alt_counts)

        return result

    def learn_calibration(self, dataset: ReadSetDataset, num_epochs, batch_size, num_workers):

        self.train(False)
        utils.freeze(self.parameters())
        utils.unfreeze(self.calibration_parameters())

        # TODO: code duplication between here and train_model
        # These are 1D tensors, one element per variant type
        artifact_to_non_artifact_ratios = torch.from_numpy(dataset.artifact_to_non_artifact_ratios()).to(self._device)
        artifact_to_non_artifact_log_prior_ratios = torch.log(artifact_to_non_artifact_ratios)

        # gather uncalibrated logits -- everything computed by the frozen part of the model -- so that we only
        # do forward and backward passes on the calibration submodule
        print("Computing uncalibrated part of model. . .")
        uncalibrated_logits_ref_alt_counts_labels_prior_ratios = []
        valid_loader = make_data_loader(dataset, utils.Epoch.VALID, batch_size, self._device.type == 'cuda', num_workers)
        pbar = tqdm(enumerate(valid_loader), mininterval=60)
        for n, batch in pbar:
            if not batch.is_labeled():
                continue
            transformed_reads = self.apply_transformer_to_reads(batch)
            logits, ref_counts, alt_counts = self.forward_from_transformed_reads_to_calibration(transformed_reads, batch)
            labels = batch.labels.to(self._device)

            # TODO: more code duplication between here and train_model
            types_one_hot = batch.variant_type_one_hot()
            log_prior_ratios = torch.sum(artifact_to_non_artifact_log_prior_ratios * types_one_hot, dim=1)

            uncalibrated_logits_ref_alt_counts_labels_prior_ratios.append((logits.detach(), ref_counts.detach(), alt_counts.detach(), labels, log_prior_ratios))

        print("Training calibration. . .")
        optimizer = torch.optim.Adam(self.calibration_parameters())
        bce = nn.BCEWithLogitsLoss(reduction='none')  # no reduction because we may want to first multiply by weights for unbalanced data
        for epoch in trange(1, num_epochs + 1, desc="Calibration epoch"):
            nll_loss = utils.StreamingAverage(device=self._device)

            pbar = tqdm(enumerate(uncalibrated_logits_ref_alt_counts_labels_prior_ratios), mininterval=60)
            for n, (logits, ref_counts, alt_counts, labels, log_prior_ratios) in pbar:
                pred = self.calibration.forward(logits, ref_counts, alt_counts)
                post_pred = pred + log_prior_ratios

                loss = torch.sum(bce(post_pred, labels))
                utils.backpropagate(optimizer, loss)
                nll_loss.record_sum(loss.detach(), len(logits))

    def train_model(self, dataset: ReadSetDataset, num_epochs, batch_size, num_workers, summary_writer: SummaryWriter, reweighting_range: float, m3_params: ArtifactModelParameters):
        bce = nn.BCEWithLogitsLoss(reduction='none')  # no reduction because we may want to first multiply by weights for unbalanced data
        train_optimizer = torch.optim.AdamW(self.training_parameters(), lr=m3_params.learning_rate)

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

        train_loader = make_data_loader(dataset, utils.Epoch.TRAIN, batch_size, self._device.type == 'cuda', num_workers)
        valid_loader = make_data_loader(dataset, utils.Epoch.VALID, batch_size, self._device.type == 'cuda', num_workers)

        for epoch in trange(1, num_epochs + 1, desc="Epoch"):
            for epoch_type in [utils.Epoch.TRAIN, utils.Epoch.VALID]:
                self.set_epoch_type(epoch_type)

                labeled_loss = utils.StreamingAverage(device=self._device)
                unlabeled_loss = utils.StreamingAverage(device=self._device)

                labeled_loss_by_type = {variant_type: utils.StreamingAverage(device=self._device) for variant_type in Variation}
                labeled_loss_by_count = {bin_idx: utils.StreamingAverage(device=self._device) for bin_idx in range(NUM_COUNT_BINS)}

                loader = train_loader if epoch_type == utils.Epoch.TRAIN else valid_loader
                pbar = tqdm(enumerate(loader), mininterval=60)
                for n, batch in pbar:
                    transformed_reads = self.apply_transformer_to_reads(batch)

                    orig_pred = self.forward_from_transformed_reads(transformed_reads, batch, weight_range=0)
                    aug1_pred = self.forward_from_transformed_reads(transformed_reads, batch, weight_range=reweighting_range)
                    aug2_pred = self.forward_from_transformed_reads(transformed_reads, batch, weight_range=reweighting_range)

                    if batch.is_labeled():
                        labels = batch.labels
                        # labeled loss: cross entropy for original and both augmented copies

                        # the variant-dependent weights multiply the (unreduced) bces inside the torch.sum below
                        # how does this work? Well the weights buy type vectors are 1D row vectors eg [SNV weight, INS weight, DEL weight]
                        # and if we broadcast multiply them by the 1-hot variant vectors we get eg
                        # [[SNV weight, 0, 0],
                        #   [0, 0, DEL weight],
                        #   [0, INS weight, 0]]
                        # taking the sum over each row then gives the weights

                        types_one_hot = batch.variant_type_one_hot()

                        log_prior_ratios = torch.sum(artifact_to_non_artifact_log_prior_ratios * types_one_hot, dim=1)
                        orig_post, aug1_post, aug2_post = orig_pred + log_prior_ratios, aug1_pred + log_prior_ratios, aug2_pred + log_prior_ratios

                        separate_losses = (bce(orig_post, labels) + bce(aug1_post, labels) + bce(aug2_post, labels))
                        loss = torch.sum(separate_losses)
                        labeled_loss.record_sum(loss.detach(), batch.size())

                        if batch.alt_count <= MAX_COUNT:
                            labeled_loss_by_count[multiple_of_three_bin_index(batch.alt_count)].record_sum(loss.detach(), batch.size())

                        losses_masked_by_type = separate_losses.reshape(batch.size(), 1) * types_one_hot
                        counts_by_type = torch.sum(types_one_hot, dim=0)
                        total_loss_by_type = torch.sum(losses_masked_by_type, dim=0)
                        variant_types = list(Variation)
                        for variant_type_idx in range(len(Variation)):
                            count_for_type = int(counts_by_type[variant_type_idx].item())
                            loss_for_type = total_loss_by_type[variant_type_idx].item()
                            labeled_loss_by_type[variant_types[variant_type_idx]].record_sum(loss_for_type, count_for_type)
                    else:
                        # unlabeled loss: consistency cross entropy between original and both augmented copies
                        loss1 = bce(aug1_pred, torch.sigmoid(orig_pred.detach()))
                        loss2 = bce(aug2_pred, torch.sigmoid(orig_pred.detach()))
                        loss3 = bce(aug1_pred, torch.sigmoid(aug2_pred.detach()))
                        loss = torch.sum(loss1 + loss2 + loss3) * labeled_to_unlabeled_ratio
                        unlabeled_loss.record_sum(loss.detach(), batch.size())

                    assert not loss.isnan().item()  # all sorts of errors produce a nan here.  This is a good place to spot it

                    if epoch_type == utils.Epoch.TRAIN:
                        utils.backpropagate(train_optimizer, loss)

                # done with one epoch type -- training or validation -- for this epoch
                summary_writer.add_scalar(epoch_type.name + "/Labeled Loss", labeled_loss.get(), epoch)
                summary_writer.add_scalar(epoch_type.name + "/Unlabeled Loss", unlabeled_loss.get(), epoch)

                for bin_idx, loss in labeled_loss_by_count.items():
                    summary_writer.add_scalar(epoch_type.name + "/Labeled Loss/By Count/" + str(multiple_of_three_bin_index_to_count(bin_idx)), loss.get(), epoch)

                for var_type, loss in labeled_loss_by_type.items():
                    summary_writer.add_scalar(epoch_type.name + "/Labeled Loss/By Type/" + var_type.name, loss.get(), epoch)

                print("Labeled loss for epoch " + str(epoch) + " of " + epoch_type.name + ": " + str(labeled_loss.get()))
            # done with training and validation for this epoch
            # note that we have not learned the AF spectrum yet
        # done with training

    # generators by name is eg {"train": train_generator, "valid": valid_generator}
    def evaluate_model_after_training(self, dataset, batch_size, num_workers, summary_writer: SummaryWriter, prefix: str = ""):
        train_loader = make_data_loader(dataset, utils.Epoch.TRAIN, batch_size, self._device.type == 'cuda', num_workers)
        valid_loader = make_data_loader(dataset, utils.Epoch.VALID, batch_size, self._device.type == 'cuda', num_workers)
        loaders_by_name = {"training": train_loader, "validation": valid_loader}
        self.freeze_all()
        self.cpu()
        self._device = "cpu"

        # round logit to nearest int, truncate to range, ending up with bins 0. . . 2*max_logit
        logit_to_bin = lambda logit: min(max(round(logit), -MAX_LOGIT), MAX_LOGIT) + MAX_LOGIT
        bin_center = lambda bin_idx: bin_idx - MAX_LOGIT

        # grid of figures -- rows are loaders, columns are variant types
        # each subplot has two line graphs of accuracy vs alt count, one each for artifact, non-artifact
        acc_vs_cnt_fig, acc_vs_cnt_axes = plt.subplots(len(loaders_by_name), len(Variation), sharex='all', sharey='all', squeeze=False)
        roc_fig, roc_axes = plt.subplots(len(loaders_by_name), len(Variation), sharex='all', sharey='all', squeeze=False)
        cal_fig, cal_axes = plt.subplots(len(loaders_by_name), len(Variation), sharex='all', sharey='all', squeeze=False)
        roc_by_cnt_fig, roc_by_cnt_axes = plt.subplots(len(loaders_by_name), len(Variation), sharex='all', sharey='all', squeeze=False, figsize=(10, 6), dpi=100)

        log_artifact_to_non_artifact_ratios = torch.from_numpy(np.log(dataset.artifact_to_non_artifact_ratios()))
        for loader_idx, (loader_name, loader) in enumerate(loaders_by_name.items()):
            # indexed by variant type, then count bin, then logit bin
            acc_vs_logit = {var_type: [[utils.StreamingAverage() for _ in range(2 * MAX_LOGIT + 1)] for _ in range(NUM_COUNT_BINS)] for var_type in Variation}

            # indexed by variant type, then call type (artifact vs variant), then count bin
            acc_vs_cnt = {var_type: defaultdict(lambda: [utils.StreamingAverage() for _ in range(NUM_COUNT_BINS)]) for var_type in Variation}

            roc_data = {var_type: [] for var_type in Variation}     # variant type -> (predicted logit, actual label)
            roc_data_by_cnt = {var_type: [[] for _ in range(NUM_COUNT_BINS)] for var_type in Variation}  # variant type, count -> (predicted logit, actual label)

            pbar = tqdm(enumerate(filter(lambda bat: bat.is_labeled(), loader)), mininterval=60)
            for n, batch in pbar:
                # In training we minimize the cross entropy loss wrt the posterior probability, accounting for priors;
                # Here we are considering artifacts and non-artifacts separately and use the uncorrected likelihood logits
                pred = self.forward(batch)
                correct = ((pred > 0) == (batch.labels > 0.5)).tolist()

                for variant_type, predicted_logit, label, correct_call in zip(batch.variant_types(), pred.tolist(), batch.labels.tolist(), correct):
                    count_bin_index = multiple_of_three_bin_index(min(MAX_COUNT, batch.alt_count))
                    acc_vs_cnt[variant_type][Call.SOMATIC if label < 0.5 else Call.ARTIFACT][count_bin_index].record(correct_call)
                    acc_vs_logit[variant_type][count_bin_index][logit_to_bin(predicted_logit)].record(correct_call)

                    roc_data[variant_type].append((predicted_logit, label))
                    roc_data_by_cnt[variant_type][count_bin_index].append((predicted_logit, label))

            # done collecting data for this particular loader, now fill in subplots for this loader's row
            for var_type in Variation:
                # data for one particular subplot (row = train / valid, column = variant type)

                non_empty_count_bins = [idx for idx in range(NUM_COUNT_BINS) if not acc_vs_cnt[var_type][label][idx].is_empty()]
                non_empty_logit_bins = [[idx for idx in range(2 * MAX_LOGIT + 1) if not acc_vs_logit[var_type][count_idx][idx].is_empty()] for count_idx in range(NUM_COUNT_BINS)]
                acc_vs_cnt_x_y_lab_tuples = [([multiple_of_three_bin_index_to_count(idx) for idx in non_empty_count_bins],
                                   [acc_vs_cnt[var_type][label][idx].get() for idx in non_empty_count_bins],
                                   label.name) for label in acc_vs_cnt[var_type].keys()]
                acc_vs_logit_x_y_lab_tuples = [([bin_center(idx) for idx in non_empty_logit_bins[count_idx]],
                                              [acc_vs_logit[var_type][count_idx][idx].get() for idx in non_empty_logit_bins[count_idx]],
                                              str(multiple_of_three_bin_index_to_count(count_idx))) for count_idx in range(NUM_COUNT_BINS)]

                plotting.simple_plot_on_axis(acc_vs_cnt_axes[loader_idx, var_type], acc_vs_cnt_x_y_lab_tuples, None, None)
                plotting.plot_accuracy_vs_accuracy_roc_on_axis([roc_data[var_type]], [None], roc_axes[loader_idx, var_type])

                plotting.plot_accuracy_vs_accuracy_roc_on_axis(roc_data_by_cnt[var_type],
                                                               [str(multiple_of_three_bin_index_to_count(idx)) for idx in range(NUM_COUNT_BINS)], roc_by_cnt_axes[loader_idx, var_type])

                # now the plot versus output logit
                plotting.simple_plot_on_axis(cal_axes[loader_idx, var_type], acc_vs_logit_x_y_lab_tuples, None, None)

        # done collecting stats for all loaders and filling in subplots

        variation_types = [var_type.name for var_type in Variation]
        loader_names = [name for (name, loader) in loaders_by_name.items()]
        plotting.tidy_subplots(acc_vs_cnt_fig, acc_vs_cnt_axes, x_label="alt count", y_label="accuracy", row_labels=loader_names, column_labels=variation_types)
        plotting.tidy_subplots(roc_fig, roc_axes, x_label="non-artifact accuracy", y_label="artifact accuracy", row_labels=loader_names, column_labels=variation_types)
        plotting.tidy_subplots(roc_by_cnt_fig, roc_by_cnt_axes, x_label="non-artifact accuracy", y_label="artifact accuracy", row_labels=loader_names, column_labels=variation_types)
        plotting.tidy_subplots(cal_fig, cal_axes, x_label="predicted logit", y_label="accuracy", row_labels=loader_names, column_labels=variation_types)

        summary_writer.add_figure("{} accuracy by alt count".format(prefix), acc_vs_cnt_fig)
        summary_writer.add_figure(prefix + " accuracy by logit output", cal_fig)
        summary_writer.add_figure(prefix + " variant accuracy vs artifact accuracy curve", roc_fig)
        summary_writer.add_figure(prefix + " variant accuracy vs artifact accuracy curves by alt count", roc_by_cnt_fig)

        # now go over just the validation data and generate feature vectors / metadata for tensorboard projectors (UMAP)
        # also generate histograms of magnitudes of average read embeddings
        pbar = tqdm(enumerate(filter(lambda bat: bat.is_labeled(), valid_loader)), mininterval=60)

        # things we will collect for the projections
        label_metadata = []     # list (extended by each batch) 1 if artifact, 0 if not
        correct_metadata = []   # list (extended by each batch), 1 if correct prediction, 0 if not
        type_metadata = []      # list of lists, strings of variant type
        truncated_count_metadata = []   # list of lists
        average_read_embedding_features = []    # list of 2D tensors (to be stacked into a single 2D tensor), average read embedding over batches
        info_embedding_features = []
        ref_seq_embedding_features = []

        # we output array figures of histograms.  The rows of each histogram are (rounded) alt counts, the columns
        # are variant types, and each subplot has overlapping histograms for non-artifacts and artifacts.

        # There are several figures of this type, which for now we enocde as tuples.  The first element is 0, 1, 2 for L0, L1, or L2 norm
        # the second is a boolean norm first for whether we take the norm then the means (if true) or vice versa (if false)
        # the third is a boolean for transformed (True) or refined (False)
        norm_types = [1, 2, 100]
        norm_first_values = [True, False]
        trans_or_refined_values = [True, False]
        histogram_tuples = [(x, y, z) for x in norm_types for y in norm_first_values for z in trans_or_refined_values]

        hists_by_type_cnt = [{var_type: [([], []) for _ in range(NUM_COUNT_BINS)] for var_type in Variation} for tup in histogram_tuples]

        for n, batch in pbar:
            types_one_hot = batch.variant_type_one_hot()
            log_prior_odds = torch.sum(log_artifact_to_non_artifact_ratios * types_one_hot, dim=1)

            pred = self.forward(batch)
            posterior_pred = pred + log_prior_odds
            correct = ((posterior_pred > 0) == (batch.labels > 0.5)).tolist()

            label_metadata.extend(["artifact" if x > 0.5 else "non-artifact" for x in batch.labels.tolist()])
            correct_metadata.extend([str(val) for val in correct])
            type_metadata.extend([Variation(idx).name for idx in batch.variant_types()])
            truncated_count_metadata.extend(batch.size() * [str(round_up_to_nearest_three(min(MAX_COUNT, batch.alt_count)))])

            read_embeddings = self.apply_transformer_to_reads(batch)

            all_read_means, alt_means, omega_info, ref_seq_embedding, effective_alt_counts, transformed_alt_3d, refined_alt_3d = \
                self.forward_from_transformed_reads_to_refined_reads(read_embeddings, batch, extra_output=True)
            average_read_embedding_features.append(alt_means)
            info_embedding_features.append(omega_info)
            ref_seq_embedding_features.append(ref_seq_embedding)

            # code for the norm histograms
            for n, (norm_type, norm_first, transformed_vs_refined) in enumerate(histogram_tuples):
                tensor_3d = transformed_alt_3d if transformed_vs_refined else refined_alt_3d

                # norm over representation dimension (2), then mean over read index (1) if norm_first
                # else mean over read index (1) then norm over the new representation dimension (1)
                output_1d = (tensor_3d.norm(dim=2, p=norm_type).mean(dim=1) if norm_first else \
                    tensor_3d.mean(dim=1).norm(dim=1, p=norm_type)) if norm_type < 100 else \
                    (tensor_3d.max(dim=2).values.mean(dim=1) if norm_first else tensor_3d.mean(dim=1).max(dim=1).values)

                for variant_type,  label, output_value, in zip(batch.variant_types(), batch.labels.tolist(), output_1d):
                    bin_idx = multiple_of_three_bin_index(min(MAX_COUNT, batch.alt_count))
                    is_artifact_index = 0 if label < 0.5 else 1
                    hists_by_type_cnt[n][variant_type][bin_idx][is_artifact_index].append(output_value.item())

        # done collecting data

        for n, (norm_type, norm_first, transformed_vs_refined) in enumerate(histogram_tuples):
            hist_fig, hist_axes = plt.subplots(NUM_COUNT_BINS, len(Variation), sharex='none', sharey='none',
                                                   squeeze=False, figsize=(15, 15), dpi=100)

            name1 = "L" + str(norm_type) + " norm"
            name2 = ("mean of " + name1) if norm_first else (name1 + " of mean")
            name3 = name2 + " transformed" if transformed_vs_refined else " refined"

            for col_idx, var_type in enumerate(Variation):
                for row_idx in range(NUM_COUNT_BINS):
                    # data for one particular subplot (row = count bin, column = variant type)
                    plotting.simple_histograms_on_axis(hist_axes[row_idx, col_idx], hists_by_type_cnt[n][var_type][row_idx], ["good", "bad"], 25)
            variation_types = [var_type.name for var_type in Variation]
            plotting.tidy_subplots(hist_fig, hist_axes, x_label=name3, y_label="",
                                   row_labels=["alt count " + str(multiple_of_three_bin_index_to_count(idx)) for idx in range(NUM_COUNT_BINS)],
                                   column_labels=variation_types, keep_axes_tick_labels=True)
            summary_writer.add_figure(name3, hist_fig)

        # downsample to a reasonable amount of UMAP data
        all_metadata=list(zip(label_metadata, correct_metadata, type_metadata, truncated_count_metadata))
        idx = np.random.choice(len(all_metadata), size=min(NUM_DATA_FOR_TENSORBOARD_PROJECTION, len(all_metadata)), replace=False)

        summary_writer.add_embedding(torch.vstack(average_read_embedding_features)[idx],
                                     metadata=[all_metadata[n] for n in idx],
                                     metadata_header=["Labels", "Correctness", "Types", "Counts"],
                                     tag="mean read embedding")

        summary_writer.add_embedding(torch.vstack(info_embedding_features)[idx],
                                     metadata=[all_metadata[n] for n in idx],
                                     metadata_header=["Labels", "Correctness", "Types", "Counts"],
                                     tag="info embedding")

        summary_writer.add_embedding(torch.vstack(ref_seq_embedding_features)[idx],
                                     metadata=[all_metadata[n] for n in idx],
                                     metadata_header=["Labels", "Correctness", "Types", "Counts"],
                                     tag="ref seq embedding")

        # read average embeddings stratified by variant type
        for variant_type in Variation:
            variant_name = variant_type.name
            indices = [n for n, type_name in enumerate(type_metadata) if type_name == variant_name]
            indices = sample_indices_for_tensorboard(indices)
            summary_writer.add_embedding(torch.vstack(average_read_embedding_features)[indices],
                                         metadata=[all_metadata[n] for n in indices],
                                         metadata_header=["Labels", "Correctness", "Types", "Counts"],
                                         tag="mean read embedding for variant type " + variant_name)

        # read average embeddings stratified by alt count
        for row_idx in range(NUM_COUNT_BINS):
            count = multiple_of_three_bin_index_to_count(row_idx)
            indices = [n for n, alt_count in enumerate(truncated_count_metadata) if alt_count == str(count)]
            indices = sample_indices_for_tensorboard(indices)
            if len(indices) > 0:
                summary_writer.add_embedding(torch.vstack(average_read_embedding_features)[indices],
                                        metadata=[all_metadata[n] for n in indices],
                                        metadata_header=["Labels", "Correctness", "Types", "Counts"],
                                        tag="mean read embedding for alt count " + str(count))


def sample_indices_for_tensorboard(indices: List[int]):
    indices_np = np.array(indices)

    if len(indices_np) <= NUM_DATA_FOR_TENSORBOARD_PROJECTION:
        return indices_np

    idx = np.random.choice(len(indices_np), size=NUM_DATA_FOR_TENSORBOARD_PROJECTION, replace=False)
    return indices_np[idx]

