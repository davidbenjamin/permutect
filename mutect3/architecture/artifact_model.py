# bug before PyTorch 1.7.1 that warns when constructing ParameterList
import warnings
from collections import defaultdict

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

NUM_DATA_FOR_TENSORBOARD_PROJECTION=10000


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
                 ref_seq_layers_strings, dropout_p, batch_normalize, learning_rate):

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
        alt_counts = torch.range(1, 100)
        for ref_count in [1, 5, 10, 25, 50, 100]:
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

        # linear transformation to convert read tensors from their initial dimensionality
        # to the embedding dimension eg data of shape [num_batches -- optional] x num_reads x read dimension
        # maps to data of shape [num_batches] x num_reads x embedding dimension)
        self.initial_read_embedding = torch.nn.Linear(in_features=num_read_features, out_features=params.read_embedding_dimension)
        self.initial_read_embedding.to(self._device)

        self.read_embedding_dimension = params.read_embedding_dimension
        self.num_transformer_heads = params.num_transformer_heads
        self.transformer_hidden_dimension = params.transformer_hidden_dimension
        self.num_transformer_layers = params.num_transformer_layers

        self.transformer_encoder_layer = torch.nn.TransformerEncoderLayer(d_model=params.read_embedding_dimension,
            nhead=params.num_transformer_heads, batch_first=True, dim_feedforward=params.transformer_hidden_dimension,
            dropout=params.dropout_p)
        self.transformer_encoder = torch.nn.TransformerEncoder(self.transformer_encoder_layer, num_layers=params.num_transformer_layers)
        self.transformer_encoder.to(self._device)

        # omega is the universal embedding of info field variant-level data
        info_layers = [self._num_info_features] + params.info_layers
        self.omega = MLP(info_layers, batch_normalize=params.batch_normalize, dropout_p=params.dropout_p)
        self.omega.to(self._device)

        self.ref_seq_cnn = DNASequenceConvolution(params.ref_seq_layer_strings, ref_sequence_length)
        self.ref_seq_cnn.to(self._device)

        # rho is the universal aggregation function
        ref_alt_info_ref_seq_embedding_dimension = 2 * self.read_embedding_dimension + self.omega.output_dimension() + self.ref_seq_cnn.output_dimension()

        # The [1] is for the output of binary classification, represented as a single artifact/non-artifact logit
        self.rho = MLP([ref_alt_info_ref_seq_embedding_dimension] + params.aggregation_layers + [1], batch_normalize=params.batch_normalize,
                       dropout_p=params.dropout_p)
        self.rho.to(self._device)

        self.calibration = Calibration()
        self.calibration.to(self._device)

    def num_read_features(self) -> int:
        return self._num_read_features

    def num_info_features(self) -> int:
        return self._num_info_features

    def ref_sequence_length(self) -> int:
        return self._ref_sequence_length

    def training_parameters(self):
        return chain(self.initial_read_embedding.parameters(),self.transformer_encoder.parameters(),
                     self.omega.parameters(), self.rho.parameters(), self.calibration.parameters())

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

        transformed_alt_reads_2d = self.transformer_encoder(alt_reads_3d).reshape(total_alt, self.read_embedding_dimension)
        transformed_ref_reads_2d = None if total_ref == 0 else self.transformer_encoder(ref_reads_3d).reshape(total_ref, self.read_embedding_dimension)

        transformed_reads_2d = transformed_alt_reads_2d if total_ref == 0 else \
            torch.vstack([transformed_ref_reads_2d, transformed_alt_reads_2d])

        # TODO: do we need the sigmoid?? We have not evaluated this for the transformer
        return 2 * (torch.sigmoid(transformed_reads_2d) - 0.5)

    def forward_from_transformed_reads_to_intermediate_layer_output(self, phi_reads: Tensor, batch: ReadSetBatch, weight_range: float = 0):
        weights = torch.ones(len(phi_reads), 1, device=self._device) if weight_range == 0 else (1 + weight_range * (1 - 2 * torch.rand(len(phi_reads), 1, device=self._device)))
        weighted_phi_reads = weights * phi_reads

        ref_count, alt_count = batch.ref_count, batch.alt_count
        total_ref = ref_count * batch.size()

        alt_wts = weights[total_ref:]
        alt_wt_sums = sums_over_chunks(alt_wts, alt_count)
        alt_wt_sq_sums = sums_over_chunks(torch.square(alt_wts), alt_count)

        # mean embedding of every read, alt and ref, at each datum
        all_read_means = ((0 if ref_count == 0 else sums_over_chunks(phi_reads[:total_ref], ref_count)) + sums_over_chunks(weighted_phi_reads[total_ref:], alt_count)) / (alt_count + ref_count)

        # weighted mean is sum of reads in a chunk divided by sum of weights in same chunk
        alt_means = sums_over_chunks(weighted_phi_reads[total_ref:], alt_count) / alt_wt_sums

        # these are fed to the calibration, since reweighting effectively reduces the read counts
        effective_alt_counts = torch.square(alt_wt_sums) / alt_wt_sq_sums

        # stack side-by-side to get 2D tensor, where each variant row is (ref mean, alt mean, info)
        omega_info = torch.sigmoid(self.omega(batch.get_info_2d().to(self._device)))

        ref_seq_embedding = self.ref_seq_cnn(batch.get_ref_sequences_2d())

        return all_read_means, alt_means, omega_info, ref_seq_embedding, effective_alt_counts

    def forward_from_transformed_reads_to_calibration(self, phi_reads: Tensor, batch: ReadSetBatch, weight_range: float = 0):
        all_read_means, alt_means, omega_info, ref_seq_embedding, effective_alt_counts = self.forward_from_transformed_reads_to_intermediate_layer_output(phi_reads, batch, weight_range)
        concatenated = torch.cat((all_read_means, alt_means, omega_info, ref_seq_embedding), dim=1)
        logits = self.rho.forward(concatenated).squeeze(dim=1)  # specify dim so that in edge case of batch size 1 we get 1D tensor, not scalar
        return logits, batch.ref_count * torch.ones_like(effective_alt_counts), effective_alt_counts

    def forward_from_transformed_reads(self, transformed_reads: Tensor, batch: ReadSetBatch, weight_range: float = 0):
        logits, ref_counts, effective_alt_counts = self.forward_from_transformed_reads_to_calibration(transformed_reads, batch, weight_range)
        return self.calibration.forward(logits, ref_counts, effective_alt_counts)

    def learn_calibration(self, dataset: ReadSetDataset, num_epochs, batch_size, num_workers):

        self.train(False)
        utils.freeze(self.parameters())
        utils.unfreeze(self.calibration_parameters())

        # TODO: code duplication between here and train_model
        labeled_artifact_to_non_artifact_ratios = dataset.artifact_to_non_artifact_ratios()
        labeled_artifact_weights_by_type = torch.from_numpy(1 / np.sqrt(labeled_artifact_to_non_artifact_ratios)).to(self._device)
        labeled_non_artifact_weights_by_type = torch.from_numpy(np.sqrt(labeled_artifact_to_non_artifact_ratios)).to(self._device)

        # gather uncalibrated logits -- everything computed by the frozen part of the model -- so that we only
        # do forward and backward passes on the calibration submodule
        print("Computing uncalibrated part of model. . .")
        uncalibrated_logits_ref_alt_counts_labels_weights = []
        valid_loader = make_data_loader(dataset, utils.Epoch.VALID, batch_size, self._device.type == 'cuda', num_workers)
        pbar = tqdm(enumerate(valid_loader), mininterval=10)
        for n, batch in pbar:
            if not batch.is_labeled():
                continue
            phi_reads = self.apply_transformer_to_reads(batch)
            logits, ref_counts, alt_counts = self.forward_from_transformed_reads_to_calibration(phi_reads, batch)
            labels = batch.labels.to(self._device)

            # TODO: more code duplication between here and train_model
            types_one_hot = batch.variant_type_one_hot()
            weights = labels * torch.sum(labeled_artifact_weights_by_type * types_one_hot, dim=1) + \
                                     (1 - labels) * torch.sum(labeled_non_artifact_weights_by_type * types_one_hot, dim=1)

            uncalibrated_logits_ref_alt_counts_labels_weights.append((logits.detach(), ref_counts.detach(), alt_counts.detach(), labels, weights))

        print("Training calibration. . .")
        optimizer = torch.optim.Adam(self.calibration_parameters())
        bce = nn.BCEWithLogitsLoss(reduction='none')  # no reduction because we may want to first multiply by weights for unbalanced data
        for epoch in trange(1, num_epochs + 1, desc="Calibration epoch"):
            nll_loss = utils.StreamingAverage(device=self._device)

            pbar = tqdm(enumerate(uncalibrated_logits_ref_alt_counts_labels_weights), mininterval=10)
            for n, (logits, ref_counts, alt_counts, labels, weights) in pbar:
                pred = self.calibration.forward(logits, ref_counts, alt_counts)

                loss = torch.sum(weights * bce(pred, labels))
                utils.backpropagate(optimizer, loss)
                nll_loss.record_sum(loss.detach(), len(logits))

    def train_model(self, dataset: ReadSetDataset, num_epochs, batch_size, num_workers, summary_writer: SummaryWriter, reweighting_range: float, m3_params: ArtifactModelParameters):
        bce = nn.BCEWithLogitsLoss(reduction='none')  # no reduction because we may want to first multiply by weights for unbalanced data
        train_optimizer = torch.optim.AdamW(self.training_parameters(), lr=m3_params.learning_rate)

        labeled_artifact_to_non_artifact_ratios = dataset.artifact_to_non_artifact_ratios()

        labeled_artifact_weights_by_type = torch.from_numpy(1 / np.sqrt(labeled_artifact_to_non_artifact_ratios)).to(self._device)
        labeled_non_artifact_weights_by_type = torch.from_numpy(np.sqrt(labeled_artifact_to_non_artifact_ratios)).to(self._device)

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

                loader = train_loader if epoch_type == utils.Epoch.TRAIN else valid_loader
                pbar = tqdm(enumerate(loader), mininterval=10)
                for n, batch in pbar:
                    phi_reads = self.apply_transformer_to_reads(batch)

                    orig_pred = self.forward_from_transformed_reads(phi_reads, batch, weight_range=0)
                    aug1_pred = self.forward_from_transformed_reads(phi_reads, batch, weight_range=reweighting_range)
                    aug2_pred = self.forward_from_transformed_reads(phi_reads, batch, weight_range=reweighting_range)

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
                        loss_balancing_factors = labels * torch.sum(labeled_artifact_weights_by_type * types_one_hot, dim=1) + \
                            (1 - labels) * torch.sum(labeled_non_artifact_weights_by_type * types_one_hot, dim=1)

                        loss = torch.sum(loss_balancing_factors * (bce(orig_pred, labels) + bce(aug1_pred, labels) + bce(aug2_pred, labels)))
                        labeled_loss.record_sum(loss.detach(), batch.size())
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

        max_count = 20  # counts above this will be truncated
        max_logit = 6

        # round logit to nearest int, truncate to range, ending up with bins 0. . . 2*max_logit
        logit_to_bin = lambda logit: min(max(round(logit), -max_logit), max_logit) + max_logit
        bin_center = lambda bin_idx: bin_idx - max_logit

        # grid of figures -- rows are loaders, columns are variant types
        # each subplot has two line graphs of accuracy vs alt count, one each for artifact, non-artifact
        acc_vs_cnt_fig, acc_vs_cnt_axes = plt.subplots(len(loaders_by_name), len(Variation), sharex='all', sharey='all', squeeze=False)
        roc_fig, roc_axes = plt.subplots(len(loaders_by_name), len(Variation), sharex='all', sharey='all', squeeze=False)
        cal_fig, cal_axes = plt.subplots(len(loaders_by_name), len(Variation), sharex='all', sharey='all', squeeze=False)
        roc_by_cnt_fig, roc_by_cnt_axes = plt.subplots(len(loaders_by_name), len(Variation), sharex='all', sharey='all', squeeze=False, figsize=(10, 6), dpi=100)

        log_artifact_to_non_artifact_ratios = torch.from_numpy(np.log(dataset.artifact_to_non_artifact_ratios()))
        for loader_idx, (loader_name, loader) in enumerate(loaders_by_name.items()):
            # indexed by variant type, then logit bin
            acc_vs_logit = {var_type: [utils.StreamingAverage() for _ in range(2*max_logit + 1)] for var_type in Variation}

            # indexed by variant type, then call type (artifact vs variant), then count bin
            acc_vs_cnt = {var_type: defaultdict(lambda: [utils.StreamingAverage() for _ in range(max_count + 1)]) for var_type in Variation}

            roc_data = {var_type: [] for var_type in Variation}     # variant type -> (predicted logit, actual label)
            roc_data_by_cnt = {var_type: [[] for _ in range(max_count + 1)] for var_type in Variation}  # variant type, count -> (predicted logit, actual label)

            pbar = tqdm(enumerate(filter(lambda bat: bat.is_labeled(), loader)), mininterval=10)
            for n, batch in pbar:
                types_one_hot = batch.variant_type_one_hot()
                log_prior_odds = torch.sum(log_artifact_to_non_artifact_ratios * types_one_hot, dim=1)

                # We weight training loss to balance artifact and non-artifact within each variant type.
                # the roc curves don't depend on the relative amounts of artifacts and non-artifacts in the evaluation data
                # and so they use the logits from the artifact model as-is.  The accuracy vs count and accuracy vs logit curves,
                # however, do depend on the (lack of) balance in the evaluation data and so for those metrics
                # we restore an unbalanced prior to mimic what the posterior model would do
                pred = self.forward(batch)
                posterior_pred = pred + log_prior_odds
                correct = ((posterior_pred > 0) == (batch.labels > 0.5)).tolist()

                for variant_type, predicted_logit, posterior_logit, label, correct_call in zip(batch.variant_types(), pred.tolist(), posterior_pred.tolist(), batch.labels.tolist(), correct):
                    truncated_count = min(max_count, batch.alt_count)
                    acc_vs_cnt[variant_type][Call.SOMATIC if label < 0.5 else Call.ARTIFACT][truncated_count].record(correct_call)
                    acc_vs_logit[variant_type][logit_to_bin(posterior_logit)].record(correct_call)

                    roc_data[variant_type].append((predicted_logit, label))
                    roc_data_by_cnt[variant_type][truncated_count].append((predicted_logit, label))

            # done collecting data for this particular loader, now fill in subplots for this loader's row
            for var_type in Variation:
                # data for one particular subplot (row = train / valid, column = variant type)

                non_empty_count_bins = [count for count in range(max_count + 1) if not acc_vs_cnt[var_type][label][count].is_empty()]
                non_empty_logit_bins = [idx for idx in range(2*max_logit + 1) if not acc_vs_logit[var_type][idx].is_empty()]
                acc_vs_cnt_x_y_lab_tuples = [(non_empty_count_bins,
                                   [acc_vs_cnt[var_type][label][count].get() for count in non_empty_count_bins],
                                   label.name) for label in acc_vs_cnt[var_type].keys()]
                acc_vs_logit_x_y_lab_tuple = [([bin_center(idx) for idx in non_empty_logit_bins],
                                              [acc_vs_logit[var_type][idx].get() for idx in non_empty_logit_bins],
                                              None)]

                plotting.simple_plot_on_axis(acc_vs_cnt_axes[loader_idx, var_type], acc_vs_cnt_x_y_lab_tuples, None, None)
                plotting.plot_accuracy_vs_accuracy_roc_on_axis([roc_data[var_type]], [None], roc_axes[loader_idx, var_type])

                roc_alt_counts = [2, 3, 4, 5, 7, 10, 15]
                plotting.plot_accuracy_vs_accuracy_roc_on_axis([roc_data_by_cnt[var_type][alt_count] for alt_count in roc_alt_counts],
                                                               [str(alt_count) for alt_count in roc_alt_counts], roc_by_cnt_axes[loader_idx, var_type])

                # now the plot versus output logit
                plotting.simple_plot_on_axis(cal_axes[loader_idx, var_type], acc_vs_logit_x_y_lab_tuple, None, None)

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
        pbar = tqdm(enumerate(filter(lambda bat: bat.is_labeled(), valid_loader)), mininterval=10)

        # things we will collect for the projections
        label_metadata = []     # list (extended by each batch) 1 if artifact, 0 if not
        correct_metadata = []   # list (extended by each batch), 1 if correct prediction, 0 if not
        type_metadata = []      # list of lists, strings of variant type
        truncated_count_metadata = []   # list of lists
        average_read_embedding_features = []    # list of 2D tensors (to be stacked into a single 2D tensor), average read embedding over batches
        info_embedding_features = []
        ref_seq_embedding_features = []

        for n, batch in pbar:
            types_one_hot = batch.variant_type_one_hot()
            log_prior_odds = torch.sum(log_artifact_to_non_artifact_ratios * types_one_hot, dim=1)

            pred = self.forward(batch)
            posterior_pred = pred + log_prior_odds
            correct = ((posterior_pred > 0) == (batch.labels > 0.5)).tolist()

            label_metadata.extend(["artifact" if x > 0.5 else "non-artifact" for x in batch.labels.tolist()])
            correct_metadata.extend([str(val) for val in correct])
            type_metadata.extend([Variation(idx).name for idx in batch.variant_types()])
            truncated_count_metadata.extend(batch.size() * [str(min(max_count, batch.alt_count))])

            phi_reads = self.apply_transformer_to_reads(batch)

            omega_info = torch.sigmoid(self.omega(batch.get_info_2d().to(self._device)))
            ref_seq_embedding = self.ref_seq_cnn(batch.get_ref_sequences_2d())

            all_read_means, alt_means, omega_info, ref_seq_embedding, effective_alt_counts = \
                self.forward_from_transformed_reads_to_intermediate_layer_output(phi_reads, batch)
            average_read_embedding_features.append(alt_means)

            omega_info = torch.sigmoid(self.omega(batch.get_info_2d().to(self._device)))
            info_embedding_features.append(omega_info)

            ref_seq_embedding = self.ref_seq_cnn(batch.get_ref_sequences_2d())
            ref_seq_embedding_features.append(ref_seq_embedding)

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
