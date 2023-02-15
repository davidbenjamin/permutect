# bug before PyTorch 1.7.1 that warns when constructing ParameterList
import warnings
from collections import defaultdict

import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from torch import nn
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


def effective_count(weights: torch.Tensor):
    return (torch.square(torch.sum(weights)) / torch.sum(torch.square(weights))).item()


# group rows into consecutive chunks to yield a 3D tensor, average over dim=1 to get
# 2D tensor of sums within each chunk
def sums_over_chunks(tensor2d: torch.Tensor, chunk_size: int):
    assert len(tensor2d) % chunk_size == 0
    return torch.sum(tensor2d.reshape([len(tensor2d) // chunk_size, chunk_size, -1]), dim=1)


# note that read layers and info layers exclude the input dimension
class ArtifactModelParameters:
    def __init__(self, read_layers, info_layers, aggregation_layers, ref_seq_layers_strings, dropout_p, batch_normalize, learning_rate):
        self.read_layers = read_layers
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

        self.coeffs = nn.Parameter(torch.Tensor([[1.0, 0.001, 0.001, 0.001], [0.001, 0.001, 0.001, 0.001], [0.001, 0.001, 0.001, 0.001], [0.001, 0.001, 0.001, 0.001]]))

        # we apply as asymptotic threshold function logit --> M * tanh(logit/M) where M is the maximum absolute
        # value of the thresholded output.  For logits << M this is the identity, and approaching M the asymptote
        # gradually turns on.  This is a continuous way to truncate the model's confidence and is part of calibration.
        # We initialize it to something large.
        self.max_logit = nn.Parameter(torch.tensor(10.0))

        # likewise, we cap the effective alt and ref counts to avoid arbitrarily large confidence
        self.max_alt = nn.Parameter(torch.tensor(20.0))
        self.max_ref = nn.Parameter(torch.tensor(20.0))

    def temperature(self, ref_counts: torch.Tensor, alt_counts: torch.Tensor):
        ref_eff = torch.squeeze(self.max_ref * torch.tanh(ref_counts / self.max_ref))
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

    def forward(self, logits, ref_counts: torch.Tensor, alt_counts: torch.Tensor):
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

    # TODO: left off here: need to make sure that ref_Sequence_length gets passed to constructor, as it should be
    def __init__(self, params: ArtifactModelParameters, num_read_features: int, num_info_features: int, ref_sequence_length: int, device=torch.device("cpu")):
        super(ArtifactModel, self).__init__()

        self._device = device
        self._num_read_features = num_read_features
        self._num_info_features = num_info_features
        self._ref_sequence_length = ref_sequence_length

        # phi is the read embedding
        read_layers = [self._num_read_features] + params.read_layers
        self.phi = MLP(read_layers, batch_normalize=params.batch_normalize, dropout_p=params.dropout_p)
        self.phi.to(self._device)

        # omega is the universal embedding of info field variant-level data
        info_layers = [self._num_info_features] + params.info_layers
        self.omega = MLP(info_layers, batch_normalize=params.batch_normalize, dropout_p=params.dropout_p)
        self.omega.to(self._device)

        self.ref_seq_cnn = DNASequenceConvolution(params.ref_seq_layer_strings, ref_sequence_length)
        self.ref_seq_cnn.to(self._device)

        # rho is the universal aggregation function
        ref_alt_info_ref_seq_embedding_dimension = 2 * self.phi.output_dimension() + self.omega.output_dimension() + self.ref_seq_cnn.output_dimension()
        alt_info_ref_seq_embedding_dimension = self.phi.output_dimension() + self.omega.output_dimension() + self.ref_seq_cnn.output_dimension()

        # The [1] is for the output of binary classification, represented as a single artifact/non-artifact logit
        self.rho = MLP([ref_alt_info_ref_seq_embedding_dimension] + params.aggregation_layers + [1], batch_normalize=params.batch_normalize,
                       dropout_p=params.dropout_p)
        self.rho.to(self._device)

        self.rho_no_ref = MLP([alt_info_ref_seq_embedding_dimension] + params.aggregation_layers + [1],
                       batch_normalize=params.batch_normalize, dropout_p=params.dropout_p)
        self.rho_no_ref.to(self._device)

        self.calibration = Calibration()
        self.calibration.to(self._device)

    def num_read_features(self) -> int:
        return self._num_read_features

    def num_info_features(self) -> int:
        return self._num_info_features

    def ref_sequence_length(self) -> int:
        return self._ref_sequence_length

    def training_parameters(self):
        return chain(self.phi.parameters(), self.omega.parameters(), self.rho.parameters(), self.rho_no_ref.parameters(), self.calibration.parameters())

    def calibration_parameters(self):
        return self.calibration.parameters()

    def freeze_all(self):
        utils.freeze(self.parameters())

    def set_epoch_type(self, epoch_type: utils.Epoch, use_ref_reads: bool = True):
        if epoch_type == utils.Epoch.TRAIN:
            self.train(True)
            utils.freeze(self.parameters())
            if use_ref_reads:
                utils.unfreeze(self.training_parameters())
            else:
                utils.unfreeze(self.rho_no_ref.parameters())
        else:
            self.freeze_all()

    # returns 1D tensor of length batch_size of log odds ratio (logits) between artifact and non-artifact
    def forward(self, batch: ReadSetBatch, use_ref_reads: bool = True):
        phi_reads = self.apply_phi_to_reads(batch)
        return self.forward_from_phi_reads(phi_reads=phi_reads, batch=batch, use_ref_reads=(use_ref_reads and batch.ref_count > 0))

    # for the sake of recycling the read embeddings when training with data augmentation, we split the forward pass
    # into 1) the expensive and recyclable embedding of every single read and 2) everything else
    # note that apply_phi_to_reads returns a 2D tensor of N x E, where E is the embedding dimensions and N is the total
    # number of reads in the whole batch.  Thus, we have to be careful to downsample within each datum.
    def apply_phi_to_reads(self, batch: ReadSetBatch):
        # note that we put the reads on GPU, apply read embedding phi, then put the result back on CPU
        return torch.sigmoid(self.phi(batch.reads.to(self._device)))

    def forward_from_phi_reads_to_calibration(self, phi_reads: torch.Tensor, batch: ReadSetBatch, weight_range: float = 0, use_ref_reads: bool = True):
        weights = torch.ones(len(phi_reads), 1, device=self._device) if weight_range == 0 else (1 + weight_range * (1 - 2 * torch.rand(len(phi_reads), 1, device=self._device)))
        weighted_phi_reads = weights * phi_reads

        ref_count, alt_count = batch.ref_count, batch.alt_count
        total_ref = ref_count * batch.size()

        ref_wts, alt_wts = weights[:total_ref], weights[total_ref:]
        ref_wt_sums, alt_wt_sums = sums_over_chunks(ref_wts, ref_count), sums_over_chunks(alt_wts, alt_count)
        ref_wt_sq_sums, alt_wt_sq_sums = sums_over_chunks(torch.square(ref_wts), ref_count), sums_over_chunks(torch.square(alt_wts), alt_count)

        # weighted mean is sum of reads in a chunk divided by sum of weights in same chunk
        ref_means = sums_over_chunks(weighted_phi_reads[:total_ref], ref_count) / ref_wt_sums if use_ref_reads else None
        alt_means = sums_over_chunks(weighted_phi_reads[total_ref:], alt_count) / alt_wt_sums

        # these are fed to the calibration, since reweighting effectively reduces the read counts
        effective_alt_counts = torch.square(alt_wt_sums) / alt_wt_sq_sums
        effective_ref_counts = torch.square(ref_wt_sums) / ref_wt_sq_sums if use_ref_reads else torch.zeros_like(effective_alt_counts)

        # stack side-by-side to get 2D tensor, where each variant row is (ref mean, alt mean, info)
        omega_info = torch.sigmoid(self.omega(batch.info.to(self._device)))

        ref_seq_embedding = self.ref_seq_cnn(batch.ref_sequences)
        concatenated = torch.cat((ref_means, alt_means, omega_info, ref_seq_embedding) if use_ref_reads else (alt_means, omega_info, ref_seq_embedding), dim=1)
        logits = (self.rho if use_ref_reads else self.rho_no_ref).forward(concatenated).squeeze(dim=1)  # specify dim so that in edge case of batch size 1 we get 1D tensor, not scalar
        return logits, effective_ref_counts, effective_alt_counts

    def forward_from_phi_reads(self, phi_reads: torch.Tensor, batch: ReadSetBatch, weight_range: float = 0, use_ref_reads: bool = True):
        logits, effective_ref_counts, effective_alt_counts = self.forward_from_phi_reads_to_calibration(phi_reads, batch, weight_range, use_ref_reads=use_ref_reads)
        return self.calibration.forward(logits, effective_ref_counts, effective_alt_counts)

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
            phi_reads = self.apply_phi_to_reads(batch)
            logits, ref_counts, alt_counts = self.forward_from_phi_reads_to_calibration(phi_reads, batch)
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
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                nll_loss.record_sum(loss.detach(), len(logits))

    def train_model(self, dataset: ReadSetDataset, num_epochs, batch_size, num_workers, summary_writer: SummaryWriter, reweighting_range: float, m3_params: ArtifactModelParameters, use_ref_reads: bool = True):
        bce = torch.nn.BCEWithLogitsLoss(reduction='none')  # no reduction because we may want to first multiply by weights for unbalanced data
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
                self.set_epoch_type(epoch_type, use_ref_reads=use_ref_reads)

                labeled_loss = utils.StreamingAverage(device=self._device)
                unlabeled_loss = utils.StreamingAverage(device=self._device)

                loader = train_loader if epoch_type == utils.Epoch.TRAIN else valid_loader
                pbar = tqdm(enumerate(loader), mininterval=10)
                for n, batch in pbar:
                    phi_reads = self.apply_phi_to_reads(batch)

                    orig_pred = self.forward_from_phi_reads(phi_reads, batch, weight_range=0, use_ref_reads=use_ref_reads)
                    aug1_pred = self.forward_from_phi_reads(phi_reads, batch, weight_range=reweighting_range, use_ref_reads=use_ref_reads)
                    aug2_pred = self.forward_from_phi_reads(phi_reads, batch, weight_range=reweighting_range, use_ref_reads=use_ref_reads)

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
                        train_optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        train_optimizer.step()

                # done with one epoch type -- training or validation -- for this epoch
                summary_writer.add_scalar(epoch_type.name + "/Labeled Loss" + ("" if use_ref_reads else "refless"), labeled_loss.get(), epoch)
                summary_writer.add_scalar(epoch_type.name + "/Unlabeled Loss" + ("" if use_ref_reads else "refless"), unlabeled_loss.get(), epoch)
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

        # replace the redundant identical SOMATIC/ARTIFACT legends on each subplot with a single legend for the figure
        handles, labels = acc_vs_cnt_axes[-1][-1].get_legend_handles_labels()
        acc_vs_cnt_fig.legend(handles, labels, loc='upper center')

        handles, labels = roc_axes[-1][-1].get_legend_handles_labels()
        roc_fig.legend(handles, labels, loc='upper center')

        handles, labels = roc_by_cnt_axes[-1][-1].get_legend_handles_labels()
        roc_by_cnt_fig.legend(handles, labels, loc='upper center')

        for ax in chain(acc_vs_cnt_fig.get_axes(), roc_fig.get_axes(), cal_fig.get_axes(), roc_by_cnt_fig.get_axes()):
            ax.label_outer()    # y tick labels only shown in leftmost column, x tick labels only shown on bottom row
            ax.legend().set_visible(False)  # hide the redundant identical subplot legends

            # remove the subplot labels and title -- these will be given manually to the whole figure and to the outer rows
            ax.set_xlabel(None)
            ax.set_ylabel(None)
            ax.set_title(None)

        for axes in acc_vs_cnt_axes, roc_axes, cal_axes, roc_by_cnt_axes:
            # make variant type column heading by setting titles on the top row of subplots
            for col_idx, var_type in enumerate(Variation):
                axes[0][col_idx].set_title(var_type.name)

            # make epoch/loader type row heading by setting y labels on leftmost column of subplots
            for row_idx, (loader_name, _) in enumerate(loaders_by_name.items()):
                axes[row_idx][0].set_ylabel(loader_name)

        acc_vs_cnt_fig.supxlabel("Alt read count")
        acc_vs_cnt_fig.supylabel("Accuracy")

        cal_fig.supxlabel("Predicted logit")
        cal_fig.supylabel("Accuracy")

        roc_fig.supxlabel("Non-artifact Accuracy")
        roc_fig.supylabel("Artifact Accuracy")

        roc_by_cnt_fig.supxlabel("Non-artifact Accuracy")
        roc_by_cnt_fig.supylabel("Artifact Accuracy")

        acc_vs_cnt_fig.tight_layout()
        roc_fig.tight_layout()
        roc_by_cnt_fig.tight_layout()
        cal_fig.tight_layout()

        summary_writer.add_figure("{} accuracy by alt count".format(prefix), acc_vs_cnt_fig)
        summary_writer.add_figure(prefix + " accuracy by logit output", cal_fig)
        summary_writer.add_figure(prefix + " variant accuracy vs artifact accuracy curve", roc_fig)
        summary_writer.add_figure(prefix + " variant accuracy vs artifact accuracy curves by alt count", roc_by_cnt_fig)