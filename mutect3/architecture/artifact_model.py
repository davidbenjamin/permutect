# bug before PyTorch 1.7.1 that warns when constructing ParameterList
import warnings

import torch
from torch.utils.tensorboard import SummaryWriter

from torch import nn
from tqdm.autonotebook import trange, tqdm
from itertools import chain

from mutect3.architecture.mlp import MLP
from mutect3.architecture.prior_model import PriorModel
from mutect3.data.read_set_batch import ReadSetBatch
from mutect3.data import read_set
from mutect3 import utils
from mutect3.metrics import plotting

warnings.filterwarnings("ignore", message="Setting attributes on ParameterList is not supported.")


def effective_count(weights: torch.Tensor):
    return (torch.square(torch.sum(weights)) / torch.sum(torch.square(weights))).item()


# note that read layers and info layers exclude the input dimension
class ArtifactModelParameters:
    def __init__(self, read_layers, info_layers, aggregation_layers, dropout_p, batch_normalize, learning_rate):
        self.read_layers = read_layers
        self.info_layers = info_layers
        self.aggregation_layers = aggregation_layers
        self.dropout_p = dropout_p
        self.batch_normalize = batch_normalize
        self.learning_rate = learning_rate


class Calibration(nn.Module):

    def __init__(self):
        super(Calibration, self).__init__()

        # take the transformed alt and ref counts (i.e. two input "features") and output
        self.mlp = MLP([2, 5, 5, 1])

        # we apply as asymptotic threshold function logit --> M * tanh(logit/M) where M is the maximum absolute
        # value of the thresholded output.  For logits << M this is the identity, and approaching M the asymptote
        # gradually turns on.  This is a continuous way to truncate the model's confidence and is part of calibration.
        # We initialize it to something large.
        self.max_logit = nn.Parameter(torch.tensor(10.0))

    def forward(self, logits, ref_counts: torch.Tensor, alt_counts: torch.Tensor):
        # based on stats 101 it's reasonable to guess that confidence depends on the sqrt of the evidence count
        # thus we apply a sqrt nonlinearity before the MLP in order to hopefully reduce the number of parameters needed.
        sqrt_counts = torch.column_stack((torch.sqrt(alt_counts), torch.sqrt(ref_counts)))

        # temperature scaling means multiplying logits -- in this case the temperature depends on alt and ref counts
        temperatures = torch.squeeze(self.mlp.forward(sqrt_counts))
        calibrated_logits = logits * temperatures
        return self.max_logit * torch.tanh(calibrated_logits / self.max_logit)


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

    def __init__(self, params: ArtifactModelParameters, device=torch.device("cpu")):
        super(ArtifactModel, self).__init__()

        self._device = device

        # phi is the read embedding
        read_layers = [read_set.NUM_READ_FEATURES] + params.read_layers
        self.phi = MLP(read_layers, batch_normalize=params.batch_normalize, dropout_p=params.dropout_p)
        self.phi.to(self._device)

        # omega is the universal embedding of info field variant-level data
        info_layers = [read_set.NUM_INFO_FEATURES] + params.info_layers
        self.omega = MLP(info_layers, batch_normalize=params.batch_normalize, dropout_p=params.dropout_p)
        self.omega.to(self._device)

        # rho is the universal aggregation function
        ref_alt_info_embedding_dimension = 2 * read_layers[-1] + info_layers[-1]

        # The [1] is for the output of binary classification, represented as a single artifact/non-artifact logit
        self.rho = MLP([ref_alt_info_embedding_dimension] + params.aggregation_layers + [1], batch_normalize=params.batch_normalize,
                       dropout_p=params.dropout_p)
        self.rho.to(self._device)

        self.calibration = Calibration()
        self.calibration.to(self._device)

        # note: prior model and normal artifact model are not part of training and thus are not ever put on GPU
        self.prior_model = PriorModel(0.0)

    def get_prior_model(self):
        return self.prior_model

    def training_parameters(self):
        return chain(self.phi.parameters(), self.omega.parameters(), self.rho.parameters(), [self.calibration.max_logit])

    def calibration_parameters(self):
        return self.calibration.parameters()

    def spectra_parameters(self):
        return self.prior_model.parameters()

    def freeze_all(self):
        utils.freeze(self.parameters())

    def set_epoch_type(self, epoch_type: utils.EpochType):
        if epoch_type == utils.EpochType.TRAIN:
            self.train(True)
            utils.freeze(self.parameters())
            utils.unfreeze(self.training_parameters())
        else:
            self.freeze_all()

    def learn_spectrum_mode(self):
        self.train(False)
        utils.freeze(self.parameters())
        utils.unfreeze(self.spectra_parameters())

    def forward(self, batch: ReadSetBatch, posterior=False):
        phi_reads = self.apply_phi_to_reads(batch)
        return self.forward_from_phi_reads(phi_reads=phi_reads, batch=batch, posterior=posterior)

    # for the sake of recycling the read embeddings when training with data augmentation, we split the forward pass
    # into 1) the expensive and recyclable embedding of every single read and 2) everything else
    # note that apply_phi_to_reads returns a 2D tensor of N x E, where E is the embedding dimensions and N is the total
    # number of reads in the whole batch.  Thus, we have to be careful to downsample within each datum.
    def apply_phi_to_reads(self, batch: ReadSetBatch):
        # note that we put the reads on GPU, apply read embedding phi, then put the result back on CPU
        return torch.sigmoid(self.phi(batch.reads().to(self._device)))

    def forward_from_phi_reads_to_calibration(self, phi_reads: torch.Tensor, batch: ReadSetBatch, weight_range: float=0):
        weights = torch.ones(len(phi_reads), 1, device=self._device) if weight_range == 0 else (1 + weight_range * (1 - 2 * torch.rand(len(phi_reads), 1, device=self._device)))
        weighted_phi_reads = weights * phi_reads

        end_indices = batch.read_end_indices().to(self._device)

        # 2D tensor of weighed sum of read embeddings within each datum -- all ref reads, then all alt reads
        read_sums = utils.chunk_sums(weighted_phi_reads, end_indices)
        weight_sums = utils.chunk_sums(weights, end_indices)
        read_means = read_sums / weight_sums

        squared_weight_sums = utils.chunk_sums(torch.square(weights), end_indices)
        effective_read_counts = torch.square(weight_sums) / squared_weight_sums

        ref_means = read_means[:batch.size(), :]
        alt_means = read_means[batch.size():, :]

        # these are fed to the calibration, since reweighting effectively reduces the read counts
        effective_ref_counts = effective_read_counts[:batch.size(), :]
        effective_alt_counts = effective_read_counts[batch.size():, :]

        # stack side-by-side to get 2D tensor, where each variant row is (ref mean, alt mean, info)
        omega_info = torch.sigmoid(self.omega(batch.info().to(self._device)))
        concatenated = torch.cat((ref_means, alt_means, omega_info), dim=1)
        logits = torch.squeeze(self.rho(concatenated))
        return logits, effective_ref_counts, effective_alt_counts

    # beta is for downsampling data augmentation
    def forward_from_phi_reads(self, phi_reads: torch.Tensor, batch: ReadSetBatch, posterior=False, weight_range: float=0):
        logits, effective_ref_counts, effective_alt_counts = self.forward_from_phi_reads_to_calibration(phi_reads, batch, weight_range)
        logits = self.calibration.forward(logits, effective_ref_counts, effective_alt_counts)

        if posterior:
            # the prior model is always on CPU, but if we are computing a posterior it's not training and hence we are not
            # using a GPU.  In principle we should put logits ont he CPU just in case, but we can be brittle for now
            # pending some refactoring.  The same applies to the normal artifact model.
            logits = self.prior_model.posterior_logits(logits, batch)

        return logits

    def learn_spectra(self, loader, num_iterations, summary_writer: SummaryWriter = None):
        self.learn_spectrum_mode()
        logits_and_batches = [(self.forward(batch=batch).detach(), batch) for batch in loader]
        optimizer = torch.optim.Adam(self.spectra_parameters())

        for epoch in trange(1, num_iterations + 1, desc="AF spectra epoch"):
            epoch_loss = utils.StreamingAverage(device=self._device)

            pbar = tqdm(enumerate(logits_and_batches), mininterval=10)
            for n, (logits, batch) in pbar:
                loss = -torch.mean(self.prior_model.log_evidence(logits, batch))
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                epoch_loss.record_sum(len(logits) * loss.detach(), len(logits))

            if summary_writer is not None:
                summary_writer.add_scalar("spectrum NLL", epoch_loss.get(), epoch)

                fig, curve = self.get_prior_model().variant_spectrum.plot_spectrum(torch.Tensor([1]), "Variant AF spectrum")
                summary_writer.add_figure("Variant AF spectrum", fig, epoch)

                for variant_type in utils.VariantType:
                    fig, curve = self.get_prior_model().artifact_spectra.plot_spectrum(variant_type.one_hot_tensor(),
                        variant_type.name + " artifact AF spectrum")
                    summary_writer.add_figure(variant_type.name + " artifact AF spectrum", fig, epoch)

    def learn_calibration(self, loader, num_epochs):
        self.train(False)
        utils.freeze(self.parameters())
        utils.unfreeze(self.calibration_parameters())

        # gather uncalibrated logits -- everything computed by the frozen part of the model -- so that we only
        # do forward and backward passes on the calibration submodule
        print("Computing uncalibrated part of model. . .")
        uncalibrated_logits_ref_alt_counts_labels = []
        pbar = tqdm(enumerate(loader), mininterval=10)
        for n, batch in pbar:
            if not batch.is_labeled():
                continue
            phi_reads = self.apply_phi_to_reads(batch)
            logits, ref_counts, alt_counts = self.forward_from_phi_reads_to_calibration(phi_reads, batch)
            uncalibrated_logits_ref_alt_counts_labels.append((logits.detach(), ref_counts.detach(), alt_counts.detach(), batch.labels().to(self._device)))

        print("Training calibration. . .")
        optimizer = torch.optim.Adam(self.calibration_parameters())
        bce = nn.BCEWithLogitsLoss()
        for epoch in trange(1, num_epochs + 1, desc="Calibration epoch"):
            nll_loss = utils.StreamingAverage(device=self._device)

            pbar = tqdm(enumerate(uncalibrated_logits_ref_alt_counts_labels), mininterval=10)
            for n, (logits, ref_counts, alt_counts, labels) in pbar:
                pred = self.calibration.forward(logits, ref_counts, alt_counts)

                loss = bce(pred, labels)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                nll_loss.record_sum(loss.detach(), len(logits))

    def calculate_logit_threshold(self, loader, summary_writer: SummaryWriter = None):
        self.train(False)
        artifact_probs = []

        print("running model over all data in loader to optimize F score")
        pbar = tqdm(enumerate(loader), mininterval=10)
        for n, batch in pbar:
            artifact_probs.extend(torch.sigmoid(self.forward(batch, posterior=True)).tolist())

        artifact_probs.sort()
        total_variants = len(artifact_probs) - sum(artifact_probs)

        # start by rejecting everything, then raise threshold one datum at a time
        threshold, tp, fp, best_f = 0.0, 0, 0, 0

        sens, prec = [], []
        for prob in artifact_probs:
            tp += (1 - prob)
            fp += prob
            sens.append(tp/(total_variants+0.0001))
            prec.append(tp/(tp+fp+0.0001))
            current_f = utils.f_score(tp, fp, total_variants)

            if current_f > best_f:
                best_f = current_f
                threshold = prob

        if summary_writer is not None:
            x_y_lab = [(sens, prec, "theoretical ROC curve according to M3's posterior probabilities")]
            fig, curve = plotting.simple_plot(x_y_lab, x_label="sensitivity", y_label="precision",
                                              title="theoretical ROC curve according to M3's posterior probabilities")
            summary_writer.add_figure("theoretical ROC curve", fig)

        return torch.logit(torch.tensor(threshold)).item()

    def train_model(self, train_loader, valid_loader, num_epochs, summary_writer: SummaryWriter, reweighting_range: float, m3_params: ArtifactModelParameters):
        bce = torch.nn.BCEWithLogitsLoss(reduction='sum')
        train_optimizer = torch.optim.AdamW(self.training_parameters(), lr=m3_params.learning_rate)

        # balance training by weighting the loss function
        total_labeled = sum(batch.size() for batch in train_loader if batch.is_labeled())
        total_unlabeled = sum(batch.size() for batch in train_loader if not batch.is_labeled())
        labeled_to_unlabeled_ratio = total_labeled / total_unlabeled

        for epoch in trange(1, num_epochs + 1, desc="Epoch"):
            for epoch_type in [utils.EpochType.TRAIN, utils.EpochType.VALID]:
                self.set_epoch_type(epoch_type)
                loader = train_loader if epoch_type == utils.EpochType.TRAIN else valid_loader
                labeled_loss = utils.StreamingAverage(device=self._device)
                unlabeled_loss = utils.StreamingAverage(device=self._device)

                pbar = tqdm(enumerate(loader), mininterval=10)
                for n, batch in pbar:
                    phi_reads = self.apply_phi_to_reads(batch)

                    # beta is for downsampling data augmentation
                    orig_pred = self.forward_from_phi_reads(phi_reads, batch, posterior=False, weight_range=0)
                    aug1_pred = self.forward_from_phi_reads(phi_reads, batch, posterior=False, weight_range=reweighting_range)
                    aug2_pred = self.forward_from_phi_reads(phi_reads, batch, posterior=False, weight_range=reweighting_range)

                    if batch.is_labeled():
                        labels = batch.labels()
                        # labeled loss: cross entropy for original and both augmented copies
                        loss = bce(orig_pred, labels) + bce(aug1_pred, labels) + bce(aug2_pred, labels)
                        labeled_loss.record_sum(loss.detach(), batch.size())
                    else:
                        # unlabeled loss: consistency cross entropy between original and both augmented copies
                        loss1 = bce(aug1_pred, torch.sigmoid(orig_pred.detach()))
                        loss2 = bce(aug2_pred, torch.sigmoid(orig_pred.detach()))
                        loss3 = bce(aug1_pred, torch.sigmoid(aug2_pred.detach()))
                        loss = (loss1 + loss2 + loss3) * labeled_to_unlabeled_ratio
                        unlabeled_loss.record_sum(loss.detach(), batch.size())

                    if epoch_type == utils.EpochType.TRAIN:
                        train_optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        train_optimizer.step()

                # done with one epoch type -- training or validation -- for this epoch
                summary_writer.add_scalar(epoch_type.name + "/Labeled Loss", labeled_loss.get(), epoch)
                summary_writer.add_scalar(epoch_type.name + "/Unlabeled Loss", unlabeled_loss.get(), epoch)
            # done with training and validation for this epoch
            # note that we have not learned the AF spectrum yet
        # done with training

    def evaluate_model_after_training(self, loader, summary_writer: SummaryWriter):
        self.freeze_all()
        self.cpu()
        self._device = "cpu"

        variant_sensitivity = utils.StreamingAverage()
        artifact_sensitivity = utils.StreamingAverage()
        high_conf_artifact_accuracy = utils.StreamingAverage()
        high_conf_variant_accuracy = utils.StreamingAverage()
        med_conf_variant_accuracy = utils.StreamingAverage()
        med_conf_artifact_accuracy = utils.StreamingAverage()
        unsure_accuracy = utils.StreamingAverage()

        pbar = tqdm(enumerate(loader), mininterval=10)
        for n, batch in pbar:
            if not batch.is_labeled():
                continue
            pred = self.forward(batch, posterior=False)
            labels = batch.labels()
            artifact_sensitivity.record_with_mask(pred > 0, labels > 0.5)
            variant_sensitivity.record_with_mask(pred < 0, labels < 0.5)

            correct = ((pred > 0) == (batch.labels() > 0.5))

            high_conf_artifact_accuracy.record_with_mask(correct, pred > 4)
            high_conf_variant_accuracy.record_with_mask(correct, pred < -4)
            med_conf_variant_accuracy.record_with_mask(correct, (pred < -1) & (pred > -4))
            med_conf_artifact_accuracy.record_with_mask(correct, (pred > 1) & (pred < 4))
            unsure_accuracy.record_with_mask(correct, (pred > -1) & (pred < 1))

        summary_writer.add_scalar("Variant Sensitivity", variant_sensitivity.get())
        summary_writer.add_scalar("Artifact Sensitivity", artifact_sensitivity.get())
        summary_writer.add_scalar("high-confidence artifact accuracy", high_conf_artifact_accuracy.get())
        summary_writer.add_scalar("high-confidence variant accuracy", high_conf_variant_accuracy.get())
        summary_writer.add_scalar("med-confidence artifact accuracy", med_conf_artifact_accuracy.get())
        summary_writer.add_scalar("med-confidence variant accuracy", med_conf_variant_accuracy.get())
        summary_writer.add_scalar("unsure accuracy", unsure_accuracy.get())




