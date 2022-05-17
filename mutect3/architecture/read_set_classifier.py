# bug before PyTorch 1.7.1 that warns when constructing ParameterList
import warnings

import torch
from torch.utils.tensorboard import SummaryWriter

from torch import nn
from tqdm.autonotebook import trange, tqdm
from itertools import chain

from mutect3 import utils
from mutect3.architecture.mlp import MLP
from mutect3.architecture.normal_artifact_model import NormalArtifactModel
from mutect3.architecture.prior_model import PriorModel
from mutect3.data.read_set_batch import ReadSetBatch
from mutect3.data.read_set_datum import NUM_READ_FEATURES, NUM_INFO_FEATURES
from mutect3.utils import freeze, unfreeze, f_score, StreamingAverage
from mutect3.metrics import plotting

warnings.filterwarnings("ignore", message="Setting attributes on ParameterList is not supported.")


def effective_count(weights: torch.Tensor):
    return (torch.square(torch.sum(weights)) / torch.sum(torch.square(weights))).item()


# note that read layers and info layers exclude the input dimension
class Mutect3Parameters:
    def __init__(self, read_layers, info_layers, aggregation_layers, dropout_p):
        self.read_layers = read_layers
        self.info_layers = info_layers
        self.aggregation_layers = aggregation_layers
        self.dropout_p = dropout_p


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


class ReadSetClassifier(nn.Module):
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

    def __init__(self, m3_params: Mutect3Parameters, na_model: NormalArtifactModel, device=torch.device("cpu")):
        super(ReadSetClassifier, self).__init__()

        self._device = device

        # phi is the read embedding
        read_layers = [NUM_READ_FEATURES] + m3_params.read_layers
        self.phi = MLP(read_layers, batch_normalize=False, dropout_p=m3_params.dropout_p)
        self.phi.to(self._device)

        # omega is the universal embedding of info field variant-level data
        info_layers = [NUM_INFO_FEATURES] + m3_params.info_layers
        self.omega = MLP(info_layers, batch_normalize=False, dropout_p=m3_params.dropout_p)
        self.omega.to(self._device)

        # rho is the universal aggregation function
        ref_alt_info_embedding_dimension = 2 * read_layers[-1] + info_layers[-1]

        # The [1] is for the output of binary classification, represented as a single artifact/non-artifact logit
        self.rho = MLP([ref_alt_info_embedding_dimension] + m3_params.aggregation_layers + [1], batch_normalize=False,
                       dropout_p=m3_params.dropout_p)
        self.rho.to(self._device)

        self.calibration = Calibration()
        self.calibration.to(self._device)

        # note: prior model and normal artifact model are not part of training and thus are not ever put on GPU
        self.prior_model = PriorModel(0.0)

        self.normal_artifact_model = na_model
        if na_model is not None:
            freeze(self.normal_artifact_model.parameters())

    def set_normal_artifact_model(self, na_model: NormalArtifactModel):
        self.normal_artifact_model = na_model

    def get_prior_model(self):
        return self.prior_model

    def training_parameters(self):
        return chain(self.phi.parameters(), self.omega.parameters(), self.rho.parameters(), [self.calibration.max_logit])

    def calibration_parameters(self):
        return self.calibration.parameters()

    def spectra_parameters(self):
        return self.prior_model.parameters()

    def freeze_all(self):
        freeze(self.parameters())

    def set_epoch_type(self, epoch_type: utils.EpochType):
        if epoch_type == utils.EpochType.TRAIN:
            self.train(True)
            freeze(self.parameters())
            unfreeze(self.training_parameters())
        else:
            self.freeze_all()

    def learn_spectrum_mode(self):
        self.train(False)
        freeze(self.parameters())
        unfreeze(self.spectra_parameters())

    def forward(self, batch: ReadSetBatch, posterior=False, normal_artifact=False):
        phi_reads = self.apply_phi_to_reads(batch)
        return self.forward_from_phi_reads(phi_reads=phi_reads, batch=batch, posterior=posterior, normal_artifact=normal_artifact)

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
    def forward_from_phi_reads(self, phi_reads: torch.Tensor, batch: ReadSetBatch, posterior=False, normal_artifact=False, weight_range: float=0):
        logits, effective_ref_counts, effective_alt_counts =  self.forward_from_phi_reads_to_calibration(phi_reads, batch, weight_range)
        logits = self.calibration.forward(logits, effective_ref_counts, effective_alt_counts)

        if posterior:
            # the prior model is always on CPU, but if we are computing a posterior it's not training and hence we are not
            # using a GPU.  In principle we should put logits ont he CPU just in case, but we can be brittle for now
            # pending some refactoring.  The same applies to the normal artifact model.
            logits = self.prior_model(logits, batch)

            if normal_artifact:
                # NORMAL ARTIFACT CALCULATION BEGINS

                # posterior probability of normal artifact given observed read counts
                # log likelihood of tumor read counts given tumor variant spectrum P(tumor counts | somatic variant)
                somatic_log_lk = self.prior_model.variant_spectrum.log_likelihood(batch.pd_tumor_alt_counts(), batch.pd_tumor_depths())

                # log likelihood of tumor read counts given the normal read counts under normal artifact sub-model
                # P(tumor counts | normal counts)
                na_batch = batch.normal_artifact_batch()
                na_log_lk = self.normal_artifact_model.log_likelihood(na_batch)

                # note that prior of normal artifact is essentially 1
                # posterior is P(artifact) = P(tumor counts | normal counts) /[P(tumor counts | normal) + P(somatic)*P(tumor counts | somatic)]
                # and posterior logits are log(post prob artifact / post prob somatic)
                # so, with n_ll = normal artifact log likelihood and som_ll = somatic log likelihood and pi = log P(somatic)
                # posterior logit = na_ll - pi - som_ll

                # TODO: WARNING: HARD-CODED MAGIC CONSTANT!!!!!
                log_somatic_prior = -11.0
                na_logits = na_log_lk - log_somatic_prior - somatic_log_lk
                # NORMAL ARTIFACT CALCULATION ENDS

                # normal artifact model is only trained and only applies when normal alt counts are non-zero
                na_mask = torch.tensor([1 if count > 0 else 0 for count in na_batch.normal_alt()])
                na_masked_logits = na_mask * na_logits - 100 * (
                            1 - na_mask)  # if no normal alt counts, turn off normal artifact

                # primitive approach -- just take whichever is greater between the two models' posteriors
                # WARNING -- commenting out the line below completely disables normal artifact filtering!!!
                logits = torch.maximum(logits, na_masked_logits)

        return logits

    def learn_spectra(self, loader, num_iterations, use_normal_artifact=False, summary_writer: SummaryWriter = None):
        self.learn_spectrum_mode()
        logits_and_batches = [(self.forward(batch=batch, normal_artifact=use_normal_artifact).detach(), batch) for batch in loader]
        optimizer = torch.optim.Adam(self.spectra_parameters())

        overall_epoch = 0
        num_fit_epochs = 10
        for iteration in trange(num_iterations, desc="AF spectra iteration"):
            tumor_afs, all_artifact_probs = [], []  # for 2D heatmap of tumor_af vs artifact prob
            probs_and_batches = []
            variant_afs = []   # debugging -- record which data are variant
            artifact_afs = []
            for logits, batch in logits_and_batches:
                posterior_logits = self.prior_model.forward(logits, batch).detach()

                # TODO: DON'T KEEP THIS DEBUG LINE WHERE WE USE THE LIKELIHOODS INSTEAD OF THE POSTERIORS!!!
                posterior_probs = torch.sigmoid(logits).detach()
                # TODO: ORIGINAL posterior_probs = torch.sigmoid(posterior_logits)
                probs_and_batches.append((posterior_probs, batch))

                data_list = batch.original_list()
                for n in range(batch.size()):
                    datum = data_list[n]
                    artifact_prob = posterior_probs[n].item()
                    tumor_af = datum.tumor_alt_count() / datum.tumor_depth()
                    tumor_afs.append(tumor_af)
                    all_artifact_probs.append(artifact_prob)

                    if artifact_prob < 0.5:
                        variant_afs.append(tumor_af)
                    else:
                        artifact_afs.append(tumor_af)
            # done computing posterior probs (the E step) of this iteration

            iter_artifacts, iter_variants, iter_sure_variants, iter_sure_artifacts = 0.0, 0.0, 0, 0

            fit_pbar = trange(num_fit_epochs, desc="AF fitting epoch")
            for epoch in fit_pbar:
                overall_epoch += 1
                epoch_loss = StreamingAverage()

                for probs, batch in probs_and_batches:
                    variant_probs = (1 - probs)*(probs < 0.2)
                    artifact_probs = probs * (probs > 0.8)

                    variant_loss = -variant_probs * self.prior_model.variant_log_likelihoods(batch)
                    artifact_loss = -artifact_probs * self.prior_model.artifact_log_likelihoods(batch, include_prior=False)

                    artifact_log_odds = self.prior_model.artifact_log_priors(batch)
                    artifact_log_priors = -torch.log1p(torch.exp(-artifact_log_odds))
                    variant_log_priors = -torch.log1p(torch.exp(artifact_log_odds))
                    prior_loss = -probs * artifact_log_priors - (1 - probs) * variant_log_priors

                    loss = torch.mean(variant_loss) + torch.mean(artifact_loss) + torch.mean(prior_loss)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    epoch_loss.record_sum(loss.detach(), batch.size())
                    # these depend on the iteration, not the inner epoch loop, and so are only computed in the last epoch
                    if summary_writer is not None and epoch == num_fit_epochs - 1:
                        iter_artifacts += torch.sum(probs).item()
                        iter_variants += torch.sum(1 - probs).item()
                        iter_sure_artifacts += torch.sum(probs > 0.99).item()
                        iter_sure_variants += torch.sum(probs < 0.01).item()
                # done with batches in this inner epoch
                summary_writer.add_scalar("spectrum NLL", epoch_loss.get(), overall_epoch)
            # done with all inner (fit) epochs
            if summary_writer is not None:
                summary_writer.add_scalar("SNV log prior", self.prior_model.prior_log_odds[0].item(), iteration)
                summary_writer.add_scalar("Insertion log prior", self.prior_model.prior_log_odds[1].item(), iteration)
                summary_writer.add_scalar("Deletion log prior", self.prior_model.prior_log_odds[2].item(), iteration)
                summary_writer.add_scalar("artifact count", iter_artifacts, iteration)
                summary_writer.add_scalar("variant count", iter_variants, iteration)
                summary_writer.add_scalar("certain artifact count", iter_sure_artifacts, iteration)
                summary_writer.add_scalar("certain variant count", iter_sure_variants, iteration)

                for variant_type in utils.VariantType:
                    fig, curve = self.get_prior_model().artifact_spectra[variant_type.value].plot_spectrum(
                        variant_type.name + " artifact AF spectrum")
                    summary_writer.add_figure(variant_type.name + " artifact AF spectrum", fig, iteration)
                fig, curve = self.get_prior_model().variant_spectrum.plot_spectrum("Variant AF spectrum")
                summary_writer.add_figure("Variant AF spectrum", fig, iteration)

                summary_writer.add_figure("Variant AFs", plotting.histogram(variant_afs, "AFs")[0], iteration)
                summary_writer.add_figure("Artifact AFs", plotting.histogram(artifact_afs, "AFs")[0], iteration)
                summary_writer.add_figure("AF vs prob", plotting.hexbin(tumor_afs, all_artifact_probs)[0], iteration)

    def learn_calibration(self, loader, num_epochs):
        self.train(False)
        freeze(self.parameters())
        unfreeze(self.calibration_parameters())

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
            uncalibrated_logits_ref_alt_counts_labels.append((logits.detach(), ref_counts.detach(), alt_counts.detach(), batch.labels.to(self._device)))

        print("Training calibration. . .")
        optimizer = torch.optim.Adam(self.calibration_parameters())
        bce = nn.BCEWithLogitsLoss()
        for epoch in trange(1, num_epochs + 1, desc="Calibration epoch"):
            nll_loss = StreamingAverage(device=self._device)

            pbar = tqdm(enumerate(uncalibrated_logits_ref_alt_counts_labels), mininterval=10)
            for n, logits_ref_alt_labels in pbar:
                logits, ref_counts, alt_counts, labels = logits_ref_alt_labels
                pred = self.calibration.forward(logits, ref_counts, alt_counts)

                loss = bce(pred, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                nll_loss.record_sum(loss.detach(), len(logits))

    def calculate_logit_threshold(self, loader, normal_artifact=False, summary_writer: SummaryWriter = None):
        self.train(False)
        artifact_probs = []

        print("running model over all data in loader to optimize F score")
        pbar = tqdm(enumerate(loader), mininterval=10)
        for n, batch in pbar:
            artifact_probs.extend(torch.sigmoid(self.forward(batch, posterior=True, normal_artifact=normal_artifact)).tolist())

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
            current_f = f_score(tp, fp, total_variants)

            if current_f > best_f:
                best_f = current_f
                threshold = prob

        if summary_writer is not None:
            x_y_lab = [(sens, prec, "theoretical ROC curve according to M3's posterior probabilities")]
            fig, curve = plotting.simple_plot(x_y_lab, x_label="sensitivity", y_label="precision",
                                              title="theoretical ROC curve according to M3's posterior probabilities")
            summary_writer.add_figure("theoretical ROC curve", fig)

        return torch.logit(torch.tensor(threshold)).item()

    def train_model(self, train_loader, valid_loader, num_epochs, summary_writer: SummaryWriter, reweighting_range: float):
        bce = torch.nn.BCEWithLogitsLoss(reduction='sum')
        train_optimizer = torch.optim.Adam(self.training_parameters())

        # balance training by weighting the loss function
        total_labeled = sum(batch.size() for batch in train_loader if batch.is_labeled())
        total_unlabeled = sum(batch.size() for batch in train_loader if not batch.is_labeled())
        labeled_to_unlabeled_ratio = total_labeled / total_unlabeled

        for epoch in trange(1, num_epochs + 1, desc="Epoch"):
            for epoch_type in [utils.EpochType.TRAIN, utils.EpochType.VALID]:
                self.set_epoch_type(epoch_type)
                loader = train_loader if epoch_type == utils.EpochType.TRAIN else valid_loader
                labeled_loss = StreamingAverage(device=self._device)
                unlabeled_loss = StreamingAverage(device=self._device)

                pbar = tqdm(enumerate(loader), mininterval=10)
                for n, batch in pbar:
                    phi_reads = self.apply_phi_to_reads(batch)

                    # beta is for downsampling data augmentation
                    orig_pred = self.forward_from_phi_reads(phi_reads, batch, posterior=False, normal_artifact=False, weight_range=0)
                    aug1_pred = self.forward_from_phi_reads(phi_reads, batch, posterior=False, normal_artifact=False, weight_range=reweighting_range)
                    aug2_pred = self.forward_from_phi_reads(phi_reads, batch, posterior=False, normal_artifact=False, weight_range=reweighting_range)

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
                        train_optimizer.zero_grad()
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

        variant_sensitivity = StreamingAverage()
        artifact_sensitivity = StreamingAverage()
        high_conf_artifact_accuracy = StreamingAverage()
        high_conf_variant_accuracy = StreamingAverage()
        med_conf_variant_accuracy = StreamingAverage()
        med_conf_artifact_accuracy = StreamingAverage()
        unsure_accuracy = StreamingAverage()

        pbar = tqdm(enumerate(loader), mininterval=10)
        for n, batch in pbar:
            if not batch.is_labeled():
                continue
            pred = self.forward(batch, posterior=False, normal_artifact=False)
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




