import torch
from matplotlib.backends.backend_pdf import PdfPages
from torch import nn
from tqdm.notebook import trange, tqdm
# bug before PyTorch 1.7.1 that warns when constructing ParameterList
import warnings

from mutect3 import utils, validation
from mutect3.architecture.mlp import MLP
from mutect3.architecture.normal_artifact_model import NormalArtifactModel
from mutect3.architecture.prior_model import PriorModel
from mutect3.data.read_set_batch import ReadSetBatch
from mutect3.data.read_set_datum import NUM_READ_FEATURES, NUM_INFO_FEATURES
from mutect3.utils import freeze, unfreeze, f_score

warnings.filterwarnings("ignore", message="Setting attributes on ParameterList is not supported.")

class Mutect3Parameters:
    def __init__(self, hidden_read_layers, hidden_info_layers, aggregation_layers, output_layers, dropout_p):
        self.hidden_read_layers = hidden_read_layers
        self.hidden_info_layers = hidden_info_layers
        self.aggregation_layers = aggregation_layers
        self.output_layers = output_layers
        self.dropout_p = dropout_p


class Calibration(nn.Module):
    MAX_ALT = 10

    def __init__(self):
        super(Calibration, self).__init__()
        # Taking the mean of read embeddings erases the alt count, by design.  However, it is fine to multiply
        # the output logits in a count-dependent way to modulate confidence.  We initialize as sqrt(alt count).
        self.confidence = nn.Parameter(torch.sqrt(torch.arange(0, Calibration.MAX_ALT + 1).float()))

        # we apply as asymptotic threshold function logit --> M * tanh(logit/M) where M is the maximum absolute
        # value of the thresholded output.  For logits << M this is the identity, and approaching M the asymptote
        # gradually turns on.  This is a continuous way to truncate the model's confidence and is part of calibration.
        # We initialize it to something large.
        self.max_logit = nn.Parameter(torch.tensor(10.0))

    def forward(self, logits, alt_counts):
        truncated_counts = torch.LongTensor([min(c, Calibration.MAX_ALT) for c in alt_counts])
        logits = logits * torch.index_select(self.confidence, 0, truncated_counts)
        return self.max_logit * torch.tanh(logits / self.max_logit)


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

    def __init__(self, m3_params: Mutect3Parameters, na_model: NormalArtifactModel):
        super(ReadSetClassifier, self).__init__()

        # phi is the read embedding
        read_layers = [NUM_READ_FEATURES] + m3_params.hidden_read_layers
        self.phi = MLP(read_layers, batch_normalize=False, dropout_p=m3_params.dropout_p)

        # omega is the universal embedding of info field variant-level data
        info_layers = [NUM_INFO_FEATURES] + m3_params.hidden_info_layers
        self.omega = MLP(info_layers, batch_normalize=False, dropout_p=m3_params.dropout_p)

        # rho is the universal aggregation function
        ref_alt_info_embedding_dimension = 2 * read_layers[-1] + info_layers[-1]
        self.rho = MLP([ref_alt_info_embedding_dimension] + m3_params.aggregation_layers, batch_normalize=False,
                       dropout_p=m3_params.dropout_p)

        # We probably don't need dropout for the final output layers
        output_layers_sizes = [m3_params.aggregation_layers[-1]] + m3_params.output_layers + [1]
        self.outputs = nn.ModuleList(MLP(output_layers_sizes) for _ in utils.VariantType)

        self.calibration = Calibration()

        self.prior_model = PriorModel(4.0)

        self.normal_artifact_model = na_model
        freeze(self.normal_artifact_model.parameters())

    def get_prior_model(self):
        return self.prior_model

    def training_parameters(self):
        result = []
        result.extend(self.phi.parameters())
        result.extend(self.omega.parameters())
        result.extend(self.rho.parameters())
        result.extend(self.outputs.parameters())
        return result

    def calibration_parameters(self):
        return [self.calibration.max_logit]

    def spectra_parameters(self):
        return self.prior_model.parameters()

    def freeze_all(self):
        freeze(self.parameters())

    def set_epoch_type(self, epoch_type: utils.EpochType):
        if epoch_type == utils.EpochType.TRAIN:
            self.training_mode()
        else:
            self.freeze_all()

    def training_mode(self):
        self.train(True)
        freeze(self.parameters())
        unfreeze(self.training_parameters())
        unfreeze(self.calibration_parameters())

    def learn_calibration_mode(self):
        self.train(False)
        freeze(self.parameters())
        unfreeze(self.calibration_parameters())

    def learn_spectrum_mode(self):
        self.train(False)
        freeze(self.parameters())
        unfreeze(self.spectra_parameters())

    def forward(self, batch: ReadSetBatch, calibrated=True, posterior=False, normal_artifact=False):
        # embed reads and take mean within each datum to get tensors of shape (batch size x embedding dimension)
        phi_reads = torch.sigmoid(self.phi(batch.reads()))
        ref_means = torch.cat([torch.mean(phi_reads[s], dim=0, keepdim=True) for s in batch.ref_slices()], dim=0)
        alt_means = torch.cat([torch.mean(phi_reads[s], dim=0, keepdim=True) for s in batch.alt_slices()], dim=0)

        # stack side-by-side to get 2D tensor, where each variant row is (ref mean, alt mean, info)
        omega_info = torch.sigmoid(self.omega(batch.info()))
        concatenated = torch.cat((ref_means, alt_means, omega_info), dim=1)

        aggregated = self.rho(concatenated)

        logits = torch.zeros(batch.size())
        for variant_type in utils.VariantType:
            # It's slightly wasteful to compute output for every variant type when we only
            # use one, but this is such a small part of the model and it lets us use batches of mixed variant types
            output = torch.squeeze(self.outputs[variant_type.value](aggregated))
            mask = torch.tensor(
                [1 if variant_type == v_type else 0 for v_type in batch.variant_type()])
            logits += mask * output

        if calibrated:
            logits = self.calibration(logits, batch.alt_counts())

        if posterior:
            logits = self.prior_model(logits, batch)

            if normal_artifact:
                # NORMAL ARTIFACT CALCULATION BEGINS

                # posterior probability of normal artifact given observed read counts
                # log likelihood of tumor read counts given tumor variant spectrum P(tumor counts | somatic variant)
                somatic_log_lk = self.prior_model.variant_spectrum.log_likelihood(batch.pd_tumor_alt_counts(), batch.pd_tumor_depths())

                # log likelihood of tumor read counts given the normal read counts under normal artifact sub-model
                # P(tumor counts | normal counts)
                na_log_lk = self.normal_artifact_model.log_likelihood(batch.normal_artifact_batch())

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
                na_mask = torch.tensor([1 if count > 0 else 0 for count in batch.normal_artifact_batch().normal_alt()])
                na_masked_logits = na_mask * na_logits - 100 * (
                            1 - na_mask)  # if no normal alt counts, turn off normal artifact

                # primitive approach -- just take whichever is greater between the two models' posteriors
                # WARNING -- commenting out the line below completely disables normal artifact filtering!!!
                logits = torch.maximum(logits, na_masked_logits)

        return logits

    def learn_spectra(self, loader, num_epochs):
        self.learn_spectrum_mode()

        logits_and_batches = [(self(batch).detach(), batch) for batch in loader]
        optimizer = torch.optim.Adam(self.spectra_parameters())

        spectra_losses = []
        epochs = []
        pbar = trange(num_epochs, desc="AF spectra epoch")
        for epoch in pbar:
            epochs.append(epoch + 1)
            epoch_loss = 0
            for logits, batch in logits_and_batches:
                loss = -torch.mean(self.prior_model.log_evidence(logits, batch))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            spectra_losses.append(epoch_loss)
            # check for convergence
            delta = 0 if epoch < 2 else abs(epoch_loss - spectra_losses[-2])
            total_delta = abs(epoch_loss - spectra_losses[0])
            if epoch > 5 and delta < 0.001 * total_delta:
                break

        # TODO: this needs to go in the report output
        validation.simple_plot([(epochs, spectra_losses, "loss")], "epoch", "loss", "AF Learning curve")

    # calculate and detach the likelihoods layers, then learn the calibration layer with SGD
    def learn_calibration(self, loader, num_epochs):
        uncalibrated_logits_labels_counts = [
            (self(batch, calibrated=False).detach(), batch.labels(),
             batch.alt_counts()) for batch in loader if batch.is_labeled()]
        optimizer = torch.optim.Adam(self.calibration_parameters())
        criterion = nn.BCEWithLogitsLoss()

        self.learn_calibration_mode()
        calibration_losses = []

        for epoch in range(num_epochs):
            epoch_loss = 0
            for logits, labels, counts in uncalibrated_logits_labels_counts:
                calibrated_logits = self.calibration(logits, counts)
                loss = criterion(calibrated_logits, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            calibration_losses.append(epoch_loss)

        epochs = range(1, num_epochs + 1)

        # TODO: this needs to go in the report output
        validation.simple_plot([(epochs, calibration_losses, "curve")], "epoch", "loss",
                               "Learning curve for calibration")

    def calculate_logit_threshold(self, loader, normal_artifact=False, roc_plot=None):
        self.train(False)
        artifact_probs = []

        print("running model over all data in loader to optimize F score")
        pbar = tqdm(enumerate(loader))
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

        if roc_plot is not None:
            x_y_lab = [(sens, prec, "theoretical ROC curve according to M3's posterior probabilities")]
            fig, curve = validation.simple_plot(x_y_lab, xlabel="sensitivity", ylabel="precision",
                                                title="theoretical ROC curve according to M3's posterior probabilities")
            with PdfPages(roc_plot) as pdf:
                pdf.savefig(fig)
        return torch.logit(torch.tensor(threshold)).item()

    def train_model(self, train_loader, valid_loader, num_epochs, beta1, beta2):
        bce = torch.nn.BCEWithLogitsLoss(reduction='sum')
        train_optimizer = torch.optim.Adam(self.training_parameters())
        training_metrics = validation.TrainingMetrics()

        # balance training by weighting the loss function
        total_labeled = sum(batch.size() for batch in train_loader if batch.is_labeled())
        total_unlabeled = sum(batch.size() for batch in train_loader if not batch.is_labeled())
        labeled_to_unlabeled_ratio = total_labeled / total_unlabeled

        for _ in trange(1, num_epochs + 1, desc="Epoch"):
            for epoch_type in [utils.EpochType.TRAIN, utils.EpochType.VALID]:
                self.set_epoch_type(epoch_type)
                loader = train_loader if epoch_type == utils.EpochType.TRAIN else valid_loader

                epoch_labeled_loss, epoch_unlabeled_loss = 0, 0
                epoch_labeled_count, epoch_unlabeled_count = 0, 0

                pbar = tqdm(enumerate(loader))
                for n, batch in pbar:
                    orig_pred = self(batch)
                    aug1_pred = self(batch.augmented_copy(beta1))
                    aug2_pred = self(batch.augmented_copy(beta2))

                    if batch.is_labeled():
                        labels = batch.labels()
                        # labeled loss: cross entropy for original and both augmented copies
                        loss = bce(orig_pred, labels) + bce(aug1_pred, labels) + bce(aug2_pred, labels)
                        epoch_labeled_count += batch.size()
                        epoch_labeled_loss += loss.item()
                    else:
                        # unlabeled loss: consistency cross entropy between original and both augmented copies
                        loss1 = bce(aug1_pred, torch.sigmoid(orig_pred.detach()))
                        loss2 = bce(aug2_pred, torch.sigmoid(orig_pred.detach()))
                        loss3 = bce(aug1_pred, torch.sigmoid(aug2_pred.detach()))
                        loss = (loss1 + loss2 + loss3) * labeled_to_unlabeled_ratio
                        epoch_unlabeled_count += batch.size()
                        epoch_unlabeled_loss += loss.item()

                    if epoch_type == utils.EpochType.TRAIN:
                        train_optimizer.zero_grad()
                        loss.backward()
                        train_optimizer.step()

                training_metrics.add("labeled NLL", epoch_type.name, epoch_labeled_loss / epoch_labeled_count)
                training_metrics.add("unlabeled NLL", epoch_type.name, epoch_unlabeled_loss / epoch_unlabeled_count)

            # done with training and validation for this epoch
            # note that we have not learned the AF spectrum yet
        # done with training
        self.learn_calibration(valid_loader, num_epochs=200)
        return training_metrics