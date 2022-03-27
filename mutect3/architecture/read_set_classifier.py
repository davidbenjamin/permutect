# bug before PyTorch 1.7.1 that warns when constructing ParameterList
import warnings

import torch
from matplotlib.backends.backend_pdf import PdfPages
from torch import nn
from tqdm.autonotebook import trange, tqdm
from itertools import chain

import mutect3.metrics.plotting
from mutect3.metrics.metrics import LearningCurves
from mutect3 import utils
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
        return chain(self.phi.parameters(), self.omega.parameters(), self.rho.parameters(), self.outputs.parameters(), [self.calibration.max_logit])

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

        logits_and_batches = [(self.forward(batch).detach(), batch) for batch in loader]
        optimizer = torch.optim.Adam(self.spectra_parameters())

        spectra_learning_curve = LearningCurves()
        pbar = trange(num_epochs, desc="AF spectra epoch")
        for epoch in pbar:
            epoch_loss = 0
            epoch_count = 0
            for logits, batch in logits_and_batches:
                loss = -torch.mean(self.prior_model.log_evidence(logits, batch))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                epoch_count += batch.size()

            # TODO: learning curves for different alt counts
            spectra_learning_curve.add("spectrum NLL", epoch_loss / epoch_count)

        return spectra_learning_curve

    def learn_calibration(self, loader, num_epochs):
        self.train(False)
        freeze(self.parameters())
        unfreeze(self.calibration_parameters())

        optimizer = torch.optim.Adam(self.calibration_parameters())
        bce = nn.BCEWithLogitsLoss()
        calibration_learning_curve = LearningCurves()

        for _ in trange(1, num_epochs + 1, desc="Epoch"):
            epoch_loss = 0
            epoch_count = 0

            # how many artifacts were predicted w/ high confidence, and how many of those predictions were correct
            # similarly for variants
            high_conf_artifact_pred, high_conf_artifact_correct = 0, 0
            high_conf_variant_pred, high_conf_variant_correct = 0, 0
            med_conf_artifact_pred, med_conf_artifact_correct = 0, 0
            med_conf_variant_pred, med_conf_variant_correct = 0, 0
            unsure_pred, unsure_correct = 0, 0
            pbar = tqdm(enumerate(loader))
            for n, batch in pbar:
                if not batch.is_labeled():
                    continue
                epoch_count += batch.size()
                pred = self.forward(batch)
                loss = bce(pred, batch.labels())
                epoch_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                high_conf_artifact = pred > 4
                high_conf_variant = pred < -4
                med_conf_artifact = (pred > 1) & (pred < 4)
                med_conf_variant = (pred < -1) & (pred > -4)
                unsure = (pred > -1) & (pred < 1)
                correct = (pred > 0) == (batch.labels() > 0.5)

                high_conf_artifact_pred += torch.sum(high_conf_artifact).item()
                high_conf_artifact_correct += torch.sum(high_conf_artifact & correct).item()
                high_conf_variant_pred += torch.sum(high_conf_variant).item()
                high_conf_variant_correct += torch.sum(high_conf_variant & correct).item()

                med_conf_artifact_pred += torch.sum(med_conf_artifact).item()
                med_conf_artifact_correct += torch.sum(med_conf_artifact & correct).item()
                med_conf_variant_pred += torch.sum(med_conf_variant).item()
                med_conf_variant_correct += torch.sum(med_conf_variant & correct).item()

                unsure_pred += torch.sum(unsure).item()
                unsure_correct += torch.sum(unsure & correct).item()

            calibration_learning_curve.add("calibration NLL", epoch_loss / epoch_count)
            calibration_learning_curve.add("high-confidence artifact accuracy", high_conf_artifact_correct / high_conf_artifact_pred)
            calibration_learning_curve.add("high-confidence variant accuracy", high_conf_variant_correct / high_conf_variant_pred)
            calibration_learning_curve.add("med-confidence artifact accuracy", med_conf_artifact_correct / med_conf_artifact_pred)
            calibration_learning_curve.add("med-confidence variant accuracy", med_conf_variant_correct / med_conf_variant_pred)
            calibration_learning_curve.add("unsure accuracy", unsure_correct / unsure_pred)
            
        return calibration_learning_curve

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
            fig, curve = mutect3.metrics.plotting.simple_plot(x_y_lab, x_label="sensitivity", y_label="precision",
                                                              title="theoretical ROC curve according to M3's posterior probabilities")
            with PdfPages(roc_plot) as pdf:
                pdf.savefig(fig)
        return torch.logit(torch.tensor(threshold)).item()

    def train_model(self, train_loader, valid_loader, num_epochs, beta1, beta2):
        bce = torch.nn.BCEWithLogitsLoss(reduction='sum')
        individual_bce = torch.nn.BCEWithLogitsLoss(reduction='none')
        train_optimizer = torch.optim.Adam(self.training_parameters())
        learning_curves = LearningCurves()

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

                # simple stratification on alt count
                epoch_less_than_five_loss, epoch_more_than_ten_loss = 0, 0
                epoch_less_than_five_count, epoch_more_than_ten_count = 0, 0

                # row/index 0 = label/index 1, column = prediction
                # 0 = variant, 1 = artifact
                # thus confusion[1][0] = count of variants classified as artifact
                epoch_confusion_matrix = [[0, 0], [0, 0]]

                pbar = tqdm(enumerate(loader))
                for n, batch in pbar:
                    orig_pred = self.forward(batch)
                    aug1_pred = self.forward(batch.augmented_copy(beta1))
                    aug2_pred = self.forward(batch.augmented_copy(beta2))

                    if batch.is_labeled():
                        labels = batch.labels()
                        # labeled loss: cross entropy for original and both augmented copies
                        loss = bce(orig_pred, labels) + bce(aug1_pred, labels) + bce(aug2_pred, labels)
                        epoch_labeled_count += batch.size()
                        epoch_labeled_loss += loss.item()

                        # convert predictions to 0/1 variant/artifact
                        binary_pred = (orig_pred > 0).int().tolist()
                        for label, pred in zip((labels > 0.5).int().tolist(), binary_pred):
                            epoch_confusion_matrix[label][pred] += 1

                        individual_loss = individual_bce(orig_pred, labels)
                        less_than_five = batch.alt_counts() < 5
                        more_than_ten = batch.alt_counts() > 10
                        epoch_less_than_five_count += torch.sum(less_than_five).item()
                        epoch_more_than_ten_count += torch.sum(more_than_ten).item()
                        epoch_less_than_five_loss += torch.sum(less_than_five * individual_loss).item()
                        epoch_more_than_ten_loss += torch.sum(more_than_ten * individual_loss).item()

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

                learning_curves.add(epoch_type.name + " labeled NLL", epoch_labeled_loss / epoch_labeled_count)
                learning_curves.add(epoch_type.name + " less than 5 alt labeled NLL", epoch_less_than_five_loss / epoch_less_than_five_count)
                learning_curves.add(epoch_type.name + " more than 10 alt labeled NLL", epoch_more_than_ten_loss / epoch_more_than_ten_count)
                learning_curves.add(epoch_type.name + " unlabeled NLL", epoch_unlabeled_loss / epoch_unlabeled_count)
                learning_curves.add(epoch_type.name + " variant accuracy", epoch_confusion_matrix[0][0] / (epoch_confusion_matrix[0][0]+epoch_confusion_matrix[0][1]))
                learning_curves.add(epoch_type.name + " artifact accuracy", epoch_confusion_matrix[1][1] / (epoch_confusion_matrix[1][0] + epoch_confusion_matrix[1][1]))

            # done with training and validation for this epoch
            # note that we have not learned the AF spectrum yet
        # done with training
        return learning_curves
