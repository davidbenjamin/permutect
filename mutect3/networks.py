import matplotlib.pyplot as plt
import torch
import enum
from enum import IntEnum
from torch import nn
from tqdm.autonotebook import trange

from mutect3 import validation, tensors, data

# bug before PyTorch 1.7.1 that warns when constructing ParameterList
import warnings
warnings.filterwarnings("ignore", message="Setting attributes on ParameterList is not supported.")


def freeze(parameters):
    for parameter in parameters:
        parameter.requires_grad = False


def unfreeze(parameters):
    for parameter in parameters:
        parameter.requires_grad = True


def f_score(tp, fp, total_true):
    fn = total_true - tp
    return tp / (tp + (fp + fn) / 2)

class EpochType(enum.Enum):
   TRAIN = "train"
   VALID = "valid"
   TEST = "test"

class VariantType(enum.IntEnum):
    SNV = 0
    INSERTION = 1
    DELETION = 2

    def is_same_type(self, site_info: tensors.SiteInfo):
        diff = len(site_info.alt()) - len(site_info.ref())
        return (self == VariantType.SNV and diff == 0) or (self == VariantType.INSERTION and diff > 0) \
               or (self == VariantType.DELETION and diff < 0)


class MLP(nn.Module):
    """
    A fully-connected network (multi-layer perceptron) that we need frequently
    as a sub-network.  It is parameterized by the dimensions of its layers, starting with
    the input layer and ending with the output.  Output is logits and as such no non-linearity
    is applied after the last linear transformation.
    """

    def __init__(self, layer_sizes, batch_normalize=False, dropout_p=None):
        super(MLP, self).__init__()

        self.layers = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.dropout = nn.ModuleList()
        for k in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[k], layer_sizes[k + 1]))

        if batch_normalize:
            for size in layer_sizes[1:]:
                self.bn.append(nn.BatchNorm1d(num_features=size))

        if dropout_p is not None:
            for n in layer_sizes[1:]:
                self.dropout.append(nn.Dropout(p=dropout_p))

    def forward(self, x):
        for n, layer in enumerate(self.layers):
            x = layer(x)
            if self.bn:
                x = self.bn[n](x)
            if self.dropout:
                x = self.dropout[n](x)
            if n < len(self.layers) - 1:
                x = nn.functional.leaky_relu(x)
        return x


# note: this function works for n, k, alpha, beta tensors of the same shape
# the result is computed element-wise ie result[i,j. . .] = beta_binomial(n[i,j..], k[i,j..], alpha[i,j..], beta[i,j..)
# often n, k will correspond to a batch dimension and alpha, beta correspond to a model, in which case
# unsqueezing is necessary
def beta_binomial(n, k, alpha, beta):
    return torch.lgamma(k + alpha) + torch.lgamma(n - k + beta) + torch.lgamma(alpha + beta) \
           - torch.lgamma(n + alpha + beta) - torch.lgamma(alpha) - torch.lgamma(beta)


class AFSpectrum(nn.Module):

    def __init__(self):
        super(AFSpectrum, self).__init__()
        # evenly-spaced beta binomials.  These are constants for now
        # TODO: should these be learned by making them Parameters?
        shapes = [(n + 1, 101 - n) for n in range(0, 100, 5)]
        self.a = torch.FloatTensor([shape[0] for shape in shapes])
        self.b = torch.FloatTensor([shape[1] for shape in shapes])

        # (pre-softmax) weights for the beta-binomial mixture.  Initialized as uniform.
        self.z = nn.Parameter(torch.ones(len(shapes)))

    # k "successes" out of n "trials" -- k and n are 1D tensors of the same size
    def log_likelihood(self, k, n):
        # note that we unsqueeze pi along the same dimension as alpha and beta
        log_pi = torch.unsqueeze(nn.functional.log_softmax(self.z, dim=0), 0)


        # n, k, a, b are all 1D tensors.  If we unsqueeze n,k along dim=1 and unsqueeze a,,b along dim=0
        # the resulting 2D log_likelihoods will have the structure
        # likelihoods[i,j] = beta_binomial(n[i],k[i],alpha[j],beta[j])
        log_likelihoods = beta_binomial(n.unsqueeze(1), k.unsqueeze(1), self.a.unsqueeze(0), self.b.unsqueeze(0))

        # by the convention above, the 0th dimension of log_likelihoods is n,k (batch) and the 1st dimension
        # is alpha, beta.  We sum over the latter, getting a 1D tensor corresponding to the batch
        return torch.logsumexp(log_pi + log_likelihoods, dim=1)

    # compute 1D tensor of log-likelihoods P(alt count|n, AF mixture model) over all data in batch
    def forward(self, batch: data.Batch):
        depths = torch.LongTensor([datum.tumor_depth() for datum in batch.mutect_info()])
        return self.log_likelihood(batch.alt_counts(), depths)

    # plot the mixture of beta densities
    def plot_spectrum(self, title):
        f = torch.arange(0.01, 0.99, 0.01)
        log_pi = nn.functional.log_softmax(self.z, dim=0)

        shapes = [(alpha, beta) for (alpha, beta) in zip(self.a.numpy(), self.b.numpy())]
        betas = [torch.distributions.beta.Beta(torch.FloatTensor([alpha]), torch.FloatTensor([beta])) for (alpha, beta)
                 in shapes]

        # list of tensors - the list is over the mixture components, the tensors are over AF values f
        unweighted_log_densities = torch.stack([beta.log_prob(f) for beta in betas], dim=0)
        # unsqueeze to make log_pi a column vector (2D tensor) for broadcasting
        weighted_log_densities = torch.unsqueeze(log_pi, 1) + unweighted_log_densities
        densities = torch.exp(torch.logsumexp(weighted_log_densities, dim=0))

        return validation.simple_plot([(f.detach().numpy(), densities.detach().numpy()," ")], "AF", "density", title)


# contains variant spectrum, artifact spectrum, and artifact/variant log prior ratio
class PriorModel(nn.Module):
    def __init__(self, initial_log_ratio=0.0):
        super(PriorModel, self).__init__()
        self.variant_spectrum = AFSpectrum()

        self.artifact_spectra = nn.ModuleList()
        self.prior_log_odds = nn.ParameterList() #log prior ratio log[P(artifact)/P(variant)] for each type
        for artifact_type in VariantType:
            self.artifact_spectra.append(AFSpectrum())
            self.prior_log_odds.append(nn.Parameter(torch.tensor(initial_log_ratio)))

    # forward pass returns posterior logits of being artifact given likelihood logits
    def forward(self, logits, batch):
        variant_ll = self.variant_spectrum(batch)

        result = torch.zeros_like(logits)
        for variant_type in VariantType:
            output = logits + self.prior_log_odds[variant_type.value] + self.artifact_spectra[variant_type.value](batch) - variant_ll
            mask = torch.tensor([1 if variant_type.is_same_type(site_info) else 0 for site_info in batch.site_info()])
            result += mask * output

        return result

    def plot_spectra(self):
        for variant_type in VariantType:
            self.artifact_spectra[variant_type.value].plot_spectrum(variant_type.name + " artifact AF spectrum")
        self.variant_spectrum.plot_spectrum("Variant AF spectrum")


class Mutect3Parameters:
    def __init__(self, hidden_read_layers, hidden_info_layers, aggregation_layers, output_layers, dropout_p):
        self.hidden_read_layers = hidden_read_layers
        self.hidden_info_layers = hidden_info_layers
        self.aggregation_layers = aggregation_layers
        self.output_layers = output_layers
        self.dropout_p = dropout_p


# calibrate the confidence of uncalibrated logits
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

    def __init__(self, m3_params: Mutect3Parameters, m2_filters_to_keep={}):
        super(ReadSetClassifier, self).__init__()

        # note: these are only used for testing, not training
        self.m2_filters_to_keep = m2_filters_to_keep

        # phi is the read embedding
        read_layers = [tensors.NUM_READ_FEATURES] + m3_params.hidden_read_layers
        self.phi = MLP(read_layers, batch_normalize=False, dropout_p=m3_params.dropout_p)

        # omega is the universal embedding of info field variant-level data
        info_layers = [tensors.NUM_INFO_FEATURES] + m3_params.hidden_info_layers
        self.omega = MLP(info_layers, batch_normalize=False, dropout_p=m3_params.dropout_p)

        # rho is the universal aggregation function
        ref_alt_info_embedding_dimension = 2 * read_layers[-1] + info_layers[-1]
        self.rho = MLP([ref_alt_info_embedding_dimension] + m3_params.aggregation_layers, batch_normalize=False,
                       dropout_p=m3_params.dropout_p)

        # We probably don't need dropout for the final output layers
        output_layers_sizes = [m3_params.aggregation_layers[-1]] + m3_params.output_layers + [1]
        self.outputs = nn.ModuleList(MLP(output_layers_sizes) for _ in VariantType)

        self.calibration = Calibration()

        self.prior_model = PriorModel(4.0)

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

    def set_epoch_type(self, epoch_type: EpochType):
        if epoch_type == EpochType.TRAIN:
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

    def forward(self, batch: data.Batch, calibrated=True, posterior=False):
        # embed reads and take mean within each datum to get tensors of shape (batch size x embedding dimension)
        phi_reads = torch.sigmoid(self.phi(batch.reads()))
        ref_means = torch.cat([torch.mean(phi_reads[s], dim=0, keepdim=True) for s in batch.ref_slices()], dim=0)
        alt_means = torch.cat([torch.mean(phi_reads[s], dim=0, keepdim=True) for s in batch.alt_slices()], dim=0)

        # stack side-by-side to get 2D tensor, where each variant row is (ref mean, alt mean, info)
        omega_info = torch.sigmoid(self.omega(batch.info()))
        concatenated = torch.cat((ref_means, alt_means, omega_info), dim=1)
        aggregated = self.rho(concatenated)

        logits = torch.zeros(batch.size())
        for variant_type in VariantType:
            # It's slightly wasteful to compute output for every variant type when we only
            # use one, but this is such a small part of the model and it lets us use batches of mixed variant types
            output = torch.squeeze(self.outputs[variant_type.value](aggregated))
            mask = torch.tensor([1 if variant_type.is_same_type(site_info) else 0 for site_info in batch.site_info()])
            logits += mask * output

        if calibrated:
            logits = self.calibration(logits, batch.alt_counts())

        if posterior:
            use_m2_filter = torch.LongTensor(
                [1 if m2.filters().intersection(self.m2_filters_to_keep) else 0 for m2 in batch.mutect_info()])
            logits = (1 - use_m2_filter) * logits + 100 * use_m2_filter
            logits = self.prior_model(logits, batch)

        return logits

    def learn_spectra(self, loader, num_epochs):
        self.learn_spectrum_mode()
        logits_and_batches = [(self(batch).detach(), batch) for batch in loader]
        optimizer = torch.optim.Adam(self.spectra_parameters())
        criterion = nn.BCEWithLogitsLoss()

        spectra_losses = []
        epochs = []
        pbar = trange(num_epochs, desc="AF spectra epoch")
        for epoch in pbar:
            epochs.append(epoch + 1)
            epoch_loss = 0
            for logits, batch in logits_and_batches:
                use_m2_filter = torch.LongTensor(
                    [1 if m2.filters().intersection(self.m2_filters_to_keep) else 0 for m2 in batch.mutect_info()])
                filter_logits = (1 - use_m2_filter) * logits + 100 * use_m2_filter
                posterior_logits = self.prior_model(filter_logits, batch)

                loss = criterion(posterior_logits, batch.labels())
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

        validation.simple_plot([(epochs, calibration_losses, "curve")], "epoch", "loss",
                               "Learning curve for calibration")

    def calculate_logit_threshold(self, loader):
        self.train(False)
        artifact_probs = []

        for batch in loader:
            artifact_probs.extend(torch.sigmoid(self(batch, posterior=True)).tolist())

        artifact_probs.sort()
        total_variants = len(artifact_probs) - sum(artifact_probs)

        # start by rejecting everything, then raise threshold one datum at a time
        threshold, tp, fp, best_f = 0, 0, 0, 0

        for prob in artifact_probs:
            tp += (1 - prob)
            fp += prob
            current_f = f_score(tp, fp, total_variants)

            if current_f > best_f:
                best_f = current_f
                threshold = prob

        return torch.logit(torch.tensor(threshold)).item()

    def train_model(self, train_loader, valid_loader, test_loader, num_epochs, beta1, beta2):
        bce = torch.nn.BCEWithLogitsLoss(reduction='sum')
        train_optimizer = torch.optim.Adam(self.training_parameters())
        training_metrics = validation.TrainingMetrics()

        # balance training by weighting the loss function
        total_labeled = sum(batch.size() for batch in train_loader if batch.is_labeled())
        total_unlabeled = sum(batch.size() for batch in train_loader if not batch.is_labeled())
        labeled_to_unlabeled_ratio = total_labeled / total_unlabeled

        for epoch in trange(1, num_epochs + 1, desc="Epoch"):
            for epoch_type in [EpochType.TRAIN, EpochType.VALID]:
                self.set_epoch_type(epoch_type)
                loader = train_loader if epoch_type == EpochType.TRAIN else valid_loader

                epoch_labeled_loss, epoch_unlabeled_loss = 0, 0
                epoch_labeled_count, epoch_unlabeled_count = 0, 0
                for batch in loader:
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

                    if epoch_type == EpochType.TRAIN:
                        train_optimizer.zero_grad()
                        loss.backward()
                        train_optimizer.step()

                training_metrics.add("labeled NLL", epoch_type.name, epoch_labeled_loss / epoch_labeled_count)
                training_metrics.add("unlabeled NLL", epoch_type.name, epoch_unlabeled_loss / epoch_unlabeled_count)

            # done with training and validation for this epoch, now calculate best F on test set
            # note that we have not learned the AF spectrum yet
            optimal_f = validation.get_optimal_f_score(self, test_loader)
            training_metrics.add("optimal F score", "test", optimal_f)
            # done with epoch
        # done with training

        self.learn_calibration(valid_loader, num_epochs=200)
        self.learn_spectra(test_loader, num_epochs=200)
        # model is trained
        return training_metrics
