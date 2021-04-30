import matplotlib.pyplot as plt
import torch
from torch import nn
from tqdm.autonotebook import trange

from mutect3 import validation, tensors, data


def freeze(parameters):
    for parameter in parameters:
        parameter.requires_grad = False


def unfreeze(parameters):
    for parameter in parameters:
        parameter.requires_grad = True


def f_score(tp, fp, total_true):
    fn = total_true - tp
    return tp / (tp + (fp + fn) / 2)


# given list of slice sizes, produce a list of index slice objects
# eg input = [2,3,1] --> [slice(0,2), slice(2,5), slice(5,6)]
def make_slices(sizes):
    slice_ends = torch.cumsum(sizes, dim=0)
    return [slice(0 if n == 0 else slice_ends[n - 1], slice_ends[n]) for n in range(len(sizes))]


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


MAX_ALT = 10


# note: this function works for alpha, beta 1D tensors of the same size, and n, k 1D tensors of the same
# size (different in general from that of alpha, beta)
# the result is a 2D tensor of the beta binomial
# result[i,j] = beta_binomial(n[i],k[i],alpha[j],beta[j])
def beta_binomial(n, k, alpha, beta):
    n2 = n.unsqueeze(1)
    k2 = k.unsqueeze(1)
    alpha2 = alpha.unsqueeze(0)
    beta2 = beta.unsqueeze(0)
    return torch.lgamma(k2 + alpha2) + torch.lgamma(n2 - k2 + beta2) + torch.lgamma(alpha2 + beta2) \
           - torch.lgamma(n2 + alpha2 + beta2) - torch.lgamma(alpha2) - torch.lgamma(beta2)


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
        log_likelihoods = beta_binomial(n, k, self.a, self.b)

        # by the convention above, the 0th dimension of log_likelihoods is n,k (batch) and the 1st dimension
        # is alpha, beta.  We sum over the latter, getting a 1D tensor corresponding to the batch
        return torch.logsumexp(log_pi + log_likelihoods, dim=1)

    # compute 1D tensor of log-likelihoods P(alt count|n, AF mixture model) over all data in batch
    def forward(self, batch):
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
        log_densities = torch.logsumexp(weighted_log_densities, dim=0)
        fig = plt.figure()
        spec = fig.gca()
        spec.plot(f.detach().numpy(), torch.exp(log_densities).detach().numpy())
        spec.set_title(title)
        return fig, spec


# contains variant spectrum, artifact spectrum, and artifact/variant log prior ratio
class PriorModel(nn.Module):
    def __init__(self, initial_log_ratio=0.0):
        super(PriorModel, self).__init__()
        self.variant_spectrum = AFSpectrum()
        self.snv_artifact_spectrum = AFSpectrum()
        self.insertion_artifact_spectrum = AFSpectrum()
        self.deletion_artifact_spectrum = AFSpectrum()

        # log prior ratio log[P(artifact)/P(variant)]
        self.prior_log_odds = nn.Parameter(torch.tensor(initial_log_ratio))

        self.snv_prior_log_odds = nn.Parameter(torch.tensor(initial_log_ratio))
        self.insertion_prior_log_odds = nn.Parameter(torch.tensor(initial_log_ratio))
        self.deletion_prior_log_odds = nn.Parameter(torch.tensor(initial_log_ratio))

    # forward pass returns posterior logits of being artifact given likelihood logits
    def forward(self, logits, batch):
        variant_spectrum_ll = self.variant_spectrum(batch)

        snv = logits + self.snv_prior_log_odds + self.snv_artifact_spectrum(batch) - variant_spectrum_ll
        insertion = logits + self.insertion_prior_log_odds + self.insertion_artifact_spectrum(batch) - variant_spectrum_ll
        deletion = logits + self.deletion_prior_log_odds + self.deletion_artifact_spectrum(batch) - variant_spectrum_ll

        variant_lengths = [len(site_info.alt()) - len(site_info.ref()) for site_info in batch.site_info()]
        snv_mask = torch.tensor([1 if length == 0 else 0 for length in variant_lengths])
        insertion_mask = torch.tensor([1 if length > 0 else 0 for length in variant_lengths])
        deletion_mask = torch.tensor([1 if length < 0 else 0 for length in variant_lengths])

        return snv_mask * snv + insertion_mask * insertion + deletion_mask * deletion

    def plot_spectra(self):
        self.snv_artifact_spectrum.plot_spectrum("SNV artifact AF spectrum")
        self.insertion_artifact_spectrum.plot_spectrum("Insertion artifact AF spectrum")
        self.deletion_artifact_spectrum.plot_spectrum("Deletion artifact AF spectrum")
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
    def __init__(self):
        super(Calibration, self).__init__()
        # Taking the mean of read embeddings erases the alt count, by design.  However, it is fine to multiply
        # the output logits in a count-dependent way to modulate confidence.  We initialize as the sqrt of the alt count.
        self.confidence = nn.Parameter(torch.sqrt(torch.arange(0, MAX_ALT + 1).float()))

        # we apply as asymptotic threshold function logit --> M * tanh(logit/M) where M is the maximum absolute
        # value of the thresholded output.  For logits << M this is the identity, and approaching M the asymptote
        # gradually turns on.  This is a continuous way to truncate the model's confidence and is part of calibration.
        # We initialize it to something large.
        self.max_logit = nn.Parameter(torch.tensor(10.0))

    def forward(self, logits, alt_counts):
        truncated_counts = torch.LongTensor([min(c, MAX_ALT) for c in alt_counts])
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
        self.snv_output = MLP(output_layers_sizes)
        self.insertion_output = MLP(output_layers_sizes)
        self.deletion_output = MLP(output_layers_sizes)

        self.calibration = Calibration()

        self.prior_model = PriorModel(4.0)

    def get_prior_model(self):
        return self.prior_model

    def training_parameters(self):
        result = []
        result.extend(self.phi.parameters())
        result.extend(self.omega.parameters())
        result.extend(self.rho.parameters())
        result.extend(self.snv_output.parameters())
        result.extend(self.insertion_output.parameters())
        result.extend(self.deletion_output.parameters())
        return result

    def calibration_parameters(self):
        return [self.calibration.max_logit]

    def spectra_parameters(self):
        return self.prior_model.parameters()

    def freeze_all(self):
        freeze(self.parameters())

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

        # embed each read and take mean within each datum to get tensors of shape (batch size x embedding dimension)
        phi_ref = torch.sigmoid(self.phi(batch.ref()))
        phi_alt = torch.sigmoid(self.phi(batch.alt()))
        ref_slices = make_slices(batch.ref_counts())
        alt_slices = make_slices(batch.alt_counts())
        ref_means = torch.cat([torch.mean(phi_ref[s], dim=0, keepdim=True) for s in ref_slices], dim=0)
        alt_means = torch.cat([torch.mean(phi_alt[s], dim=0, keepdim=True) for s in alt_slices], dim=0)

        # stack side-by-side to get 2D tensor, where each variant row is (ref mean, alt mean, info)
        omega_info = torch.sigmoid(self.omega(batch.info()))
        concatenated = torch.cat((ref_means, alt_means, omega_info), dim=1)
        aggregated = self.rho(concatenated)

        # squeeze to get 1D tensor.  It's slightly wasteful to compute output for every variant type when we only
        # use one, but this is such a small part of the model and it lets us use batches of mixed variant types
        # without fancy code
        snv = torch.squeeze(self.snv_output(aggregated))
        insertion = torch.squeeze(self.insertion_output(aggregated))
        deletion = torch.squeeze(self.deletion_output(aggregated))

        variant_lengths = [len(site_info.alt()) - len(site_info.ref()) for site_info in batch.site_info()]
        snv_mask = torch.tensor([1 if length == 0 else 0 for length in variant_lengths])
        insertion_mask = torch.tensor([1 if length > 0 else 0 for length in variant_lengths])
        deletion_mask = torch.tensor([1 if length < 0 else 0 for length in variant_lengths])

        logits = snv_mask * snv + insertion_mask * insertion + deletion_mask * deletion

        if calibrated:
            logits = self.calibration(logits, batch.alt_counts())

        # TODO: maybe have variant type-dependent prior models?
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
            if epoch > 5:
                delta = abs(epoch_loss - spectra_losses[-2])
                total_delta = abs(epoch_loss - spectra_losses[0])
                if delta < 0.001 * total_delta:
                    break

        fig, curve = validation.simple_plot([(epochs, spectra_losses, "AF learning curve")], "epoch", "loss", "AF Learning curve")

    # calculate and detach the likelihoods layers, then learn the calibration layer with SGD
    def learn_calibration(self, loader, num_epochs):
        uncalibrated_logits_labels_counts = [
            (self(batch.original_batch(), calibrated=False).detach(), batch.original_batch().labels(),
             batch.original_batch().alt_counts()) for batch in loader if batch.is_labeled()]
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
        variant_probs = []

        for batch in loader:
            logits = self(batch, posterior=True)
            true_probs = 1 - torch.sigmoid(logits)

            for n in range(batch.size()):
                variant_probs.append(true_probs[n].item())

        variant_probs.sort()
        total_variants = sum(variant_probs)

        # we are going to start by accepting everything -- the threshold is just below the smallest probability
        threshold = 0  # must be greater than or equal to this threshold for true variant probability
        tp = total_variants
        fp = len(variant_probs) - total_variants
        best_F = f_score(tp, fp, total_variants)

        for prob in variant_probs:  # we successively reject each probability and increase the threshold
            tp = tp - prob
            fp = fp - (1 - prob)
            F = f_score(tp, fp, total_variants)

            if F > best_F:
                best_F = F
                threshold = prob

        # we have calculate a variant probability threshold but we want an artifact logit threshold
        return torch.logit(1 - torch.tensor(threshold)).item()

    def train_model(self, train_loader, valid_loader, test_loader, num_epochs):
        bce = torch.nn.BCEWithLogitsLoss(reduction='sum')
        train_optimizer = torch.optim.Adam(self.training_parameters())
        training_metrics = validation.TrainingMetrics()

        # In case the DataLoader is not balanced between labeled and unlabeled, we do so with the loss function
        total_labeled, total_unlabeled = 0, 0
        for batch in train_loader:
            if batch.is_labeled():
                total_labeled += batch.size()
            else:
                total_unlabeled += batch.size()
        labeled_to_unlabeled_ratio = total_labeled / total_unlabeled

        for epoch in trange(1, num_epochs + 1, desc="Epoch"):
            # training epoch, then validation epoch
            for train_vs_valid in [True, False]:
                loader = train_loader if train_vs_valid else valid_loader

                if train_vs_valid:
                    self.training_mode()
                else:
                    self.freeze_all()

                epoch_labeled_loss, epoch_unlabeled_loss = 0, 0
                epoch_labeled_count, epoch_unlabeled_count = 0, 0
                for batch in loader:
                    # batch is data.AugmentedBatch
                    orig_pred = self(batch.original_batch())
                    aug1_pred = self(batch.first_augmented_batch())
                    aug2_pred = self(batch.second_augmented_batch())

                    if (batch.is_labeled()):
                        labels = batch.original_batch().labels()
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

                    if train_vs_valid:
                        train_optimizer.zero_grad()
                        loss.backward()
                        train_optimizer.step()

                type = "training" if train_vs_valid else "validating"
                training_metrics.add("labeled NLL", type, epoch_labeled_loss / epoch_labeled_count)
                training_metrics.add("unlabeled NLL", type, epoch_unlabeled_loss / epoch_unlabeled_count)

            # done with training and validation for this epoch, now calculate best F on test set
            # note that we have not learned the AF spectrum yet
            optimal_f = validation.get_optimal_f_score(self, test_loader)
            training_metrics.add(validation.TrainingMetrics.F, "test", optimal_f)
            # done with epoch
        # done with training

        # TODO: this is a lot of epochs.  Make this more efficient
        self.learn_calibration(valid_loader, num_epochs=100)
        self.learn_spectra(test_loader, num_epochs=200)
        # model is trained
        return training_metrics
