import torch
import math
from torch import nn
from mutect3 import validation, tensors
import matplotlib.pyplot as plt


def freeze(parameters):
    for parameter in parameters:
        parameter.requires_grad = False

def unfreeze(parameters):
    for parameter in parameters:
        parameter.requires_grad = True

def F_score(tp, fp, total_true):
    fn = total_true - tp
    return tp / (tp + (fp + fn) / 2)

class MLP(nn.Module):
    """
    A fully-connected network (multi-layer perceptron) that we need frequently
    as a sub-network.  It is parameterized by the dimensions of its layers, starting with
    the input layer and ending with the output.  Output is logits and as such no non-linearity
    is applied after the last linear transformation.
    """

    def __init__(self, layer_sizes, batch_normalize=False):
        super(MLP, self).__init__()

        self.layers = nn.ModuleList()
        for k in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[k], layer_sizes[k + 1]))

        self.bn = nn.ModuleList()
        if batch_normalize:
            for size in layer_sizes[1:]:
                self.bn.append(nn.BatchNorm1d(num_features=size))

    def forward(self, x):
        for n, layer in enumerate(self.layers):
            x = layer(x)
            if self.bn:
                x = self.bn[n](x)
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
        return self.log_likelihood(batch.alt_counts(), depths )

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
    def __init__(self, initial_log_ratio = 0.0):
        super(PriorModel, self).__init__()
        self.variant_spectrum = AFSpectrum()
        self.artifact_spectrum = AFSpectrum()

        # log of prior ratio P(artifact)/P(variant)
        self.prior_log_odds = nn.Parameter(torch.tensor(initial_log_ratio))

    # forward pass returns tuple log [P(variant)*P(alt count|variant)], log [P(artifact)*P(alt count|artifact)]
    def forward(self, batch):
        log_artifact_prior = nn.functional.logsigmoid(self.prior_log_odds)
        log_variant_prior = nn.functional.logsigmoid(-self.prior_log_odds)

        artifact_log_likelihood = self.artifact_spectrum(batch)
        variant_log_likelihood = self.variant_spectrum(batch)

        return log_variant_prior + variant_log_likelihood, log_artifact_prior + artifact_log_likelihood

    def plot_spectra(self):
        self.artifact_spectrum.plot_spectrum("Artifact AF spectrum")
        self.variant_spectrum.plot_spectrum("Variant AF spectrum")

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

    def __init__(self, hidden_read_layers, hidden_info_layers, aggregation_layers, output_layers, m2_filters_to_keep={}):
        super(ReadSetClassifier, self).__init__()

        read_layers = [tensors.NUM_READ_FEATURES] + hidden_read_layers
        info_layers = [tensors.NUM_INFO_FEATURES] + hidden_info_layers

        # note: these are only used for testing, not training
        self.m2_filters_to_keep = m2_filters_to_keep

        # phi is the read embedding
        self.phi = MLP(read_layers, batch_normalize=False)

        # omega is the universal embedding of info field variant-level data
        self.omega = MLP(info_layers, batch_normalize=False)

        # rho is the universal aggregation function
        # the *2 is for the use of both ref and alt reads
        # the [1] is the final binary classification in logit space
        self.rho = MLP([2 * read_layers[-1] + info_layers[-1]] + aggregation_layers, batch_normalize=False)

        self.snv_output = MLP([aggregation_layers[-1]] + output_layers + [1])
        self.insertion_output = MLP([aggregation_layers[-1]] + output_layers + [1])
        self.deletion_output = MLP([aggregation_layers[-1]] + output_layers + [1])

        # since we take the mean of read embeddings we lose all information about alt count.  This is intentional
        # because a somatic classifier must be largely unaware of the allele fraction in order to avoid simply
        # calling everything with high allele fraction as good.  However, once we apply the aggregation function
        # and get logits we can multiply the output logits in an alt count-dependent way to change the confidence
        # in our predictions.  We initialize the confidence as the sqrt of the alt count which is vaguely in line
        # with statistical intuition.
        self.confidence = nn.Parameter(torch.sqrt(torch.arange(0, MAX_ALT + 1).float()), requires_grad=False)

        # we apply as asymptotic threshold function logit --> M * tanh(logit/M) where M is the maximum absolute
        # value of the thresholded output.  For logits << M this is the identity, and approaching M the asymptote
        # gradually turns on.  This is a continuous way to truncate the model's confidence and is part of calibration.
        # We initialize it to something large.
        self.max_logit = nn.Parameter(torch.tensor(10.0), requires_grad=False)


        # after a count-dependent confidence, one final logit --> logit mapping of the confidence.  This is initialized
        # as the identity function but after calibration we want the output logits to saturate, representing the fact
        # that our model can never be certain.

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
        #return [self.confidence, self.max_logit]
        return [self.max_logit]

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


    # see the custom collate_fn for information on the batched input
    def forward(self, batch, calibrated=True, posterior=False):

        # broadcast the embedding to each read
        num_sets = batch.size()
        phi_ref = torch.sigmoid(self.phi(batch.ref()))
        phi_alt = torch.sigmoid(self.phi(batch.alt()))
        omega_info = torch.sigmoid(self.omega(batch.info()))

        ref_end = torch.cumsum(batch.ref_counts(), dim=0)
        ref_slices = [slice(0 if n == 0 else ref_end[n - 1], ref_end[n]) for n in range(num_sets)]
        alt_end = torch.cumsum(batch.alt_counts(), dim=0)
        alt_slices = [slice(0 if n == 0 else alt_end[n - 1], alt_end[n]) for n in range(num_sets)]

        # note that after taking means, tensors are now num_sets x embedding dimension
        ref_means = torch.cat([torch.mean(phi_ref[s], dim=0, keepdim=True) for s in ref_slices], dim=0)
        alt_means = torch.cat([torch.mean(phi_alt[s], dim=0, keepdim=True) for s in alt_slices], dim=0)

        # stack side-by-side to get 2D tensor, where each variant row is (ref mean, alt mean, info
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

        # apply alt count-dependent confidence calibration
        if calibrated:
            truncated_counts = torch.LongTensor([min(c, MAX_ALT) for c in batch.alt_counts()])
            logits = logits * torch.index_select(self.confidence, 0, truncated_counts)
            logits = self.max_logit * torch.tanh(logits / self.max_logit)
        log_evidence = 0    # this won't be used unless overwritten for prior model case

        # TODO: maybe have variant type-dependent prior models?
        if posterior:
            # 1 if we use an M2 filter regardless of M3, 0 otherwise
            use_m2_filter = torch.LongTensor([1 if m2.filters().intersection(self.m2_filters_to_keep) else 0 for m2 in batch.mutect_info()])
            logits = (1 - use_m2_filter) * logits + 100 * use_m2_filter

            log_variant_factor, log_artifact_factor = self.prior_model(batch)
            log_artifact_likelihood = nn.functional.logsigmoid(logits)
            log_variant_likelihood = nn.functional.logsigmoid(-logits)
            logits = logits + log_artifact_factor - log_variant_factor

            # these are 1D tensors, one element for each
            log_artifact_evidence = log_artifact_factor + log_artifact_likelihood
            log_variant_evidence = log_variant_factor + log_variant_likelihood
            log_evidence = torch.mean(torch.logsumexp(torch.stack([log_variant_evidence, log_artifact_evidence]), 0))

        return logits, log_evidence

    def learn_spectra(self, loader, num_epochs):
        self.learn_spectrum_mode()

        logits_and_batches = [(self(batch)[0].detach(), batch) for batch in loader]
        optimizer = torch.optim.Adam(self.spectra_parameters())

        spectra_losses = []
        converged = False
        for epoch in range(num_epochs):
            if converged:
                spectra_losses.append(spectra_losses[-1])
                continue

            print("Prior log odds: " + str(self.prior_model.prior_log_odds.item()))
            epoch_loss = 0
            # each batch gets an E step in which we apply the prior model to the logits to get posteriors and
            # an M step in which we do gradient descent on the prior model parameters
            total_variant, total_artifact = 0, 0
            for logits, batch in logits_and_batches:
                # E step -- note that we detach at the end!
                use_m2_filter = torch.LongTensor(
                    [1 if m2.filters().intersection(self.m2_filters_to_keep) else 0 for m2 in batch.mutect_info()])
                filter_logits = (1 - use_m2_filter) * logits + 100 * use_m2_filter
                log_variant_factor, log_artifact_factor = self.prior_model(batch)
                posterior_logits = filter_logits + log_artifact_factor - log_variant_factor
                artifact_probs = torch.sigmoid(posterior_logits).detach()
                num_artifacts = torch.sum(artifact_probs).item()
                num_variants = len(logits) - num_artifacts
                total_artifact += num_artifacts
                total_variant += num_variants

                # M step
                variant_ll, artifact_ll = self.prior_model(batch)
                weighted_ll = variant_ll * (1 - artifact_probs) + artifact_ll * artifact_probs
                loss = -torch.mean(weighted_ll)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            self.prior_model.prior_log_odds = nn.Parameter(torch.tensor(math.log(total_artifact/total_variant)))

            # check for convergence
            if epoch > 5:
                delta = abs(epoch_loss - spectra_losses[-1])
                total_delta = abs(epoch_loss - spectra_losses[0])
                if delta < 0.001 * total_delta:
                    converged = True

            spectra_losses.append(epoch_loss)

        # TODO horrible code duplication!!!
        epochs = range(1, num_epochs + 1)
        fig = plt.figure()
        curve = fig.gca()

        curve.plot(epochs, spectra_losses)
        curve.set_title("Learning curve for AF spectrum")
        curve.set_xlabel("epoch")
        curve.set_ylabel("loss")

    # calculate and detach the likelihoods layers, then learn the top calibration layer through several SGD epochs
    def learn_calibration(self, loader, num_epochs):

        # the training pass may cause overconfidence and spoil our calibration. Let's re-calibrate.
        uncalibrated_logits_labels_counts = [(self(batch, calibrated=False)[0].detach(), batch.labels(), \
                                              torch.LongTensor([min(c, MAX_ALT) for c in batch.alt_counts()]))
                                             for batch in loader]
        optimizer = torch.optim.Adam(self.calibration_parameters())
        criterion = nn.BCEWithLogitsLoss()

        self.learn_calibration_mode()
        calibration_losses = []

        for epoch in range(num_epochs):
            epoch_loss = 0
            for logits, labels, counts in uncalibrated_logits_labels_counts:
                #TODO: these two lines are duplicated from forward!!!
                calibrated_logits = logits * torch.index_select(self.confidence, 0, counts)
                calibrated_logits = self.max_logit * torch.tanh(calibrated_logits / self.max_logit)
                loss = criterion(calibrated_logits, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            calibration_losses.append(epoch_loss)

        epochs = range(1, num_epochs + 1)
        fig = plt.figure()
        curve = fig.gca()

        curve.plot(epochs, calibration_losses)
        curve.set_title("Learning curve for calibration")
        curve.set_xlabel("epoch")
        curve.set_ylabel("loss")

    def calculate_logit_threshold(self, loader):
        self.train(False)
        variant_probs = []

        for batch in loader:
            logits, _ = self(batch, posterior=True)
            true_probs = 1 - torch.sigmoid(logits)

            for n in range(batch.size()):
                variant_probs.append(true_probs[n].item())

        variant_probs.sort()
        total_variants = sum(variant_probs)

        # we are going to start by accepting everything -- the threshold is just below the smallest probability
        threshold = 0  # must be greater than or equal to this threshold for true variant probability
        tp = total_variants
        fp = len(variant_probs) - total_variants
        best_F = F_score(tp, fp, total_variants)

        for prob in variant_probs:  # we successively reject each probability and increase the threshold
            tp = tp - prob
            fp = fp - (1 - prob)
            F = F_score(tp, fp, total_variants)

            if F > best_F:
                best_F = F
                threshold = prob

        # we have calculate a variant probability threshold but we want an artifact logit threshold
        return torch.logit(1 - torch.tensor(threshold)).item()

    def train_model(self, train_loader, valid_loader, test_loader, num_epochs, batch_size):
        criterion = torch.nn.BCEWithLogitsLoss()
        train_optimizer = torch.optim.Adam(self.training_parameters())

        # we optimize the AF spectra -- without supervision -- on the test set
        test_optimizer = torch.optim.Adam(self.spectra_parameters())

        training_metrics = validation.TrainingMetrics()
        for epoch in range(1, num_epochs + 1):
            print("Epoch " + str(epoch))

            # training epoch, then validation epoch
            for train_vs_valid in [True, False]:
                loader = train_loader if train_vs_valid else valid_loader

                if train_vs_valid:
                    self.training_mode()
                else:
                    self.freeze_all()

                epoch_loss = 0
                for batch_number, batch in enumerate(loader):
                    predictions, _ = self(batch)
                    loss = criterion(predictions, batch.labels())
                    epoch_loss += loss.item()

                    if train_vs_valid:
                        train_optimizer.zero_grad()
                        loss.backward()
                        train_optimizer.step()

                training_metrics.add(validation.TrainingMetrics.NLL, "training" if train_vs_valid else "validating",
                                     epoch_loss / (len(loader) * batch_size))

            # done with training and validation for this epoch, now calculate best F on test set
            # note that we have not learned the AF spectrum yet
            optimal_f = validation.get_optimal_f_score(self, test_loader)
            training_metrics.add(validation.TrainingMetrics.F, "test", optimal_f)
            # done with epoch
        # done with training

        #TODO: this is a lot of epochs.  Make this more efficient
        self.learn_calibration(valid_loader, num_epochs=100)
        self.learn_spectra(test_loader, num_epochs=200)
        # model is trained
        return training_metrics