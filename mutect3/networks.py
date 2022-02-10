from typing import List

import torch
from torch import nn
from tqdm.autonotebook import tqdm, trange
from matplotlib.backends.backend_pdf import PdfPages

from mutect3 import validation, data, utils

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
        return self.log_likelihood(batch.pd_tumor_alt_counts(), batch.pd_tumor_depths())

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

        return validation.simple_plot([(f.detach().numpy(), densities.detach().numpy(), " ")], "AF", "density", title)


# contains variant spectrum, artifact spectrum, and artifact/variant log prior ratio
class PriorModel(nn.Module):
    def __init__(self, initial_log_ratio=0.0):
        super(PriorModel, self).__init__()
        self.variant_spectrum = AFSpectrum()

        self.artifact_spectra = nn.ModuleList()
        self.prior_log_odds = nn.ParameterList()  # log prior ratio log[P(artifact)/P(variant)] for each type
        for artifact_type in utils.VariantType:
            self.artifact_spectra.append(AFSpectrum())
            self.prior_log_odds.append(nn.Parameter(torch.tensor(initial_log_ratio)))

    # calculate log likelihoods for all variant types and then apply a mask to select the correct
    # type for each datum in a batch
    def artifact_log_likelihoods(self, batch: data.Batch):
        result = torch.zeros(batch.size())
        for variant_type in utils.VariantType:
            output = self.prior_log_odds[variant_type.value] + self.artifact_spectra[variant_type.value](batch)
            mask = torch.tensor([1 if variant_type == datum.variant_type() else 0 for datum in batch.original_list()])
            result += mask * output

        return result

    # forward pass returns posterior logits of being artifact given likelihood logits
    def forward(self, logits, batch):
        return logits + self.artifact_log_likelihoods(batch) - self.variant_spectrum(batch)

    # with fixed logits from the ReadSetClassifier, the log probability of seeing the observed tensors and counts
    # This is our objective to maximize when learning the prior model
    def log_evidence(self, logits, batch):
        term1 = torch.logsumexp(torch.column_stack((logits + self.artifact_log_likelihoods(batch), self.variant_spectrum(batch))), dim=1)

        prior_log_odds = torch.zeros_like(logits)
        for variant_type in utils.VariantType:
            mask = torch.tensor([1 if variant_type == datum.variant_type() else 0 for datum in batch.original_list()])
            prior_log_odds += mask * self.prior_log_odds[variant_type.value]

        term2 = torch.logsumexp(torch.column_stack((torch.zeros_like(logits), prior_log_odds)), dim=1)
        return term1 - term2

    # returns list of fig, curve tuples
    def plot_spectra(self):
        result = []
        for variant_type in utils.VariantType:
            result.append(self.artifact_spectra[variant_type.value].plot_spectrum(variant_type.name + " artifact AF spectrum"))
        result.append(self.variant_spectrum.plot_spectrum("Variant AF spectrum"))
        return result


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


class NormalArtifactModel(nn.Module):

    def __init__(self, hidden_layers: List[int], dropout_p: float = None):
        super(NormalArtifactModel, self).__init__()

        # number of mixture components
        self.num_components = 3

        # we will convert the normal alt and normal depth into the shape parameters of
        # the beta distribution they define.
        input_to_output_layer_sizes = [2] + hidden_layers + [self.num_components]

        # the alpha, beta, and z layers take as input 2D BATCH_SIZE x 2 tensors (the 2 comes from the input dimension
        # of normal alt count, normal depth, which we transform into normal mu, sigma) and output
        # 2D BATCH_SIZE x num_components tensors.  Each row has all the component alphas (or betas, or z)
        # for one datum in the batch
        self.mlp_alpha = MLP(input_to_output_layer_sizes, batch_normalize=False, dropout_p=dropout_p)
        self.mlp_beta = MLP(input_to_output_layer_sizes, batch_normalize=False, dropout_p=dropout_p)
        self.mlp_z = MLP(input_to_output_layer_sizes, batch_normalize=False, dropout_p=dropout_p)

    def forward(self, batch: data.NormalArtifactBatch):
        return self.log_likelihood(batch)

    def get_beta_parameters(self, batch: data.NormalArtifactBatch):
        # beta posterior of normal counts with flat 1,1 prior
        # alpha, bet, mu, sigma are all 1D tensors
        alpha = batch.normal_alt() + 1
        alpha = alpha.float()
        beta = batch.normal_depth() - batch.normal_alt() + 1
        beta = beta.float()
        mu = alpha / (alpha + beta)
        sigma = torch.sqrt(alpha * beta / ((alpha + beta) * (alpha + beta) * (alpha + beta + 1)))

        # parametrize the input as the mean and std of this beta
        # each row is one datum of the batch
        mu_sigma = torch.stack((mu, sigma), dim=1)

        # parametrize the generative model in terms of the beta shape parameters
        # note the exp to make it positive and the squeeze to make it 1D, as required by the beta binomial
        output_alpha = torch.exp(self.mlp_alpha(mu_sigma))
        output_beta = torch.exp(self.mlp_beta(mu_sigma))
        output_z = self.mlp_z(mu_sigma)
        log_pi = nn.functional.log_softmax(output_z, dim=1)

        # These are all 2D tensors -- BATCH_SIZE x num_components
        return output_alpha, output_beta, log_pi

    # given normal alts, normal depths, tumor depths, what is the log likelihood of given tumor alt counts
    # that is, this returns the 1D tensor of log likelihoods
    def log_likelihood(self, batch: data.NormalArtifactBatch):
        output_alpha, output_beta, log_pi = self.get_beta_parameters(batch)

        n = batch.tumor_depth().unsqueeze(1)
        k = batch.tumor_alt().unsqueeze(1)

        component_log_likelihoods = beta_binomial(n, k, output_alpha, output_beta)
        # 0th dimension is batch, 1st dimension is component.  Sum over the latter
        return torch.logsumexp(log_pi + component_log_likelihoods, dim=1)

    # plot the beta mixture density of tumor AF given normal data
    def plot_spectrum(self, datum: data.NormalArtifactDatum, title):
        f = torch.arange(0.01, 0.99, 0.01)

        # make a singleton batch
        batch = data.NormalArtifactBatch([datum])
        output_alpha, output_beta, log_pi = self.get_beta_parameters(batch)

        # remove dummy batch dimension
        output_alpha = output_alpha.squeeze()
        output_beta = output_beta.squeeze()
        log_pi = log_pi.squeeze()

        # list of component beta distributions
        betas = [torch.distributions.beta.Beta(torch.FloatTensor([alpha]), torch.FloatTensor([beta])) for (alpha, beta)
                 in zip(output_alpha.detach().numpy(), output_beta.detach().numpy())]

        # list of tensors - the list is over the mixture components, the tensors are over AF values f
        unweighted_log_densities = torch.stack([beta.log_prob(f) for beta in betas], dim=0)
        # unsqueeze to make log_pi a column vector (2D tensor) for broadcasting
        weighted_log_densities = torch.unsqueeze(log_pi, 1) + unweighted_log_densities
        densities = torch.exp(torch.logsumexp(weighted_log_densities, dim=0))

        return validation.simple_plot([(f.detach().numpy(), densities.detach().numpy(), " ")], "AF", "density", title)

    def train_model(self, train_loader, valid_loader, num_epochs):
        optimizer = torch.optim.Adam(self.parameters())
        training_metrics = validation.TrainingMetrics()

        for epoch in trange(1, num_epochs + 1, desc="Epoch"):
            print("Normal artifact epoch " + str(epoch))
            for epoch_type in [utils.EpochType.TRAIN, utils.EpochType.VALID]:
                loader = train_loader if epoch_type == utils.EpochType.TRAIN else valid_loader

                epoch_loss = 0
                epoch_count = 0
                pbar = tqdm(loader)
                for batch in pbar:
                    log_likelihoods = self(batch)
                    weights = 1 / batch.downsampling()
                    loss = -torch.mean(weights * log_likelihoods)
                    epoch_loss += loss.item()
                    epoch_count += 1

                    if epoch_type == utils.EpochType.TRAIN:
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                training_metrics.add("NLL", epoch_type.name, epoch_loss / epoch_count)
            # done with epoch
        # done with training
        # model is trained
        return training_metrics


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
        read_layers = [data.NUM_READ_FEATURES] + m3_params.hidden_read_layers
        self.phi = MLP(read_layers, batch_normalize=False, dropout_p=m3_params.dropout_p)

        # omega is the universal embedding of info field variant-level data
        info_layers = [data.NUM_INFO_FEATURES] + m3_params.hidden_info_layers
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

    def forward(self, batch: data.Batch, calibrated=True, posterior=False, normal_artifact=False):
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
                ### NORMAL ARTIFACT CALCULATION BEGINS

                # posterior probability of normal artifact given observed read counts
                # log likelihood of tumor read counts given tumor variant spectrum P(tumor counts | somatic variant)
                somatic_log_lk = self.prior_model.variant_spectrum.log_likelihood(batch.alt_counts(),
                                                                                  batch.alt_counts() + batch.ref_counts())

                # log likelihood of tumor read counts given the normal read counts under normal artifact sub-model
                # P(tumor counts | normal counts)
                na_log_lk = self.normal_artifact_model.log_likelihood(batch.normal_artifact_batch())

                # note that prior of normal artifact is essentially 1
                # posterior is P(artifact) = P(tumor counts | normal counts) /[P(tumor counts | normal) + P(somatic)*P(tumor counts | somatic)]
                # and posterior logits are log(post prob artifact / post prob somatic)
                # so, with n_ll = normal artifact log likelhood and som_ll = somatic log likelihood and pi = log P(somatic)
                # posterior logit = na_ll - pi - som_ll

                # TODO: WARNING: HARD-CODED MAGIC CONSTANT!!!!!
                log_somatic_prior = -11.0
                na_logits = na_log_lk - log_somatic_prior - somatic_log_lk
                ### NORMAL ARTIFACT CALCULATION ENDS

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

        for epoch in trange(1, num_epochs + 1, desc="Epoch"):
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
