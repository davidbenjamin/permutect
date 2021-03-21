import torch
from torch import nn, optim
from mutect3 import spectrum


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


class ReadSetClassifier(nn.Module):
    """
    DeepSets framework for reads and variant info.  We embed each read and concatenate the mean ref read
    embedding, mean alt read embedding, and variant info embedding, then apply an aggregation function to
    this concatenation.

    read_layers: dimensions of layers for embedding reads, including input dimension, which is the
    size of each read's 1D tensor

    info_layers: dimensions of layers for embedding variant info, including input dimension, which is the
    size of variant info 1D tensor

    aggregation_layers: dimensions of layers for aggregation, excluding its input which is determined by the
    read and info embeddings.
    """

    def __init__(self, read_layers, info_layers, aggregation_layers):
        super(ReadSetClassifier, self).__init__()

        # phi is the read embedding
        self.phi = MLP(read_layers, batch_normalize=False)

        # omega is the universal embedding of info field variant-level data
        self.omega = MLP(info_layers, batch_normalize=False)

        # rho is the universal aggregation function
        # the *2 is for the use of both ref and alt reads
        # the [1] is the final binary classification in logit space
        self.rho = MLP([2 * read_layers[-1] + info_layers[-1]] + aggregation_layers + [1], batch_normalize=False)

        # since we take the mean of read embeddings we lose all information about alt count.  This is intentional
        # because a somatic classifier must be largely unaware of the allele fraction in order to avoid simply
        # calling everything with high allele fraction as good.  However, once we apply the aggregation function
        # and get logits we can multiply the output logits in an alt count-dependent way to change the confidence
        # in our predictions.  We initialize the confidence as the sqrt of the alt count which is vaguely in line
        # with statistical intuition.
        self.confidence = nn.Parameter(torch.sqrt(torch.range(0, MAX_ALT)), requires_grad=False)

    # see the custom collate_fn for information on the batched input
    def forward(self, batch):

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

        # scale the logits to express greater certainty ~sqrt(N) with increasing alt count.  We might get rid of this.
        output = self.rho(concatenated)

        # TODO: get rid of squeezing once we have multi-class logit output?
        output = torch.squeeze(output)

        truncated_counts = torch.LongTensor([min(c, MAX_ALT) for c in batch.alt_counts()])
        return output * torch.index_select(self.confidence, 0, truncated_counts)

    # the self.confidence calibration layer is sort of a hyperparameter that we can learn from the
    # validation set.  We do not want to learn it from the training set!  This method lets us freeze the whole
    # model except for calibration.
    def freeze_everything_except_calibration(self):
        for param in self.parameters():
            param.requires_grad = False
        self.confidence.requires_grad = True

    def unfreeze_everything_except_calibration(self):
        for param in self.parameters():
            param.requires_grad = True
        self.confidence.requires_grad = False


    def make_posterior_model(self, test_loader, logit_threshold):
        iterations = 2

        result = self
        for n in range(iterations):
            artifact_proportion, artifact_spectrum, variant_spectrum = \
                spectrum.learn_af_spectra(result, test_loader, m2_filters_to_keep={'normal_artifact'}, threshold=logit_threshold)
            result = PriorAdjustedReadSetClassifier(result, artifact_proportion, artifact_spectrum,variant_spectrum)

        return result


# modify the output of a wrapped read set classifier (may be temperature-scaled) to account for
# 1. overall prior probability of artifact vs variant
# 2. artifact AF spectrum
# 3. variant AF spectrum
class PriorAdjustedReadSetClassifier(nn.Module):
    """
    A thin decorator, which wraps the above model with temperature scaling
    NB: Output of the neural network should be the classification logits,
            NOT the softmax (or log softmax)!
    """

    def __init__(self, model, artifact_proportion, artifact_spectrum, variant_spectrum):
        super(PriorAdjustedReadSetClassifier, self).__init__()
        self.model = model
        self.log_artifact_proportion = torch.log(torch.FloatTensor([artifact_proportion]))
        self.log_variant_proportion = torch.log(torch.FloatTensor([1 - artifact_proportion]))
        self.artifact_spectrum = artifact_spectrum
        self.variant_spectrum = variant_spectrum

    def forward(self, batch):
        # these logits are from a model trained on a balanced data set i.e. they are (implicitly)
        # the posterior probability of an artifact when the priors are flat
        # that is, they represent log likelihood ratio log(P(data|artifact)/P(data|non-artifact))
        artifact_to_variant_log_likelihood_ratios = self.model(batch)

        alt_counts = batch.alt_counts().numpy()
        depths = [datum.tumor_depth() for datum in batch.mutect_info()]

        # these are relative log priors of artifacts and variants to have k alt reads out of n total
        artifact_log_priors = torch.FloatTensor(
            [self.log_artifact_proportion + self.artifact_spectrum.log_likelihood(k, n).item() for (k, n) in
             zip(alt_counts, depths)])
        variant_log_priors = torch.FloatTensor(
            [self.log_variant_proportion + self.variant_spectrum.log_likelihood(k, n).item() for (k, n) in
             zip(alt_counts, depths)])
        artifact_to_variant_log_prior_ratios = artifact_log_priors - variant_log_priors

        # the sum of the log prior ratio and the log likelihood ratio is the log posterior ratio
        # that is, it is the output we want, in logit form
        return artifact_to_variant_log_prior_ratios + artifact_to_variant_log_likelihood_ratios

    def get_artifact_spectrum(self):
        return self.artifact_spectrum

    def get_variant_spectrum(self):
        return self.variant_spectrum

