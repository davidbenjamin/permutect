import torch
from torch import nn, optim

class LinearStack(nn.Module):
    # layer_sizes starts from the input and ends with the output
    # optional batch normalization
    def __init__(self, layer_sizes, batch_normalize=False, batch_normalize_input=False):

        super(LinearStack, self).__init__()

        self.layers = nn.ModuleList()
        for k in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[k], layer_sizes[k + 1]))

        self.bn = nn.ModuleList()

        self.batch_normalize_input = False
        if batch_normalize_input:
            self.batch_normalize_input = True
            self.bn_input = nn.BatchNorm1d(num_features=layer_sizes[0])

        if batch_normalize:
            # batch normalize after every linear transformation
            for size in layer_sizes[1:]:
                self.bn.append(nn.BatchNorm1d(num_features=size))

    def forward(self, x):
        if self.batch_normalize_input:
            x = self.bn_input(x)

        # Feedforward -- note that last layer has no non-linearity
        for n, layer in enumerate(self.layers):
            x = layer(x)
            if self.bn:
                x = self.bn[n](x)
            if n < len(self.layers) - 1:
                x = nn.functional.relu(x)

        return x

MAX_ALT = 10

class ReadSetClassifier(nn.Module):
    # embedding layer sizes include input, aggregation hidden layer sizes do not include its input which is
    # the embedding output, nor its output which is a binary classification
    def __init__(self, embedding_layer_sizes, info_embedding_layer_sizes, aggregation_hidden_layer_sizes):
        super(ReadSetClassifier, self).__init__()

        # phi is the universal read embedding function
        self.phi = LinearStack(embedding_layer_sizes, batch_normalize=False, batch_normalize_input=False)
        self.embedding_dimension = embedding_layer_sizes[-1]  # this is the embedding dimension of one read

        # omega is the universal embedding of info field variant-level data
        self.omega = LinearStack(info_embedding_layer_sizes, batch_normalize=False, batch_normalize_input=False)
        self.info_embedding_dimension = info_embedding_layer_sizes[-1]

        # rho is the universal aggregation function
        # the *2 is for the use of both ref and alt reads
        # the [1] is the final binary classification in logit space
        self.rho = LinearStack(
            [2 * self.embedding_dimension + self.info_embedding_dimension] + aggregation_hidden_layer_sizes + [1],
            batch_normalize=False, batch_normalize_input=False)

        # temperature scaling logit output for calibrating on validation set
        self.temperature = None

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
        ref_means = torch.cat([torch.mean(phi_ref[slice], dim=0, keepdim=True) for slice in ref_slices], dim=0)
        alt_means = torch.cat([torch.mean(phi_alt[slice], dim=0, keepdim=True) for slice in alt_slices], dim=0)

        # stack the ref and alt means and info side-by-side so that each row of the resulting
        # 2D tensor is (mean ref 1, . . . mean ref D, mean alt 1, . . . mean alt D, ref count)
        stacked_embeddings = torch.cat((ref_means, alt_means, omega_info), dim=1)

        # broadcast the aggregation over the batch
        logits = self.rho(stacked_embeddings)

        # scale the logits to express greater certainty ~sqrt(N) with increasing alt count.  We might get rid of this.
        output = logits * torch.sqrt(torch.unsqueeze(batch.alt_counts(), 1).float())

        # note that no non-linearity is applied here because nn.BCEWithLogitsLoss() includes
        # the sigmoid.  Thus when classifying we will have to apply the sigmoid explicitly

        if self.temperature is not None:
            truncated_counts = torch.LongTensor([min(c, MAX_ALT) for c in batch.alt_counts()])
            return torch.squeeze(output) / torch.index_select(self.temperature, 0, truncated_counts)
        else:
            return torch.squeeze(output)

    def set_temperature(self, valid_loader):

        self.temperature = torch.ones(MAX_ALT + 1)

        # First: collect all the logits and labels for the validation set
        logits_list, labels_list, alt_counts_list = [], [], []
        with torch.no_grad():
            for batch in valid_loader:
                logits_list.append(self.forward(batch))
                labels_list.append(batch.labels())
                alt_counts_list.append(batch.alt_counts())
            logits = torch.cat(logits_list)
            labels = torch.cat(labels_list)
            alt_counts = torch.cat(alt_counts_list)

        self.temperature.requires_grad = True
        nll_criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=50)

        def eval():
            truncated_counts = torch.LongTensor([min(c, MAX_ALT) for c in alt_counts])
            temperatures = torch.index_select(self.temperature, 0, truncated_counts)
            loss = nll_criterion(logits / temperatures, labels)
            loss.backward()
            return loss

        optimizer.step(eval)
        self.temperature.requires_grad = False

    def uncalibrate(self):
        self.temperature = None

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
        depths = [datum.tumor_depth() for datum in batch.mutect2_data()]

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
