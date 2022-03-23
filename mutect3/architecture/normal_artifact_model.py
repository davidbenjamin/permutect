from typing import List

import torch
from torch import nn
from tqdm.autonotebook import trange, tqdm

import mutect3.metrics.plotting
import mutect3.metrics.training_metrics
from mutect3 import utils
from mutect3.architecture.mlp import MLP
from mutect3.data.normal_artifact_datum import NormalArtifactDatum
from mutect3.data.normal_artifact_batch import NormalArtifactBatch
from mutect3.utils import beta_binomial


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

    def forward(self, batch: NormalArtifactBatch):
        return self.log_likelihood(batch)

    def get_beta_parameters(self, batch: NormalArtifactBatch):
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
    def log_likelihood(self, batch: NormalArtifactBatch):
        output_alpha, output_beta, log_pi = self.get_beta_parameters(batch)

        n = batch.tumor_depth().unsqueeze(1)
        k = batch.tumor_alt().unsqueeze(1)

        component_log_likelihoods = beta_binomial(n, k, output_alpha, output_beta)
        # 0th dimension is batch, 1st dimension is component.  Sum over the latter
        return torch.logsumexp(log_pi + component_log_likelihoods, dim=1)

    # TODO: this is not used
    # plot the beta mixture density of tumor AF given normal data
    def plot_spectrum(self, datum: NormalArtifactDatum, title):
        f = torch.arange(0.01, 0.99, 0.01)

        # make a singleton batch
        batch = NormalArtifactBatch([datum])
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

        return mutect3.metrics.plotting.simple_plot([(f.detach().numpy(), densities.detach().numpy(), " ")], "AF", "density", title)

    def train_model(self, train_loader, valid_loader, num_epochs):
        optimizer = torch.optim.Adam(self.parameters())
        training_metrics = mutect3.metrics.training_metrics.TrainingMetrics()

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

                training_metrics.add("Normal_artifact_NLL", epoch_type.name, epoch_loss / epoch_count)
            # done with epoch
        # done with training
        # model is trained
        return training_metrics
