from typing import List

import torch
from torch.utils.tensorboard import SummaryWriter
from torch import nn
from tqdm.autonotebook import trange, tqdm

from mutect3.metrics.plotting import simple_plot
from mutect3 import utils
from mutect3.architecture.mlp import MLP
from mutect3.data.normal_artifact_batch import NormalArtifactBatch
from mutect3.utils import beta_binomial, StreamingAverage


class NormalArtifactModel(nn.Module):

    def __init__(self, hidden_layers: List[int], dropout_p: float = None):
        super(NormalArtifactModel, self).__init__()

        # the shapes are hard-coded and immutable -- the idea is that we're not really interested in modeling
        # the tumor AF spectrum perfectly, just in figuring out coarsely whether to expect a significant amount
        # of tumor alt reads or not
        shapes = [(1, 1000), (1, 100), (5, 50), (10, 45), (15, 40), (5, 2)]
        self.num_components = len(shapes)
        self.a = torch.FloatTensor([shape[0] for shape in shapes])
        self.b = torch.FloatTensor([shape[1] for shape in shapes])

        # MLP for computing (pre-softmax) weights for the beta-binomial mixture.
        # input is the 1-element tensor -- just the normal AF -- though that may change
        # output are the pre-softmax mixture component weights
        self.mlp_z = MLP([1] + hidden_layers + [self.num_components], batch_normalize=False, dropout_p=dropout_p)

    # note: copied from af_spectrum.py
    # k "successes" out of n "trials" -- k and n are 1D tensors of the same size
    # normal AF is 2D, batch size x 1 (the second dimension is just the normal AF)
    def log_likelihood_given_normal_af(self, tumor_alt_counts: torch.IntTensor, tumor_depths: torch.IntTensor, normal_af: torch.Tensor):
        # z is a 2D tensor batch size x number of mixture components
        z = self.mlp_z(normal_af)

        # log_weights is the same shape as z but after a log softmax
        # each row is the log mixture weights for one datum in the batch
        log_weights = nn.functional.log_softmax(z, dim=1)

        # tumor depths, tumor alt counts, a, b are all 1D tensors.  If we unsqueeze n,k along dim=1 and unsqueeze a,,b along dim=0
        # the resulting 2D log_likelihoods will have the structure
        # likelihoods[i,j] = beta_binomial(n[i],k[i],alpha[j],beta[j])
        # this means that the first dimension is the batch and the second is the mixture component
        log_likelihoods = beta_binomial(tumor_depths.unsqueeze(1), tumor_alt_counts.unsqueeze(1), self.a.unsqueeze(0), self.b.unsqueeze(0))

        # by the convention above, the 0th dimension of log_likelihoods is n,k (batch) and the 1st dimension
        # is alpha, beta.  We sum over the latter, getting a 1D tensor corresponding to the batch
        return torch.logsumexp(log_weights + log_likelihoods, dim=1)

    # given normal alts, normal depths, tumor depths, what is the log likelihood of given tumor alt counts
    # that is, this returns the 1D tensor of log likelihoods
    def log_likelihood(self, batch: NormalArtifactBatch):
        normal_af = (batch.normal_alt() / (batch.normal_depth() + 0.1)).unsqueeze(dim=1)
        return self.log_likelihood_given_normal_af(batch.tumor_alt(), batch.tumor_depth(), normal_af)

    def forward(self, batch: NormalArtifactBatch):
        return self.log_likelihood(batch)

    # plot the beta mixture density of tumor AF given normal data
    def plot_spectrum(self, normal_af: float, title):
        f = torch.arange(0.005, 0.999, 0.001)

        # z is a 2D tensor: 1 x number of mixture components, where '1' is a dummy singleton batch
        z = self.mlp_z(torch.Tensor([[normal_af]]))

        # squeeze removes the dummy batch dimension.  Now it's 1D
        log_weights = nn.functional.log_softmax(z, dim=1).squeeze()

        # list of component beta distributions
        betas = [torch.distributions.beta.Beta(torch.FloatTensor([alpha]), torch.FloatTensor([beta])) for (alpha, beta)
                 in zip(self.a.numpy(), self.b.numpy())]

        # list of tensors - the list is over the mixture components, the tensors are over AF values f
        unweighted_log_densities = torch.stack([beta.log_prob(f) for beta in betas], dim=0)
        # unsqueeze to make log_pi a column vector (2D tensor) for broadcasting
        weighted_log_densities = torch.unsqueeze(log_weights, 1) + unweighted_log_densities
        densities = torch.exp(torch.logsumexp(weighted_log_densities, dim=0))

        return simple_plot([(f.detach().numpy(), densities.detach().numpy(), " ")], "AF", "density", title)

    def train_model(self, train_loader, valid_loader, num_epochs, summary_writer: SummaryWriter):
        optimizer = torch.optim.Adam(self.parameters())

        for epoch in trange(1, num_epochs + 1, desc="Epoch"):
            print("Normal artifact epoch " + str(epoch))
            for epoch_type in [utils.EpochType.TRAIN, utils.EpochType.VALID]:
                loader = train_loader if epoch_type == utils.EpochType.TRAIN else valid_loader

                epoch_loss = StreamingAverage()
                pbar = tqdm(loader, mininterval=10)
                for batch in pbar:
                    log_likelihoods = self.forward(batch)
                    weights = 1 / batch.downsampling()
                    loss = -torch.mean(weights * log_likelihoods)
                    epoch_loss.record(loss.item())

                    if epoch_type == utils.EpochType.TRAIN:
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                summary_writer.add_scalar(epoch_type.name + "/Normal artifact NLL", epoch_loss.get(), epoch)
            # done with epoch
        # done with training
        # model is trained

        for normal_af in [0.0, 0.03, 0.06, 0.1, 0.15, 0.25, 0.5, 0.75]:
            fig, curve = self.plot_spectrum(normal_af, "NA modeled tumor AF given normal AF = " + str(normal_af))
            summary_writer.add_figure("NA modeled tumor AF given normal AF = " + str(normal_af), fig)
