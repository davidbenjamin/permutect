from torch import nn
import torch
import math

from tqdm.autonotebook import trange, tqdm
from mutect3 import utils
from mutect3.architecture.mlp import MLP
from typing import List


class VAE(nn.Module):

    # layer sizes starting from the input dimension and ending with the latent dimension
    def __init__(self, layer_sizes: List[int], device, batch_normalize=False):
        super(VAE, self).__init__()
        self.device = device
        self.input_dim = layer_sizes[0]
        self.latent_dim = layer_sizes[-1]

        self.encoder = MLP(layer_sizes, batch_normalize=batch_normalize)
        self.encoder.to(self.device)
        self.decoder = MLP(layer_sizes[::-1], batch_normalize=batch_normalize)
        self.decoder.to(self.device)

        # this has nothing to do with stochastically generating the latent representation z
        # from the mean and log variance output by the encoder.  Rather it is a model parameter
        # regarding the noise in the reconstruction.  It is often blithely set to 1/2 for
        # convenience so that the reconstruction loss is simply mean squared error, but we'll
        # be scrupulous here.
        self.sigma_squared = nn.Parameter(torch.tensor(0.5))

        # in order to avoid overflow nan errors, we set a learnable scale for the mean and log variance
        # They will be truncated by sigmoids, but the model can learn a multiplicative factor to
        # dilate the sigmoids
        self.mean_scale = nn.Parameter(torch.tensor(1.0))
        self.logvar_scale = nn.Parameter(torch.tensor(1.0))

        # latent mean and variance
        self.mean_layer = nn.Linear(self.latent_dim, self.latent_dim)
        self.mean_layer.to(self.device)
        self.logvar_layer = nn.Linear(self.latent_dim, self.latent_dim)
        self.logvar_layer.to(self.device)

    # return the mean and standard deviation of the embedding into latent space
    def encode(self, x):
        hidden = self.encoder(x)
        return self.mean_scale * torch.sigmoid(self.mean_layer(hidden)), self.logvar_scale * torch.sigmoid(self.logvar_layer(hidden))

    def compress(self, x):
        return self.encoder(x)

    def sample(self, mean, std):
        epsilon = torch.randn_like(std).to(self.device)
        return mean + std * epsilon

    def decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        mean, logvar = self.encode(x)
        std = torch.exp(0.5 * logvar)
        z = self.sample(mean, std)
        reconstruction = self.decode(z)
        return reconstruction, mean, logvar

    def loss_function(self, input_x, reconstructed_x, mean, logvar):
        # this will be the mean over all inputs in the batch
        diff = reconstructed_x - input_x
        reconstruction_loss = 0.5 * torch.log(2 * math.pi * self.sigma_squared) + \
            torch.mean(torch.sum(torch.square(diff), dim=1)) / (2 * self.sigma_squared)

        kl_loss = torch.mean(torch.sum(torch.square(mean) + torch.exp(logvar) - (logvar + 1), dim=1))/2

        return reconstruction_loss + kl_loss

    def forward_to_loss(self, input_x):
        reconstructed_x, mean, logvar = self.forward(input_x)
        return self.loss_function(input_x, reconstructed_x, mean, logvar)

    def train_model_one_epoch(self, input_batch_generator, learning_rate=0.001):
        optimizer = torch.optim.AdamW(self.parameters(), lr=learning_rate)

        average_loss = utils.StreamingAverage(device=self.device)
        pbar = tqdm(enumerate(input_batch_generator), mininterval=10)
        for n, input_x in pbar:
            loss = self.forward_to_loss(input_x)
            utils.backpropagate(optimizer, loss)
            average_loss.record_sum(loss, len(input_x))
        print("loss: " + str(average_loss.get()))




