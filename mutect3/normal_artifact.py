import pickle
import random
from typing import List
import torch
from torch import nn
import math

from torch.utils.data import Dataset, DataLoader
from tqdm.autonotebook import tqdm, trange

import mutect3.utils
from mutect3 import networks, validation


class NormalArtifactDatum:
    def __init__(self, normal_alt_count: int, normal_depth: int, tumor_alt_count: int, tumor_depth: int, downsampling: float, variant_type: str):
        self._normal_alt_count = normal_alt_count
        self._normal_depth = normal_depth
        self._tumor_alt_count = tumor_alt_count
        self._tumor_depth = tumor_depth
        self._downsampling = downsampling
        self._variant_type = variant_type

    def normal_alt_count(self) -> int:
        return self._normal_alt_count

    def normal_depth(self) -> int:
        return self._normal_depth

    def tumor_alt_count(self) -> int:
        return self._tumor_alt_count

    def tumor_depth(self) -> int:
        return self._tumor_depth

    def downsampling(self) -> float:
        return self._downsampling

    def variant_type(self) -> str:
        return self._variant_type

def make_normal_artifact_pickle(file, datum_list):
    with open(file, 'wb') as f:
        pickle.dump(datum_list, f)

def load_normal_artifact_pickle(file):
    with open(file, 'rb') as f:
        return pickle.load(f)


class NormalArtifactTableReader:
    def __init__(self, header_tokens):
        self.normal_alt_idx = header_tokens.index("normal_alt")
        self.normal_dp_idx = header_tokens.index("normal_dp")
        self.tumor_alt_idx = header_tokens.index("tumor_alt")
        self.tumor_dp_idx = header_tokens.index("tumor_dp")
        self.downsampling_idx = header_tokens.index("downsampling")
        self.type_idx = header_tokens.index("type")

    def normal_alt_count(self, tokens):
        return int(tokens[self.normal_alt_idx])

    def normal_depth(self, tokens):
        return int(tokens[self.normal_dp_idx])

    def tumor_alt_count(self, tokens):
        return int(tokens[self.tumor_alt_idx])

    def tumor_depth(self, tokens):
        return int(tokens[self.tumor_dp_idx])

    def downsampling(self, tokens):
        return float(tokens[self.downsampling_idx])

    def variant_type(self, tokens):
        return tokens[self.type_idx]


# this takes a table from VariantsToTable and produces a Python list of Datum objects
def read_data(table_file, shuffle=True) -> List[NormalArtifactDatum]:
    data = []

    with open(table_file) as fp:
        reader = NormalArtifactTableReader(fp.readline().split())

        pbar = tqdm(enumerate(fp))
        for n, line in pbar:

            tokens = line.split()

            normal_alt_count = reader.normal_alt_count(tokens)
            normal_depth = reader.normal_depth(tokens)
            tumor_alt_count = reader.tumor_alt_count(tokens)
            tumor_depth = reader.tumor_depth(tokens)
            downsampling = reader.downsampling(tokens)
            variant_type = reader.variant_type(tokens)

            data.append(NormalArtifactDatum(normal_alt_count,normal_depth, tumor_alt_count, tumor_depth, downsampling, variant_type))

    if shuffle:
        random.shuffle(data)
    print("Done")
    return data

def generate_normal_artifact_pickle(table_file, pickle_file):
    data = read_data(table_file)
    make_normal_artifact_pickle(pickle_file, data)

class NormalArtifactDataset(Dataset):
    def __init__(self, pickled_file):
        self.data = load_normal_artifact_pickle(pickled_file)
        random.shuffle(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


# take a list of normal artifact data and package it as several 1D tensors
class NormalArtifactBatch:

    def __init__(self, data: List[NormalArtifactDatum]):
        self._normal_alt = torch.IntTensor([datum.normal_alt_count() for datum in data])
        self._normal_depth = torch.IntTensor([datum.normal_depth() for datum in data])
        self._tumor_alt = torch.IntTensor([datum.tumor_alt_count() for datum in data])
        self._tumor_depth = torch.IntTensor([datum.tumor_depth() for datum in data])
        self._downsampling = torch.FloatTensor([datum.downsampling() for datum in data])
        self._variant_type = [datum.variant_type() for datum in data]
        self._size = len(data)

    def size(self):
        return self._size

    def normal_alt(self):
        return self._normal_alt

    def normal_depth(self):
        return self._normal_depth

    def tumor_alt(self):
        return self._tumor_alt

    def tumor_depth(self):
        return self._tumor_depth

    def downsampling(self):
        return self._downsampling

    def variant_type(self):
        return self._variant_type

def make_normal_artifact_data_loader(dataset: NormalArtifactDataset, batch_size):
    return DataLoader(dataset=dataset, batch_size=batch_size, collate_fn=NormalArtifactBatch)


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
        self.mlp_alpha = networks.MLP(input_to_output_layer_sizes, batch_normalize=False, dropout_p=dropout_p)
        self.mlp_beta = networks.MLP(input_to_output_layer_sizes, batch_normalize=False, dropout_p=dropout_p)
        self.mlp_z = networks.MLP(input_to_output_layer_sizes, batch_normalize=False, dropout_p=dropout_p)

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

        component_log_likelihoods = networks.beta_binomial(n, k, output_alpha, output_beta)
        # 0th dimension is batch, 1st dimension is component.  Sum over the latter
        return torch.logsumexp(log_pi + component_log_likelihoods, dim=1)

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
        betas = [torch.distributions.beta.Beta(torch.FloatTensor([alpha]), torch.FloatTensor([beta])) for (alpha, beta) in zip(output_alpha.detach().numpy(), output_beta.detach().numpy())]

        # list of tensors - the list is over the mixture components, the tensors are over AF values f
        unweighted_log_densities = torch.stack([beta.log_prob(f) for beta in betas], dim=0)
        # unsqueeze to make log_pi a column vector (2D tensor) for broadcasting
        weighted_log_densities = torch.unsqueeze(log_pi, 1) + unweighted_log_densities
        densities = torch.exp(torch.logsumexp(weighted_log_densities, dim=0))

        return validation.simple_plot([(f.detach().numpy(), densities.detach().numpy()," ")], "AF", "density", title)


    def train_model(self, train_loader, valid_loader, num_epochs):
        optimizer = torch.optim.Adam(self.parameters())
        training_metrics = validation.TrainingMetrics()


        for epoch in trange(1, num_epochs + 1, desc="Epoch"):
            for epoch_type in [mutect3.utils.EpochType.TRAIN, mutect3.utils.EpochType.VALID]:
                loader = train_loader if epoch_type == mutect3.utils.EpochType.TRAIN else valid_loader

                epoch_loss = 0
                epoch_count = 0
                for batch in loader:
                    log_likelihoods = self(batch)
                    weights = 1/batch.downsampling()
                    loss = -torch.mean(weights * log_likelihoods)
                    epoch_loss += loss.item()
                    epoch_count += 1

                    if epoch_type == mutect3.utils.EpochType.TRAIN:
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                training_metrics.add("NLL", epoch_type.name, epoch_loss / epoch_count)
            # done with epoch
        # done with training
        # model is trained
        return training_metrics



