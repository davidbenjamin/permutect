import pickle
import random
from typing import List
import torch
from torch import nn
import math

from torch.utils.data import Dataset, DataLoader
from tqdm.autonotebook import tqdm, trange
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

def generate_normal_artifact_pickle(table_file, pickle_dir, pickle_prefix):
    data = read_data(table_file)
    make_normal_artifact_pickle(pickle_dir + pickle_prefix + '-normal-artifact.pickle', data)

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
        # we will convert the normal alt and normal depth into the shape parameters of
        # the beta distribution they define.
        mu_sigma_1_layers = [2] + hidden_layers + [1]
        self.mlp_alpha = networks.MLP(mu_sigma_1_layers, batch_normalize=False, dropout_p=dropout_p)
        self.mlp_beta = networks.MLP(mu_sigma_1_layers, batch_normalize=False, dropout_p=dropout_p)

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
        output_alpha = torch.squeeze(torch.exp(self.mlp_alpha(mu_sigma)))
        output_beta = torch.squeeze(torch.exp(self.mlp_beta(mu_sigma)))

        return output_alpha, output_beta

    # given normal alts, normal depths, tumor depths, what is the log likelihood of given tumor alt counts
    # that is, this returns the 1D tensor of log likelihoods
    def log_likelihood(self, batch: NormalArtifactBatch):
        output_alpha, output_beta = self.get_beta_parameters(batch)

        # the log likelihood is the beta binomial log likelihood
        tumor_alt_count = batch.tumor_alt()
        tumor_depth = batch.tumor_depth()

        # depth and alt count are the n and k of the beta binomial, 1D tensors indexed by the batch
        # output alpha and output beta are 1D tensors of the same length
        # beta_binomial(n,k,alpha,beta)[i,j] is the beta binomial log likelihood of n[i],k[i], alpha[j], beta[j]
        return networks.beta_binomial(tumor_depth, tumor_alt_count, output_alpha, output_beta)

    def train_model(self, train_loader, valid_loader, num_epochs):
        optimizer = torch.optim.Adam(self.parameters())
        training_metrics = validation.TrainingMetrics()


        for epoch in trange(1, num_epochs + 1, desc="Epoch"):
            for epoch_type in [networks.EpochType.TRAIN, networks.EpochType.VALID]:
                loader = train_loader if epoch_type == networks.EpochType.TRAIN else valid_loader

                epoch_loss = 0
                epoch_count = 0
                for batch in loader:
                    log_likelihoods = self(batch)
                    loss = -torch.mean(log_likelihoods)
                    epoch_loss += loss.item()
                    epoch_count += 1

                    if epoch_type == networks.EpochType.TRAIN:
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                training_metrics.add("NLL", epoch_type.name, epoch_loss / epoch_count)
            # done with epoch
        # done with training
        # model is trained
        return training_metrics



