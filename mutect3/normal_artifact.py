import pickle
import random
from typing import List
import torch
from torch import nn
import math

from torch.utils.data import Dataset
from tqdm.autonotebook import tqdm
from mutect3 import networks


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
        return self._normal_alt_count

    def tumor_depth(self) -> int:
        return self._normal_depth

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

class NormalArtifactModel(nn.Module):

    def __init__(self, hidden_layers: List[int], dropout_p: float = None):
        super(NormalArtifactModel, self).__init__()
        # we will convert the normal alt and normal depth into the mean and standard deviation of
        # the beta distribution they define.  Then we will predict a mixture of
        all_layers = [2] + hidden_layers + [1]
        self.phi = networks.MLP(all_layers, batch_normalize=False, dropout_p=dropout_p)

    # given normal alt, normal depth, tumor depth, what is the log likelihood of a given tumor alt count
    def log_likelihood(self, datum: NormalArtifactDatum):
        # beta posterior of normal counts with flat 1,1 prior
        alpha = datum.normal_alt_count() + 1
        beta = datum.normal_depth() - datum.normal_alt_count() + 1
        mu = alpha / (alpha + beta)
        sigma = math.sqrt(alpha*beta/((alpha+beta)*(alpha+beta)*(alpha + beta + 1)))

        # parametrize the input as the mean and std of this beta
        normal_tensor = torch.tensor([mu, sigma])

        tumor_alt_count = datum.tumor_alt_count()
        tumor_depth = datum.tumor_depth()

