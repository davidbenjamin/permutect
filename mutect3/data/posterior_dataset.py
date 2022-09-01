from torch.utils.data import Dataset, DataLoader
from typing import Iterable
import random

from mutect3.data.posterior_batch import PosteriorBatch
from mutect3.data.posterior_datum import PosteriorDatum


class PosteriorDataset(Dataset):
    def __init__(self, data: Iterable[PosteriorDatum], shuffle: bool = True):
        self.data = data

        if shuffle:
            random.shuffle(self.data)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> PosteriorDatum:
        return self.data[index]


def make_posterior_data_loader(dataset: PosteriorDataset, batch_size: int):
    return DataLoader(dataset=dataset, batch_size=batch_size, collate_fn=PosteriorBatch)