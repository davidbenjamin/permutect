import random
from typing import List, Iterable

import pandas as pd
from torch.utils.data import Dataset, DataLoader

from mutect3.data.normal_artifact_batch import NormalArtifactBatch
from mutect3.data.normal_artifact_datum import NormalArtifactDatum


def read_normal_artifact_data(table_file, shuffle=True) -> List[NormalArtifactDatum]:
    df = pd.read_table(table_file, header=0)
    df = df.astype({"normal_alt": int, "normal_dp": int, "tumor_alt": int, "tumor_dp": int, "downsampling": float,
                    "type": str})

    data = []
    for _, row in df.iterrows():
        data.append(NormalArtifactDatum(row['normal_alt'], row['normal_dp'], row['tumor_alt'], row['tumor_dp'],
                                        row['downsampling'], row['type']))

    if shuffle:
        random.shuffle(data)
    return data


class NormalArtifactDataset(Dataset):
    def __init__(self, table_files=[], data: Iterable[NormalArtifactDatum] = []):
        self.data = []
        for table_file in table_files:
            self.data.extend(read_normal_artifact_data(table_file))
        self.data.extend(data)
        random.shuffle(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


def make_normal_artifact_data_loader(na_dataset, batch_size):
    return DataLoader(dataset=na_dataset, batch_size=batch_size, collate_fn=NormalArtifactBatch)