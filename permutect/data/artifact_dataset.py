import math
import random
from typing import List

import numpy as np
from torch.utils.data import Dataset, DataLoader, Sampler

from permutect.architecture.base_model import BaseModel
from permutect.data.base_datum import ArtifactDatum, ArtifactBatch
from permutect.data.base_dataset import BaseDataset, chunk


# given a ReadSetDataset, apply a BaseModel to get a dataset (in RAM, maybe implement memory map later)
# of RepresentationReadSets
class ArtifactDataset(Dataset):
    def __init__(self, base_dataset: BaseDataset, base_model: BaseModel, folds_to_use: List[int] = None):

        self.artifact_totals = base_dataset.artifact_totals
        self.non_artifact_totals = base_dataset.non_artifact_totals
        self.artifact_data = []
        self.num_folds = base_dataset.num_folds
        self.labeled_indices = [[] for _ in range(self.num_folds)]  # one list for each fold
        self.unlabeled_indices = [[] for _ in range(self.num_folds)]    # ditto
        self.num_base_features = base_model.output_dimension()

        index = 0

        loader = base_dataset.make_data_loader(base_dataset.all_folds() if folds_to_use is None else folds_to_use, batch_size=256)
        for base_batch in loader:
            representations = base_model.calculate_representations(base_batch).detach()
            for representation, base_datum in zip(representations, base_batch.original_list()):
                artifact_datum = ArtifactDatum(base_datum, representation)
                self.artifact_data.append(artifact_datum)
                fold = index % self.num_folds
                if artifact_datum.is_labeled():
                    self.labeled_indices[fold].append(index)
                else:
                    self.unlabeled_indices[fold].append(index)
                index += 1

    def __len__(self):
        return len(self.artifact_data)

    def __getitem__(self, index):
        return self.artifact_data[index]

    def artifact_to_non_artifact_ratios(self):
        return self.artifact_totals / self.non_artifact_totals

    def total_labeled_and_unlabeled(self):
        total_labeled = np.sum(self.artifact_totals + self.non_artifact_totals)
        return total_labeled, len(self) - total_labeled

    # it is often convenient to arbitrarily use the last fold for validation
    def last_fold_only(self):
        return [self.num_folds - 1]  # use the last fold for validation

    def all_but_the_last_fold(self):
        return list(range(self.num_folds - 1))

    def all_but_one_fold(self, fold_to_exclude: int):
        return list(range(fold_to_exclude)) + list(range(fold_to_exclude + 1, self.num_folds))

    def all_folds(self):
        return list(range(self.num_folds))

    def make_data_loader(self, folds_to_use: List[int], batch_size: int, pin_memory=False, num_workers: int = 0):
        sampler = SemiSupervisedRepresentationBatchSampler(self, batch_size, folds_to_use)
        return DataLoader(dataset=self, batch_sampler=sampler, collate_fn=ArtifactBatch, pin_memory=pin_memory, num_workers=num_workers)


# make RepresentationReadSetBatches that are all supervised or all unsupervised -- ref and alt counts may be disparate
class SemiSupervisedRepresentationBatchSampler(Sampler):
    def __init__(self, dataset: ArtifactDataset, batch_size, folds_to_use: List[int]):
        # combine the index lists of all relevant folds
        self.labeled_indices = []
        self.unlabeled_indices = []
        for fold in folds_to_use:
            self.labeled_indices.extend(dataset.labeled_indices[fold])
            self.unlabeled_indices.extend(dataset.unlabeled_indices[fold])

        self.batch_size = batch_size
        self.num_batches = sum(math.ceil(len(indices) // self.batch_size) for indices in
                               (self.labeled_indices, self.unlabeled_indices))

    def __iter__(self):
        batches = []    # list of lists of indices -- each sublist is a batch
        for index_list in (self.labeled_indices, self.unlabeled_indices):
            random.shuffle(index_list)
            batches.extend(chunk(index_list, self.batch_size))
        random.shuffle(batches)

        return iter(batches)

    def __len__(self):
        return self.num_batches

