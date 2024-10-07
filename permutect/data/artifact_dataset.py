import math
import random
from typing import List
from tqdm.autonotebook import tqdm

import numpy as np
from torch.utils.data import Dataset, DataLoader, Sampler

from permutect.architecture.base_model import BaseModel
from permutect.data.base_datum import ArtifactDatum, ArtifactBatch
from permutect.data.base_dataset import BaseDataset, chunk




# given a ReadSetDataset, apply a BaseModel to get an ArtifactDataset (in RAM, maybe implement memory map later)
# of RepresentationReadSets
class ArtifactDataset(Dataset):
    def __init__(self, base_dataset: BaseDataset, base_model: BaseModel, folds_to_use: List[int] = None):
        self.counts_by_source = base_dataset.counts_by_source

        self.totals = base_dataset.totals
        self.weights = base_dataset.weights

        self.artifact_totals = base_dataset.artifact_totals
        self.unlabeled_totals = base_dataset.unlabeled_totals
        self.non_artifact_totals = base_dataset.non_artifact_totals
        self.unlabeled_totals_by_count = base_dataset.unlabeled_totals_by_count
        self.artifact_totals_by_count = base_dataset.artifact_totals_by_count
        self.non_artifact_totals_by_count = base_dataset.non_artifact_totals_by_count

        self.artifact_data = []
        self.num_folds = base_dataset.num_folds
        self.labeled_indices = [[] for _ in range(self.num_folds)]  # one list for each fold
        self.unlabeled_indices = [[] for _ in range(self.num_folds)]    # ditto
        self.num_base_features = base_model.output_dimension()
        self.num_ref_alt_features = base_model.ref_alt_seq_embedding_dimension()


        index = 0

        loader = base_dataset.make_data_loader(base_dataset.all_folds() if folds_to_use is None else folds_to_use, batch_size=256)
        print("making artifact dataset from base dataset")
        pbar = tqdm(enumerate(loader), mininterval=60)
        for n, base_batch in pbar:
            representations, ref_alt_seq_embeddings = base_model.calculate_representations(base_batch)
            for representation, ref_alt_emb, base_datum in zip(representations.detach(), ref_alt_seq_embeddings.detach(), base_batch.original_list()):
                artifact_datum = ArtifactDatum(base_datum, representation.detach(), ref_alt_emb)
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

    def artifact_to_non_artifact_ratios_by_count(self, count: int):
        # giving each a regularizing pseudocount of 1 -- not sure if this is a wise idea
        return (self.artifact_totals_by_count[count] + 1) / (self.non_artifact_totals_by_count[count] + 1)

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

    def make_data_loader(self, folds_to_use: List[int], batch_size: int, pin_memory=False, num_workers: int = 0, labeled_only: bool = False):
        sampler = SemiSupervisedArtifactBatchSampler(self, batch_size, folds_to_use, labeled_only)
        return DataLoader(dataset=self, batch_sampler=sampler, collate_fn=ArtifactBatch, pin_memory=pin_memory, num_workers=num_workers)


# make ArtifactBatches that mix different ref, alt counts, labeled, unlabeled
# with an option to emit only labeled data
class SemiSupervisedArtifactBatchSampler(Sampler):
    def __init__(self, dataset: ArtifactDataset, batch_size, folds_to_use: List[int], labeled_only: bool = False):
        # combine the index lists of all relevant folds
        self.indices_to_use = []

        for fold in folds_to_use:
            self.indices_to_use.extend(dataset.labeled_indices[fold])
            if not labeled_only:
                self.indices_to_use.extend(dataset.unlabeled_indices[fold])

        self.batch_size = batch_size
        self.num_batches = math.ceil(len(self.indices_to_use) // self.batch_size)

    def __iter__(self):
        random.shuffle(self.indices_to_use)
        batches = chunk(self.indices_to_use, self.batch_size)   # list of lists of indices -- each sublist is a batch
        random.shuffle(batches)

        return iter(batches)

    def __len__(self):
        return self.num_batches

