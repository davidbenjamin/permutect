import math
import random
from typing import List

import torch
from tqdm.autonotebook import tqdm
from torch.utils.data import Dataset, DataLoader, Sampler

from permutect.architecture.permutect_model import PermutectModel
from permutect.data.features_batch import FeaturesBatch
from permutect.data.features_datum import FeaturesDatum
from permutect.data.reads_batch import ReadsBatch
from permutect.data.reads_dataset import ReadsDataset, chunk
from permutect.data.prefetch_generator import prefetch_generator


class FeaturesDataset(Dataset):
    def __init__(self, base_dataset: ReadsDataset,
                 model: PermutectModel,
                 folds_to_use: List[int] = None,
                 base_loader_num_workers=0,
                 base_loader_batch_size=8192):
        self.counts_by_source = base_dataset.counts_by_source
        self.totals_sclt = base_dataset.totals_sclt
        self.label_balancing_weights_sclt = base_dataset.label_balancing_weights_sclt
        self.source_balancing_weights_sct = base_dataset.source_balancing_weights_sct

        self.artifact_data = []
        self.num_folds = base_dataset.num_folds
        self.labeled_indices = [[] for _ in range(self.num_folds)]  # one list for each fold
        self.unlabeled_indices = [[] for _ in range(self.num_folds)]    # ditto
        self.num_base_features = model.pooling_dimension()

        index = 0

        loader = base_dataset.make_data_loader(base_dataset.all_folds() if folds_to_use is None else folds_to_use,
                                               batch_size=base_loader_batch_size,
                                               num_workers=base_loader_num_workers)
        print("making artifact dataset from base dataset")

        is_cuda = model._device.type == 'cuda'
        print(f"Is base model using CUDA? {is_cuda}")

        reads_batch: ReadsBatch
        for reads_batch in tqdm(prefetch_generator(loader), mininterval=60, total=len(loader)):
            with torch.inference_mode():
                representations, _ = model.calculate_representations(reads_batch)

            for representation, data_1d in zip(representations.detach().cpu(), reads_batch.get_data_2d()):
                features_datum = FeaturesDatum(data_1d, representation.detach())
                self.artifact_data.append(features_datum)
                fold = index % self.num_folds
                if features_datum.is_labeled():
                    self.labeled_indices[fold].append(index)
                else:
                    self.unlabeled_indices[fold].append(index)
                index += 1

    def __len__(self):
        return len(self.artifact_data)

    def __getitem__(self, index):
        return self.artifact_data[index]

    # it is often convenient to arbitrarily use the last fold for validation
    def last_fold_only(self):
        return [self.num_folds - 1]  # use the last fold for validation

    def all_but_the_last_fold(self):
        return list(range(self.num_folds - 1))

    def all_but_one_fold(self, fold_to_exclude: int):
        return list(range(fold_to_exclude)) + list(range(fold_to_exclude + 1, self.num_folds))

    def all_folds(self):
        return list(range(self.num_folds))

    def validate_sources(self) -> int:
        num_sources = len(self.counts_by_source.keys())
        if num_sources == 1:
            print("Data come from a single source")
        else:
            sources_list = list(self.counts_by_source.keys())
            sources_list.sort()
            assert sources_list[0] == 0, "There is no source 0"
            assert sources_list[1] == num_sources - 1, f"sources should be 0, 1, 2. . . without gaps, but sources are {sources_list}."

            print(f"Data come from multiple sources, with counts {self.counts_by_source}.")
        return num_sources

    def make_data_loader(self, folds_to_use: List[int], batch_size: int, pin_memory=False, num_workers: int = 0, labeled_only: bool = False, sources_to_use: List[int] = None):
        sampler = SemiSupervisedArtifactBatchSampler(self, batch_size, folds_to_use, labeled_only, sources_to_use)
        return DataLoader(dataset=self, batch_sampler=sampler, collate_fn=FeaturesBatch, pin_memory=pin_memory, num_workers=num_workers)


# make ArtifactBatches that mix different ref, alt counts, labeled, unlabeled
# with an option to emit only labeled data
class SemiSupervisedArtifactBatchSampler(Sampler):
    def __init__(self, dataset: FeaturesDataset, batch_size, folds_to_use: List[int], labeled_only: bool = False, sources_to_use: List[int] = None):
        # combine the index lists of all relevant folds
        self.indices_to_use = []
        source_set = None if sources_to_use is None else set(sources_to_use)

        for fold in folds_to_use:
            indices_in_fold = dataset.labeled_indices[fold] if labeled_only else (dataset.labeled_indices[fold] + dataset.unlabeled_indices[fold])
            if sources_to_use is None:
                source_indices_in_fold = indices_in_fold
            else:
                source_indices_in_fold = [idx for idx in indices_in_fold if dataset[idx].get_source() in source_set]
            self.indices_to_use.extend(source_indices_in_fold)

        self.batch_size = batch_size
        self.num_batches = math.ceil(len(self.indices_to_use) // self.batch_size)

    def __iter__(self):
        random.shuffle(self.indices_to_use)
        batches = chunk(self.indices_to_use, self.batch_size)   # list of lists of indices -- each sublist is a batch
        random.shuffle(batches)

        return iter(batches)

    def __len__(self):
        return self.num_batches

