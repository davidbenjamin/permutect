import math
import psutil
import random
from typing import  List, Generator

import torch
from torch.distributed.rpc import get_worker_info
from torch.utils.data import Dataset, DataLoader, IterableDataset
from torch.utils.data.sampler import Sampler

from permutect.data.memory_mapped_data import MemoryMappedData
from permutect.data.reads_datum import ReadsDatum
from permutect.data.reads_batch import ReadsBatch
from permutect.data.batch import BatchProperty, BatchIndexedTensor
from permutect.utils.enums import Variation, Label

WEIGHT_PSEUDOCOUNT = 10


# it is often convenient to arbitrarily use the last fold for validation
def last_fold_only(num_folds: int):
    return [num_folds - 1]  # use the last fold for validation

def all_but_the_last_fold(num_folds: int):
    return list(range(num_folds - 1))

def all_but_one_fold(num_folds: int, fold_to_exclude: int):
    return list(range(fold_to_exclude)) + list(range(fold_to_exclude + 1, num_folds))

def all_folds(num_folds: int):
    return list(range(num_folds))

def ratio_with_pseudocount(a, b):
    return (a + WEIGHT_PSEUDOCOUNT) / (b + WEIGHT_PSEUDOCOUNT)


class ReadsDataset(IterableDataset):
    def __init__(self, memory_mapped_data: MemoryMappedData, num_folds: int = 1, folds_to_use: List[int] = None):
        """
        :param num_folds:
        """
        super(ReadsDataset, self).__init__()
        self.totals_slvra = BatchIndexedTensor.make_zeros(num_sources=1, include_logits=False, device=torch.device('cpu'))
        self.memory_mapped_data = memory_mapped_data
        self._size = memory_mapped_data.num_data
        self._read_end_indices = memory_mapped_data.read_end_indices

        available_memory = psutil.virtual_memory().available
        fits_in_ram = memory_mapped_data.size_in_bytes() < 0.6 * available_memory
        self._memory_map_mode = fits_in_ram
        print(f"Data occupy {memory_mapped_data.size_in_bytes() // 1000000} Mb and the system has {available_memory // 1000000} Mb of RAM available.")

        # copy memory-mapped data to RAM if space allows, otherwise use the memory-mapped data
        self._stacked_reads_re = memory_mapped_data.reads_mmap.copy() if fits_in_ram else memory_mapped_data.reads_mmap
        self._stacked_data_ve = memory_mapped_data.data_mmap.copy() if fits_in_ram else memory_mapped_data.data_mmap

        self.indices_in_use = []

        folds_set = set(all_folds(num_folds) if folds_to_use is None else folds_to_use)

        # TODO: does this still work when we implement it as an IterableDataset?
        for n, datum in enumerate(self):
            self.totals_slvra.record_datum(datum)
            fold = n % num_folds
            if fold in folds_set:
                self.indices_in_use.append(n)

        first_datum: ReadsDatum = self[0]
        self.num_read_features = first_datum.num_read_features()
        self.num_info_features = len(first_datum.get_info_1d())
        self.haplotypes_length = len(first_datum.get_haplotypes_1d())

    # this is not required for an IterableDataset, but it can't hurt!
    def __len__(self):
        return self._size

    # TODO: I hope this is never used since we have deleted it
    # TODO: I think it was omnly used indirectly through the Sampler/DataLoader
    #def __getitem__(self, index):
    #    read_start_index = 0 if index == 0 else self._read_end_indices[index - 1]
    #    read_end_index = self._read_end_indices[index]

    #    return ReadsDatum(datum_array=self._stacked_data_ve[index],
    #                          compressed_reads_re=self._stacked_reads_re[read_start_index:read_end_index])

    def __iter__(self):
        worker_info = get_worker_info()
        worker_id = 0 if worker_info is None else worker_info.id
        num_workers = 1 if worker_info is None else worker_info.num_workers

        num_data_per_worker = self._size // num_workers
        num_bytes_per_worker = self.memory_mapped_data.size_in_bytes() // num_workers

        # note: this is the total available system memory, not per process
        total_available_memory_in_bytes = psutil.virtual_memory().available
        available_memory_per_worker = total_available_memory_in_bytes // num_workers

        # we want the amount of memory loaded to RAM at any given time to be well below the total available memory
        # thus we multiply by a cautious fudge factor
        fudge_factor = 4
        chunks_per_worker = 1 + ((fudge_factor * num_bytes_per_worker) // available_memory_per_worker)

        # The way multiple DataLoader workers work in PyTorch is not so obvious.
        # 1) There is an original Dataset
        # 2) There is a DataLoader associated with that Dataset that creates one copy of the Dataset for each worker
        #    process.  I believe that tensors and numpy arrays in RAM are deep-copied in this process, and I'm pretty
        #   sure that the memory-map file backing the dataset is not copied.
        # 3) when iter() is called it is usually done so implicitly via the DataLoader, in which case "self" refers to
        # one of the copies and we are only responsible for part of the memory-mapped data.
        # 4) this part of the memory-mapped data might be too big for available RAM, so we load one contiguous chunk
        # (this is a chunk within the subset of data that this worker is responsible for) into RAM at a time.

        # example of specific number: num_data = 21, num_workers = 4, then data_per_worker is 5 and the index ranges
        # are [0,5), [5,10), [10,15), [15,21)
        worker_start_idx = worker_id * num_data_per_worker
        worker_end_idx = (worker_id + 1) * num_data_per_worker if worker_id < num_workers - 1 else self._size

        num_data_for_this_worker = worker_end_idx - worker_start_idx
        data_per_chunk = num_data_per_worker // chunks_per_worker
        chunks = list(range(chunks_per_worker))
        random.shuffle(chunks)
        for chunk in chunks:
            chunk_start_idx = worker_start_idx + chunk * data_per_chunk
            chunk_end_idx = (worker_start_idx + (chunk + 1) * data_per_chunk) if (chunk == chunks_per_worker - 1) else worker_end_idx
            chunk_read_start_idx = 0 if chunk_start_idx == 0 else self._read_end_indices[chunk_start_idx - 1]
            chunk_read_end_idx = self._read_end_indices[chunk_end_idx]

            # TODO: I think the .copy() is necessary to copy the slice of the memory-map from disk into RAM
            # these operations should be really fast because it's all sequential access
            chunk_data_ve = self._stacked_data_ve[chunk_start_idx:chunk_end_idx].copy()
            chunk_reads_re = self._stacked_reads_re[chunk_read_start_idx:chunk_read_end_idx].copy()
            chunk_read_end_indices = self._read_end_indices[chunk_start_idx:chunk_end_idx]

            # now that it's all in RAM, we can yield in randomly-accessed order
            indices = list(range(len(chunk_data_ve)))
            random.shuffle(indices)

            for idx in indices:
                read_start_idx = 0 if idx == 0 else chunk_read_end_indices[idx - 1]
                read_end_idx = chunk_read_end_indices[idx]
                yield ReadsDatum(datum_array=chunk_data_ve[idx], compressed_reads_re=chunk_reads_re[read_start_idx:read_end_idx])



    def num_sources(self) -> int:
        return self.totals_slvra.num_sources()

    def report_totals(self):
        totals_slv = self.totals_slvra.get_marginal((BatchProperty.SOURCE, BatchProperty.LABEL, BatchProperty.VARIANT_TYPE))
        for source in range(len(totals_slv)):
            print(f"Data counts for source {source}:")
            for var_type in Variation:
                print(f"Data counts for variant type {var_type.name}:")
                for label in Label:
                    print(f"{label.name}: {int(totals_slv[source, label, var_type].item())}")


    def validate_sources(self) -> int:
        num_sources = self.num_sources()
        totals_by_source_s = self.totals_slvra.get_marginal((BatchProperty.SOURCE,))
        if num_sources == 1:
            print("Data come from a single source")
        else:
            for source in range(self.num_sources()):
                assert totals_by_source_s[source].item() >= 1, f"No data for source {source}."
            print(f"Data come from multiple sources, with counts {totals_by_source_s.cpu().tolist()}.")
        return num_sources

    def make_data_loader(self, batch_size: int, pin_memory=False, num_workers: int = 0, labeled_only: bool = False):
        sampler = SemiSupervisedBatchSampler(self, batch_size, labeled_only)
        return DataLoader(dataset=self, batch_sampler=sampler, collate_fn=ReadsBatch, pin_memory=pin_memory, num_workers=num_workers)


# from a generator that yields BaseDatum(s), create a generator that yields the two numpy arrays needed to reconstruct the datum
def make_flattened_tensor_generator(reads_data_generator: Generator[ReadsDatum, None, None]):
    for reads_datum in reads_data_generator:
        yield reads_datum.get_compressed_reads_re()
        yield reads_datum.get_array_1d()


# ex: chunk([a,b,c,d,e], 3) = [[a,b,c], [d,e]]
def chunk(lis, chunk_size):
    return [lis[i:i + chunk_size] for i in range(0, len(lis), chunk_size)]


# Labeled and unlabeled data are mixed.
# the artifact model handles weighting the losses to compensate for class imbalance between supervised and unsupervised
# thus the sampler is not responsible for balancing the data
class SemiSupervisedBatchSampler(Sampler):
    def __init__(self, dataset: ReadsDataset, batch_size: int, labeled_only: bool = False):
        # combine the index maps of all relevant folds
        self.indices_to_use = []
        for idx in dataset.indices_in_use:
            if not (labeled_only and dataset[idx].get_label() == Label.UNLABELED):
                self.indices_to_use.append(idx)

        self.batch_size = batch_size
        self.num_batches = math.ceil(len(self.indices_to_use) / self.batch_size)

    def __iter__(self):
        batches = []    # list of lists of indices -- each sublist is a batch
        random.shuffle(self.indices_to_use)
        batches.extend(chunk(self.indices_to_use, self.batch_size))
        random.shuffle(batches)

        return iter(batches)

    def __len__(self):
        return self.num_batches

