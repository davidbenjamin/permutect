import psutil
import random
from typing import  List

import torch
from torch.utils.data import DataLoader, IterableDataset

from permutect.data.memory_mapped_data import MemoryMappedData
from permutect.data.reads_datum import ReadsDatum
from permutect.data.reads_batch import ReadsBatch
from permutect.data.batch import BatchProperty, BatchIndexedTensor
from permutect.misc_utils import ConsistentValue
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
    def __init__(self, memory_mapped_data: MemoryMappedData, num_folds: int = None, folds_to_use: List[int] = None):
        """
        :param num_folds:
        """
        super(ReadsDataset, self).__init__()
        self.totals_slvra = BatchIndexedTensor.make_zeros(num_sources=1, include_logits=False, device=torch.device('cpu'))
        # if no folds, no copying is done; otherwise this creates a new file on disk
        self.memory_mapped_data = memory_mapped_data.restrict_to_folds(num_folds, folds_to_use)
        self._size = self.memory_mapped_data.num_data
        self._read_end_indices = self.memory_mapped_data.read_end_indices

        available_memory = psutil.virtual_memory().available
        print(f"Data occupy {memory_mapped_data.size_in_bytes() // 1000000} Mb and the system has {available_memory // 1000000} Mb of RAM available.")

        # copy memory-mapped data to RAM if space allows, otherwise use the memory-mapped data
        self._stacked_reads_re = self.memory_mapped_data.reads_mmap
        self._stacked_data_ve = self.memory_mapped_data.data_mmap

        self._num_read_features, self._num_info_features, self._haplotypes_length = ConsistentValue(), ConsistentValue(), ConsistentValue()
        for datum in self.memory_mapped_data.generate_reads_data():
            self.totals_slvra.record_datum(datum)
            self._num_read_features.check(datum.num_read_features())
            self._num_info_features.check(len(datum.get_info_1d()))
            self._haplotypes_length.check(len(datum.get_haplotypes_1d()))


    def num_read_features(self) -> int:
        return self._num_read_features.value

    def num_info_features(self) -> int:
        return self._num_info_features.value

    def haplotypes_length(self) -> int:
        return self._haplotypes_length.value

    # this is not required for an IterableDataset, but it can't hurt!
    def __len__(self):
        return self._size

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
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
            chunk_read_end_idx = self._read_end_indices[chunk_end_idx - 1]

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

    def make_data_loader(self, batch_size: int, pin_memory=False, num_workers: int = 0):
        return DataLoader(dataset=self, batch_size=batch_size, collate_fn=ReadsBatch, pin_memory=pin_memory,
                          num_workers=num_workers, prefetch_factor=2 if num_workers > 0 else None, persistent_workers=num_workers > 0)


# ex: chunk([a,b,c,d,e], 3) = [[a,b,c], [d,e]]
def chunk(lis, chunk_size):
    return [lis[i:i + chunk_size] for i in range(0, len(lis), chunk_size)]


