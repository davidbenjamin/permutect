from __future__ import annotations
import os
import tarfile
import tempfile
from tempfile import NamedTemporaryFile
from typing import Generator, List

import numpy as np
import torch

from permutect.data.datum import Datum, DATUM_ARRAY_DTYPE
from permutect.data.reads_datum import ReadsDatum, READS_ARRAY_DTYPE

# numpy.save appends .npy if the extension doesn't already include it.  We preempt this behavior.
SUFFIX_FOR_DATA_MMAP = ".data_mmap.npy"
SUFFIX_FOR_READS_MMAP = ".reads_mmap.npy"
SUFFIX_FOR_METADATA = ".metadata.npy"


class MemoryMappedData:
    """
    wrapper for
        1) memory-mapped numpy file for stacked 1D data arrays.  The nth row of this array is the 1D array for the nth Datum.
        2) memory-mapped numpy file for 2D stacked reads array.  The rows of this array are the ref reads of the 0th ReadsDatum,
            the alt reads of the 0th ReadsDatum, the ref reads of the 1st ReadsDatum etc.

        NOTE: the memory-mapped files may be larger than necessary.  That is, there may be junk space in the files that
        does not correspond to actual data.  The num_data and num_reads tell us how far into the files is actual data.
    """

    def __init__(self, data_mmap, num_data, reads_mmap, num_reads):
        self.data_mmap = data_mmap
        self.reads_mmap = reads_mmap
        self.num_data = num_data
        self.num_reads = num_reads

        # note: this can hold indices up to a bit over 4 billion, which is probably bigger than any training dataset we'll need
        self.read_end_indices = np.zeros(shape=(num_data,), dtype=np.uint32)
        idx = 0
        for n in range(num_data):
            idx += Datum(data_mmap[n]).get_read_count()
            self.read_end_indices[n] = idx

    def __len__(self):
        return self.num_data

    def size_in_bytes(self):
        return self.data_mmap.nbytes + self.reads_mmap.nbytes

    def generate_reads_data(self, num_folds: int = None, folds_to_use: List[int] = None) -> Generator[ReadsDatum, None, None]:
        folds_set = None if folds_to_use is None else set(folds_to_use)
        print("Generating ReadsDatum objects from memory-mapped data.")
        assert self.reads_mmap.dtype == READS_ARRAY_DTYPE
        count = 0
        for idx in range(self.num_data):
            if folds_to_use is None or (idx % num_folds in folds_set):
                data_array = self.data_mmap[idx]
                reads_array = self.reads_mmap[0 if idx == 0 else self.read_end_indices[idx - 1]:self.read_end_indices[idx]]
                yield ReadsDatum(datum_array=data_array, compressed_reads_re=reads_array)
                count += 1
        print(f"generated {count} objects.")

    def restrict_to_folds(self, num_folds: int = None, folds_to_use: List[int] = None) -> MemoryMappedData:
        if folds_to_use is None:
            return self
        else:
            print(f"Restricting to folds {folds_to_use} out of {num_folds} total folds.")
            proportion = len(folds_to_use) / num_folds
            fudge_factor = 1.1
            estimated_num_data = int(self.num_data * proportion * fudge_factor)
            estimated_num_reads = int(self.num_reads * proportion * fudge_factor)
            reads_datum_source = self.generate_reads_data(num_folds=num_folds, folds_to_use=folds_to_use)
            return MemoryMappedData.from_generator(reads_datum_source, estimated_num_data, estimated_num_reads)

    def restrict_to_labeled_only(self) -> MemoryMappedData:
        print("Restricting dataset to labeled data only.")
        labeled_count, total = 0, 0
        # estimated the proportion of labeled data
        for n, datum in enumerate(self.generate_reads_data()):
            if n > 1000:
                break
            total += 1
            labeled_count += 1 if datum.is_labeled() else 0

        labeled_proportion = labeled_count / total
        fudge_factor = 1.1
        estimated_num_reads = self.num_reads * labeled_proportion * fudge_factor
        estimated_num_data = self.num_data * labeled_proportion * fudge_factor

        reads_datum_source = (datum for datum in self.generate_reads_data() if datum.is_labeled())
        return MemoryMappedData.from_generator(reads_datum_source, estimated_num_data, estimated_num_reads)

    def save_to_tarfile(self, output_tarfile):
        """
        It seems a little odd to save to disk when memory-mapped files are already on disk, but:
            1) the files don't know their dtype and shape
            2) the files don't know how much of the data are actually used
            3) the files might be temporary files and won't persist after the Python program executes
            4) it's convenient to package things as a single tarfile
        :return:
        """
        # num_data, data dimension; num_reads, reads dimension
        metadata = np.array([self.num_data, self.data_mmap.shape[-1], self.num_reads, self.reads_mmap.shape[-1]], dtype=np.uint32)
        metadata_file = NamedTemporaryFile(suffix=SUFFIX_FOR_METADATA)
        torch.save(metadata, metadata_file.name)

        # For some reason, self.data_mmap.filename and self.reads_mmap.filename point to an empty file.  I have no clue why this is,
        # so at risk of redundant copying I just use numpy's save function, followed later by numpy.open_memmap
        data_file = NamedTemporaryFile(suffix=SUFFIX_FOR_DATA_MMAP)
        np.save(data_file.name, self.data_mmap)

        reads_file = NamedTemporaryFile(suffix=SUFFIX_FOR_READS_MMAP)
        np.save(reads_file.name, self.reads_mmap)

        with tarfile.open(output_tarfile, "w") as output_tar:
            output_tar.add(metadata_file.name, arcname=("metadata" + SUFFIX_FOR_METADATA))
            output_tar.add(data_file.name, arcname=("data_array" + SUFFIX_FOR_DATA_MMAP))
            output_tar.add(reads_file.name, arcname=("reads_array" + SUFFIX_FOR_READS_MMAP))

    # Load the list of objects back from the .npy file
    # Remember to set allow_pickle=True when loading as well
    @classmethod
    def load_from_tarfile(cls, data_tarfile) -> MemoryMappedData:
        temp_dir = tempfile.TemporaryDirectory()

        with tarfile.open(data_tarfile, 'r') as tar:
            for member in tar.getmembers():
                if member.isfile():
                    tar.extract(member, path=temp_dir.name)

        metadata_files = [os.path.abspath(os.path.join(temp_dir.name, p)) for p in os.listdir(temp_dir.name) if p.endswith(SUFFIX_FOR_METADATA)]
        data_files = [os.path.abspath(os.path.join(temp_dir.name, p)) for p in os.listdir(temp_dir.name) if p.endswith(SUFFIX_FOR_DATA_MMAP)]
        reads_files = [os.path.abspath(os.path.join(temp_dir.name, p)) for p in os.listdir(temp_dir.name) if p.endswith(SUFFIX_FOR_READS_MMAP)]
        assert len(metadata_files) == 1
        assert len(data_files) == 1
        assert len(reads_files) == 1

        loaded_metadata = torch.load(metadata_files[0])
        num_data, data_dim, num_reads, reads_dim = loaded_metadata[0], loaded_metadata[1], loaded_metadata[2], loaded_metadata[3]

        # NOTE: the original file may have had excess space due to the O(N) amortized growing scheme
        # if we load the same file with the actual num_data, as opposed to the capacity, it DOES work correctly
        data_mmap = np.lib.format.open_memmap(data_files[0], mode='r', shape=(num_data, data_dim))
        reads_mmap = np.lib.format.open_memmap(reads_files[0], mode='r', shape=(num_reads, reads_dim))

        return cls(data_mmap=data_mmap, num_data=num_data, reads_mmap=reads_mmap, num_reads=num_reads)

    @classmethod
    def from_generator(cls, reads_datum_source, estimated_num_data, estimated_num_reads) -> MemoryMappedData:
        """
        Write RawUnnormalizedReadsDatum or ReadsDatum data to memory maps.  We set the file sizes to initial guesses but if these are outgrown we copy
        data to larger files, just like the amortized O(N) append operation on lists.

        :param reads_datum_source: an Iterable or Generator of ReadsDatum
        :param estimated_num_data: initial estimate of how much capacity is needed
        :param estimated_num_reads:
        :return:
        """
        num_data, num_reads = 0, 0
        data_capacity, reads_capacity = 0, 0
        data_mmap, reads_mmap = None, None

        datum: ReadsDatum
        for datum in reads_datum_source:
            data_array = datum.get_array_1d()
            reads_array = datum.get_reads_array_re()    # this works both for raw unnormalized data and the compressed reads of ReadsDatum

            num_data += 1
            num_reads += len(reads_array)

            # double capacity or set to initial estimate, create new file and mmap, copy old data
            if num_data > data_capacity:
                data_capacity = estimated_num_data if data_capacity == 0 else data_capacity*2
                data_file = NamedTemporaryFile(suffix=SUFFIX_FOR_DATA_MMAP)
                old_data_mmap = data_mmap
                data_mmap = np.memmap(data_file.name, dtype=data_array.dtype, mode='w+', shape=(data_capacity, data_array.shape[-1]))
                if old_data_mmap is not None:
                    data_mmap[:len(old_data_mmap)] = old_data_mmap

            # likewise for reads
            if num_reads > reads_capacity:
                reads_capacity = estimated_num_reads if reads_capacity == 0 else reads_capacity * 2
                reads_file = NamedTemporaryFile(suffix=SUFFIX_FOR_READS_MMAP)
                old_reads_mmap = reads_mmap
                reads_mmap = np.memmap(reads_file.name, dtype=reads_array.dtype, mode='w+', shape=(reads_capacity, reads_array.shape[-1]))
                if old_reads_mmap is not None:
                    reads_mmap[:len(old_reads_mmap)] = old_reads_mmap

            # write new data
            data_mmap[num_data - 1] = data_array
            reads_mmap[num_reads - len(reads_array):num_reads] = reads_array

        data_mmap.flush()
        reads_mmap.flush()

        return cls(data_mmap=data_mmap, num_data=num_data, reads_mmap=reads_mmap, num_reads=num_reads)