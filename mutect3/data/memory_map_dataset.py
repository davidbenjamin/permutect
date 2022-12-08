import tarfile

import torch
import os
from torch.utils.data import Dataset
from mmap_ninja.ragged import RaggedMmap
import tempfile

from mutect3.data.read_set import ReadSet, load_list_of_read_sets


TENSORS_PER_READ_SET = 5


class MemoryMapDataset(Dataset):
    def __init__(self, data_tarfile, metadata_file):
        super(MemoryMapDataset, self).__init__()

        self.memory_map_dir = tempfile.TemporaryDirectory()

        RaggedMmap.from_generator(out_dir=self.memory_map_dir.name,
            sample_generator=make_flattened_tensor_generator(make_read_set_generator_from_tarfile(data_tarfile)),
            batch_size=10000,
            verbose=True)

        self.memory_map = RaggedMmap(self.memory_map_dir.name)

        # get metadata -- this parallels the order of saving in preprocess_dataset.py
        self.num_read_features, self.num_info_features, self.ref_sequence_length, self.num_training_data, \
            self.artifact_totals, self.non_artifact_totals = torch.load(metadata_file)

    def __getitem__(self, item):
        bottom_index = item * TENSORS_PER_READ_SET

        # The order here corresponds to the order of yield statements within make_flattened_tensor_generator()
        return ReadSet(ref_sequence_tensor=self.memory_map[bottom_index + 2],
                       ref_tensor=self.memory_map[bottom_index],
                       alt_tensor=self.memory_map[bottom_index + 1],
                       info_tensor=self.memory_map[bottom_index + 3],
                       label=self.memory_map[bottom_index + 4])


# from a generator that yields read sets, create a generator that yields
# ref tensor, alt tensor, ref sequence tensor, info tensor, label tensor, ref tensor alt tensor. . .
def make_flattened_tensor_generator(read_set_generator):
    for read_set in read_set_generator:
        yield read_set.ref_tensor
        yield read_set.alt_tensor
        yield read_set.ref_sequence_tensor
        yield read_set.info_tensor

        # TODO: this is currently not a tensor of any sort, let a lone a numpy tensor!!!
        yield read_set.label

        
def make_read_set_generator_from_tarfile(data_tarfile):
    # extract the tarfile to a temporary directory that will be cleaned up when the program ends
    temp_dir = tempfile.TemporaryDirectory()
    tar = tarfile.open(data_tarfile)
    tar.extractall(temp_dir.name)
    tar.close()
    data_files = [os.path.abspath(os.path.join(temp_dir.name, p)) for p in os.listdir(temp_dir.name)]

    # recall each data file saves a list of ReadSets via ReadSet.save_list_of_read_sets
    # we reverse it with ReadSet.load_list_of_read_sets
    for file in data_files:
        for datum in load_list_of_read_sets(file):
            yield datum


