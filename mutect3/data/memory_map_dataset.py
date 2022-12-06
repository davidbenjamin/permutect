from torch.utils.data import Dataset
from mmap_ninja.ragged import RaggedMmap
import tempfile

from mutect3.data.read_set import ReadSet


TENSORS_PER_READ_SET = 5


# from a generator that yields read sets, create a generator that yields
# ref tensor, alt tensor, ref sequence tensor, info tensor, label tensor, ref tensor alt tensor. . .
def make_flattened_tensor_generator(read_set_generator):
    for read_set in read_set_generator:
        # TODO: these need to be numpy tensors.  Perhaps easiest is to make them numpy tensors from the beginning
        # TODO: and only switch to PyTorch once we actually create a batch
        yield read_set.ref_tensor
        yield read_set.alt_tensor
        yield read_set.ref_sequence_tensor
        yield read_set.info_tensor

        # TODO: this is currently not a tensor of any sort, let a lone a numpy tensor!!!
        yield read_set.label


class MemoryMapDataset(Dataset):
    def __init__(self, read_set_generator, shuffle: bool = True, normalize: bool = True):
        super(MemoryMapDataset, self).__init__()

        self.memory_map_dir = tempfile.TemporaryDirectory()

        RaggedMmap.from_generator(out_dir=self.memory_map_dir.name,
            sample_generator=make_flattened_tensor_generator(read_set_generator),
            batch_size=10000,
            verbose=True)

        self.memory_map = RaggedMmap(self.memory_map_dir.name)

    def __getitem__(self, item):
        bottom_index = item * TENSORS_PER_READ_SET

        # The order here corresponds to the order of yield statements within make_flattened_tensor_generator()
        return ReadSet(ref_sequence_tensor=self.memory_map[bottom_index + 2],
                       ref_tensor=self.memory_map[bottom_index],
                       alt_tensor=self.memory_map[bottom_index + 1],
                       info_tensor=self.memory_map[bottom_index + 3],
                       label=self.memory_map[bottom_index + 4])



