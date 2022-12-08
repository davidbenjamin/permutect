import tarfile

import torch
import os
from torch.utils.data import Dataset
from mmap_ninja.ragged import RaggedMmap
import tempfile

from mutect3.data.read_set import ReadSet, load_list_of_read_sets





class MemoryMapDataset(Dataset):






