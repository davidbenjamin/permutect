from torch.utils.data import DataLoader

from permutect.data.datum import DEFAULT_GPU_FLOAT, DEFAULT_CPU_FLOAT
from permutect.data.reads_batch import ReadsBatch
from permutect.misc_utils import gpu_if_available


def prefetch_generator(dataloader: DataLoader, device=gpu_if_available()):
    """
    prefetch and send batches to GPU in the background
    dataloader must yield betches that have a copy_to method
    """
    print("Entering prefetch generator...")
    is_cuda = device.type == 'cuda'
    dtype = DEFAULT_GPU_FLOAT if is_cuda else DEFAULT_CPU_FLOAT
    batch_cpu: ReadsBatch
    for batch_cpu in dataloader:
        batch_gpu = batch_cpu.copy_to(device, dtype=dtype)
        yield batch_gpu


    """
    loader_iter = iter(dataloader)
    print("about to get the first batch for the prefetch generator...")
    next_batch_cpu = next(loader_iter, None)
    next_batch = None if next_batch_cpu is None else next_batch_cpu.copy_to(device=device, dtype=dtype)
    print("Iterating over the rest of the loader in the prefetch generator...")
    for _ in range(len(dataloader)):
        # the prefetched + sent-to-GPU batch is processed
        batch = next_batch
        # but in the background we'll fetch the next batch and send to GPU
        # the default None will come up on the final batch where there is no next batch to prefetch
        next_batch_cpu = next(loader_iter, None)
        next_batch = None if next_batch_cpu is None else next_batch_cpu.copy_to(device, dtype=dtype)
        yield batch
    """