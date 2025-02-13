from torch.utils.data import DataLoader

from permutect.data.datum import DEFAULT_GPU_FLOAT, DEFAULT_CPU_FLOAT
from permutect.misc_utils import gpu_if_available


def prefetch_generator(dataloader: DataLoader, device=gpu_if_available()):
    """
    prefetch and send batches to GPU in the background
    dataloader must yield betches that have a copy_to method
    """
    loader_iter = iter(dataloader)
    is_cuda = device.type == 'cuda'
    dtype = DEFAULT_GPU_FLOAT if is_cuda else DEFAULT_CPU_FLOAT
    next_batch_cpu = next(loader_iter, None)
    next_batch = None if next_batch_cpu is None else next_batch_cpu.copy_to(device=device, dtype=dtype)
    for _ in range(len(dataloader)):
        # the prefetched + sent-to-GPU batch is processed
        batch = next_batch
        # but in the background we'll fetch the next batch and send to GPU
        # the default None will come up on the final batch where there is no next batch to prefetch
        next_batch_cpu = next(loader_iter, None)
        next_batch = None if next_batch_cpu is None else (next_batch_cpu.copy_to(device, dtype=dtype) if is_cuda else next_batch_cpu)
        yield batch