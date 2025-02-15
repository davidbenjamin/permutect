import numpy as np
import torch


def index_2d_array(tens, idx0, idx1):
    dim0, dim1 = tens.shape
    flattened_indices = (idx0 * dim1) + idx1
    return tens.view(-1)[flattened_indices]


def index_3d_array(tens, idx0, idx1, idx2):
    """
    given 3d tensor T_ijk and 1D index tensors I, J, K, return the 1D tensor:
    result[n] = T[I[n], J[n], K[n]]
    """
    dim0, dim1, dim2 = tens.shape
    flattened_indices = (idx0 * dim1 * dim2) + (idx1 * dim2) + idx2
    return tens.view(-1)[flattened_indices]


def index_4d_array(tens, idx0, idx1, idx2, idx3):
    dim0, dim1, dim2, dim3 = tens.shape
    flattened_indices = (idx0 * dim1 * dim2 * dim3) + (idx1 * dim2 * dim3) + (idx2 * dim3) + idx3
    return tens.view(-1)[flattened_indices]


def index_5d_array(tens, idx0, idx1, idx2, idx3, idx4):
    dim0, dim1, dim2, dim3, dim4 = tens.shape
    flattened_indices = (idx0 * dim1 * dim2 * dim3 * dim4) + (idx1 * dim2 * dim3 * dim4) + (idx2 * dim3 * dim4) + (idx3 * dim4) + idx4
    return tens.view(-1)[flattened_indices]


# add in-place
# note that the flattened view(-1) shares memory with the original tensor
def add_to_2d_array(tens: torch.Tensor, idx0, idx1, values):
    dim0, dim1 = tens.shape
    flattened_indices = (idx0 * dim1) + idx1
    return tens.view(-1).index_add_(dim=0, index=flattened_indices, source=values)


def add_to_3d_array(tens: torch.Tensor, idx0, idx1, idx2, values):
    """
    given 3d tensor T_ijk and 1D index tensors I, J, K, achieve the effect
    T[I[n], J[n], K[n]] += values[n]
    """
    dim0, dim1, dim2 = tens.shape
    flattened_indices = (idx0 * dim1 * dim2) + (idx1 * dim2) + idx2
    return tens.view(-1).index_add_(dim=0, index= flattened_indices, source=values)


def add_to_4d_array(tens: torch.Tensor, idx0, idx1, idx2, idx3, values):
    dim0, dim1, dim2, dim3 = tens.shape
    flattened_indices = (idx0 * dim1 * dim2 * dim3) + (idx1 * dim2 * dim3) + (idx2 * dim3) + idx3
    return tens.view(-1).index_add_(dim=0, index= flattened_indices, source=values)


def add_to_5d_array(tens, idx0, idx1, idx2, idx3, idx4, values):
    dim0, dim1, dim2, dim3, dim4 = tens.shape
    flattened_indices = (idx0 * dim1 * dim2 * dim3 * dim4) + (idx1 * dim2 * dim3 * dim4) + (idx2 * dim3 * dim4) + (idx3 * dim4) + idx4
    return tens.view(-1).index_add_(dim=0, index= flattened_indices, source=values)


def downsample_tensor(tensor2d: np.ndarray, new_length: int):
    if tensor2d is None or new_length >= len(tensor2d):
        return tensor2d
    perm = np.random.permutation(len(tensor2d))
    return tensor2d[perm[:new_length]]


# for tensor of shape (R, C...) and row counts n1, n2. . nK, return a tensor of shape (K, C...) whose 1st row is the sum of the
# first n1 rows of the input, 2nd row is the sum of the next n2 rows etc
# note that this works for arbitrary C, including empty.  That is, it works for 1D, 2D, 3D etc input.
def sums_over_rows(input_tensor: torch.Tensor, counts: torch.IntTensor):
    range_ends = torch.cumsum(counts, dim=0)
    assert range_ends[-1] == len(input_tensor)   # the counts need to add up!

    row_cumsums = torch.cumsum(input_tensor, dim=0)

    # if counts are eg 1, 2, 3 then range ends are 1, 3, 6 and we are interested in cumsums[0, 2, 5]
    relevant_cumsums = row_cumsums[(range_ends - 1).long()]

    # if counts are eg 1, 2, 3 we now have, the sum of the first 1, 3, and 6 rows.  To get the sums of row 0, rows 1-2, rows 3-5
    # we need the consecutive differences, with a row of zeroes prepended
    row_of_zeroes = torch.zeros_like(relevant_cumsums[0])[None] # the [None] makes it (1xC)
    relevant_sums = torch.diff(relevant_cumsums, dim=0, prepend=row_of_zeroes)
    return relevant_sums


def cumsum_starting_from_zero(x: torch.Tensor):
    return (torch.sum(x) - torch.cumsum(x.flip(dims=(0,)), dim=0)).flip(dims=(0,))
