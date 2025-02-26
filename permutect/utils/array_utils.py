from typing import Tuple

import numpy as np
import torch
from torch import Tensor


def index_2d_array(tens, idx0, idx1):
    dim0, dim1 = tens.shape
    flattened_indices = idx1 + dim1 * idx0
    return tens.view(-1)[flattened_indices]


def index_3d_array(tens, idx0, idx1, idx2):
    """
    given 3d tensor T_ijk and 1D index tensors I, J, K, return the 1D tensor:
    result[n] = T[I[n], J[n], K[n]]
    """
    dim0, dim1, dim2 = tens.shape
    flattened_indices = idx2 + dim2 * (idx1 + dim1 * idx0)
    return tens.view(-1)[flattened_indices]


def index_4d_array(tens, idx0, idx1, idx2, idx3):
    dim0, dim1, dim2, dim3 = tens.shape
    flattened_indices = idx3 + dim3 * (idx2 + dim2 * (idx1 + dim1 * idx0))
    return tens.view(-1)[flattened_indices]


def index_5d_array(tens, idx0, idx1, idx2, idx3, idx4):
    dim0, dim1, dim2, dim3, dim4 = tens.shape
    flattened_indices = idx4 + dim4 * (idx3 + dim3 * (idx2 + dim2 * (idx1 + dim1 * idx0)))
    return tens.view(-1)[flattened_indices]


def index_6d_array(tens, idx0, idx1, idx2, idx3, idx4, idx5):
    dim0, dim1, dim2, dim3, dim4, dim5 = tens.shape
    flattened_indices = idx5 + dim5 * (idx4 + dim4 * (idx3 + dim3 * (idx2 + dim2 * (idx1 + dim1 * idx0))))
    return tens.view(-1)[flattened_indices]


# add in-place
# note that the flattened view(-1) shares memory with the original tensor
def add_to_2d_array(tens: Tensor, idx0, idx1, values):
    dim0, dim1 = tens.shape
    flattened_indices = idx1 + dim1 * idx0
    return tens.view(-1).index_add_(dim=0, index=flattened_indices, source=values)


def add_to_3d_array(tens: Tensor, idx0, idx1, idx2, values):
    """
    given 3d tensor T_ijk and 1D index tensors I, J, K, achieve the effect
    T[I[n], J[n], K[n]] += values[n]
    """
    dim0, dim1, dim2 = tens.shape
    flattened_indices = idx2 + dim2 * (idx1 + dim1 * idx0)
    return tens.view(-1).index_add_(dim=0, index= flattened_indices, source=values)


def add_to_4d_array(tens: Tensor, idx0, idx1, idx2, idx3, values):
    dim0, dim1, dim2, dim3 = tens.shape
    flattened_indices = idx3 + dim3 * (idx2 + dim2 * (idx1 + dim1 * idx0))
    return tens.view(-1).index_add_(dim=0, index= flattened_indices, source=values)


def add_to_5d_array(tens, idx0, idx1, idx2, idx3, idx4, values):
    dim0, dim1, dim2, dim3, dim4 = tens.shape
    flattened_indices = idx4 + dim4 * (idx3 + dim3 * (idx2 + dim2 * (idx1 + dim1 * idx0)))
    return tens.view(-1).index_add_(dim=0, index=flattened_indices, source=values)


def add_to_6d_array(tens, idx0, idx1, idx2, idx3, idx4, idx5, values):
    dim0, dim1, dim2, dim3, dim4, dim5 = tens.shape
    flattened_indices = idx5 + dim5 * (idx4 + dim4 * (idx3 + dim3 * (idx2 + dim2 * (idx1 + dim1 * idx0))))
    return tens.view(-1).index_add_(dim=0, index=flattened_indices, source=values)


def downsample_tensor(tensor2d: np.ndarray, new_length: int):
    if tensor2d is None or new_length >= len(tensor2d):
        return tensor2d
    perm = np.random.permutation(len(tensor2d))
    return tensor2d[perm[:new_length]]


# for tensor of shape (R, C...) and row counts n1, n2. . nK, return a tensor of shape (K, C...) whose 1st row is the sum of the
# first n1 rows of the input, 2nd row is the sum of the next n2 rows etc
# note that this works for arbitrary C, including empty.  That is, it works for 1D, 2D, 3D etc input.
def sums_over_rows(input_tensor: Tensor, counts: torch.IntTensor):
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


def cumsum_starting_from_zero(x: Tensor):
    return (torch.sum(x) - torch.cumsum(x.flip(dims=(0,)), dim=0)).flip(dims=(0,))


def select_and_sum(x: Tensor, select: dict[int, int]={}, sum: Tuple[int]=()):
    """
    select specific indices over certain dimensions and sum over others.  For example suppose
    x = [ [[1,2], [3,4]],
            [[5,6], [7,8]]]
    Then we want:
        select_and_sum(x, select={0:0}, sum=(2)) = [3, 7]
        select_and_sum(x, select={0:1,1:0}, sum=()) = [5,6]
        select_and_sum(x, select={}, sum=(1,2)) = [10,26]
    :param x: the input tensor
    :param select: dict of dimension:index of that dimension to select
    :param sum tuple of dimensions over which to sum
    :return:

    select_indices and sum_dims must be disjoint but need not include all input dimensions.
    """

    # initialize indexing to be complete slices i.e. select everything, then use the given select_indices
    indices = [slice(dim_size) for dim_size in x.shape]
    for select_dim, select_index in select.items():
        indices[select_dim] = slice(select_index, select_index + 1)     # one-element slice

    selected = x[tuple(indices)]    # retains original dimensions; selected dimensions have length 1
    summed = selected if len(sum) == 0 else torch.sum(selected, dim=sum, keepdim=True)  # still retain original dimension

    # Finally, select element 0 from the selected and summed axes to contract the dimensions
    for sum_dim in sum:
        indices[sum_dim] = 0
    for select_dim in select.keys():
        indices[select_dim] = 0

    return summed[tuple(indices)]
