from typing import Tuple

import numpy as np
import torch
from torch import Tensor, IntTensor


def flattened_indices(shape: Tuple[int], idx: Tuple[IntTensor]):
    dim = len(shape)
    if dim < 4:
        if dim == 2:
            return idx[1] + shape[1] * idx[0]
        elif dim == 3:
            return idx[2] + shape[2] * (idx[1] + shape[1] * idx[0])
    elif dim == 5:
        return idx[4] + shape[4] * (idx[3] + shape[3] * (idx[2] + shape[2] * (idx[1] + shape[1] * idx[0])))
    elif dim == 6:
        return idx[5] + shape[5] * (idx[4] + shape[4] * (idx[3] + shape[3] * (idx[2] + shape[2] * (idx[1] + shape[1] * idx[0]))))
    elif dim == 4:
        return idx[3] + shape[3] * (idx[2] + shape[2] * (idx[1] + shape[1] * idx[0]))
    else:
        raise Exception("Not implemented yet.")


def index_tensor(tens: Tensor, idx: Tuple[IntTensor]) -> Tensor:
    return tens.view(-1)[flattened_indices(tens.shape, idx)]


# add in-place
# note that the flattened view(-1) shares memory with the original tensor
def add_at_index(tens: Tensor, idx: Tuple[IntTensor], values: Tensor) -> Tensor:
    return tens.view(-1).index_add_(dim=0, index=flattened_indices(tens.shape, idx), source=values)


def downsample_tensor(tensor2d: np.ndarray, new_length: int):
    if tensor2d is None or new_length >= len(tensor2d):
        return tensor2d
    perm = np.random.permutation(len(tensor2d))
    return tensor2d[perm[:new_length]]


# for tensor of shape (R, C...) and row counts n1, n2. . nK, return a tensor of shape (K, C...) whose 1st row is the sum of the
# first n1 rows of the input, 2nd row is the sum of the next n2 rows etc
# note that this works for arbitrary C, including empty.  That is, it works for 1D, 2D, 3D etc input.
def sums_over_rows(input_tensor: Tensor, counts: IntTensor):
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
