import sys
from typing import Union, Iterable

import numpy as np
import torch


# Note: these are unbatched, intended for Tensors sized (N, N)
@torch.no_grad()
def _mat_fro_norm(A):  # torch's version is incorrect
    return torch.sqrt((A ** 2).sum())

@torch.no_grad()
def _mat_inf_norm(A):
    return torch.abs(A).sum(dim=-1).max(dim=-1)[0]

@torch.no_grad()
def _mat_l1_norm(A):
    return torch.abs(A).sum()

@torch.no_grad()
def matrices_norm(A, norm: str='tr'):
    if norm == 'inf':  # decently tight bound on largest eig for diag dom
        A_norm = _mat_inf_norm(A)
    elif norm == 'fro':  # looser bound on largest eig
        A_norm = _mat_fro_norm(A)
    elif norm == 'l1':  # looser bound on largest eig
        A_norm = _mat_l1_norm(A)
    else:  # trace, even looser bound on largest eig
        A_norm = torch.trace(A)
    return A_norm

@torch.no_grad()
def symmetrize(A):
    return A / 2 + A.t() / 2

@torch.no_grad()
def _matrices_info(A: torch.Tensor, norm: str='tr'):
    # check if A is 3D, return size and norm
    A_sz = A.size()
    A_dim = A.dim()
    if A_dim > 2:
        caller = sys._getframe(1).f_code.co_name
        raise ValueError("{} supports matrices only! Input dim: {}".format(caller, A_dim))
    A_norm = matrices_norm(A, norm=norm)
    A_norm = A_norm[None, None]
    A_dev = A.device

    return A_sz, A_dim, A_norm, A_dev

@torch.no_grad()
def mat_pow(A: torch.Tensor, p: int):
    # performs A^p on symmetric matrices using O(log_2(p)) matmuls
    if p < 1:
        raise ValueError("mat_pow only valid for positive integer powers!")
    if p == 1:
        return A
    # string:
    pb = bin(p)[2:]
    X_prev = A  # tmp var tracking A^(2^i))
    Xpow = None  # accumulator
    for i, save in enumerate(pb[::-1]):
        if save == '1':
            if Xpow is None:
                Xpow = X_prev
            else:
                Xpow = Xpow.mm(X_prev)
        if i < len(pb) - 1:
            # stays more symmetric:
            X_prev = X_prev.mm(X_prev.transpose(-2, -1))
    return Xpow


@torch.no_grad()
def power_iter(mat_g, error_tolerance=1e-6, num_iters=100):
  """Power iteration.

  Compute the maximum eigenvalue of mat, for scaling.
  v is a random vector with values in (-1, 1)

  Args:
    mat_g: the symmetric PSD matrix.
    error_tolerance: Iterative exit condition.
    num_iters: Number of iterations.

  Returns:
    eigen vector, eigen value, num_iters
  """
  v = torch.rand(list(mat_g.shape)[0], device=mat_g.device).type_as(mat_g) * 2 - 1
  error = 1
  iters = 0
  singular_val = 0
  while error > error_tolerance and iters < num_iters:
    v = v / torch.norm(v)
    mat_v = torch.mv(mat_g, v)
    s_v = torch.dot(v, mat_v)
    error = torch.abs(s_v - singular_val)
    v = mat_v
    singular_val = s_v
    iters += 1
  return singular_val, v / torch.norm(v), iters



@torch.no_grad()
def mat_inv_root(mat_g, p,
                 iter_count=100,
                 error_tolerance=1e-6,
                 ridge_epsilon=1e-6,
                 double=False):
    """A method to compute G^{-1/p} using a coupled Newton iteration.

    See for example equation 3.2 on page 9 of:
    A Schur-Newton Method for the Matrix p-th Root and its Inverse
    by Chun-Hua Guo and Nicholas J. Higham
    SIAM Journal on Matrix Analysis and Applications,
    2006, Vol. 28, No. 3 : pp. 788-804
    https://pdfs.semanticscholar.org/0abe/7f77433cf5908bfe2b79aa91af881da83858.pdf

    Args:
        mat_g: A square positive semidefinite matrix
        p: a positive integer
        iter_count: Stop iterating after this many rounds.
        error_tolerance: Threshold for stopping iteration
        ridge_epsilon: We add this times I to G, to make is positive definite.
                     For scaling, we multiply it by the largest eigenvalue of G.
    Returns:
        (mat_g + rI)^{-1/p} (r = ridge_epsilon * max_eigenvalue of mat_g).
    """
    orig_type = mat_g.dtype
    if double:
        mat_g = mat_g.type(torch.float64)
    shape = list(mat_g.shape)
    if len(shape) == 1:
        return torch.pow(mat_g + ridge_epsilon, -1/p)
    identity = torch.eye(shape[0], device=mat_g.device).type_as(mat_g)
    if shape[0] == 1:
        return identity
    alpha = -1.0/p
    max_ev, _, _ = power_iter(mat_g)
    ridge_epsilon *= max_ev
    mat_g += ridge_epsilon * identity
    z = (1 + p) / (2 * torch.norm(mat_g))
    # The best value for z is
    # (1 + p) * (c_max^{1/p} - c_min^{1/p}) /
    #            (c_max^{1+1/p} - c_min^{1+1/p})
    # where c_max and c_min are the largest and smallest singular values of
    # mat_g.
    # The above estimate assumes that c_max > c_min * 2^p
    # Can replace above line by the one below, but it is less accurate,
    # hence needs more iterations to converge.
    # z = (1 + p) / tf.trace(mat_g)
    # If we want the method to always converge, use z = 1 / norm(mat_g)
    # or z = 1 / tf.trace(mat_g), but these can result in many
    # extra iterations.

    mat_root = identity * torch.pow(z, 1.0/p)
    mat_m = mat_g * z
    error = torch.max(torch.abs(mat_m - identity))
    count = 0
    while error > error_tolerance and count < iter_count:
        tmp_mat_m = (1 - alpha) * identity + alpha * mat_m
        new_mat_root = symmetrize(torch.matmul(mat_root, tmp_mat_m))
        mat_m = symmetrize(torch.matmul(mat_pow(tmp_mat_m, p), mat_m))
        #tmp_mat_p2 = mat_pow(tmp_mat_m, p//2)
        #mat_m = symmetrize(tmp_mat_p2.mm(mat_m).mm(tmp_mat_p2.T))
        new_error = torch.max(torch.abs(mat_m - identity))
        if new_error > error * 1.2:
            break
        mat_root = new_mat_root
        error = new_error
        count += 1
    return mat_root.type(orig_type)

@torch.no_grad()
def matrix_power_svd(matrix: torch.Tensor, power: Union[float, Iterable[float]], double: bool=False, matrix_eps=1e-8, eig_eps=0, **unused) -> torch.Tensor:
    if singleton := not(isinstance(power, Iterable) and len(power) > 0):
        pows = [power]
    else:
        pows = power
    # Really, use hermitian eigenvalue decomposition
    if unused:
        print(f'warning, `matrix_power_svd` got unused keywords: {list(unused.keys())}')
    orig_type = matrix.dtype
    precision = torch.float64 if double else torch.float32
    try:
        mat = matrix.type(precision) + matrix_eps*torch.eye(matrix.size()[0], device=matrix.device)
        L, V = torch.linalg.eigh(mat)
    except Exception as err:
        if not matrix.isfinite().all():
            print('matrix', matrix)
        raise err
    L = torch.maximum(L, eig_eps * torch.ones(1, device=L.device))
    if singleton:
        return ((V * L.pow(power)) @ V.T).type(orig_type)
    else:
        return tuple(((V * L.pow(pow)) @ V.T).type(orig_type) for pow in pows)
