## Simple Kradagrad++ that extends official unoptimized Shampoo implementation

from collections import defaultdict

import torch

from . import math_utils as mu
from .math_utils import positive_matrix_functions as mf
from . import batched_matrix_functions as bmf
from .third_party.shampoo import (
    Shampoo, Preconditioner,
    STEP, MOMENTUM, PRECONDITIONER, GRAFT,
    LayerwiseGrafting, AdamGraft, Graft
)


class KrADPreconditioner(Preconditioner):

    @torch.no_grad()
    def __init__(self, var, hps, eps_override=1.0, debug=False):
        """
        debug mode print tensor values likely to be non-finite, raises exceptions
        eps_override changes initialization of preconditioner diagonal (usually 1.0)
        """
        super().__init__(var, hps, eps_override=eps_override)
        self.debug = debug
        
    @torch.no_grad()
    def exponent_for_preconditioner(self):
        """Returns exponent to use for pth root M^{1/p}.
        If `inverse_exponent_override` is set, use ONS-style update
        """
        if self._hps.inverse_exponent_override > 0:
            return len(self._transformed_shape)
        return 2 * len(self._transformed_shape)

    @torch.no_grad()
    def add_statistics(self, grad):
        """Compute inverse KrAD statistics from gradients and add to the correct state entries.

        Args:
          grad: Gradient to compute statistics from.
        """
        if not self.statistics: return  # sgd
        # kradagrad
        partitioned_grads = self.partition_grad(grad)
        w1 = self._hps.beta2
        damping = self._hps.matrix_eps > 0
        rank = len(self._transformed_shape)
        for j, grad in enumerate(partitioned_grads):
            if self._hps.double:
                grad = grad.double()
            if self._hps.bf16:
                grad = grad.bfloat16()
            for i in range(rank):
                axes = list(range(i)) + list(range(i + 1, rank))
                stat = self.statistics[j*rank + i]
                if w1 < 1:
                    stat.mul_(1/w1)

                #if damping:
                if False:
                    if self._hps.bf16:
                        stat = stat.bfloat16()
                    stat_2_eps = stat.mul(-self._hps.matrix_eps).mm(stat.T)
                    if self.debug and not stat_2_eps.isfinite().all():
                        print('stat2eps', stat_2_eps, '\nstat', stat)
                        raise ValueError('stat2eps broke')
                    self.statistics[j*rank + i].add_(stat_2_eps)
                    stat = self.statistics[j*rank + i]

                ggt = torch.tensordot(grad, grad, [axes, axes])
                if self.debug and not ggt.isfinite().all():
                    print('ggt', ggt, '\ngrad', grad)
                    raise ValueError('ggt broke')

                ggtl = ggt.mm(stat.type_as(ggt).T)

                #t_k = -(1 + mf.matrices_norm(ggtl, 'tr'))
                t_k = -(1 + mf.matrices_norm(ggtl, 'tr')) * stat.size()[0]
                ggtl.mul_(1/t_k)
                lggtl = stat.type_as(ggtl).mm(ggtl)
                self.statistics[j*rank + i].add_(lggtl)



    @torch.no_grad()
    def compute_preconditioners(self, **kwargs):
        """Compute L^{1/exp} for each statistics matrix L"""
        if not self.statistics: return  # sgd
        # kradagrad
        exp = self.exponent_for_preconditioner()
        for i, stat in enumerate(self.statistics):
            if self._hps.iterative_matrix_roots:
                self.preconditioners[i] = (
                    mf.symmetrize(mf.mat_pow(mf.mat_inv_root(
                        stat, exp,
                        double=self._hps.double, iter_count=10
                    ), exp-1).mm(stat)))
            else:
                try:
                    if self._hps.bf16:
                        stat = stat.bfloat16()
                    precon = mf.matrix_power_svd(stat, 1 / exp, double=self._hps.double) if exp > 1 else stat
                    if self._hps.double:
                        precon = precon.double()
                    self.preconditioners[i] = precon
                except Exception as err:
                    if self.debug:
                        print('stat', stat)
                    raise err


    @torch.no_grad()
    def partition_grad(self, grad):
        """Partition gradient.
        Args:
          grad: Gradient to partition.
        """
        reshaped_grad = torch.reshape(grad.detach(), self._transformed_shape)
        partitioned_grads = self._partitioner.partition(reshaped_grad)
        return partitioned_grads

class KradagradPP(Shampoo):
    r"""Implements a simple version of Kradagrad++ Optimizer Algorithm.
    Extends the unoptimized official pytorch implementation of Shampoo
    """
    @torch.no_grad()
    def __init__(self, params, tensor_batch_size=None, **kwargs):
        # extra accouting for batch processing:
        # group by exponent, sort by size
        # group similar sizes together to minimize padding
        self.cpu = kwargs.get('cpu', False)
        self.debug = kwargs.get('debug', False)
        self.initialized = False
        self._step = 0
        self.no_batch = not bool(tensor_batch_size)
        self.tensor_batch_size = tensor_batch_size if tensor_batch_size else 0
        
        # store references to Tensors for batch processing
        self.param_master_dict = defaultdict(list)
        self.stat_master_dict = defaultdict(list)
        self.cond_master_dict = defaultdict(list)
        
        super().__init__(params, **kwargs)

    @torch.no_grad()
    def init_var_state(self, var, state):
        """Initialize the PyTorch state of for a single variable."""
        # group by exponent for batching
        # we can further improve by sorting by size, but that requires advanced bookkeeping
        prec = KrADPreconditioner(var, self.hps, eps_override=1/self.hps.matrix_eps, debug=self.debug)
        if not (self.cpu or self.no_batch):
            exp = prec.exponent_for_preconditioner()
            if len(prec.statistics) > 1:
                self.stat_master_dict[exp].extend(prec.statistics)
                self.cond_master_dict[exp].extend(prec.preconditioners)

        # original, pared down        
        var = var.detach()
        state[MOMENTUM] = torch.zeros_like(var.data, device=var.device)
        state[PRECONDITIONER] = prec
        if self.cpu or self.no_batch:
            state[STEP] = 0
            if self.hps.graft_type == LayerwiseGrafting.ADAM:
                state[GRAFT] = AdamGraft(self.hps, var)
            else:
                state[GRAFT] = Graft(self.hps, var)

    # reimplementation for batch processing on gpu
    @torch.no_grad()
    def step(self, closure=None):
        if self.cpu or self.no_batch:
            super().step(closure)
            return

        # GPU batch processing, currently no grafting

        # we can still initialize in series
        if not self.initialized:
            for group in self.param_groups:
                lr = group['lr']
                for p in group['params']:
                    if p.grad is None: continue
                    if not (state := self.state[p]):
                        self.init_var_state(p, state)
            self.initialized = True
        for group in self.param_groups:
            # populate grads in series
            partitioned_grads_dict = defaultdict(list)
            for p in group['params']:
                if p.grad is None: continue
                prec = self.state[p][PRECONDITIONER]
                exp = prec.exponent_for_preconditioner()
                sh_ = prec._transformed_shape
                # list of lists of partitioned grads
                partitioned_grads_dict[exp].append(
                    prec._partitioner.partition(
                        p.grad.detach().reshape(sh_)
                    )
                )

            # process stats and precon in batches
            if self._step % self.hps.statistics_compute_steps == 0:
                self.batch_add_statistics(partitioned_grads_dict)
            if self._step % self.hps.preconditioning_compute_steps == 0:
                self.batch_compute_preconditioners(step=self.step)
        self._step += 1
        # back to series processing (easier, not as much gain from batched)
        for group in self.param_groups:
            lr = group['lr']

            for p in group['params']:
                grad = p.grad.detach()
                state = self.state[p]
                precon = state[PRECONDITIONER]

                krad_grad = grad
                # Precondition
                if self._step >= self.hps.start_preconditioning_step:
                    krad_grad = precon.preconditioned_grad(grad)

                # Weight decay
                if self.hps.weight_decay != 0.0:
                    krad_grad.add_(p.data, alpha=self.hps.weight_decay)

                # Momentum and Nesterov momentum, if needed
                state[MOMENTUM].mul_(group['momentum']).add_(krad_grad)

                momentum_update = state[MOMENTUM]
                wd_update = krad_grad

                if self.hps.nesterov:
                    momentum_update.mul_(group['momentum']).add_(wd_update)

                # Final update
                p.data.add_(momentum_update, alpha=-lr)

    def batch_add_statistics(self, partitioned_grads_dict):
        bs = self.tensor_batch_size
        w1 = self.hps.beta2
        damping = self.hps.matrix_eps > 0
        for exp in partitioned_grads_dict.keys():
            # list of lists of partitioned grads
            grads_list_list = partitioned_grads_dict[exp]
            
            grads_list = flatten(grads_list_list)
            rank = grads_list[0].ndim
            if rank == 1:  # no statistics for vectors
                continue

            # list of stats
            stats_list = self.stat_master_dict[exp]

            n_grad = len(grads_list)
            n_batch = mu.roundup(n_grad, bs)
            for i in range(n_batch):
                # bookkeeping for sizes
                grads_this = grads_list[bs*i:bs*(i+1)]
                
                # all dimensions
                stats_this = stats_list[bs*rank*i:bs*rank*(i+1)]

                # actual batch size (only != bs for last batch)
                bs_ = len(grads_this)
                
                grads_sizes = [list(x.size()) for x in grads_this]
                max_grads_sizes = [max(x) for x in zip(*grads_sizes)]
                batch_grads = torch.zeros([bs_] + max_grads_sizes)
                
                stats_sizes = [list(x.size()) for x in stats_this]
                # each dimension gets its own batch
                stats_sizes_by_rank = [stats_sizes[k::rank] for k in range(rank)]
                    
                max_stats_sizes_by_rank = [[max(x) for x in zip(*stats_sizes_)] for stats_sizes_ in stats_sizes_by_rank]
                batch_stats_by_rank = [torch.zeros([bs_] + max_stats_sizes_) for max_stats_sizes_ in max_stats_sizes_by_rank]

                # fill batch of grads
                for j in range(bs_):
                    batch_grads[[j] + _slicer(grads_this[j])] = grads_this[j]
                # fill batches of stats for each rank
                for k in range(rank):
                    for j in range(bs_):
                        batch_stats_by_rank[k][[j] + _slicer(stats_this[j * rank + k])] = stats_this[j * rank + k]

                # process in batch
                for k in range(rank):
                    # just this dimension
                    batch_stats_this_rank = batch_stats_by_rank[k]
                    if w1 < 1:
                        batch_stats_this_rank.mul_(1/w1)
                    if damping:
                        batch_stats_2_eps = batch_stats_this_rank.mul(-self.hps.matrix_eps).bmm(batch_stats_this_rank.transpose(-2,-1))
                        batch_stats_this_rank.add_(batch_stats_2_eps)
                    
                    
                    batch_grad_mat = bmf.batch_matricize(batch_grads, k)
                    ggt = batch_grad_mat.bmm(batch_grad_mat.transpose(-2, -1))
                    ggtl = ggt.bmm(batch_stats_this_rank.transpose(-2, -1))
                    t_batch = -(1 + bmf.matrices_norm(ggtl, 'fro'))
                    ggtl.mul_((1/t_batch)[..., None, None])
                    lggtl = batch_stats_this_rank.bmm(ggtl)
                    updated_stats = batch_stats_this_rank.add(lggtl)
                    # update in batch of stats for this rank in series
                    for j in range(bs_):
                        ix = j * rank + k
                        stats_this[ix].copy_(updated_stats[[j] + _slicer(stats_this[ix])])
    
    def batch_compute_preconditioners(self, step):
        bs = self.tensor_batch_size
        for exp in self.stat_master_dict.keys():
            stats_list = self.stat_master_dict[exp]
            precs_list = self.cond_master_dict[exp]
            n_stat = len(stats_list)
            n_batch = mu.roundup(n_stat, bs)
            for i in range(n_batch):
                # bookkeeping for sizes
                stats_this = stats_list[bs*i:bs*(i+1)]
                precs_this = precs_list[bs*i:bs*(i+1)]
                bs_ = len(stats_this)
                sizes = [x.size()[0] for x in stats_this]
                max_size = max(*sizes)
                batch_stats = torch.zeros([bs_, max_size, max_size])

                # fill batch of stats
                for j in range(bs_):
                    batch_stats[j, 0:sizes[j], 0:sizes[j]] = stats_this[j]
                batch_root = bmf.mat_root(
                    batch_stats, exp, None,  # skipping warm start
                    double=self._hps.double,
                    iters=10, tol=1e-4, inner_iters=20, inner_tol=1e-6
                )
                for j in range(bs_):
                    precs_this[j].copy_(batch_root[j, 0:sizes[j], 0:sizes[j]])
    
    # future work
    def batch_preconditioned_grad():
        pass
    
    def batch_merge_partitions():
        pass
    
    def unbatch():
        pass

def flatten(list_of_lists):
    return [x for y in list_of_lists for x in y]

def _slicer(tensor):
    return [slice(0, x) for x in tensor.size()]
