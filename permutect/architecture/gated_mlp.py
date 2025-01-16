"""
---
title: Pay Attention to MLPs (gMLP)
summary: >
  This is an annotated implementation/tutorial of Pay Attention to MLPs (gMLP) in PyTorch.
---

# Pay Attention to MLPs (gMLP)

This is a [PyTorch](https://pytorch.org) implementation of the paper
[Pay Attention to MLPs](https://arxiv.org/abs/2105.08050).

This paper introduces a Multilayer Perceptron (MLP) based architecture with gating,
which they name **gMLP**. It consists of a stack of $L$ *gMLP* blocks.

Here is [the training code](experiment.html) for a gMLP model based autoregressive model.
"""
# copied from https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/transformers/gmlp/__init__.py
# then modified for the symmetric case

import torch
from torch import nn

from permutect import utils
from permutect.sets.ragged_sets import RaggedSets


class GatedMLPBlock(nn.Module):
    """
    ## gMLP Block

    Each block does the following transformations to input embeddings
    $X \in \mathbb{R}^{n \times d}$ where $n$ is the sequence length
    and $d$ is the dimensionality of the embeddings:

    \begin{align}
    Z &= \sigma(XU) \\
    \tilde{Z} &= s(Z) \\
    Y &= \tilde{Z}V \\
    \end{align}

    where $V$ and $U$ are learnable projection weights.
    $s(\cdot)$ is the Spacial Gating Unit defined below.
    Output dimensionality of $s(\cdot)$ will be half of $Z$.
    $\sigma$ is an activation function such as
    [GeLU](https://pytorch.org/docs/stable/generated/torch.nn.GELU.html).
    """

    def __init__(self, d_model: int, d_ffn: int):
        """
        * `d_model` is the dimensionality ($d$) of $X$ i.e. the embedding dimension of each read
        * `d_ffn` is the dimensionality of $Z$, that is, the hidden dimension of each block
        """
        super(GatedMLPBlock, self).__init__()
        # Normalization layer fro Pre-Norm
        self.norm = nn.LayerNorm([d_model])
        # Activation function $\sigma$
        self.activation = nn.SELU()
        # Projection layer for $Z = \sigma(XU)$
        self.proj1 = nn.Linear(d_model, d_ffn)
        # Spacial Gating Unit $s(\cdot)$
        self.sgu = SpacialGatingUnit(d_ffn)
        # Projection layer for $Y = \tilde{Z}V$
        self.proj2 = nn.Linear(d_ffn // 2, d_model)
        # Embedding size (required by [Encoder](../models.html#Encoder).
        # We use the encoder module from transformer architecture and plug
        # *gMLP* block as a replacement for the [Transformer Layer](../models.html#Encoder).
        self.size = d_model

    # X is 2D, counts are the numbers of elements in each consecutive group of rows that form a self-attention group
    # that is, is X has 10 rows and counts = [2,3,5], elements 0-1, 2-4, and 5-9 form independent self-attention groups
    # In other words, all the reads of a batch are flattened together in X -- the batch information is in counts
    def forward(self, x_re: torch.Tensor, counts: torch.IntTensor):
        """
        * `x_bre` is the input read embedding tensor of shape Batch x Reads x Embedding
        """
        # Norm, projection to d_ffn, and activation $Z = \sigma(XU)$
        z_rd = self.activation(self.proj1(self.norm(x_re)))
        # Spacial Gating Unit $\tilde{Z} = s(Z)$
        gated_rd = self.sgu.forward(z_rd, counts)
        # Final projection $Y = \tilde{Z}V$ back to embedding dimension
        gated_re = self.proj2(gated_rd)

        # Add the shortcut connection
        return x_re + gated_re


class SpacialGatingUnit(nn.Module):
    """
    ## Spatial Gating Unit
    ORIGINAL:
    $$s(Z) = Z_1 \odot f_{W,b}(Z_2)$$

    where $f_{W,b}(Z) = W Z + b$ is a linear transformation along the sequence dimension,
    and $\odot$ is element-wise multiplication.
    $Z$ is split into to parts of equal size $Z_1$ and $Z_2$ along the channel dimension (embedding dimension).

    MODIFIED: f_{W,b} must be permutation-invariant, and the only way to achieve this is if W has a constant diagonal element
    and a constant off-diagonal element.  That is: WZ = a*(mean of Z along sequence dimension) + b Z + bias

    Due to taking the mean the model no longer needs a constant sequence length.
    """
    def __init__(self, d_z: int):
        """
        * `d_z` is the dimensionality of $Z$, which is d_ffn of the SGU block
        * `seq_len` is the sequence length
        """
        super(SpacialGatingUnit, self).__init__()
        # Normalization layer before applying $f_{W,b}(\cdot)$
        self.norm = nn.LayerNorm([d_z // 2])
        # Weight $W$ in $f_{W,b}(\cdot)$.

        # TODO: shouldn't alpha and beta be element-by-element???
        self.alpha = nn.Parameter(torch.tensor(0.01))
        self.beta = nn.Parameter(torch.tensor(0.01))

    # Z is 2D, counts are the numbers of elements in each consecutive group of rows that form a self-attention group
    # that is, is X has 10 rows and counts = [2,3,5], elements 0-1, 2-4, and 5-9 form independent self-attention groups
    def forward(self, z_rd: torch.Tensor, counts: torch.IntTensor):
        # Split $Z$ into $Z_1$ and $Z_2$ over the hidden dimension and normalize $Z_2$ before $f_{W,b}(\cdot)$
        z1_rd, z2_rd = torch.chunk(z_rd, 2, dim=-1)
        z2_rd = self.norm(z2_rd)

        # TODO: self.beta needs to multiply the mean field here!!!
        z2_rd = 1 + self.alpha * z2_rd + utils.means_over_rows(z2_rd, counts, keepdim=True)

        # $Z_1 \odot f_{W,b}(Z_2)$
        return z1_rd * z2_rd


class GatedMLP(nn.Module):
    def __init__(self, d_model: int, d_ffn: int, num_blocks: int):
        super(GatedMLP, self).__init__()

        self.blocks = nn.ModuleList([GatedMLPBlock(d_model, d_ffn) for _ in range(num_blocks)])

    # X is 2D, counts are the numbers of elements in each consecutive group of rows that form a self-attention group
    # that is, is X has 10 rows and counts = [2,3,5], elements 0-1, 2-4, and 5-9 form independent self-attention groups
    def forward(self, x, counts):
        for block in self.blocks:
            x = block.forward(x, counts)
        return x


class GatedRefAltMLPBlock(nn.Module):
    """
    Like the above, but ref reads see the mean field of ref reads and alt reads see the mean fields of both ref and alt
    Note that using mean fields implies the model never *counts* ref or alt reads, nor knows their relative frequencies
    """

    def __init__(self, d_model: int, d_ffn: int):
        """
        * `d_model` is the dimensionality of read embeddings
        * `d_ffn` is the hidden dimension of each block
        """
        super(GatedRefAltMLPBlock, self).__init__()
        # Normalization layer fro Pre-Norm
        self.norm = nn.LayerNorm([d_model])
        # Activation function $\sigma$
        self.activation = nn.SELU()
        # Projection layer for $Z = \sigma(XU)$
        self.proj1_ref = nn.Linear(d_model, d_ffn)
        self.proj1_alt = nn.Linear(d_model, d_ffn)
        # Spacial Gating Unit $s(\cdot)$
        self.sgu = SpacialGatingUnitRefAlt(d_ffn)
        # Projection layer for $Y = \tilde{Z}V$
        self.proj2_ref = nn.Linear(d_ffn // 2, d_model)
        self.proj2_alt = nn.Linear(d_ffn // 2, d_model)
        # Embedding size (required by [Encoder](../models.html#Encoder).
        # We use the encoder module from transformer architecture and plug
        # *gMLP* block as a replacement for the [Transformer Layer](../models.html#Encoder).
        self.size = d_model

    def forward(self, ref_brf: RaggedSets, alt_brf: RaggedSets) -> tuple[RaggedSets, RaggedSets]:
        """
        :param ref_brf: reference read data indexed (conceptually) by batch, read within set, feature
        :param alt_brf: similar for alt read data
        :return: transformed ref and alt sets
        * `x_bre` is the input read embedding tensor of shape Batch x Reads x Embedding
        """
        # Norm, projection to d_ffn, and activation $Z = \sigma(XU)$
        zref_brd = ref_brf.apply_elementwise(self.norm).apply_elementwise(self.proj1_ref).apply_elementwise(self.activation)
        zalt_brd = alt_brf.apply_elementwise(self.norm).apply_elementwise(self.proj1_alt).apply_elementwise(self.activation)

        # Spacial Gating Unit $\tilde{Z} = s(Z)$
        gated_ref_brd, gated_alt_brd = self.sgu.forward(zref_brd, zalt_brd)

        # Final projection $Y = \tilde{Z}V$ back to embedding dimension
        gated_ref_bre = gated_ref_brd.apply_elementwise(self.proj2_ref)
        gated_alt_bre = gated_alt_brd.apply_elementwise(self.proj2_alt)

        # Add the shortcut connection
        return ref_brf.add_elementwise(gated_ref_bre), alt_brf.add_elementwise(gated_alt_bre)


class SpacialGatingUnitRefAlt(nn.Module):
    """
    """
    def __init__(self, d_z: int):
        """
        * `d_z` is the dimensionality of $Z$, which is d_ffn of the SGU block
        * `seq_len` is the sequence length
        """
        super(SpacialGatingUnitRefAlt, self).__init__()
        # Normalization layer before applying $f_{W,b}(\cdot)$
        self.norm = nn.LayerNorm([d_z // 2])
        # Weight $W$ in $f_{W,b}(\cdot)$.

        # TODO: maybe let these parameters be element-by-element vectors?
        self.alpha_ref = nn.Parameter(torch.tensor(0.01))
        self.alpha_alt = nn.Parameter(torch.tensor(0.01))
        self.beta_ref = nn.Parameter(torch.tensor(0.01))
        self.beta_alt = nn.Parameter(torch.tensor(0.01))
        self.gamma = nn.Parameter(torch.tensor(0.01))

        # regularizer / sort of imputed value for when there are no ref counts
        self.ref_regularizer = nn.Parameter(0.1 * torch.ones(d_z // 2))
        self.regularizer_weight_pre_exp = nn.Parameter(torch.log(torch.tensor(0.1)))

    def forward(self, zref_brd: RaggedSets, zalt_brd: RaggedSets) -> tuple[RaggedSets, RaggedSets]:

        # Split $Z$ into $Z_1$ and $Z_2$ over the hidden dimension and normalize $Z_2$ before $f_{W,b}(\cdot)$
        z1_ref_brd, z2_ref_brd = zref_brd.split_in_two_by_features()
        z1_alt_brd, z2_alt_brd = zalt_brd.split_in_two_by_features()
        z2_ref_brd = z2_ref_brd.apply_elementwise(self.norm)
        z2_alt_brd = z2_alt_brd.apply_elementwise(self.norm)

        reg_weight = torch.exp(self.regularizer_weight_pre_exp) + 0.25
        # means over reads for each variant (batch index)
        ref_mean_field_bd = z2_ref_brd.means_over_sets(regularizer_f=self.ref_regularizer, regularizer_weight=reg_weight)
        alt_mean_field_bd = z2_alt_brd.means_over_sets()

        # same as above except now there is an additional term for the ref mean field influence on alt
        # maybe later also let alt mean field influence ref
        ref_gate_brd = (z2_ref_brd * self.alpha_ref + 1).broadcast_add(self.beta_ref * ref_mean_field_bd)
        alt_gate_brd = (z2_alt_brd * self.alpha_alt + 1).broadcast_add(self.beta_alt * alt_mean_field_bd).\
            broadcast_add(self.gamma * ref_mean_field_bd)

        # $Z_1 \odot f_{W,b}(Z_2)$
        return z1_ref_brd.multiply_elementwise(ref_gate_brd), z1_alt_brd.multiply_elementwise(alt_gate_brd)


class GatedRefAltMLP(nn.Module):
    def __init__(self, d_model: int, d_ffn: int, num_blocks: int):
        super(GatedRefAltMLP, self).__init__()

        self.blocks = nn.ModuleList([GatedRefAltMLPBlock(d_model, d_ffn) for _ in range(num_blocks)])

    def forward(self, ref_brf: RaggedSets, alt_brf: RaggedSets) -> tuple[RaggedSets, RaggedSets]:
        """
        :param ref_brf: reference read data indexed (conceptually) by batch, read within set, feature
        :param alt_brf: similar for alt read data
        :return: transformed ref and alt sets
        """
        block: GatedRefAltMLPBlock
        for block in self.blocks:
            ref_brf, alt_brf = block.forward(ref_brf, alt_brf)
        return ref_brf, alt_brf
