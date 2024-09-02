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

from typing import Optional

import torch
from torch import nn


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
        self.activation = nn.GELU()
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

    def forward(self, x_bre: torch.Tensor):
        """
        * `x_bre` is the input read embedding tensor of shape Batch x Reads x Embedding
        """
        # Norm, projection to d_ffn, and activation $Z = \sigma(XU)$
        z_brd = self.activation(self.proj1(self.norm(x_bre)))
        # Spacial Gating Unit $\tilde{Z} = s(Z)$
        gated_brd = self.sgu(z_brd)
        # Final projection $Y = \tilde{Z}V$ back to embedding dimension
        gated_bre = self.proj2(gated_brd)

        # Add the shortcut connection
        return x_bre + gated_bre


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
        #
        self.alpha = nn.Parameter(torch.tensor(0.01))
        self.beta = nn.Parameter(torch.tensor(0.01))

    def forward(self, z_brd: torch.Tensor):
        """
        * `z_brd` is the input tensor of shape Batch x Reads x Dimension
        `[seq_len, batch_size, d_z]`
        """

        # Split $Z$ into $Z_1$ and $Z_2$ over the hidden dimension and normalize $Z_2$ before $f_{W,b}(\cdot)$
        z1_brd, z2_brd = torch.chunk(z_brd, 2, dim=-1)
        z2_brd = self.norm(z2_brd)

        z2_brd = 1 + self.alpha * z2_brd + torch.mean(z2_brd, dim=1)[:, None, :]

        # $Z_1 \odot f_{W,b}(Z_2)$
        return z1_brd * z2_brd


class GatedMLP(nn.Module):
    def __init__(self, d_model: int, d_ffn: int, num_blocks: int):
        super(GatedMLP, self).__init__()

        self.blocks = nn.ModuleList([GatedMLPBlock(d_model, d_ffn) for _ in range(num_blocks)])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x