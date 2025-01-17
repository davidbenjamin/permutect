import torch
from typing import List

from permutect.architecture.mlp import MLP
from permutect.sets.ragged_sets import RaggedSets

'''
Pytorch module that takes in 2D set tensor X of shape (N, F) and 1D counts tensor Counts, where N is the total *flattened*
number of elements in a batch, summed over all sets.

That is, if the batch consists of N sets with Counts = m_1, m_2. . . m_N, and individual elements have F features, then
N = m_1 + m_2 . . . + m_N.  The pooling operation is done within each set independently of the rest of the batch.  The flattening
is only for the sake of numerical efficiency, but different sets never mix.

Although X is implemented as 2D, conceptually it is a ragged 3D array X_bsf, where b is the index of sets within the batch,
s is the index of elements within a set, and f is the feature index.  Here the range of values of s is different for each set in
the batch, while the range of feature indices f is constant.

Anyway, the operation we wish to carry out is:

    pooling_result_bd = g_3[sum_s{g_1(X_bsf) * Softmax(g_2(X_bsf, dim=-2)}]

where d is the output feature dimension and

    g_1: X_bsf -> Y_bsd is an arbitrary element-wise mapping (in practice an MLP)
    g_2: X_bsf -> Z_bsd is another element-wise mapping (also an MLP)
    Softmax(Z_bsd, dim=-2) -> W_bsd such that sum_s W_bsd = 1 for all b,d is a way to get a normalized collection of weights
        that sum to 1
    sum_s(Y_bsd * W_bsd) -> S_bd takes the weighted average over all s of Y_bsd, using weights W_bsd
    g_3: S_bd -> R_bd is a final nonlinearity (in practice an MLP)

We can think of this as a nonlinear permutation-invariant way of feature-wise choosing the influence of each element.  It
is easy ot see that this scheme can reduce to simple averaging by learning g_1 as the identity and g_2 as a constant.  In fact,
this inspires the appropriate initialization.  It can also express featurewise max-pooling when g_1 is the identity and g_2
is multiplication by a large constant such that the softmax essentially picks out only the largest s for each b and f.  It can also
compute the featurewise variance by calculating second moments in g_1 and averaging.

Given the raggedness of X_bsf the implementation is non-trivial. Fortunately, I wrote a RaggedSets class to handle all this!
'''


class SetPooling(torch.nn.Module):
    def __init__(self, input_dim: int, mlp_layers: List[int], final_mlp_layers: List[int], batch_normalize: bool = False, dropout_p: float = 0):
        super(SetPooling, self).__init__()
        # TODO: layer norm??
        self.mlp1 = MLP([input_dim] + mlp_layers, batch_normalize, dropout_p)
        self.mlp2 = MLP([input_dim] + mlp_layers, batch_normalize, dropout_p)

        self.mlp3 = MLP([self.mlp1.output_dimension()] + final_mlp_layers, batch_normalize, dropout_p)

    def forward(self, x_bsf: RaggedSets) -> torch.Tensor:
        values_bsd = x_bsf.apply_elementwise(self.mlp1)
        weights_bsd = x_bsf.apply_elementwise(self.mlp2).softmax_within_sets()

        weighted_values_bd = values_bsd.multiply_elementwise(weights_bsd).sums_over_sets()
        return self.mlp3.forward(weighted_values_bd)

    def output_dimension(self) -> int:
        return self.mlp3.output_dimension()

