import torch
from torch import IntTensor
from torch import Tensor
from torch.distributions import Beta
from torch.nn import Module
from torch.nn import Parameter
from torch.nn.parameter import Buffer
from torch.nn.utils import parametrize

from permutect import misc_utils
from permutect.architecture.parameterizations import LogWeights
from permutect.data.batch import Batch
from permutect.data.batch import BatchIndexedTensor
from permutect.data.count_binning import COUNT_BIN_SKIP
from permutect.data.count_binning import MAX_ALT_COUNT
from permutect.data.count_binning import MAX_REF_COUNT
from permutect.data.count_binning import MIN_ALT_COUNT
from permutect.data.count_binning import NUM_ALT_COUNT_BINS
from permutect.data.count_binning import NUM_REF_COUNT_BINS
from permutect.data.count_binning import alt_count_bin_index
from permutect.data.count_binning import ref_count_bin_index
from permutect.misc_utils import backpropagate
from permutect.utils.stats_utils import beta_binomial_log_lk


class Downsampler(Module):
    # downsampling is done as a mixture of beta binomials with a *fixed* set of basis beta distributions
    BETA_BASIS_SHAPES = torch.tensor([[1.0, 1.0], [1.0, 5.0], [5.0, 1.0], [5.0, 5.0]])

    def __init__(self, num_sources: int):
        super(Downsampler, self).__init__()
        self.num_sources = num_sources

        # copy this as a member so that it gets moved to the correct device with the downsampler
        self.beta_basis = Buffer(Downsampler.BETA_BASIS_SHAPES)

        # these are 3D -- with two dummy read count indices -- for broadcasting
        alpha_k11 = self.beta_basis[:, 0].view(-1, 1, 1)
        beta_k11 = self.beta_basis[:, 1].view(-1, 1, 1)

        # these are 'raw' as opposed to binned
        raw_ref_counts_r = IntTensor(range(MAX_REF_COUNT + 1))
        raw_alt_counts_a = IntTensor(range(MAX_ALT_COUNT + 1))

        # we're about to have a lot of indices.  'r' and 'a' denote ref and alt counts as usual.  'y' and 'z' denote downsampled
        # ref and alt counts, so for example an alt downsampling transition matrix is indexed as k, a, v, where k is the
        # basis index.  "trans," by the way, is short for "transition" in the sense of transition to downsampled read counts

        ref_kry = raw_ref_counts_r.view(1, -1, 1)  # k, y are dummy indices for broadcasting
        downref_kry = raw_ref_counts_r.view(1, 1, -1)  # k, r are dummy indices for broadcasting
        alt_haz = raw_alt_counts_a.view(1, -1, 1)  # h, v are dummy indices for broadcasting
        downalt_haz = raw_alt_counts_a.view(1, 1, -1)  # H, a are dummy indices for broadcasting

        ref_trans_kry = torch.where(
            ref_kry >= downref_kry,
            torch.exp(beta_binomial_log_lk(n=ref_kry, k=downref_kry, alpha=alpha_k11, beta=beta_k11)),
            0,
        )
        alt_trans_haz = torch.where(
            alt_haz >= downalt_haz,
            torch.exp(beta_binomial_log_lk(n=alt_haz, k=downalt_haz, alpha=alpha_k11, beta=beta_k11)),
            0,
        )

        # now we need to bin this.  If you think about it carefully, the appropriate thing to do is *sum* over downsampled
        # counts that correspond to the same bin and to *average* over original counts.  We could do this in some fancy vectorized
        # way, but because this is a one-time cost during construction we employ a slower but clearer approach.
        # we implement the average by summing over everything and dividing by the count bin skip, which is the number
        # of counts per bin

        binned_ref_trans_kry = torch.zeros(len(self.beta_basis), NUM_REF_COUNT_BINS, NUM_REF_COUNT_BINS)
        binned_alt_trans_haz = torch.zeros(len(self.beta_basis), NUM_ALT_COUNT_BINS, NUM_ALT_COUNT_BINS)

        for ref in range(MAX_REF_COUNT + 1):
            ref_bin = ref_count_bin_index(ref)
            for downref in range(MAX_REF_COUNT + 1):
                downref_bin = ref_count_bin_index(downref)
                binned_ref_trans_kry[:, ref_bin, downref_bin] += ref_trans_kry[:, ref, downref]

        for alt in range(MAX_ALT_COUNT + 1):
            alt_bin = 0 if alt < MIN_ALT_COUNT else alt_count_bin_index(alt)
            for downalt in range(MAX_ALT_COUNT + 1):
                # in the downsampling we guarantee at least one alt read count, so a beta binomial that samples no
                # alt reads end up with one read, hence is in bin 0.
                downalt_bin = 0 if downalt < MIN_ALT_COUNT else alt_count_bin_index(downalt)
                binned_alt_trans_haz[:, alt_bin, downalt_bin] += alt_trans_haz[:, alt, downalt]

        self.binned_ref_trans_kry = Buffer(binned_ref_trans_kry / COUNT_BIN_SKIP)
        self.binned_alt_trans_haz = Buffer(binned_alt_trans_haz / COUNT_BIN_SKIP)

        # for each source/label/var type/ ref/alt bin we have mixture component weights for both ref and alt downsampling
        # 'k' and 'h' are both indices to denote the mixture components
        slvra_shape = BatchIndexedTensor.shape_without_logits(num_sources)
        slvrak_shape = slvra_shape + (len(self.beta_basis),)

        # the log weights tensors should never be changed when training a model and as such should usually never
        # have gradients.  However, 1) defining them as parameters lets us parametrize them with LogWeights and
        # 2) we do train them when fitting the downsampler.  Thus we define them as Parameters rather than Buffers
        # BUT we always detach() them.  Furthermore, when training a model the optimizer should only involve the
        # model's parameters, not the downsampler's
        self.log_ref_weights_slvrak = Parameter(torch.zeros(slvrak_shape))
        parametrize.register_parametrization(self, "log_ref_weights_slvrak", LogWeights())
        self.log_alt_weights_slvrah = Parameter(torch.zeros(slvrak_shape))
        parametrize.register_parametrization(self, "log_alt_weights_slvrah", LogWeights())

    def weights_parameters(self):
        return [
            self.parametrizations.log_ref_weights_slvrak.original,
            self.parametrizations.log_alt_weights_slvrah.original,
        ]

    # this is used in model training, NOT for fitting the downsampler, so for security we detach the weights
    def calculate_downsampling_fractions(self, batch: Batch) -> tuple[Tensor, Tensor]:
        # we will flatten all the batch indices -- slvra, but not k -- to get a 2D tensor indexed by the batch's
        # flattened indices ('f') and the downsampler's mixture index k
        ref_weights_fk = torch.exp(self.log_ref_weights_slvrak.detach()).view(-1, len(self.beta_basis))
        alt_weights_fk = torch.exp(self.log_alt_weights_slvrah.detach()).view(-1, len(self.beta_basis))

        # use the weights for choosing from random samples
        ref_weights_bk = ref_weights_fk[batch.batch_indices().flattened_idx]
        alt_weights_bk = alt_weights_fk[batch.batch_indices().flattened_idx]

        refk_b = torch.multinomial(ref_weights_bk, num_samples=1)  # one sample for each batch element
        altk_b = torch.multinomial(alt_weights_bk, num_samples=1)

        # The '2' indicates that each beta shape has two elements, alpha and beta
        ref_shapes_b2 = self.beta_basis[refk_b.flatten()]
        alt_shapes_b2 = self.beta_basis[altk_b.flatten()]
        ref_fracs_b = Beta(ref_shapes_b2[:, 0], ref_shapes_b2[:, 1]).sample().view(-1)
        alt_fracs_b = Beta(alt_shapes_b2[:, 0], alt_shapes_b2[:, 1]).sample().view(-1)
        return ref_fracs_b, alt_fracs_b

    # used to fit the downsampler, log weights are NOT detach()ed
    def calculate_expected_downsampled_counts(self, counts_slvra: BatchIndexedTensor):
        ref_weights_slvrak = torch.exp(self.log_ref_weights_slvrak)
        alt_weights_slvrah = torch.exp(self.log_alt_weights_slvrah)

        # in words, we're calculating (recall that y, z are downsampled ref, alt count bins)
        # result_slvyz = sum_{rakh} counts_slvra * ref_weights_slvrak * alt_weights_slvrah * ref_trans_kry * alt_trans_haz
        result_slvyz = torch.einsum(
            "slvra, slvrak, slvrah, kry, haz->slvyz",
            counts_slvra,
            ref_weights_slvrak,
            alt_weights_slvrah,
            self.binned_ref_trans_kry,
            self.binned_alt_trans_haz,
        )
        return result_slvyz

    def optimize_downsampling_balance(self, counts_slvra: BatchIndexedTensor):
        optimizer = torch.optim.AdamW(self.parameters())
        # TODO: magic constant -- choose when to end optimization more intelligently
        for step in range(10000):
            expected_slvyz = self.calculate_expected_downsampled_counts(counts_slvra)

            # divide by the total over all counts for each slv bin to get a probability distribution over output r/a count bins
            normalized_slvyz = expected_slvyz / torch.sum(expected_slvyz, dim=(-2, -1), keepdim=True)

            # we want downsampled counts to be as even as possible among all ref and alt count bins.  This is equivalent
            # to minimizing the sum of squared downsampled counts.  Since the downsampling
            # preserves total probability, it can't minimize this loss by scaling down the result.  It can only minimize
            # it by redistributing.
            sums_of_squares_slv = torch.sum(torch.square(normalized_slvyz), dim=(-2, -1))
            loss = torch.sum(sums_of_squares_slv)
            backpropagate(optimizer, loss)
