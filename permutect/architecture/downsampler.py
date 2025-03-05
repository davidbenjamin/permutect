import torch
from torch import IntTensor
from torch.nn import Module, Parameter

from permutect.data.batch import BatchIndexedTensor
from permutect.data.count_binning import MAX_REF_COUNT, MIN_ALT_COUNT, MAX_ALT_COUNT, NUM_REF_COUNT_BINS, \
    NUM_ALT_COUNT_BINS, ref_count_bin_index, alt_count_bin_index, COUNT_BIN_SKIP
from permutect.utils.enums import Label, Variation
from permutect.utils.stats_utils import beta_binomial_log_lk


class Downsampler(Module):
    # downsampling is done as a mixture of beta binomials with a *fixed* set of basis beta distributions
    BETA_BASIS_SHAPES = [(1, 1), (1, 5), (5, 1), (5, 5)]

    def __init__(self, num_sources: int):
        super(Downsampler, self).__init__()
        self.num_sources = num_sources

        # these are 3D -- with two dummy read count indices -- for broadcasting
        alpha_k11 = torch.tensor([shape[0] for shape in Downsampler.BETA_BASIS_SHAPES]).view(-1, 1, 1)
        beta_k11 = torch.tensor([shape[1] for shape in Downsampler.BETA_BASIS_SHAPES]).view(-1, 1, 1)

        # these are 'raw' as opposed to binned
        raw_ref_counts_r = IntTensor(range(MAX_REF_COUNT + 1))
        raw_alt_counts_a = IntTensor(range(MAX_ALT_COUNT + 1))

        # we're about to have a lot of indices.  'r' and 'a' denote ref and alt counts as usual.  'y' and 'z' denote downsampled
        # ref and alt counts, so for example an alt downsampling transition matrix is indexed as k, a, v, where k is the
        # basis index.  "trans," by the way, is short for "transition" in the sense of transition to downsampled read counts

        ref_kry = raw_ref_counts_r.view(1, -1, 1)       # k, y are dummy indices for broadcasting
        downref_kry = raw_ref_counts_r.view(1, 1, -1)   # k, r are dummy indices for broadcasting
        alt_haz = raw_alt_counts_a.view(1, -1, 1)       # h, v are dummy indices for broadcasting
        downalt_haz = raw_alt_counts_a.view(1, 1, -1)   # H, a are dummy indices for broadcasting

        ref_trans_kry = torch.exp(beta_binomial_log_lk(n=ref_kry, k=downref_kry, alpha=alpha_k11, beta=beta_k11))
        alt_trans_haz = torch.exp(beta_binomial_log_lk(n=alt_haz, k=downalt_haz, alpha=alpha_k11, beta=beta_k11))

        # now we need to bin this.  If you think about it carefully, the appropriate thing to do is *sum* over downsampled
        # counts that correspond to the same bin and to *average* over original counts.  We could do this in some fancy vectorized
        # way, but because this is a one-time cost during construction we employ a slower but clearer approach.
        # we implement the average by summing over everything and dividing by the count bin skip, which is the number
        # of counts per bin

        binned_ref_trans_kry = torch.zeros(len(Downsampler.BETA_BASIS_SHAPES), NUM_REF_COUNT_BINS, NUM_REF_COUNT_BINS)
        binned_alt_trans_haz = torch.zeros(len(Downsampler.BETA_BASIS_SHAPES), NUM_ALT_COUNT_BINS, NUM_ALT_COUNT_BINS)

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

        self.binned_ref_trans_kru = Parameter(binned_ref_trans_kry/ COUNT_BIN_SKIP, requires_grad=False)
        self.binned_alt_trans_hav = Parameter(binned_alt_trans_haz / COUNT_BIN_SKIP, requires_grad=False)

        # for each source/label/var type/ ref/alt bin we have mixture component weights for both ref and alt downsampling
        # 'k' and 'h' are both indices to denote the mixture components
        self.ref_weights_pre_softmax_slvrak = Parameter(torch.zeros(num_sources, len(Label), len(Variation), NUM_REF_COUNT_BINS,
                NUM_ALT_COUNT_BINS, len(Downsampler.BETA_BASIS_SHAPES)), requires_grad=True)
        self.alt_weights_pre_softmax_slvrah = Parameter(torch.zeros(num_sources, len(Label), len(Variation), NUM_REF_COUNT_BINS,
                NUM_ALT_COUNT_BINS, len(Downsampler.BETA_BASIS_SHAPES)), requires_grad=True)

        # TODO: check that the code below is unneeded.  I think we get exp(-inf) = 0 when k > n, which is what we want.
        # note the "where" here -- transition probability is zero when the downsampled count exceeds the original count
        # ref_trans_kry = torch.where(ref_kry >= downref_kru, torch.exp(log_ref_trans_kru), 0)

    def calculate_expected_downsampled_counts(self, counts_slvra: BatchIndexedTensor):
        ref_weights_slvrak = torch.softmax(self.ref_weights_pre_softmax_slvrak, dim=-1)
        alt_weights_slvrah = torch.softmax(self.alt_weights_pre_softmax_slvrah, dim=-1)

        # in words, we're calculating (recall that y, z are downsampled ref, alt count bins)
        # result_slvyz = sum_{rakh} counts_slvra * ref_weights_slvrak * alt_weights_slvrah * ref_trans_kry * alt_trans_haz

        # TODO: look up howto einsum this

