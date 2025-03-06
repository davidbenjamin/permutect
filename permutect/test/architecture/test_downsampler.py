from random import random

from permutect.architecture.downsampler import Downsampler
from permutect.data.batch import BatchIndexedTensor
from permutect.data.count_binning import NUM_REF_COUNT_BINS, NUM_ALT_COUNT_BINS
from permutect.utils.enums import Label, Variation


def test_learning():
    downsampler = Downsampler(num_sources=1)

    counts_slvra = BatchIndexedTensor.make_zeros(num_sources=1)
    for label in Label:
        for var_type in Variation:
            for ref_count_bin in range(NUM_REF_COUNT_BINS):
                for alt_count_bin in range(NUM_ALT_COUNT_BINS):
                    counts_slvra[0, label, var_type, ref_count_bin, alt_count_bin] = 1000 * random()

    downsampler.optimize_downsampling_balance(counts_slvra)