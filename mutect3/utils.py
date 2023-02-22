import enum
import numpy as np
from mutect3.data.posterior import PosteriorDatum
import cyvcf2
import tempfile
import tarfile
import os


# variant size is alt - ref length
import torch
from torch.utils.data import random_split


class ConsistentValue:
    """
    Tracks a value that once initialized, is consistent among eg all members of a dataset.  For example, all tensors
    must have the same number of columns.
    """
    def __init__(self, value=None):
        self.value = value

    def check(self, value):
        if self.value is None:
            self.value = value
        else:
            assert self.value == value, "inconsistent values"


def downsample_tensor(tensor2d: np.ndarray, new_length: int):
    if tensor2d is None or new_length >= len(tensor2d):
        return tensor2d
    perm = np.random.permutation(len(tensor2d))
    return tensor2d[perm[:new_length]]


def get_variant_type(alt_allele, ref_allele):
    variant_size = len(alt_allele) - len(ref_allele)
    if variant_size == 0:
        return Variation.SNV
    else:
        return Variation.INSERTION if variant_size > 0 else Variation.DELETION


class Variation(enum.IntEnum):
    SNV = 0
    INSERTION = 1
    DELETION = 2
    BIG_INSERTION = 3
    BIG_DELETION = 4

    def one_hot_tensor(self):
        result = np.zeros(len(Variation))
        result[self.value] = 1
        return result

    @staticmethod
    def get_type(ref_allele: str, alt_allele: str):
        diff = len(alt_allele) - len(ref_allele)
        if diff == 0:
            return Variation.SNV
        elif diff > 0:
            return Variation.BIG_INSERTION if diff > 1 else Variation.INSERTION
        else:
            return Variation.BIG_DELETION if diff < -1 else Variation.DELETION


class Call(enum.IntEnum):
    SOMATIC = 0
    ARTIFACT = 1
    SEQ_ERROR = 2
    GERMLINE = 3


class Epoch(enum.IntEnum):
    TRAIN = 0
    VALID = 1
    TEST = 2


class Label(enum.IntEnum):
    ARTIFACT = 0
    VARIANT = 1
    UNLABELED = 2

    @staticmethod
    def get_label(label_str: str):
        for label in Label:
            if label_str == label.name:
                return label

        raise ValueError('label is invalid: %s' % label_str)

    @staticmethod
    def is_label(label_str: str):
        for label in Label:
            if label_str == label.name:
                return True

        return False


def freeze(parameters):
    for parameter in parameters:
        parameter.requires_grad = False


def unfreeze(parameters):
    for parameter in parameters:
        parameter.requires_grad = True


def f_score(tp, fp, total_true):
    fn = total_true - tp
    return tp / (tp + (fp + fn) / 2)


# note: this function works for n, k, alpha, beta tensors of the same shape
# the result is computed element-wise ie result[i,j. . .] = beta_binomial(n[i,j..], k[i,j..], alpha[i,j..], beta[i,j..)
# often n, k will correspond to a batch dimension and alpha, beta correspond to a model, in which case
# unsqueezing is necessary
# NOTE: this excludes the nCk factor
def beta_binomial(n, k, alpha, beta):
    return torch.lgamma(k + alpha) + torch.lgamma(n - k + beta) + torch.lgamma(alpha + beta) \
           - torch.lgamma(n + alpha + beta) - torch.lgamma(alpha) - torch.lgamma(beta)


class StreamingAverage:
    def __init__(self, device="cpu"):
        self._count = 0
        self._sum = torch.tensor([0.0], device=device)

    def is_empty(self):
        return self._count == 0

    # this is the only method where, if we're on GPU, self._sum is transferred to the CPU
    def get(self):
        return self._sum.item() / (self._count + 0.0001)

    # value should live on same device as self._sum
    def record(self, value: torch.Tensor):
        self._count += 1
        self._sum += value

    # value_sum should live on same device as self._sum
    def record_sum(self, value_sum: torch.Tensor, count: int):
        self._count += count
        self._sum += value_sum

    # record only values masked as true
    # values and mask should live on same device as self._sum
    def record_with_mask(self, values: torch.Tensor, mask: torch.Tensor):
        self._count += torch.sum(mask)
        self._sum += torch.sum(values*mask)


def log_binomial_coefficient(n: torch.Tensor, k: torch.Tensor):
    return (n + 1).lgamma() - (k + 1).lgamma() - ((n - k) + 1).lgamma()


def encode(contig: str, position: int, alt: str):
    return contig + ':' + str(position) + ':' + alt


def encode_datum(datum: PosteriorDatum):
    return encode(datum.contig, datum.position, datum.alt)


def encode_variant(v: cyvcf2.Variant, zero_based=False):
    alt = v.ALT[0]  # TODO: we're assuming biallelic
    start = (v.start + 1) if zero_based else v.start
    return encode(v.CHROM, start, alt)


def extract_to_temp_dir(tar_file, directory):
    tar = tarfile.open(tar_file)
    tar.extractall(directory)
    tar.close()
    return [os.path.abspath(os.path.join(directory, p)) for p in os.listdir(directory)]
