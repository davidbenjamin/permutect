import enum


# variant size is alt - ref length
import torch
from torch.utils.data import random_split


def get_variant_type(alt_allele, ref_allele):
    variant_size = len(alt_allele) - len(ref_allele)
    if variant_size == 0:
        return VariantType.SNV
    else:
        return VariantType.INSERTION if variant_size > 0 else VariantType.DELETION


class VariantType(enum.IntEnum):
    SNV = 0
    INSERTION = 1
    DELETION = 2


class EpochType(enum.Enum):
    TRAIN = "train"
    VALID = "valid"
    TEST = "test"


def split_dataset_into_train_and_valid(dataset, train_fraction=0.9):
    train_len = int(0.9 * len(dataset))
    valid_len = len(dataset) - train_len
    return random_split(dataset, lengths=[train_len, valid_len])


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
def beta_binomial(n, k, alpha, beta):
    return torch.lgamma(k + alpha) + torch.lgamma(n - k + beta) + torch.lgamma(alpha + beta) \
           - torch.lgamma(n + alpha + beta) - torch.lgamma(alpha) - torch.lgamma(beta)