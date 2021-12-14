import enum


# variant size is alt - ref length
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