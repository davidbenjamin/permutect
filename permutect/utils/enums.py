import enum


class Variation(enum.IntEnum):
    SNV = 0
    INSERTION = 1
    DELETION = 2
    BIG_INSERTION = 3
    BIG_DELETION = 4

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
    NORMAL_ARTIFACT = 4


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