import enum

# variant size is alt - ref length
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