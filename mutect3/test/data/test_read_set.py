import torch

from mutect3 import utils
from mutect3.data import read_set
from mutect3.utils import VariantType


def test_read_set_datum():
    num_ref_reads = 6
    num_alt_reads = 8
    num_read_features = 11

    ref_tensor = torch.rand(num_ref_reads, num_read_features)
    alt_tensor = torch.rand(num_alt_reads, num_read_features)
    gatk_info_tensor = torch.rand(read_set.NUM_GATK_INFO_FEATURES)
    label = "artifact"

    snv_datum = read_set.ReadSet(VariantType.SNV, ref_tensor, alt_tensor, gatk_info_tensor, label)

    assert torch.equal(snv_datum.ref_tensor(), ref_tensor)
    assert torch.equal(snv_datum.alt_tensor(), alt_tensor)
    assert torch.equal(snv_datum.info_tensor(), gatk_info_tensor)
    assert snv_datum.label() == label
    assert snv_datum.variant_type() == VariantType.SNV

    insertion_datum = read_set.ReadSet(VariantType.INSERTION, ref_tensor, alt_tensor, gatk_info_tensor, label)
    assert insertion_datum.variant_type() == VariantType.INSERTION

    deletion_datum = read_set.ReadSet(VariantType.DELETION, ref_tensor, alt_tensor, gatk_info_tensor, label)
    assert deletion_datum.variant_type() == VariantType.DELETION
