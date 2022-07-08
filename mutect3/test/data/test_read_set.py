import torch

from mutect3 import utils


def test_read_set_datum():
    contig = "contig"
    position = 100
    ref = "A"
    alt = "C"

    num_ref_reads = 6
    num_alt_reads = 8

    ref_tensor = torch.rand(num_ref_reads, read_set_datum.NUM_READ_FEATURES)
    alt_tensor = torch.rand(num_alt_reads, read_set_datum.NUM_READ_FEATURES)
    gatk_info_tensor = torch.rand(read_set_datum.NUM_GATK_INFO_FEATURES)
    label = "artifact"

    tumor_depth = 50
    tumor_alt_count = 15

    normal_depth = 70
    normal_alt_count = 1

    snv_datum = read_set_datum.ReadSet(contig, position, ref, alt, ref_tensor, alt_tensor, gatk_info_tensor, label,
                                       tumor_depth, tumor_alt_count, normal_depth, normal_alt_count)

    assert snv_datum.contig() == contig
    assert snv_datum.position() == position
    assert snv_datum.ref() == ref
    assert snv_datum.alt() == alt
    assert torch.equal(snv_datum.ref_tensor(), ref_tensor)
    assert torch.equal(snv_datum.alt_tensor(), alt_tensor)
    assert torch.equal(snv_datum.info_tensor(), gatk_info_tensor)
    assert snv_datum.label() == label
    assert snv_datum.tumor_depth() == tumor_depth
    assert snv_datum.tumor_alt_count() == tumor_alt_count
    assert snv_datum.normal_depth() == normal_depth
    assert snv_datum.normal_alt_count() == normal_alt_count

    new_label = "somatic"
    snv_datum.set_label(new_label)
    assert snv_datum.label() == new_label

    assert snv_datum.variant_type() == utils.VariantType.SNV

    insertion_datum = read_set_datum.ReadSet(contig, position, "A", "AC", ref_tensor, alt_tensor, gatk_info_tensor,
                                             label, tumor_depth, tumor_alt_count, normal_depth, normal_alt_count)

    assert insertion_datum.variant_type() == utils.VariantType.INSERTION

    deletion_datum = read_set_datum.ReadSet(contig, position, "AC", "A", ref_tensor, alt_tensor, gatk_info_tensor, label,
                                            tumor_depth, tumor_alt_count, normal_depth, normal_alt_count)

    assert deletion_datum.variant_type() == utils.VariantType.DELETION
