import numpy as np
import torch

import permutect.utils.allele_utils
from permutect.data.reads_datum import ReadsDatum
from permutect.utils.enums import Variation, Label


def test_reads_datum():
    num_ref_reads = 6
    num_alt_reads = 8
    num_read_features = 11
    num_info_features = 9

    ref_tensor = torch.rand(num_ref_reads, num_read_features)
    alt_tensor = torch.rand(num_alt_reads, num_read_features)
    gatk_info_tensor = torch.rand(num_info_features)
    label = Label.ARTIFACT
    source = 0

    snv_datum = ReadsDatum.from_gatk("AC", Variation.SNV, ref_tensor, alt_tensor, gatk_info_tensor, label, source)

    assert torch.equal(snv_datum.reads_re, np.vstack([ref_tensor, alt_tensor]))
    assert permutect.utils.allele_utils.get_variant_type() == Variation.SNV
    assert snv_datum.get_label() == label

    insertion_datum = ReadsDatum.from_gatk("GT", Variation.INSERTION, ref_tensor, alt_tensor, gatk_info_tensor, label, source)
    deletion_datum = ReadsDatum.from_gatk("TT", Variation.DELETION, ref_tensor, alt_tensor, gatk_info_tensor, label, source)
    assert permutect.utils.allele_utils.get_variant_type() == Variation.INSERTION
    assert permutect.utils.allele_utils.get_variant_type() == Variation.DELETION

