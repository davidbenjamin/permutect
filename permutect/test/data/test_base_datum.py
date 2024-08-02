import numpy as np
import torch

from permutect.data import base_datum
from permutect.utils import Variation, Label


def test_base_datum():
    num_ref_reads = 6
    num_alt_reads = 8
    num_read_features = 11
    num_info_features = 9

    ref_tensor = torch.rand(num_ref_reads, num_read_features)
    alt_tensor = torch.rand(num_alt_reads, num_read_features)
    gatk_info_tensor = torch.rand(num_info_features)
    label = Label.ARTIFACT

    snv_datum = base_datum.BaseDatum.from_gatk("AC", Variation.SNV, ref_tensor, alt_tensor, gatk_info_tensor, label)

    assert torch.equal(snv_datum.get_ref_sequence_1d(), torch.Tensor([0,1]))
    assert torch.equal(snv_datum.reads_2d, np.vstack([ref_tensor, alt_tensor]))
    assert torch.equal(snv_datum.variant_type_one_hot(), gatk_info_tensor)
    assert snv_datum.label == label

    insertion_datum = base_datum.BaseDatum.from_gatk("GT", Variation.INSERTION, ref_tensor, alt_tensor, gatk_info_tensor, label)
    deletion_datum = base_datum.BaseDatum.from_gatk("TT", Variation.DELETION, ref_tensor, alt_tensor, gatk_info_tensor, label)

    insertion_datum_variant_one_hot = insertion_datum.variant_type_one_hot()
    assert insertion_datum_variant_one_hot[Variation.INSERTION.value] == 1
    assert insertion_datum_variant_one_hot[Variation.DELETION.value] == 0
    assert insertion_datum_variant_one_hot[Variation.SNV.value] == 0

    deletion_datum_variant_one_hot = deletion_datum.variant_type_one_hot()

    assert deletion_datum_variant_one_hot[Variation.INSERTION.value] == 0
    assert deletion_datum_variant_one_hot[Variation.DELETION.value] == 1
    assert deletion_datum_variant_one_hot[Variation.SNV.value] == 0
