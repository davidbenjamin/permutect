import torch

from mutect3.data import read_set
from mutect3.utils import Variation, Label


def test_read_set():
    num_ref_reads = 6
    num_alt_reads = 8
    num_read_features = 11
    num_info_features = 9

    ref_tensor = torch.rand(num_ref_reads, num_read_features)
    alt_tensor = torch.rand(num_alt_reads, num_read_features)
    gatk_info_tensor = torch.rand(num_info_features)
    label = Label.ARTIFACT

    snv_datum = read_set.ReadSet.from_gatk("AC", Variation.SNV, ref_tensor, alt_tensor, gatk_info_tensor, label)

    assert torch.equal(snv_datum.ref_sequence_tensor, torch.Tensor([[1, 0], [0, 1], [0, 0], [0, 0]]))
    assert torch.equal(snv_datum.ref_tensor(), ref_tensor)
    assert torch.equal(snv_datum.alt_tensor(), alt_tensor)
    assert torch.equal(snv_datum.info_tensor()[:-len(Variation)], gatk_info_tensor)
    assert snv_datum.label() == label

    insertion_datum = read_set.ReadSet.from_gatk("GT", Variation.INSERTION, ref_tensor, alt_tensor, gatk_info_tensor, label)
    deletion_datum = read_set.ReadSet.from_gatk("TT", Variation.DELETION, ref_tensor, alt_tensor, gatk_info_tensor, label)

    assert insertion_datum.info_tensor()[-len(Variation) + Variation.INSERTION.value] == 1
    assert insertion_datum.info_tensor()[-len(Variation) + Variation.DELETION.value] == 0
    assert insertion_datum.info_tensor()[-len(Variation) + Variation.SNV.value] == 0

    assert deletion_datum.info_tensor()[-len(Variation) + Variation.INSERTION.value] == 0
    assert deletion_datum.info_tensor()[-len(Variation) + Variation.DELETION.value] == 1
    assert deletion_datum.info_tensor()[-len(Variation) + Variation.SNV.value] == 0
