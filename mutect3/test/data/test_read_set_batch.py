import torch
from mutect3.data import read_set, read_set_batch
from mutect3.data.read_set import ReadSet


# make a three-datum batch
from mutect3.utils import VariantType


def test_read_set_batch():
    size = 3

    variant_types = [VariantType.SNV, VariantType.SNV, VariantType.INSERTION]
    num_read_features = 11
    ref_counts = [6, 11, 7]
    alt_counts = [2, 15, 6]

    ref_tensors = [torch.rand(n, num_read_features) for n in ref_counts]
    alt_tensors = [torch.rand(n, num_read_features) for n in alt_counts]

    gatk_info_tensors = [torch.rand(read_set.NUM_GATK_NFO_FEATURES) for _ in range(size)]
    labels = ["ARTIFACT", "SOMATIC", "ARTIFACT"]

    data = [ReadSet(variant_types[n], ref_tensors[n], alt_tensors[n], gatk_info_tensors[n], labels[n]) for n in range(size)]

    batch = read_set_batch.ReadSetBatch(data)

    assert batch.is_labeled()
    assert batch.size() == 3

    assert batch.original_list() == data

    assert batch.reads().shape[0] == sum(ref_counts) + sum(alt_counts)
    assert batch.reads().shape[1] == num_read_features

    assert batch.info().shape[0] == 3

    assert batch.labels().tolist() == [1.0, 0.0, 1.0]

