import torch

import permutect.data.reads_batch
import permutect.data.reads_datum
from permutect.data.reads_datum import ReadsDatum


# make a three-datum batch
from permutect.utils.enums import Variation, Label


def test_reads_batch():
    size = 3
    num_gatk_info_features = 5

    variant_types = [Variation.SNV, Variation.SNV, Variation.INSERTION]
    num_read_features = 11

    # TODO: test different counts and also test that mixed counts fail
    ref_counts = [11, 11, 11]
    alt_counts = [6, 6, 6]
    ref_sequence_strings = ["ACC", "GTG", "TAA"]

    ref_tensors = [torch.rand(n, num_read_features) for n in ref_counts]
    alt_tensors = [torch.rand(n, num_read_features) for n in alt_counts]

    gatk_info_tensors = [torch.rand(num_gatk_info_features) for _ in range(size)]
    labels = [Label.ARTIFACT, Label.VARIANT, Label.ARTIFACT]
    sources = [0,0,0]

    data = [ReadsDatum.from_gatk(ref_sequence_strings[n], variant_types[n], ref_tensors[n], alt_tensors[n], gatk_info_tensors[n], labels[n], sources[n]) for n in range(size)]

    batch = permutect.data.reads_batch.ReadsBatch(data)

    assert torch.equal(batch.get_one_hot_haplotypes_2d(),
                       torch.Tensor([
                           [[1,0,0],[0,1,1],[0,0,0],[0,0,0]],
                           [[0,0,0],[0,0,0],[1,0,1],[0,1,0]],
                           [[0,1,1],[0,0,0],[0,0,0],[1,0,0]]
                           ])
                       )
    assert batch.size() == 3

    assert batch.get_reads_2d().shape[0] == sum(ref_counts) + sum(alt_counts)
    assert batch.get_reads_2d().shape[1] == num_read_features

    assert batch.get_info_2d().shape[0] == 3

    assert batch.labels.tolist() == [1.0, 0.0, 1.0]

