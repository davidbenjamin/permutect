import torch
from mutect3.data import read_set_datum, read_set_batch
from mutect3.utils import VariantType


# make a three-datum batch
def test_read_set_batch():
    size = 3
    contigs = ['a', 'b', 'c']
    positions = [10, 20, 30]
    refs = ['A', 'C', 'G']
    alts = ['C', 'G', 'TT']

    ref_counts = [6, 11, 7]
    alt_counts = [2, 15, 6]

    ref_tensors = [torch.rand(n, read_set_datum.NUM_READ_FEATURES) for n in ref_counts]
    alt_tensors = [torch.rand(n, read_set_datum.NUM_READ_FEATURES) for n in alt_counts]

    gatk_info_tensors = [torch.rand(read_set_datum.NUM_GATK_NFO_FEATURES) for _ in range(size)]
    labels = ["ARTIFACT", "SOMATIC", "ARTIFACT"]

    # pre-downsampling counts
    tumor_depths = [50, 60, 70]
    tumor_alt_counts = [15, 25, 9]

    normal_depths = [70, 60, 50]
    normal_alt_counts = [0, 1, 2]

    data = [read_set_datum.ReadSetDatum(contigs[n], positions[n], refs[n], alts[n], ref_tensors[n], alt_tensors[n],
                                       gatk_info_tensors[n], labels[n], tumor_depths[n], tumor_alt_counts[n], normal_depths[n],
                                       normal_alt_counts[n]) for n in range(size)]

    batch = read_set_batch.ReadSetBatch(data)

    assert batch.is_labeled()
    assert batch.size() == 3

    assert batch.pd_tumor_depths().tolist() == tumor_depths
    assert batch.pd_tumor_alt_counts().tolist() == tumor_alt_counts

    assert batch.original_list() == data

    assert batch.reads().shape[0] == sum(ref_counts) + sum(alt_counts)
    assert batch.reads().shape[1] == read_set_datum.NUM_READ_FEATURES

    assert batch.info().shape[0] == 3

    assert batch.labels().tolist() == [1.0, 0.0, 1.0]

