from typing import List

import torch

from mutect3 import utils
from mutect3.data.normal_artifact_batch import NormalArtifactBatch
from mutect3.data.normal_artifact_datum import NormalArtifactDatum
from mutect3.data.read_set_datum import ReadSetDatum


# given list of slice sizes, produce a list of index slice objects
# eg input = [2,3,1] --> [slice(0,2), slice(2,5), slice(5,6)]
def make_slices(sizes, offset=0):
    slice_ends = offset + torch.cumsum(sizes, dim=0)
    return [slice(offset if n == 0 else slice_ends[n - 1], slice_ends[n]) for n in range(len(sizes))]


# Read sets have different sizes so we can't form a batch by naively stacking tensors.  We need a custom way
# to collate a list of Datum into a Batch

# collated batch contains:
# 2D tensors of ALL ref (alt) reads, not separated by set.
# number of reads in ref (alt) read sets, in same order as read tensors
# info: 2D tensor of info fields, one row per variant
# labels: 1D tensor of 0 if non-artifact, 1 if artifact
# lists of original mutect2_data and site info
# Example: if we have two input data, one with alt reads [[0,1,2], [3,4,5] and the other with
# alt reads [[6,7,8], [9,10,11], [12,13,14] then the output alt reads tensor is
# [[0,1,2], [3,4,5], [6,7,8], [9,10,11], [12,13,14]] and the output counts are [2,3]
# inside the model, the counts will be used to separate the reads into sets
class ReadSetBatch:

    def __init__(self, data: List[ReadSetDatum]):
        self._original_list = data  # keep this for downsampling augmentation
        self.labeled = data[0].label() != "UNLABELED"
        for datum in data:
            if (datum.label() != "UNLABELED") != self.labeled:
                raise Exception("Batch may not mix labeled and unlabeled")

        self._ref_counts = torch.IntTensor([len(item.ref_tensor()) for item in data])
        self._alt_counts = torch.IntTensor([len(item.alt_tensor()) for item in data])
        self._ref_slices = make_slices(self._ref_counts)
        self._alt_slices = make_slices(self._alt_counts, torch.sum(self._ref_counts))
        self._reads = torch.cat([item.ref_tensor() for item in data] + [item.alt_tensor() for item in data], dim=0)
        self._info = torch.stack([item.info_tensor() for item in data], dim=0)
        self._labels = torch.FloatTensor([1.0 if item.label() == "ARTIFACT" else 0.0 for item in data]) if self.labeled else None
        self._ref = [item.ref() for item in data]
        self._alt = [item.alt() for item in data]
        self._variant_type = [item.variant_type() for item in data]
        self._size = len(data)

        # pre-downsampled allele counts
        self._pd_tumor_depths = torch.IntTensor([item.tumor_depth() for item in data])
        self._pd_tumor_alt_counts = torch.IntTensor([item.tumor_alt_count() for item in data])

        # TODO: variant type needs to go in constructor -- and maybe it should be utils.VariantType, not str
        # TODO: we might need to change the counts in this constructor
        normal_artifact_data = [NormalArtifactDatum(item.normal_alt_count(), item.normal_depth(),
                                                    item.tumor_alt_count(),item.tumor_depth(),
                                                    1.0, item.variant_type) for item in data]
        self._normal_artifact_batch = NormalArtifactBatch(normal_artifact_data)

    def augmented_copy(self, beta):
        return ReadSetBatch([datum.downsampled_copy(beta) for datum in self._original_list])

    def original_list(self) -> List[ReadSetDatum]:
        return self._original_list

    def is_labeled(self) -> bool:
        return self.labeled

    def size(self) -> int:
        return self._size

    def reads(self) -> torch.Tensor:
        return self._reads

    def ref_slices(self) -> List[slice]:
        return self._ref_slices

    def alt_slices(self) -> List[slice]:
        return self._alt_slices

    def ref_counts(self) -> torch.IntTensor:
        return self._ref_counts

    def alt_counts(self) -> torch.IntTensor:
        return self._alt_counts

    def pd_tumor_depths(self) -> torch.IntTensor:
        return self._pd_tumor_depths

    def pd_tumor_alt_counts(self) -> torch.IntTensor:
        return self._pd_tumor_alt_counts

    def info(self) -> torch.Tensor:
        return self._info

    def labels(self) -> torch.Tensor:
        return self._labels

    def variant_type(self) -> List[utils.VariantType]:
        return self._variant_type

    def normal_artifact_batch(self) -> NormalArtifactBatch:
        return self._normal_artifact_batch
