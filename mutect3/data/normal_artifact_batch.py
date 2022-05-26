from typing import List

import torch

from mutect3.data.normal_artifact_datum import NormalArtifactDatum


class NormalArtifactBatch:

    def __init__(self, data: List[NormalArtifactDatum]):
        self._normal_alt = torch.IntTensor([datum.normal_alt_count() for datum in data])
        self._normal_depth = torch.IntTensor([datum.normal_depth() for datum in data])
        self._tumor_alt = torch.IntTensor([datum.tumor_alt_count() for datum in data])
        self._tumor_depth = torch.IntTensor([datum.tumor_depth() for datum in data])
        self._downsampling = torch.FloatTensor([datum.downsampling() for datum in data])
        self._variant_type = [datum.variant_type() for datum in data]
        self._size = len(data)

    def size(self) -> int:
        return self._size

    def normal_alt(self) -> torch.IntTensor:
        return self._normal_alt

    def normal_depth(self) -> torch.IntTensor:
        return self._normal_depth

    def tumor_alt(self) -> torch.IntTensor:
        return self._tumor_alt

    def tumor_depth(self) -> torch.IntTensor:
        return self._tumor_depth

    def downsampling(self) -> torch.FloatTensor:
        return self._downsampling

    def variant_type(self) -> List[str]:
        return self._variant_type