from typing import List

import torch
from mutect3.data.posterior_datum import PosteriorDatum


class PosteriorBatch:

    def __init__(self, data: List[PosteriorDatum]):
        self._original_list = data  # keep this for downsampling augmentation
        self._size = len(data)

        self.depths = torch.IntTensor([item.depth for item in data])
        self.alt_counts = torch.IntTensor([item.alt_count for item in data])
        self.normal_depths = torch.IntTensor([item.normal_depth for item in data])
        self.normal_alt_counts = torch.IntTensor([item.normal_alt_count for item in data])

        self._variant_type_one_hot = torch.vstack([item.variant_type.one_hot_tensor() for item in self._original_list])

        self.seq_error_log_likelihoods = torch.Tensor([item.seq_error_log_likelihood for item in self._original_list])
        self.normal_seq_error_log_likelihoods = torch.Tensor([item.normal_seq_error_log_likelihood for item in self._original_list])
        self.allele_frequencies = torch.Tensor([item.allele_frequency for item in self._original_list])
        self.artifact_logits = torch.Tensor([item.artifact_logit for item in self._original_list])

    def original_list(self) -> List[PosteriorDatum]:
        return self._original_list

    def size(self) -> int:
        return self._size

    def ref_counts(self) -> torch.IntTensor:
        return self.depths - self.alt_counts

    def normal_ref_counts(self) -> torch.IntTensor:
        return self.normal_depths - self.normal_alt_counts

    def variant_type_one_hot(self):
        return self._variant_type_one_hot

    def variant_type_mask(self, variant_type):
        return torch.BoolTensor([item.variant_type == variant_type for item in self._original_list])

