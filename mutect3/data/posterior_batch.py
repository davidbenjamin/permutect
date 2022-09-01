from typing import List

import torch
from mutect3.data.posterior_datum import PosteriorDatum


class PosteriorBatch:

    def __init__(self, data: List[PosteriorDatum]):
        self._original_list = data  # keep this for downsampling augmentation
        self._size = len(data)

        self._pd_tumor_depths = torch.IntTensor([item.tumor_depth() for item in data])
        self._pd_tumor_alt_counts = torch.IntTensor([item.tumor_alt_count() for item in data])

        self._variant_type_one_hot = torch.vstack([item.variant_type().one_hot_tensor() for item in self._original_list])

        self._seq_error_log_likelihood = torch.Tensor([item.seq_error_log_likelihood() for item in self._original_list])
        self._normal_seq_error_log_likelihood = torch.Tensor([item.normal_seq_error_log_likelihood() for item in self._original_list])
        self._allele_frequencies = torch.Tensor([item.allele_frequency() for item in self._original_list])
        self._artifact_logits = torch.Tensor([item.artifact_logit() for item in self._original_list])

    def original_list(self) -> List[PosteriorDatum]:
        return self._original_list

    def size(self) -> int:
        return self._size

    def pd_tumor_depths(self) -> torch.IntTensor:
        return self._pd_tumor_depths

    def pd_tumor_alt_counts(self) -> torch.IntTensor:
        return self._pd_tumor_alt_counts

    def pd_tumor_ref_counts(self) -> torch.IntTensor:
        return self._pd_tumor_depths - self._pd_tumor_alt_counts

    def variant_type_one_hot(self):
        return self._variant_type_one_hot

    def variant_type_mask(self, variant_type):
        return torch.BoolTensor([item.variant_type() == variant_type for item in self._original_list])

    def seq_error_log_likelihoods(self) -> torch.Tensor:
        return self._seq_error_log_likelihood

    def normal_seq_error_log_likelihoods(self) -> torch.Tensor:
        return self._normal_seq_error_log_likelihood

    def allele_frequencies(self) -> torch.Tensor:
        return self._allele_frequencies

    def artifact_logits(self) -> torch.Tensor:
        return self._artifact_logits
