from mutect3 import utils


class PosteriorDatum:
    # info tensor comes from GATK and does not include one-hot encoding of variant type
    def __init__(self, contig: str, position: int, ref: str, alt: str,
                 tumor_depth: int, tumor_alt_count: int, normal_depth: int, normal_alt_count: int,
                 seq_error_log_likelihood: float, normal_seq_error_log_likelihood: float, allele_frequency: float =  None,
                 artifact_logit: float = None):
        self._contig = contig
        self._position = position
        self._ref = ref
        self._alt = alt
        self._variant_type = utils.VariantType.get_type(ref, alt)

        self._tumor_depth = tumor_depth
        self._tumor_alt_count = tumor_alt_count
        self._normal_depth = normal_depth
        self._normal_alt_count = normal_alt_count

        # this is used only on filtering datasets for fitting the AF spectra.
        self._seq_error_log_likelihood = seq_error_log_likelihood
        self._normal_seq_error_log_likelihood = normal_seq_error_log_likelihood

        self._allele_frequency = allele_frequency
        self._artifact_logit = artifact_logit

    def contig(self) -> str:
        return self._contig

    def position(self) -> int:
        return self._position

    def ref(self) -> str:
        return self._ref

    def alt(self) -> str:
        return self._alt

    def variant_type(self) -> utils.VariantType:
        return self._variant_type

    def tumor_depth(self) -> int:
        return self._tumor_depth

    def tumor_alt_count(self) -> int:
        return self._tumor_alt_count

    def normal_depth(self) -> int:
        return self._normal_depth

    def normal_alt_count(self) -> int:
        return self._normal_alt_count

    def seq_error_log_likelihood(self):
        return self._seq_error_log_likelihood

    def normal_seq_error_log_likelihood(self):
        return self._normal_seq_error_log_likelihood

    def set_allele_frequency(self, af: float):
        self._allele_frequency = af

    def allele_frequency(self) -> float:
        return self._allele_frequency

    def set_artifact_logit(self, logit: float):
        self._artifact_logit = logit

    def artifact_logit(self) -> float:
        return self._artifact_logit
