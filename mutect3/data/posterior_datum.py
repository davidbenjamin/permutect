from mutect3 import utils


class PosteriorDatum:
    def __init__(self, contig: str, position: int, ref: str, alt: str,
                 depth: int, alt_count: int, normal_depth: int, normal_alt_count: int,
                 seq_error_log_likelihood: float, normal_seq_error_log_likelihood: float, allele_frequency: float = None,
                 artifact_logit: float = None):
        self.contig = contig
        self.position = position
        self.ref = ref
        self.alt = alt
        self.variant_type = utils.Variation.get_type(ref, alt)

        self.depth = depth
        self.alt_count = alt_count
        self.normal_depth = normal_depth
        self.normal_alt_count = normal_alt_count

        self.seq_error_log_likelihood = seq_error_log_likelihood
        self.normal_seq_error_log_likelihood = normal_seq_error_log_likelihood

        self.allele_frequency = allele_frequency
        self.artifact_logit = artifact_logit

    def set_allele_frequency(self, af: float):
        self.allele_frequency = af

    def set_artifact_logit(self, logit: float):
        self.artifact_logit = logit

