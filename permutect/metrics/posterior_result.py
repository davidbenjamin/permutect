class PosteriorResult:
    """
    simple container class for holding results of the posterior model and other things that get output to the VCF and
    tensorboard analysis
    """
    def __init__(self, artifact_logit: float, posterior_probabilities, log_priors, spectra_lls, normal_lls, label, alt_count, depth, var_type, embedding):
        self.artifact_logit = artifact_logit
        self.posterior_probabilities = posterior_probabilities
        self.log_priors = log_priors
        self.spectra_lls = spectra_lls
        self.normal_lls = normal_lls
        self.label = label
        self.alt_count = alt_count
        self.depth = depth
        self.variant_type = var_type
        self.embedding = embedding