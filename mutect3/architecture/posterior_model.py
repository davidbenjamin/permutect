import torch

from mutect3 import utils
from mutect3.architecture.beta_binomial_mixture import BetaBinomialMixture
from mutect3.data.read_set_batch import ReadSetBatch
from mutect3.architecture.artifact_model import ArtifactModel


class PosteriorModel(torch.nn.Module):
    """

    """

    def __init__(self, artifact_model: ArtifactModel):
        super(PosteriorModel, self).__init__()

        self._artifact_model = artifact_model