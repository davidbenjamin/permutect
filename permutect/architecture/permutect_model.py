import torch

from permutect import constants
from permutect.architecture.artifact_model import ArtifactModel
from permutect.architecture.posterior_model import PosteriorModel
from permutect.data.datum import DEFAULT_CPU_FLOAT
from permutect.data.datum import DEFAULT_GPU_FLOAT
from permutect.misc_utils import gpu_if_available


class PermutectModel(torch.nn.Module):
    def __init__(self, artifact_model: ArtifactModel, posterior_model: PosteriorModel, device=None):
        super(PermutectModel, self).__init__()

        if device is None:
            device = gpu_if_available()

        self._device = device
        self._dtype = DEFAULT_GPU_FLOAT if device != torch.device("cpu") else DEFAULT_CPU_FLOAT

        self.artifact_model = artifact_model
        self.posterior_model  = posterior_model
        self.artifact_model.to(device=self._device, dtype=self._dtype)
        self.posterior_model.to(device=self._device, dtype=self._dtype)

    def save_model(self, path):
        artifact_model = self.artifact_model
        posterior_model = self.posterior_model
        saved_dict = {
                constants.STATE_DICT_NAME: artifact_model.state_dict(),
                constants.HYPERPARAMS_NAME: artifact_model._params,
                constants.NUM_READ_FEATURES_NAME: artifact_model.read_embedding.input_dimension(),
                constants.NUM_INFO_FEATURES_NAME: artifact_model.info_embedding.input_dimension(),
                constants.REF_SEQUENCE_LENGTH_NAME: artifact_model.haplotypes_length(),
                constants.POSTERIOR_STATE_DICT_NAME: posterior_model.state_dict(),
                constants.POSTERIOR_PARAMS_NAME: posterior_model._params,
            }

        torch.save(saved_dict, path)

def load_model(path, device: torch.device = None) -> PermutectModel:
    if device is None:
        device = gpu_if_available()
    saved = torch.load(path, map_location=device, weights_only=False)

    artifact_model = ArtifactModel(
        params=saved[constants.HYPERPARAMS_NAME],
        num_read_features=saved[constants.NUM_READ_FEATURES_NAME],
        num_info_features=saved[constants.NUM_INFO_FEATURES_NAME],
        haplotypes_length=saved[constants.REF_SEQUENCE_LENGTH_NAME],
        device=device,
    )
    artifact_model.load_state_dict(saved[constants.STATE_DICT_NAME])

    posterior_model = PosteriorModel(posterior_params=saved[constants.POSTERIOR_PARAMS_NAME], device=device)
    posterior_model.load_state_dict(saved[constants.POSTERIOR_STATE_DICT_NAME])

    model = PermutectModel(artifact_model, posterior_model, device)

    return model
