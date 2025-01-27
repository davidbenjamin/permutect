"""
Utilities for saving and loading base + artifact model

Note that they are always saved and loaded together, which perhaps suggests more unifying refactoring is in order. . .
"""
import torch

from permutect import utils, constants
from permutect.architecture.artifact_model import ArtifactModel
from permutect.architecture.base_model import BaseModel

BASE_PREFIX = "base"
ARTIFACT_PREFIX = "artifact"


# save a base model AND artifact model (optionally with artifact log priors and spectra) together
def save(path, base_model: BaseModel, artifact_model: ArtifactModel, artifact_log_priors=None, artifact_spectra=None):
    base_dict = base_model.make_dict_for_saving(prefix=BASE_PREFIX)
    artifact_dict = artifact_model.make_dict_for_saving(artifact_log_priors, artifact_spectra, prefix=ARTIFACT_PREFIX)
    torch.save({**artifact_dict, **base_dict}, path)


def _base_model_from_saved_dict(saved, prefix: str = "", device: torch.device = utils.gpu_if_available()):
    hyperparams = saved[prefix + constants.HYPERPARAMS_NAME]
    num_read_features = saved[prefix + constants.NUM_READ_FEATURES_NAME]
    num_info_features = saved[prefix + constants.NUM_INFO_FEATURES_NAME]
    ref_sequence_length = saved[prefix + constants.REF_SEQUENCE_LENGTH_NAME]

    model = BaseModel(hyperparams, num_read_features=num_read_features, num_info_features=num_info_features,
                      ref_sequence_length=ref_sequence_length, device=device)
    model.load_state_dict(saved[prefix + constants.STATE_DICT_NAME])

    # in case the state dict had the wrong dtype for the device we're on now eg base model was pretrained on GPU
    # and we're now on CPU
    model.to(model._dtype)

    return model


def _artifact_model_from_saved_dict(saved, prefix: str = "artifact"):
    model_params = saved[prefix + constants.HYPERPARAMS_NAME]
    num_base_features = saved[prefix + constants.NUM_BASE_FEATURES_NAME]
    model = ArtifactModel(model_params, num_base_features)
    model.load_state_dict(saved[prefix + constants.STATE_DICT_NAME])

    artifact_log_priors = saved[prefix + constants.ARTIFACT_LOG_PRIORS_NAME]  # possibly None
    artifact_spectra_state_dict = saved[prefix + constants.ARTIFACT_SPECTRA_STATE_DICT_NAME]  # possibly None
    return model, artifact_log_priors, artifact_spectra_state_dict


def load_models(path, device):
    saved = torch.load(path, map_location=device)
    base_model = _base_model_from_saved_dict(saved, prefix=BASE_PREFIX)
    artifact_model, artifact_log_priors, artifact_spectra = _artifact_model_from_saved_dict(saved, prefix=ARTIFACT_PREFIX)
    return base_model, artifact_model, artifact_log_priors, artifact_spectra