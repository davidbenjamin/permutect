from mutect3.test.test_utils import artificial_data
from mutect3.data.read_set_dataset import ReadSetDataset, make_semisupervised_data_loader, make_test_data_loader
from mutect3.data.read_set_datum import ReadSetDatum
from typing import Iterable
from torch.distributions.beta import Beta
from mutect3.architecture.normal_artifact_model import NormalArtifactModel
from mutect3.architecture.read_set_classifier import ReadSetClassifier, Mutect3Parameters
from mutect3 import utils
from mutect3.tools.train_model import TrainingParameters

BATCH_SIZE = 64
NUM_EPOCHS = 100
NUM_SPECTRUM_EPOCHS=200
TRAINING_PARAMS = TrainingParameters(batch_size=BATCH_SIZE, num_epochs=NUM_EPOCHS, beta1=Beta(5, 1), beta2=Beta(5, 1))

SMALL_MODEL_PARAMS = Mutect3Parameters(hidden_read_layers=[5, 5], hidden_info_layers=[5, 5], aggregation_layers=[5, 5],
                                       output_layers=[5, 5], dropout_p=0.2)


# Note that the test methods in this class also cover batching, samplers, datasets, and data loaders
def train_model_and_return_metrics(m3_params: Mutect3Parameters, training_params: TrainingParameters,
                                   data: Iterable[ReadSetDatum]):
    dataset = ReadSetDataset(data=data)
    training, valid = utils.split_dataset_into_train_and_valid(dataset, 0.9)
    na_model = NormalArtifactModel([10, 10, 10])

    train_loader = make_semisupervised_data_loader(training, TRAINING_PARAMS.batch_size)
    valid_loader = make_semisupervised_data_loader(valid, TRAINING_PARAMS.batch_size)
    model = ReadSetClassifier(m3_params=m3_params, na_model=na_model).float()
    training_metrics = model.train_model(train_loader, valid_loader, TRAINING_PARAMS.num_epochs, TRAINING_PARAMS.beta1, TRAINING_PARAMS.beta2)
    calibration_metrics = model.learn_calibration(valid_loader, num_epochs=50)
    return model, training_metrics, calibration_metrics


def test_separate_gaussian_data():
    data = artificial_data.make_two_gaussian_data(10000)
    params = SMALL_MODEL_PARAMS
    training_params = TRAINING_PARAMS
    model, training_metrics, calibration_metrics = \
        train_model_and_return_metrics(m3_params=params, training_params=training_params, data=data)

    assert training_metrics.metrics.get("TRAIN variant accuracy")[training_params.num_epochs - 1] > 0.99
    assert training_metrics.metrics.get("TRAIN artifact accuracy")[training_params.num_epochs - 1] > 0.99
    assert training_metrics.metrics.get("VALID variant accuracy")[training_params.num_epochs-1] > 0.99
    assert training_metrics.metrics.get("VALID artifact accuracy")[training_params.num_epochs - 1] > 0.99

    test_data = artificial_data.make_two_gaussian_data(1000, is_training_data=False, vaf=0.5, unlabeled_fraction=0.0)
    test_dataset = ReadSetDataset(data=test_data)
    test_loader = make_test_data_loader(test_dataset, BATCH_SIZE)
    model.learn_spectra(test_loader, NUM_SPECTRUM_EPOCHS)

    j = 90


def test_wide_and_narrow_gaussian_data():
    data = artificial_data.make_wide_and_narrow_gaussian_data(1000)
    params = SMALL_MODEL_PARAMS
    training_params = TRAINING_PARAMS
    model, training_metrics, calibration_metrics = \
        train_model_and_return_metrics(m3_params=params, training_params=training_params, data=data)

    assert training_metrics.metrics.get("TRAIN variant accuracy")[training_params.num_epochs - 1] > 0.90
    assert training_metrics.metrics.get("TRAIN artifact accuracy")[training_params.num_epochs - 1] > 0.90
    assert training_metrics.metrics.get("VALID variant accuracy")[training_params.num_epochs-1] > 0.90
    assert training_metrics.metrics.get("VALID artifact accuracy")[training_params.num_epochs - 1] > 0.90

    test_data = artificial_data.make_two_gaussian_data(1000, is_training_data=False, vaf=0.25, unlabeled_fraction=0.0)
    test_dataset = ReadSetDataset(data=test_data)
    test_loader = make_test_data_loader(test_dataset, BATCH_SIZE)
    model.learn_spectra(test_loader, NUM_SPECTRUM_EPOCHS)
    p = 90
