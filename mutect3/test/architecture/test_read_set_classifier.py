from mutect3.test.test_utils import artificial_data
from mutect3.data.read_set_dataset import ReadSetDataset, make_semisupervised_data_loader
from mutect3.data.read_set_datum import ReadSetDatum
from typing import Iterable
import torch
from matplotlib.backends.backend_pdf import PdfPages
from torch.distributions.beta import Beta
from mutect3.architecture.normal_artifact_model import NormalArtifactModel
from mutect3.architecture.read_set_classifier import ReadSetClassifier, Mutect3Parameters
from mutect3 import utils
from mutect3.tools.train_model import TrainingParameters

BATCH_SIZE=64
NUM_EPOCHS = 100
TRAINING_PARAMS = TrainingParameters(batch_size=BATCH_SIZE, num_epochs=NUM_EPOCHS, beta1=Beta(5, 1), beta2=Beta(5, 1))

SMALL_MODEL_PARAMS = Mutect3Parameters(hidden_read_layers=[5, 5], hidden_info_layers=[5, 5], aggregation_layers=[5, 5],
                                       output_layers=[5, 5], dropout_p=0.2)

def train_model_and_return_metrics(m3_params: Mutect3Parameters, training_params: TrainingParameters,
                                   data: Iterable[ReadSetDatum])
    dataset = ReadSetDataset(data=data)
    training, valid = utils.split_dataset_into_train_and_valid(dataset, 0.9)
    na_model = NormalArtifactModel([10, 10, 10])

    train_loader = make_semisupervised_data_loader(training, TRAINING_PARAMS.batch_size)
    valid_loader = make_semisupervised_data_loader(valid, TRAINING_PARAMS.batch_size)
    model = ReadSetClassifier(SMALL_MODEL_PARAMS, na_model).float()
    training_metrics = model.train_model(train_loader, valid_loader, TRAINING_PARAMS.num_epochs, TRAINING_PARAMS.beta1, TRAINING_PARAMS.beta2)
    calibration_metrics = model.learn_calibration(valid_loader, num_epochs=50)
    return model, training_metrics, calibration_metrics

def test_separate_gaussian_data():
    data = artificial_data.make_two_gaussian_data(10000)
    model, training_metrics, calibration_metrics = \
        train_model_and_return_metrics(m3_params=SMALL_MODEL_PARAMS, training_params=TRAINING_PARAMS, data=data)

    dataset = ReadSetDataset(data=data)
    training, valid = utils.split_dataset_into_train_and_valid(dataset, 0.9)

    na_model = NormalArtifactModel([10, 10, 10])

    train_loader = make_semisupervised_data_loader(training, params.batch_size)
    valid_loader = make_semisupervised_data_loader(valid, params.batch_size)
    model = ReadSetClassifier(m3_params, na_model).float()

    training_metrics = model.train_model(train_loader, valid_loader, params.num_epochs, params.beta1, params.beta2)
    calibration_metrics = model.learn_calibration(valid_loader, num_epochs=50)

    assert training_metrics.metrics.get("TRAIN variant accuracy")[params.num_epochs - 1] > 0.99
    assert training_metrics.metrics.get("TRAIN artifact accuracy")[params.num_epochs - 1] > 0.99
    assert training_metrics.metrics.get("VALID variant accuracy")[params.num_epochs-1] > 0.99
    assert training_metrics.metrics.get("VALID artifact accuracy")[params.num_epochs - 1] > 0.99


def test_wide_and_narrow_gaussian_data():

    m3_params = Mutect3Parameters(hidden_read_layers=[5,5], hidden_info_layers=[5,5], aggregation_layers=[5,5],
                                  output_layers=[5,5], dropout_p=0.2)

    beta1 = Beta(5, 1)
    beta2 = Beta(5, 1)
    params = TrainingParameters(batch_size=BATCH_SIZE, num_epochs=NUM_EPOCHS, beta1=beta1, beta2=beta2)

    data = artificial_data.make_wide_and_narrow_gaussian_data(10000)
    dataset = ReadSetDataset(data=data)
    training, valid = utils.split_dataset_into_train_and_valid(dataset, 0.9)

    na_model = NormalArtifactModel([10, 10, 10])

    train_loader = make_semisupervised_data_loader(training, params.batch_size)
    valid_loader = make_semisupervised_data_loader(valid, params.batch_size)
    model = ReadSetClassifier(m3_params, na_model).float()

    training_metrics = model.train_model(train_loader, valid_loader, params.num_epochs, params.beta1, params.beta2)
    calibration_metrics = model.learn_calibration(valid_loader, num_epochs=50)

    assert training_metrics.metrics.get("TRAIN variant accuracy")[params.num_epochs - 1] > 0.93
    assert training_metrics.metrics.get("TRAIN artifact accuracy")[params.num_epochs - 1] > 0.93
    assert training_metrics.metrics.get("VALID variant accuracy")[params.num_epochs-1] > 0.93
    assert training_metrics.metrics.get("VALID artifact accuracy")[params.num_epochs - 1] > 0.93
