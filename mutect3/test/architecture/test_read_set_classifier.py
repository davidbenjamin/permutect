from mutect3.test.test_utils import artificial_data
from mutect3.data.read_set_dataset import ReadSetDataset, make_semisupervised_data_loader
import torch
from matplotlib.backends.backend_pdf import PdfPages
from torch.distributions.beta import Beta
from mutect3.architecture.normal_artifact_model import NormalArtifactModel
from mutect3.architecture.read_set_classifier import ReadSetClassifier, Mutect3Parameters
from mutect3 import utils
from mutect3.tools.train_model import TrainingParameters

BATCH_SIZE=64
NUM_EPOCHS = 100


def test_separate_gaussian_data():

    m3_params = Mutect3Parameters(hidden_read_layers=[5,5], hidden_info_layers=[5,5], aggregation_layers=[5,5],
                                  output_layers=[5,5], dropout_p=0.2)

    beta1 = Beta(5, 1)
    beta2 = Beta(5, 1)
    params = TrainingParameters(batch_size=BATCH_SIZE, num_epochs=NUM_EPOCHS, beta1=beta1, beta2=beta2)

    data = artificial_data.make_two_gaussian_data(1000)
    dataset = ReadSetDataset(data=data)
    training, valid = utils.split_dataset_into_train_and_valid(dataset, 0.9)

    na_model = NormalArtifactModel([10, 10, 10])

    train_loader = make_semisupervised_data_loader(training, params.batch_size)
    valid_loader = make_semisupervised_data_loader(valid, params.batch_size)
    model = ReadSetClassifier(m3_params, na_model).float()

    training_metrics = model.train_model(train_loader, valid_loader, params.num_epochs, params.beta1, params.beta2)
    calibration_metrics = model.learn_calibration(valid_loader, num_epochs=50)
    h = 90