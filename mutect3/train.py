from mutect3 import validation, networks, data, normal_artifact
from torch.distributions.beta import Beta
import torch

class TrainingParameters:
    def __init__(self, batch_size, num_epochs, beta1: Beta, beta2: Beta):
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.beta1 = beta1
        self.beta2 = beta2

def run_evaluation(training_pickles, test_pickle, normal_artifact_pickle, params: TrainingParameters, m3_params: networks.Mutect3Parameters):
    #TODO: encapsulate this stuff
    print("Loading normal artifact dataset from pickled files")
    train_and_valid = normal_artifact.NormalArtifactDataset(normal_artifact_pickle)
    train_len = int(0.9 * len(train_and_valid))
    valid_len = len(train_and_valid) - train_len
    na_train, na_valid = torch.utils.data.random_split(train_and_valid, lengths=[train_len, valid_len])

    print("Training normal artifact model")
    na_batch_size = 64
    na_train_loader = normal_artifact.make_normal_artifact_data_loader(na_train, na_batch_size)
    na_valid_loader = normal_artifact.make_normal_artifact_data_loader(na_valid, na_batch_size)
    na_model = normal_artifact.NormalArtifactModel([10, 10, 10])
    na_training_metrics = na_model.train_model(na_train_loader, na_valid_loader, num_epochs=10)
    na_training_metrics.plot_all_metrics()

    print("Loading datasets from pickled files")
    train, valid, test = data.make_datasets(training_pickles, test_pickle)

    train_false_artifacts = sum(
        [1 for datum in train if 'PASS' in datum.mutect_info().filters() and datum.artifact_label() == 1])
    print("Training data includes " + str(train_false_artifacts) + " PASS variants labelled as artifacts.")

    train_loader, valid_loader, test_loader = data.make_data_loaders(train, valid, test, params.batch_size)
    model = networks.ReadSetClassifier(m3_params, na_model).float()
    training_metrics = model.train_model(train_loader, valid_loader, test_loader, params.num_epochs, params.beta1, params.beta2)
    training_metrics.plot_all_metrics()
    model.get_prior_model().plot_spectra()

    # plot purported ROC curve
    validation.get_optimal_f_score(model, test_loader, make_plot=True, normal_artifact=True)

    logit_threshold = model.calculate_logit_threshold(test_loader)
    print("Optimal logit threshold: " + str(logit_threshold))

    validation.show_validation_plots(model, test_loader, logit_threshold)

    return model

