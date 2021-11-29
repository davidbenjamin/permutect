import torch
from torch.distributions.beta import Beta
from mutect3 import validation, networks, data, normal_artifact


class TrainingParameters:
    def __init__(self, batch_size, num_epochs, beta1: Beta, beta2: Beta):
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.beta1 = beta1
        self.beta2 = beta2


def run_evaluation(model: networks.ReadSetClassifier, test_pickle, params: TrainingParameters):
    test = data.Mutect3Dataset([test_pickle])
    test_loader = data.make_test_data_loader(test, params.batch_size)
    model.learn_spectra(test_loader, num_epochs=200)
    model.get_prior_model().plot_spectra()
    validation.plot_roc_curve(model, test_loader, normal_artifact=True)

    logit_threshold = model.calculate_logit_threshold(test_loader)
    print("Optimal logit threshold: " + str(logit_threshold))

    validation.show_validation_plots(model, test_loader, logit_threshold)


# note: this does not include learning the AF spectra, which is part of the posterior model
# that is particular to one specific callset or test situation
def make_trained_mutect3_model(m3_params, training_pickles, normal_artifact_pickles, params):
    na_dataset = normal_artifact.NormalArtifactDataset(normal_artifact_pickles)
    na_train, na_valid = data.split_dataset_into_train_and_valid(na_dataset, 0.9)

    print("Training normal artifact model")
    na_batch_size = 64
    na_train_loader = normal_artifact.make_normal_artifact_data_loader(na_train, na_batch_size)
    na_valid_loader = normal_artifact.make_normal_artifact_data_loader(na_valid, na_batch_size)

    #TODO: should have NA params class
    na_model = normal_artifact.NormalArtifactModel([10, 10, 10])
    na_training_metrics = na_model.train_model(na_train_loader, na_valid_loader, num_epochs=10)
    na_training_metrics.plot_all_metrics()

    train, valid = data.make_training_and_validation_datasets(training_pickles)
    train_loader = data.make_semisupervised_data_loader(train, params.batch_size)
    valid_loader = data.make_semisupervised_data_loader(valid, params.batch_size)
    model = networks.ReadSetClassifier(m3_params, na_model).float()
    training_metrics = model.train_model(train_loader, valid_loader, params.num_epochs, params.beta1, params.beta2)
    training_metrics.plot_all_metrics()
    return model


def save_mutect3_model(model, m3_params, path):
    #TODO: introduce constants
    torch.save({
        'model_state_dict': model.state_dict(),
        'm3_params': m3_params
    }, path)


# this presumes that we have a ReadSetClassifier model and we have saved it via save_mutect3_model
def load_saved_model(path):
    saved = torch.load(path)
    m3_params = saved['m3_params']
    #TODO: this should not be hard-coded.  See above above introducing na_params
    na_model = normal_artifact.NormalArtifactModel([10, 10, 10])
    model = networks.ReadSetClassifier(m3_params, na_model)
    model.load_state_dict(saved['model_state_dict'])
    return model

