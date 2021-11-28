from torch.distributions.beta import Beta
from mutect3 import validation, networks, data, normal_artifact


class TrainingParameters:
    def __init__(self, batch_size, num_epochs, beta1: Beta, beta2: Beta):
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.beta1 = beta1
        self.beta2 = beta2


def run_evaluation(training_pickles, test_pickle, normal_artifact_pickles, params: TrainingParameters, m3_params: networks.Mutect3Parameters):
    model = make_trained_mutect3_model(m3_params, normal_artifact_pickles, params, training_pickles)

    test = data.Mutect3Dataset([test_pickle])
    test_loader = data.make_test_data_loader(test, params.batch_size)
    model.learn_spectra(test_loader, num_epochs=200)
    model.get_prior_model().plot_spectra()
    validation.plot_roc_curve(model, test_loader, normal_artifact=True)

    logit_threshold = model.calculate_logit_threshold(test_loader)
    print("Optimal logit threshold: " + str(logit_threshold))

    validation.show_validation_plots(model, test_loader, logit_threshold)

    return model


# note: this does not include learning the AF spectra, which is part of the posterior model
# that is particular to one specific callset or test situation
def make_trained_mutect3_model(m3_params, normal_artifact_pickles, params, training_pickles):
    na_dataset = normal_artifact.NormalArtifactDataset(normal_artifact_pickles)
    na_train, na_valid = data.split_dataset_into_train_and_valid(na_dataset, 0.9)

    print("Training normal artifact model")
    na_batch_size = 64
    na_train_loader = normal_artifact.make_normal_artifact_data_loader(na_train, na_batch_size)
    na_valid_loader = normal_artifact.make_normal_artifact_data_loader(na_valid, na_batch_size)
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

