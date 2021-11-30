import torch
from torch.distributions.beta import Beta

import mutect3.networks
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


# this presumes that we have a ReadSetClassifier model and we have saved it via save_mutect3_model
def load_saved_model(path):
    saved = torch.load(path)
    m3_params = saved['m3_params']
    #TODO: this should not be hard-coded.  See above above introducing na_params
    na_model = mutect3.networks.NormalArtifactModel([10, 10, 10])
    model = networks.ReadSetClassifier(m3_params, na_model)
    model.load_state_dict(saved['model_state_dict'])
    return model

