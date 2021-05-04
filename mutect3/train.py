from mutect3 import validation, networks, data
from torch.distributions.beta import Beta

class TrainingParameters:
    def __init__(self, batch_size, num_epochs, beta1: Beta, beta2: Beta):
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.beta1 = beta1
        self.beta2 = beta2

def run_evaluation(training_pickles, test_pickle, params: TrainingParameters, m3_params: networks.Mutect3Parameters):
    # Load data
    print("Loading datasets from pickled files")
    train, valid, test = data.make_datasets(training_pickles, test_pickle)

    train_false_artifacts = sum(
        [1 for datum in train if 'PASS' in datum.mutect_info().filters() and datum.artifact_label() == 1])
    print("Training data includes " + str(train_false_artifacts) + " PASS variants labelled as artifacts.")

    train_loader, valid_loader, test_loader = data.make_data_loaders(train, valid, test, params.batch_size)
    model = networks.ReadSetClassifier(m3_params, m2_filters_to_keep={'normal_artifact'}).float()
    training_metrics = model.train_model(train_loader, valid_loader, test_loader, params.num_epochs, params.beta1, params.beta2)
    training_metrics.plot_all_metrics()
    model.get_prior_model().plot_spectra()

    # plot purported ROC curve
    validation.get_optimal_f_score(model, test_loader, make_plot=True)

    logit_threshold = model.calculate_logit_threshold(test_loader)
    print("Optimal logit threshold: " + str(logit_threshold))

    validation.show_validation_plots(model, test_loader, logit_threshold)

    return model

