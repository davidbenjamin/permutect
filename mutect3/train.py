from mutect3 import validation, networks, data
from torch.distributions.beta import Beta

class TrainingParameters:
    def __init__(self, batch_size, num_epochs, beta1: Beta, beta2: Beta, hidden_read_layers, \
                 hidden_info_layers, aggregation_layers, output_layers, dropout_p):
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.beta1 = beta1
        self.beta2 = beta2
        self.hidden_read_layers = hidden_read_layers
        self.hidden_info_layers = hidden_info_layers
        self.aggregation_layers = aggregation_layers
        self.output_layers = output_layers
        self.dropout_p = dropout_p

def run_evaluation(training_pickles, test_pickle, params: TrainingParameters):
    # Load data
    print("Loading datasets from pickled files")
    train, valid, test = data.make_datasets(training_pickles, test_pickle)

    train_false_artifacts = sum(
        [1 for datum in train if 'PASS' in datum.mutect_info().filters() and datum.artifact_label() == 1])
    print("Training data includes " + str(train_false_artifacts) + " PASS variants labelled as artifacts.")

    train_loader, valid_loader, test_loader = data.make_data_loaders(train, valid, test, params.batch_size, params.beta1, params.beta2)
    model = networks.ReadSetClassifier(hidden_read_layers=params.hidden_read_layers, hidden_info_layers=params.hidden_info_layers,
                                       aggregation_layers=params.aggregation_layers, output_layers=params.output_layers,
                                       m2_filters_to_keep={'normal_artifact'}, dropout_p=params.dropout_p).float()
    training_metrics = model.train_model(train_loader, valid_loader, test_loader, params.num_epochs)
    training_metrics.plot_all_metrics()
    model.get_prior_model().plot_spectra()

    # plot purported ROC curve
    validation.get_optimal_f_score(model, test_loader, make_plot=True)

    logit_threshold = model.calculate_logit_threshold(test_loader)
    print("Optimal logit threshold: " + str(logit_threshold))

    validation.show_validation_plots(model, test_loader, logit_threshold)

    return model

