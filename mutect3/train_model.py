import argparse

import torch
from matplotlib.backends.backend_pdf import PdfPages
from torch.distributions.beta import Beta

from mutect3 import data, networks, utils


class TrainingParameters:
    def __init__(self, batch_size, num_epochs, beta1: Beta, beta2: Beta):
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.beta1 = beta1
        self.beta2 = beta2


def make_trained_mutect3_model(m3_params: networks.Mutect3Parameters, training_datasets, normal_artifact_tables,
                               params: TrainingParameters, report_pdf=None):
    na_dataset = data.NormalArtifactDataset(normal_artifact_tables)
    na_train, na_valid = utils.split_dataset_into_train_and_valid(na_dataset, 0.9)

    print("Training normal artifact model")
    na_batch_size = 64
    na_train_loader = data.make_normal_artifact_data_loader(na_train, na_batch_size)
    na_valid_loader = data.make_normal_artifact_data_loader(na_valid, na_batch_size)

    # TODO: should have NA params class
    na_model = networks.NormalArtifactModel([10, 10, 10])
    na_training_metrics = na_model.train_model(na_train_loader, na_valid_loader, num_epochs=10)

    print("Loading datasets")

    # make our training, validation, and testing data
    train_and_valid = data.Mutect3Dataset(training_datasets)
    training, valid = utils.split_dataset_into_train_and_valid(train_and_valid, 0.9)

    unlabeled_count = sum([1 for datum in train_and_valid if datum.label() == "UNLABELED"])
    print("Unlabeled data: " + str(unlabeled_count) + ", labeled data: " + str(len(train_and_valid) - unlabeled_count))
    print("Dataset sizes -- training: " + str(len(training)) + ", validation: " + str(len(valid)))

    train_loader = data.make_semisupervised_data_loader(training, params.batch_size)
    valid_loader = data.make_semisupervised_data_loader(valid, params.batch_size)
    model = networks.ReadSetClassifier(m3_params, na_model).float()

    print("Training model")
    training_metrics = model.train_model(train_loader, valid_loader, params.num_epochs, params.beta1, params.beta2)
    if report_pdf is not None:
        with PdfPages(report_pdf) as pdf:
            for metrics in (training_metrics, na_training_metrics):
                for metric_type in metrics.metrics.keys():
                    fig, curve = metrics.plot_metrics(metric_type)
                    pdf.savefig(fig)

    return model


def save_mutect3_model(model, m3_params, path):
    # TODO: introduce constants
    torch.save({
        'model_state_dict': model.state_dict(),
        'm3_params': m3_params
    }, path)


def main():
    parser = argparse.ArgumentParser()

    # architecture hyperparameters
    parser.add_argument('--hidden-read-layers', nargs='+', type=int, required=True)
    parser.add_argument('--hidden-info-layers', nargs='+', type=int, required=True)
    parser.add_argument('--aggregation-layers', nargs='+', type=int, required=True)
    parser.add_argument('--output-layers', nargs='+', type=int, required=True)
    parser.add_argument('--dropout-p', type=float, default=0.0, required=False)

    # Training data inputs
    parser.add_argument('--training-datasets', nargs='+', type=str, required=True)
    parser.add_argument('--normal-artifact-tables', nargs='+', type=str, required=True)

    # training hyperparameters
    parser.add_argument('--alpha1', type=float, default=5.0, required=False)
    parser.add_argument('--beta1', type=float, default=1.0, required=False)
    parser.add_argument('--alpha2', type=float, default=5.0, required=False)
    parser.add_argument('--beta2', type=float, default=1.0, required=False)
    parser.add_argument('--batch_size', type=int, default=64, required=False)
    parser.add_argument('--num_epochs', type=int, required=True)

    # path to saved model
    parser.add_argument('--output', required=True)
    parser.add_argument('--report_pdf', required=False)

    args = parser.parse_args()

    m3_params = networks.Mutect3Parameters(args.hidden_read_layers, args.hidden_info_layers,
                                           args.aggregation_layers, args.output_layers, args.dropout_p)

    beta1 = Beta(args.alpha1, args.beta1)
    beta2 = Beta(args.alpha2, args.beta2)
    params = TrainingParameters(args.batch_size, args.num_epochs, beta1, beta2)

    model = make_trained_mutect3_model(m3_params, args.training_datasets, args.normal_artifact_tables, params,
                                       args.report_pdf)

    save_mutect3_model(model, m3_params, args.output)


if __name__ == '__main__':
    main()
