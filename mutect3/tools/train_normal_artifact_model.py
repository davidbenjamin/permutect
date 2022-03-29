import argparse
import torch
from matplotlib.backends.backend_pdf import PdfPages
import mutect3.architecture.normal_artifact_model
import mutect3.architecture.read_set_classifier
from mutect3 import utils
from mutect3.data import normal_artifact_dataset


def make_trained_normal_artifact_model(normal_artifact_datasets, num_epochs, hidden_layers, batch_size, report_pdf=None):
    na_dataset = normal_artifact_dataset.NormalArtifactDataset(normal_artifact_datasets)
    na_train, na_valid = utils.split_dataset_into_train_and_valid(na_dataset, 0.9)

    print("Training normal artifact model")
    na_train_loader = normal_artifact_dataset.make_normal_artifact_data_loader(na_train, batch_size)
    na_valid_loader = normal_artifact_dataset.make_normal_artifact_data_loader(na_valid, batch_size)

    na_model = mutect3.architecture.normal_artifact_model.NormalArtifactModel(hidden_layers=hidden_layers)
    na_training_metrics = na_model.train_model(na_train_loader, na_valid_loader, num_epochs=num_epochs)

    if report_pdf is not None:
        with PdfPages(report_pdf) as pdf:
            for metric_type in na_training_metrics.metrics.keys():
                fig, curve = na_training_metrics.plot_curves(metric_type)
                pdf.savefig(fig)

            for normal_alt, normal_depth in [(0, 30), (1, 30), (2, 30), (3, 30), (4, 30), (5, 30), (10, 30), (15, 30)]:
                fig, curve = na_model.plot_spectrum(
                    normal_alt, normal_depth, "NA modeled tumor AF given normal alt count, normal depth = " +
                                              str(normal_alt) + ", " + str(normal_depth))
                pdf.savefig(fig)

    return na_model


def save_na_model(na_model, hidden_layers, path):
    # TODO: introduce constants
    torch.save({
        'model_state_dict': na_model.state_dict(),
        'hidden_layers': hidden_layers
    }, path)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--hidden-layers', nargs='+', type=int, required=True)
    parser.add_argument('--normal-artifact-datasets', nargs='+', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=64, required=False)
    parser.add_argument('--num_epochs', type=int, required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--report_pdf', required=False)

    args = parser.parse_args()
    model = make_trained_normal_artifact_model(args.normal_artifact_datasets, args.num_epochs,
                                               args.hidden_layers, args.batch_size, args.report_pdf)
    save_na_model(model, args.hidden_layers, args.output)


if __name__ == '__main__':
    main()
