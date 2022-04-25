import argparse
import torch
from matplotlib.backends.backend_pdf import PdfPages
import mutect3.architecture.normal_artifact_model
import mutect3.architecture.read_set_classifier
from mutect3 import utils, constants
from mutect3.data import normal_artifact_dataset


def train_na_model(normal_artifact_datasets, num_epochs, hidden_layers, batch_size, tensorboard_dir):
    na_dataset = normal_artifact_dataset.NormalArtifactDataset(normal_artifact_datasets)
    na_train, na_valid = utils.split_dataset_into_train_and_valid(na_dataset, 0.9)

    print("Training normal artifact model")
    na_train_loader = normal_artifact_dataset.make_normal_artifact_data_loader(na_train, batch_size)
    na_valid_loader = normal_artifact_dataset.make_normal_artifact_data_loader(na_valid, batch_size)

    na_model = mutect3.architecture.normal_artifact_model.NormalArtifactModel(hidden_layers=hidden_layers)
    na_training_metrics = na_model.train_model(na_train_loader, na_valid_loader, num_epochs=num_epochs)

    # TODO: FIX
        # with PdfPages(report_pdf) as pdf:
        #    for fig, curve in na_training_metrics.plot_curves():
        #        pdf.savefig(fig)

        #    for normal_af in [0.0, 0.03, 0.06, 0.1, 0.15, 0.25, 0.5, 0.75]:
        #        fig, curve = na_model.plot_spectrum(
        #            normal_af, "NA modeled tumor AF given normal AF = " + str(normal_af))
        #        pdf.savefig(fig)
    # TODO: end of TODO
    return na_model


def save_na_model(na_model, hidden_layers, path):
    torch.save({
        constants.STATE_DICT_NAME: na_model.state_dict(),
        constants.HIDDEN_LAYERS_NAME: hidden_layers
    }, path)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--' + constants.HIDDEN_LAYERS_NAME, nargs='+', type=int, required=True)
    parser.add_argument('--' + constants.NORMAL_ARTIFACT_DATASETS_NAME, nargs='+', type=str, required=True)
    parser.add_argument('--' + constants.BATCH_SIZE_NAME, type=int, default=64, required=False)
    parser.add_argument('--' + constants.NUM_EPOCHS_NAME, type=int, required=True)
    parser.add_argument('--' + constants.OUTPUT_NAME, required=True)
    parser.add_argument('--' + constants.TENSORBOARD_DIR_NAME, type=str, default='tensorboard', required=False)

    return parser.parse_args()


def main():
    args = parse_arguments()
    model = train_na_model(getattr(args, constants.NORMAL_ARTIFACT_DATASETS_NAME), getattr(args, constants.NUM_EPOCHS_NAME),
                           getattr(args, constants.HIDDEN_LAYERS_NAME), getattr(args, constants.BATCH_SIZE_NAME), getattr(args, constants.TENSORBOARD_DIR_NAME))
    save_na_model(model, getattr(args, constants.HIDDEN_LAYERS_NAME), getattr(args, constants.OUTPUT_NAME))


if __name__ == '__main__':
    main()
