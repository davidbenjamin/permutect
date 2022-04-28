from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import tempfile
from torch.utils.tensorboard import SummaryWriter
from numpy.random import binomial
from mutect3.data.normal_artifact_datum import NormalArtifactDatum
import random

import mutect3.architecture.normal_artifact_model
import mutect3.architecture.read_set_classifier
from mutect3 import utils
from mutect3.data import normal_artifact_dataset


# make fake data with generative model:
# normal and tumor depth are always 100
# 1 in 5 chance of normal artifact, in which case
#   i) normal alt count is binomial(normal_depth, 0.1)
#   ii) tumor depth is binomial(tumor depth, 0.1) with some probability and 0 otherwise
# (ie normal alts don't always imply tumor artifact)
# 4 in 5 chance of no artifact, in which case alt counts are 0
def test_normal_artifact():
    depth = 100
    data = []
    size = 100000
    artifact_fraction = 0.3
    # artifact_af = 0.15
    normal_imply_tumor_prob = 0.5
    batch_size = 64
    num_epochs = 100
    hidden_layers = [5, 5, 5]
    for _ in range(size):
        artifact = random.uniform(0, 1) < artifact_fraction
        if artifact:
            artifact_af = random.uniform(0.1, 0.5)
            normal_alt_count = binomial(depth, artifact_af)
            tumor_artifact = random.uniform(0, 1) < normal_imply_tumor_prob
            tumor_alt_count = binomial(depth, artifact_af) if tumor_artifact else 0
            datum = NormalArtifactDatum(normal_alt_count=normal_alt_count, normal_depth=depth, tumor_alt_count=tumor_alt_count,
                                        tumor_depth=depth, downsampling=1, variant_type="SNV")
        else:
            datum = NormalArtifactDatum(normal_alt_count=0, normal_depth=depth, tumor_alt_count=0, tumor_depth=depth,
                                        downsampling=1, variant_type="SNV")
        data.append(datum)

    na_dataset = normal_artifact_dataset.NormalArtifactDataset(data=data)
    na_train, na_valid = utils.split_dataset_into_train_and_valid(na_dataset, 0.9)

    na_train_loader = normal_artifact_dataset.make_normal_artifact_data_loader(na_train, batch_size)
    na_valid_loader = normal_artifact_dataset.make_normal_artifact_data_loader(na_valid, batch_size)

    na_model = mutect3.architecture.normal_artifact_model.NormalArtifactModel(hidden_layers=hidden_layers)

    with tempfile.TemporaryDirectory() as tensorboard_dir:
        summary_writer = SummaryWriter(tensorboard_dir)
        na_model.train_model(na_train_loader, na_valid_loader, num_epochs=num_epochs, summary_writer=summary_writer)
        [na_model.plot_spectrum(af, "af = " + str(af) + " plot") for af in [0.0, 0.05, 0.1, 0.15, 0.2, 0.5]]

        j = 90
