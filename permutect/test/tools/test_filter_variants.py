import tempfile
from argparse import Namespace

from permutect import constants
from permutect.test.test_file_names import *
from permutect.tools import filter_variants


def test_filtering_on_dream1_chr20():
    # NOTE: EXPERIMENTAL_MODEL does not currently exist in the repo, so this
    # test will fail with FileNotFoundError until a compatible model is committed.
    artifact_model = EXPERIMENTAL_MODEL

    permutect_vcf = tempfile.NamedTemporaryFile()
    tensorboard_dir = tempfile.TemporaryDirectory()

    filtering_args = Namespace()
    setattr(filtering_args, constants.INPUT_NAME, MUTECT2_CHR20_FILTERED_VCF)
    setattr(filtering_args, constants.TEST_DATASET_NAME, DREAM_1_CHR20_PLAIN_TEXT_DATA)
    setattr(filtering_args, constants.ARTIFACT_MODEL_NAME, artifact_model)
    setattr(filtering_args, constants.TRAINABLE_PARAMETERS_NAME, ["CLUSTERING"])
    setattr(filtering_args, constants.OUTPUT_NAME, permutect_vcf.name)
    setattr(filtering_args, constants.TENSORBOARD_DIR_NAME, tensorboard_dir.name)
    setattr(filtering_args, constants.NUM_SPECTRUM_ITERATIONS_NAME, 2)
    setattr(filtering_args, constants.RECALL_WEIGHT_NAME, 1.0)
    setattr(filtering_args, constants.HET_BETA_NAME, 10)
    setattr(filtering_args, constants.SPECTRUM_LEARNING_RATE_NAME, 0.001)
    setattr(filtering_args, constants.INITIAL_LOG_VARIANT_PRIOR_NAME, -10.0)
    setattr(filtering_args, constants.INITIAL_LOG_ARTIFACT_PRIOR_NAME, -10.0)
    setattr(filtering_args, constants.GENOMIC_SPAN_NAME, 60000000)
    setattr(filtering_args, constants.MAF_SEGMENTS_NAME, None)
    setattr(filtering_args, constants.CONTIGS_TABLE_NAME, CONTIGS_TABLE)
    setattr(filtering_args, constants.NORMAL_MAF_SEGMENTS_NAME, None)
    setattr(filtering_args, constants.GERMLINE_MODE_NAME, False)
    setattr(filtering_args, constants.NO_GERMLINE_MODE_NAME, False)

    # training hyperparameters
    setattr(filtering_args, constants.BATCH_SIZE_NAME, 64)
    setattr(filtering_args, constants.NUM_WORKERS_NAME, 2)
    setattr(filtering_args, constants.NUM_EPOCHS_NAME, 2)
    setattr(filtering_args, constants.LEARNING_RATE_NAME, 0.001)
    setattr(filtering_args, constants.WEIGHT_DECAY_NAME, 0.01)

    filter_variants.main_without_parsing(filtering_args)
