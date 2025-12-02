import tempfile
from argparse import Namespace

from permutect import constants
from permutect.tools import filter_variants
from permutect.test.test_file_names import *


def test_filtering_on_dream1_chr20():
    # Inputs
    artifact_model = ARTIFACT_MODEL_V_0_4_0
    mutect2_vcf = MUTECT2_CHR20_FILTERED_VCF
    maf_segments = SEGMENTS_TABLE
    contigs_table = CONTIGS_TABLE
    filtering_dataset = DREAM_1_CHR20_PLAIN_TEXT_DATA

    # Outputs
    permutect_vcf = tempfile.NamedTemporaryFile()
    tensorboard_dir = tempfile.TemporaryDirectory()

    filtering_args = Namespace()
    setattr(filtering_args, constants.INPUT_NAME, mutect2_vcf)
    setattr(filtering_args, constants.TEST_DATASET_NAME, filtering_dataset)
    setattr(filtering_args, constants.ARTIFACT_MODEL_NAME, artifact_model)
    setattr(filtering_args, constants.OUTPUT_NAME, permutect_vcf.name)
    setattr(filtering_args, constants.TENSORBOARD_DIR_NAME, tensorboard_dir.name)
    setattr(filtering_args, constants.BATCH_SIZE_NAME, 64)
    setattr(filtering_args, constants.NUM_WORKERS_NAME, 0)
    setattr(filtering_args, constants.NUM_SPECTRUM_ITERATIONS_NAME, 2)
    setattr(filtering_args, constants.HET_BETA_NAME, 10)
    setattr(filtering_args, constants.SPECTRUM_LEARNING_RATE_NAME, 0.001)
    setattr(filtering_args, constants.INITIAL_LOG_VARIANT_PRIOR_NAME, -10.0)
    setattr(filtering_args, constants.INITIAL_LOG_ARTIFACT_PRIOR_NAME, -10.0)
    setattr(filtering_args, constants.GENOMIC_SPAN_NAME, 60000000)
    setattr(filtering_args, constants.MAF_SEGMENTS_NAME, None)
    setattr(filtering_args, constants.CONTIGS_TABLE_NAME, contigs_table)
    setattr(filtering_args, constants.NORMAL_MAF_SEGMENTS_NAME, None)
    setattr(filtering_args, constants.GERMLINE_MODE_NAME, False)
    setattr(filtering_args, constants.NO_GERMLINE_MODE_NAME, False)

    filter_variants.main_without_parsing(filtering_args)
    h = 9

test_filtering_on_dream1_chr20()