import tempfile

import cyvcf2
import pytest

from permutect import constants
from permutect.test.test_file_names import *
from permutect.tools import filter_variants
from permutect.tools import pre_permutect_filtering
from permutect.utils.enums import Call


def test_filtering_on_dream1_chr20():
    artifact_model = PERMUTECT_V0_9_2_ARTIFACT_MODEL

    permutect_vcf = tempfile.NamedTemporaryFile()
    tensorboard_dir = tempfile.TemporaryDirectory()

    filtering_args = filter_variants.parse_arguments(
        [
            "--" + constants.INPUT_NAME,
            MUTECT2_CHR20_FILTERED_VCF,
            "--" + constants.TEST_DATASET_NAME,
            DREAM_1_CHR20_PLAIN_TEXT_DATA,
            "--" + constants.ARTIFACT_MODEL_NAME,
            artifact_model,
            "--" + constants.TRAINABLE_PARAMETERS_NAME,
            "CLUSTERING",
            "--" + constants.OUTPUT_NAME,
            permutect_vcf.name,
            "--" + constants.TENSORBOARD_DIR_NAME,
            tensorboard_dir.name,
            "--" + constants.GENOMIC_SPAN_NAME,
            "60000000",
            "--" + constants.CONTIGS_TABLE_NAME,
            HG19_CONTIGS_TABLE,
            "--" + constants.BATCH_SIZE_NAME,
            "64",
            "--" + constants.NUM_WORKERS_NAME,
            "2",
            "--" + constants.NUM_EPOCHS_NAME,
            "2",
            "--" + constants.WEIGHT_DECAY_NAME,
            "0.01",
        ]
    )

    filter_variants.main_without_parsing(filtering_args)


def test_filtering_on_seqc2_wes_chr20():
    artifact_model = PERMUTECT_V0_9_2_ARTIFACT_MODEL

    pre_filtered_vcf = tempfile.NamedTemporaryFile(suffix=".vcf")
    permutect_vcf = tempfile.NamedTemporaryFile()
    tensorboard_dir = tempfile.TemporaryDirectory()

    pre_filtering_args = pre_permutect_filtering.parse_arguments(
        [
            "--" + constants.INPUT_NAME,
            MUTECT2_SEQC2_WES_VCF,
            "--" + constants.REFERENCE_NAME,
            HG38_CHR20_REFERENCE,
            "--" + constants.OUTPUT_NAME,
            pre_filtered_vcf.name,
        ]
    )
    pre_permutect_filtering.main_without_parsing(pre_filtering_args)

    filtering_args = filter_variants.parse_arguments(
        [
            "--" + constants.INPUT_NAME,
            pre_filtered_vcf.name,
            "--" + constants.TEST_DATASET_NAME,
            SEQC2_WES_PLAIN_TEXT_DATA,
            "--" + constants.ARTIFACT_MODEL_NAME,
            artifact_model,
            "--" + constants.TRAINABLE_PARAMETERS_NAME,
            "CLUSTERING",
            "--" + constants.OUTPUT_NAME,
            permutect_vcf.name,
            "--" + constants.TENSORBOARD_DIR_NAME,
            tensorboard_dir.name,
            "--" + constants.GENOMIC_SPAN_NAME,
            "60000000",
            "--" + constants.CONTIGS_TABLE_NAME,
            HG38_CONTIGS_TABLE,
            "--" + constants.BATCH_SIZE_NAME,
            "64",
            "--" + constants.NUM_WORKERS_NAME,
            "2",
            "--" + constants.NUM_EPOCHS_NAME,
            "2",
            "--" + constants.WEIGHT_DECAY_NAME,
            "0.01",
        ]
    )

    filter_variants.main_without_parsing(filtering_args)

    # variants with trusted non-artifact filters are output unchanged, while variants with trusted artifact filters
    # keep them and are seen by the posterior model as definite artifacts or normal artifacts
    input_records = {(v.CHROM, v.POS, v.REF, v.ALT[0]): str(v) for v in cyvcf2.VCF(pre_filtered_vcf.name)}
    num_non_artifact_filtered, num_artifact_filtered = 0, 0
    for v in cyvcf2.VCF(permutect_vcf.name):
        filters = set(v.FILTER.split(";")) if v.FILTER else set()
        if filters & filter_variants.TRUSTED_NON_ARTIFACT_FILTERS:
            num_non_artifact_filtered += 1
            assert str(v) == input_records[(v.CHROM, v.POS, v.REF, v.ALT[0])]
        elif filters & filter_variants.TRUSTED_ARTIFACT_FILTERS:
            num_artifact_filtered += 1
            posterior_probs = v.INFO.get(filter_variants.POST_PROB_INFO_KEY)
            assert posterior_probs[Call.ARTIFACT] + posterior_probs[Call.NORMAL_ARTIFACT] == pytest.approx(1, abs=0.01)
            assert v.INFO.get(filter_variants.ARTIFACT_LOD_INFO_KEY) is None
    assert num_non_artifact_filtered > 0 and num_artifact_filtered > 0

    target_records = [v for v in cyvcf2.VCF(permutect_vcf.name) if v.CHROM == "chr20" and v.POS == 4932703]
    assert len(target_records) == 1
    assert pre_permutect_filtering.HAIRPIN_FILTER in target_records[0].FILTER.split(";")
