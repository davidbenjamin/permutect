import os
import tempfile

from permutect import constants
from permutect.test.test_file_names import *
from permutect.tools import filter_variants


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
            "--" + constants.REFERENCE_NAME,
            HG19_CHR20_REFERENCE,
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

    permutect_vcf = tempfile.NamedTemporaryFile()
    tensorboard_dir = tempfile.TemporaryDirectory()

    filtering_args = filter_variants.parse_arguments(
        [
            "--" + constants.INPUT_NAME,
            MUTECT2_SEQC2_WES_VCF,
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
            "--" + constants.REFERENCE_NAME,
            HG38_CHR20_REFERENCE,
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

    target_records = []
    with open(permutect_vcf.name) as output_vcf:
        for line in output_vcf:
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if fields[0] == "chr20" and fields[1] == "4932703":
                target_records.append(fields)

    assert len(target_records) == 1
    assert "hairpin" in target_records[0][6].split(";"), (
        "chr20:4932703 did not receive the hairpin filter; inspect its nearby StemLoop energy threshold"
    )
