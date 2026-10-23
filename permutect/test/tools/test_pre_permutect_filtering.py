import tempfile

import cyvcf2
import pytest

from permutect import constants
from permutect.test.test_file_names import *
from permutect.tools import pre_permutect_filtering


def test_hairpin_filtering_on_seqc2_wes_chr20():
    output_vcf = tempfile.NamedTemporaryFile(suffix=".vcf")

    args = pre_permutect_filtering.parse_arguments(
        [
            "--" + constants.INPUT_NAME,
            MUTECT2_SEQC2_WES_VCF,
            "--" + constants.REFERENCE_NAME,
            HG38_CHR20_REFERENCE,
            "--" + constants.OUTPUT_NAME,
            output_vcf.name,
        ]
    )
    pre_permutect_filtering.main_without_parsing(args)

    target_records = [v for v in cyvcf2.VCF(output_vcf.name) if v.CHROM == "chr20" and v.POS == 4932703]
    assert len(target_records) == 1
    record = target_records[0]
    assert pre_permutect_filtering.HAIRPIN_FILTER in record.FILTER.split(";"), (
        "chr20:4932703 did not receive the hairpin filter; inspect its nearby StemLoop energy threshold"
    )
    assert record.INFO.get(pre_permutect_filtering.STEM_LOOP_FREE_ENERGY_INFO_KEY) == pytest.approx(1.3)
    assert record.INFO.get(pre_permutect_filtering.STEM_LOOP_DISTANCE_INFO_KEY) == 0


def test_vcf_without_format_dp_header():
    # like the output of SplitMultiallelics in permutect.wdl, this VCF has FORMAT/DP values but no FORMAT/DP header line
    output_vcf = tempfile.NamedTemporaryFile(suffix=".vcf")

    args = pre_permutect_filtering.parse_arguments(
        [
            "--" + constants.INPUT_NAME,
            MUTECT2_CHR20_FILTERED_VCF,
            "--" + constants.REFERENCE_NAME,
            HG19_CHR20_REFERENCE,
            "--" + constants.OUTPUT_NAME,
            output_vcf.name,
        ]
    )
    pre_permutect_filtering.main_without_parsing(args)

    num_input_records = sum(1 for _ in cyvcf2.VCF(MUTECT2_CHR20_FILTERED_VCF))
    assert sum(1 for _ in cyvcf2.VCF(output_vcf.name)) == num_input_records
