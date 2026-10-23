import json
import os
import subprocess

import cyvcf2
import pytest

from permutect.test.test_file_names import *
from permutect.tools.filter_variants import POST_PROB_INFO_KEY
from permutect.tools.pre_permutect_filtering import HAIRPIN_FILTER

PERMUTECT_TEST_DOCKER = os.environ.get("PERMUTECT_TEST_DOCKER", "permutect:wdl-test")
CROMWELL_JAR = os.environ.get("CROMWELL_JAR")
PERMUTECT_WDL = os.path.abspath(os.path.join(test_dir, "..", "..", "scripts", "permutect.wdl"))

pytestmark = pytest.mark.skipif(
    CROMWELL_JAR is None, reason="WDL tests need Cromwell and a Permutect docker image; see `pixi run -e wdl test-wdl`"
)


def run_permutect_wdl(working_dir, filter_pcr_errors: bool) -> dict:
    """
    Run permutect.wdl with Cromwell on SEQC2 WES chr20 data, giving cached Mutect2 outputs so that Mutect2 is skipped.
    Returns the Cromwell workflow metadata.
    """
    # the workflow requires reads, but only Mutect2 uses them
    placeholder_bam, placeholder_bai = os.path.join(working_dir, "reads.bam"), os.path.join(working_dir, "reads.bai")
    for placeholder in (placeholder_bam, placeholder_bai):
        open(placeholder, "w").close()

    inputs = {
        "Permutect.artifact_model": PERMUTECT_V0_9_2_ARTIFACT_MODEL,
        "Permutect.ref_fasta": HG38_CHR20_REFERENCE,
        "Permutect.ref_fai": HG38_CHR20_REFERENCE_INDEX,
        "Permutect.ref_dict": HG38_CHR20_REFERENCE_DICT,
        "Permutect.primary_bam": placeholder_bam,
        "Permutect.primary_bai": placeholder_bai,
        "Permutect.scatter_count": 1,
        "Permutect.batch_size": 64,
        # data loader workers share tensors through a Unix socket in TMPDIR, which Cromwell's Local backend puts too
        # deep in the execution directory for the socket path length limit
        "Permutect.num_workers": 0,
        "Permutect.filter_pcr_errors": filter_pcr_errors,
        "Permutect.cached_plain_text_test_dataset": SEQC2_WES_PLAIN_TEXT_DATA,
        "Permutect.cached_mutect2_vcf": MUTECT2_SEQC2_WES_VCF,
        "Permutect.cached_mutect2_vcf_idx": MUTECT2_SEQC2_WES_VCF_INDEX,
        "Permutect.cached_contigs_table": HG38_CONTIGS_TABLE,
        "Permutect.cached_maf_segments": SEQC2_WES_SEGMENTS_TABLE,
        "Permutect.cached_normal_maf_segments": SEQC2_WES_SEGMENTS_TABLE,
        "Permutect.cached_mutect_stats": MUTECT2_SEQC2_WES_STATS,
        "Permutect.permutect_docker": PERMUTECT_TEST_DOCKER,
    }
    inputs_json, metadata_json = os.path.join(working_dir, "inputs.json"), os.path.join(working_dir, "metadata.json")
    with open(inputs_json, "w") as file:
        json.dump(inputs, file)

    # Cromwell writes its executions and logs into the working directory.  Looking up the docker image's hash in a
    # registry is pointless for a locally built image.
    cromwell_command = ["java", "-Ddocker.hash-lookup.enabled=false", "-jar", os.path.abspath(CROMWELL_JAR), "run"]
    subprocess.run(
        cromwell_command + [PERMUTECT_WDL, "--inputs", inputs_json, "--metadata-output", metadata_json],
        cwd=working_dir,
        timeout=3600,
    )
    assert os.path.exists(metadata_json), "Cromwell did not run the workflow; see its log above"
    with open(metadata_json) as file:
        metadata = json.load(file)
    assert metadata["status"] == "Succeeded", json.dumps(metadata.get("failures"), indent=2)
    return metadata


def output_vcf_records(metadata: dict) -> list[cyvcf2.Variant]:
    for output_file in ("output_vcf", "output_vcf_idx", "tensorboard_report"):
        assert os.path.exists(metadata["outputs"]["Permutect." + output_file])
    return list(cyvcf2.VCF(metadata["outputs"]["Permutect.output_vcf"]))


def has_filter(v: cyvcf2.Variant, filter_name: str) -> bool:
    return v.FILTER is not None and filter_name in v.FILTER.split(";")


@pytest.fixture(scope="module")
def metadata_with_pcr_error_filtering(tmp_path_factory):
    return run_permutect_wdl(tmp_path_factory.mktemp("with_pcr_error_filtering"), filter_pcr_errors=True)


@pytest.fixture(scope="module")
def metadata_without_pcr_error_filtering(tmp_path_factory):
    return run_permutect_wdl(tmp_path_factory.mktemp("without_pcr_error_filtering"), filter_pcr_errors=False)


def test_permutect_wdl_with_pcr_error_filtering(metadata_with_pcr_error_filtering):
    assert "Permutect.PrePermutectFiltering" in metadata_with_pcr_error_filtering["calls"]
    records = output_vcf_records(metadata_with_pcr_error_filtering)
    assert any(v.INFO.get(POST_PROB_INFO_KEY) is not None for v in records)

    target_records = [v for v in records if v.CHROM == "chr20" and v.POS == 4932703]
    assert len(target_records) == 1
    assert has_filter(target_records[0], HAIRPIN_FILTER)


def test_permutect_wdl_without_pcr_error_filtering(metadata_without_pcr_error_filtering):
    assert "Permutect.PrePermutectFiltering" not in metadata_without_pcr_error_filtering["calls"]
    records = output_vcf_records(metadata_without_pcr_error_filtering)
    assert any(v.INFO.get(POST_PROB_INFO_KEY) is not None for v in records)
    assert not any(has_filter(v, HAIRPIN_FILTER) for v in records)
