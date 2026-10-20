import os

# this ought to be .../permutect/permutect/test/, where the first "permutect" is the github repo folder
test_dir = os.path.dirname(__file__)
integration_test_dir = os.path.abspath(os.path.join(test_dir, "..", "..", "integration-tests"))

SMALL_PLAIN_TEXT_DATA = os.path.join(integration_test_dir, "hiseqx-NA12878-8000-data-plain-text.dataset")
PREPROCESSED_DATA = os.path.join(integration_test_dir, "preprocessed-dataset.tar")
SMALL_ARTIFACT_MODEL = os.path.join(integration_test_dir, "small_artifact_model.pt")

HG19_CONTIGS_TABLE = os.path.join(integration_test_dir, "contigs.table")
HG38_CONTIGS_TABLE = os.path.join(integration_test_dir, "hg38-contigs.table")
SEGMENTS_TABLE = os.path.join(integration_test_dir, "segments.table")
DREAM_1_CHR20_PLAIN_TEXT_DATA = os.path.join(integration_test_dir, "dream-1-chr-20-plain-text.dataset")
HG19_CHR20_REFERENCE = os.path.join(integration_test_dir, "Homo_sapiens_assembly19_chr20.fasta")
HG38_CHR20_REFERENCE = os.path.join(integration_test_dir, "Homo_sapiens_assembly38_chr20.fasta")
SEQC2_WES_PLAIN_TEXT_DATA = os.path.join(integration_test_dir, "seqc2-wes-chr20-plain-text.dataset")

ARTIFACT_MODEL_V_0_4_0 = os.path.join(integration_test_dir, "artifact-model-v0.4.0.pt")
PERMUTECT_V0_9_2_ARTIFACT_MODEL = os.path.join(integration_test_dir, "permutect-v0.9.2-artifact-model.pt")
EXPERIMENTAL_MODEL = os.path.join(integration_test_dir, "experimental-model.pt")

MUTECT2_CHR20_FILTERED_VCF = os.path.join(integration_test_dir, "mutect2_chr20.vcf")
MUTECT2_SEQC2_WES_VCF = os.path.join(integration_test_dir, "mutect2_seqc2_wes_chr20.vcf")

# for testing SNV-dependent priors, this VCF only has true _ -> T substitution.
MUTECT2_CHR20_FILTERED_VCF_ONLY_T_SNVS = os.path.join(integration_test_dir, "mutect2_chr20_only_T_snvs.vcf")
