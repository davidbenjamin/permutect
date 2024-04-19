import tempfile

from permutect.tools import filter_variants


# TODO: update the saved models in this test -- they are obsolete and incompatible
def test_filtering():
    saved_artifact_model = "/Users/davidben/permutect/just-dream-1/permutect.pt"
    test_dataset_file = "/Users/davidben/mutect3/just-dream-1/small-test.dataset"
    # test_dataset_file = "/Users/davidben/permutect/just-dream-1/big.dataset"
    unfiltered_vcf = "/Users/davidben/permutect/just-dream-1/small-m2-calls.vcf"
    # unfiltered_vcf = "/Users/davidben/permutect/just-dream-1/filtered_m2_calls.vcf"
    segmentation_file = "/Users/davidben/mutect3/just-dream-1/segments.table"

    segmentation = filter_variants.get_segmentation(segmentation_file)

    with tempfile.TemporaryDirectory() as tensorboard_dir, tempfile.TemporaryFile() as output_vcf:
        filter_variants.make_filtered_vcf(saved_artifact_model=saved_artifact_model,
                                          initial_log_variant_prior=-13.0,
                                          initial_log_artifact_prior=-13.0,
                                          test_dataset_file=test_dataset_file,
                                          input_vcf=unfiltered_vcf,
                                          output_vcf=output_vcf,
                                          batch_size=64,
                                          chunk_size=100002,
                                          num_spectrum_iterations=10,
                                          tensorboard_dir=tensorboard_dir,
                                          genomic_span=float("2.0E9"),
                                          segmentation=segmentation)
        dummy_breakpoint = 77