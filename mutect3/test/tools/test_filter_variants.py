import tempfile

from mutect3.tools import filter_variants


def test_filtering():
    saved_artifact_model = "/Users/davidben/mutect3/just-dream-1/mutect3.pt"
    test_dataset_file = "/Users/davidben/mutect3/just-dream-1/small-test.dataset"
    # test_dataset_file = "/Users/davidben/mutect3/just-dream-1/big.dataset"
    unfiltered_vcf = "/Users/davidben/mutect3/just-dream-1/small-m2-calls.vcf"
    # unfiltered_vcf = "/Users/davidben/mutect3/just-dream-1/filtered_m2_calls.vcf"
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
                                          num_spectrum_iterations=10,
                                          tensorboard_dir=tensorboard_dir,
                                          num_ignored_sites=float("2.0E9"),
                                          segmentation=segmentation)
        dummy_breakpoint = 77