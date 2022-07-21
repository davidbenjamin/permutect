from mutect3.tools import filter_variants
from mutect3.architecture import artifact_model

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import tempfile
from torch.utils.tensorboard import SummaryWriter


def test_filtering():
    saved_artifact_model = "/Users/davidben/mutect3/just-dream-1/mutect3.pt"
    test_dataset_file = "/Users/davidben/mutect3/just-dream-1/small-test.dataset"
    unfiltered_vcf = "/Users/davidben/mutect3/just-dream-1/small-m2-calls.vcf"

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
                                          num_ignored_sites=2_000_000_000)
        dummy_breakpoint = 77