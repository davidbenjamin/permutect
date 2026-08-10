import argparse
from collections import defaultdict

import cyvcf2
import numpy as np
import torch
from intervaltree import IntervalTree
from torch.utils.tensorboard import SummaryWriter
from tqdm.autonotebook import tqdm

from permutect import constants
from permutect.architecture.artifact_model import ArtifactModel
from permutect.architecture.permutect_model import load_model
from permutect.data import plain_text_data
from permutect.data.batch import Batch
from permutect.data.count_binning import MAX_ALT_COUNT
from permutect.data.count_binning import alt_count_bin_index
from permutect.data.count_binning import alt_count_bin_name
from permutect.data.datum import COMPRESSED_READS_ARRAY_DTYPE
from permutect.data.datum import Data
from permutect.data.datum import Datum
from permutect.data.memory_mapped_data import MemoryMappedData
from permutect.data.prefetch_generator import prefetch_generator
from permutect.data.reads_dataset import ReadsDataset
from permutect.metrics.evaluation_metrics import EmbeddingMetrics
from permutect.metrics.evaluation_metrics import EvaluationMetrics
from permutect.metrics.loss_metrics import AccuracyMetrics
from permutect.metrics.posterior_result import PosteriorResult
from permutect.misc_utils import Timer
from permutect.misc_utils import encode_datum
from permutect.misc_utils import encode_variant
from permutect.misc_utils import gpu_if_available
from permutect.misc_utils import overlapping_filters
from permutect.misc_utils import report_memory_usage
from permutect.parameters import PosteriorModelParameters
from permutect.parameters import TrainingParameters
from permutect.parameters import add_posterior_model_params_to_parser
from permutect.parameters import add_training_params_to_parser
from permutect.parameters import parse_posterior_model_params
from permutect.parameters import parse_training_params
from permutect.training.model_training import train_artifact_model
from permutect.utils.allele_utils import find_variant_type
from permutect.utils.enums import Call
from permutect.utils.enums import Epoch
from permutect.utils.enums import Label
from permutect.utils.enums import Variation
from permutect.utils.math_utils import inverse_sigmoid
from permutect.utils.math_utils import prob_to_logit

TRUSTED_M2_FILTERS = {"contamination", "slippage"}

POST_PROB_INFO_KEY = "POST"
ARTIFACT_LOD_INFO_KEY = "ARTLOD"
LOG_PRIOR_INFO_KEY = "PRIOR"
SPECTRA_LOG_LIKELIHOOD_INFO_KEY = "SPECLL"
NORMAL_LOG_LIKELIHOOD_INFO_KEY = "NORMLL"

FILTER_NAMES = [call_type.name.lower() for call_type in Call]


def parse_arguments():
    parser = argparse.ArgumentParser()
    add_training_params_to_parser(parser)  # most of these are only relevant in case of test time adaptation
    add_posterior_model_params_to_parser(parser)
    parser.add_argument("--" + constants.INPUT_NAME, required=True, help="unfiltered input Mutect2 VCF")
    parser.add_argument(
        "--" + constants.TEST_DATASET_NAME,
        required=True,
        help="plain text dataset file corresponding to variants in input VCF",
    )
    parser.add_argument(
        "--" + constants.ARTIFACT_MODEL_NAME,
        required=True,
        help="Permutect artifact model from train_artifact_model.py",
    )
    parser.add_argument(
        "--" + constants.CONTIGS_TABLE_NAME,
        required=True,
        help="table of contig names vs integer indices",
    )
    parser.add_argument("--" + constants.OUTPUT_NAME, required=True, help="path to output filtered VCF")
    parser.add_argument(
        "--" + constants.TENSORBOARD_DIR_NAME,
        type=str,
        default="tensorboard",
        required=False,
        help="path to output tensorboard",
    )
    parser.add_argument(
        "--" + constants.GENOMIC_SPAN_NAME,
        type=float,
        required=True,
        help="number of sites considered by Mutect2, including those lacking variation or artifacts, hence absent from input dataset.  "
        "Necessary for learning priors since otherwise rates of artifacts and variants would be overinflated.",
    )
    parser.add_argument(
        "--" + constants.MAF_SEGMENTS_NAME,
        required=False,
        help="copy-number segmentation file from GATK containing minor allele fractions.  "
        "Useful for modeling germline variation as the minor allele fraction determines the distribution of germline allele counts.",
    )
    parser.add_argument(
        "--" + constants.NORMAL_MAF_SEGMENTS_NAME,
        required=False,
        help="copy-number segmentation file from GATK containing minor allele fractions in the normal/control sample",
    )
    parser.add_argument(
        "--" + constants.RECALL_WEIGHT_NAME,
        type=float,
        default=1.0,
        required=False,
        help="Relative weight of recall vs precision in optimizing F_beta score.",
    )
    parser.add_argument(
        "--" + constants.GERMLINE_MODE_NAME,
        action="store_true",
        help="flag for genotyping both somatic and somatic variants distinctly but considering both "
             "as non-errors (true positives), which affects the posterior threshold set by optimal F1 score",
    )
    return parser.parse_args()


def get_segmentation(segments_file) -> defaultdict:
    result = defaultdict(IntervalTree)
    if segments_file is None:
        return result

    print("reading segmentation file")
    with open(segments_file, "r") as file:
        for line in file:
            if line.startswith("#") or (line.startswith("contig") and "minor_allele_fraction" in line):
                continue
            tokens = line.split()
            contig, start, stop, maf = tokens[0], int(tokens[1]), int(tokens[2]), float(tokens[3])
            if stop > start:  # IntervalTree throws error if start == stop
                result[contig][start:stop] = maf

    return result


def main_without_parsing(args):
    # default for filtering is not to retrain any of the model on test data
    training_params = parse_training_params(args, default_training_params=[])
    posterior_params = parse_posterior_model_params(args)

    make_filtered_vcf(
        artifact_model_path=getattr(args, constants.ARTIFACT_MODEL_NAME),
        training_params=training_params,
        posterior_params=posterior_params,
        test_dataset_file=getattr(args, constants.TEST_DATASET_NAME),
        contigs_table=getattr(args, constants.CONTIGS_TABLE_NAME),
        input_vcf=getattr(args, constants.INPUT_NAME),
        output_vcf=getattr(args, constants.OUTPUT_NAME),
        tensorboard_dir=getattr(args, constants.TENSORBOARD_DIR_NAME),
        genomic_span=getattr(args, constants.GENOMIC_SPAN_NAME),
        germline_mode=getattr(args, constants.GERMLINE_MODE_NAME),
        recall_weight=getattr(args, constants.RECALL_WEIGHT_NAME),
        segmentation=get_segmentation(getattr(args, constants.MAF_SEGMENTS_NAME)),
        normal_segmentation=get_segmentation(getattr(args, constants.NORMAL_MAF_SEGMENTS_NAME)),
    )


def make_filtered_vcf(
    artifact_model_path,
    training_params: TrainingParameters,
    posterior_params: PosteriorModelParameters,
    test_dataset_file,
    contigs_table,
    input_vcf,
    output_vcf,
    tensorboard_dir,
    genomic_span: int,
    germline_mode: bool = False,
    recall_weight: float = 1.0,
    segmentation=None,
    normal_segmentation=None,
):
    if segmentation is None:
        segmentation = defaultdict(IntervalTree)
    if normal_segmentation is None:
        normal_segmentation = defaultdict(IntervalTree)
    print("Loading artifact model and test dataset")
    contig_index_to_name_map = {}
    with open(contigs_table) as file:
        while line := file.readline().strip():
            contig, index = line.split()
            contig_index_to_name_map[int(index)] = contig

    device = gpu_if_available()
    model = load_model(artifact_model_path, device=device)

    annotated_dataset = make_annotated_dataset(
        dataset_file=test_dataset_file,
        input_vcf=input_vcf,
        contig_index_to_name_map=contig_index_to_name_map,
        segmentation=segmentation,
        normal_segmentation=normal_segmentation,
    )

    model.assert_compatible(annotated_dataset)

    # optional test-time domain adaptation
    if training_params.trainable_parameter_sets:
        summary_writer = SummaryWriter(tensorboard_dir, filename_suffix="_adaptation")
        train_artifact_model(
            model=model,
            train_dataset=annotated_dataset,
            valid_dataset=None,
            training_params=training_params,
            summary_writer=summary_writer,
            epochs_per_evaluation=5,
        )

    # TODO: reset model.posterior_model to values specified in posterior_params

    posterior_data_loader = make_posterior_data_loader(
        annotated_dataset=annotated_dataset,
        model=model.artifact_model,
        batch_size=training_params.batch_size,
        num_workers=training_params.num_workers,
    )

    print("Learning AF spectra")
    summary_writer = SummaryWriter(tensorboard_dir)

    num_ignored_sites = genomic_span - len(posterior_data_loader.dataset)
    # here is where pretrained artifact priors and spectra are used if given

    model.posterior_model.learn_priors_and_spectra(
        posterior_data_loader,
        num_iterations=training_params.num_spectrum_iterations,
        summary_writer=summary_writer,
        ignored_to_non_ignored_ratio=num_ignored_sites / len(posterior_data_loader.dataset),
        learning_rate=training_params.spectrum_learning_rate,
    )

    print("Calculating optimal logit threshold")
    error_probability_thresholds = model.posterior_model.calculate_probability_thresholds(
        posterior_data_loader, summary_writer, germline_mode=germline_mode, recall_weight=recall_weight
    )
    print(f"Optimal probability threshold: {error_probability_thresholds}")
    apply_filtering_to_vcf(
        input_vcf,
        output_vcf,
        contig_index_to_name_map,
        error_probability_thresholds,
        posterior_data_loader,
        model.posterior_model,
        summary_writer=summary_writer,
        germline_mode=germline_mode,
    )


@torch.inference_mode()
def generate_posterior_data(dataset, model: ArtifactModel, batch_size: int, num_workers: int):
    # pass through the dataset, running the artifact model
    # to get artifact logits, which we record in a dict keyed by variant strings.  These will later be added to PosteriorDatum objects.
    loader = dataset.make_data_loader(batch_size, pin_memory=torch.cuda.is_available(), num_workers=num_workers)

    print("creating posterior data...")
    batch: Batch
    for batch in tqdm(prefetch_generator(loader), mininterval=60, total=len(loader)):
        output = model.compute_batch_output(batch)
        for int_array, float_array, logit, embedding in zip(
            batch.get_int_array_be(),
            batch.get_float_array_be(),
            output.logits_b.detach().tolist(),
            output.features_be.cpu(),
        ):
            # make a Datum with no reads or haplotypes whose 1D info array is the embedding
            empty_reads = np.zeros((0, 0), dtype=COMPRESSED_READS_ARRAY_DTYPE)
            output_datum = Datum(
                int_array=int_array,
                float_array=float_array,
                reads_re=empty_reads,
                compressed=True,
            )
            output_datum.set(Data.REF_COUNT, 0)
            output_datum.set(Data.ALT_COUNT, 0)
            output_datum.set(Data.CACHED_ARTIFACT_LOGIT, logit)
            output_datum.set_info_1d(embedding)
            yield output_datum


def make_annotated_dataset(
    dataset_file,
    input_vcf,
    contig_index_to_name_map,
    segmentation=None,
    normal_segmentation=None,
):
    if segmentation is None:
        segmentation = defaultdict(IntervalTree)
    if normal_segmentation is None:
        normal_segmentation = defaultdict(IntervalTree)
    normalizing_timer = Timer("Normalizing data. . .")
    normalized_mmap_data = plain_text_data.make_normalized_mmap_data(dataset_files=[dataset_file])
    normalizing_timer.report("Time to normalize test data:")

    report_memory_usage("Creating ReadsDataset with AF and MAF.")
    annotation_timer = Timer("Annotating data with AF and MAF from VCF. . .")
    annotated_mmap_data = normalized_mmap_data.make_vcf_annotate_memory_mapped_data(
        input_vcf,
        contig_index_to_name_map,
        filters_to_exclude=TRUSTED_M2_FILTERS,
        segmentation=segmentation,
        normal_segmentation=normal_segmentation,
    )
    annotated_dataset = ReadsDataset(memory_mapped_data=annotated_mmap_data)
    annotation_timer.report("Time to annotate data with AF and MAF:")
    return annotated_dataset


@torch.inference_mode()
def make_posterior_data_loader(
    annotated_dataset: ReadsDataset,
    model: ArtifactModel,
    batch_size: int,
    num_workers: int,
):
    # Generate Datum objects without reads or haplotypes, where the INFO array is the embedding, and with the
    # cached artifact logit computed from the model
    # TODO: posterior data should include information about clustering once artifact clusters have their own spectra
    posterior_generator = generate_posterior_data(annotated_dataset, model, batch_size, num_workers)
    posterior_mmap = MemoryMappedData.from_generator(
        posterior_generator, estimated_num_data=len(annotated_dataset), estimated_num_reads=0
    )
    print(f"Size of filtering dataset: {len(posterior_mmap)}")

    posterior_dataset = ReadsDataset(posterior_mmap)
    report_memory_usage("Finished creating posterior ReadsDataset.")
    return posterior_dataset.make_data_loader(batch_size, pin_memory=torch.cuda.is_available(), num_workers=num_workers)


# error probability thresholds is a dict from Variant type to error probability threshold (float)
@torch.inference_mode()
def apply_filtering_to_vcf(
    input_vcf,
    output_vcf,
    contig_index_to_name_map,
    error_probability_thresholds,
    posterior_loader,
    posterior_model,
    summary_writer: SummaryWriter,
    germline_mode: bool = False,
):
    print("Computing final error probabilities")
    passing_call_type = Call.GERMLINE if germline_mode else Call.SOMATIC
    evaluation_metrics = EvaluationMetrics(num_sources=1)

    # Note: using BatchIndexedTotals in a hacky way, with Call replacing Source!
    artifact_logit_metrics = AccuracyMetrics.create(num_sources=len(Call))
    encoding_to_posterior_results = {}

    batch: Batch
    for batch in tqdm(prefetch_generator(posterior_loader), mininterval=60, total=len(posterior_loader)):
        # posterior, along with intermediate tensors for debugging/interpretation
        log_priors_bc, spectra_log_lks_bc, normal_log_lks_bc, log_posteriors_bc = (
            posterior_model.log_posterior_and_ingredients(batch)
        )

        posterior_probs_bc = torch.nn.functional.softmax(log_posteriors_bc, dim=1)
        error_probs_b = 1 - posterior_probs_bc[:, passing_call_type]
        error_logits_b = inverse_sigmoid(error_probs_b)
        # this does nothing if the test dataset was generated without a truth VCF and thus has no labels
        # note that here we use error logits, not artifact logits
        # TODO: maybe also have an option to record relative to the computed probability thresholds.
        # TODO: this code here treats posterior_prob = 1/2 as the threshold
        # TODO: we could perhaps subtract the threshold to re-center at zero
        evaluation_metrics.record_batch(Epoch.TEST, batch, logits=error_logits_b, use_original_counts=True)

        most_confident_probs_b, most_confident_calls_b = torch.max(posterior_probs_bc, dim=-1)
        artifact_logit_metrics.record_with_sources_and_logits(
            batch,
            values=most_confident_probs_b,
            sources_override=most_confident_calls_b,
            logits=batch.get(Data.CACHED_ARTIFACT_LOGIT),
            use_original_counts=True,
        )

        artifact_logits = batch.get(Data.CACHED_ARTIFACT_LOGIT).cpu().tolist()
        data = [
            Datum(int_array, float_array)
            for (int_array, float_array) in zip(batch.get_int_array_be(), batch.get_float_array_be())
        ]
        # NOTE: for posterior data, batch.get_info_be actually gets the embedding array!!!!!
        # TODO: perhaps make this safer
        for datum, post_probs, logit, log_prior, log_spec, log_normal, embedding in zip(
            data,
            posterior_probs_bc,
            artifact_logits,
            log_priors_bc,
            spectra_log_lks_bc,
            normal_log_lks_bc,
            batch.get_info_be(),
        ):
            encoding = encode_datum(datum, contig_index_to_name_map)
            encoding_to_posterior_results[encoding] = PosteriorResult(
                artifact_logit=logit,
                posterior_probabilities=post_probs.tolist(),
                log_priors=log_prior,
                spectra_lls=log_spec,
                normal_lls=log_normal,
                label=datum.get(Data.LABEL),
                alt_count=datum.get(Data.ORIGINAL_ALT_COUNT),
                depth=datum.get(Data.ORIGINAL_DEPTH),
                var_type=datum.get(Data.VARIANT_TYPE),
                embedding=embedding,
            )

    print("Applying threshold")
    unfiltered_vcf = cyvcf2.VCF(input_vcf)

    # This is specific to Mutect2, which outputs a line:
    # ##tumor_sample=___________
    tumor_sample_name = ""
    all_samples = []
    for header_line in unfiltered_vcf.raw_header.split("\n"):
        if header_line.startswith("##tumor_sample"):
            tumor_sample_name = header_line.split("=")[-1]
            print(tumor_sample_name)
        elif header_line.startswith("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t"):
            prefix = "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t"
            all_samples = header_line[len(prefix) :].split()
    tumor_sample_index = 0
    for n, sample in enumerate(all_samples):
        if sample == tumor_sample_name:
            print(f"tumor sample found in VCF header: name {sample}, index {n}")
            tumor_sample_index = n

    all_types = [call_type.name for call_type in Call]
    unfiltered_vcf.add_format_to_header({"ID": "DP", "Description": "depth", "Type": "Integer", "Number": "1"})
    unfiltered_vcf.add_info_to_header(
        {
            "ID": POST_PROB_INFO_KEY,
            "Description": "Mutect3 posterior probability of {" + ", ".join(all_types) + "}",
            "Type": "Float",
            "Number": "A",
        }
    )
    unfiltered_vcf.add_info_to_header(
        {
            "ID": LOG_PRIOR_INFO_KEY,
            "Description": "Log priors of {" + ", ".join(all_types) + "}",
            "Type": "Float",
            "Number": "A",
        }
    )
    unfiltered_vcf.add_info_to_header(
        {
            "ID": SPECTRA_LOG_LIKELIHOOD_INFO_KEY,
            "Description": "Log spectra likelihoods of {" + ", ".join(all_types) + "}",
            "Type": "Float",
            "Number": "A",
        }
    )
    unfiltered_vcf.add_info_to_header(
        {
            "ID": NORMAL_LOG_LIKELIHOOD_INFO_KEY,
            "Description": "Log normal likelihoods of {" + ", ".join(all_types) + "}",
            "Type": "Float",
            "Number": "A",
        }
    )
    unfiltered_vcf.add_info_to_header(
        {
            "ID": ARTIFACT_LOD_INFO_KEY,
            "Description": "Mutect3 artifact log odds",
            "Type": "Float",
            "Number": "A",
        }
    )

    for n, filter_name in enumerate(FILTER_NAMES):
        if n != passing_call_type:
            unfiltered_vcf.add_filter_to_header({"ID": filter_name, "Description": filter_name})

    writer = cyvcf2.Writer(output_vcf, unfiltered_vcf)  # input vcf is a template for the header
    pbar = tqdm(enumerate(unfiltered_vcf), mininterval=60)
    labeled_truth = False
    embedding_metrics = EmbeddingMetrics()  # only if there is labeled truth for evaluation

    missing_encodings = []
    for n, v in pbar:
        filters = overlapping_filters(v, TRUSTED_M2_FILTERS)

        # TODO: in germline mode, somatic doesn't exist (or is just highly irrelevant) and germline is not an error!
        encoding = encode_variant(v, zero_based=True)  # cyvcf2 is zero-based
        if encoding in encoding_to_posterior_results:
            posterior_result = encoding_to_posterior_results[encoding]
            post_probs = posterior_result.posterior_probabilities
            v.INFO[POST_PROB_INFO_KEY] = ",".join(map(lambda prob: "{:.3f}".format(prob), post_probs))
            v.INFO[LOG_PRIOR_INFO_KEY] = ",".join(map(lambda pri: "{:.3f}".format(pri), posterior_result.log_priors))
            v.INFO[SPECTRA_LOG_LIKELIHOOD_INFO_KEY] = ",".join(
                map(lambda ll: "{:.3f}".format(ll), posterior_result.spectra_lls)
            )
            v.INFO[ARTIFACT_LOD_INFO_KEY] = "{:.3f}".format(posterior_result.artifact_logit)
            v.INFO[NORMAL_LOG_LIKELIHOOD_INFO_KEY] = ",".join(
                map(lambda ll: "{:.3f}".format(ll), posterior_result.normal_lls)
            )

            label = Label(posterior_result.label)  # this is the Label enum, might be UNLABELED
            error_prob = 1 - post_probs[passing_call_type]
            variant_type = find_variant_type(v)
            called_as_error = error_prob > error_probability_thresholds[variant_type]

            error_call = None

            if called_as_error:
                # get the error type with the largest posterior probability
                highest_prob_indices = torch.topk(torch.tensor(post_probs), 2).indices.tolist()
                highest_prob_index = (
                    highest_prob_indices[1] if highest_prob_indices[0] == passing_call_type else highest_prob_indices[0]
                )
                error_call = list(Call)[highest_prob_index]
                filters.add(FILTER_NAMES[highest_prob_index])

            # note that this excludes the correctness part of embedding metrics, which is below
            embedding_metrics.label_metadata.append(label.name)
            embedding_metrics.type_metadata.append(variant_type.name)
            embedding_metrics.truncated_count_metadata.append(
                alt_count_bin_name(alt_count_bin_index(min(MAX_ALT_COUNT, posterior_result.alt_count)))
            )
            embedding_metrics.features.append(posterior_result.embedding)
            # TODO: we don't yet record ref features but we could eventually. . .

            correctness_label = "unknown"
            if label != Label.UNLABELED:
                labeled_truth = True
                clipped_error_prob = 0.5 + 0.9999999 * (error_prob - 0.5)  # noqa: F841

                # TODO: this is sloppy -- it only works because when we label the posterior dataset (if truth is available)
                # TODO: we stretch the definitions so that "Label.ARTIFACT" simply means "something we shouldn't call", including
                # TODO: artifact or germline (in the somatic calling case), and "Label.VARIANT" means "something we should call"
                is_correct = (called_as_error and label == Label.ARTIFACT) or (
                    not called_as_error and label == Label.VARIANT
                )

                # TODO: double-check the logic here
                if is_correct:
                    if label == Label.VARIANT:
                        correctness_label = EmbeddingMetrics.TRUE_POSITIVE
                    elif error_call == Call.ARTIFACT or error_call == Call.NORMAL_ARTIFACT:
                        correctness_label = EmbeddingMetrics.TRUE_NEGATIVE_ARTIFACT
                    # elif error_call == Call.SEQ_ERROR:
                    #    correctness_label = EmbeddingMetrics.TRUE_NEGATIVE_SEQ_ERROR
                    # we don't do anything for germline (in somatic mode) or seq error --
                else:
                    if called_as_error:
                        if error_call == Call.ARTIFACT or error_call == Call.NORMAL_ARTIFACT:
                            correctness_label = EmbeddingMetrics.FALSE_NEGATIVE_ARTIFACT
                    else:
                        correctness_label = EmbeddingMetrics.FALSE_POSITIVE
                    # TODO: this is only right for somatic calling
                    bad_call = error_call if called_as_error else Call.SOMATIC
                    evaluation_metrics.record_mistake(posterior_result, bad_call)
            embedding_metrics.correct_metadata.append(correctness_label)
        else:
            # It is possible due to various quirks of Mutect2 assembly and flags such as --genotype-germline-sites etc
            # that a site with zero alt depth can end up in the output VCF.  However, Permutect exludes such sites from
            # the test dataset.  Therefore, we manually check for such sites and make sure they get filtered!
            total_alt_depth = np.sum(v.format("AD")[tumor_sample_index][1:])
            if total_alt_depth == 0:
                filters.add(FILTER_NAMES[Call.SEQ_ERROR])
            missing_encodings.append(encoding)
        v.FILTER = ";".join(filters) if filters else "PASS"
        writer.write_record(v)
    print("closing resources")
    writer.close()
    unfiltered_vcf.close()

    embedding_metrics.output_to_summary_writer(summary_writer, is_filter_variants=True)

    # recall that "sources" is really call type here
    artifact_logit_metrics = artifact_logit_metrics.cpu()
    evaluation_metrics.put_on_cpu()
    metrics: AccuracyMetrics
    for call_type, metrics in zip(Call, artifact_logit_metrics.split_over_sources()):
        hist_fig, hist_ax = metrics.make_logit_histograms()
        summary_writer.add_figure(f"artifact logit histograms for call type {call_type.name}", hist_fig)

    if labeled_truth:
        given_thresholds = {var_type: prob_to_logit(error_probability_thresholds[var_type]) for var_type in Variation}
        evaluation_metrics.make_plots(summary_writer, given_thresholds, sens_prec=True)
        evaluation_metrics.make_mistake_histograms(summary_writer)


def main():
    args = parse_arguments()
    main_without_parsing(args)


if __name__ == "__main__":
    main()
