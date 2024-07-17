import argparse
import math
from collections import defaultdict
from typing import Set

import cyvcf2
import torch
from intervaltree import IntervalTree
from torch.utils.tensorboard import SummaryWriter
from tqdm.autonotebook import tqdm

from permutect import constants
from permutect.architecture.artifact_model import ArtifactModel, load_artifact_model
from permutect.architecture.posterior_model import PosteriorModel
from permutect.architecture.base_model import BaseModel, load_base_model
from permutect.data import base_dataset, plain_text_data
from permutect.data.posterior import PosteriorDataset, PosteriorDatum
from permutect.data.artifact_dataset import ArtifactDataset
from permutect.metrics.evaluation_metrics import EvaluationMetrics, PosteriorResult, EmbeddingMetrics, \
    round_up_to_nearest_three, MAX_COUNT
from permutect.utils import Call, find_variant_type, Label, Variation, Epoch

TRUSTED_M2_FILTERS = {'contamination'}

POST_PROB_INFO_KEY = 'POST'
ARTIFACT_LOD_INFO_KEY = 'ARTLOD'
LOG_PRIOR_INFO_KEY = 'PRIOR'
SPECTRA_LOG_LIKELIHOOD_INFO_KEY = 'SPECLL'
NORMAL_LOG_LIKELIHOOD_INFO_KEY = 'NORMLL'

FILTER_NAMES = [call_type.name.lower() for call_type in Call]


# the inverse of the sigmoid function.  Convert a probability to a logit.
def prob_to_logit(prob: float):
    return math.log(prob / (1 - prob))


def get_first_numeric_element(variant, key):
    tuple_or_scalar = variant.INFO[key]
    return tuple_or_scalar[0] if type(tuple_or_scalar) is tuple else tuple_or_scalar


# if alt and ref alleles are not in minimal representation ie have redundant matching bases at the end, trim them
def trim_alleles_on_right(ref: str, alt: str):
    trimmed_ref, trimmed_alt = ref, alt
    while len(ref) > 1 and len(alt) > 1 and trimmed_alt[-1] == trimmed_ref[-1]:
        trimmed_ref, trimmed_alt = trimmed_ref[:-1], trimmed_alt[:-1]
    return trimmed_ref, trimmed_alt


# TODO: contigs stored as integer index must be converted back to string to compare VCF variants with dataset variants!!!
def encode(contig: str, position: int, ref: str, alt: str):
    trimmed_ref, trimmed_alt = trim_alleles_on_right(ref, alt)
    return contig + ':' + str(position) + ':' + trimmed_alt


def encode_datum(datum: PosteriorDatum, contig_index_to_name_map):
    contig_name = contig_index_to_name_map[datum.contig]
    return encode(contig_name, datum.position, datum.ref, datum.alt)


def encode_variant(v: cyvcf2.Variant, zero_based=False):
    alt = v.ALT[0]  # TODO: we're assuming biallelic
    ref = v.REF
    start = (v.start + 1) if zero_based else v.start
    return encode(v.CHROM, start, ref, alt)


def filters_to_keep_from_m2(v: cyvcf2.Variant) -> Set[str]:
    return set([]) if v.FILTER is None else set(v.FILTER.split(";")).intersection(TRUSTED_M2_FILTERS)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--' + constants.INPUT_NAME, required=True, help='unfiltered input Mutect2 VCF')
    parser.add_argument('--' + constants.TEST_DATASET_NAME, required=True,
                        help='plain text dataset file corresponding to variants in input VCF')
    parser.add_argument('--' + constants.M3_MODEL_NAME, required=True, help='trained Mutect3 artifact model from train_model.py')
    parser.add_argument('--' + constants.BASE_MODEL_NAME, type=str, help='Base model from train_base_model.py')
    parser.add_argument('--' + constants.CONTIGS_TABLE_NAME, required=True, help='table of contig names vs integer indices')
    parser.add_argument('--' + constants.OUTPUT_NAME, required=True, help='path to output filtered VCF')
    parser.add_argument('--' + constants.TENSORBOARD_DIR_NAME, type=str, default='tensorboard', required=False, help='path to output tensorboard')
    parser.add_argument('--' + constants.BATCH_SIZE_NAME, type=int, default=64, required=False, help='batch size')
    parser.add_argument('--' + constants.CHUNK_SIZE_NAME, type=int, default=100000, required=False, help='size in bytes of intermediate binary datasets')
    parser.add_argument('--' + constants.NUM_SPECTRUM_ITERATIONS, type=int, default=10, required=False,
                        help='number of epochs for fitting allele fraction spectra')
    parser.add_argument('--' + constants.INITIAL_LOG_VARIANT_PRIOR_NAME, type=float, default=-10.0, required=False,
                        help='initial value for natural log prior of somatic variants')
    parser.add_argument('--' + constants.INITIAL_LOG_ARTIFACT_PRIOR_NAME, type=float, default=-10.0, required=False,
                        help='initial value for natural log prior of artifacts')
    parser.add_argument('--' + constants.GENOMIC_SPAN_NAME, type=float, required=True,
                        help='number of sites considered by Mutect2, including those lacking variation or artifacts, hence absent from input dataset.  '
                             'Necessary for learning priors since otherwise rates of artifacts and variants would be overinflated.')
    parser.add_argument('--' + constants.MAF_SEGMENTS_NAME, required=False,
                        help='copy-number segmentation file from GATK containing minor allele fractions.  '
                             'Useful for modeling germline variation as the minor allele fraction determines the distribution of germline allele counts.')
    parser.add_argument('--' + constants.NORMAL_MAF_SEGMENTS_NAME, required=False,
                        help='copy-number segmentation file from GATK containing minor allele fractions in the normal/control sample')

    parser.add_argument('--' + constants.GERMLINE_MODE_NAME, action='store_true',
                        help='flag for genotyping both somatic and somatic variants distinctly but considering both '
                             'as non-errors (true positives), which affects the posterior threshold set by optimal F1 score')

    parser.add_argument('--' + constants.NO_GERMLINE_MODE_NAME, action='store_true',
                        help='flag for not genotyping germline events so that the only possibilities considered are '
                             'somatic, artifact, and sequencing error.  This is useful for certain validation where '
                             'pseudo-somatic events are created by mixing germline events at varying fractions')
    return parser.parse_args()


def get_segmentation(segments_file) -> defaultdict:

    result = defaultdict(IntervalTree)
    if segments_file is None:
        return result

    print(" reading segmentation file")
    with open(segments_file, 'r') as file:
        for line in file:
            if line.startswith("#") or (line.startswith("contig") and "minor_allele_fraction" in line):
                continue
            tokens = line.split()
            contig, start, stop, maf = tokens[0], int(tokens[1]), int(tokens[2]), float(tokens[3])
            if stop > start:    # IntervalTree throws error if start == stop
                result[contig][start:stop] = maf

    return result


def main_without_parsing(args):
    make_filtered_vcf(saved_artifact_model_path=getattr(args, constants.M3_MODEL_NAME),
                      base_model_path=getattr(args, constants.BASE_MODEL_NAME),
                      initial_log_variant_prior=getattr(args, constants.INITIAL_LOG_VARIANT_PRIOR_NAME),
                      initial_log_artifact_prior=getattr(args, constants.INITIAL_LOG_ARTIFACT_PRIOR_NAME),
                      test_dataset_file=getattr(args, constants.TEST_DATASET_NAME),
                      contigs_table=getattr(args, constants.CONTIGS_TABLE_NAME),
                      input_vcf=getattr(args, constants.INPUT_NAME),
                      output_vcf=getattr(args, constants.OUTPUT_NAME),
                      batch_size=getattr(args, constants.BATCH_SIZE_NAME),
                      chunk_size=getattr(args, constants.CHUNK_SIZE_NAME),
                      num_spectrum_iterations=getattr(args, constants.NUM_SPECTRUM_ITERATIONS),
                      tensorboard_dir=getattr(args, constants.TENSORBOARD_DIR_NAME),
                      genomic_span=getattr(args, constants.GENOMIC_SPAN_NAME),
                      germline_mode=getattr(args, constants.GERMLINE_MODE_NAME),
                      no_germline_mode=getattr(args, constants.NO_GERMLINE_MODE_NAME),
                      segmentation=get_segmentation(getattr(args, constants.MAF_SEGMENTS_NAME)),
                      normal_segmentation=get_segmentation(getattr(args, constants.NORMAL_MAF_SEGMENTS_NAME)))


def make_filtered_vcf(saved_artifact_model_path, base_model_path, initial_log_variant_prior: float, initial_log_artifact_prior: float,
                      test_dataset_file, contigs_table, input_vcf, output_vcf, batch_size: int, chunk_size: int, num_spectrum_iterations: int, tensorboard_dir,
                      genomic_span: int, germline_mode: bool = False, no_germline_mode: bool = False, segmentation=defaultdict(IntervalTree),
                      normal_segmentation=defaultdict(IntervalTree)):
    print("Loading artifact model and test dataset")
    base_model = load_base_model(base_model_path)
    contig_index_to_name_map = {}
    with open(contigs_table) as file:
        while line := file.readline().strip():
            contig, index = line.split()
            contig_index_to_name_map[int(index)] = contig

    artifact_model, artifact_log_priors, artifact_spectra_state_dict = load_artifact_model(saved_artifact_model_path)
    posterior_model = PosteriorModel(initial_log_variant_prior, initial_log_artifact_prior, no_germline_mode=no_germline_mode, num_base_features=artifact_model.num_base_features)
    posterior_data_loader = make_posterior_data_loader(test_dataset_file, input_vcf, contig_index_to_name_map,
        base_model, artifact_model, batch_size, chunk_size=chunk_size, segmentation=segmentation, normal_segmentation=normal_segmentation)

    print("Learning AF spectra")
    summary_writer = SummaryWriter(tensorboard_dir)

    num_ignored_sites = genomic_span - len(posterior_data_loader.dataset)
    # here is where pretrained artifact priors and spectra are used if given

    posterior_model.learn_priors_and_spectra(posterior_data_loader, num_iterations=num_spectrum_iterations,
        summary_writer=summary_writer, ignored_to_non_ignored_ratio=num_ignored_sites/len(posterior_data_loader.dataset),
        artifact_log_priors=artifact_log_priors, artifact_spectra_state_dict=artifact_spectra_state_dict)

    print("Calculating optimal logit threshold")
    error_probability_thresholds = posterior_model.calculate_probability_thresholds(posterior_data_loader, summary_writer, germline_mode=germline_mode)
    print("Optimal probability threshold: " + str(error_probability_thresholds))
    apply_filtering_to_vcf(input_vcf, output_vcf, contig_index_to_name_map, error_probability_thresholds, posterior_data_loader, posterior_model, summary_writer=summary_writer, germline_mode=germline_mode)


def make_posterior_data_loader(dataset_file, input_vcf, contig_index_to_name_map, base_model: BaseModel, artifact_model: ArtifactModel,
                               batch_size: int, chunk_size: int, segmentation=defaultdict(IntervalTree), normal_segmentation=defaultdict(IntervalTree)):
    print("Reading test dataset")

    m2_filtering_to_keep = set()
    allele_frequencies = {}

    print("recording M2 filters and allele frequencies from input VCF")
    for v in cyvcf2.VCF(input_vcf):
        encoding = encode_variant(v, zero_based=True)
        if filters_to_keep_from_m2(v):
            m2_filtering_to_keep.add(encoding)
        allele_frequencies[encoding] = 10 ** (-get_first_numeric_element(v, "POPAF"))

    # pass through the plain text dataset, normalizing and creating ReadSetDatasets as we go, running the artifact model
    # to get artifact logits, which we record in a dict keyed by variant strings.  These will later be added to PosteriorDatum objects.
    print("reading dataset and calculating artifact logits")
    posterior_data = []
    for list_of_base_data in plain_text_data.generate_normalized_data([dataset_file], chunk_size):
        raw_dataset = base_dataset.BaseDataset(data_in_ram=list_of_base_data)
        artifact_dataset = ArtifactDataset(raw_dataset, base_model)
        artifact_loader = artifact_dataset.make_data_loader(artifact_dataset.all_folds(), batch_size, pin_memory=False, num_workers=0)

        for artifact_batch in artifact_loader:
            artifact_logits = artifact_model.forward(batch=artifact_batch).detach().tolist()

            labels = ([Label.ARTIFACT if x > 0.5 else Label.VARIANT for x in artifact_batch.labels]) if artifact_batch.is_labeled() else (artifact_batch.size()*[Label.UNLABELED])
            for variant, counts_and_seq_lks, logit, label, embedding in zip(artifact_batch.variants, artifact_batch.counts_and_likelihoods, artifact_logits, labels, artifact_batch.get_representations_2d()):
                contig_name = contig_index_to_name_map[variant.contig]
                encoding = encode(contig_name, variant.position, variant.ref, variant.alt)
                if encoding in allele_frequencies and encoding not in m2_filtering_to_keep:
                    allele_frequency = allele_frequencies[encoding]

                    # these are default dicts, so if there's no segmentation for the contig we will get no overlaps but not an error
                    # For a general IntervalTree there is a list of potentially multiple overlaps but here there is either one or zero
                    segmentation_overlaps = segmentation[contig_name][variant.position]
                    normal_segmentation_overlaps = normal_segmentation[contig_name][variant.position]
                    maf = list(segmentation_overlaps)[0].data if segmentation_overlaps else 0.5
                    normal_maf = list(normal_segmentation_overlaps)[0].data if normal_segmentation_overlaps else 0.5

                    posterior_datum = PosteriorDatum(variant, counts_and_seq_lks, allele_frequency, logit, embedding, label, maf, normal_maf)
                    posterior_data.append(posterior_datum)

    print("Size of filtering dataset: " + str(len(posterior_data)))
    posterior_dataset = PosteriorDataset(posterior_data)
    return posterior_dataset.make_data_loader(batch_size)


# error probability thresholds is a dict from Variant type to error probability threshold (float)
def apply_filtering_to_vcf(input_vcf, output_vcf, contig_index_to_name_map, error_probability_thresholds, posterior_loader, posterior_model, summary_writer: SummaryWriter, germline_mode: bool = False):
    print("Computing final error probabilities")
    passing_call_type = Call.GERMLINE if germline_mode else Call.SOMATIC
    encoding_to_posterior_results = {}

    pbar = tqdm(enumerate(posterior_loader), mininterval=60)
    for n, batch in pbar:
        # posterior, along with intermediate tensors for debugging/interpretation
        log_priors, spectra_lls, normal_lls, log_posteriors = \
            posterior_model.log_posterior_and_ingredients(batch)

        posterior_probs = torch.nn.functional.softmax(log_posteriors, dim=1)

        encodings = [encode_datum(datum, contig_index_to_name_map) for datum in batch.original_list()]
        artifact_logits = [datum.artifact_logit for datum in batch.original_list()]
        var_types = [datum.variant_type for datum in batch.original_list()]
        labels = [datum.label for datum in batch.original_list()]
        alt_counts = batch.alt_counts.tolist()
        depths = batch.depths.tolist()

        for encoding, post_probs, logit, log_prior, log_spec, log_normal, label, alt_count, depth, var_type, embedding in zip(encodings, posterior_probs, artifact_logits, log_priors, spectra_lls, normal_lls, labels, alt_counts, depths, var_types, batch.embeddings):
            encoding_to_posterior_results[encoding] = PosteriorResult(logit, post_probs.tolist(), log_prior, log_spec, log_normal, label, alt_count, depth, var_type, embedding)

    print("Applying threshold")
    unfiltered_vcf = cyvcf2.VCF(input_vcf)

    all_types = [call_type.name for call_type in Call]
    unfiltered_vcf.add_format_to_header( {'ID': "DP", 'Description': "depth", 'Type': 'Integer', 'Number': '1'})
    unfiltered_vcf.add_info_to_header({'ID': POST_PROB_INFO_KEY, 'Description': 'Mutect3 posterior probability of {' + ', '.join(all_types) + '}',
                                       'Type': 'Float', 'Number': 'A'})
    unfiltered_vcf.add_info_to_header({'ID': LOG_PRIOR_INFO_KEY, 'Description': 'Log priors of {' + ', '.join(all_types) + '}',
         'Type': 'Float', 'Number': 'A'})
    unfiltered_vcf.add_info_to_header({'ID': SPECTRA_LOG_LIKELIHOOD_INFO_KEY, 'Description': 'Log spectra likelihoods of {' + ', '.join(all_types) + '}',
         'Type': 'Float', 'Number': 'A'})
    unfiltered_vcf.add_info_to_header({'ID': NORMAL_LOG_LIKELIHOOD_INFO_KEY, 'Description': 'Log normal likelihoods of {' + ', '.join(all_types) + '}',
         'Type': 'Float', 'Number': 'A'})
    unfiltered_vcf.add_info_to_header({'ID': ARTIFACT_LOD_INFO_KEY, 'Description': 'Mutect3 artifact log odds',
         'Type': 'Float', 'Number': 'A'})

    for n, filter_name in enumerate(FILTER_NAMES):
        if n != passing_call_type:
            unfiltered_vcf.add_filter_to_header({'ID': filter_name, 'Description': filter_name})

    writer = cyvcf2.Writer(output_vcf, unfiltered_vcf)  # input vcf is a template for the header
    evaluation_metrics = EvaluationMetrics()
    pbar = tqdm(enumerate(unfiltered_vcf), mininterval=60)
    labeled_truth = False
    embedding_metrics = EmbeddingMetrics() # only if there is labeled truth for evaluation

    for n, v in pbar:
        filters = filters_to_keep_from_m2(v)

        # TODO: in germline mode, somatic doesn't exist (or is just highly irrelevant) and germline is not an error!
        encoding = encode_variant(v, zero_based=True)  # cyvcf2 is zero-based
        if encoding in encoding_to_posterior_results:
            posterior_result = encoding_to_posterior_results[encoding]
            post_probs = posterior_result.posterior_probabilities
            v.INFO[POST_PROB_INFO_KEY] = ','.join(map(lambda prob: "{:.3f}".format(prob), post_probs))
            v.INFO[LOG_PRIOR_INFO_KEY] = ','.join(map(lambda pri: "{:.3f}".format(pri), posterior_result.log_priors))
            v.INFO[SPECTRA_LOG_LIKELIHOOD_INFO_KEY] = ','.join(map(lambda ll: "{:.3f}".format(ll), posterior_result.spectra_lls))
            v.INFO[ARTIFACT_LOD_INFO_KEY] = "{:.3f}".format(posterior_result.artifact_logit)
            v.INFO[NORMAL_LOG_LIKELIHOOD_INFO_KEY] = ','.join(map(lambda ll: "{:.3f}".format(ll), posterior_result.normal_lls))

            label = posterior_result.label    # this is the Label enum, might be UNLABELED
            error_prob = 1 - post_probs[passing_call_type]
            variant_type = find_variant_type(v)
            called_as_error = error_prob > error_probability_thresholds[variant_type]

            error_call = None

            if called_as_error:
                # get the error type with the largest posterior probability
                highest_prob_indices = torch.topk(torch.Tensor(post_probs), 2).indices.tolist()
                highest_prob_index = highest_prob_indices[1] if highest_prob_indices[0] == passing_call_type else highest_prob_indices[0]
                error_call = list(Call)[highest_prob_index]
                filters.add(FILTER_NAMES[highest_prob_index])

            # note that this excludes the correctness part of embedding metrics, which is below
            embedding_metrics.label_metadata.append(label.name)
            embedding_metrics.type_metadata.append(variant_type.name)
            embedding_metrics.truncated_count_metadata.append(
                str(round_up_to_nearest_three(min(MAX_COUNT, posterior_result.alt_count))))
            embedding_metrics.representations.append(posterior_result.embedding)

            if label != Label.UNLABELED:
                labeled_truth = True
                clipped_error_prob = 0.5 + 0.9999999 * (error_prob - 0.5)
                error_logit = prob_to_logit(clipped_error_prob)
                float_label = 1.0 if label == Label.ARTIFACT else 0.0
                is_correct = (called_as_error and label == Label.ARTIFACT) or (not called_as_error and label == Label.VARIANT)
                evaluation_metrics.record_call(Epoch.TEST, variant_type, error_logit, float_label, is_correct, posterior_result.alt_count)

                if not is_correct:
                    bad_call = error_call if called_as_error else Call.SOMATIC
                    evaluation_metrics.record_mistake(posterior_result, bad_call)
                embedding_metrics.correct_metadata.append("correct" if is_correct else "incorrect")
            else:
                embedding_metrics.correct_metadata.append("unknown")

        v.FILTER = ';'.join(filters) if filters else 'PASS'
        writer.write_record(v)
    print("closing resources")
    writer.close()
    unfiltered_vcf.close()

    embedding_metrics.output_to_summary_writer(summary_writer)

    if labeled_truth:
        given_thresholds = {var_type: prob_to_logit(error_probability_thresholds[var_type]) for var_type in Variation}
        evaluation_metrics.make_plots(summary_writer, given_thresholds, sens_prec=True)
        evaluation_metrics.make_mistake_histograms(summary_writer)



def main():
    args = parse_arguments()
    main_without_parsing(args)


if __name__ == '__main__':
    main()
