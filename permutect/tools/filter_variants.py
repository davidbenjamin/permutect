import argparse
from collections import defaultdict
from typing import Set

import cyvcf2
import torch
from intervaltree import IntervalTree
from torch.utils.tensorboard import SummaryWriter
from tqdm.autonotebook import tqdm

from permutect import constants
from permutect.architecture.artifact_model import ArtifactModel
from permutect.architecture.posterior_model import PosteriorModel
from permutect.data import read_set_dataset, plain_text_data
from permutect.data.posterior import PosteriorDataset, PosteriorDatum
from permutect.utils import Call, find_variant_type, Label

TRUSTED_M2_FILTERS = {'contamination'}

POST_PROB_INFO_KEY = 'POST'
ARTIFACT_LOD_INFO_KEY = 'ARTLOD'
LOG_PRIOR_INFO_KEY = 'PRIOR'
SPECTRA_LOG_LIKELIHOOD_INFO_KEY = 'SPECLL'
NORMAL_LOG_LIKELIHOOD_INFO_KEY = 'NORMLL'

FILTER_NAMES = [call_type.name.lower() for call_type in Call]


# this presumes that we have an ArtifactModel and we have saved it via save_permutect_model as in train_model.py
# it includes log artifact priors and artifact spectra, but these may be None
def load_artifact_model(path) -> ArtifactModel:
    saved = torch.load(path)
    hyperparams = saved[constants.HYPERPARAMS_NAME]
    num_read_features = saved[constants.NUM_READ_FEATURES_NAME]
    num_info_features = saved[constants.NUM_INFO_FEATURES_NAME]
    ref_sequence_length = saved[constants.REF_SEQUENCE_LENGTH_NAME]

    model = ArtifactModel(hyperparams, num_read_features=num_read_features, num_info_features=num_info_features, ref_sequence_length=ref_sequence_length)
    model.load_state_dict(saved[constants.STATE_DICT_NAME])

    artifact_log_priors = saved[constants.ARTIFACT_LOG_PRIORS_NAME]     # possibly None
    artifact_spectra_state_dict = saved[constants.ARTIFACT_SPECTRA_STATE_DICT_NAME]     #possibly None
    return model, artifact_log_priors, artifact_spectra_state_dict


def get_first_numeric_element(variant, key):
    tuple_or_scalar = variant.INFO[key]
    return tuple_or_scalar[0] if type(tuple_or_scalar) is tuple else tuple_or_scalar


# if alt and ref alleles are not in minimal representation ie have redundant matching bases at the end, trim them
def trim_alleles_on_right(ref: str, alt: str):
    trimmed_ref, trimmed_alt = ref, alt
    while len(ref) > 1 and len(alt) > 1 and trimmed_alt[-1] == trimmed_ref[-1]:
        trimmed_ref, trimmed_alt = trimmed_ref[:-1], trimmed_alt[:-1]
    return trimmed_ref, trimmed_alt


def encode(contig: str, position: int, ref: str, alt: str):
    trimmed_ref, trimmed_alt = trim_alleles_on_right(ref, alt)
    return contig + ':' + str(position) + ':' + trimmed_alt


def encode_datum(datum: PosteriorDatum):
    return encode(datum.contig, datum.position, datum.ref, datum.alt)


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
    make_filtered_vcf(saved_artifact_model=getattr(args, constants.M3_MODEL_NAME),
                      initial_log_variant_prior=getattr(args, constants.INITIAL_LOG_VARIANT_PRIOR_NAME),
                      initial_log_artifact_prior=getattr(args, constants.INITIAL_LOG_ARTIFACT_PRIOR_NAME),
                      test_dataset_file=getattr(args, constants.TEST_DATASET_NAME),
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


def make_filtered_vcf(saved_artifact_model, initial_log_variant_prior: float, initial_log_artifact_prior: float,
                      test_dataset_file, input_vcf, output_vcf, batch_size: int, chunk_size: int, num_spectrum_iterations: int, tensorboard_dir,
                      genomic_span: int, germline_mode: bool = False, no_germline_mode: bool = False, segmentation=defaultdict(IntervalTree),
                      normal_segmentation=defaultdict(IntervalTree)):
    print("Loading artifact model and test dataset")
    artifact_model, artifact_log_priors, artifact_spectra_state_dict = load_artifact_model(saved_artifact_model)
    posterior_model = PosteriorModel(initial_log_variant_prior, initial_log_artifact_prior, segmentation=segmentation, normal_segmentation=normal_segmentation, no_germline_mode=no_germline_mode)
    posterior_data_loader = make_posterior_data_loader(test_dataset_file, input_vcf, artifact_model, batch_size, chunk_size=chunk_size)

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
    apply_filtering_to_vcf(input_vcf, output_vcf, error_probability_thresholds, posterior_data_loader, posterior_model, germline_mode=germline_mode)

    # TODO: if the posterior data have truth labels, the summary writer should now generate an analysis of the calls,
    # TODO: similar to how we evaluate a model after training


def make_posterior_data_loader(dataset_file, input_vcf, artifact_model: ArtifactModel, batch_size: int, chunk_size: int):
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
    for list_of_read_sets in plain_text_data.generate_normalized_data([dataset_file], chunk_size):
        artifact_dataset = read_set_dataset.ReadSetDataset(data_in_ram=list_of_read_sets)
        artifact_loader = read_set_dataset.make_data_loader(artifact_dataset, artifact_dataset.all_folds(), batch_size, pin_memory=False, num_workers=0)

        for artifact_batch in artifact_loader:
            artifact_logits = artifact_model.forward(batch=artifact_batch).detach().tolist()

            labels = ([Label.ARTIFACT if x > 0.5 else Label.VARIANT for x in artifact_batch.labels]) if artifact_batch.is_labeled() else (artifact_batch.size()*[Label.UNLABELED])
            for variant, counts_and_seq_lks, index, logit, label in zip(artifact_batch.variants, artifact_batch.counts_and_likelihoods, artifact_batch.indices, artifact_logits, labels):
                encoding = encode(variant.contig, variant.position, variant.ref, variant.alt)
                if encoding in allele_frequencies and encoding not in m2_filtering_to_keep:
                    allele_frequency = allele_frequencies[encoding]
                    posterior_datum = PosteriorDatum(variant, counts_and_seq_lks, index, allele_frequency, logit)
                    posterior_data.append(posterior_datum)

    print("Size of filtering dataset: " + str(len(posterior_data)))
    posterior_dataset = PosteriorDataset(posterior_data)
    return posterior_dataset.make_data_loader(batch_size)


# error probability thresholds is a dict from Variant type to error probability threshold (float)
def apply_filtering_to_vcf(input_vcf, output_vcf, error_probability_thresholds, posterior_loader, posterior_model, germline_mode: bool = False):
    print("Computing final error probabilities")
    passing_call_type = Call.GERMLINE if germline_mode else Call.SOMATIC
    encoding_to_post_prob = {}
    encoding_to_artifact_logit = {}
    encoding_to_log_priors = {}
    encoding_to_spectra_lls = {}
    encoding_to_normal_lls = {}
    encoding_to_labels = {}
    pbar = tqdm(enumerate(posterior_loader), mininterval=60)
    for n, batch in pbar:
        # posterior, along with intermediate tensors for debugging/interpretation
        log_priors, spectra_lls, normal_lls, log_posteriors = \
            posterior_model.log_posterior_and_ingredients(batch)

        posterior_probs = torch.nn.functional.softmax(log_posteriors, dim=1)

        encodings = [encode_datum(datum) for datum in batch.original_list()]
        artifact_logits = [datum.artifact_logit for datum in batch.original_list()]
        labels = [datum.label for datum in batch.original_list()]

        for encoding, post_probs, logit, log_prior, log_spec, log_normal, label in zip(encodings, posterior_probs, artifact_logits, log_priors, spectra_lls, normal_lls, labels):
            encoding_to_post_prob[encoding] = post_probs.tolist()
            encoding_to_artifact_logit[encoding] = logit
            encoding_to_log_priors[encoding] = log_prior
            encoding_to_spectra_lls[encoding] = log_spec
            encoding_to_normal_lls[encoding] = log_normal
            encoding_to_labels[encoding] = label

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
    pbar = tqdm(enumerate(unfiltered_vcf), mininterval=60)
    for n, v in pbar:
        filters = filters_to_keep_from_m2(v)

        # TODO: in germline mode, somatic doesn't exist (or is just highly irrelevant) and germline is not an error!
        encoding = encode_variant(v, zero_based=True)  # cyvcf2 is zero-based
        if encoding in encoding_to_post_prob:
            post_probs = encoding_to_post_prob[encoding]
            v.INFO[POST_PROB_INFO_KEY] = ','.join(map(lambda prob: "{:.3f}".format(prob), post_probs))
            v.INFO[LOG_PRIOR_INFO_KEY] = ','.join(map(lambda pri: "{:.3f}".format(pri), encoding_to_log_priors[encoding]))
            v.INFO[SPECTRA_LOG_LIKELIHOOD_INFO_KEY] = ','.join(map(lambda ll: "{:.3f}".format(ll), encoding_to_spectra_lls[encoding]))
            v.INFO[ARTIFACT_LOD_INFO_KEY] = "{:.3f}".format(encoding_to_artifact_logit[encoding])
            v.INFO[NORMAL_LOG_LIKELIHOOD_INFO_KEY] = ','.join(map(lambda ll: "{:.3f}".format(ll), encoding_to_normal_lls[encoding]))

            error_prob = 1 - post_probs[passing_call_type]
            # TODO: threshold by variant type
            if error_prob > error_probability_thresholds[find_variant_type(v)]:
                # get the error type with the largest posterior probability
                highest_prob_indices = torch.topk(torch.Tensor(post_probs), 2).indices.tolist()
                highest_prob_index = highest_prob_indices[1] if highest_prob_indices[0] == passing_call_type else highest_prob_indices[0]
                filters.add(FILTER_NAMES[highest_prob_index])

        v.FILTER = ';'.join(filters) if filters else 'PASS'
        writer.write_record(v)
    print("closing resources")
    writer.close()
    unfiltered_vcf.close()


def main():
    args = parse_arguments()
    main_without_parsing(args)


if __name__ == '__main__':
    main()
