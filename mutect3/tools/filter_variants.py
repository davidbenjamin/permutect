import argparse
from typing import Set
from intervaltree import IntervalTree
from collections import defaultdict

import torch
from torch.utils.tensorboard import SummaryWriter
from cyvcf2 import VCF, Writer, Variant
from tqdm.autonotebook import tqdm

from torch.utils.data import DataLoader

from mutect3.architecture.artifact_model import ArtifactModel
from mutect3.architecture.posterior_model import PosteriorModel
from mutect3.data import read_set, read_set_dataset
from mutect3 import constants
from mutect3.utils import CallType

# TODO: eventually M3 can handle multiallelics
TRUSTED_M2_FILTERS = {'contamination', 'multiallelic'}

POST_PROB_INFO_KEY = 'POST'
FILTER_NAMES = [call_type.name.lower() for call_type in CallType]


# this presumes that we have an ArtifactModel and we have saved it via save_mutect3_model as in train_model.py
def load_artifact_model(path) -> ArtifactModel:
    saved = torch.load(path)
    m3_params = saved[constants.M3_PARAMS_NAME]
    model = ArtifactModel(m3_params)
    model.load_state_dict(saved[constants.STATE_DICT_NAME])
    return model


def encode(contig: str, position: int, alt: str):
    # TODO: restore the alt eventually once we handle multiallelics intelligently eg by splitting
    # return contig + ':' + str(position) + ':' + alt
    return contig + ':' + str(position)


def encode_datum(datum: read_set.ReadSet):
    return encode(datum.contig(), datum.position(), datum.alt())


def encode_variant(v: Variant, zero_based=False):
    alt = v.ALT[0]  # TODO: we're assuming biallelic
    start = (v.start + 1) if zero_based else v.start
    return encode(v.CHROM, start, alt)


def filters_to_keep_from_m2(v: Variant) -> Set[str]:
    return set([]) if v.FILTER is None else set(v.FILTER.split(";")).intersection(TRUSTED_M2_FILTERS)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--' + constants.INPUT_NAME, help='VCF from GATK', required=True)
    parser.add_argument('--' + constants.TEST_DATASET_NAME, help='test dataset file from GATK', required=True)
    parser.add_argument('--' + constants.M3_MODEL_NAME, help='trained Mutect3 model', required=True)
    parser.add_argument('--' + constants.OUTPUT_NAME, help='output filtered vcf', required=True)
    parser.add_argument('--' + constants.TENSORBOARD_DIR_NAME, type=str, default='tensorboard', required=False)
    parser.add_argument('--' + constants.BATCH_SIZE_NAME, type=int, default=64, required=False)
    parser.add_argument('--' + constants.NUM_SPECTRUM_ITERATIONS, type=int, default=10, required=False)
    parser.add_argument('--' + constants.INITIAL_LOG_VARIANT_PRIOR_NAME, type=float, default=-10.0, required=False)
    parser.add_argument('--' + constants.INITIAL_LOG_ARTIFACT_PRIOR_NAME, type=float, default=-10.0, required=False)
    parser.add_argument('--' + constants.NUM_IGNORED_SITES_NAME, type=float, required=True)
    parser.add_argument('--' + constants.MAF_SEGMENTS_NAME, required=False)
    parser.add_argument('--' + constants.GERMLINE_MODE_NAME, action='store_true')
    return parser.parse_args()


def get_segmentation(segments_file) -> defaultdict:
    result = defaultdict(IntervalTree)
    if segments_file is None:
        return result

    with open(segments_file, 'r') as file:
        for line in file:
            if line.startswith("#") or (line.startswith("contig") and line.endswith("minor_allele_fraction")):
                continue
            tokens = line.split()
            contig, start, stop, maf = tokens[0], int(tokens[1]), int(tokens[2]), float(tokens[3])
            result[contig][start:stop] = maf

    return result


def main():
    args = parse_arguments()

    make_filtered_vcf(saved_artifact_model=getattr(args, constants.M3_MODEL_NAME),
                      initial_log_variant_prior=getattr(args, constants.INITIAL_LOG_VARIANT_PRIOR_NAME),
                      initial_log_artifact_prior=getattr(args, constants.INITIAL_LOG_ARTIFACT_PRIOR_NAME),
                      test_dataset_file=getattr(args, constants.TEST_DATASET_NAME),
                      input_vcf=getattr(args, constants.INPUT_NAME),
                      output_vcf=getattr(args, constants.OUTPUT_NAME),
                      batch_size=getattr(args, constants.BATCH_SIZE_NAME),
                      num_spectrum_iterations=getattr(args, constants.NUM_SPECTRUM_ITERATIONS),
                      tensorboard_dir=getattr(args, constants.TENSORBOARD_DIR_NAME),
                      num_ignored_sites=getattr(args, constants.NUM_IGNORED_SITES_NAME),
                      maf_segments=getattr(args, constants.MAF_SEGMENTS_NAME),
                      germline_mode=getattr(args, constants.GERMLINE_MODE_NAME),
                      segmentation=get_segmentation(getattr(args, constants.MAF_SEGMENTS_NAME)))


def make_filtered_vcf(saved_artifact_model, initial_log_variant_prior: float, initial_log_artifact_prior: float,
                      test_dataset_file, input_vcf, output_vcf, batch_size: int, num_spectrum_iterations: int, tensorboard_dir,
                      num_ignored_sites: int, maf_segments, germline_mode: bool, segmentation=defaultdict(IntervalTree)):
    print("Loading artifact model and test dataset")
    artifact_model = load_artifact_model(saved_artifact_model)
    posterior_model = PosteriorModel(artifact_model, initial_log_variant_prior, initial_log_artifact_prior, segmentation=segmentation)
    filtering_data_loader = make_filtering_data_loader(test_dataset_file, input_vcf, batch_size)

    print("Learning AF spectra")
    summary_writer = SummaryWriter(tensorboard_dir)
    posterior_model.learn_priors_and_spectra(filtering_data_loader, num_iterations=num_spectrum_iterations,
        summary_writer=summary_writer, ignored_to_non_ignored_ratio=num_ignored_sites/len(filtering_data_loader.dataset))

    print("Calculating optimal logit threshold")
    error_probability_threshold = posterior_model.calculate_probability_threshold(filtering_data_loader, summary_writer, germline_mode=germline_mode)
    print("Optimal probability threshold: " + str(error_probability_threshold))
    apply_filtering_to_vcf(input_vcf, output_vcf, error_probability_threshold, filtering_data_loader, posterior_model, germline_mode=germline_mode)


def make_filtering_data_loader(dataset_file, input_vcf, batch_size: int) -> DataLoader:
    print("Reading test dataset")
    unfiltered_test_data = read_set_dataset.read_data(dataset_file)
    # record variants that M2 filtered as germline or contamination.  Mutect3 ignores these
    m2_filtering_to_keep = set([encode_variant(v, zero_based=True) for v in VCF(input_vcf) if
                                filters_to_keep_from_m2(v)])

    allele_frequencies = {encode_variant(v, zero_based=True): 10 ** (-v.INFO["POPAF"]) for v in VCF(input_vcf)}

    # choose which variants to proceed to M3 -- those that M2 didn't filter as germline or contamination
    filtering_variants = []
    for datum in unfiltered_test_data:
        encoding = encode_datum(datum)
        if encoding not in m2_filtering_to_keep:
            datum.set_allele_frequency(allele_frequencies[encoding])
            filtering_variants.append(datum)
    print("Size of filtering dataset: " + str(len(filtering_variants)))
    filtering_dataset = read_set_dataset.ReadSetDataset(data=filtering_variants)
    filtering_data_loader = read_set_dataset.make_test_data_loader(filtering_dataset, batch_size)
    return filtering_data_loader


def apply_filtering_to_vcf(input_vcf, output_vcf, error_probability_threshold, filtering_data_loader, posterior_model, germline_mode: bool = False):
    print("Computing final error probabilities")
    passing_call_type = CallType.GERMLINE if germline_mode else CallType.SOMATIC
    encoding_to_post_prob_dict = {}
    pbar = tqdm(enumerate(filtering_data_loader), mininterval=10)
    for n, batch in pbar:
        posterior_probs = posterior_model.posterior_probabilities(batch)
        encodings = [encode_datum(datum) for datum in batch.original_list()]
        for encoding, post_probs in zip(encodings, posterior_probs):
            encoding_to_post_prob_dict[encoding] = post_probs.tolist()
    print("Applying threshold")
    unfiltered_vcf = VCF(input_vcf)

    all_types = [call_type.name for call_type in CallType]
    unfiltered_vcf.add_info_to_header({'ID': POST_PROB_INFO_KEY, 'Description': 'Mutect3 posterior probability of {' + ', '.join(all_types) + '}',
                                       'Type': 'Float', 'Number': 'A'})

    for n, filter_name in enumerate(FILTER_NAMES):
        if n != passing_call_type:
            unfiltered_vcf.add_filter_to_header({'ID': filter_name, 'Description': filter_name})

    writer = Writer(output_vcf, unfiltered_vcf)  # input vcf is a template for the header
    pbar = tqdm(enumerate(unfiltered_vcf), mininterval=10)
    for n, v in pbar:
        filters = filters_to_keep_from_m2(v)

        # TODO: in germline mode, somatic doesn't exist (or is just highly irrelevant) and germline is not an error!
        encoding = encode_variant(v, zero_based=True)  # cyvcf2 is zero-based
        if encoding in encoding_to_post_prob_dict:
            post_probs = encoding_to_post_prob_dict[encoding]
            v.INFO[POST_PROB_INFO_KEY] = post_probs

            error_prob = 1 - post_probs[passing_call_type]
            if error_prob > error_probability_threshold:
                # get the error type with the largest posterior probability
                highest_prob_indices = torch.topk(post_probs, 2).indices.tolist()
                highest_prob_index = highest_prob_indices[1] if highest_prob_indices[0] == passing_call_type else highest_prob_indices[0]
                filters.add(FILTER_NAMES[highest_prob_index])

        v.FILTER = ';'.join(filters) if filters else 'PASS'
        writer.write_record(v)
    print("closing resources")
    writer.close()
    unfiltered_vcf.close()


if __name__ == '__main__':
    main()
