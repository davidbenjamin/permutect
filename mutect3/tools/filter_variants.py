import argparse
import math
from typing import Set
from intervaltree import IntervalTree
from collections import defaultdict
import psutil

import torch
from torch.utils.tensorboard import SummaryWriter
import cyvcf2
from tqdm.autonotebook import tqdm

from mutect3.architecture.artifact_model import ArtifactModel
from mutect3.architecture.posterior_model import PosteriorModel
from mutect3.data import read_set_dataset
from mutect3 import constants
from mutect3.data.posterior_dataset import PosteriorDataset
from mutect3.data.posterior_datum import PosteriorDatum
from mutect3.utils import Call

TRUSTED_M2_FILTERS = {'contamination'}

POST_PROB_INFO_KEY = 'POST'
FILTER_NAMES = [call_type.name.lower() for call_type in Call]

CHUNK_SIZE = 100000


# this presumes that we have an ArtifactModel and we have saved it via save_mutect3_model as in train_model.py
def load_artifact_model(path) -> ArtifactModel:
    saved = torch.load(path)
    m3_params = saved[constants.M3_PARAMS_NAME]
    num_read_features = saved[constants.NUM_READ_FEATURES_NAME]
    num_info_features = saved[constants.NUM_INFO_FEATURES_NAME]
    ref_sequence_length = saved[constants.REF_SEQUENCE_LENGTH_NAME]
    model = ArtifactModel(m3_params, num_read_features=num_read_features, num_info_features=num_info_features, ref_sequence_length=ref_sequence_length)
    model.load_state_dict(saved[constants.STATE_DICT_NAME])
    return model


def encode(contig: str, position: int, alt: str):
    # TODO: restore the alt eventually once we handle multiallelics intelligently eg by splitting
    # return contig + ':' + str(position) + ':' + alt
    return contig + ':' + str(position)


def encode_datum(datum: PosteriorDatum):
    return encode(datum.contig(), datum.position(), datum.alt())


def encode_variant(v: cyvcf2.Variant, zero_based=False):
    alt = v.ALT[0]  # TODO: we're assuming biallelic
    start = (v.start + 1) if zero_based else v.start
    return encode(v.CHROM, start, alt)


def get_first_numeric_element(variant, key):
    tuple_or_scalar = variant.INFO[key]
    return tuple_or_scalar[0] if type(tuple_or_scalar) is tuple else tuple_or_scalar


def filters_to_keep_from_m2(v: cyvcf2.Variant) -> Set[str]:
    return set([]) if v.FILTER is None else set(v.FILTER.split(";")).intersection(TRUSTED_M2_FILTERS)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--' + constants.INPUT_NAME, help='VCF from GATK', required=True)
    parser.add_argument('--' + constants.TEST_DATASET_NAME, help='test dataset file from GATK', required=True)
    parser.add_argument('--' + constants.M3_MODEL_NAME, help='trained Mutect3 model', required=True)
    parser.add_argument('--' + constants.OUTPUT_NAME, help='output filtered vcf', required=True)
    parser.add_argument('--' + constants.TENSORBOARD_DIR_NAME, type=str, default='tensorboard', required=False)
    parser.add_argument('--' + constants.BATCH_SIZE_NAME, type=int, default=64, required=False)
    parser.add_argument('--' + constants.CHUNK_SIZE_NAME, type=int, default=100000, required=False)
    parser.add_argument('--' + constants.NUM_SPECTRUM_ITERATIONS, type=int, default=10, required=False)
    parser.add_argument('--' + constants.INITIAL_LOG_VARIANT_PRIOR_NAME, type=float, default=-10.0, required=False)
    parser.add_argument('--' + constants.INITIAL_LOG_ARTIFACT_PRIOR_NAME, type=float, default=-10.0, required=False)
    parser.add_argument('--' + constants.NUM_IGNORED_SITES_NAME, type=float, required=True)
    parser.add_argument('--' + constants.MAF_SEGMENTS_NAME, required=False)
    parser.add_argument('--' + constants.NORMAL_MAF_SEGMENTS_NAME, required=False)
    parser.add_argument('--' + constants.GERMLINE_MODE_NAME, action='store_true')
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
                      chunk_size=getattr(args, constants.CHUNK_SIZE_NAME),
                      num_spectrum_iterations=getattr(args, constants.NUM_SPECTRUM_ITERATIONS),
                      tensorboard_dir=getattr(args, constants.TENSORBOARD_DIR_NAME),
                      num_ignored_sites=getattr(args, constants.NUM_IGNORED_SITES_NAME),
                      germline_mode=getattr(args, constants.GERMLINE_MODE_NAME),
                      segmentation=get_segmentation(getattr(args, constants.MAF_SEGMENTS_NAME)),
                      normal_segmentation=get_segmentation(getattr(args, constants.NORMAL_MAF_SEGMENTS_NAME)))


def make_filtered_vcf(saved_artifact_model, initial_log_variant_prior: float, initial_log_artifact_prior: float,
                      test_dataset_file, input_vcf, output_vcf, batch_size: int, chunk_size: int, num_spectrum_iterations: int, tensorboard_dir,
                      num_ignored_sites: int, germline_mode: bool = False, segmentation=defaultdict(IntervalTree),
                      normal_segmentation=defaultdict(IntervalTree)):
    print("Loading artifact model and test dataset")
    artifact_model = load_artifact_model(saved_artifact_model)
    posterior_model = PosteriorModel(initial_log_variant_prior, initial_log_artifact_prior, segmentation=segmentation, normal_segmentation=normal_segmentation)
    posterior_data_loader = make_posterior_data_loader(test_dataset_file, input_vcf, artifact_model, batch_size, chunk_size=chunk_size)

    print("Learning AF spectra")
    summary_writer = SummaryWriter(tensorboard_dir)

    # TODO: filtering data loader is now a filtering dataset!!!
    posterior_model.learn_priors_and_spectra(posterior_data_loader, num_iterations=num_spectrum_iterations,
        summary_writer=summary_writer, ignored_to_non_ignored_ratio=num_ignored_sites/len(posterior_data_loader.dataset))

    print("Calculating optimal logit threshold")
    error_probability_threshold = posterior_model.calculate_probability_threshold(posterior_data_loader, summary_writer, germline_mode=germline_mode)
    print("Optimal probability threshold: " + str(error_probability_threshold))
    apply_filtering_to_vcf(input_vcf, output_vcf, error_probability_threshold, posterior_data_loader, posterior_model, germline_mode=germline_mode)


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

    # TODO chunk size should be command line argument
    num_data = read_set_dataset.count_data(dataset_file)
    num_chunks = math.ceil(num_data / chunk_size)

    print("preparing dataset to pass to Mutect3 posterior probability model")
    read_sets_buffer = []
    posterior_buffer = []
    posterior_data = []

    data_count = 0
    chunk_count = 0
    for read_set, posterior_datum in read_set_dataset.read_data(dataset_file, posterior=True, yield_nones=True):
        data_count += 1

        if read_set is not None:
            encoding = encode_datum(posterior_datum)
            if encoding not in m2_filtering_to_keep:
                posterior_datum.set_allele_frequency(allele_frequencies[encoding])
                posterior_buffer.append(posterior_datum)
                read_sets_buffer.append(read_set)

        if data_count == math.ceil(num_data * (chunk_count + 1) / num_chunks):
            print("memory usage percent: " + str(psutil.virtual_memory().percent))
            print(posterior_buffer[-1].contig() + ":" + str(posterior_buffer[-1].position()))

            artifact_dataset = read_set_dataset.ReadSetDataset(data=read_sets_buffer)
            logits = []
            for artifact_batch in read_set_dataset.make_test_data_loader(artifact_dataset, batch_size):
                logits.extend(artifact_model.forward(batch=artifact_batch).detach().tolist())
            for logit, posterior in zip(logits, posterior_buffer):
                posterior.set_artifact_logit(logit)
            posterior_data.extend(posterior_buffer)

            chunk_count += 1
            read_sets_buffer = []
            posterior_buffer = []

    assert data_count == num_data and chunk_count == num_chunks and len(read_sets_buffer) == 0  # should be nothing left over
    print("Size of filtering dataset: " + str(len(posterior_data)))
    posterior_dataset = PosteriorDataset(posterior_data)
    return posterior_dataset.make_data_loader(batch_size)


def apply_filtering_to_vcf(input_vcf, output_vcf, error_probability_threshold, posterior_loader, posterior_model, germline_mode: bool = False):
    print("Computing final error probabilities")
    passing_call_type = Call.GERMLINE if germline_mode else Call.SOMATIC
    encoding_to_post_prob_dict = {}
    pbar = tqdm(enumerate(posterior_loader), mininterval=10)
    for n, batch in pbar:
        posterior_probs = posterior_model.posterior_probabilities(batch)

        encodings = [encode_datum(datum) for datum in batch.original_list()]
        for encoding, post_probs in zip(encodings, posterior_probs):
            encoding_to_post_prob_dict[encoding] = post_probs.tolist()
    print("Applying threshold")
    unfiltered_vcf = cyvcf2.VCF(input_vcf)

    all_types = [call_type.name for call_type in Call]
    unfiltered_vcf.add_info_to_header({'ID': POST_PROB_INFO_KEY, 'Description': 'Mutect3 posterior probability of {' + ', '.join(all_types) + '}',
                                       'Type': 'Float', 'Number': 'A'})

    for n, filter_name in enumerate(FILTER_NAMES):
        if n != passing_call_type:
            unfiltered_vcf.add_filter_to_header({'ID': filter_name, 'Description': filter_name})

    writer = cyvcf2.Writer(output_vcf, unfiltered_vcf)  # input vcf is a template for the header
    pbar = tqdm(enumerate(unfiltered_vcf), mininterval=10)
    for n, v in pbar:
        filters = filters_to_keep_from_m2(v)

        # TODO: in germline mode, somatic doesn't exist (or is just highly irrelevant) and germline is not an error!
        encoding = encode_variant(v, zero_based=True)  # cyvcf2 is zero-based
        if encoding in encoding_to_post_prob_dict:
            post_probs = encoding_to_post_prob_dict[encoding]
            v.INFO[POST_PROB_INFO_KEY] = ','.join(map(lambda prob: "{:.4f}".format(prob), post_probs))

            error_prob = 1 - post_probs[passing_call_type]
            if error_prob > error_probability_threshold:
                # get the error type with the largest posterior probability
                highest_prob_indices = torch.topk(torch.Tensor(post_probs), 2).indices.tolist()
                highest_prob_index = highest_prob_indices[1] if highest_prob_indices[0] == passing_call_type else highest_prob_indices[0]
                filters.add(FILTER_NAMES[highest_prob_index])

        v.FILTER = ';'.join(filters) if filters else 'PASS'
        writer.write_record(v)
    print("closing resources")
    writer.close()
    unfiltered_vcf.close()


if __name__ == '__main__':
    main()
