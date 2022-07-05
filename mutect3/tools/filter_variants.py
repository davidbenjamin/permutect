import argparse
from typing import Set

import torch
from torch.utils.tensorboard import SummaryWriter
from cyvcf2 import VCF, Writer, Variant
from tqdm.autonotebook import tqdm

from mutect3.architecture.artifact_model import ArtifactModel
from mutect3.architecture.posterior_model import PosteriorModel
from mutect3.data import read_set, read_set_dataset
from mutect3 import constants

# TODO: eventually M3 can handle multiallelics
TRUSTED_M2_FILTERS = {'contamination', 'germline', 'multiallelic'}


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


def main():
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
    args = parser.parse_args()

    # record variants that M2 filtered as germline or contamination.  Mutect3 ignores these
    m2_filtering_to_keep = set([encode_variant(v, zero_based=True) for v in VCF(getattr(args, constants.INPUT_NAME)) if filters_to_keep_from_m2(v)])

    print("Loading artifact model")
    artifact_model = load_artifact_model(getattr(args, constants.M3_MODEL_NAME))
    posterior_model = PosteriorModel(artifact_model, variant_log_prior=getattr(args, constants.INITIAL_LOG_VARIANT_PRIOR_NAME),
                                     artifact_log_prior=getattr(args, constants.INITIAL_LOG_ARTIFACT_PRIOR_NAME))

    print("Reading test dataset")
    unfiltered_test_data = read_set_dataset.read_data(getattr(args, constants.TEST_DATASET_NAME))

    # TODO: START SILLY STUFF
    # THIS IS SOME RIDICULOUS ONE-OFF STUFF TO CHECK THE LIKELIHOODS MODEL FOR SINGULAR
    #all_data_loader = read_set_dataset.make_test_data_loader(unfiltered_test_data, getattr(args, constants.BATCH_SIZE_NAME))

    #print("Calculating all the logits")
    #pbar = tqdm(enumerate(all_data_loader), mininterval=10)
    #for n, batch in pbar:
    #    logits = artifact_model.forward(batch, posterior=False).detach()
    #    encodings = [encode_datum(datum) for datum in batch.original_list()]
    #    for encoding, logit in zip(encodings, logits):
    #        print(encoding + ": " + str(logit.item()))

    # TODO: END SILLY STUFF

    # choose which variants to proceed to M3 -- those that M2 didn't filter as germline or contamination
    filtering_variants = []
    for datum in unfiltered_test_data:
        encoding = encode_datum(datum)
        if encoding not in m2_filtering_to_keep:
            filtering_variants.append(datum)

    print("Size of filtering dataset: " + str(len(filtering_variants)))
    filtering_dataset = read_set_dataset.ReadSetDataset(data=filtering_variants)
    filtering_data_loader = read_set_dataset.make_test_data_loader(filtering_dataset, getattr(args, constants.BATCH_SIZE_NAME))

    print("Learning AF spectra")
    num_spectrum_iterations = getattr(args, constants.NUM_SPECTRUM_ITERATIONS)
    summary_writer = SummaryWriter(getattr(args, constants.TENSORBOARD_DIR_NAME))
    posterior_model.learn_priors_and_spectra(filtering_data_loader, num_iterations=num_spectrum_iterations,
        summary_writer=summary_writer, ignored_to_non_ignored_ratio=getattr(args, constants.NUM_IGNORED_SITES_NAME)/len(filtering_variants))

    print("Calculating optimal logit threshold")
    error_probability_threshold = posterior_model.calculate_probability_threshold(filtering_data_loader, summary_writer)
    print("Optimal probability threshold: " + str(error_probability_threshold))

    print("Computing final error probabilities")
    encoding_to_error_prob_dict = {}
    pbar = tqdm(enumerate(filtering_data_loader), mininterval=10)
    for n, batch in pbar:
        error_probs = posterior_model.error_probabilities(batch)
        encodings = [encode_datum(datum) for datum in batch.original_list()]
        for encoding, error_prob in zip(encodings, error_probs):
            encoding_to_error_prob_dict[encoding] = error_prob.item()

    print("Applying threshold")
    unfiltered_vcf = VCF(getattr(args, constants.INPUT_NAME))
    unfiltered_vcf.add_info_to_header({'ID': 'ERROR_PROB', 'Description': 'Mutect3 posterior error probability',
                            'Type': 'Float', 'Number': 'A'})
    unfiltered_vcf.add_filter_to_header({'ID': 'mutect3', 'Description': 'fails Mutect3 deep learning filter'})

    writer = Writer(getattr(args, constants.OUTPUT_NAME), unfiltered_vcf)  # input vcf is a template for the header

    pbar = tqdm(enumerate(unfiltered_vcf), mininterval=10)
    for n, v in pbar:
        filters = filters_to_keep_from_m2(v)

        encoding = encode_variant(v, zero_based=True)   # cyvcf2 is zero-based
        if encoding in encoding_to_error_prob_dict:
            error_prob = encoding_to_error_prob_dict[encoding]
            v.INFO["ERROR_PROB"] = error_prob

            # TODO: distinguish between artifact and weak evidence
            if error_prob > error_probability_threshold:
                filters.add("mutect3")

        v.FILTER = ';'.join(filters) if filters else 'PASS'
        writer.write_record(v)

    print("closing resources")
    writer.close()
    unfiltered_vcf.close()


if __name__ == '__main__':
    main()
