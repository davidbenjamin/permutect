import argparse
from typing import Set

import torch
from torch.utils.tensorboard import SummaryWriter
from cyvcf2 import VCF, Writer, Variant
from tqdm.autonotebook import tqdm

import mutect3.architecture.read_set_classifier
from mutect3.data import read_set_datum, read_set_dataset
from mutect3 import constants

# TODO: eventually M3 can handle multiallelics
TRUSTED_M2_FILTERS = {'contamination', 'germline', 'weak_evidence', 'multiallelic'}


# this presumes that we have a ReadSetClassifier model and we have saved it via save_mutect3_model as in train_model.py
def load_m3_model(path):
    saved = torch.load(path)
    m3_params = saved[constants.M3_PARAMS_NAME]
    model = mutect3.architecture.read_set_classifier.ReadSetClassifier(m3_params)
    model.load_state_dict(saved[constants.STATE_DICT_NAME])
    return model


def encode(contig: str, position: int, alt: str):
    # TODO: restore the alt eventually once we handle multiallelics intelligently eg by splitting
    # return contig + ':' + str(position) + ':' + alt
    return contig + ':' + str(position)


def encode_datum(datum: read_set_datum.ReadSetDatum):
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
    parser.add_argument('--' + constants.NA_MODEL_NAME, help='trained normal artifact model', required=True)
    parser.add_argument('--' + constants.OUTPUT_NAME, help='output filtered vcf', required=True)
    parser.add_argument('--' + constants.TENSORBOARD_DIR_NAME, type=str, default='tensorboard', required=False)
    parser.add_argument('--' + constants.BATCH_SIZE_NAME, type=int, default=64, required=False)
    parser.add_argument('--' + constants.NUM_SPECTRUM_ITERATIONS, type=int, default=10, required=False)
    args = parser.parse_args()

    # record encodings of variants that M2 filtered as germline, contamination, or weak evidence
    # Mutect3 ignores these
    m2_filtering_to_keep = set([encode_variant(v, zero_based=True) for v in VCF(getattr(args, constants.INPUT_NAME)) if filters_to_keep_from_m2(v)])

    print("Loading model")
    model = load_m3_model(getattr(args, constants.M3_MODEL_NAME))

    print("Reading test dataset")
    unfiltered_test_data = read_set_dataset.read_data(getattr(args, constants.TEST_DATASET_NAME))

    # TODO: START SILLY STUFF
    # THIS IS SOME RIDICULOUS ONE-OFF STUFF TO CHECK THE LIKELIHOODS MODEL FOR SINGULAR
    all_data_loader = read_set_dataset.make_test_data_loader(unfiltered_test_data, getattr(args, constants.BATCH_SIZE_NAME))

    print("Calculating all the logits")
    pbar = tqdm(enumerate(all_data_loader), mininterval=10)
    for n, batch in pbar:
        logits = model.forward(batch, posterior=False)
        encodings = [encode_datum(datum) for datum in batch.original_list()]
        for encoding, logit in zip(encodings, logits):
            print(encoding + ": " + str(logit))

    # TODO: END SILLY STUFF

    m3_variants = []
    for datum in unfiltered_test_data:
        encoding = encode_datum(datum)
        if encoding not in m2_filtering_to_keep:
            m3_variants.append(datum)

    print("Size of test dataset: " + str(len(m3_variants)))

    dataset = read_set_dataset.ReadSetDataset(data=m3_variants)
    data_loader = read_set_dataset.make_test_data_loader(dataset, getattr(args, constants.BATCH_SIZE_NAME))


    # The AF spectrum was, of course, not pre-trained with the rest of the model
    print("Learning AF spectra")
    num_spectrum_iterations = getattr(args, constants.NUM_SPECTRUM_ITERATIONS)
    summary_writer = SummaryWriter(getattr(args, constants.TENSORBOARD_DIR_NAME))
    model.learn_spectra(data_loader, num_iterations=num_spectrum_iterations, summary_writer=summary_writer)

    print("Calculating optimal logit threshold")
    logit_threshold = model.calculate_logit_threshold(loader=data_loader, summary_writer=summary_writer)
    print("Optimal logit threshold: " + str(logit_threshold))

    encoding_to_logit_dict = {}

    print("Running final calls")
    pbar = tqdm(enumerate(data_loader), mininterval=10)
    for n, batch in pbar:
        logits = model.forward(batch, posterior=True)

        encodings = [encode_datum(datum) for datum in batch.original_list()]
        for encoding, logit in zip(encodings, logits):
            encoding_to_logit_dict[encoding] = logit.item()

    print("Applying computed logits")

    unfiltered_vcf = VCF(getattr(args, constants.INPUT_NAME))
    unfiltered_vcf.add_info_to_header({'ID': 'LOGIT', 'Description': 'Mutect3 posterior logit',
                            'Type': 'Float', 'Number': 'A'})
    unfiltered_vcf.add_filter_to_header({'ID': 'mutect3', 'Description': 'fails Mutect3 deep learning filter'})

    writer = Writer(getattr(args, constants.OUTPUT_NAME), unfiltered_vcf)  # input vcf is a template for the header

    pbar = tqdm(enumerate(unfiltered_vcf), mininterval=10)
    for n, v in pbar:
        filters = filters_to_keep_from_m2(v)

        encoding = encode_variant(v, zero_based=True)   # cyvcf2 is zero-based
        if encoding in encoding_to_logit_dict:
            logit = encoding_to_logit_dict[encoding]
            v.INFO["LOGIT"] = logit

            if logit > logit_threshold:
                filters.add("mutect3")

        v.FILTER = ';'.join(filters) if filters else 'PASS'
        writer.write_record(v)

    print("closing resources")
    writer.close()
    unfiltered_vcf.close()


if __name__ == '__main__':
    main()
