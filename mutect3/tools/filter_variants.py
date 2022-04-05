import argparse
from typing import Set
import time

import torch
from cyvcf2 import VCF, Writer, Variant
from matplotlib.backends.backend_pdf import PdfPages
from tqdm.autonotebook import tqdm

import mutect3.architecture.normal_artifact_model
import mutect3.architecture.read_set_classifier
from mutect3.data import read_set_datum, read_set_dataset

# TODO: eventually M3 can handle multiallelics
TRUSTED_M2_FILTERS = {'contamination', 'germline', 'weak_evidence', 'multiallelic'}


# this presumes that we have a ReadSetClassifier model and we have saved it via save_mutect3_model as in train_model.py
def load_saved_model(path):
    saved = torch.load(path)
    m3_params = saved['m3_params']
    model = mutect3.architecture.read_set_classifier.ReadSetClassifier(m3_params, None)
    model.load_state_dict(saved['model_state_dict'])
    return model


# this presumes that we have a NormalArtifact model and we have saved it via save_normal_artifact_model as in train_normal_artifact_model.py
def load_saved_normal_artifact_model(path):
    saved = torch.load(path)
    hidden_layers = saved['hidden_layers']
    na_model = mutect3.architecture.normal_artifact_model.NormalArtifactModel(hidden_layers)
    na_model.load_state_dict(saved['model_state_dict'])
    return na_model


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
    parser.add_argument('--input', help='VCF from GATK', required=True)
    parser.add_argument('--test_dataset', help='test dataset file from GATK', required=True)
    parser.add_argument('--trained_m3_model', help='trained Mutect3 model', required=True)
    parser.add_argument('--trained_normal_artifact_model', help='trained normal artifact model', required=True)
    parser.add_argument('--output', help='output filtered vcf', required=True)
    parser.add_argument('--report_pdf', required=False)
    parser.add_argument('--roc_pdf', required=False)
    parser.add_argument('--batch_size', type=int, default=64, required=False)
    parser.add_argument('--turn_off_normal_artifact', action='store_true')
    args = parser.parse_args()

    # record encodings of variants that M2 filtered as germline, contamination, or weak evidence
    # Mutect3 ignores these
    m2_filtering_to_keep = set([encode_variant(v, zero_based=True) for v in VCF(args.input) if filters_to_keep_from_m2(v)])

    print("Reading test dataset")
    unfiltered_test_data = read_set_dataset.read_data(args.test_dataset)
    m3_variants = []
    for datum in unfiltered_test_data:
        encoding = encode_datum(datum)
        if encoding not in m2_filtering_to_keep:
            m3_variants.append(datum)

    print("Size of test dataset: " + str(len(m3_variants)))

    dataset = read_set_dataset.ReadSetDataset(data=m3_variants)
    data_loader = read_set_dataset.make_test_data_loader(dataset, args.batch_size)

    na_model = load_saved_normal_artifact_model(args.trained_normal_artifact_model)
    model = load_saved_model(args.trained_m3_model)
    model.set_normal_artifact_model(na_model)
    use_normal_artifact = (not args.turn_off_normal_artifact)

    # The AF spectrum was, of course, not pre-trained with the rest of the model
    print("Learning AF spectra")
    spectrum_metrics, epoch_spectra = model.learn_spectra(data_loader, num_epochs=25, use_normal_artifact=use_normal_artifact)

    print("generating plots")
    if args.report_pdf is not None:
        with PdfPages(args.report_pdf) as pdf:
            for fig, curve in spectrum_metrics.plot_curves():
                pdf.savefig(fig)

            spectra_plots = model.get_prior_model().plot_spectra()
            for fig, curve in spectra_plots:
                pdf.savefig(fig)

            for fig, curve in epoch_spectra:
                pdf.savefig(fig)

    print("Calculating optimal logit threshold")
    logit_threshold = model.calculate_logit_threshold(loader=data_loader, normal_artifact=use_normal_artifact, roc_plot=args.roc_pdf)
    print("Optimal logit threshold: " + str(logit_threshold))

    encoding_to_logit_dict = {}

    print("Running final calls")
    pbar = tqdm(enumerate(data_loader), mininterval=10)
    for n, batch in pbar:
        logits = model.forward(batch, posterior=True, normal_artifact=use_normal_artifact)

        encodings = [encode_datum(datum) for datum in batch.original_list()]
        for encoding, logit in zip(encodings, logits):
            encoding_to_logit_dict[encoding] = logit.item()

    print("Applying computed logits")

    unfiltered_vcf = VCF(args.input)
    unfiltered_vcf.add_info_to_header({'ID': 'LOGIT', 'Description': 'Mutect3 posterior logit',
                            'Type': 'Float', 'Number': 'A'})
    unfiltered_vcf.add_filter_to_header({'ID': 'mutect3', 'Description': 'fails Mutect3 deep learning filter'})

    writer = Writer(args.output, unfiltered_vcf)  # input vcf is a template for the header

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
