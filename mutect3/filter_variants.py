import argparse
from typing import Set
import torch
from matplotlib.backends.backend_pdf import PdfPages
from tqdm.autonotebook import tqdm

from mutect3 import networks, data
from cyvcf2 import VCF, Writer, Variant

TRUSTED_M2_FILTERS = {'contamination', 'germline', 'weak_evidence'}


# this presumes that we have a ReadSetClassifier model and we have saved it via save_mutect3_model
def load_saved_model(path):
    saved = torch.load(path)
    m3_params = saved['m3_params']
    # TODO: this should not be hard-coded.  See above above introducing na_params
    na_model = networks.NormalArtifactModel([10, 10, 10])
    model = networks.ReadSetClassifier(m3_params, na_model)
    model.load_state_dict(saved['model_state_dict'])
    return model


def encode(contig: str, position: int, alt: str):
    return contig + ':' + str(position) + ':' + alt


def encode_datum(datum: data.Datum):
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
    parser.add_argument('--output', help='output filtered vcf', required=True)
    parser.add_argument('--report_pdf', required=False)
    parser.add_argument('--roc_pdf', required=False)
    parser.add_argument('--batch_size', type=int, default=64, required=False)
    args = parser.parse_args()

    # record encodings of variants that M2 filtered as germline, contamination, or weak evidence
    # Mutect3 ignores these
    m2_filtering_to_keep = set([encode_variant(v) for v in VCF(args.input) if filters_to_keep_from_m2(v)])

    print("Reading test dataset")
    m3_variants = filter(lambda d: encode_datum(d) not in m2_filtering_to_keep, data.read_data(args.test_dataset))
    dataset = data.Mutect3Dataset(data=m3_variants)
    data_loader = data.make_test_data_loader(dataset, args.batch_size)

    model = load_saved_model(args.trained_m3_model)

    # The AF spectrum was, of course, not pre-trained with the rest of the model
    print("Learning AF spectra")
    model.learn_spectra(data_loader, num_epochs=200)

    print("generating plots")
    if args.report_pdf is not None:
        spectra_plots = model.get_prior_model().plot_spectra()
        with PdfPages(args.report_pdf) as pdf:
            for fig, curve in spectra_plots:
                pdf.savefig(fig)

    print("Calculating optimal logit threshold")
    logit_threshold = model.calculate_logit_threshold(loader=data_loader, normal_artifact=True, roc_plot=args.roc_pdf)
    print("Optimal logit threshold: " + str(logit_threshold))

    encoding_to_logit_dict = {}

    print("Running final calls")
    pbar = tqdm(enumerate(data_loader))
    for n, batch in pbar:
        logits = model(batch, posterior=True, normal_artifact=True)

        encodings = [encode_datum(datum) for datum in batch.original_list()]
        for encoding, logit in zip(encodings, logits):
            print(encoding)
            encoding_to_logit_dict[encoding] = logit.item()

    print("Applying computed logits")

    unfiltered_vcf = VCF(args.input)
    unfiltered_vcf.add_info_to_header({'ID': 'LOGIT', 'Description': 'Mutect3 posterior logit',
                            'Type': 'Float', 'Number': 'A'})
    unfiltered_vcf.add_filter_to_header({'ID': 'mutect3', 'Description': 'fails Mutect3 deep learning filter'})

    writer = Writer(args.output, unfiltered_vcf) # input vcf is a template for the header

    pbar = tqdm(enumerate(unfiltered_vcf))
    for n, v in pbar:
        filters = filters_to_keep_from_m2(v)

        encoding = encode_variant(v, zero_based=True)   # cyvcf2 is zero-based
        print(encoding)
        if encoding in encoding_to_logit_dict:
            logit = encoding_to_logit_dict[encoding]
            v.INFO["LOGIT"] = logit

            if logit > logit_threshold:
                filters.add("mutect3")

        v.FILTER = ';'.join(filters) if filters else '.'
        writer.write_record(v)

    print("closing resources")
    writer.close()
    unfiltered_vcf.close()


    '''
    with open(args.input) as unfiltered_vcf, open(args.output, "w") as filtered_vcf:
        for line in unfiltered_vcf:
            # header lines
            info_added, filter_added = False, False
            if line.startswith('#'):
                if (not filter_added) and line.startswith('##FILTER'):
                    filtered_vcf.write('##FILTER=<ID=mutect3,Description="Technical artifact according to Mutect3 deep sets model">')
                    filter_added = True
                if (not info_added) and line.startswith('##INFO'):
                    filtered_vcf.write('##INFO=<ID=LOGIT,Number=1,Type=Float,Description="logit for M3 posterior probability of technical artifact">')
                    info_added = True
                filtered_vcf.write(line)
            #non-header lines
            else:
                tokens = line.strip().split('\t')
                contig, position, alts = tokens[0], tokens[1], tokens[4]
                filters = set(tokens[6].split(';')).intersection(TRUSTED_M2_FILTERS)
                encoding = contig + ':' + position + ':' + alts.split(',')[0]
                if encoding in encoding_to_logit_dict:
                    logit = encoding_to_logit_dict[encoding]
                    tokens[7] = tokens[7] + ';LOGIT=' + str(logit)    # add LOGIT INFO
                    if logit > logit_threshold: # fails Mutect3
                        filters.add('mutect3')
                if not filters:
                    filters.add('PASS')
                tokens[6] = ';'.join(filters)

                filtered_vcf.write('\t'.join(tokens) + '\n')
    '''


if __name__ == '__main__':
    main()
