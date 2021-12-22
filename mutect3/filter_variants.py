import argparse

import torch
from matplotlib.backends.backend_pdf import PdfPages
from tqdm.autonotebook import tqdm

from mutect3 import networks, data

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', help='VCF from GATK', required=True)
    parser.add_argument('--trained_m3_model', help='trained Mutect3 model', required=True)
    parser.add_argument('--tumor', help='tumor sample name', required=True)
    parser.add_argument('--normal', help='normal sample name')
    parser.add_argument('--output', help='output filtered vcf', required=True)
    parser.add_argument('--report_pdf', required=False)
    parser.add_argument('--roc_pdf', required=False)
    parser.add_argument('--batch_size', type=int, default=64, required=False)
    args = parser.parse_args()

    print("Reading tensors from VCF")
    dataset = #TODO: need to write a method that gets dataset from M2 text file output.
    # TODO: it probably doesn't have to be specific to test data

    print("Creating data loader")
    data_loader = data.make_test_data_loader(dataset, args.batch_size)

    print("Loading saved model")
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
    logit_threshold = model.calculate_logit_threshold(data_loader, args.roc_pdf)
    print("Optimal logit threshold: " + str(logit_threshold))

    encoding_to_logit_dict = {}

    print("Running final calls")
    pbar = tqdm(enumerate(data_loader))
    for n, batch in pbar:
        logits = model(batch, posterior=True, normal_artifact=True)

        # encoding has form contig:position:alt
        # TODO write method
        encodings = [datum.locus() + ':' + datum.alt() for datum in batch]
        for encoding, logit in zip(encodings, logits):
            encoding_to_logit_dict[encoding] = logit.item()

    with open(args.input) as unfiltered_vcf, open(args.output, "w") as filtered_vcf:
        for line in unfiltered_vcf:
            #header lines
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

if __name__ == '__main__':
    main()
