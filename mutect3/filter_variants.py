import argparse
import pysam
from matplotlib.backends.backend_pdf import PdfPages
from tqdm.autonotebook import tqdm
import torch
from mutect3 import tensors, networks, validation, data

REF_DOWNSAMPLE = 20
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
    parser.add_argument('--batch_size', type=int, default=64, required=False)
    args = parser.parse_args()

    print("Reading tensors from VCF")
    dataset = get_test_dataset(args.input, args.tumor, args.normal, REF_DOWNSAMPLE)

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
        roc_fig, roc_curve = validation.plot_roc_curve(model, data_loader, normal_artifact=True)
        with PdfPages(args.report_pdf) as pdf:
            for fig, curve in spectra_plots:
                pdf.savefig(fig)
            pdf.savefig(roc_fig)

    print("Calculating optimal logit threshold")
    logit_threshold = model.calculate_logit_threshold(data_loader)
    # print("Optimal logit threshold: " + str(logit_threshold))

    encoding_to_logit_dict = {}

    print("Running final calls")
    pbar = tqdm(enumerate(data_loader))
    for n, batch in pbar:
        logits = model(batch, posterior=True, normal_artifact=True)

        # encoding has form contig:position:alt
        # TODO write method
        encodings = [site.locus() + ':' + site.alt() for site in batch.site_info()]
        for encoding, logit in zip(encodings, logits):
            encoding_to_logit_dict[encoding] = logit

    vcf_in = pysam.VariantFile(args.input)
    vcf_out = pysam.VariantFile(args.output, 'w', header=vcf_in.header)

    vcf_out.header.add_meta(key="INFO", items=[('ID', 'LOGIT'),
                                               ('Number', '1'),
                                               ('Type', 'Float'),
                                               ('Description',
                                                'logit for M3 posterior probability of technical artifact')])

    vcf_out.header.add_meta(key="FILTER", items=[('ID', 'mutect3'),
                                                 ('Description', 'filtered by Mutect3 as technical artifact')])

    for rec in vcf_in:
        encoding = rec.contig + ':' + rec.pos + ':' + rec.alleles[1]

        # throw out all of M2's technical artifact filters
        rec.filter = set(rec.filter).intersection(TRUSTED_M2_FILTERS)

        if encoding in encoding_to_logit_dict:
            logit = encoding_to_logit_dict[encoding]
            rec.info["LOGIT"] = logit
            if logit > logit_threshold:
                rec.filter.add('mutect3')

        vcf_out.write(rec)

    vcf_out.close()


# tensorize all the possible somatic variants and technical artifacts, skipping germline,
# contamination, and weak evidence variants, for whose filters we trust Mutect2.
def get_test_dataset(vcf, tumor, normal, ref_downsample):
    data_list = []
    vcf_input = pysam.VariantFile(vcf)
    for n, rec in enumerate(vcf_input):
        if n % 10000 == 0:
            print(rec.contig + ':' + str(rec.pos))
        datum = tensors.unlabeled_datum_from_vcf(rec, tumor, normal, ref_downsample)
        filters = datum.mutect_info().filters()
        if filters.isdisjoint(TRUSTED_M2_FILTERS):
            data_list.append(datum)

    return data.Mutect3Dataset(data_list, shuffle=True)


if __name__ == '__main__':
    main()
