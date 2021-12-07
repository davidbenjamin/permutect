import argparse
import pysam


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--truth_vcf', help='truth VCF', required=True)
    parser.add_argument('--mutect3_vcf', help='Mutect3 VCF', required=True)
    parser.add_argument('--mutect2_vcf', help='Mutect2 VCF')
    args = parser.parse_args()

    encoded_truth_variants = set()
    print("Reading in truth VCF")
    truth_input = pysam.VariantFile(args.truth_vcf)
    for n, rec in enumerate(truth_input):
        if n % 10000 == 0:
            print(rec.contig + ':' + str(rec.pos))
        #TODO: check whether variant is filtered in truth
        encoding = rec.contig + ':' + str(rec.pos) + ':' + rec.alleles[1]
        encoded_truth_variants.add(encoding)

    print("Reading in Mutect3 VCF")
    m3_tp, m3_fn, m3_fp = count_true_and_false(args.mutect3_vcf, encoded_truth_variants)
    m3_sensitivity, m3_precision = m3_tp / (m3_tp + m3_fn), m3_tp / (m3_tp + m3_fp)

    print("Reading in Mutect2 VCF")
    m2_tp, m2_fn, m2_fp = count_true_and_false(args.mutect2_vcf, encoded_truth_variants)
    m2_sensitivity, m2_precision = m2_tp / (m2_tp + m2_fn), m2_tp / (m2_tp + m2_fp)

    print("Mutect3 sensitivity: " + str(m3_sensitivity) + ", precision: " + str(m3_precision))
    print("Mutect2 sensitivity: " + str(m2_sensitivity) + ", precision: " + str(m2_precision))


def count_true_and_false(vcf, encoded_truth_variants):
    tp, fn, fp = 0, 0, 0
    for n, rec in enumerate(pysam.VariantFile(vcf)):
        if n % 10000 == 0:
            print(rec.contig + ':' + str(rec.pos))
        encoding = rec.contig + ':' + str(rec.pos) + ':' + rec.alleles[1]
        variant_in_truth = encoding in encoded_truth_variants

        filters = set(rec.filter)
        if len(filters) == 1 and 'PASS' in filters: #Variant passes Mutect 3 filters
            if variant_in_truth:
                tp += 1  # true positive
            else:
                fp += 1  # false positive
        elif len(filters) > 0 and variant_in_truth:
            fn += 1      # false

    return tp, fn, fp

if __name__ == '__main__':
    main()