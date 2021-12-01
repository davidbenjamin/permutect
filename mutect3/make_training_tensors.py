import argparse
from mutect3 import tensors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', help='VCF from GATK', required=True)
    parser.add_argument('--tumor', help='tumor sample name', required=True)
    parser.add_argument('--normal', help='normal sample name')
    parser.add_argument('--output', help='output path for pickled list of tensors', required=True)
    args = parser.parse_args()

    # if --normal not specified, args.normal is None, which works all the way through
    data = tensors.make_training_tensors_from_vcf(args.input, args.tumor, args.normal)
    tensors.make_pickle(args.output, data)


if __name__ == '__main__':
    main()