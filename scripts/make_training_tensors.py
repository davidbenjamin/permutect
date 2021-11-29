import argparse
from mutect3 import tensors


def main(table_file, tumor_sample, normal_sample, pickle_file):
    data = tensors.make_tensors(table_file, is_training=True, sample_name=tumor_sample, normal_sample_name=normal_sample)
    tensors.make_pickle(pickle_file, data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--table', help='table of GATK output', required=True)
    parser.add_argument('--tumor', help='tumor sample name', required=True)
    parser.add_argument('--normal', help='normal sample name')
    parser.add_argument('--output', help='output path for pickled list of tensors', required=True)
    args = parser.parse_args()

    # if --normal not specified, args.normal is None, which works all the way through
    main(args.table, args.tumor, args.normal, args.output)