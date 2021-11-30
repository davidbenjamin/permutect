import argparse
from mutect3 import tensors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--table', help='table of GATK output', required=True)
    parser.add_argument('--tumor', help='tumor sample name', required=True)
    parser.add_argument('--normal', help='normal sample name')
    parser.add_argument('--output', help='output path for pickled list of tensors', required=True)
    args = parser.parse_args()

    # if --normal not specified, args.normal is None, which works all the way through
    data = tensors.make_tensors(args.table, is_training=True, sample_name=args.tumor, normal_sample_name=args.normal)

    print(args.tumor)
    print(args.normal)
    print(len(data))
    print(data[0])
    tensors.make_pickle(args.output, data)


if __name__ == '__main__':
    main()