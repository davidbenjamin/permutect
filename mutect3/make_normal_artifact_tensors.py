import argparse

import mutect3.tensors
from mutect3 import normal_artifact


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--table', help='table of GATK output', required=True)
    parser.add_argument('--output', help='output path for pickled list of tensors', required=True)
    args = parser.parse_args()

    mutect3.tensors.generate_normal_artifact_pickle(args.table, args.output)


if __name__ == '__main__':
    main()