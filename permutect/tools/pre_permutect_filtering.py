import argparse

import cyvcf2
import numpy as np
import pysam
from tqdm.autonotebook import tqdm

from permutect import constants
from permutect.misc_utils import declare_format_dp
from permutect.misc_utils import get_tumor_sample_index
from permutect.utils.sequence_utils import StemLoop
from permutect.utils.sequence_utils import find_near_stem_loop

HAIRPIN_FILTER = "hairpin"
STEM_LOOP_FREE_ENERGY_INFO_KEY = "SLFE"
STEM_LOOP_DISTANCE_INFO_KEY = "SLDIST"


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(
        description="Apply filters to a Mutect2 VCF before filter_variants, which trusts these filters and excludes "
        "the variants they filter from its posterior model."
    )
    parser.add_argument("--" + constants.INPUT_NAME, required=True, help="unfiltered input Mutect2 VCF")
    parser.add_argument(
        "--" + constants.REFERENCE_NAME,
        required=True,
        help="reference genome .fasta file, accompanied by a .fai index",
    )
    parser.add_argument("--" + constants.OUTPUT_NAME, required=True, help="path to output VCF")
    parser.add_argument(
        "--stem-loop-max-energy",
        dest=constants.STEM_LOOP_MAX_ENERGY_NAME,
        type=float,
        default=1.5,
        help="maximum total stem-loop energy for applying the hairpin filter (strictly less than this value)",
    )
    return parser.parse_args(argv)


def main_without_parsing(args):
    apply_pre_permutect_filters(
        input_vcf=getattr(args, constants.INPUT_NAME),
        ref_fasta=getattr(args, constants.REFERENCE_NAME),
        output_vcf=getattr(args, constants.OUTPUT_NAME),
        stem_loop_max_energy=getattr(args, constants.STEM_LOOP_MAX_ENERGY_NAME),
    )


def find_hairpin(
    v: cyvcf2.Variant, tumor_sample_index: int, reference: pysam.FastaFile, stem_loop_max_energy: float
) -> tuple[StemLoop, int] | None:
    """
    (stem loop, distance to stem) if the variant has a low allele fraction and is near a stem loop, otherwise None
    """
    tumor_allele_depths = v.format("AD")[tumor_sample_index]
    total_alt_depth = np.sum(tumor_allele_depths[1:])
    total_depth = tumor_allele_depths[0] + total_alt_depth
    # check the allele fraction first because folding the reference sequence is much more expensive
    if total_depth == 0 or total_alt_depth / total_depth >= 0.1:
        return None
    return find_near_stem_loop(reference, v.CHROM, v.start, max_energy=stem_loop_max_energy)


def apply_pre_permutect_filters(input_vcf, ref_fasta, output_vcf, stem_loop_max_energy: float = 1.5):
    reference = pysam.FastaFile(ref_fasta)
    vcf = cyvcf2.VCF(input_vcf)
    tumor_sample_index = get_tumor_sample_index(vcf)

    declare_format_dp(vcf)
    vcf.add_info_to_header(
        {
            "ID": STEM_LOOP_FREE_ENERGY_INFO_KEY,
            "Description": "Total free energy of the selected nearby stem loop",
            "Type": "Float",
            "Number": "1",
        }
    )
    vcf.add_info_to_header(
        {
            "ID": STEM_LOOP_DISTANCE_INFO_KEY,
            "Description": "Distance in bases from variant start to the selected stem",
            "Type": "Integer",
            "Number": "1",
        }
    )
    vcf.add_filter_to_header({"ID": HAIRPIN_FILTER, "Description": "near low-alt-fraction stem-loop"})

    writer = cyvcf2.Writer(output_vcf, vcf)  # input vcf is a template for the header
    for v in tqdm(vcf, mininterval=60):
        hairpin = find_hairpin(v, tumor_sample_index, reference, stem_loop_max_energy)
        if hairpin is not None:
            stem_loop, stem_distance = hairpin
            v.FILTER = HAIRPIN_FILTER if v.FILTER is None else v.FILTER + ";" + HAIRPIN_FILTER
            v.INFO[STEM_LOOP_FREE_ENERGY_INFO_KEY] = stem_loop.total_energy
            v.INFO[STEM_LOOP_DISTANCE_INFO_KEY] = stem_distance
        writer.write_record(v)
    writer.close()
    vcf.close()
    reference.close()


def main():
    args = parse_arguments()
    main_without_parsing(args)


if __name__ == "__main__":
    main()
