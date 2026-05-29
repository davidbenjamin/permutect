"""
We assume that the AoU VCF already has records ith AC=0 removed

Run this script as python process_aou_vcf.py aou_input.vcf aou.vcf, where aou.vcf is the desired output file name
"""
import argparse

parser = argparse.ArgumentParser()
# positional argument: input AoU VCF
parser.add_argument('input', type=str, help='The name of the raw input AoU VCF')
# second positional argument: output file name
parser.add_argument('output', type=str, help='The name of the output file')
args = parser.parse_args()


# AoU forbids revealing AC of variants with fewer than 20 participants, so we obscure AF with a constant value
RARE_VARIANT_AF = 10 / 500000
with open(args.input, 'r') as reader, open(args.output, 'w') as writer:
    line_count = 0
    for line in reader:
        # header line -- don't modify
        if line.startswith('#'):
            writer.write(line)
        else:
            line_count += 1
            tokens = line.strip().split("\t")
            if line_count % 1000000 == 0:
                print(f"Processed {line_count} lines")
                print(f"Position is {tokens[0]}:{tokens[1]}.")

            # the first 7 entries don't need to be changed.  Write them tab-separated.
            writer.write("\t".join(tokens[:7]) + "\t")

            # INFO fields are semicolon-separated.  We care about the first four: AC, AN, AF, Hom
            info_tokens = tokens[7].split(";")

            # grab the numbers to the right of the equals sign
            ac, an, af, hom = map(lambda x: float(x.split("=")[1]), info_tokens[:4])
            num_participants = ac - hom
            af_to_use = af if num_participants > 20 else RARE_VARIANT_AF

            writer.write(f"AF={af_to_use:.3e}\n")