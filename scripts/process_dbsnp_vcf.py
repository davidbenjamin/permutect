"""
We assume that the dbSNP VCF already has records AF=0 removed

Run this script as python process_aou_vcf.py aou_input.vcf aou.vcf, where aou.vcf is the desired output file name
"""
import argparse

parser = argparse.ArgumentParser()
# positional argument: input AoU VCF
parser.add_argument('input', type=str, help='The name of the raw input dbSNP VCF')
# second positional argument: output file name
parser.add_argument('output', type=str, help='The name of the output file')
args = parser.parse_args()


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

            # contigs are refseq accession numbers like NC_000001.11
            # here the .11 means the hg38 version (it's not necessarily 11) and is ignored.  The numbers after
            # "NC_" are 1 for chr1. . . 22 for chr22, 23 for chrX, and 24 for chrY

            refseq_num = int(tokens[0].split('.')[0].removeprefix("NC_"))
            contig = None
            if refseq_num <= 22:
                contig = f"chr{refseq_num}"
            elif refseq_num == 23:
                contig = "chrX"
            elif refseq_num == 24:
                contig = "chrY"

            # tokens 1 is position, tokens 2 if dbSNP ID which we discard and replace with ., tokens 3-4 are ref alt
            # tokens
            if contig is not None:
                writer.write("\t".join([contig, tokens[1], '.', tokens[3], tokens[4], '.', '.']) + "\t")

            # INFO fields are semicolon-separated.  We care about the first four: AC, AN, AF, Hom
            info_tokens = tokens[7].split(";")

            # example INFO
            # RS=1280576611;dbSNPBuildID=151;SSR=0;PSEUDOGENEINFO=WASH7P:653635;VC=SNV;INT;GNO;FREQ=TOMMO:1,2.613e-05|dbGaP_PopFreq:0.9998,0.0001686

            freq_entry = next((token for token in reversed(info_tokens) if token.startswith("FREQ")), None)

            af_tokens = freq_entry.split('=')[-1].split('|')
            writer.write(f"AF={af_to_use:.2e}\n")