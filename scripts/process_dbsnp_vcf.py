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
            refseq_string = tokens[0].split('.')[0].removeprefix("NC_")

            if not refseq_string.isdigit():
                continue

            refseq_num = int(refseq_string)
            contig = None
            if refseq_num <= 22:
                contig = f"chr{refseq_num}"
            elif refseq_num == 23:
                contig = "chrX"
            elif refseq_num == 24:
                contig = "chrY"

            if contig is None:
                continue

            alt_alleles = tokens[4].split(',')
            alt_count = len(alt_alleles)

            # INFO fields are semicolon-separated.  We only care about the FREQ field
            info_tokens = tokens[7].split(";")

            # example INFO
            # RS=1280576611;dbSNPBuildID=151;SSR=0;PSEUDOGENEINFO=WASH7P:653635;VC=SNV;INT;GNO;FREQ=TOMMO:1,2.613e-05|dbGaP_PopFreq:0.9998,0.0001686

            freq_entry = next((token for token in reversed(info_tokens) if token.startswith("FREQ")), None)

            # for an entry with e.g. one ref and two alt alleles, each AF token looks like, for example
            # TOMMO:0.98,0.01,0.01
            if freq_entry is None:
                continue
            af_tokens = freq_entry.split('=')[-1].split('|')

            max_alt_afs = list(0.0 for _ in range(alt_count))
            for af_token in af_tokens:
                project = af_token.split(':')[0]

                # AFs for the Simons Genetic Diversity Project are reported weirdly -- often as 1.0 for the minor
                # allele, which is just wrong.  It's a cohort of 300 genomes.
                max_alt_af = 1/300 if project.startswith("SGDP") else 0.99

                alt_afs = map(lambda tok: 0.0 if tok == '.' else float(tok), af_token.split(':')[-1].split(',')[1:])
                for n, af in enumerate(alt_afs):
                    max_alt_afs[n] = max(max_alt_afs[n], min(af, max_alt_af))

            # each alt allele gets its own line
            for n, alt_allele in enumerate(alt_alleles):
                if max_alt_afs[n] > 0:
                    # tokens 1 is position, tokens 2 if dbSNP ID which we discard and replace with ., tokens 3-4 are ref alt
                    # tokens
                    writer.write("\t".join([contig, tokens[1], '.', tokens[3], alt_allele, '.', '.']) + "\t")
                    writer.write(f"AF={max_alt_afs[n]}\n")
