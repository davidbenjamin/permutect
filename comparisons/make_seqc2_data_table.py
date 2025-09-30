sample_table = "sample.tsv"
pair_table = "pair.tsv"
participant = "seqc2"

high_confidence_bedfile = "gs://broad-dsp-david-benjamin/seqc2/High-Confidence_Regions_v1.2.bed"
high_confidence_exome_bedfile = "gs://broad-dsp-david-benjamin/seqc2/High-Confidence_Exonic_Regions_v1.2.bed"
truth_vcf = "gs://broad-dsp-david-benjamin/seqc2/high-confidence_SNV_INDEL_in_HC_regions_v1.2.1.vcf"
truth_vcf_idx = truth_vcf + ".idx"

seqc2_bucket = "gs://broad-dsp-david-benjamin/seqc2"

ffpe_wes_bucket = seqc2_bucket + "/ffpe/wes"
ffpe_wgs_bucket = seqc2_bucket + "/ffpe/wgs"
titration_100x_bucket = seqc2_bucket + "/titration/100x"
titration_50x_bucket = seqc2_bucket + "/titration/50x"
titration_30x_bucket = seqc2_bucket + "/titration/30x"
wgs_hiseq_bucket = seqc2_bucket + "/wgs/hiseq"
wgs_novaseq_bucket = seqc2_bucket + "/wgs/novaseq"
wes_hiseq_bucket = seqc2_bucket + "/wes/hiseq"
wes_novaseq_bucket = seqc2_bucket + "/wes/novaseq"

with open(sample_table, 'w') as sample_file, open(pair_table, 'w') as pair_file:
    # write the headers
    sample_file.write('entity:sample_id\tbam\tbai\tevaluation_truth\tevaluation_truth_idx\tparticipant\n')
    pair_file.write('entity:pair_id\tcalling_intervals\tcase_sample\tcontrol_sample\tparticipant')

    # first do the 2x2 combinations of WGS/WES and hiseq/novaseq
    for platform, platform_string in [("hiseq", "IL"), ("novaseq", "NV")]:
        for target, target_string in [("wgs", "WGS"), ("wes", "WES")]:
            high_conf_bed = high_confidence_exome_bedfile if target == "wes" else high_confidence_bedfile

            bucket = f"{seqc2_bucket}/{target}/{platform}"
            for n in (1, 2, 3):
                tumor_bam = f"{bucket}/{target_string}_{platform_string}_T_{n}.bwa.dedup.bam"
                tumor_bai = f"{bucket}/{target_string}_{platform_string}_T_{n}.bwa.dedup.bai"
                normal_bam = f"{bucket}/{target_string}_{platform_string}_N_{n}.bwa.dedup.bam"
                normal_bai = f"{bucket}/{target_string}_{platform_string}_N_{n}.bwa.dedup.bai"

                tumor_id = f"{target_string}_{platform_string}_T_{n}"
                normal_id = f"{target_string}_{platform_string}_N_{n}"
                pair_id = f"{target_string}_{platform_string}_TN_{n}"

                sample_file.write(f"{tumor_id}\t{tumor_bam}\t{tumor_bai}\t{truth_vcf}\t{truth_vcf_idx}\t{participant}\n")
                sample_file.write(f"{normal_id}\t{normal_bam}\t{normal_bai}\t\t\t{participant}\n")
                pair_file.write(f"{pair_id}\t{high_conf_bed}\t{tumor_id}\t{normal_id}\t{participant}")

    # now the titration series
    for coverage, suffix in [(30, ".s0.3"), (50, ".s0.5"), (100, "")]:
        bucket = f"{seqc2_bucket}/titration/{coverage}x"
        normal_sample_id = None
        for tumor_normal_ratio in ("0-1", "1-19", "1-9", "1-4", "1-1", "3-1", "1-0"):
            bam = f"{bucket}/SPP_GT_{tumor_normal_ratio}_1.bwa.dedup{suffix}.bam"
            bai = bam + ".bai"
            sample_id = f"SPP_GT_{coverage}x_{tumor_normal_ratio}_1"
            sample_file.write(f"{sample_id}\t{bam}\t{bai}\t{truth_vcf}\t{truth_vcf_idx}\t{participant}\n")

            if tumor_normal_ratio == "0-1":     # this is the pure normal
                normal_sample_id = sample_id
            else:           # form a pair with the normal
                pair_id = sample_id + "_TN"
                pair_file.write(f"{pair_id}\t{high_confidence_bedfile}\t{sample_id}\t{normal_sample_id}\t{participant}")




