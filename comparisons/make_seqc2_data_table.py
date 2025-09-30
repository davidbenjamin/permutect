sample_table = "sample.tsv"
pair_table = "pair.tsv"

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


    # first do the 2x2 combinations of WGS/WES and hiseq/novaseq
    for platform, platform_string in [("hiseq", "IL"), ("novaseq", "NV")]:
        for target, target_string in [("wgs", "WGS"), ("wes", "WES")]:
            bucket = f"{seqc2_bucket}/{target}/{platform}"
            for n in (1, 2, 3):
                tumor_bam = f"{bucket}/{target_string}_{platform_string}_T_{n}.bwa.dedup.bam"
                tumor_bai = f"{bucket}/{target_string}_{platform_string}_T_{n}.bwa.dedup.bai"
                normal_bam = f"{bucket}/{target_string}_{platform_string}_N_{n}.bwa.dedup.bam"
                normal_bai = f"{bucket}/{target_string}_{platform_string}_N_{n}.bwa.dedup.bai"
