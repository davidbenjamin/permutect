version 1.0

workflow DeepSomatic {
    input {
        File ref_fasta
        File ref_fai
        File ref_dict

        File tumor_bam
        File tumor_bai
        File normal_bam
        File normal_bai

        # Can be WGS,WES,PACBIO,ONT,FFPE_WGS,FFPE_WES,WGS_TUMOR_ONLY,PACBIO_TUMOR_ONLY,ONT_TUMOR_ONLY
        String model_type

        File intervals
        File? masks

        String deepsomatic_extra_args

        File? truth_vcf    # used for evaluation
        File? truth_vcf_idx

        String gatk_docker = "us.gcr.io/broad-gatk/gatk"
        String deepsomatic_docker = "us.gcr.io/broad-dsde-methods/davidben/deepsomatic"
        String? gcs_project_for_requester_pays

        # WDL version 1.0 does not have an empty Optional literal
        # such a literal is very useful because Terra has a bug where whenever a data table is updated, empty values
        # silently and invisibly get converted to empty strings "".  Thus it is useful to recognize empty strings and
        # declare empty Optionals.  The only way to do this in WDL 1.0 is to get an empty Optional as a variable from the
        # workflow inputs.  These inputs should NEVER be filled in!!!!!
        File? EMPTY_STRING_HACK
    }

    call IntervalListToBed {
        input:
            gatk_docker = gatk_docker,
            intervals = intervals
    }

    call GetSampleName {
        input:
            ref_fasta = ref_fasta,
            ref_fai = ref_fai,
            ref_dict = ref_dict,
            tumor_bam = tumor_bam,
            tumor_bai = tumor_bai,
            normal_bam = normal_bam,
            normal_bai = normal_bai,
            gcs_project_for_requester_pays = gcs_project_for_requester_pays,
            gatk_docker = gatk_docker
    }

    call Deepsomatic {
        input:
            ref_fasta = ref_fasta,
            ref_fai = ref_fai,
            ref_dict = ref_dict,
            tumor_bam = tumor_bam,
            tumor_bai = tumor_bai,
            tumor_sample = GetSampleName.tumor_sample,
            normal_bam = normal_bam,
            normal_bai = normal_bai,
            normal_sample = GetSampleName.normal_sample,
            intervals_bed = IntervalListToBed.output_bed,
            intervals_bed_idx = IntervalListToBed.output_bed_idx,
            model_type = model_type,
            deepsomatic_extra_args = deepsomatic_extra_args,
            deepsomatic_docker = deepsomatic_docker
    }

    if (defined(truth_vcf)){
        call Concordance {
            input:
                intervals = intervals,
                masks = if masks == "" then EMPTY_STRING_HACK else masks,
                truth_vcf = select_first([truth_vcf]),
                truth_vcf_idx = select_first([truth_vcf_idx]),
                eval_vcf = Deepsomatic.output_vcf,
                eval_vcf_idx = Deepsomatic.output_vcf_idx,
                gatk_docker = gatk_docker
        }
    }

    output {
        File deepsomatic_calls_vcf = Deepsomatic.output_vcf
        File deepsomatic_calls_vcf_idx = Deepsomatic.output_vcf_idx

        File? fn = Concordance.fn
        File? fn_idx = Concordance.fn_idx
        File? fp = Concordance.fp
        File? fp_idx = Concordance.fp_idx
        File? tp = Concordance.tp
        File? tp_idx = Concordance.tp_idx
        File? ffn = Concordance.ffn
        File? ffn_idx = Concordance.ffn_idx
        File? ftn = Concordance.ftn
        File? ftn_idx = Concordance.ftn_idx
        File? concordance_summary = Concordance.summary
        File? filter_analysis = Concordance.filter_analysis
    }
}

task IntervalListToBed {
    input {
        String gatk_docker
        File intervals

        Int cpu = 2
        Int mem_gb = 4
        Int disk_gb = 100
        Int boot_disk_gb = 4
        Int max_retries = 0
        Int preemptible = 0
    }

    command <<<
        gatk IntervalListToBed --INPUT ~{intervals} --OUTPUT intervals.bed

        sort -k 1,1 -k 2,2n -k 3,3n intervals.bed | bgzip -c > sorted.bed.gz
        tabix -pbed sorted.bed.gz
    >>>

    runtime {
        docker: gatk_docker
        bootDiskSizeGb: boot_disk_gb
        memory: mem_gb + " GB"
        disks: "local-disk " + disk_gb + " SSD"
        preemptible: preemptible
        maxRetries: max_retries
        cpu: cpu
    }

    output {
        File output_bed = "sorted.bed.gz"
        File output_bed_idx = "sorted.bed.gz.tbi"
    }
}



task Deepsomatic {
    input {
        File ref_fasta
        File ref_fai
        File ref_dict

        File tumor_bam
        File tumor_bai
        String tumor_sample
        File normal_bam
        File normal_bai
        String normal_sample
        File intervals_bed
        File intervals_bed_idx
        String model_type
        String deepsomatic_extra_args

        String deepsomatic_docker

        Int cpu = 4
        Int mem_gb = 16
        Int disk_gb = 1000
        Int boot_disk_gb = 10
        Int max_retries = 0
        Int preemptible = 0
    }

    command <<<
        run_deepsomatic \
            --model_type=~{model_type} \
            --ref=~{ref_fasta} \
            --reads_normal=~{normal_bam} \
            --reads_tumor=~{tumor_bam} \
            --output_vcf=output/output.vcf.gz \
            --output_gvcf=output/output.g.vcf.gz \
            --sample_name_tumor=~{tumor_sample} \
            --sample_name_normal=~{normal_sample} \
            --num_shards=~{cpu} \
            --logging_dir=output/logs \
            --intermediate_results_dir output/intermediate_results_dir \
            ‑‑interval‑file=~{intervals_bed} \
            --use_default_pon_filtering=false \
            --dry_run=false \
            ~{deepsomatic_extra_args}

    >>>

    runtime {
        docker: deepsomatic_docker
        bootDiskSizeGb: boot_disk_gb
        memory: mem_gb + " GB"
        disks: "local-disk " + disk_gb + " SSD"
        preemptible: preemptible
        maxRetries: max_retries
        cpu: cpu
    }

    output {
        File output_vcf = "output/output.vcf.gz"
        File output_vcf_idx = "output/output.vcf.gz.tbi"
    }
}



task Concordance {
    input {
    	File? intervals
    	File? masks
    	File truth_vcf
    	File truth_vcf_idx
    	File eval_vcf
    	File eval_vcf_idx

    	# runtime
    	String gatk_docker = "us.gcr.io/broad-gatk/gatk"
    	Int preemptible = 0
	}

    command {
        gatk --java-options "-Xmx2g" Concordance \
            ~{"-L " + intervals} \
            ~{"-XL " + masks} \
            -truth ~{truth_vcf} -eval ~{eval_vcf} \
            -tpfn "tpfn.vcf" \
            -tpfp "tpfp.vcf" \
            -ftnfn "ftnfn.vcf" \
            -filter-analysis "filter-analysis.txt" \
            -summary "summary.txt"

        grep '#' tpfn.vcf > HEAD
        grep STATUS=FN tpfn.vcf > BODY
        cat HEAD BODY > false_negatives.vcf

        grep '#' tpfp.vcf > HEAD
        grep STATUS=FP tpfp.vcf > BODY
        cat HEAD BODY > false_positives.vcf

        grep '#' tpfp.vcf > HEAD
        grep STATUS=TP tpfp.vcf > BODY
        cat HEAD BODY > true_positives.vcf

        grep '#' ftnfn.vcf > HEAD
        grep STATUS=FFN ftnfn.vcf > BODY
        cat HEAD BODY > filtered_false_negatives.vcf
        grep STATUS=FTN ftnfn.vcf > BODY
        cat HEAD BODY > filtered_true_negatives.vcf

        for vcf in false_negatives.vcf false_positives.vcf true_positives.vcf filtered_false_negatives.vcf filtered_true_negatives.vcf; do
            #HACK: IndexFeatureFile throws error if vcf is empty, which is possible here especially in the case of false negatives
            gatk --java-options "-Xmx2g" SelectVariants -V $vcf -O tmp.vcf
            mv tmp.vcf $vcf
            mv tmp.vcf.idx $vcf.idx
        done
    }

    runtime {
        memory: "5 GB"
        bootDiskSizeGb: 12
        docker: "${gatk_docker}"
        disks: "local-disk " + 100 + " HDD"
        preemptible: select_first([preemptible, 2])
    }

    output {
        File fn = "false_negatives.vcf"
        File fn_idx = "false_negatives.vcf.idx"
        File fp = "false_positives.vcf"
        File fp_idx = "false_positives.vcf.idx"
        File tp = "true_positives.vcf"
        File tp_idx = "true_positives.vcf.idx"
        File ffn = "filtered_false_negatives.vcf"
        File ffn_idx = "filtered_false_negatives.vcf.idx"
        File ftn = "filtered_true_negatives.vcf"
        File ftn_idx = "filtered_true_negatives.vcf.idx"
        File summary = "summary.txt"
        File filter_analysis = "filter-analysis.txt"
    }
}

task GetSampleName {
    input {
        File ref_fasta
        File ref_fai
        File ref_dict
        File tumor_bam
        File tumor_bai
        File? normal_bam
        File? normal_bai
        String? gcs_project_for_requester_pays

        String gatk_docker
        Int mem = 2
        Int boot_disk_size = 10
        Int preemptible = 0
        Int max_retries = 0
        Int disk_space = 10
        Int cpu = 1
    }


    parameter_meta{
        ref_fasta: {localization_optional: true}
        ref_fai: {localization_optional: true}
        ref_dict: {localization_optional: true}
        tumor_reads: {localization_optional: true}
        tumor_reads_index: {localization_optional: true}
        normal_reads: {localization_optional: true}
        normal_reads_index: {localization_optional: true}
    }

    command <<<
        touch normal_names.txt
        if [[ ! -z "~{normal_bam}" ]]; then
            gatk GetSampleName -R ~{ref_fasta} -I ~{normal_bam} -O normal_names.txt -encode \
                ~{"--gcs-project-for-requester-pays " + gcs_project_for_requester_pays}
        fi

        gatk GetSampleName -R ~{ref_fasta} -I ~{tumor_bam} -O tumor_names.txt -encode \
                ~{"--gcs-project-for-requester-pays " + gcs_project_for_requester_pays}
    >>>

    runtime {
        docker: gatk_docker
        bootDiskSizeGb: boot_disk_size
        memory: mem + " GB"
        disks: "local-disk " + disk_space + " SSD"
        preemptible: preemptible
        maxRetries: max_retries
        cpu: cpu
    }

    output {
        String normal_sample = read_string("normal_names.txt")
        String tumor_sample = read_string("tumor_names.txt")
    }
}