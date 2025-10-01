version 1.0

workflow Strelka {
    input {
        File ref_fasta
        File ref_fai
        File ref_dict

        File tumor_bam
        File tumor_bai
        File normal_bam
        File normal_bai
        File intervals
        File? masks
        String manta_extra_args
        String strelka_extra_args

        File? truth_vcf    # used for evaluation
        File? truth_vcf_idx

        String gatk_docker = "us.gcr.io/broad-gatk/gatk"
        String strelka_docker = "us.gcr.io/broad-dsde-methods/davidben/strelka"
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

    call Manta {
        input:
            ref_fasta = ref_fasta,
            ref_fai = ref_fai,
            ref_dict = ref_dict,
            tumor_bam = tumor_bam,
            tumor_bai = tumor_bai,
            normal_bam = normal_bam,
            normal_bai = normal_bai,
            intervals_bed = IntervalListToBed.output_bed,
            intervals_bed_idx = IntervalListToBed.output_bed_idx,
            manta_extra_args = manta_extra_args,
            strelka_docker = strelka_docker
    }

    call Strelka {
        input:
            ref_fasta = ref_fasta,
            ref_fai = ref_fai,
            ref_dict = ref_dict,
            tumor_bam = tumor_bam,
            tumor_bai = tumor_bai,
            normal_bam = normal_bam,
            normal_bai = normal_bai,
            intervals_bed = IntervalListToBed.output_bed,
            intervals_bed_idx = IntervalListToBed.output_bed_idx,
            manta_vcf = Manta.manta_vcf,
            manta_vcf_idx = Manta.manta_vcf_idx,
            strelka_extra_args = strelka_extra_args,
            strelka_docker = strelka_docker
    }

    call Merge {
        input:
            gatk_docker = gatk_docker,
            snvs_vcf = Strelka.snvs_vcf,
            snvs_vcf_idx = Strelka.snvs_vcf_idx,
            indels_vcf = Strelka.indels_vcf,
            indels_vcf_idx = Strelka.indels_vcf_idx
    }

    if (defined(truth_vcf)){
        call Concordance {
            input:
                intervals = intervals,
                masks = if masks == "" then EMPTY_STRING_HACK else masks,
                truth_vcf = select_first([truth_vcf]),
                truth_vcf_idx = select_first([truth_vcf_idx]),
                eval_vcf = Merge.merged_vcf,
                eval_vcf_idx = Merge.merged_vcf_idx,
                gatk_docker = gatk_docker
        }
    }

    output {
        File strelka_calls_vcf = Merge.merged_vcf
        File strelka_calls_vcf_idx = Merge.merged_vcf_idx

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


task Manta {
    input {
        File ref_fasta
        File ref_fai
        File ref_dict

        File tumor_bam
        File tumor_bai
        File normal_bam
        File normal_bai
        File intervals_bed
        File intervals_bed_idx
        String manta_extra_args

        String strelka_docker

        Int cpu = 4
        Int mem_gb = 16
        Int disk_gb = 1000
        Int boot_disk_gb = 10
        Int max_retries = 0
        Int preemptible = 0
    }

    command <<<
        configManta.py \
                --normalBam ~{normal_bam} \
                --tumorBam ~{tumor_bam} \
                --referenceFasta ~{ref_fasta} \
                --runDir output \
                --callRegions ~{intervals_bed} \
                ~{manta_extra_args}

        echo "contents of output directory after configuring workflow:"
        ls output

        echo "About to run output/runWorkflow.py"

        output/runWorkflow.py -m local -j ~{cpu}

        echo "contents of output directory after running workflow:"
        ls output

        echo "contents of output/results:"
        ls output/results

        echo "contents of output/variants:"
        ls output/variants

        echo "contents of output/results/variants:"
        ls output/results/variants

        echo "contents of output/results/stats:"
        ls output/results/stats

        echo "contents of output/results/evidence:"
        ls output/results/evidence

        echo "We're going to cat output/runWorkflow.py:"
        cat output/runWorkflow.py
    >>>

    runtime {
        docker: strelka_docker
        bootDiskSizeGb: boot_disk_gb
        memory: mem_gb + " GB"
        disks: "local-disk " + disk_gb + " SSD"
        preemptible: preemptible
        maxRetries: max_retries
        cpu: cpu
    }

    output {
        File manta_vcf = "output/results/variants/candidateSmallIndels.vcf.gz"
        File manta_vcf_idx = "output/results/variants/candidateSmallIndels.vcf.gz.tbi"
    }
}


task Strelka {
    input {
        File ref_fasta
        File ref_fai
        File ref_dict

        File tumor_bam
        File tumor_bai
        File normal_bam
        File normal_bai
        File intervals_bed
        File intervals_bed_idx
        File manta_vcf
        File manta_vcf_idx

        String strelka_extra_args

        String strelka_docker

        Int cpu = 4
        Int mem_gb = 16
        Int disk_gb = 1000
        Int boot_disk_gb = 10
        Int max_retries = 0
        Int preemptible = 0
    }

    command <<<
        configureStrelkaSomaticWorkflow.py \
                --normalBam ~{normal_bam} \
                --tumorBam ~{tumor_bam} \
                --referenceFasta ~{ref_fasta} \
                --indelCandidates ~{manta_vcf} \
                --runDir output \
                --callRegions ~{intervals_bed} \
                ~{strelka_extra_args}

        echo "contents of output directory after configuring workflow:"
        ls output

        output/runWorkflow.py -m local -j ~{cpu}

        echo "contents of output directory after running workflow:"
        ls output

        echo "contents of output/results:"
        ls output/results

        echo "contents of output/results/variants:"
        ls output/results/variants
    >>>

    runtime {
        docker: strelka_docker
        bootDiskSizeGb: boot_disk_gb
        memory: mem_gb + " GB"
        disks: "local-disk " + disk_gb + " SSD"
        preemptible: preemptible
        maxRetries: max_retries
        cpu: cpu
    }

    output {
        File snvs_vcf = "output/results/variants/somatic.snvs.vcf.gz"
        File snvs_vcf_idx = "output/results/variants/somatic.snvs.vcf.gz.tbi"
        File indels_vcf = "output/results/variants/somatic.indels.vcf.gz"
        File indels_vcf_idx = "output/results/variants/somatic.indels.vcf.gz.tbi"
    }
}

task Merge {
    input {
        String gatk_docker
        File snvs_vcf
        File snvs_vcf_idx
        File indels_vcf
        File indels_vcf_idx

        Int cpu = 1
        Int mem_gb = 4
        Int disk_gb = 100
        Int boot_disk_gb = 10
        Int max_retries = 0
        Int preemptible = 0
    }

    command <<<
        gatk MergeVcfs -I ~{snvs_vcf} -I ~{indels_vcf} -O merged.vcf
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
        File merged_vcf = "merged.vcf"
        File merged_vcf_idx = "merged.vcf.idx"
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