version 1.0

workflow HaplotypeCaller {
    input {
        File? intervals
        File ref_fasta
        File ref_fai
        File ref_dict
        File bam
        File bai

        # runtime
        File? gatk_override
        Int scatter_count

        String? gcs_project_for_requester_pays

    }


    call SplitIntervals {
        input:
            intervals = intervals,
            ref_fasta = ref_fasta,
            ref_fai = ref_fai,
            ref_dict = ref_dict,
            scatter_count = scatter_count
    }

    scatter (subintervals in SplitIntervals.interval_files ) {
        call HC {
            input:
                intervals = subintervals,
                ref_fasta = ref_fasta,
                ref_fai = ref_fai,
                ref_dict = ref_dict,
                bam = bam,
                bai = bai,
                gatk_override = gatk_override,
                gcs_project_for_requester_pays = gcs_project_for_requester_pays
        }
    }

    call MergeVCFs {
        input:
            input_vcfs = HC.output_vcf,
            input_vcf_indices = HC.output_vcf_idx
    }

    output {
        File output_vcf = MergeVCFs.merged_vcf
        File output_vcf_idx = MergeVCFs.merged_vcf_idx

    }
}

task SplitIntervals {
    input {
      File? intervals
      File ref_fasta
      File ref_fai
      File ref_dict
      Int scatter_count
      String? split_intervals_extra_args

    }

    command {
        set -e

        mkdir interval-files
        gatk SplitIntervals \
            -R ~{ref_fasta} \
            ~{"-L " + intervals} \
            -scatter ~{scatter_count} \
            -O interval-files \
            ~{split_intervals_extra_args}
        cp interval-files/*.interval_list .
    }

    runtime {
        docker: "broadinstitute/gatk:latest"
        bootDiskSizeGb: 6
        memory: 4 + " GB"
        disks: "local-disk " + 10 + " SSD"
        preemptible: 0
        maxRetries: 0
        cpu: 1
    }

    output {
        Array[File] interval_files = glob("*.interval_list")
    }
}

task HC {
    input {
        File? intervals
        File ref_fasta
        File ref_fai
        File ref_dict
        File bam
        File bai

        String? extra_args
        File? gatk_override
        String? gcs_project_for_requester_pays

        # runtime
        Int? preemptible
        Int? disk_space
    }

    parameter_meta{
        intervals: {localization_optional: true}
        ref_fasta: {localization_optional: true}
        ref_fai: {localization_optional: true}
        ref_dict: {localization_optional: true}
        bam: {localization_optional: true}
        bai: {localization_optional: true}
    }

    command <<<
        set -e

        export GATK_LOCAL_JAR=~{default="/root/gatk.jar" gatk_override}

        gatk HaplotypeCaller \
            -R ~{ref_fasta} \
            -I ~{bam} \
            ~{"-L " + intervals} \
            -O output.vcf \
            ~{extra_args} \
            ~{"--gcs-project-for-requester-pays " + gcs_project_for_requester_pays}
    >>>

    runtime {
        docker: "broadinstitute/gatk:latest"
        bootDiskSizeGb: 10
        memory: 8 + " GB"
        disks: "local-disk " + select_first([disk_space, 100]) + " SSD"
        preemptible: select_first([preemptible, 1])
        maxRetries: 0
        cpu: 1
    }

    output {
        File output_vcf = "output.vcf"
        File output_vcf_idx = "output.vcf.idx"
    }
}

task MergeVCFs {
    input {
      Array[File] input_vcfs
      Array[File] input_vcf_indices
    }

    command {
        set -e
        gatk MergeVcfs -I ~{sep=' -I ' input_vcfs} -O merged.vcf
    }

    runtime {
        docker: "broadinstitute/gatk:latest"
        bootDiskSizeGb: 10
        memory: 8 + " GB"
        disks: "local-disk " + 100 + " SSD"
        preemptible: 0
        maxRetries: 0
        cpu: 1
    }

    output {
        File merged_vcf = "merged.vcf"
        File merged_vcf_idx = "merged.vcf.idx"
    }
}


