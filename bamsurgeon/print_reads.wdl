version 1.0

workflow PrintReads {
    input {
        File ref_fasta
        File ref_fai
        File ref_dict

        File base_bam
        File base_bam_index
        String intervals

        String gatk_docker = "us.gcr.io/broad-gatk/gatk"
        String? gcs_project_for_requester_pays
    }

    call PrintReads {
        input:
            original_bam = base_bam,
            original_bam_idx = base_bam_index,
            ref_fasta = ref_fasta,
            ref_fai = ref_fai,
            ref_dict = ref_dict,
            gatk_docker = gatk_docker,
            intervals = intervals
    }

    output {
        File output_bam = PrintReads.output_bam
        File output_bam_idx = PrintReads.output_bam_idx
    }
}

task PrintReads {
    input {
        String gatk_docker
        String? gcs_project_for_requester_pays
        File original_bam       # this can be a BAM or CRAM
        File original_bam_idx
        File ref_fasta          # GATK PrintReads requires a reference for CRAMs
        File ref_fai
        File ref_dict
        String intervals

        Int cpu = 2
        Int mem_gb = 4
        Int disk_gb = 100
        Int boot_disk_gb = 10
        Int max_retries = 0
        Int preemptible = 1
    }

    parameter_meta{
        ref_fasta: {localization_optional: true}
        ref_fai: {localization_optional: true}
        ref_dict: {localization_optional: true}
        original_bam: {localization_optional: true}
        original_bam_idx: {localization_optional: true}
    }

    command <<<
        gatk PrintReads -R ~{ref_fasta} -I ~{original_bam} -L ~{intervals} -O output.bam \
            ~{"--gcs-project-for-requester-pays " + gcs_project_for_requester_pays}
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
        File output_bam = "output.bam"
        File output_bam_idx = "output.bai"
    }
}