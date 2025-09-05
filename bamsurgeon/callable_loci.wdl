version 1.0

workflow CallableLoci {
    input {
        File ref_fasta
        File ref_fai
        File ref_dict

        File bam
        File bam_idx
        File? intervals
        String extra_args = ""

        String gatk_docker = "us.gcr.io/broad-gatk/gatk"
        String? gcs_project_for_requester_pays
    }

    call CallableLoci {
        input:
            bam = bam,
            bam_idx = bam_idx,
            ref_fasta = ref_fasta,
            ref_fai = ref_fai,
            ref_dict = ref_dict,
            gatk_docker = gatk_docker,
            intervals = intervals,
            extra_args = extra_args
    }

    output {
        File good_regions_bed = CallableLoci.good_regions_bed
        File bad_regions_bed = CallableLoci.bad_regions_bed
        File summary = CallableLoci.summary
    }
}

task CallableLoci {
    input {
        String gatk_docker
        String? gcs_project_for_requester_pays
        File bam       # this can be a BAM or CRAM
        File bam_idx
        File ref_fasta
        File ref_fai
        File ref_dict
        File? intervals
        String extra_args

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
        bam: {localization_optional: true}
        bam_idx: {localization_optional: true}
    }

    command <<<
        gatk CallableLoci -R ~{ref_fasta} -I ~{bam} ~{" -L " + intervals} -O callable_status.bed --summary summary.txt ~{extra_args}
        grep CALLABLE callable_status.bed > good_regions.bed
        grep -v CALLABLE callable_status.bed > bad_regions.bed
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
        File good_regions_bed = "good_regions.bed"
        File bad_regions_bed = "bad_regions.bed"
        File summary = "summary.txt"
    }
}