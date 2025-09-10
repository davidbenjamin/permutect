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

        String gatk_docker = "us.gcr.io/broad-gatk/gatk"
        String strelka_docker
        String? gcs_project_for_requester_pays
    }

    call IntervalListToBed {
        input:
            gatk_docker = gatk_docker,
            intervals = intervals
    }

    output {
        
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
        File output_bed = "intervals.bed"
    }
}