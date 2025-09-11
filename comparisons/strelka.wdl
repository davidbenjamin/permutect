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
        String manta_extra_args
        String strelka_extra_args

        String gatk_docker = "us.gcr.io/broad-gatk/gatk"
        String strelka_docker
        String? gcs_project_for_requester_pays
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
            manta_vcf = Manta.manta_vcf,
            manta_vcf_idx = Manta.manta_vcf_idx,
            strelka_extra_args = strelka_extra_args,
            strelka_docker = strelka_docker
    }

    output {
        File snvs_vcf = Strelka.snvs_vcf
        File snvs_vcf_idx = Strelka.snvs_vcf_idx
        File indels_vcf = Strelka.indels_vcf
        File indels_vcf_idx = Strelka.indels_vcf_idx
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
        String manta_extra_args

        String strelka_docker

        Int cpu = 2
        Int mem_gb = 4
        Int disk_gb = 100
        Int boot_disk_gb = 4
        Int max_retries = 0
        Int preemptible = 0
    }

    command <<<
        python configManta.py \
                --normalBam ~{normal_bam} \
                --tumorBam ~{tumor_bam} \
                --referenceFasta ~{ref_fasta} \
                --runDir output \
                --callRegions ~{intervals_bed} \
                ~{manta_extra_args}

        ls output

        python output/runWorkflow.py -j ~{cpu}

        ls output
        ls output/results
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
        File manta_vcf = "output/results/variants/candidateSmallIndes.vcf.gz"
        File manta_vcf_idx = "output/results/variants/candidateSmallIndes.vcf.gz.tbi"
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
        File manta_vcf
        File manta_vcf_idx

        String strelka_extra_args

        String strelka_docker

        Int cpu = 2
        Int mem_gb = 4
        Int disk_gb = 100
        Int boot_disk_gb = 4
        Int max_retries = 0
        Int preemptible = 0
    }

    command <<<
        python configManta.py \
                --normalBam ~{normal_bam} \
                --tumorBam ~{tumor_bam} \
                --referenceFasta ~{ref_fasta} \
                --indelCandidates ~{manta_vcf} \
                --runDir output \
                --callRegions ~{intervals_bed} \
                ~{strelka_extra_args}

        ls output

        python output/runWorkflow.py -m local -j ~{cpu}

        ls output
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
        File snvs_vcf = "output/somatic.snvs.vcf.gz"
        File snvs_vcf_idx = "output/somatic.snvs.vcf.gz.tbi"
        File indels_vcf = "output/somatic.indels.vcf.gz"
        File indels_vcf_idx = "output/somatic.indels.vcf.gz.tbi"
    }
}