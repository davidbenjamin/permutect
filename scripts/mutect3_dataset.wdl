version 1.0

import "https://raw.githubusercontent.com/gatk-workflows/gatk4-somatic-snvs-indels/2.6.0/mutect2.wdl" as m2

workflow Mutect3Dataset {
    input {
        File? intervals
        File? masks
        File ref_fasta
        File ref_fai
        File ref_dict
        Int scatter_count
        File primary_bam
        File primary_bai
        File control_bam
        File control_bai
        File? gnomad
        File? gnomad_idx
        String? m2_extra_args
        String? split_intervals_extra_args
        Boolean training_data_mode

        # runtime
        String gatk_docker
        String mutect3_docker
        File? gatk_override
        Int? preemptible
        Int? max_retries
    }

    Runtime small_runtime = {"gatk_docker": gatk_docker, "gatk_override": gatk_override,
                                "max_retries": 2, "preemptible": 0, "cpu": 2,
                                "machine_mem": 4000, "command_mem": 3500,
                                "disk": 100, "boot_disk_size": 12}

    call m2.SplitIntervals as Split {
        input:
            intervals = intervals,
            ref_fasta = ref_fasta,
            ref_fai = ref_fai,
            ref_dict = ref_dict,
            scatter_count = scatter_count,
            split_intervals_extra_args = split_intervals_extra_args,
            runtime_params = small_runtime
    }

    scatter (subintervals in Split.interval_files ) {
        call MakeDataset {
            input:
                intervals = subintervals,
                ref_fasta = ref_fasta,
                ref_fai = ref_fai,
                ref_dict = ref_dict,
                tumor_reads = primary_bam,
                tumor_reads_index = primary_bai,
                normal_reads = control_bam,
                normal_reads_index = control_bai,
                gnomad = gnomad,
                gnomad_idx = gnomad_idx,
                m2_extra_args = m2_extra_args,
                training_data_mode = training_data_mode,

                preemptible = preemptible,
                max_retries = max_retries,
                gatk_override = gatk_override,
                gatk_docker = gatk_docker
        }
    }

    call Concatenate {
        input:
            input_files = MakeDataset.dataset,
            gatk_docker = gatk_docker
    }


    output {
        File mutect3Dataset = Concatenate.concatenated
    }
}

task MakeDataset {
    input {
        File? intervals
        File ref_fasta
        File ref_fai
        File ref_dict
        File tumor_reads
        File tumor_reads_index
        File? normal_reads
        File? normal_reads_index
        File? gnomad
        File? gnomad_idx
        String? m2_extra_args
        Boolean training_data_mode


      File? gatk_override
      String? gcs_project_for_requester_pays

      # runtime
      String gatk_docker
      Int? mem
      Int? preemptible
      Int? max_retries
      Int? disk_space
      Int? cpu
      Boolean use_ssd = false
    }

    # Mem is in units of GB but our command and memory runtime values are in MB
    Int machine_mem = if defined(mem) then mem * 1000 else 3500
    Int command_mem = machine_mem - 500

    parameter_meta{
      intervals: {localization_optional: true}
      ref_fasta: {localization_optional: true}
      ref_fai: {localization_optional: true}
      ref_dict: {localization_optional: true}
      tumor_reads: {localization_optional: true}
      tumor_reads_index: {localization_optional: true}
      normal_reads: {localization_optional: true}
      normal_reads_index: {localization_optional: true}
      gnomad: {localization_optional: true}
      gnomad_idx: {localization_optional: true}
    }

    command <<<
        set -e

        export GATK_LOCAL_JAR=~{default="/root/gatk.jar" gatk_override}

        if [[ ! -z "~{normal_reads}" ]]; then
            gatk --java-options "-Xmx~{command_mem}m" GetSampleName -R ~{ref_fasta} -I ~{normal_reads} -O normal_name.txt -encode \
            ~{"--gcs-project-for-requester-pays " + gcs_project_for_requester_pays}
            normal_command_line="-I ~{normal_reads} -normal `cat normal_name.txt`"
        fi

        gatk --java-options "-Xmx~{command_mem}m" Mutect2 \
            -R ~{ref_fasta} \
            -I ~{tumor_reads} \
            $normal_command_line \
            ~{"--germline-resource " + gnomad} \
            ~{"-L " + intervals} \
            -O output.vcf \
            --mutect3-dataset dataset.txt \
            ~{true='--mutect3-training-mode' false='' training_data_mode} \
            ~{m2_extra_args} \
            ~{"--gcs-project-for-requester-pays " + gcs_project_for_requester_pays}

    >>>

    runtime {
        docker: gatk_docker
        bootDiskSizeGb: 12
        memory: machine_mem + " MB"
        disks: "local-disk " + select_first([disk_space, 100]) + if use_ssd then " SSD" else " HDD"
        preemptible: select_first([preemptible, 10])
        maxRetries: select_first([max_retries, 0])
        cpu: select_first([cpu, 1])
    }

    output {
        File dataset = "dataset.txt"
    }
}


task Concatenate {
    input {
        Array[File] input_files
        Int? mem
        String gatk_docker
    }

    Int machine_mem = if defined(mem) then mem * 1000 else 7000

    command {
        cat ~{sep=' ' input_files} > output.txt
    }

    runtime {
        docker: gatk_docker
        bootDiskSizeGb: 12
        memory: machine_mem + " MB"
        disks: "local-disk 100 HDD"
        preemptible: 1
        maxRetries: 1
        cpu: 2
    }

    output {
        File concatenated = "output.txt"
    }
}