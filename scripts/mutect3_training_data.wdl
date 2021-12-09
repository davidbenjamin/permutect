version 1.0

import "https://raw.githubusercontent.com/gatk-workflows/gatk4-somatic-snvs-indels/2.6.0/mutect2.wdl" as m2

workflow Mutect3TrainingData {
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
        File? pon
        File? gnomad
        File? variants_for_contamination
        String ref_downsample
        Boolean? run_orientation_bias_mixture_model_filter
        File? realignment_index_bundle
        String? realignment_extra_args
        String? m2_extra_args
        String? m2_extra_filtering_args
        String? normal_artifact_extra_args
        String? split_intervals_extra_args
        Boolean? make_bamout
        Boolean? compress_vcfs
        Int? max_na_records

        # runtime
        String gatk_docker
        String mutect3_docker
        File? gatk_override
        Int? preemptible
        Int? max_retries
    }

    String m2_extra_args_with_training_mode = select_first([m2_extra_args, ""]) + " --training-data-mode --training-data-mode-ref-downsample " + ref_downsample

    Runtime small_runtime = {"gatk_docker": gatk_docker, "gatk_override": gatk_override,
                                "max_retries": 2, "preemptible": 0, "cpu": 2,
                                "machine_mem": 4000, "command_mem": 3500,
                                "disk": 100, "boot_disk_size": 12}

    # call on the tumor (with normal if present) to get tumor read data and M2 filtering
    call m2.Mutect2 as mutect2 {
        input:
            ref_fasta = ref_fasta,
            ref_fai = ref_fai,
            ref_dict = ref_dict,
            scatter_count = scatter_count,
            tumor_reads = primary_bam,
            tumor_reads_index = primary_bai,
            normal_reads = control_bam,
            normal_reads_index = control_bai,
            intervals = intervals,
            pon = pon,
            gnomad = gnomad,
            variants_for_contamination = variants_for_contamination,
            run_orientation_bias_mixture_model_filter = run_orientation_bias_mixture_model_filter,
            realignment_index_bundle = realignment_index_bundle,
            realignment_extra_args = realignment_extra_args,
            preemptible = preemptible,
            max_retries = max_retries,
            m2_extra_args = m2_extra_args_with_training_mode,
            m2_extra_filtering_args = m2_extra_filtering_args,
            make_bamout = make_bamout,
            compress_vcfs = compress_vcfs,
            gatk_override = gatk_override,
            gatk_docker = gatk_docker
    }

    call GetSampleNames {
        input:
            ref_fasta = ref_fasta,
            ref_fai = ref_fai,
            ref_dict = ref_dict,
            tumor_reads = primary_bam,
            tumor_reads_index = primary_bai,
            normal_reads = control_bam,
            normal_reads_index = control_bai,
            preemptible = preemptible,
            max_retries = max_retries,
            gatk_override = gatk_override,
            gatk_docker = gatk_docker
    }

    call MakeTrainingTensors {
        input:
            mutect2_vcf = mutect2.filtered_vcf,
            mutect2_vcf_idx = mutect2.filtered_vcf_idx,
            tumor_sample = GetSampleNames.tumor_sample,
            normal_sample = GetSampleNames.normal_sample,
            mutect3_docker = mutect3_docker
    }


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
        call GetNormalArtifactData {
            input:
                ref_fasta = ref_fasta,
                ref_fai = ref_fai,
                ref_dict = ref_dict,
                tumor_reads = primary_bam,
                tumor_reads_index = primary_bai,
                normal_reads = select_first([control_bam]),
                normal_reads_index = select_first([control_bai]),
                intervals = subintervals,
                preemptible = preemptible,
                max_retries = max_retries,
                extra_args = normal_artifact_extra_args,
                gatk_override = gatk_override,
                gatk_docker = gatk_docker
        }
    }

    call MergeNormalArtifactData {
        input:
            input_tables = GetNormalArtifactData.table,
            runtime_params = small_runtime
    }

    call MakeNormalArtifactTensors {
        input:
            input_table = MergeNormalArtifactData.merged_table,
            mutect3_docker = mutect3_docker,
            max_records = max_na_records
    }

    output {
        File train_pickle = MakeTrainingTensors.train_pickle
        File normal_artifact_pickle = MakeNormalArtifactTensors.normal_artifact_pickle
    }
}


task GetNormalArtifactData {
    input {
        File? intervals
        File ref_fasta
        File ref_fai
        File ref_dict
        File tumor_reads
        File tumor_reads_index
        File normal_reads
        File normal_reads_index
        String? extra_args

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
    }

    command <<<
        set -e

        export GATK_LOCAL_JAR=~{default="/root/gatk.jar" gatk_override}

        if [[ ! -z "~{normal_reads}" ]]; then
        gatk --java-options "-Xmx~{command_mem}m" GetSampleName -R ~{ref_fasta} -I ~{normal_reads} -O normal_name.txt -encode \
        ~{"--gcs-project-for-requester-pays " + gcs_project_for_requester_pays}
        normal_sample="`cat normal_name.txt`"
        fi

        gatk --java-options "-Xmx~{command_mem}m" GetNormalArtifactData \
        -R ~{ref_fasta} ~{"-L " + intervals} -I ~{tumor_reads} -I ~{normal_reads} -O normal_artifact.table \
        -normal $normal_sample \
        ~{extra_args} ~{"--gcs-project-for-requester-pays " + gcs_project_for_requester_pays}
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
        File table = "normal_artifact.table"
    }
}

task MergeNormalArtifactData {
    input {
        Array[File] input_tables
        Runtime runtime_params
    }

    command {
        set -e
        export GATK_LOCAL_JAR=~{default="/root/gatk.jar" runtime_params.gatk_override}

        gatk --java-options "-Xmx~{runtime_params.command_mem}m" GatherNormalArtifactData \
        -I ~{sep=' -I ' input_tables} \
        -O normal_artifact.table
    }

    runtime {
        docker: runtime_params.gatk_docker
        bootDiskSizeGb: runtime_params.boot_disk_size
        memory: runtime_params.machine_mem + " MB"
        disks: "local-disk " + runtime_params.disk + " HDD"
        preemptible: runtime_params.preemptible
        maxRetries: runtime_params.max_retries
        cpu: runtime_params.cpu
    }

    output {
        File merged_table = "normal_artifact.table"
    }
}

task GetSampleNames {
    input {
      File ref_fasta
      File ref_fai
      File ref_dict
      File tumor_reads
      File tumor_reads_index
      File? normal_reads
      File? normal_reads_index

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
      ref_fasta: {localization_optional: true}
      ref_fai: {localization_optional: true}
      ref_dict: {localization_optional: true}
      tumor_reads: {localization_optional: true}
      tumor_reads_index: {localization_optional: true}
      normal_reads: {localization_optional: true}
      normal_reads_index: {localization_optional: true}
    }

    command <<<
        set -e

        export GATK_LOCAL_JAR=~{default="/root/gatk.jar" gatk_override}

        gatk --java-options "-Xmx~{command_mem}m" GetSampleName -R ~{ref_fasta} -I ~{tumor_reads} -O tumor_name.txt -encode \
            ~{"--gcs-project-for-requester-pays " + gcs_project_for_requester_pays}
        tumor_command_line="-I ~{tumor_reads} -tumor `cat tumor_name.txt`"

        gatk --java-options "-Xmx~{command_mem}m" GetSampleName -R ~{ref_fasta} -I ~{normal_reads} -O normal_name.txt -encode \
            ~{"--gcs-project-for-requester-pays " + gcs_project_for_requester_pays}
        normal_command_line="-I ~{normal_reads} -normal `cat normal_name.txt`"
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
        String tumor_sample = read_string("tumor_name.txt")
        String normal_sample = read_string("normal_name.txt")
    }
}

task MakeTrainingTensors {
    input {
        File mutect2_vcf
        File mutect2_vcf_idx
        String tumor_sample
        String normal_sample

        # runtime
        String mutect3_docker
        Int? mem
        Int? preemptible
        Int? max_retries
        Int? disk_space
        Int? cpu
        Boolean use_ssd = false
    }

    # Mem is in units of GB but our command and memory runtime values are in MB
    Int machine_mem = if defined(mem) then mem * 1000 else 8000
    Int command_mem = machine_mem - 500

    command <<<
        set -e

        make_training_tensors --input ~{mutect2_vcf} --tumor ~{tumor_sample} --normal ~{normal_sample} --output "train.pickle"
    >>>

    runtime {
        docker: mutect3_docker
        bootDiskSizeGb: 12
        memory: machine_mem + " MB"
        disks: "local-disk " + select_first([disk_space, 100]) + if use_ssd then " SSD" else " HDD"
        preemptible: select_first([preemptible, 10])
        maxRetries: select_first([max_retries, 0])
        cpu: select_first([cpu, 1])
    }

    output {
        File train_pickle = "train.pickle"
    }
}

task MakeNormalArtifactTensors {
    input {
        File input_table
        String mutect3_docker
        Int? mem
        Int? preemptible
        Int? max_retries
        Int? disk_space
        Int? cpu
        Int? max_records
        Boolean use_ssd = false
    }

    # Mem is in units of GB but our command and memory runtime values are in MB
    Int machine_mem = if defined(mem) then mem * 1000 else 8000
    Int command_mem = machine_mem - 500

    command <<<
        set -e

        if [[ ! -z "~{max_records}" ]]; then
            head -n ~{max_records} ~{input_table} > truncated.txt
            input="truncated.txt
        else
            input=~{input_table}
        make_normal_artifact_tensors --table $input --output "normal_artifact.pickle"
    >>>

    runtime {
        docker: mutect3_docker
        bootDiskSizeGb: 12
        memory: machine_mem + " MB"
        disks: "local-disk " + select_first([disk_space, 100]) + if use_ssd then " SSD" else " HDD"
        preemptible: select_first([preemptible, 10])
        maxRetries: select_first([max_retries, 0])
        cpu: select_first([cpu, 1])
    }

    output {
        File normal_artifact_pickle = "normal_artifact.pickle"
    }
}