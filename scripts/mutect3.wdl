version 1.0

import "https://api.firecloud.org/ga4gh/v1/tools/davidben:mutect2/versions/1/plain-WDL/descriptor" as m2


workflow Mutect3 {
    input {
        File mutect3_model

        File? intervals
        File? masks
        File ref_fasta
        File ref_fai
        File ref_dict
        Int scatter_count
        File primary_bam
        File primary_bai
        File? control_bam
        File? control_bai
        File? gnomad
        File? gnomad_idx
        File? variants_for_contamination
        File? variants_for_contamination_idx
        File? realignment_index_bundle
        String? realignment_extra_args
        Boolean? run_orientation_bias_mixture_model_filter
        String? m2_extra_args
        String? split_intervals_extra_args
        Int batch_size


        String gatk_docker
        File? gatk_override
        String mutect3_docker
        Int? preemptible
        Int? max_retries
    }

    call m2.Mutect2 {
        input:
            make_m3_training_dataset = false,
            make_m3_test_dataset = true,
            intervals = intervals,
            ref_fasta = ref_fasta,
            ref_fai = ref_fai,
            ref_dict = ref_dict,
            tumor_reads = primary_bam,
            tumor_reads_index = primary_bai,
            normal_reads = control_bam,
            normal_reads_index = control_bai,

            scatter_count = scatter_count,
            gnomad = gnomad,
            gnomad_idx = gnomad_idx,
            variants_for_contamination = variants_for_contamination,
            variants_for_contamination_idx = variants_for_contamination_idx,
            realignment_index_bundle = realignment_index_bundle,
            realignment_extra_args = realignment_extra_args,
            run_orientation_bias_mixture_model_filter = run_orientation_bias_mixture_model_filter,
            m2_extra_args = m2_extra_args,
            make_bamout = false,

            gatk_docker = gatk_docker,
            gatk_override = gatk_override,
            preemptible = preemptible,
            max_retries = max_retries
    }

    call Mutect3Filtering {
        input:
            mutect2_vcf = Mutect2.filtered_vcf,
            mutect2_vcf_idx = Mutect2.filtered_vcf_idx,
            mutect3_model = mutect3_model,
            test_dataset = select_first([Mutect2.m3_dataset]),
            maf_segments = Mutect2.maf_segments,
            mutect_stats = Mutect2.mutect_stats,
            batch_size = batch_size,
            mutect3_docker = mutect3_docker,
    }

    call IndexVCF {
        input:
            unindexed_vcf = Mutect3Filtering.output_vcf,
            gatk_docker = gatk_docker
    }

    output {
        File output_vcf = IndexVCF.vcf
        File output_vcf_idx = IndexVCF.vcf_index
        File tensorboard_report = Mutect3Filtering.tensorboard_report
    }
}

task Mutect3Filtering {
    input {
        File mutect3_model
        File test_dataset
        File mutect2_vcf
        File mutect2_vcf_idx
        File? maf_segments
        File mutect_stats
        Int batch_size

        String mutect3_docker
        Int? preemptible
        Int? max_retries
        Int? disk_space
        Int? cpu
        Int? mem
        Boolean use_ssd = false
    }

    # Mem is in units of GB but our command and memory runtime values are in MB
    Int machine_mem = if defined(mem) then mem * 1000 else 16000
    Int command_mem = machine_mem - 500

    command <<<
        set -e

        num_ignored=`grep "callable" ~{mutect_stats} | while read name value; do echo $value; done`

        filter_variants \
            --input ~{mutect2_vcf} \
            --test_dataset ~{test_dataset} \
            --batch_size ~{batch_size} \
            --m3_model ~{mutect3_model} \
            --num_ignored_sites $num_ignored \
            --output mutect3-filtered.vcf \
            --tensorboard_dir tensorboard
    >>>

    runtime {
        docker: mutect3_docker
        bootDiskSizeGb: 12
        memory: machine_mem + " MB"
        disks: "local-disk " + select_first([disk_space, 100]) + if use_ssd then " SSD" else " HDD"
        preemptible: select_first([preemptible, 3])
        maxRetries: select_first([max_retries, 1])
        cpu: select_first([cpu, 2])
    }

    output {
        File output_vcf = "mutect3-filtered.vcf"
        File tensorboard_report = glob("tensorboard/*tfevents*")[0]
    }
}

task IndexVCF {
    input {
        File unindexed_vcf
        String gatk_docker
        Int? preemptible
        Int? max_retries
        Int? disk_space
        Int? cpu
        Int? mem
        Boolean use_ssd = false
    }

    # Mem is in units of GB but our command and memory runtime values are in MB
    Int machine_mem = if defined(mem) then mem * 1000 else 4000
    Int command_mem = machine_mem - 500

    command <<<

        cp ~{unindexed_vcf} output.vcf

        gatk --java-options "-Xmx~{command_mem}m" IndexFeatureFile -I output.vcf

        set -e
    >>>

    runtime {
        docker: gatk_docker
        bootDiskSizeGb: 12
        memory: machine_mem + " MB"
        disks: "local-disk " + select_first([disk_space, 100]) + if use_ssd then " SSD" else " HDD"
        preemptible: select_first([preemptible, 3])
        maxRetries: select_first([max_retries, 1])
        cpu: select_first([cpu, 2])
    }

    output {
        File vcf = "output.vcf"
        File vcf_index = "output.vcf.idx"
    }
}
