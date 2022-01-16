version 1.0

import "https://raw.githubusercontent.com/gatk-workflows/gatk4-somatic-snvs-indels/2.6.0/mutect2.wdl" as m2
import "https://api.firecloud.org/ga4gh/v1/tools/davidben:mutect3-dataset/versions/2/plain-WDL/descriptor" as dataset


workflow Mutect3 {
    input {
        # circumvent running Mutect2 within this workflow
        File? precomputed_mutect2_vcf
        File? precomputed_mutect2_vcf_idx

        File mutect3_model
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
        File? variants_for_contamination
        File? variants_for_contamination_idx
        File? realignment_index_bundle
        String? realignment_extra_args
        Boolean? run_orientation_bias_mixture_model_filter
        String? m2_extra_args
        String? split_intervals_extra_args
        Int? max_na_records
        Int batch_size


        String gatk_docker
        File? gatk_override
        String mutect3_docker
        Int? preemptible
        Int? max_retries
    }

    if (!defined(precomputed_mutect2_vcf)) {
        call m2.Mutect2 {
            input:
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
    }

    call dataset.Mutect3Dataset as TestDataset {
        input:
            intervals = intervals,
            masks = masks,
            ref_fasta = ref_fasta,
            ref_fai = ref_fai,
            ref_dict = ref_dict,
            scatter_count = scatter_count,
            primary_bam = primary_bam,
            primary_bai = primary_bai,
            control_bam = control_bam,
            control_bai = control_bai,
            gnomad = gnomad,
            gnomad_idx = gnomad_idx,
            m2_extra_args = m2_extra_args,
            split_intervals_extra_args = split_intervals_extra_args,
            training_data_mode = false,

            gatk_docker = gatk_docker,
            mutect3_docker = mutect3_docker,
            gatk_override = gatk_override,
            preemptible = preemptible,
            max_retries = max_retries
    }

    call Mutect3Filtering {
        input:
            mutect2_vcf = select_first([precomputed_mutect2_vcf, Mutect2.filtered_vcf]),
            mutect2_vcf_idx = select_first([precomputed_mutect2_vcf_idx, Mutect2.filtered_vcf_idx]),
            mutect3_model = mutect3_model,
            test_dataset = TestDataset.mutect3Dataset,
            batch_size = batch_size,
            mutect3_docker = mutect3_docker,
    }

    output {
        File output_vcf = Mutect3Filtering.output_vcf
        File report_pdf = Mutect3Filtering.report_pdf
        File roc_pdf = Mutect3Filtering.roc_pdf
    }
}

task Mutect3Filtering {
    input {
        File mutect3_model
        File test_dataset
        File mutect2_vcf
        File mutect2_vcf_idx
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

        filter_variants \
            --input ~{mutect2_vcf} \
            --test_dataset ~{test_dataset} \
            --batch_size ~{batch_size} \
            --trained_m3_model ~{mutect3_model} \
            --batch_size ~{batch_size} \
            --output mutect3-filtered.vcf \
            --report_pdf report.pdf \
            --roc_pdf roc.pdf
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
        File report_pdf = "report.pdf"
        File roc_pdf = "roc.pdf"
    }
}
