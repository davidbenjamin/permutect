version 1.0

import "https://raw.githubusercontent.com/gatk-workflows/gatk4-somatic-snvs-indels/2.6.0/mutect2.wdl" as m2


workflow Mutect3Filtering {
    input {
        File mutect3_model
        Int num_epochs
        Int batch_size
        Float dropout_p
        Array[Int] hidden_read_layers
        Array[Int] hidden_info_layers
        Array[Int] aggregation_layers
        Array[Int] output_layers

        String mutect3_docker
        Int? preemptible
        Int? max_retries
    }

    call TrainMutect3 {
        input:
            train_pickle = train_pickle,
            normal_artifact_pickle = normal_artifact_pickle,
            mutect3_docker = mutect3_docker,
            preemptible = preemptible,
            max_retries = max_retries,
            num_epochs = num_epochs,
            batch_size = batch_size,
            dropout_p = dropout_p,
            hidden_read_layers = hidden_read_layers,
            hidden_info_layers = hidden_info_layers,
            aggregation_layers = aggregation_layers,
            output_layers = output_layers
    }


    output {
        File mutect3_model = TrainMutect3.mutect3_model
        File training_report = TrainMutect3.training_report
    }
}

task Mutect3Filtering {
    input {
        File mutect3_model
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

        train_and_save_model \
            --input ~{mutect2_vcf} \
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
        preemptible: select_first([preemptible, 10])
        maxRetries: select_first([max_retries, 0])
        cpu: select_first([cpu, 1])
    }

    output {
        File output_vcf = "mutect3-filtered.vcf"
        File training_report = "training-report.pdf"
        File report_pdf = "report.pdf"
        File roc_pdf = "roc.pdf"
    }
}




filter_variants \
    --trained_m3_model saved/dream1-saved.pt \
    --tumor "synthetic.challenge.set1.tumor" \
    --normal "synthetic.challenge.set1.normal" \
