version 1.0


workflow TrainMutect3 {
    input {
        File train_pickle
        File normal_artifact_pickle
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

task TrainMutect3 {
    input {
        File train_pickle
        File normal_artifact_pickle

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
            --training-pickles ~{train_pickle} \
            --normal-artifact-pickles ~{normal_artifact_pickle} \
            --hidden-read-layers ~{sep=' ' hidden_read_layers} \
            --hidden-info-layers ~{sep=' ' hidden_info_layers} \
            --aggregation-layers ~{sep=' ' aggregation_layers} \
            --output-layers ~{sep=' ' output_layers} \
            --dropout-p ~{dropout_p} \
            --batch_size ~{batch_size} \
            --num_epochs ~{num_epochs} \
            --output mutect3.pt \
            --report_pdf training-report.pdf
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
        File mutect3_model = "mutect3.pt"
        File training_report = "training-report.pdf"
    }
}
