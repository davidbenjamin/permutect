version 1.0


workflow TrainMutect3 {
    input {
        Array[File] training_datasets
        Array[File] normal_artifact_datasets
        Int num_epochs
        Int num_normal_artifact_epochs
        Int batch_size
        Int normal_artifact_batch_size
        Float dropout_p
        Float reweighting_range
        Array[Int] hidden_read_layers
        Array[Int] hidden_info_layers
        Array[Int] aggregation_layers
        Array[Int] output_layers
        Array[Int] hidden_normal_artifact_layers

        String mutect3_docker
        Int? preemptible
        Int? max_retries
    }

    call TrainMutect3 {
        input:
            training_datasets = training_datasets,
            mutect3_docker = mutect3_docker,
            preemptible = preemptible,
            max_retries = max_retries,
            num_epochs = num_epochs,
            batch_size = batch_size,
            dropout_p = dropout_p,
            reweighting_range = reweighting_range,
            hidden_read_layers = hidden_read_layers,
            hidden_info_layers = hidden_info_layers,
            aggregation_layers = aggregation_layers,
            output_layers = output_layers
    }

    call TrainNormalArtifact {
        input:
            normal_artifact_datasets = normal_artifact_datasets,
            mutect3_docker = mutect3_docker,
            preemptible = preemptible,
            max_retries = max_retries,
            num_epochs = num_normal_artifact_epochs,
            batch_size = normal_artifact_batch_size,
            hidden_normal_artifact_layers = hidden_normal_artifact_layers,
    }


    output {
        File mutect3_model = TrainMutect3.mutect3_model
        File normal_artifact_model = TrainNormalArtifact.normal_artifact_model
        File training_report = TrainMutect3.training_report
        File normal_artifact_training_report = TrainNormalArtifact.training_report
    }
}

task TrainMutect3 {
    input {
        Array[File] training_datasets

        Int num_epochs
        Int batch_size
        Float dropout_p
        Float reweighting_range
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

        train_model \
            --training-datasets ~{sep=' ' training_datasets} \
            --hidden-read-layers ~{sep=' ' hidden_read_layers} \
            --hidden-info-layers ~{sep=' ' hidden_info_layers} \
            --aggregation-layers ~{sep=' ' aggregation_layers} \
            --output-layers ~{sep=' ' output_layers} \
            --dropout-p ~{dropout_p} \
            --reweighting_range ~{reweighting_range} \
            --batch-size ~{batch_size} \
            --num-epochs ~{num_epochs} \
            --output mutect3.pt \
            --tensorboard-dir tensorboard
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
        File training_report = glob("tensorboard/*tfevents*")[0]
    }
}

task TrainNormalArtifact {
    input {
        Array[File] normal_artifact_datasets

        Int num_epochs
        Int batch_size
        Array[Int] hidden_normal_artifact_layers

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

        train_normal_artifact_model \
            --normal-artifact-datasets ~{sep=' ' normal_artifact_datasets} \
            --hidden-layers ~{sep=' ' hidden_normal_artifact_layers} \
            --batch-size ~{batch_size} \
            --num-epochs ~{num_epochs} \
            --output normal_artifact.pt \
            --tensorboard-dir tensorboard
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
        File normal_artifact_model = "normal_artifact.pt"
        File training_report = glob("tensorboard/*tfevents*")[0]
    }
}
