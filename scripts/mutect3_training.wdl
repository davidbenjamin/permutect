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
        Array[Int] read_layers
        Array[Int] info_layers
        Array[Int] aggregation_layers
        Array[Int] hidden_normal_artifact_layers
        String? train_m3_extra_args
        String? train_na_extra_args

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
            read_layers = read_layers,
            info_layers = info_layers,
            aggregation_layers = aggregation_layers,
            extra_args = train_m3_extra_args
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
            extra_args = train_na_extra_args
    }


    output {
        File mutect3_model = TrainMutect3.mutect3_model
        File normal_artifact_model = TrainNormalArtifact.normal_artifact_model
        Array[File] training_tensorboard = TrainMutect3.training_tensorboard
        Array[File] na_tensorboard = TrainNormalArtifact.na_tensorboard
    }
}

task TrainMutect3 {
    input {
        Array[File] training_datasets

        Int num_epochs
        Int batch_size
        Float dropout_p
        Float reweighting_range
        Array[Int] read_layers
        Array[Int] info_layers
        Array[Int] aggregation_layers
        String? extra_args

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
            --training_datasets ~{sep=' ' training_datasets} \
            --read_layers ~{sep=' ' read_layers} \
            --info_layers ~{sep=' ' info_layers} \
            --aggregation_layers ~{sep=' ' aggregation_layers} \
            --dropout_p ~{dropout_p} \
            --reweighting_range ~{reweighting_range} \
            --batch_size ~{batch_size} \
            --num_epochs ~{num_epochs} \
            --output mutect3.pt \
            --tensorboard_dir tensorboard \
            ~{extra_args}
    >>>

    runtime {
        docker: mutect3_docker
        bootDiskSizeGb: 12
        memory: machine_mem + " MB"
        disks: "local-disk " + select_first([disk_space, 100]) + if use_ssd then " SSD" else " HDD"
        preemptible: select_first([preemptible, 10])
        maxRetries: select_first([max_retries, 0])
        cpu: select_first([cpu, 1])
        gpuType: "nvidia-tesla-t4"
        gpuCount: 1
    }

    output {
        File mutect3_model = "mutect3.pt"
        Array[File] training_tensorboard = glob("tensorboard/*")
    }
}

task TrainNormalArtifact {
    input {
        Array[File] normal_artifact_datasets

        Int num_epochs
        Int batch_size
        Array[Int] hidden_normal_artifact_layers
        String? extra_args

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
            --normal_artifact_datasets ~{sep=' ' normal_artifact_datasets} \
            --hidden_layers ~{sep=' ' hidden_normal_artifact_layers} \
            --batch_size ~{batch_size} \
            --num_epochs ~{num_epochs} \
            --output normal_artifact.pt \
            --tensorboard_dir tensorboard \
            ~{extra_args}
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
        Array[File] na_tensorboard = glob("tensorboard/*")
    }
}
