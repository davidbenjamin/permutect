version 1.0


workflow TrainArtifactModel {
    input {
        File train_tar
        File? pretrained_model
        Int num_epochs
        Int batch_size
        Int inference_batch_size
        Int? num_workers
        Float dropout_p
        Float reweighting_range
        Array[Int] read_layers
        Int self_attention_hidden_dimension
        Int num_self_attention_layers
        Array[Int] info_layers
        Array[Int] aggregation_layers
        Array[Int] calibration_layers
        Array[String] ref_seq_layer_strings
        String? extra_args
        Int? gpu_count

        String permutect_docker
        Int? preemptible
        Int? max_retries
    }

    call Train {
        input:
            train_tar = train_tar,
            pretrained_model = pretrained_model,
            permutect_docker = permutect_docker,
            preemptible = preemptible,
            max_retries = max_retries,
            num_epochs = num_epochs,
            batch_size = batch_size,
            inference_batch_size = inference_batch_size,
            num_workers = num_workers,
            gpu_count = gpu_count,
            dropout_p = dropout_p,
            reweighting_range = reweighting_range,
            read_layers = read_layers,
            self_attention_hidden_dimension = self_attention_hidden_dimension,
            num_self_attention_layers = num_self_attention_layers,
            info_layers = info_layers,
            aggregation_layers = aggregation_layers,
            calibration_layers = calibration_layers,
            ref_seq_layer_strings = ref_seq_layer_strings,
            extra_args = extra_args
    }

    output {
        File permutect_model = Train.permutect_model
        File training_tensorboard_tar = Train.tensorboard_tar
    }
}


task Train {
    input {
        File train_tar
        File? pretrained_model

        Int num_epochs
        Int batch_size
        Int inference_batch_size
        Int? num_workers
        Int? gpu_count
        Float dropout_p
        Float reweighting_range
        Array[Int] read_layers
        Int self_attention_hidden_dimension
        Int num_self_attention_layers
        Array[Int] info_layers
        Array[Int] aggregation_layers
        Array[Int] calibration_layers
        Array[String] ref_seq_layer_strings

        String? extra_args

        String permutect_docker
        Int? preemptible
        Int? max_retries
        Int? disk_space
        Int? cpu
        Int? mem
    }

    # Mem is in units of GB but our command and memory runtime values are in MB
    Int machine_mem = if defined(mem) then mem * 1000 else 16000
    Int command_mem = machine_mem - 500

    command <<<
        set -e

        train_artifact_model \
            --train_tar ~{train_tar} \
            ~{"--saved_model " + pretrained_model} \
            --read_layers ~{sep=' ' read_layers} \
            --self_attention_hidden_dimension ~{self_attention_hidden_dimension} \
            --num_self_attention_layers ~{num_self_attention_layers} \
            --info_layers ~{sep=' ' info_layers} \
            --aggregation_layers ~{sep=' ' aggregation_layers} \
            --calibration_layers ~{sep=' ' calibration_layers} \
            --ref_seq_layer_strings ~{sep=' ' ref_seq_layer_strings} \
            --dropout_p ~{dropout_p} \
            --reweighting_range ~{reweighting_range} \
            --batch_size ~{batch_size} \
            --inference_batch_size ~{inference_batch_size} \
            ~{"--num_workers " + num_workers} \
            --num_epochs ~{num_epochs} \
            --output artifact_model.pt \
            --tensorboard_dir tensorboard \
            ~{extra_args}

        tar cvf tensorboard.tar tensorboard/
    >>>

    runtime {
        docker: permutect_docker
        bootDiskSizeGb: 12
        memory: machine_mem + " MB"
        disks: "local-disk " + select_first([disk_space, 100]) + " SSD"
        preemptible: select_first([preemptible, 0])
        maxRetries: select_first([max_retries, 0])
        cpu: select_first([cpu, 1])
        gpuType: "nvidia-tesla-t4"
        gpuCount: select_first([gpu_count, 1])
        nvidiaDriverVersion: "535.183.01"
        zones : ["us-central1-a", "us-central1-b", "us-central1-c"]
    }

    output {
        File permutect_model = "artifact_model.pt"
        File tensorboard_tar = "tensorboard.tar"
    }
}

