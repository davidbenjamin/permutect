version 1.0


workflow TrainArtifactModel {
    input {
        String permutect_docker
        File train_tar
        File? pretrained_model

        Int? num_epochs
        Int? batch_size
        Int? num_workers

        Array[Int]? read_layers
        Array[Int]? info_layers
        Array[String]? ref_seq_layer_strings
        Array[Int]? aggregation_layers
        Int? self_attention_hidden_dimension
        Int? num_self_attention_layers
        Float? dropout_p
        String? extra_args
    }

    call Train {
        input:
            permutect_docker = permutect_docker,
            train_tar = train_tar,
            pretrained_model = pretrained_model,
            num_epochs = num_epochs,
            batch_size = batch_size,
            num_workers = num_workers,
            read_layers = read_layers,
            info_layers = info_layers,
            ref_seq_layer_strings = ref_seq_layer_strings,
            aggregation_layers = aggregation_layers,
            self_attention_hidden_dimension = self_attention_hidden_dimension,
            num_self_attention_layers = num_self_attention_layers,
            dropout_p = dropout_p,
            extra_args = extra_args
    }

    output {
        File artifact_model = Train.artifact_model
        File training_tensorboard_tar = Train.tensorboard_tar
    }
}


task Train {
    input {
        String permutect_docker
        File train_tar
        File? pretrained_model

        Int? num_epochs
        Int? batch_size
        Int? num_workers

        Array[Int]? read_layers
        Array[Int]? info_layers
        Array[String]? ref_seq_layer_strings
        Array[Int]? aggregation_layers
        Int? self_attention_hidden_dimension
        Int? num_self_attention_layers
        Float? dropout_p
        String? extra_args

        Int? gpu_count
        Int? preemptible
        Int? max_retries
        Int? disk_space
        Int? cpu
        Int? mem
    }

    command <<<
        set -e

        train_artifact_model \
            --train_tar ~{train_tar} \
            ~{"--pretrained_artifact_model " + pretrained_model} \
            ~{true="--read_layers " false="" defined(read_layers)}~{sep=" " read_layers} \
            ~{true="--info_layers " false="" defined(info_layers)}~{sep=" " info_layers} \
            ~{true="--ref_seq_layer_strings " false="" defined(ref_seq_layer_strings)}~{sep=" " ref_seq_layer_strings} \
            ~{true="--aggregation_layers " false="" defined(aggregation_layers)}~{sep=" " aggregation_layers} \
            ~{"--self_attention_hidden_dimension " + self_attention_hidden_dimension} \
            ~{"--num_self_attention_layers " + num_self_attention_layers} \
            ~{"--dropout_p " + dropout_p} \
            ~{"--batch_size " + batch_size} \
            ~{"--num_workers " + num_workers} \
            ~{"--num_epochs " + num_epochs} \
            --output artifact_model.pt \
            --tensorboard_dir tensorboard \
            ~{extra_args}

        tar cvf tensorboard.tar tensorboard/
    >>>

    runtime {
        docker: permutect_docker
        bootDiskSizeGb: 12
        memory: mem + " GB"
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
        File artifact_model = "artifact_model.pt"
        File tensorboard_tar = "tensorboard.tar"
    }
}

