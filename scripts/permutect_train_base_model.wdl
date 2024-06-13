version 1.0


workflow TrainPermutectBaseModel {
    input {
        File train_tar
        File? pretrained_model
        Int num_epochs
        Int batch_size
        Int? num_workers
        Float dropout_p
        Int? alt_downsample
        Float reweighting_range
        Int read_embedding_dimension
        Int num_transformer_heads
        Int transformer_hidden_dimension
        Int num_transformer_layers
        Array[Int] info_layers
        Array[Int] aggregation_layers
        Array[String] ref_seq_layer_strings
        String? train_m3_extra_args
        Boolean use_gpu

        String permutect_docker
        Int? preemptible
        Int? max_retries
    }

    if (use_gpu) {
        call TrainPermutectBaseGPU {
            input:
                train_tar = train_tar,
                pretrained_model = pretrained_model,
                permutect_docker = permutect_docker,
                preemptible = preemptible,
                max_retries = max_retries,
                num_epochs = num_epochs,
                batch_size = batch_size,
                num_workers = num_workers,
                dropout_p = dropout_p,
                alt_downsample = alt_downsample,
                reweighting_range = reweighting_range,
                read_embedding_dimension = read_embedding_dimension,
                num_transformer_heads = num_transformer_heads,
                transformer_hidden_dimension = transformer_hidden_dimension,
                num_transformer_layers = num_transformer_layers,
                info_layers = info_layers,
                aggregation_layers = aggregation_layers,
                ref_seq_layer_strings = ref_seq_layer_strings,
                extra_args = train_m3_extra_args
        }
    }

        if (!use_gpu) {
        call TrainPermutectBaseCPU {
            input:
                train_tar = train_tar,
                pretrained_model = pretrained_model,
                permutect_docker = permutect_docker,
                preemptible = preemptible,
                max_retries = max_retries,
                num_epochs = num_epochs,
                batch_size = batch_size,
                num_workers = num_workers,
                dropout_p = dropout_p,
                alt_downsample = alt_downsample,
                reweighting_range = reweighting_range,
                read_embedding_dimension = read_embedding_dimension,
                num_transformer_heads = num_transformer_heads,
                transformer_hidden_dimension = transformer_hidden_dimension,
                num_transformer_layers = num_transformer_layers,
                info_layers = info_layers,
                aggregation_layers = aggregation_layers,
                ref_seq_layer_strings = ref_seq_layer_strings,
                extra_args = train_m3_extra_args
        }
    }


    output {
        File base_model = select_first([TrainPermutectBaseGPU.base_model, TrainPermutectBaseCPU.base_model])
        File training_tensorboard_tar = select_first([TrainPermutectBaseGPU.tensorboard_tar, TrainPermutectBaseCPU.tensorboard_tar])
    }
}

## HORRIBLE HACK: because there is no way in Terra to set gpuCount to 0, in order to optionally use GPU we have to write
## two nearly-identical tasks, one for CPU and one for GPU.  See https://github.com/broadinstitute/cromwell/issues/6679
task TrainPermutectBaseGPU {
    input {
        File train_tar
        File? pretrained_model

        Int num_epochs
        Int batch_size
        Int? num_workers
        Float dropout_p
        Int? alt_downsample
        Float reweighting_range
        Int read_embedding_dimension
        Int num_transformer_heads
        Int transformer_hidden_dimension
        Int num_transformer_layers
        Array[Int] info_layers
        Array[Int] aggregation_layers
        Array[String] ref_seq_layer_strings

        String? extra_args

        String permutect_docker
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

        train_base_model \
            --train_tar ~{train_tar} \
            ~{"--pretrained_model " + pretrained_model} \
            --read_embedding_dimension ~{read_embedding_dimension} \
            --num_transformer_heads ~{num_transformer_heads} \
            --transformer_hidden_dimension ~{transformer_hidden_dimension} \
            --num_transformer_layers ~{num_transformer_layers} \
            --info_layers ~{sep=' ' info_layers} \
            --aggregation_layers ~{sep=' ' aggregation_layers} \
            --ref_seq_layer_strings ~{sep=' ' ref_seq_layer_strings} \
            --dropout_p ~{dropout_p} \
            ~{"--alt_downsample " + alt_downsample} \
            --reweighting_range ~{reweighting_range} \
            --batch_size ~{batch_size} \
            ~{"--num_workers " + num_workers} \
            --num_epochs ~{num_epochs} \
            --output base_model.pt \
            --tensorboard_dir tensorboard \
            ~{extra_args}

        tar cvf tensorboard.tar tensorboard/
    >>>

    runtime {
        docker: permutect_docker
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
        File base_model = "base_model.pt"
        File tensorboard_tar = "tensorboard.tar"
    }
}

task TrainPermutectBaseCPU {
    input {
        File train_tar
        File? pretrained_model

        Int num_epochs
        Int batch_size
        Int? num_workers
        Float dropout_p
        Int? alt_downsample
        Float reweighting_range
        Int read_embedding_dimension
        Int num_transformer_heads
        Int transformer_hidden_dimension
        Int num_transformer_layers
        Array[Int] info_layers
        Array[Int] aggregation_layers
        Array[String] ref_seq_layer_strings
        String? extra_args

        String permutect_docker
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

        train_base_model \
            --train_tar ~{train_tar} \
            ~{"--pretrained_model " + pretrained_model} \
            --read_embedding_dimension ~{read_embedding_dimension} \
            --num_transformer_heads ~{num_transformer_heads} \
            --transformer_hidden_dimension ~{transformer_hidden_dimension} \
            --num_transformer_layers ~{num_transformer_layers} \
            --info_layers ~{sep=' ' info_layers} \
            --aggregation_layers ~{sep=' ' aggregation_layers} \
            --ref_seq_layer_strings ~{sep=' ' ref_seq_layer_strings} \
            --dropout_p ~{dropout_p} \
            ~{"--alt_downsample " + alt_downsample} \
            --reweighting_range ~{reweighting_range} \
            --batch_size ~{batch_size} \
            ~{"--num_workers " + num_workers} \
            --num_epochs ~{num_epochs} \
            --output base_model.pt \
            --tensorboard_dir tensorboard \
            ~{extra_args}

        tar cvf tensorboard.tar tensorboard/
    >>>

    runtime {
        docker: permutect_docker
        bootDiskSizeGb: 12
        memory: machine_mem + " MB"
        disks: "local-disk " + select_first([disk_space, 100]) + if use_ssd then " SSD" else " HDD"
        preemptible: select_first([preemptible, 10])
        maxRetries: select_first([max_retries, 0])
        cpu: select_first([cpu, 1])
    }

    output {
        File base_model = "base_model.pt"
        File tensorboard_tar = "tensorboard.tar"
    }
}
