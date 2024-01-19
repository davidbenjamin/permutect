version 1.0


workflow TrainMutect3 {
    input {
        File train_tar
        File artifact_tar
        Int num_epochs
        Int num_calibration_epochs
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
        Array[Int] calibration_layers
        Array[String] ref_seq_layer_strings
        String? train_m3_extra_args
        Boolean use_gpu
        Boolean learn_artifact_spectra
        Float? genomic_span

        String mutect3_docker
        Int? preemptible
        Int? max_retries
    }

    if (use_gpu) {
        call TrainMutect3GPU {
            input:
                train_tar = train_tar,
                artifact_tar = artifact_tar,
                mutect3_docker = mutect3_docker,
                preemptible = preemptible,
                max_retries = max_retries,
                num_epochs = num_epochs,
                num_calibration_epochs = num_calibration_epochs,
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
                calibration_layers = calibration_layers,
                ref_seq_layer_strings = ref_seq_layer_strings,
                extra_args = train_m3_extra_args,
                learn_artifact_spectra = learn_artifact_spectra,
                genomic_span = genomic_span
        }
    }

        if (!use_gpu) {
        call TrainMutect3CPU {
            input:
                train_tar = train_tar,
                artifact_tar = artifact_tar,
                mutect3_docker = mutect3_docker,
                preemptible = preemptible,
                max_retries = max_retries,
                num_epochs = num_epochs,
                num_calibration_epochs = num_calibration_epochs,
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
                calibration_layers = calibration_layers,
                ref_seq_layer_strings = ref_seq_layer_strings,
                extra_args = train_m3_extra_args,
                learn_artifact_spectra = learn_artifact_spectra,
                genomic_span = genomic_span
        }
    }


    output {
        File mutect3_model = select_first([TrainMutect3GPU.mutect3_model, TrainMutect3CPU.mutect3_model])
        File training_tensorboard_tar = select_first([TrainMutect3GPU.tensorboard_tar, TrainMutect3CPU.tensorboard_tar])
    }
}

## HORRIBLE HACK: because there is no way in Terra to set gpuCount to 0, in order to optionally use GPU we have to write
## two nearly-identical tasks, one for CPU and one for GPU.  See https://github.com/broadinstitute/cromwell/issues/6679
task TrainMutect3GPU {
    input {
        File train_tar
        File artifact_tar

        Int num_epochs
        Int num_calibration_epochs
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
        Array[Int] calibration_layers
        Array[String] ref_seq_layer_strings
        Boolean learn_artifact_spectra
        Float? genomic_span

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
    String learn_artifact_cmd = if learn_artifact_spectra then "--learn_artifact_spectra"  else ""

    command <<<
        set -e

        train_model \
            --train_tar ~{train_tar} \
            --artifact_tar ~{artifact_tar} \
            --read_embedding_dimension ~{read_embedding_dimension} \
            --num_transformer_heads ~{num_transformer_heads} \
            --transformer_hidden_dimension ~{transformer_hidden_dimension} \
            --num_transformer_layers ~{num_transformer_layers} \
            --info_layers ~{sep=' ' info_layers} \
            --aggregation_layers ~{sep=' ' aggregation_layers} \
            --calibration_layers ~{sep=' ' calibration_layers} \
            --ref_seq_layer_strings ~{sep=' ' ref_seq_layer_strings} \
            --dropout_p ~{dropout_p} \
            ~{"--alt_downsample " + alt_downsample} \
            --reweighting_range ~{reweighting_range} \
            --batch_size ~{batch_size} \
            ~{"--num_workers " + num_workers} \
            --num_epochs ~{num_epochs} \
            --num_calibration_epochs ~{num_calibration_epochs} \
            --output mutect3.pt \
            --tensorboard_dir tensorboard \
            ~{"--genomic_span " + genomic_span} \
            ~{learn_artifact_cmd} \
            ~{extra_args}

        tar cvf tensorboard.tar tensorboard/
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
        File tensorboard_tar = "tensorboard.tar"
    }
}

task TrainMutect3CPU {
    input {
        File train_tar
        File artifact_tar

        Int num_epochs
        Int num_calibration_epochs
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
        Array[Int] calibration_layers
        Array[String] ref_seq_layer_strings
        Boolean learn_artifact_spectra
        Float? genomic_span
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
    String learn_artifact_cmd = if learn_artifact_spectra then "--learn_artifact_spectra" else ""

    command <<<
        set -e

        train_model \
            --train_tar ~{train_tar} \
            --artifact_tar ~{artifact_tar} \
            --read_embedding_dimension ~{read_embedding_dimension} \
            --num_transformer_heads ~{num_transformer_heads} \
            --transformer_hidden_dimension ~{transformer_hidden_dimension} \
            --num_transformer_layers ~{num_transformer_layers} \
            --info_layers ~{sep=' ' info_layers} \
            --aggregation_layers ~{sep=' ' aggregation_layers} \
            --calibration_layers ~{sep=' ' calibration_layers} \
            --ref_seq_layer_strings ~{sep=' ' ref_seq_layer_strings} \
            --dropout_p ~{dropout_p} \
            ~{"--alt_downsample " + alt_downsample} \
            --reweighting_range ~{reweighting_range} \
            --batch_size ~{batch_size} \
            ~{"--num_workers " + num_workers} \
            --num_epochs ~{num_epochs} \
            --num_calibration_epochs ~{num_calibration_epochs} \
            --output mutect3.pt \
            --tensorboard_dir tensorboard \
            ~{"--genomic_span " + genomic_span} \
            ~{learn_artifact_cmd} \
            ~{extra_args}

        tar cvf tensorboard.tar tensorboard/
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
        File tensorboard_tar = "tensorboard.tar"
    }
}
