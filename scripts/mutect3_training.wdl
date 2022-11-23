version 1.0


workflow TrainMutect3 {
    input {
        Array[File] training_datasets
        Int num_epochs
        Int batch_size
        Int? num_workers
        Int chunk_size
        Float dropout_p
        Float reweighting_range
        Array[Int] read_layers
        Array[Int] info_layers
        Array[Int] aggregation_layers
        Array[String] ref_seq_layer_strings
        String? train_m3_extra_args
        Boolean use_gpu

        String mutect3_docker
        Int? preemptible
        Int? max_retries
    }

    if (use_gpu) {
        call TrainMutect3GPU {
            input:
                training_datasets = training_datasets,
                mutect3_docker = mutect3_docker,
                preemptible = preemptible,
                max_retries = max_retries,
                num_epochs = num_epochs,
                batch_size = batch_size,
                num_workers = num_workers,
                chunk_size = chunk_size,
                dropout_p = dropout_p,
                reweighting_range = reweighting_range,
                read_layers = read_layers,
                info_layers = info_layers,
                aggregation_layers = aggregation_layers,
                ref_seq_layer_strings = ref_seq_layer_strings,
                extra_args = train_m3_extra_args
        }
    }

        if (!use_gpu) {
        call TrainMutect3CPU {
            input:
                training_datasets = training_datasets,
                mutect3_docker = mutect3_docker,
                preemptible = preemptible,
                max_retries = max_retries,
                num_epochs = num_epochs,
                batch_size = batch_size,
                num_workers = num_workers,
                chunk_size = chunk_size,
                dropout_p = dropout_p,
                reweighting_range = reweighting_range,
                read_layers = read_layers,
                info_layers = info_layers,
                aggregation_layers = aggregation_layers,
                ref_seq_layer_strings = ref_seq_layer_strings,
                extra_args = train_m3_extra_args
        }
    }


    output {
        File mutect3_model = select_first([TrainMutect3GPU.mutect3_model, TrainMutect3CPU.mutect3_model])
        Array[File] training_tensorboard = select_first([TrainMutect3GPU.training_tensorboard, TrainMutect3CPU.training_tensorboard])
    }
}

## HORRIBLE HACK: because there is no way in Terra to set gpuCount to 0, in order to optionally use GPU we have to write
## two nearly-identical tasks, one for CPU and one for GPU.  See https://github.com/broadinstitute/cromwell/issues/6679
task TrainMutect3GPU {
    input {
        Array[File] training_datasets

        Int num_epochs
        Int batch_size
        Int? num_workers
        Int chunk_size
        Float dropout_p
        Float reweighting_range
        Array[Int] read_layers
        Array[Int] info_layers
        Array[Int] aggregation_layers
        Array[String] ref_seq_layer_strings

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
            --ref_seq_layer_strings ~{sep=' ' ref_seq_layer_strings} \
            --dropout_p ~{dropout_p} \
            --reweighting_range ~{reweighting_range} \
            --batch_size ~{batch_size} \
            ~{"--num_workers " + num_workers} \
            --chunk_size ~{chunk_size} \
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

task TrainMutect3CPU {
    input {
        Array[File] training_datasets

        Int num_epochs
        Int batch_size
        Int chunk_size
        Int? num_workers
        Float dropout_p
        Float reweighting_range
        Array[Int] read_layers
        Array[Int] info_layers
        Array[Int] aggregation_layers
        Array[String] ref_seq_layer_strings
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
            --ref_seq_layer_strings ~{sep=' ' ref_seq_layer_strings} \
            --dropout_p ~{dropout_p} \
            --reweighting_range ~{reweighting_range} \
            --batch_size ~{batch_size} \
            ~{"--num_workers " + num_workers} \
            --chunk_size ~{chunk_size} \
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
    }

    output {
        File mutect3_model = "mutect3.pt"
        Array[File] training_tensorboard = glob("tensorboard/*")
    }
}
