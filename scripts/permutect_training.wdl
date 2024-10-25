version 1.0


workflow TrainPermutect {
    input {
        File train_tar
        File base_model
        Int num_epochs
        Int num_calibration_epochs
        Int batch_size
        Int? num_workers
        Int? gpu_count
        Float dropout_p
        Array[Int] aggregation_layers
        Array[Int] calibration_layers
        String? train_m3_extra_args
        Boolean use_gpu
        Boolean learn_artifact_spectra
        Float? genomic_span

        String permutect_docker
        Int? preemptible
        Int? max_retries
    }

    if (use_gpu) {
        call TrainPermutectGPU {
            input:
                train_tar = train_tar,
                base_model = base_model,
                permutect_docker = permutect_docker,
                preemptible = preemptible,
                max_retries = max_retries,
                num_epochs = num_epochs,
                num_calibration_epochs = num_calibration_epochs,
                batch_size = batch_size,
                num_workers = num_workers,
                gpu_count = gpu_count,
                dropout_p = dropout_p,
                aggregation_layers = aggregation_layers,
                calibration_layers = calibration_layers,
                extra_args = train_m3_extra_args,
                learn_artifact_spectra = learn_artifact_spectra,
                genomic_span = genomic_span
        }
    }

        if (!use_gpu) {
        call TrainPermutectCPU {
            input:
                train_tar = train_tar,
                base_model = base_model,
                permutect_docker = permutect_docker,
                preemptible = preemptible,
                max_retries = max_retries,
                num_epochs = num_epochs,
                num_calibration_epochs = num_calibration_epochs,
                batch_size = batch_size,
                num_workers = num_workers,
                dropout_p = dropout_p,
                aggregation_layers = aggregation_layers,
                calibration_layers = calibration_layers,
                extra_args = train_m3_extra_args,
                learn_artifact_spectra = learn_artifact_spectra,
                genomic_span = genomic_span
        }
    }


    output {
        File artifact_model = select_first([TrainPermutectGPU.artifact_model, TrainPermutectCPU.artifact_model])
        File training_tensorboard_tar = select_first([TrainPermutectGPU.tensorboard_tar, TrainPermutectCPU.tensorboard_tar])
    }
}

## HORRIBLE HACK: because there is no way in Terra to set gpuCount to 0, in order to optionally use GPU we have to write
## two nearly-identical tasks, one for CPU and one for GPU.  See https://github.com/broadinstitute/cromwell/issues/6679
task TrainPermutectGPU {
    input {
        File train_tar
        File base_model

        Int num_epochs
        Int num_calibration_epochs
        Int batch_size
        Int? num_workers
        Int? gpu_count
        Float dropout_p
        Array[Int] aggregation_layers
        Array[Int] calibration_layers
        Boolean learn_artifact_spectra
        Float? genomic_span

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
    String learn_artifact_cmd = if learn_artifact_spectra then "--learn_artifact_spectra"  else ""

    command <<<
        set -e

        train_model \
            --train_tar ~{train_tar} \
            --base_model ~{base_model} \
            --aggregation_layers ~{sep=' ' aggregation_layers} \
            --calibration_layers ~{sep=' ' calibration_layers} \
            --dropout_p ~{dropout_p} \
            --batch_size ~{batch_size} \
            ~{"--num_workers " + num_workers} \
            --num_epochs ~{num_epochs} \
            --num_calibration_epochs ~{num_calibration_epochs} \
            --output artifact.pt \
            --tensorboard_dir tensorboard \
            ~{"--genomic_span " + genomic_span} \
            ~{learn_artifact_cmd} \
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
        gpuType: "nvidia-tesla-p4"
        gpuCount: select_first([gpu_count, 1])
        nvidiaDriverVersion: "535.183.01"
        zones : ["us-central1-a", "us-central1-b", "us-central1-c"]
    }

    output {
        File artifact_model = "artifact.pt"
        File tensorboard_tar = "tensorboard.tar"
    }
}

task TrainPermutectCPU {
    input {
        File train_tar
        File base_model

        Int num_epochs
        Int num_calibration_epochs
        Int batch_size
        Int? num_workers
        Float dropout_p
        Array[Int] aggregation_layers
        Array[Int] calibration_layers
        Boolean learn_artifact_spectra
        Float? genomic_span
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
    String learn_artifact_cmd = if learn_artifact_spectra then "--learn_artifact_spectra" else ""

    command <<<
        set -e

        train_model \
            --train_tar ~{train_tar} \
            --base_model ~{base_model} \
            --aggregation_layers ~{sep=' ' aggregation_layers} \
            --calibration_layers ~{sep=' ' calibration_layers} \
            --dropout_p ~{dropout_p} \
            --batch_size ~{batch_size} \
            ~{"--num_workers " + num_workers} \
            --num_epochs ~{num_epochs} \
            --num_calibration_epochs ~{num_calibration_epochs} \
            --output artifact.pt \
            --tensorboard_dir tensorboard \
            ~{"--genomic_span " + genomic_span} \
            ~{learn_artifact_cmd} \
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
        File artifact_model = "artifact.pt"
        File tensorboard_tar = "tensorboard.tar"
    }
}
