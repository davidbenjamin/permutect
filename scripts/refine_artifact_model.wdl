version 1.0


workflow RefineArtifactModel {
    input {
        File train_tar
        File saved_model
        Int num_epochs
        Int num_calibration_epochs
        Int? calibration_source
        Int batch_size
        Int inference_batch_size
        Int? num_workers
        Int? gpu_count
        String? extra_args
        Boolean learn_artifact_spectra
        Float? genomic_span

        String permutect_docker
        Int? preemptible
        Int? max_retries
        Int? mem
    }

    call Refine {
        input:
            train_tar = train_tar,
            saved_model = saved_model,
            permutect_docker = permutect_docker,
            preemptible = preemptible,
            max_retries = max_retries,
            mem = mem,
            num_epochs = num_epochs,
            num_calibration_epochs = num_calibration_epochs,
            calibration_source = calibration_source,
            batch_size = batch_size,
            inference_batch_size = inference_batch_size,
            num_workers = num_workers,
            gpu_count = gpu_count,
            extra_args = extra_args,
            learn_artifact_spectra = learn_artifact_spectra,
            genomic_span = genomic_span
    }


    output {
        File permutect_model = Refine.permutect_model
        File training_tensorboard_tar = Refine.tensorboard_tar
    }
}


task Refine {
    input {
        File train_tar
        File saved_model

        Int num_epochs
        Int num_calibration_epochs
        Int? calibration_source
        Int batch_size
        Int inference_batch_size
        Int? num_workers
        Int? gpu_count
        Boolean learn_artifact_spectra
        Float? genomic_span

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
    String learn_artifact_cmd = if learn_artifact_spectra then "--learn_artifact_spectra"  else ""

    command <<<
        set -e

        refine_artifact_model \
            --train_tar ~{train_tar} \
            --saved_model ~{saved_model} \
            --batch_size ~{batch_size} \
            --inference_batch_size ~{inference_batch_size} \
            ~{"--num_workers " + num_workers} \
            --num_epochs ~{num_epochs} \
            --num_calibration_epochs ~{num_calibration_epochs} \
            ~{"--calibration_sources " + calibration_source} \
            --output artifact_model.pt \
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
