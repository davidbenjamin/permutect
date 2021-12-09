version 1.0


workflow TrainMutect3 {
    input {


        # runtime
        File mutect3_package_tar_gz
        File? gatk_override
        Int? preemptible
        Int? max_retries
    }


    Runtime small_runtime = {"gatk_docker": gatk_docker, "gatk_override": gatk_override,
                                "max_retries": 2, "preemptible": 0, "cpu": 2,
                                "machine_mem": 4000, "command_mem": 3500,
                                "disk": 100, "boot_disk_size": 12}


    output {

    }
}
