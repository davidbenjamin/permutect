version 1.0

workflow FTPDownload {
    input {
        Array[String] paths
        String docker = "ubuntu:16.04"
    }

    call FTP {
        input:
            docker = docker,
            paths = paths
    }

    output {
        Array[File] downloaded_files = FTP.downloaded_files
    }
}

task FTP {
    input {
        Array[String] paths
        String docker

        Int mem_gb = 4
        Int disk_gb = 1000
        Int boot_disk_gb = 10
        Int max_retries = 0
        Int preemptible = 0
    }


    command <<<
        mkdir output
        cd output

        for path in ~{sep=' ' paths}; do wget ${path}; done
    >>>

    runtime {
        docker: docker
        bootDiskSizeGb: boot_disk_gb
        memory: mem_gb + " GB"
        disks: "local-disk " + disk_gb + " SSD"
        preemptible: preemptible
        maxRetries: max_retries
    }

    output {
        Array[File] downloaded_files = glob("output/*")
    }
}