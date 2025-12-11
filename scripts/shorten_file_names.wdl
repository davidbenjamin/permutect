version 1.0

workflow ShortenFileName {
    input {
        File input_file
    }

    call Shorten {input: input_file=input_file}

    output {
        File output_file = Shorten.output_file
    }
}

task Shorten {
    input {
        File input_file
        String docker = "ubuntu:16.04"
    }

    command <<<
        mkdir output
        newname=$(basename ~{input_file})
        cp ~{input_file} output/${newname}
    >>>

    runtime {
        docker: docker
        preemptible: 1
        maxRetries: 0
        cpu: 1
    }

    output {
        File output_file = glob("output/*")[0]
    }
}
