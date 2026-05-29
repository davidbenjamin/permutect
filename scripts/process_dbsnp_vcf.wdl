version 1.0

workflow Process {
	input {
		File aou_vcf
		File aou_vcf_idx
	}

	call process { input: aou_vcf=aou_vcf, aou_vcf_idx=aou_vcf_idx}

	output {
		File output_vcf = process.output_vcf
	}
}

task process {
	input {
		File aou_vcf
		File aou_vcf_idx
	}

	command <<<
        gunzip ~{aou_vcf}

        grep -v '#' *.vcf | head -n 100 > result.txt
	>>>

    runtime {
        docker: "continuumio/anaconda:latest"
        disks: "local-disk " + 500 + " SSD"
    }

    output {
        File output_vcf = "result.txt"
    }
}