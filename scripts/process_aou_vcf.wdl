version 1.0

workflow Process {
	input {
		File aou_vcf
		File aou_vcf_idx
		File python_script
	}

	call process { input: aou_vcf=aou_vcf, aou_vcf_idx=aou_vcf_idx, python_script=python_script}

	call compress_and_index { input: aou_vcf=process.output_vcf}

	output {
		File output_vcf = compress_and_index.output_vcf
		File output_vcf_idx = compress_and_index.output_vcf_idx
	}
}

task process {
	input {
		File aou_vcf
		File aou_vcf_idx
		File python_script
	}

	command <<<
		gunzip -c ~{aou_vcf} > unzipped.vcf

		grep -v -e 'AC=0' unzipped.vcf > variants.vcf

		python ~{python_script} variants.vcf aou_processed.vcf
	>>>

    runtime {
        docker: "continuumio/anaconda3:latest"
        disks: "local-disk " + 2000 + " SSD"
    }

    output {
        File output_vcf = "aou_processed.vcf"
    }
}

task compress_and_index {
	input {
		File aou_vcf
	}

	command <<<
		bcftools view -O z -o aou.vcf.bgz ~{aou_vcf}
		tabix -p vcf aou.vcf.bgz
	>>>

    runtime {
        docker: "biocontainers/bcftools:v1.9-1-deb_cv1"
        disks: "local-disk " + 1000 + " SSD"
    }

    output {
        File output_vcf = "aou.vcf.bgz"
		File output_vcf_idx = "aou.vcf.bgz.tbi"
    }
}