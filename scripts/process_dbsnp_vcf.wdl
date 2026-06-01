version 1.0

workflow Process {
	input {
		File dbsnp_vcf
		File dbsnp_vcf_idx
		File python_script
	}

	call process { input: dbsnp_vcf=dbsnp_vcf, dbsnp_vcf_idx=dbsnp_vcf_idx, python_script=python_script}

	call compress_and_index { input: dbsnp_vcf=process.output_vcf}

	output {
		File output_vcf = compress_and_index.output_vcf
		File output_vcf_idx = compress_and_index.output_vcf_idx
	}
}

task process {
	input {
		File dbsnp_vcf
		File dbsnp_vcf_idx
		File python_script
	}

	command <<<
		gunzip -c ~{dbsnp_vcf} > unzipped.vcf

		grep -v 'FREQ=dbGaP_PopFreq:1' unzipped.vcf > variants.vcf

		python ~{python_script} variants.vcf dbsnp_processed.vcf
	>>>

    runtime {
        docker: "continuumio/anaconda3:latest"
        disks: "local-disk " + 2000 + " SSD"
    }

    output {
        File output_vcf = "dbsnp_processed.vcf"
    }
}

task compress_and_index {
	input {
		File dbsnp_vcf
	}

	command <<<
		bcftools view -O z -o dbsnp.vcf.bgz ~{dbsnp_vcf}
		tabix -p vcf dbsnp.vcf.bgz
	>>>

    runtime {
        docker: "biocontainers/bcftools:v1.9-1-deb_cv1"
        disks: "local-disk " + 1000 + " SSD"
    }

    output {
        File output_vcf = "dbsnp.vcf.bgz"
		File output_vcf_idx = "dbsnp.vcf.bgz.tbi"
    }
}