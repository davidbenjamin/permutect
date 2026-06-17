version 1.0

workflow MergeVCFs {
	input {
		Array[File] input_vcfs
        Array[File] input_vcf_indices
	}

	call merge { input: input_vcfs = input_vcfs, input_vcf_indices = input_vcf_indices}

	output {
		File merged_vcf = merge.merged_vcf
        File merged_vcf_idx = merge.merged_vcf_idx
	}
}

task merge {
        input {
            Array[File] input_vcfs
            Array[File] input_vcf_indices
            File ref_dict
        }

    command {
        set -e
        gatk MergeVcfs -I ~{sep=' -I ' input_vcfs} -D ~{ref_dict} -O merged.vcf.gz
    }

    runtime {
        docker: "broadinstitute/gatk:latest"
        bootDiskSizeGb: 6
        memory: 8 + " GB"
        disks: "local-disk " + 200 + " SSD"
        preemptible: 0
        maxRetries: 0
        cpu: 1
    }

    output {
        File merged_vcf = "merged.vcf.gz"
        File merged_vcf_idx = "merged.vcf.gz.tbi"
    }
}