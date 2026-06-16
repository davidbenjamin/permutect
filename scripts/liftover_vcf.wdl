version 1.0

workflow LiftoverVCF {
      input {
          File input_vcf
          File input_vcf_idx
          File? intervals

          File source_ref_fasta
          File source_ref_fai
          File source_ref_dict
          File target_ref_fasta
          File target_ref_fai
          File target_ref_dict

          File source_to_target_chain_file

          Int scatter_count
      }

    call SplitIntervals {
        input:
            intervals = intervals,
            ref_fasta = source_ref_fasta,
            ref_fai = source_ref_fai,
            ref_dict = source_ref_dict,
            scatter_count = scatter_count
    }

    scatter (subintervals in SplitIntervals.interval_files ) {
        call RestrictVCF {
            input:
                intervals = subintervals,
                input_vcf = input_vcf,
                input_vcf_idx = input_vcf_idx,
                ref_fasta = source_ref_fasta,
                ref_fai = source_ref_fai,
                ref_dict = source_ref_dict
        }

        call Liftover {
            input:
                input_vcf = RestrictVCF.output_vcf,
                input_vcf_idx = RestrictVCF.output_vcf_idx,
                source_ref_fasta = source_ref_fasta,
                source_ref_fai = source_ref_fai,
                source_ref_dict = source_ref_dict,
                target_ref_fasta = target_ref_fasta,
                target_ref_fai = target_ref_fai,
                target_ref_dict = target_ref_dict,
                source_to_target_chain_file = source_to_target_chain_file
        }
    }

    call MergeVCFs {
        input:
            input_vcfs = Liftover.output_vcf
    }

    output {
        File lifted_vcf = MergeVCFs.merged_vcf
        File lifted_vcf_idx = MergeVCFs.merged_vcf_idx
    }


}

task SplitIntervals {
    input {
      File? intervals
      File ref_fasta
      File ref_fai
      File ref_dict
      Int scatter_count
      String? split_intervals_extra_args
    }

    command {
        set -e

        mkdir interval-files
        gatk SplitIntervals \
            -R ~{ref_fasta} \
            ~{"-L " + intervals} \
            -scatter ~{scatter_count} \
            -O interval-files \
            ~{split_intervals_extra_args}
        cp interval-files/*.interval_list .
    }

    runtime {
        docker: "broadinstitute/gatk:latest"
        bootDiskSizeGb: 6
        memory: 4 + " GB"
        disks: "local-disk " + 100 + " SSD"
        preemptible: 0
        maxRetries: 0
        cpu: 1
    }

    output {
        Array[File] interval_files = glob("*.interval_list")
    }
}

task RestrictVCF {
    input {
        File intervals
        File input_vcf
        File input_vcf_idx

        File ref_fasta
        File ref_fai
        File ref_dict
    }

    command {
        set -e

        gatk SelectVariants \
            -R ~{ref_fasta} \
            ~{"-L " + intervals} \
            -V ~{input_vcf} \
            -O output.vcf.gz
    }

    runtime {
        docker: "broadinstitute/gatk:latest"
        bootDiskSizeGb: 6
        memory: 4 + " GB"
        disks: "local-disk " + 100 + " SSD"
        preemptible: 0
        maxRetries: 0
        cpu: 1
    }

    output {
        File output_vcf = "output.vcf.gz"
        File output_vcf_idx = "output.vcf.gz.tbi"
    }
}

task Liftover {
    input {
        File input_vcf
        File input_vcf_idx

        File source_ref_fasta
        File source_ref_fai
        File source_ref_dict
        File target_ref_fasta
        File target_ref_fai
        File target_ref_dict

        File source_to_target_chain_file
    }

    command {
        set -e

        bcftools norm -f ~{source_ref_fasta} -m -any ~{input_vcf} --check-ref s -Oz -o normalized.vcf.gz

        bcftools +liftover -Ou normalized.vcf.gz -- \
            -s ~{source_ref_fasta} \
            -f ~{target_ref_fasta} \
            -c ~{source_to_target_chain_file} \
            --reject rejected.vcf | bcftools sort -Oz -o lifted.vcf.gz -W=tbi
    }

    runtime {
        docker: "quay.io/biocontainers/bcftools-liftover-plugin:1.22--hb66fcc3_0"
        bootDiskSizeGb: 6
        memory: 4 + " GB"
        disks: "local-disk " + 100 + " SSD"
        preemptible: 0
        maxRetries: 0
        cpu: 1
    }

    output {
        File output_vcf = "lifted.vcf.gz"
        File rejected = "rejected.vcf"
    }
}

task MergeVCFs {
    input {
      Array[File] input_vcfs
    }

    command {
        set -e

        for file in ~{sep=' ' input_vcfs}; do
            gatk IndexFeatureFile -I $file
        done

        gatk  MergeVcfs -I ~{sep=' -I ' input_vcfs} -O merged.vcf.gz
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

