workflow CheckBamsurgeon {
    input {
        # basic inputs
        File bamsurgeon_bam
        File bamsurgeon_bam_idx
        File bamsurgeon_truth_vcf
        File bamsurgeon_truth_vcf_idx
        File ref_fasta
        File ref_fai
        File ref_dict

        # extra arguments
        File? dragstr_model
        String? m2_extra_args
        Boolean make_bamout = false

        # runtime
        String gatk_docker
        File? gatk_override
        String? gcs_project_for_requester_pays

    }

    call M2 {
        input:
            bamsurgeon_bam = bamsurgeon_bam,
            bamsurgeon_bam_idx = bamsurgeon_bam_idx,
            bamsurgeon_truth_vcf = bamsurgeon_truth_vcf,
            bamsurgeon_truth_vcf_idx = bamsurgeon_truth_vcf_idx,
            ref_fasta = ref_fasta,
            ref_fai = ref_fai,
            ref_dict = ref_dict,
            m2_extra_args = m2_extra_args,
            make_bamout = make_bamout,
            dragstr_model = dragstr_model,
            gatk_override = gatk_override,
            gcs_project_for_requester_pays = gcs_project_for_requester_pays
    }


    output {
        File output_vcf = M2.output_vcf
        File output_vcf_idx = M2.output_vcf_idx
        File bamout = M2.bamout
    }
}





task M2 {
    input {
        File bamsurgeon_bam
        File bamsurgeon_bam_idx
        File bamsurgeon_truth_vcf
        File bamsurgeon_truth_vcf_idx
        File ref_fasta
        File ref_fai
        File ref_dict

        String? m2_extra_args
        Boolean? make_bamout
        File? dragstr_model
        File? gatk_override
        String? gcs_project_for_requester_pays


        # runtime
        String gatk_docker
        Int mem = 4
        Int preemptible = 1
        Int max_retries = 0
        Int disk_space = 100
        Int cpu = 1
        Boolean use_ssd = false
    }

    # Mem is in units of GB but our command and memory runtime values are in MB
    Int machine_mem = mem * 1000
    Int command_mem = machine_mem - 500

    parameter_meta{
        ref_fasta: {localization_optional: true}
        ref_fai: {localization_optional: true}
        ref_dict: {localization_optional: true}
        bamsurgeon_bam: {localization_optional: true}
        bamsurgeon_bam_idx: {localization_optional: true}
        bamsurgeon_truth_vcf: {localization_optional: true}
        bamsurgeon_truth_vcf_idx: {localization_optional: true}
    }

    command <<<
        set -e

        export GATK_LOCAL_JAR=~{default="/root/gatk.jar" gatk_override}

        # We need to create these files regardless, even if they stay empty
        touch bamout.bam

        gatk --java-options "-Xmx~{command_mem}m" Mutect2 \
            -R ~{ref_fasta} \
            -I ~{bamsurgeon_bam} \
            ~{"-L " + bamsurgeon_truth_vcf} \
            ~{"--alleles " + bamsurgeon_truth_vcf} \
            ~{"--dragstr-params-path " + dragstr_model} \
            -O force_call.vcf \
            ~{true='--bam-output bamout.bam' false='' make_bamout} \
            ~{m2_extra_args} \
            ~{"--gcs-project-for-requester-pays " + gcs_project_for_requester_pays}

    >>>

    runtime {
        docker: gatk_docker
        bootDiskSizeGb: 12
        memory: machine_mem + " MB"
        disks: "local-disk " + disk_space + " SSD"
        preemptible: preemptible
        maxRetries: max_retries
        cpu: cpu
    }

    output {
        File output_vcf = "force_call.vcf"
        File output_vcf_idx = "force_call.vcf.idx"
        File bamout = "bamout.bam"
    }
}