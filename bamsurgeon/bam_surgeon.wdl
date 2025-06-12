version 1.0

workflow BamSurgeon {
    input {
        File ref_fasta
        File ref_fai
        File ref_dict

        File base_bam
        File base_bam_index
        File target_regions_bed

        Int num_snvs
        Int num_indels
        Float somatic_allele_fraction
        Int seed

        Boolean use_print_reads = true

        String bam_surgeon_docker = "us.gcr.io/broad-dsde-methods/davidben/bam_surgeon"
        String gatk_docker = "us.gcr.io/broad-gatk/gatk"
        String? gcs_project_for_requester_pays
    }

    if (use_print_reads) {
        call PrintReads {
            input:
                gatk_docker = gatk_docker,
                gcs_project_for_requester_pays = gcs_project_for_requester_pays,
                original_bam = base_bam,
                original_bam_idx = base_bam_index,
                ref_fasta = ref_fasta,
                ref_fai = ref_fai,
                ref_dict = ref_dict,
                intervals = target_regions_bed
        }
    }

    call IndexReference {
        input:
            ref_fasta = ref_fasta,
            ref_fai = ref_fai,
            ref_dict = ref_dict,
            bam_surgeon_docker = bam_surgeon_docker
    }

    call Bamsurgeon {
        input:
            ref_tar = IndexReference.ref_tar,
            base_bam = select_first([PrintReads.output_bam, base_bam]),
            base_bam_index = select_first([PrintReads.output_bam_idx, base_bam_index]),
            target_regions_bed = target_regions_bed,
            num_snvs = num_snvs,
            num_indels = num_indels,
            somatic_allele_fraction = somatic_allele_fraction,
            seed = seed,
            bam_surgeon_docker = bam_surgeon_docker
    }

    call LeftAlignAndTrim {
    input:
        gatk_docker = gatk_docker,
        truth_vcf = Bamsurgeon.truth_vcf,
        ref_fasta = ref_fasta,
        ref_fai = ref_fai,
        ref_dict = ref_dict
    }

    output {
        File synthetic_tumor_bam = Bamsurgeon.synthetic_tumor_bam
        File synthetic_tumor_bam_index = Bamsurgeon.synthetic_tumor_bam_index
        File truth_vcf = LeftAlignAndTrim.aligned_truth_vcf
        File truth_vcf_idx = LeftAlignAndTrim.aligned_truth_vcf_idx
    }
}

task IndexReference {
    input {
        File ref_fasta
        File ref_fai
        File ref_dict

        String bam_surgeon_docker
        Int preemptible_tries = 0
        Int cpu = 4
        Int disk_space = 500
        Int mem = 8
    }


    command <<<
        # calls to BWA within bam surgeon require not not ref fasta, dict, but other auxiliary files
        echo "Indexing reference"
        bwa index ~{ref_fasta}

        ref_dir=`dirname ~{ref_fasta}`
        echo "contents of reference directory ${ref_dir}:"
        ls ${ref_dir}

        mkdir reference
        mv ${ref_dir}/* reference

        echo "contents of reference/ directory after move"
        ls reference

        tar cvf reference.tar reference/

        # debug: expand it
        rm -r reference/

        tar -xvf reference.tar
        ls .
        ls reference/
  >>>

  runtime {
     docker: bam_surgeon_docker
     disks: "local-disk " + disk_space + " SSD"
     memory: mem + " GB"
     preemptible: preemptible_tries
     cpu: cpu
  }

  output {
      File ref_tar = "reference.tar"
  }
}

task Bamsurgeon {
    input {
        File ref_tar
        File base_bam
        File base_bam_index
        File target_regions_bed
        Int num_snvs
        Int num_indels
        Float somatic_allele_fraction
        Int seed


        String bam_surgeon_docker
        Int preemptible_tries = 0
        Int cpu = 4
        Int disk_space = 500
        Int mem = 8
    }

    Int snv_seed = seed
    Int indel_seed = snv_seed + 1
    Int mem_mb = mem * 1000

    command <<<
        tar -xvf ~{ref_tar}
        ref_fasta=`ls reference/*.fasta`

        # if the inout intervals are not a bed file we need to convert
        if [[ ~{target_regions_bed} == *.bed ]]; then
            echo "input intervals file is already in bed format"
            bed_file=~{target_regions_bed}
        else
            echo "input intervals must be converted to bed format"
            java -jar /usr/local/bin/picard.jar IntervalListToBed I=~{target_regions_bed} O=bed_regions.bed
            bed_file=bed_regions.bed
        fi

        # super annoying: bam surgeon expects bam index to end in .bam.bai or .cram.crai, not just .bai or .crai
        if [[ ~{base_bam_index} == *.bai ]]; then
            mv ~{base_bam_index} ~{base_bam}.bai
        elif [[ ~{base_bam_index} == *.crai ]]; then
            mv ~{base_bam_index} ~{base_bam}.crai
        else
            echo "TERRIBLE ERROR.  Sequencing data is neither .bam nor .cram"
        fi


        echo "making random sites"
        python3.6 /bamsurgeon/scripts/randomsites.py --genome ${ref_fasta} --bed $bed_file \
            --seed ~{snv_seed} --numpicks ~{num_snvs} --avoidN snv > addsnv_input.bed

        python3.6 /bamsurgeon/scripts/randomsites.py --genome ${ref_fasta} --bed $bed_file \
            --seed ~{indel_seed} --numpicks ~{num_indels} --avoidN indel > addindel_input.bed

        echo "contents of current directory:"
        ls

        echo "adding synthetic SNVs"
        python3.6 /bamsurgeon/bin/addsnv.py --varfile addsnv_input.bed --bamfile ~{base_bam} \
            --reference ${ref_fasta} --outbam snv.bam \
            --procs 8 \
            --insane \
            --ignorepileup \
            --snvfrac 0.2 \
            --mutfrac ~{somatic_allele_fraction} \
            --haplosize 50 \
            --picardjar /picard.jar \
            --minmutreads 2 \
            --coverdiff 0.1 \
            --ignoresnps --tagreads \
            --aligner mem \
            --seed 1 \

        echo "contents of current directory:"
        ls

        # I believe the output is called snv.addsnv.addsnv_input.vcf, but the wildcard lets us be more general
        mv snv.*.vcf snvs.vcf

        echo "sorting SNV-added bam"
        samtools sort -@ ~{cpu} --output-fmt BAM snv.bam > snv_sorted.bam

        echo "contents of current directory:"
        ls

        echo "indexing SNV-added bam"
        samtools index snv_sorted.bam

        echo "contents of current directory:"
        ls

        echo "adding synthetic indels"
        python3.6 /bamsurgeon/bin/addindel.py --varfile addindel_input.bed --bamfile snv_sorted.bam --reference ${ref_fasta} \
            --outbam snv_indel.bam \
            --procs 8 \
            --insane \
            --ignorepileup \
            --snvfrac 0.2 \
            --mutfrac ~{somatic_allele_fraction} \
            --picardjar /picard.jar \
            --minmutreads 2 \
            --tagreads \
            --aligner mem \
            --seed 1

        echo "contents of current directory:"
        ls

        # Likewise (see above) I believe the exact VCF output is snv_indel.addindel.addindel_input.vcf
        # note that the BAM is named snv_indel (since it has the cumulative effect of SNV and indel addition), whereas
        # the VCF only contains the indels added at this last step.
        mv snv_indel.*.vcf indels.vcf

        echo "sorting BAM with SNVs and indels added"
        samtools sort -@ ~{cpu} --output-fmt BAM snv_indel.bam > result.bam

        echo "contents of current directory:"
        ls

        echo "indexing BAM"
        samtools index result.bam

        echo "contents of current directory:"
        ls

        echo "sorting VCF"
        java -jar /picard.jar SortVcf I=snvs.vcf I=indels.vcf O=variants.vcf

        echo "contents of current directory:"
        ls
  >>>

  runtime {
     docker: bam_surgeon_docker
     disks: "local-disk " + disk_space + " SSD"
     memory: mem_mb + " MB"
     preemptible: preemptible_tries
     cpu: cpu
  }

  output {
      File synthetic_tumor_bam = "result.bam"
      File synthetic_tumor_bam_index = "result.bam.bai"
      File truth_vcf = "variants.vcf"
  }
}

task LeftAlignAndTrim {
    input {
        String gatk_docker
        File truth_vcf = "variants.vcf"
        File ref_fasta
        File ref_fai
        File ref_dict

        Int cpu = 2
        Int mem_gb = 4
        Int disk_gb = 100
        Int boot_disk_gb = 10
        Int max_retries = 0
        Int preemptible = 1
    }

    parameter_meta{
        ref_fasta: {localization_optional: true}
        ref_fai: {localization_optional: true}
        ref_dict: {localization_optional: true}
    }

    command <<<
        cp ~{truth_vcf} ./orig.vcf
        gatk IndexFeatureFile -I orig.vcf

        gatk LeftAlignAndTrimVariants -R ~{ref_fasta} -V orig.vcf -O truth.vcf
    >>>

    runtime {
        docker: gatk_docker
        bootDiskSizeGb: boot_disk_gb
        memory: mem_gb + " GB"
        disks: "local-disk " + disk_gb + " SSD"
        preemptible: preemptible
        maxRetries: max_retries
        cpu: cpu
    }

    output {
        File aligned_truth_vcf = "truth.vcf"
        File aligned_truth_vcf_idx = "truth.vcf.idx"
    }
}


task PrintReads {
    input {
        String gatk_docker
        String? gcs_project_for_requester_pays
        File original_bam       # this can be a BAM or CRAM
        File original_bam_idx
        File ref_fasta          # GATK PrintReads requires a reference for CRAMs
        File ref_fai
        File ref_dict
        File intervals

        Int cpu = 2
        Int mem_gb = 4
        Int disk_gb = 100
        Int boot_disk_gb = 10
        Int max_retries = 0
        Int preemptible = 1
    }

    parameter_meta{
        intervals: {localization_optional: true}
        ref_fasta: {localization_optional: true}
        ref_fai: {localization_optional: true}
        ref_dict: {localization_optional: true}
        original_bam: {localization_optional: true}
        original_bam_idx: {localization_optional: true}
    }

    command <<<
        # this command also produces the accompanying index hla.bai
        # the PairedReadFilter is necessary for SamtoFastq to succeed
        gatk PrintReads -R ~{ref_fasta} -I ~{original_bam} -L ~{intervals} -O output.bam \
            ~{"--gcs-project-for-requester-pays " + gcs_project_for_requester_pays}

    >>>

    runtime {
        docker: gatk_docker
        bootDiskSizeGb: boot_disk_gb
        memory: mem_gb + " GB"
        disks: "local-disk " + disk_gb + " SSD"
        preemptible: preemptible
        maxRetries: max_retries
        cpu: cpu
    }

    output {
        File output_bam = "output.bam"
        File output_bam_idx = "output.bai"
    }
}
