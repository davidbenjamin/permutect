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

        # this docker built from the Dockerfile in the bam surgeon github
        String bam_surgeon_docker = "us.gcr.io/broad-dsde-methods/davidben/bam_surgeon"
    }

    call RandomSitesAndAddVariants {
        input:
            ref_fasta = ref_fasta,
            ref_fai = ref_fai,
            ref_dict = ref_dict,
            base_bam = base_bam,
            base_bam_index = base_bam_index,
            target_regions_bed = target_regions_bed,
            num_snvs = num_snvs,
            num_indels = num_indels,
            somatic_allele_fraction = somatic_allele_fraction,
            seed = seed,
            bam_surgeon_docker = bam_surgeon_docker
    }

    output {
        File synthetic_tumor_bam = RandomSitesAndAddVariants.synthetic_tumor_bam
        File synthetic_tumor_bam_index = RandomSitesAndAddVariants.synthetic_tumor_bam_index
        File truth_vcf = RandomSitesAndAddVariants.truth_vcf
    }
}

task RandomSitesAndAddVariants {
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
        echo $PWD

        python3 /bamsurgeon/scripts/randomsites.py –g ~{ref_fasta} –b ~{target_regions_bed} –n ~{num_snvs} –avoidN –s ~{snv_seed} snv > addsnv_input.bed

        python3 /bamsurgeon/scripts/randomsites.py –g ~{ref_fasta} –b ~{target_regions_bed} –n ~{num_indels} –avoidN –s ~{indel_seed} indel > addindel_input.bed

        python3 /bamsurgeon/bin/addsnv.py –r ~{ref_fasta} –-bamfile ~{base_bam} --varfile addsnv_input.bed \
            --mutfrac ~{somatic_allele_fraction} \
            --snvfrac 0.2 \
            --haplosize 50 \
            –ignoresnps –tagreads –ignorepileup \
            –picardjar picard.jar \
            –aligner mem \
            –minmutreads 1 –seed 1 \
            –o snv.bam --vcf snvs.vcf

        samtools sort -@ ~{cpu} --output-fmt BAM snv.bam > snv_sorted.bam

        samtools index snv_sorted.bam

        python3 /bamsurgeon/bin/addindel.py –r ~{ref_fasta} –-bamfile snv_sorted.bam --varfile addindel_input.bed \
            --mutfrac ~{somatic_allele_fraction} \
            --snvfrac 0.2 \
            --haplosize 50 \
            –tagreads –ignorepileup \
            –picardjar picard.jar \
            –aligner mem \
            –minmutreads 1 –seed 1 \
            –o snv_indel.bam --vcf indels.vcf

        samtools sort -@ ~{cpu} --output-fmt BAM snv_indel.bam > snv_indel_sorted.bam

        samtools index snv_indel_sorted.bam

        java -jar picard.jar MergeVcfs I=snvs.vcf I=indels.vcf O=variants.vcf
  >>>

  runtime {
     docker: bam_surgeon_docker
     disks: "local-disk " + disk_space + " SSD"
     memory: mem_mb + " MB"
     preemptible: preemptible_tries
     cpu: cpu
  }

  output {
      File synthetic_tumor_bam = "snv_indel_sorted.bam"
      File synthetic_tumor_bam_index = "snv_indel_sorted.bam.bai"
      File truth_vcf = "variants.vcf"
  }
}
