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

        # calls to BWA within bam surgeon require not not ref fasta, dict, but other auxiliary files
        echo "Indexing reference"
        bwa index ~{ref_fasta}

        # if the inout intervals are not a bed file we need to convert
        if [[ ~{target_regions_bed} == *.bed ]]; then
            echo "input intervals file is already in bed format"
            bed_file=~{target_regions_bed}
        else
            echo "input intervals must be converted to bed format"
            java -jar /usr/local/bin/picard.jar IntervalListToBed I=~{target_regions_bed} O=bed_regions.bed
            bed_file=bed_regions.bed
        fi

        # super annoying: bam surgeon expects bam index to end in .bam.bai, not just .bai
        mv ~{base_bam_index} ~{base_bam}.bai

        echo "making random sites"
        python3.6 /bamsurgeon/scripts/randomsites.py --genome ~{ref_fasta} --bed $bed_file \
            --seed ~{snv_seed} --numpicks ~{num_snvs} --avoidN snv > addsnv_input.bed

        python3.6 /bamsurgeon/scripts/randomsites.py --genome ~{ref_fasta} --bed $bed_file \
            --seed ~{indel_seed} --numpicks ~{num_indels} --avoidN indel > addindel_input.bed

        echo "adding synthetic SNVs"
        python3.6 /bamsurgeon/bin/addsnv.py --varfile addsnv_input.bed --bamfile ~{base_bam} \
            --reference ~{ref_fasta} --outbam snv.bam \
            --snvfrac 0.2 \
            --mutfrac ~{somatic_allele_fraction} \
            --haplosize 50 \
            --picardjar /picard.jar \
            --minmutreads 2 \
            --coverdiff 0.1 \
            --ignoresnps --tagreads --ignorepileup \
            --force --insane \
            --aligner mem \
            --seed 1 \

        # I believe the output is called snv.addsnv.addsnv_input.vcf, but the wildcard lets us be more general
        mv snv.*.vcf snvs.vcf

        echo "sorting SNV-added bam"
        samtools sort -@ ~{cpu} --output-fmt BAM snv.bam > snv_sorted.bam

        echo "indexing SNV-added bam"
        samtools index snv_sorted.bam

        echo "adding synthetic indels"
        python3.6 /bamsurgeon/bin/addindel.py --varfile addindel_input.bed --bamfile snv_sorted.bam --reference ~{ref_fasta} \
            --outbam snv_indel.bam \
            --snvfrac 0.2 \
            --mutfrac ~{somatic_allele_fraction} \
            --haplosize 50 \
            --picardjar /picard.jar \
            --minmutreads 2 \
            --tagreads --ignorepileup \
            --force --insane \
            --aligner mem \
            --seed 1

        # Likewise (see above)
        mv indels.*.vcf indels.vcf

        echo "sorting BAM"
        samtools sort -@ ~{cpu} --output-fmt BAM snv_indel.bam > snv_indel_sorted.bam

        echo "indexing BAM"
        samtools index snv_indel_sorted.bam

        echo "sorting VCF"
        java -jar /picard.jar SortVcf I=snvs.vcf I=indels.vcf O=variants.vcf
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
