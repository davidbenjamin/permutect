version 1.0

workflow BamSurgeon {
    input {
        File ref_fasta
        File ref_fai
        File ref_dict

        File base_bam
        File base_bam_index
        File intervals

        Int scatter_count

        Int num_snvs
        Int num_indels
        Float somatic_allele_fraction
        Int seed

        Boolean use_print_reads = true

        String bam_surgeon_docker = "us.gcr.io/broad-dsde-methods/davidben/bam_surgeon"
        String gatk_docker = "us.gcr.io/broad-gatk/gatk"
        File? gatk_override
        File? dragstr_model
        String? gcs_project_for_requester_pays
    }

    call IndexReference {
        input:
            ref_fasta = ref_fasta,
            ref_fai = ref_fai,
            ref_dict = ref_dict,
            bam_surgeon_docker = bam_surgeon_docker
    }

    call SplitIntervals {
        input:
            intervals = intervals,
            ref_fasta = ref_fasta,
            ref_fai = ref_fai,
            ref_dict = ref_dict,
            scatter_count = scatter_count,
            gatk_docker = gatk_docker
    }

    # generate all SNV and all indels -- the Bamsurgeon task in each scatter must use the overlap between these and the
    # particular scatter interval
    call RandomSites {
        input:
            ref_tar = IndexReference.ref_tar,
            intervals = intervals,
            num_snvs = num_snvs,
            num_indels = num_indels,
            seed = seed,
            bam_surgeon_docker = bam_surgeon_docker
    }

    scatter (subintervals in SplitIntervals.interval_files ) {
        # make a chunk of the overall bam containing the intervals in this scatter
        # and the subsets of the SNV and indel BED files int his scatter
        call RestrictBamAndBedsToSubIntervals {
            input:
                gatk_docker = gatk_docker,
                gcs_project_for_requester_pays = gcs_project_for_requester_pays,
                original_bam = base_bam,
                original_bam_idx = base_bam_index,
                original_snv_bed = RandomSites.snv_bed,
                original_indel_bed = RandomSites.indel_bed,
                ref_fasta = ref_fasta,
                ref_fai = ref_fai,
                ref_dict = ref_dict,
                subintervals = subintervals
        }

        call Bamsurgeon {
            input:
                ref_tar = IndexReference.ref_tar,
                base_bam = RestrictBamAndBedsToSubIntervals.output_bam,
                base_bam_index = RestrictBamAndBedsToSubIntervals.output_bam_idx,
                intervals = subintervals,
                snv_bed = RestrictBamAndBedsToSubIntervals.output_snv_bed,
                indel_bed = RestrictBamAndBedsToSubIntervals.output_indel_bed,
                somatic_allele_fraction = somatic_allele_fraction,
                bam_surgeon_docker = bam_surgeon_docker
        }
    }

    # combine the scattered truth VCFs
    call MergeVCFs {
        input:
            input_vcfs = Bamsurgeon.truth_vcf,
            input_vcf_indices = Bamsurgeon.truth_vcf_idx,
            gatk_docker = gatk_docker
    }

    call MergeBams {
        input:
            ref_fasta = ref_fasta,
            ref_fai = ref_fai,
            ref_dict = ref_dict,
            bams = Bamsurgeon.synthetic_tumor_bam,
            bam_indices = Bamsurgeon.synthetic_tumor_bam_index,
            gatk_docker = gatk_docker
    }

    call LeftAlignAndTrim {
        input:
            gatk_docker = gatk_docker,
            input_vcf = MergeVCFs.merged_vcf,
            input_vcf_idx = MergeVCFs.merged_vcf_idx,
            ref_fasta = ref_fasta,
            ref_fai = ref_fai,
            ref_dict = ref_dict
    }

    call RemovePhantoms {
        input:
            bamsurgeon_bam = MergeBams.merged_bam,
            bamsurgeon_bam_idx = MergeBams.merged_bai,
            bamsurgeon_truth_vcf = LeftAlignAndTrim.aligned_vcf,
            bamsurgeon_truth_vcf_idx = LeftAlignAndTrim.aligned_vcf_idx,
            ref_fasta = ref_fasta,
            ref_fai = ref_fai,
            ref_dict = ref_dict,
            dragstr_model = dragstr_model,
            gatk_override = gatk_override,
            gcs_project_for_requester_pays = gcs_project_for_requester_pays,
            gatk_docker = gatk_docker
    }

    output {
        File synthetic_tumor_bam = MergeBams.merged_bam
        File synthetic_tumor_bam_index = MergeBams.merged_bai
        File truth_vcf = LeftAlignAndTrim.aligned_vcf
        File truth_vcf_idx = LeftAlignAndTrim.aligned_vcf_idx
        File phantom_check_vcf = RemovePhantoms.output_vcf
        File phantom_check_vcf_idx = RemovePhantoms.output_vcf_idx
        File phantom_check_bamout = RemovePhantoms.bamout
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

task SplitIntervals {
    input {
        File? intervals
        File? masked_intervals
        File ref_fasta
        File ref_fai
        File ref_dict
        Int scatter_count

        # if intervals aren't split, and if there is more than a read length between intervals, then scatters don't have
        # overlapping reads and we can gather bams without sorting / worrying about edge effects
        String split_intervals_extra_args = "--subdivision-mode BALANCING_WITHOUT_INTERVAL_SUBDIVISION --min-contig-size 100000"

        String gatk_docker
        Int boot_disk_size = 10
        Int mem = 4
        Int disk = 10
        Int preemptible = 0
        Int cpu = 1
        Int max_retries = 0
    }

    command {
        set -e

        mkdir interval-files
        gatk SplitIntervals \
            -R ~{ref_fasta} \
            ~{"-L " + intervals} \
            ~{"-XL " + masked_intervals} \
            -scatter ~{scatter_count} \
            -O interval-files \
            ~{split_intervals_extra_args}
        cp interval-files/*.interval_list .
    }

    runtime {
        docker: gatk_docker
        bootDiskSizeGb: boot_disk_size
        memory: mem + " GB"
        disks: "local-disk " + disk + " SSD"
        preemptible: preemptible
        maxRetries: max_retries
        cpu: cpu
    }

    output {
        Array[File] interval_files = glob("*.interval_list")
    }
}

task RandomSites {
    input {
        File ref_tar
        File intervals
        Int num_snvs
        Int num_indels
        Int seed

        String bam_surgeon_docker
        Int preemptible_tries = 0
        Int cpu = 1
        Int disk_space = 100
        Int mem = 4
    }

    Int snv_seed = seed
    Int indel_seed = snv_seed + 1
    Int mem_mb = mem * 1000

    command <<<
        tar -xvf ~{ref_tar}
        ref_fasta=`ls reference/*.fasta`

        # if the inout intervals are not a bed file we need to convert
        if [[ ~{intervals} == *.bed ]]; then
            echo "input intervals file is already in bed format"
            bed_file=~{intervals}
        else
            echo "input intervals must be converted to bed format"
            java -jar /usr/local/bin/picard.jar IntervalListToBed I=~{intervals} O=bed_regions.bed
            bed_file=bed_regions.bed
        fi

        echo "making random sites"
        python3.6 /bamsurgeon/scripts/randomsites.py --genome ${ref_fasta} --bed $bed_file \
            --seed ~{snv_seed} --numpicks ~{num_snvs} --avoidN snv > addsnv.bed

        python3.6 /bamsurgeon/scripts/randomsites.py --genome ${ref_fasta} --bed $bed_file \
            --seed ~{indel_seed} --numpicks ~{num_indels} --avoidN indel > addindel.bed
  >>>

  runtime {
     docker: bam_surgeon_docker
     disks: "local-disk " + disk_space + " SSD"
     memory: mem_mb + " MB"
     preemptible: preemptible_tries
     cpu: cpu
  }

  output {
      File snv_bed = "addsnv.bed"
      File indel_bed = "addindel.bed"
  }
}

task Bamsurgeon {
    input {
        File ref_tar
        File base_bam
        File base_bam_index
        File intervals
        File snv_bed
        File indel_bed

        Float somatic_allele_fraction

        String bam_surgeon_docker
        Int preemptible_tries = 0
        Int cpu = 4
        Int disk_space = 500
        Int mem = 8
    }

    Int mem_mb = mem * 1000

    command <<<
        tar -xvf ~{ref_tar}
        ref_fasta=`ls reference/*.fasta`

        echo "contents of current directory:"
        ls

        # BAMSurgeon expects .bam.bai for the index -- it will error if it's just plain .bai!
        mv ~{base_bam_index} ~{base_bam}.bai

        echo "adding synthetic SNVs"
        python3.6 /bamsurgeon/bin/addsnv.py --varfile ~{snv_bed} --bamfile ~{base_bam} \
            --reference ${ref_fasta} --outbam snv.bam \
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
        python3.6 /bamsurgeon/bin/addindel.py --varfile ~{indel_bed} --bamfile snv_sorted.bam --reference ${ref_fasta} \
            --outbam snv_indel.bam \
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
        java -jar /picard.jar SortVcf I=snvs.vcf I=indels.vcf O=truth.vcf

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
      File truth_vcf = "truth.vcf"
      File truth_vcf_idx = "truth.vcf.idx"
  }
}

task LeftAlignAndTrim {
    input {
        String gatk_docker
        File input_vcf
        File input_vcf_idx
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
        gatk LeftAlignAndTrimVariants -R ~{ref_fasta} -V ~{input_vcf} -O truth.vcf
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
        File aligned_vcf = "truth.vcf"
        File aligned_vcf_idx = "truth.vcf.idx"
    }
}

task RestrictBamAndBedsToSubIntervals {
    input {
        String gatk_docker
        String? gcs_project_for_requester_pays
        File original_bam       # this can be a BAM or CRAM
        File original_bam_idx
        File original_snv_bed
        File original_indel_bed
        File ref_fasta          # GATK PrintReads requires a reference for CRAMs
        File ref_fai
        File ref_dict
        File subintervals

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
        original_bam: {localization_optional: true}
        original_bam_idx: {localization_optional: true}
    }

    command <<<

        # if the inout intervals are not a bed file we need to convert
        if [[ ~{subintervals} == *.bed ]]; then
            echo "input subintervals file is already in bed format"
            subintervals_bed_file=~{subintervals}
        else
            echo "input intervals must be converted to bed format"
            gatk IntervalListToBed -I ~{subintervals} -O subintervals.bed
            subintervals_bed_file=subintervals.bed
        fi



        # HACK: bedtools does not accept 6-column BED, while the add-indel.bed file from BAMSurgeon has 6 columns,
        # the auxiliary columns being VAF, DEL/INS, and inserted bases (if insertion)  we merge the last three columns
        # and separate by underscores for processing with bedtools
        while read a b c d e f; do printf "${a}\t${b}\t${c}\t${d}_${e}_${f}\n"; done < ~{original_indel_bed} > hack-indel.bed

        # bedtools is in the GATK docker
        # -wa means write the entire original entry from the -a argument
        bedtools intersect -wa -a ~{original_snv_bed} -b ${subintervals_bed_file} > restricted_snv.bed
        bedtools intersect -wa -a hack-indel.bed -b ${subintervals_bed_file} > hack-restricted_indel.bed

        # now convert the underscores back to tabs and remove trailing tabs for DEL records to undo the hack
        sed 's/_/\t/g' hack-restricted_indel.bed | sed 's/[[:space:]]*$//' > restricted_indel.bed


        # this command also produces the accompanying index hla.bai
        # the PairedReadFilter is necessary for SamtoFastq to succeed
        gatk PrintReads -R ~{ref_fasta} -I ~{original_bam} -L ~{subintervals} -O output.bam \
            ~{"--gcs-project-for-requester-pays " + gcs_project_for_requester_pays}

        echo "Here is the subintervals file:"
        cat ${subintervals_bed_file}

        echo "Here is the restricted SNV bed file:"
        cat restricted_snv.bed

        echo "Here is the restricted indel bed file:"
        cat restricted_indel.bed

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
        File output_snv_bed = "restricted_snv.bed"
        File output_indel_bed = "restricted_indel.bed"
    }
}

task MergeBams {
    input {
        File ref_fasta
        File ref_fai
        File ref_dict
        Array[File]+ bams
        Array[File]+ bam_indices
        String gatk_docker
        File? gatk_override
        Int disk_space = 1000
        Int boot_disk_size = 10
        Int preemptible = 0
        Int cpu = 1
        Int max_retries = 0
        Int mem = 16
    }

    Int machine_mem = mem * 1000
    Int command_mem = machine_mem - 500

    command <<<
        # This command block assumes that there is at least one file in bam_outs.
        #  Do not call this task if len(bam_outs) == 0
        set -e
        export GATK_LOCAL_JAR=~{default="/root/gatk.jar" gatk_override}
        gatk --java-options "-Xmx~{command_mem}m" GatherBamFiles \
            -I ~{sep=" -I " bams} -O merged.bam -R ~{ref_fasta}

        # We must sort because adjacent scatters may have overlapping reads

        gatk --java-options "-Xmx~{command_mem}m" SortSam -I merged.bam \
            -O merged-sorted.bam --SORT_ORDER coordinate -VALIDATION_STRINGENCY LENIENT

        mv merged-sorted.bam merged.bam

        gatk --java-options "-Xmx~{command_mem}m" BuildBamIndex -I merged.bam -VALIDATION_STRINGENCY LENIENT
    >>>

    runtime {
        docker: gatk_docker
        bootDiskSizeGb: boot_disk_size
        memory: machine_mem + " MB"
        disks: "local-disk " + disk_space + " SSD"
        preemptible: preemptible
        maxRetries: max_retries
        cpu: cpu
    }

    output {
        File merged_bam = "merged.bam"
        File merged_bai = "merged.bai"
    }
}

task MergeVCFs {
    input {
        Array[File] input_vcfs
        Array[File] input_vcf_indices
        String gatk_docker
        Int disk_space = 100
        Int boot_disk_size = 10
        Int preemptible = 0
        Int cpu = 1
        Int max_retries = 0
        Int mem = 16
    }

    Int machine_mem = mem * 1000
    Int command_mem = machine_mem - 500

    # using MergeVcfs instead of GatherVcfs so we can create indices
    command {
        set -e
        gatk --java-options "-Xmx~{command_mem}m" MergeVcfs -I ~{sep=' -I ' input_vcfs} -O merged.vcf
    }

    runtime {
        docker: gatk_docker
        bootDiskSizeGb: boot_disk_size
        memory: machine_mem + " MB"
        disks: "local-disk " + disk_space + " SSD"
        preemptible: preemptible
        maxRetries: max_retries
        cpu: cpu
    }

    output {
        File merged_vcf = "merged.vcf"
        File merged_vcf_idx = "merged.vcf.idx"
    }
}

task RemovePhantoms {
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