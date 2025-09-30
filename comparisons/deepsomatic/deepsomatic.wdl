version 1.0

workflow DeepSomatic {
    input {
        File ref_fasta
        File ref_dict
        File ref_fai

        File tumor_bam
        File tumor_bai
        File? normal_bam
        File? normal_bai

        # Can be WGS,WES,PACBIO,ONT,FFPE_WGS,FFPE_WES,WGS_TUMOR_ONLY,PACBIO_TUMOR_ONLY,ONT_TUMOR_ONLY
        String model_type

        File intervals
        File? masks

        String deepsomatic_extra_args

        File? truth_vcf    # used for evaluation
        File? truth_vcf_idx

        String gatk_docker = "us.gcr.io/broad-gatk/gatk"

        String deepsomatic_docker = "nvcr.io/nvidia/clara/clara-parabricks:4.3.1-1"
        #String deepsomatic_docker = "us.gcr.io/broad-dsde-methods/davidben/deepsomatic-gpu"

        # see https://github.com/clara-parabricks-workflows/parabricks-wdl/blob/main/wdl/germline_calling.wdl
        # and https://docs.nvidia.com/clara/parabricks/4.5.1/gettingstarted/installationrequirements.html
        String nvidia_driver_version = "525.60.13"
        String? gcs_project_for_requester_pays

        # WDL version 1.0 does not have an empty Optional literal
        # such a literal is very useful because Terra has a bug where whenever a data table is updated, empty values
        # silently and invisibly get converted to empty strings "".  Thus it is useful to recognize empty strings and
        # declare empty Optionals.  The only way to do this in WDL 1.0 is to get an empty Optional as a variable from the
        # workflow inputs.  These inputs should NEVER be filled in!!!!!
        File? EMPTY_STRING_HACK
    }

    call GetSampleName {
        input:
            ref_fasta = ref_fasta,
            ref_fai = ref_fai,
            ref_dict = ref_dict,
            tumor_bam = tumor_bam,
            tumor_bai = tumor_bai,
            normal_bam = normal_bam,
            normal_bai = normal_bai,
            gcs_project_for_requester_pays = gcs_project_for_requester_pays,
            gatk_docker = gatk_docker
    }

    call SplitContigs {
      input:
        ref_fasta_index = ref_fai,
        threads = 2
    }

    #call IntervalListToBed {
    #    input:
    #        gatk_docker = gatk_docker,
    #        intervals = intervals
    #}

        scatter (ctg in SplitContigs.contigs){
            call DeepSomaticPacBio {
                input:
                    tumor_bam = tumor_bam,
                    normal_bam = normal_bam,
                    tumor_bam_index = tumor_bai,
                    normal_bam_index = normal_bai,
                    tumor_sample = GetSampleName.tumor_sample,
                    normal_sample = GetSampleName.normal_sample,
                    model_type = model_type,
                    ref_fasta = ref_fasta,
                    ref_fasta_index = ref_fai,
                    contig = ctg
            }
        }

    call MergeVCFs {
        input:
            input_vcfs = DeepSomaticPacBio.deepsomatic_vcf,
            input_vcf_indices = DeepSomaticPacBio.deepsomatic_vcf_idx
    }


    #call DeepsomaticParabricks {
    #    input:
    #        ref_tarball = ref_tarball,
    #        tumor_bam = tumor_bam,
    #        tumor_bai = tumor_bai,
    #        normal_bam = normal_bam,
    #        normal_bai = normal_bai,
    #        intervals = IntervalListToBed.output_bed,
    #        deepsomatic_extra_args = deepsomatic_extra_args,
    #        deepsomatic_docker = deepsomatic_docker,
    #        nvidia_driver_version = nvidia_driver_version
    #}

    if (defined(truth_vcf)){
        call Concordance {
            input:
                intervals = intervals,
                masks = if masks == "" then EMPTY_STRING_HACK else masks,
                truth_vcf = select_first([truth_vcf]),
                truth_vcf_idx = select_first([truth_vcf_idx]),
                eval_vcf = MergeVCFs.merged_vcf,
                eval_vcf_idx = MergeVCFs.merged_vcf_idx,
                gatk_docker = gatk_docker
        }
    }

    output {
        File deepsomatic_calls_vcf = MergeVCFs.merged_vcf
        File deepsomatic_calls_vcf_idx = MergeVCFs.merged_vcf_idx

        File? fn = Concordance.fn
        File? fn_idx = Concordance.fn_idx
        File? fp = Concordance.fp
        File? fp_idx = Concordance.fp_idx
        File? tp = Concordance.tp
        File? tp_idx = Concordance.tp_idx
        File? ffn = Concordance.ffn
        File? ffn_idx = Concordance.ffn_idx
        File? ftn = Concordance.ftn
        File? ftn_idx = Concordance.ftn_idx
        File? concordance_summary = Concordance.summary
        File? filter_analysis = Concordance.filter_analysis
    }
}

task DeepsomaticParabricks {
    input {
        File ref_tarball

        File tumor_bam
        File tumor_bai
        File normal_bam
        File normal_bai
        File intervals
        String deepsomatic_extra_args

        String deepsomatic_docker
        String nvidia_driver_version

        Int cpu = 24
        Int gpu_count = 4
        Int mem_gb = 120
        Int disk_gb = 1000
        Int max_retries = 0
        Int preemptible = 0
    }

    String ref = basename(ref_tarball, ".tar")
    String localTarball = basename(ref_tarball)

    command <<<
        mv ~{ref_tarball} ~{localTarball} && \
        time tar xvf ~{localTarball} && \
        time pbrun deepsomatic \
            --ref ~{ref} \
            --in-tumor-bam ~{tumor_bam} \
            --in-normal-bam ~{normal_bam} \
            --out-variants output.vcf \
            ~{deepsomatic_extra_args}


        # report nvidia driver stuff, cuda compatibility etc for troubleshooting
        # nvidia-smi
        # we removed             ‑‑interval‑file ~{intervals} \
        # echo "contents of current directory"
        # ls .
        # echo "name of ref: " ~{ref}
        # echo "local tarball: " ~{localTarball}
        # pbrun deepsomatic -h

    >>>

    runtime {
        docker: deepsomatic_docker
        memory: mem_gb + " GB"
        disks: "local-disk " + disk_gb + " SSD"
        preemptible: preemptible
        maxRetries: max_retries
        cpu: cpu
        gpuType: "nvidia-tesla-t4"
        gpuCount: gpu_count
        nvidiaDriverVersion: nvidia_driver_version
        zones : ["us-central1-a", "us-central1-b", "us-central1-c"]
    }

    output {
        File output_vcf = "output.vcf"
        File output_vcf_idx = "output/output.vcf.idx"
    }
}

task Deepsomatic {
    input {
        File ref_tarball

        File tumor_bam
        File tumor_bai
        String tumor_sample
        File normal_bam
        File normal_bai
        String normal_sample
        File intervals
        String model_type
        String deepsomatic_extra_args

        String deepsomatic_docker
        String nvidia_driver_version

        Int cpu = 4
        Int mem_gb = 32
        Int disk_gb = 1000
        Int boot_disk_gb = 10
        Int max_retries = 0
        Int preemptible = 0
    }

    String ref = basename(ref_tarball, ".tar")
    String localTarball = basename(ref_tarball)

    command <<<
        mv ~{ref_tarball} ~{localTarball}
        tar xvf ~{localTarball}

        run_deepsomatic \
            --model_type=~{model_type} \
            --ref=~{ref} \
            ‑‑interval‑file=~{intervals} \
            --reads_normal=~{normal_bam} \
            --reads_tumor=~{tumor_bam} \
            --output_vcf=output/output.vcf.gz \
            --output_gvcf=output/output.g.vcf.gz \
            --sample_name_tumor=~{tumor_sample} \
            --sample_name_normal=~{normal_sample} \
            --num_shards=~{cpu} \
            --logging_dir=output/logs \
            --intermediate_results_dir output/intermediate_results_dir \
            --use_default_pon_filtering=false \
            --dry_run=false \
            ~{deepsomatic_extra_args}

    >>>

    runtime {
        docker: deepsomatic_docker
        bootDiskSizeGb: boot_disk_gb
        memory: mem_gb + " GB"
        disks: "local-disk " + disk_gb + " SSD"
        preemptible: preemptible
        maxRetries: max_retries
        cpu: cpu
        gpuType: "nvidia-tesla-t4"
        gpuCount: 1
        nvidiaDriverVersion: nvidia_driver_version
        zones : ["us-central1-a", "us-central1-b", "us-central1-c"]
    }

    output {
        File output_vcf = "output/output.vcf.gz"
        File output_vcf_idx = "output/output.vcf.gz.tbi"
    }
}



task Concordance {
    input {
    	File? intervals
    	File? masks
    	File truth_vcf
    	File truth_vcf_idx
    	File eval_vcf
    	File eval_vcf_idx

    	# runtime
    	String gatk_docker = "us.gcr.io/broad-gatk/gatk"
    	Int preemptible = 0
	}

    command {
        gatk --java-options "-Xmx2g" Concordance \
            ~{"-L " + intervals} \
            ~{"-XL " + masks} \
            -truth ~{truth_vcf} -eval ~{eval_vcf} \
            -tpfn "tpfn.vcf" \
            -tpfp "tpfp.vcf" \
            -ftnfn "ftnfn.vcf" \
            -filter-analysis "filter-analysis.txt" \
            -summary "summary.txt"

        grep '#' tpfn.vcf > HEAD
        grep STATUS=FN tpfn.vcf > BODY
        cat HEAD BODY > false_negatives.vcf

        grep '#' tpfp.vcf > HEAD
        grep STATUS=FP tpfp.vcf > BODY
        cat HEAD BODY > false_positives.vcf

        grep '#' tpfp.vcf > HEAD
        grep STATUS=TP tpfp.vcf > BODY
        cat HEAD BODY > true_positives.vcf

        grep '#' ftnfn.vcf > HEAD
        grep STATUS=FFN ftnfn.vcf > BODY
        cat HEAD BODY > filtered_false_negatives.vcf
        grep STATUS=FTN ftnfn.vcf > BODY
        cat HEAD BODY > filtered_true_negatives.vcf

        for vcf in false_negatives.vcf false_positives.vcf true_positives.vcf filtered_false_negatives.vcf filtered_true_negatives.vcf; do
            #HACK: IndexFeatureFile throws error if vcf is empty, which is possible here especially in the case of false negatives
            gatk --java-options "-Xmx2g" SelectVariants -V $vcf -O tmp.vcf
            mv tmp.vcf $vcf
            mv tmp.vcf.idx $vcf.idx
        done
    }

    runtime {
        memory: "5 GB"
        bootDiskSizeGb: 12
        docker: "${gatk_docker}"
        disks: "local-disk " + 100 + " HDD"
        preemptible: select_first([preemptible, 2])
    }

    output {
        File fn = "false_negatives.vcf"
        File fn_idx = "false_negatives.vcf.idx"
        File fp = "false_positives.vcf"
        File fp_idx = "false_positives.vcf.idx"
        File tp = "true_positives.vcf"
        File tp_idx = "true_positives.vcf.idx"
        File ffn = "filtered_false_negatives.vcf"
        File ffn_idx = "filtered_false_negatives.vcf.idx"
        File ftn = "filtered_true_negatives.vcf"
        File ftn_idx = "filtered_true_negatives.vcf.idx"
        File summary = "summary.txt"
        File filter_analysis = "filter-analysis.txt"
    }
}

task GetSampleName {
    input {
        File ref_fasta
        File ref_fai
        File ref_dict
        File tumor_bam
        File tumor_bai
        File? normal_bam
        File? normal_bai
        String? gcs_project_for_requester_pays

        String gatk_docker
        Int mem = 2
        Int boot_disk_size = 10
        Int preemptible = 0
        Int max_retries = 0
        Int disk_space = 10
        Int cpu = 1
    }


    parameter_meta{
        ref_fasta: {localization_optional: true}
        ref_fai: {localization_optional: true}
        ref_dict: {localization_optional: true}
        tumor_bam: {localization_optional: true}
        tumor_bai: {localization_optional: true}
        normal_bam: {localization_optional: true}
        normal_bai: {localization_optional: true}
    }

    command <<<
        touch normal_names.txt
        if [[ ! -z "~{normal_bam}" ]]; then
            gatk GetSampleName -R ~{ref_fasta} -I ~{normal_bam} -O normal_names.txt -encode \
                ~{"--gcs-project-for-requester-pays " + gcs_project_for_requester_pays}
        fi

        gatk GetSampleName -R ~{ref_fasta} -I ~{tumor_bam} -O tumor_names.txt -encode \
                ~{"--gcs-project-for-requester-pays " + gcs_project_for_requester_pays}
    >>>

    runtime {
        docker: gatk_docker
        bootDiskSizeGb: boot_disk_size
        memory: mem + " GB"
        disks: "local-disk " + disk_space + " SSD"
        preemptible: preemptible
        maxRetries: max_retries
        cpu: cpu
    }

    output {
        String normal_sample = read_string("normal_names.txt")
        String tumor_sample = read_string("tumor_names.txt")
    }
}


task IntervalListToBed {
    input {
        String gatk_docker
        File intervals

        Int cpu = 2
        Int mem_gb = 4
        Int disk_gb = 100
        Int boot_disk_gb = 4
        Int max_retries = 0
        Int preemptible = 0
    }

    command <<<
        gatk IntervalListToBed --INPUT ~{intervals} --OUTPUT intervals.bed
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
        File output_bed = "intervals.bed"
    }
}

# copied from PacBio's DeepSomatic WDL: https://github.com/PacificBiosciences/HiFi-somatic-WDL/blob/main/tasks/common.wdl
# Use bedtools to split contigs
task SplitContigs {
  input {
    File ref_fasta_index
    Int chunk_size = 75000000
    Int threads
  }

  Float file_size = ceil(size(ref_fasta_index, "GB") + 10)

  command <<<
  set -euxo pipefail

  bedtools --version

  echo "Splitting contigs for ~{ref_fasta_index}"
  bedtools makewindows -g ~{ref_fasta_index} -w ~{chunk_size} > contigs.bed
  grep -v -E "random|chrUn|chrM|chrEBV" contigs.bed > noalt.bed
  # Split the contig bed files into one file for each line
  split -l 1 noalt.bed contigs_split.
  # Add .bed to all the contigs_split file
  for file in $(ls contigs_split.*); do mv $file $file.bed; done
  >>>

  output {
    Array[File] contigs = glob("contigs_split.*.bed")
  }

  runtime {
    docker: "quay.io/biocontainers/bedtools:2.31.0--hf5e1c6e_2"
    cpu: threads
    memory: "~{threads * 4} GB"
    disk: file_size + " GB"
    maxRetries: 2
    preemptible: 1
  }
}

task DeepSomaticPacBio {
    input {
        File tumor_bam
        File? normal_bam
        File tumor_bam_index
        File? normal_bam_index
        String tumor_sample
        String? normal_sample
        String model_type
        File ref_fasta
        File ref_fasta_index
        File? contig
        Int threads = 16
    }

    Float file_size = ceil(size(tumor_bam, "GB") * 2 + size(normal_bam, "GB") * 2 + size(ref_fasta, "GB") + size(contig, "GB") + 20)

    command <<<
    set -euxo pipefail

    /opt/deepvariant/bin/deepsomatic/run_deepsomatic --version

    /opt/deepvariant/bin/deepsomatic/run_deepsomatic \
        --model_type=~{model_type} \
        ~{if defined(normal_bam) then "" else "--use_default_pon_filtering=true"} \
        --ref=~{ref_fasta} \
        ~{if (defined(normal_bam)) then "--reads_normal=~{normal_bam}" else ""} \
        --reads_tumor=~{tumor_bam} \
        --output_vcf=deepsomatic.vcf.gz \
        --output_gvcf=deepsomatic.g.vcf.gz \
        --sample_name_tumor=~{tumor_sample} \
        ~{if (defined(normal_bam)) then "--sample_name_normal=~{normal_sample}" else ""} \
        --num_shards=~{threads} \
        --postprocess_variants_extra_args="--cpus=~{threads / 2},--num_partitions=~{threads / 2}" \
        --logging_dir=logs \
        ~{"--regions=" + contig}
    >>>

    output {
        File deepsomatic_vcf = "deepsomatic.vcf.gz"
        File deepsomatic_vcf_idx = "deepsomatic.vcf.gz.tbi"
    }

    runtime {
        docker: "google/deepsomatic@sha256:d9797b8950bf615ec7010d1336b7ee0a2f12ea09323dc3585f7e9fe39b082bde"
        cpu: threads
        memory: "~{threads * 8} GB"
        disk: file_size + " GB"
        maxRetries: 0
        preemptible: 1
    }
}

task MergeVCFs {
    input {
        Array[File] input_vcfs
        Array[File] input_vcf_indices

        Int mem = 8
        Int boot_disk_size = 8
        Int disk = 100
        Int preemptible = 0
        Int max_retries =  1
        Int cpu = 1
    }

    command {
        set -e
        gatk MergeVcfs -I ~{sep=' -I ' input_vcfs} -O merged.vcf
    }

    runtime {
        docker: "us.gcr.io/broad-gatk/gatk"
        bootDiskSizeGb: boot_disk_size
        memory: mem + " GB"
        disks: "local-disk " + disk + " HDD"
        preemptible: preemptible
        maxRetries: max_retries
        cpu: cpu
    }

    output {
        File merged_vcf = "merged.vcf"
        File merged_vcf_idx = "merged.vcf.idx"
    }
}

