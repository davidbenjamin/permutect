version 1.0


struct Runtime {
    String gatk_docker
    File? gatk_override
    Int max_retries
    Int preemptible
    Int cpu
    Int machine_mem
    Int command_mem
    Int disk
    Int boot_disk_size
}

workflow Permutect {
    input {
        File permutect_model

        File? intervals
        File? masks     # masked_intervals
        File ref_fasta
        File ref_fai
        File ref_dict
        Int scatter_count
        Int? num_spectrum_iterations
        Float? spectrum_learning_rate
        File primary_bam    # tumor_reads
        File primary_bai    # tumors_reads_index
        File? control_bam   # normals_reads
        File? control_bai   # normals_reads_index
        File? gnomad
        File? gnomad_idx
        File? variants_for_contamination
        File? variants_for_contamination_idx
        File? realignment_index_bundle
        File? dragstr_model
        String? realignment_extra_args
        Boolean make_bamout = false
        Boolean compress_vcfs = false
        Boolean run_orientation_bias_mixture_model_filter = false
        String? m2_extra_args
        String? m2_extra_filtering_args
        String? split_intervals_extra_args
        Int batch_size
        Int num_workers
        Int? gpu_count
        Int chunk_size
        File? test_dataset_truth_vcf
        File? test_dataset_truth_vcf_idx
        Boolean skip_filtering = true

        String? permutect_filtering_extra_args
        String gatk_docker
        String? gcs_project_for_requester_pays
        String bcftools_docker
        File? gatk_override
        String permutect_docker
        Int preemptible = 2
        Int max_retries = 0
        Int small_task_cpu = 2
        Int small_task_mem = 4
        Int small_task_disk = 100
        Int boot_disk_size = 12
        Int learn_read_orientation_mem = 8000
        Int filter_alignment_artifacts_mem = 9000
        String basic_bash_docker = "ubuntu:16.04"
        Int emergency_extra_disk = 0
        File? obscene_hack_leave_unset
    }

    File? masked_intervals = masks
    File tumor_reads = primary_bam
    File tumor_reads_index = primary_bai
    File? normal_reads = control_bam
    File? normal_reads_index = control_bai

    # Disk sizes used for dynamic sizing
    Int ref_size = ceil(size(ref_fasta, "GB") + size(ref_dict, "GB") + size(ref_fai, "GB"))
    Int tumor_reads_size = ceil(size(tumor_reads, "GB") + size(tumor_reads_index, "GB"))
    Int gnomad_vcf_size = if defined(gnomad) then ceil(size(gnomad, "GB")) else 0
    Int normal_reads_size = if defined(control_bam) then ceil(size(control_bam, "GB") + size(control_bai, "GB")) else 0

    # This is added to every task as padding, should increase if systematically you need more disk for every call
    Int disk_pad = 10 + emergency_extra_disk

    Runtime standard_runtime = {"gatk_docker": gatk_docker, "gatk_override": gatk_override,
            "max_retries": max_retries, "preemptible": preemptible, "cpu": small_task_cpu,
            "machine_mem": small_task_mem * 1000, "command_mem": small_task_mem * 1000 - 500,
            "disk": small_task_disk + disk_pad, "boot_disk_size": boot_disk_size}

    Int tumor_reads_size = ceil(size(tumor_reads, "GB") + size(tumor_reads_index, "GB"))
    Int normal_reads_size = if defined(control_bam) then ceil(size(control_bam, "GB") + size(control_bai, "GB")) else 0

    Int m2_output_size = tumor_reads_size / scatter_count
    #TODO: do we need to change this disk size now that NIO is always going to happen (for the google backend only)
    Int m2_per_scatter_size = (tumor_reads_size + normal_reads_size) + ref_size + gnomad_vcf_size + m2_output_size + disk_pad

    call SplitIntervals {
        input:
            intervals = intervals,
            masked_intervals = masked_intervals,
            ref_fasta = ref_fasta,
            ref_fai = ref_fai,
            ref_dict = ref_dict,
            scatter_count = scatter_count,
            split_intervals_extra_args = split_intervals_extra_args,
            runtime_params = standard_runtime
    }

    scatter (subintervals in SplitIntervals.interval_files ) {
        call M2 {
            input:
                intervals = subintervals,
                ref_fasta = ref_fasta,
                ref_fai = ref_fai,
                ref_dict = ref_dict,
                tumor_reads = tumor_reads,
                tumor_reads_index = tumor_reads_index,
                normal_reads = normal_reads,
                normal_reads_index = normal_reads_index,
                gnomad = gnomad,
                gnomad_idx = gnomad_idx,
                preemptible = preemptible,
                max_retries = max_retries,
                m2_extra_args = m2_extra_args,
                variants_for_contamination = variants_for_contamination,
                variants_for_contamination_idx = variants_for_contamination_idx,
                dragstr_model = dragstr_model,
                make_bamout = make_bamout,
                run_ob_filter = run_orientation_bias_mixture_model_filter,
                compress_vcfs = compress_vcfs,
                make_permutect_training_dataset = false,
                make_permutect_test_dataset = true,
                permutect_test_dataset_truth_vcf = test_dataset_truth_vcf,
                permutect_test_dataset_truth_vcf_idx = test_dataset_truth_vcf_idx,
                gatk_override = gatk_override,
                gatk_docker = gatk_docker,
                disk_space = m2_per_scatter_size,
                gcs_project_for_requester_pays = gcs_project_for_requester_pays
        }
    }

    Int merged_vcf_size = ceil(size(M2.unfiltered_vcf, "GB"))
    Int merged_bamout_size = ceil(size(M2.output_bamOut, "GB"))

    if (run_orientation_bias_mixture_model_filter && (!skip_filtering)) {
        call LearnReadOrientationModel {
            input:
                f1r2_tar_gz = M2.f1r2_counts,
                runtime_params = standard_runtime,
                mem = learn_read_orientation_mem
        }
    }

    call MergeVCFs {
        input:
            input_vcfs = M2.unfiltered_vcf,
            input_vcf_indices = M2.unfiltered_vcf_idx,
            compress_vcfs = compress_vcfs,
            runtime_params = standard_runtime
    }

    if (make_bamout) {
        call MergeBamOuts {
            input:
                ref_fasta = ref_fasta,
                ref_fai = ref_fai,
                ref_dict = ref_dict,
                bam_outs = M2.output_bamOut,
                runtime_params = standard_runtime,
                disk_space = ceil(merged_bamout_size * 4) + disk_pad,
        }
    }

    call MergeStats { input: stats = M2.stats, runtime_params = standard_runtime }

    if (defined(variants_for_contamination) && (!skip_filtering)) {
        call MergePileupSummaries as MergeTumorPileups {
            input:
                input_tables = flatten(M2.tumor_pileups),
                output_name = "tumor-pileups",
                ref_dict = ref_dict,
                runtime_params = standard_runtime
        }

        if (defined(normal_reads)){
            call MergePileupSummaries as MergeNormalPileups {
                input:
                    input_tables = flatten(M2.normal_pileups),
                    output_name = "normal-pileups",
                    ref_dict = ref_dict,
                    runtime_params = standard_runtime
            }
        }

        call CalculateContamination {
            input:
                tumor_pileups = MergeTumorPileups.merged_table,
                normal_pileups = MergeNormalPileups.merged_table,
                runtime_params = standard_runtime
        }

        # call CalculateContamination with the normal as the "tumor" in order to get normal MAF segments
        if (defined(normal_reads)){
            call CalculateContamination as GetNormalMAFSegments {
                input:
                    tumor_pileups = select_first([MergeNormalPileups.merged_table]),
                    runtime_params = standard_runtime
            }
        }
    }

    call Concatenate as ConcatenatePermutectTestData {
        input:
            input_files = M2.permutect_test_dataset,
            gatk_docker = gatk_docker
    }

    if (!skip_filtering) {
        call Filter {
            input:
                ref_fasta = ref_fasta,
                ref_fai = ref_fai,
                ref_dict = ref_dict,
                intervals = intervals,
                unfiltered_vcf = MergeVCFs.merged_vcf,
                unfiltered_vcf_idx = MergeVCFs.merged_vcf_idx,
                compress_vcfs = compress_vcfs,
                mutect_stats = MergeStats.merged_stats,
                contamination_table = CalculateContamination.contamination_table,
                maf_segments = CalculateContamination.maf_segments,
                artifact_priors_tar_gz = LearnReadOrientationModel.artifact_prior_table,
                m2_extra_filtering_args = m2_extra_filtering_args,
                runtime_params = standard_runtime,
                disk_space = ceil(size(MergeVCFs.merged_vcf, "GB") * 4) + disk_pad
        }

        if (defined(realignment_index_bundle)) {
            call FilterAlignmentArtifacts {
                input:
                    ref_fasta = ref_fasta,
                    ref_fai = ref_fai,
                    ref_dict = ref_dict,
                    reads = tumor_reads,
                    reads_index = tumor_reads_index,
                    realignment_index_bundle = select_first([realignment_index_bundle]),
                    realignment_extra_args = realignment_extra_args,
                    compress_vcfs = compress_vcfs,
                    input_vcf = Filter.filtered_vcf,
                    input_vcf_idx = Filter.filtered_vcf_idx,
                    runtime_params = standard_runtime,
                    mem = filter_alignment_artifacts_mem,
                    gcs_project_for_requester_pays = gcs_project_for_requester_pays
            }
        }
    }

    # THIS WAS THE OUTPUT BLOCK OF Mutect2
    File mutect2_output_vcf = select_first([FilterAlignmentArtifacts.filtered_vcf, Filter.filtered_vcf, MergeVCFs.merged_vcf])
    File mutect2_output_vcf_idx = select_first([FilterAlignmentArtifacts.filtered_vcf_idx, Filter.filtered_vcf_idx,MergeVCFs.merged_vcf_idx])
    File mutect2_permutect_test_dataset = ConcatenatePermutectTestData.concatenated
    File? mutect2_bamout = MergeBamOuts.merged_bam_out
    File? mutect2_bamout_index = MergeBamOuts.merged_bam_out_index
    File? mutect2_filtering_stats = Filter.filtering_stats
    File mutect2_mutect_stats = MergeStats.merged_stats
    File? mutect2_contamination_table = CalculateContamination.contamination_table

    File? mutect2_maf_segments = CalculateContamination.maf_segments
    File? mutect2_normal_maf_segments = GetNormalMAFSegments.maf_segments
    File? mutect2_read_orientation_model_params = LearnReadOrientationModel.artifact_prior_table
    File? mutect2_permutect_test_dataset = ConcatenatePermutectTestData.concatenated
    File mutect2_permutect_contigs_table = select_first(M2.permutect_contigs_table)
    File mutect2_permutect_read_groups_table = select_first(M2.permutect_read_groups_table)


    call SplitMultiallelics {
        input:
            input_vcf = mutect2_output_vcf,
            input_vcf_idx = mutect2_output_vcf_idx,
            ref_fasta = ref_fasta,
            ref_fai = ref_fai,
            ref_dict = ref_dict,
            bcftools_docker = bcftools_docker
    }

    call IndexVCF as IndexAfterSplitting {
        input:
            unindexed_vcf = SplitMultiallelics.output_vcf,
            gatk_docker = gatk_docker
    }

    call PermutectFiltering {
        input:
            mutect2_vcf = IndexAfterSplitting.vcf,
            mutect2_vcf_idx = IndexAfterSplitting.vcf_index,
            permutect_model = permutect_model,
            test_dataset = mutect2_permutect_test_dataset,
            contigs_table = mutect2_permutect_contigs_table,
            maf_segments = mutect2_maf_segments,
            normal_maf_segments = mutect2_normal_maf_segments,
            mutect_stats = mutect2_mutect_stats,
            batch_size = batch_size,
            num_workers = num_workers,
            gpu_count = gpu_count,
            num_spectrum_iterations = num_spectrum_iterations,
            spectrum_learning_rate = spectrum_learning_rate,
            chunk_size = chunk_size,
            permutect_filtering_extra_args = permutect_filtering_extra_args,
            permutect_docker = permutect_docker,
    }

    call IndexVCF as IndexAfterFiltering {
        input:
            unindexed_vcf = PermutectFiltering.output_vcf,
            gatk_docker = gatk_docker
    }

    output {
        File output_vcf = IndexAfterFiltering.vcf
        File output_vcf_idx = IndexAfterFiltering.vcf_index
        File tensorboard_report = PermutectFiltering.tensorboard_report
        File test_dataset = mutect2_permutect_test_dataset
        File mutect2_vcf = mutect2_output_vcf
        File mutect2_vcf_idx = mutect2_output_vcf_idx
        File? maf_segments = mutect2_maf_segments
        File? normal_maf_segments = mutect2_normal_maf_segments
    }
}

 task PermutectFiltering {
    input {
        File permutect_model
        File test_dataset
        File contigs_table
        File mutect2_vcf
        File mutect2_vcf_idx
        File? maf_segments
        File? normal_maf_segments
        File mutect_stats
        Int? num_spectrum_iterations
        Float? spectrum_learning_rate
        Int batch_size
        Int num_workers
        Int? gpu_count
        Int chunk_size
        String? permutect_filtering_extra_args

        String permutect_docker
        Int? preemptible
        Int? max_retries
        Int? disk_space
        Int? cpu
        Int? mem
    }

    # Mem is in units of GB but our command and memory runtime values are in MB
    Int machine_mem = if defined(mem) then mem * 1000 else 16000
    Int command_mem = machine_mem - 500

    command <<<
        # set -e
        genomic_span=`grep "callable" ~{mutect_stats} | while read name value; do echo $value; done`

        filter_variants --input ~{mutect2_vcf} --test_dataset ~{test_dataset} \
            --saved_model ~{permutect_model} \
            --contigs_table ~{contigs_table} \
            --output permutect-filtered.vcf \
            --tensorboard_dir tensorboard \
            --batch_size ~{batch_size} --num_workers ~{num_workers} --chunk_size ~{chunk_size} \
            ~{" --num_spectrum_iterations " + num_spectrum_iterations} \
            ~{" --spectrum_learning_rate " + spectrum_learning_rate} \
            ~{" --maf_segments " + maf_segments} ~{" --normal_maf_segments " + normal_maf_segments} \
            --genomic_span $genomic_span ~{permutect_filtering_extra_args}

        tar cvf tensorboard.tar tensorboard/
    >>>

    runtime {
        docker: permutect_docker
        bootDiskSizeGb: 12
        memory: machine_mem + " MB"
        disks: "local-disk " + select_first([disk_space, 100]) + " SSD"
        preemptible: select_first([preemptible, 0])
        maxRetries: select_first([max_retries, 0])
        cpu: select_first([cpu, 2])
        gpuType: "nvidia-tesla-t4"
        gpuCount: select_first([gpu_count, 1])
        nvidiaDriverVersion: "535.183.01"
        zones : ["us-central1-a", "us-central1-b", "us-central1-c"]
    }

    output {
        File output_vcf = "permutect-filtered.vcf"
        File tensorboard_report = "tensorboard.tar"
    }
}

task SplitMultiallelics {
    input {
        File input_vcf
        File input_vcf_idx
        File ref_fasta
        File ref_fai
        File ref_dict
        String bcftools_docker
        Int? preemptible
        Int? max_retries
        Int? disk_space
        Int? cpu
        Int? mem
    }

    # Mem is in units of GB but our command and memory runtime values are in MB
    Int machine_mem = if defined(mem) then mem * 1000 else 4000
    Int command_mem = machine_mem - 500

    command <<<

        bcftools norm -m -any -f ~{ref_fasta} ~{input_vcf} > output.vcf

    >>>

    runtime {
        docker: bcftools_docker
        bootDiskSizeGb: 12
        memory: machine_mem + " MB"
        disks: "local-disk " + select_first([disk_space, 100]) + " SSD"
        preemptible: select_first([preemptible, 0])
        maxRetries: select_first([max_retries, 0])
        cpu: select_first([cpu, 2])
    }

    output {
        File output_vcf = "output.vcf"
    }
}

task IndexVCF {
    input {
        File unindexed_vcf
        String gatk_docker
        Int? preemptible
        Int? max_retries
        Int? disk_space
        Int? cpu
        Int? mem
    }

    # Mem is in units of GB but our command and memory runtime values are in MB
    Int machine_mem = if defined(mem) then mem * 1000 else 4000
    Int command_mem = machine_mem - 500

    command <<<

        cp ~{unindexed_vcf} indexed.vcf

        gatk --java-options "-Xmx~{command_mem}m" IndexFeatureFile -I indexed.vcf

        gatk --java-options "-Xmx~{command_mem}m" SelectVariants -V indexed.vcf -O output.vcf --lenient \
            -DGA DP -DGA AF -DGA F1R2 -DGA F2R1 -DGA FAD -DGA SB \
            -DA AS_FilterStatus -DA AS_SB_TABLE -DA ECNT -DA GERMQ -DA MBQ -DA MFRL -DA MMQ -DA MPOS

        set -e
    >>>

    runtime {
        docker: gatk_docker
        bootDiskSizeGb: 12
        memory: machine_mem + " MB"
        disks: "local-disk " + select_first([disk_space, 100]) + " SSD"
        preemptible: select_first([preemptible, 0])
        maxRetries: select_first([max_retries, 0])
        cpu: select_first([cpu, 1])
    }

    output {
        File vcf = "output.vcf"
        File vcf_index = "output.vcf.idx"
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
      String? split_intervals_extra_args

      # runtime
      Runtime runtime_params
    }

    command {
        set -e
        export GATK_LOCAL_JAR=~{default="/root/gatk.jar" runtime_params.gatk_override}

        mkdir interval-files
        gatk --java-options "-Xmx~{runtime_params.command_mem}m" SplitIntervals \
            -R ~{ref_fasta} \
            ~{"-L " + intervals} \
            ~{"-XL " + masked_intervals} \
            -scatter ~{scatter_count} \
            -O interval-files \
            ~{split_intervals_extra_args}
        cp interval-files/*.interval_list .
    }

    runtime {
        docker: runtime_params.gatk_docker
        bootDiskSizeGb: runtime_params.boot_disk_size
        memory: runtime_params.machine_mem + " MB"
        disks: "local-disk " + runtime_params.disk + " HDD"
        preemptible: runtime_params.preemptible
        maxRetries: runtime_params.max_retries
        cpu: runtime_params.cpu
    }

    output {
        Array[File] interval_files = glob("*.interval_list")
    }
}

task M2 {
    input {
        File? intervals
        File ref_fasta
        File ref_fai
        File ref_dict
        File tumor_reads
        File tumor_reads_index
        File? normal_reads
        File? normal_reads_index
        File? pon
        File? pon_idx
        File? gnomad
        File? gnomad_idx
        String? m2_extra_args
        String? getpileupsummaries_extra_args
        Boolean? make_bamout
        Boolean? run_ob_filter
        Boolean compress_vcfs
        File? gga_vcf
        File? gga_vcf_idx
        File? variants_for_contamination
        File? variants_for_contamination_idx
        File? dragstr_model

        File? gatk_override

        String? gcs_project_for_requester_pays

        Boolean make_permutect_training_dataset = false
        Boolean make_permutect_test_dataset = false
        File? permutect_training_dataset_truth_vcf
        File? permutect_training_dataset_truth_vcf_idx
        File? permutect_test_dataset_truth_vcf
        File? permutect_test_dataset_truth_vcf_idx

        # runtime
        String gatk_docker
        Int? mem
        Int? preemptible
        Int? max_retries
        Int? disk_space
        Int? cpu
        Boolean use_ssd = false
    }

    String output_vcf = "output" + if compress_vcfs then ".vcf.gz" else ".vcf"
    String output_vcf_idx = output_vcf + if compress_vcfs then ".tbi" else ".idx"

    String output_stats = output_vcf + ".stats"

    # Mem is in units of GB but our command and memory runtime values are in MB
    Int machine_mem = if defined(mem) then mem * 1000 else 3500
    Int command_mem = machine_mem - 500

    parameter_meta{
        intervals: {localization_optional: true}
        ref_fasta: {localization_optional: true}
        ref_fai: {localization_optional: true}
        ref_dict: {localization_optional: true}
        tumor_reads: {localization_optional: true}
        tumor_reads_index: {localization_optional: true}
        normal_reads: {localization_optional: true}
        normal_reads_index: {localization_optional: true}
        pon: {localization_optional: true}
        pon_idx: {localization_optional: true}
        gnomad: {localization_optional: true}
        gnomad_idx: {localization_optional: true}
        gga_vcf: {localization_optional: true}
        gga_vcf_idx: {localization_optional: true}
        variants_for_contamination: {localization_optional: true}
        variants_for_contamination_idx: {localization_optional: true}
        permutect_training_dataset_truth_vcf: {localization_optional: true}
        permutect_training_dataset_truth_vcf_idx: {localization_optional: true}
    }

    command <<<
        set -e

        export GATK_LOCAL_JAR=~{default="/root/gatk.jar" gatk_override}

        # We need to create these files regardless, even if they stay empty
        touch bamout.bam
        touch f1r2.tar.gz
        touch training-dataset.txt
        touch test-dataset.txt
        touch contigs.table
        touch read-groups.table

        if [[ ! -z "~{normal_reads}" ]]; then
            gatk --java-options "-Xmx~{command_mem}m" GetSampleName -R ~{ref_fasta} -I ~{normal_reads} -O normal_names.txt -encode \
            ~{"--gcs-project-for-requester-pays " + gcs_project_for_requester_pays}
            # add "-normal " to the start of each line and " " to the end, then remove newlines
            # to get -normal sample1 -normal sample2 etc
            normal_sample_line=`awk '{ print "-normal", $0 }' normal_names.txt | tr '\n' ' '`
        fi

        gatk --java-options "-Xmx~{command_mem}m" Mutect2 \
            -R ~{ref_fasta} \
            -I ~{tumor_reads} \
            ~{"-I " + normal_reads} \
            $normal_sample_line \
            ~{"--germline-resource " + gnomad} \
            ~{"-pon " + pon} \
            ~{"-L " + intervals} \
            ~{"--alleles " + gga_vcf} \
            ~{"--dragstr-params-path " + dragstr_model} \
            -O "~{output_vcf}" \
            ~{true='--bam-output bamout.bam' false='' make_bamout} \
            ~{true='--f1r2-tar-gz f1r2.tar.gz' false='' run_ob_filter} \
            ~{true='--permutect-training-dataset training-dataset.txt' false='' make_permutect_training_dataset} \
            ~{true='--permutect-test-dataset test-dataset.txt' false='' make_permutect_test_dataset} \
            ~{"--permutect-training-truth " + permutect_training_dataset_truth_vcf} \
            ~{"--permutect-test-truth " + permutect_test_dataset_truth_vcf} \
            ~{m2_extra_args} \
            ~{"--gcs-project-for-requester-pays " + gcs_project_for_requester_pays}

        m2_exit_code=$?

        ### GetPileupSummaries

        # If the variants for contamination and the intervals for this scatter don't intersect, GetPileupSummaries
        # throws an error.  However, there is nothing wrong with an empty intersection for our purposes; it simply doesn't
        # contribute to the merged pileup summaries that we create downstream.  We implement this via array outputs.
        # If the tool errors, no table is created and the glob yields an empty array.
        set +e

        if [[ ! -z "~{variants_for_contamination}" ]]; then
            gatk --java-options "-Xmx~{command_mem}m" GetPileupSummaries -R ~{ref_fasta} -I ~{tumor_reads} ~{"--interval-set-rule INTERSECTION -L " + intervals} \
                -V ~{variants_for_contamination} -L ~{variants_for_contamination} -O tumor-pileups.table ~{getpileupsummaries_extra_args} \
                ~{"--gcs-project-for-requester-pays " + gcs_project_for_requester_pays}


            if [[ ! -z "~{normal_reads}" ]]; then
                gatk --java-options "-Xmx~{command_mem}m" GetPileupSummaries -R ~{ref_fasta} -I ~{normal_reads} ~{"--interval-set-rule INTERSECTION -L " + intervals} \
                    -V ~{variants_for_contamination} -L ~{variants_for_contamination} -O normal-pileups.table ~{getpileupsummaries_extra_args} \
                    ~{"--gcs-project-for-requester-pays " + gcs_project_for_requester_pays}
            fi
        fi

        # the script only fails if Mutect2 itself fails
        exit $m2_exit_code
    >>>

    runtime {
        docker: gatk_docker
        bootDiskSizeGb: 12
        memory: machine_mem + " MB"
        disks: "local-disk " + select_first([disk_space, 100]) + if use_ssd then " SSD" else " HDD"
        preemptible: select_first([preemptible, 10])
        maxRetries: select_first([max_retries, 0])
        cpu: select_first([cpu, 1])
    }

    output {
        File unfiltered_vcf = "~{output_vcf}"
        File unfiltered_vcf_idx = "~{output_vcf_idx}"
        File output_bamOut = "bamout.bam"
        File stats = "~{output_stats}"
        File f1r2_counts = "f1r2.tar.gz"
        Array[File] tumor_pileups = glob("*tumor-pileups.table")
        Array[File] normal_pileups = glob("*normal-pileups.table")
        File permutect_training_dataset = "training-dataset.txt"
        File permutect_test_dataset = "test-dataset.txt"
        File permutect_contigs_table = "contigs.table"
        File permutect_read_groups_table = "read-groups.table"
    }
}

task MergeVCFs {
    input {
      Array[File] input_vcfs
      Array[File] input_vcf_indices
      Boolean compress_vcfs
      Runtime runtime_params
    }

    String output_vcf = if compress_vcfs then "merged.vcf.gz" else "merged.vcf"
    String output_vcf_idx = output_vcf + if compress_vcfs then ".tbi" else ".idx"

    # using MergeVcfs instead of GatherVcfs so we can create indices
    # WARNING 2015-10-28 15:01:48 GatherVcfs  Index creation not currently supported when gathering block compressed VCFs.
    command {
        set -e
        export GATK_LOCAL_JAR=~{default="/root/gatk.jar" runtime_params.gatk_override}
        gatk --java-options "-Xmx~{runtime_params.command_mem}m" MergeVcfs -I ~{sep=' -I ' input_vcfs} -O ~{output_vcf}
    }

    runtime {
        docker: runtime_params.gatk_docker
        bootDiskSizeGb: runtime_params.boot_disk_size
        memory: runtime_params.machine_mem + " MB"
        disks: "local-disk " + runtime_params.disk + " HDD"
        preemptible: runtime_params.preemptible
        maxRetries: runtime_params.max_retries
        cpu: runtime_params.cpu
    }

    output {
        File merged_vcf = "~{output_vcf}"
        File merged_vcf_idx = "~{output_vcf_idx}"
    }
}

task MergeBamOuts {
    input {
      File ref_fasta
      File ref_fai
      File ref_dict
      Array[File]+ bam_outs
      Runtime runtime_params
      Int? disk_space   #override to request more disk than default small task params
    }

    command <<<
        # This command block assumes that there is at least one file in bam_outs.
        #  Do not call this task if len(bam_outs) == 0
        set -e
        export GATK_LOCAL_JAR=~{default="/root/gatk.jar" runtime_params.gatk_override}
        gatk --java-options "-Xmx~{runtime_params.command_mem}m" GatherBamFiles \
            -I ~{sep=" -I " bam_outs} -O unsorted.out.bam -R ~{ref_fasta}

        # We must sort because adjacent scatters may have overlapping (padded) assembly regions, hence
        # overlapping bamouts

        gatk --java-options "-Xmx~{runtime_params.command_mem}m" SortSam -I unsorted.out.bam \
            -O bamout.bam --SORT_ORDER coordinate -VALIDATION_STRINGENCY LENIENT
        gatk --java-options "-Xmx~{runtime_params.command_mem}m" BuildBamIndex -I bamout.bam -VALIDATION_STRINGENCY LENIENT
    >>>

    runtime {
        docker: runtime_params.gatk_docker
        bootDiskSizeGb: runtime_params.boot_disk_size
        memory: runtime_params.machine_mem + " MB"
        disks: "local-disk " + select_first([disk_space, runtime_params.disk]) + " HDD"
        preemptible: runtime_params.preemptible
        maxRetries: runtime_params.max_retries
        cpu: runtime_params.cpu
    }

    output {
        File merged_bam_out = "bamout.bam"
        File merged_bam_out_index = "bamout.bai"
    }
}


task MergeStats {
    input {
      Array[File]+ stats
      Runtime runtime_params
    }

    command {
        set -e
        export GATK_LOCAL_JAR=~{default="/root/gatk.jar" runtime_params.gatk_override}


        gatk --java-options "-Xmx~{runtime_params.command_mem}m" MergeMutectStats \
            -stats ~{sep=" -stats " stats} -O merged.stats
    }

    runtime {
        docker: runtime_params.gatk_docker
        bootDiskSizeGb: runtime_params.boot_disk_size
        memory: runtime_params.machine_mem + " MB"
        disks: "local-disk " + runtime_params.disk + " HDD"
        preemptible: runtime_params.preemptible
        maxRetries: runtime_params.max_retries
        cpu: runtime_params.cpu
    }

    output {
        File merged_stats = "merged.stats"
    }
}

task MergePileupSummaries {
    input {
      Array[File] input_tables
      String output_name
      File ref_dict
      Runtime runtime_params
    }

    command {
        set -e
        export GATK_LOCAL_JAR=~{default="/root/gatk.jar" runtime_params.gatk_override}

        gatk --java-options "-Xmx~{runtime_params.command_mem}m" GatherPileupSummaries \
        --sequence-dictionary ~{ref_dict} \
        -I ~{sep=' -I ' input_tables} \
        -O ~{output_name}.tsv
    }

    runtime {
        docker: runtime_params.gatk_docker
        bootDiskSizeGb: runtime_params.boot_disk_size
        memory: runtime_params.machine_mem + " MB"
        disks: "local-disk " + runtime_params.disk + " HDD"
        preemptible: runtime_params.preemptible
        maxRetries: runtime_params.max_retries
        cpu: runtime_params.cpu
    }

    output {
        File merged_table = "~{output_name}.tsv"
    }
}

# Learning step of the orientation bias mixture model, which is the recommended orientation bias filter as of September 2018
task LearnReadOrientationModel {
    input {
      Array[File] f1r2_tar_gz
      Runtime runtime_params
      Int? mem  #override memory
    }

    Int machine_mem = select_first([mem, runtime_params.machine_mem])
    Int command_mem = machine_mem - 1000

    command {
        set -e
        export GATK_LOCAL_JAR=~{default="/root/gatk.jar" runtime_params.gatk_override}

        gatk --java-options "-Xmx~{command_mem}m" LearnReadOrientationModel \
            -I ~{sep=" -I " f1r2_tar_gz} \
            -O "artifact-priors.tar.gz"
    }

    runtime {
        docker: runtime_params.gatk_docker
        bootDiskSizeGb: runtime_params.boot_disk_size
        memory: machine_mem + " MB"
        disks: "local-disk " + runtime_params.disk + " HDD"
        preemptible: runtime_params.preemptible
        maxRetries: runtime_params.max_retries
        cpu: runtime_params.cpu
    }

    output {
        File artifact_prior_table = "artifact-priors.tar.gz"
    }

}

task CalculateContamination {
    input {
      String? intervals
      File tumor_pileups
      File? normal_pileups
      Runtime runtime_params
    }

    command {
        set -e

        export GATK_LOCAL_JAR=~{default="/root/gatk.jar" runtime_params.gatk_override}

        gatk --java-options "-Xmx~{runtime_params.command_mem}m" CalculateContamination -I ~{tumor_pileups} \
        -O contamination.table --tumor-segmentation segments.table ~{"-matched " + normal_pileups}
    }

    runtime {
        docker: runtime_params.gatk_docker
        bootDiskSizeGb: runtime_params.boot_disk_size
        memory: runtime_params.machine_mem + " MB"
        disks: "local-disk " + runtime_params.disk + " HDD"
        preemptible: runtime_params.preemptible
        maxRetries: runtime_params.max_retries
        cpu: runtime_params.cpu
    }

    output {
        File contamination_table = "contamination.table"
        File maf_segments = "segments.table"
    }
}

task Filter {
    input {
        File? intervals
        File ref_fasta
        File ref_fai
        File ref_dict
        File unfiltered_vcf
        File unfiltered_vcf_idx
        Boolean compress_vcfs
        File? mutect_stats
        File? artifact_priors_tar_gz
        File? contamination_table
        File? maf_segments
        String? m2_extra_filtering_args

        Runtime runtime_params
        Int? disk_space
    }

    String output_vcf = if compress_vcfs then "filtered.vcf.gz" else "filtered.vcf"
    String output_vcf_idx = output_vcf + if compress_vcfs then ".tbi" else ".idx"

    parameter_meta{
      ref_fasta: {localization_optional: true}
      ref_fai: {localization_optional: true}
      ref_dict: {localization_optional: true}
    }

    command {
        set -e

        export GATK_LOCAL_JAR=~{default="/root/gatk.jar" runtime_params.gatk_override}

        gatk --java-options "-Xmx~{runtime_params.command_mem}m" FilterMutectCalls -V ~{unfiltered_vcf} \
            -R ~{ref_fasta} \
            -O ~{output_vcf} \
            ~{"--contamination-table " + contamination_table} \
            ~{"--tumor-segmentation " + maf_segments} \
            ~{"--ob-priors " + artifact_priors_tar_gz} \
            ~{"-stats " + mutect_stats} \
            --filtering-stats filtering.stats \
            ~{m2_extra_filtering_args}
    }

    runtime {
        docker: runtime_params.gatk_docker
        bootDiskSizeGb: runtime_params.boot_disk_size
        memory: runtime_params.machine_mem + " MB"
        disks: "local-disk " + select_first([disk_space, runtime_params.disk]) + " HDD"
        preemptible: runtime_params.preemptible
        maxRetries: runtime_params.max_retries
        cpu: runtime_params.cpu
    }

    output {
        File filtered_vcf = "~{output_vcf}"
        File filtered_vcf_idx = "~{output_vcf_idx}"
        File filtering_stats = "filtering.stats"
    }
}

task FilterAlignmentArtifacts {
    input {
      File ref_fasta
      File ref_fai
      File ref_dict
      File input_vcf
      File input_vcf_idx
      File reads
      File reads_index
      Boolean compress_vcfs
      File realignment_index_bundle
      String? realignment_extra_args
      String? gcs_project_for_requester_pays
      Runtime runtime_params
      Int mem
    }

    String output_vcf = if compress_vcfs then "filtered.vcf.gz" else "filtered.vcf"
    String output_vcf_idx = output_vcf +  if compress_vcfs then ".tbi" else ".idx"

    Int machine_mem = mem
    Int command_mem = machine_mem - 500

    parameter_meta{
      ref_fasta: {localization_optional: true}
      ref_fai: {localization_optional: true}
      ref_dict: {localization_optional: true}
      input_vcf: {localization_optional: true}
      input_vcf_idx: {localization_optional: true}
      reads: {localization_optional: true}
      reads_index: {localization_optional: true}
    }

    command {
        set -e

        export GATK_LOCAL_JAR=~{default="/root/gatk.jar" runtime_params.gatk_override}

        gatk --java-options "-Xmx~{command_mem}m" FilterAlignmentArtifacts \
            -R ~{ref_fasta} \
            -V ~{input_vcf} \
            -I ~{reads} \
            --bwa-mem-index-image ~{realignment_index_bundle} \
            ~{realignment_extra_args} \
            -O ~{output_vcf} \
            ~{"--gcs-project-for-requester-pays " + gcs_project_for_requester_pays}
    }

    runtime {
        docker: runtime_params.gatk_docker
        bootDiskSizeGb: runtime_params.boot_disk_size
        memory: machine_mem + " MB"
        disks: "local-disk " + runtime_params.disk + " HDD"
        preemptible: runtime_params.preemptible
        maxRetries: runtime_params.max_retries
        cpu: runtime_params.cpu
    }

    output {
        File filtered_vcf = "~{output_vcf}"
        File filtered_vcf_idx = "~{output_vcf_idx}"
    }
}

task Concatenate {
    input {
        Array[File] input_files
        Int? mem
        String gatk_docker
    }

    Int machine_mem = if defined(mem) then mem * 1000 else 7000

    command {
        cat ~{sep=' ' input_files} > output.txt
    }

    runtime {
        docker: gatk_docker
        bootDiskSizeGb: 12
        memory: machine_mem + " MB"
        disks: "local-disk 100 HDD"
        preemptible: 1
        maxRetries: 1
        cpu: 2
    }

    output {
        File concatenated = "output.txt"
    }
}
