version 1.0

import "https://api.firecloud.org/ga4gh/v1/tools/davidben:mutect2/versions/20/plain-WDL/descriptor" as m2

workflow Permutect {
    input {
        File artifact_model

        File? intervals
        File? masks
        File ref_fasta
        File ref_fai
        File ref_dict
        Int scatter_count
        Int? num_spectrum_iterations
        Float? spectrum_learning_rate
        File primary_bam
        File primary_bai
        File? control_bam
        File? control_bai
        File? gnomad
        File? gnomad_idx
        File? variants_for_contamination
        File? variants_for_contamination_idx
        File? realignment_index_bundle
        File? dragstr_model
        String? realignment_extra_args
        Boolean skip_m2_filtering = true
        Boolean run_orientation_bias_mixture_model_filter = false
        Boolean filter_pcr_errors = false   # run pre_permutect_filtering, which applies the hairpin filter
        String? m2_extra_args
        String? split_intervals_extra_args
        Int batch_size
        Int num_workers
        Int? gpu_count
        File? test_dataset_truth_vcf    # used for evaluation
        File? test_dataset_truth_vcf_idx
        String? concordance_header

        # These HACKS are required to get around Terra's unreliable call-caching
        # note: defining them as Strings instead of Files is essential because otherwise
        # a cached input gs://[some hash]/final-outputs/file.txt is written as
        # gs://[some hash]/final-outputs/[another hash]/final-outputs/file.txt
        # and eventually the file name gets too long and throws an error in Terra/cromwell.
        # if we define them as Strings it uses the literal value
        String? cached_plain_text_test_dataset
        String? cached_mutect2_vcf
        String? cached_mutect2_vcf_idx
        String? cached_contigs_table
        String? cached_maf_segments
        String? cached_normal_maf_segments
        String? cached_mutect_stats

        String? permutect_filtering_extra_args
        String? gcs_project_for_requester_pays
        File? gatk_override
        String permutect_docker
        Int? preemptible
        Int? max_retries

        # WDL version 1.0 does not have an empty Optional literal
        # such a literal is very useful because Terra has a bug where whenever a data table is updated, empty values
        # silently and invisibly get converted to empty strings "".  Thus it is useful to recognize empty strings and
        # declare empty Optionals.  The only way to do this in WDL 1.0 is to get an empty Optional as a variable from the
        # workflow inputs.  These inputs should NEVER be filled in!!!!!
        File? EMPTY_STRING_HACK
    }

    # I (David Benjamin, that is) always run Permutect on Terra, which has the following extremely irritating bug:
    # whenever you upload new data to your Terra workspace, values in the table that were previously unset
    # (i.e. None, empty Optional) are **silently and invisibly** (there's no sign in the GUI) rewritten as
    # empty String literals.  That is, your Nones become ""s.  This ruins workflows with optional inputs, of course.
    # The workaround is to explicitly test for Strings equal to "", which are actually empty Optionals that
    # Terra converted.

    File? intervals_hack = if (select_first([intervals,""]) == "") then EMPTY_STRING_HACK else intervals
    File? masks_hack = if (select_first([masks,""]) == "") then EMPTY_STRING_HACK else masks

    File? control_bam_hack = if select_first([control_bam, ""]) == "" then EMPTY_STRING_HACK else control_bam
    File? control_bai_hack = if select_first([control_bai, ""]) == "" then EMPTY_STRING_HACK else control_bai
    File? gnomad_hack = if select_first([gnomad, ""]) == "" then EMPTY_STRING_HACK else gnomad
    File? gnomad_idx_hack = if select_first([gnomad_idx, ""]) == "" then EMPTY_STRING_HACK else gnomad_idx

    File? variants_for_contamination_hack = if select_first([variants_for_contamination, ""]) == "" then EMPTY_STRING_HACK else variants_for_contamination
    File? variants_for_contamination_idx_hack = if select_first([variants_for_contamination_idx, ""]) == "" then EMPTY_STRING_HACK else variants_for_contamination_idx
    File? realignment_index_bundle_hack = if select_first([realignment_index_bundle, ""]) == "" then EMPTY_STRING_HACK else realignment_index_bundle
    File? dragstr_model_hack = if select_first([dragstr_model, ""]) == "" then EMPTY_STRING_HACK else dragstr_model
    File? test_dataset_truth_vcf_hack = if select_first([test_dataset_truth_vcf, ""]) == "" then EMPTY_STRING_HACK else test_dataset_truth_vcf
    File? test_dataset_truth_vcf_idx_hack = if select_first([test_dataset_truth_vcf_idx, ""]) == "" then EMPTY_STRING_HACK else test_dataset_truth_vcf_idx

    File? cached_dataset_hack = if (select_first([cached_plain_text_test_dataset,""]) == "") then EMPTY_STRING_HACK else cached_plain_text_test_dataset
    String? cached_mutect2_vcf_hack = if (select_first([cached_mutect2_vcf,""]) == "") then EMPTY_STRING_HACK else cached_mutect2_vcf
    String? cached_mutect2_vcf_idx_hack = if (select_first([cached_mutect2_vcf_idx,""]) == "") then EMPTY_STRING_HACK else cached_mutect2_vcf_idx
    String? cached_contigs_table_hack = if (select_first([cached_contigs_table,""]) == "") then EMPTY_STRING_HACK else cached_contigs_table
    String? cached_maf_segments_hack = if (select_first([cached_maf_segments,""]) == "") then EMPTY_STRING_HACK else cached_maf_segments
    String? cached_normal_maf_segments_hack = if (select_first([cached_normal_maf_segments,""]) == "") then EMPTY_STRING_HACK else cached_normal_maf_segments
    String? cached_mutect_stats_hack = if (select_first([cached_mutect_stats,""]) == "") then EMPTY_STRING_HACK else cached_mutect_stats

    String? gcs_project_for_requester_pays_hack = if select_first([gcs_project_for_requester_pays, ""]) == "" then EMPTY_STRING_HACK else gcs_project_for_requester_pays

    Boolean use_cache = defined(cached_dataset_hack) && defined(cached_mutect2_vcf_hack) && defined(cached_contigs_table_hack) && defined(cached_mutect_stats_hack)
    if(!use_cache) {
        call m2.Mutect2 {
            input:
                make_permutect_training_dataset = false,
                make_permutect_test_dataset = true,
                permutect_test_dataset_truth_vcf = test_dataset_truth_vcf_hack,
                permutect_test_dataset_truth_vcf_idx = test_dataset_truth_vcf_idx_hack,
                intervals = intervals_hack,
                masked_intervals = masks_hack,
                ref_fasta = ref_fasta,
                ref_fai = ref_fai,
                ref_dict = ref_dict,
                tumor_reads = primary_bam,
                tumor_reads_index = primary_bai,
                normal_reads = control_bam_hack,
                normal_reads_index = control_bai_hack,

                scatter_count = scatter_count,
                gnomad = gnomad_hack,
                gnomad_idx = gnomad_idx_hack,
                variants_for_contamination = variants_for_contamination_hack,
                variants_for_contamination_idx = variants_for_contamination_idx_hack,
                skip_filtering = skip_m2_filtering,
                realignment_index_bundle = realignment_index_bundle_hack,
                realignment_extra_args = realignment_extra_args,
                dragstr_model = dragstr_model_hack,
                run_orientation_bias_mixture_model_filter = run_orientation_bias_mixture_model_filter,
                m2_extra_args = m2_extra_args,
                make_bamout = false,

                gatk_docker = permutect_docker,
                gcs_project_for_requester_pays = gcs_project_for_requester_pays_hack,
                gatk_override = gatk_override,
                preemptible = preemptible,
                max_retries = max_retries
        }
    }

    call SplitMultiallelics {
        input:
            input_vcf = select_first([Mutect2.output_vcf, cached_mutect2_vcf_hack]),
            input_vcf_idx = select_first([Mutect2.output_vcf_idx, cached_mutect2_vcf_idx_hack]),
            ref_fasta = ref_fasta,
            ref_fai = ref_fai,
            ref_dict = ref_dict,
            gatk_docker = permutect_docker
    }

    if (filter_pcr_errors) {
        call PrePermutectFiltering {
            input:
                input_vcf = SplitMultiallelics.output_vcf,
                ref_fasta = ref_fasta,
                ref_fai = ref_fai,
                permutect_docker = permutect_docker
        }
    }

    call PermutectFiltering {
        input:
            mutect2_vcf = select_first([PrePermutectFiltering.output_vcf, SplitMultiallelics.output_vcf]),
            mutect2_vcf_idx = select_first([PrePermutectFiltering.output_vcf_idx, SplitMultiallelics.output_vcf_idx]),
            artifact_model = artifact_model,
            test_dataset = select_first([Mutect2.permutect_test_dataset, cached_dataset_hack]),
            contigs_table = select_first([Mutect2.permutect_contigs_table, cached_contigs_table_hack]),
            maf_segments = select_first([Mutect2.maf_segments, cached_maf_segments_hack]),
            normal_maf_segments = select_first([Mutect2.normal_maf_segments, cached_normal_maf_segments_hack]),
            mutect_stats = select_first([Mutect2.mutect_stats, cached_mutect_stats_hack]),
            batch_size = batch_size,
            num_workers = num_workers,
            gpu_count = gpu_count,
            num_spectrum_iterations = num_spectrum_iterations,
            spectrum_learning_rate = spectrum_learning_rate,
            permutect_filtering_extra_args = permutect_filtering_extra_args,
            permutect_docker = permutect_docker,
    }

    if (defined(test_dataset_truth_vcf_hack)){
        call Concordance as PermutectConcordance {
            input:
                intervals = intervals_hack,
                masks = masks_hack,
                truth_vcf = select_first([test_dataset_truth_vcf_hack]),
                truth_vcf_idx = select_first([test_dataset_truth_vcf_idx_hack]),
                eval_vcf = PermutectFiltering.output_vcf,
                eval_vcf_idx = PermutectFiltering.output_vcf_idx,
                gatk_docker = permutect_docker,
                concordance_header = concordance_header,
        }

        call Concordance as M2Concordance {
            input:
                intervals = intervals_hack,
                masks = masks_hack,
                truth_vcf = select_first([test_dataset_truth_vcf_hack]),
                truth_vcf_idx = select_first([test_dataset_truth_vcf_idx_hack]),
                eval_vcf = select_first([Mutect2.output_vcf, cached_mutect2_vcf_hack]),
                eval_vcf_idx = select_first([Mutect2.output_vcf_idx, cached_mutect2_vcf_idx_hack]),
                gatk_docker = permutect_docker,
                concordance_header = concordance_header
        }
    }

    output {
        File output_vcf = PermutectFiltering.output_vcf
        File output_vcf_idx = PermutectFiltering.output_vcf_idx
        File tensorboard_report = PermutectFiltering.tensorboard_report
        String test_dataset = select_first([Mutect2.permutect_test_dataset, cached_dataset_hack])
        String mutect2_vcf = select_first([Mutect2.output_vcf, cached_mutect2_vcf_hack])
        String mutect2_vcf_idx = select_first([Mutect2.output_vcf_idx, cached_mutect2_vcf_idx_hack])
        String contigs_table = select_first([Mutect2.permutect_contigs_table, cached_contigs_table_hack])
        String maf_segments = select_first([Mutect2.maf_segments, cached_maf_segments_hack])
        String normal_maf_segments = select_first([Mutect2.normal_maf_segments, cached_normal_maf_segments_hack])
        String mutect_stats = select_first([Mutect2.mutect_stats, cached_mutect_stats_hack])

        File? fn = PermutectConcordance.fn
        File? fn_idx = PermutectConcordance.fn_idx
        File? fp = PermutectConcordance.fp
        File? fp_idx = PermutectConcordance.fp_idx
        File? tp = PermutectConcordance.tp
        File? tp_idx = PermutectConcordance.tp_idx
        File? ffn = PermutectConcordance.ffn
        File? ffn_idx = PermutectConcordance.ffn_idx
        File? ftn = PermutectConcordance.ftn
        File? ftn_idx = PermutectConcordance.ftn_idx
        File? concordance_summary = PermutectConcordance.summary
        File? filter_analysis = PermutectConcordance.filter_analysis

        File? m2_concordance_summary = M2Concordance.summary
    }
}

 task PermutectFiltering {
    input {
        File artifact_model
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
            --artifact_model ~{artifact_model} \
            --contigs_table ~{contigs_table} \
            --output permutect-filtered.vcf \
            --tensorboard_dir tensorboard \
            --batch_size ~{batch_size} --num_workers ~{num_workers} \
            ~{" --num_spectrum_iterations " + num_spectrum_iterations} \
            ~{" --spectrum_learning_rate " + spectrum_learning_rate} \
            ~{" --maf_segments " + maf_segments} ~{" --normal_maf_segments " + normal_maf_segments} \
            --genomic_span $genomic_span ~{permutect_filtering_extra_args}

        tar cvf tensorboard.tar tensorboard/

        # compress
        bcftools view permutect-filtered.vcf -Oz -o permutect-filtered.vcf.gz

        bcftools index -t permutect-filtered.vcf.gz
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
        File output_vcf = "permutect-filtered.vcf.gz"
        File output_vcf_idx = "permutect-filtered.vcf.gz.tbi"
        File tensorboard_report = "tensorboard.tar"
    }
}

task PrePermutectFiltering {
    input {
        File input_vcf
        File ref_fasta
        File ref_fai
        String permutect_docker
        Int? preemptible
        Int max_retries = 0
        Int disk_space = 100
        Int cpu = 1
        Int mem = 4
    }

    # Mem is in units of GB but our command and memory runtime values are in MB
    Int machine_mem = mem * 1000
    Int command_mem = machine_mem - 500

    command <<<
        set -e

        pre_permutect_filtering --input ~{input_vcf} --reference ~{ref_fasta} --output pre-filtered.vcf

        gatk --java-options "-Xmx~{command_mem}m" IndexFeatureFile -I pre-filtered.vcf
    >>>

    runtime {
        docker: permutect_docker
        bootDiskSizeGb: 12
        memory: machine_mem + " MB"
        disks: "local-disk " + disk_space + " SSD"
        preemptible: select_first([preemptible, 0])
        maxRetries: max_retries
        cpu: cpu
    }

    output {
        File output_vcf = "pre-filtered.vcf"
        File output_vcf_idx = "pre-filtered.vcf.idx"
    }
}

task SplitMultiallelics {
    input {
        File input_vcf
        File input_vcf_idx
        File ref_fasta
        File ref_fai
        File ref_dict
        String gatk_docker
        Int? preemptible
        Int max_retries = 0
        Int disk_space = 100
        Int cpu = 1
        Int mem = 4
    }

    # Mem is in units of GB but our command and memory runtime values are in MB
    Int machine_mem = mem * 1000
    Int command_mem = machine_mem - 500

    command <<<

        bcftools norm -m -any -f ~{ref_fasta} ~{input_vcf} > split.vcf

        gatk --java-options "-Xmx~{command_mem}m" IndexFeatureFile -I split.vcf

        gatk --java-options "-Xmx~{command_mem}m" SelectVariants -V split.vcf -O output.vcf --lenient \
            -DGA DP -DGA AF -DGA F1R2 -DGA F2R1 -DGA FAD -DGA SB \
            -DA AS_FilterStatus -DA AS_SB_TABLE -DA ECNT -DA GERMQ -DA MBQ -DA MFRL -DA MMQ -DA MPOS

        set -e

    >>>

    runtime {
        docker: gatk_docker
        bootDiskSizeGb: 12
        memory: machine_mem + " MB"
        disks: "local-disk " + disk_space + " SSD"
        preemptible: select_first([preemptible, 0])
        maxRetries: max_retries
        cpu: cpu
    }

    output {
        File output_vcf = "output.vcf"
        File output_vcf_idx = "output.vcf.idx"
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
        String? concordance_header

    	# runtime
    	String gatk_docker = "us.gcr.io/broad-gatk/gatk"
    	Int preemptible = 0
	}

    command {
        # this shouldn't be necessary but the Terra hacks on top of Terra hacks to deal with all the bugs recently
        # have reached a breaking point
        gatk IndexFeatureFile -I ~{eval_vcf}

        gatk --java-options "-Xmx2g" Concordance \
            ~{"-L " + intervals} \
            ~{"-XL " + masks} \
            -truth ~{truth_vcf} -eval ~{eval_vcf} \
            -tpfn "tpfn.vcf" \
            -tpfp "tpfp.vcf" \
            -ftnfn "ftnfn.vcf" \
            -filter-analysis "filter-analysis.txt" \
            -summary "summary_no_header.txt"

        echo ~{concordance_header} > TMP
        cat TMP summary_no_header.txt > summary.txt

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