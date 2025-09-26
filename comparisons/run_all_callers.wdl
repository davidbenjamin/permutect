version 1.0

import "https://api.firecloud.org/ga4gh/v1/tools/davidben:call-variants-with-permutect/versions/2/plain-WDL/descriptor" as PermutectWorkflow
import "https://api.firecloud.org/ga4gh/v1/tools/davidben:strelka/versions/10/plain-WDL/descriptor" as StrelkaWorkflow
import "https://api.firecloud.org/ga4gh/v1/tools/davidben:deepsomatic/versions/1/plain-WDL/descriptor" as DeepSomaticWorkflow



workflow RunAllCallers {
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
        Boolean skip_m2_filtering = false
        Boolean run_orientation_bias_mixture_model_filter = false
        String? m2_extra_args
        String strelka_extra_args = ""
        String manta_extra_args = ""
        String deepsomatic_extra_args = ""

        # Can be WGS,WES,PACBIO,ONT,FFPE_WGS,FFPE_WES,WGS_TUMOR_ONLY,PACBIO_TUMOR_ONLY,ONT_TUMOR_ONLY
        String deepsomatic_model_type
        String? split_intervals_extra_args
        Int batch_size
        Int num_workers
        Int? gpu_count
        Int chunk_size
        File? test_dataset_truth_vcf    # used for evaluation
        File? test_dataset_truth_vcf_idx

        String? permutect_filtering_extra_args
        String gatk_docker = "us.gcr.io/broad-gatk/gatk"
        String strelka_docker = "us.gcr.io/broad-dsde-methods/davidben/strelka"
        String deepsomatic_docker = "us.gcr.io/broad-dsde-methods/davidben/deepsomatic-gpu"

        String? gcs_project_for_requester_pays
        File? gatk_override
        String permutect_docker
        Int? preemptible
        Int? max_retries
    }

    call PermutectWorkflow.Permutect {
        input:
            artifact_model = artifact_model,
            intervals = intervals,
            masks = masks,
            ref_fasta = ref_fasta,
            ref_fai = ref_fai,
            ref_dict = ref_dict,
            scatter_count = scatter_count,
            num_spectrum_iterations = num_spectrum_iterations,
            spectrum_learning_rate = spectrum_learning_rate,
            primary_bam = primary_bam,
            primary_bai = primary_bai,
            control_bam = control_bam,
            control_bai = control_bai,
            gnomad = gnomad,
            gnomad_idx = gnomad_idx,
            variants_for_contamination = variants_for_contamination,
            variants_for_contamination_idx = variants_for_contamination_idx,
            realignment_index_bundle = realignment_index_bundle,
            dragstr_model = dragstr_model,
            realignment_extra_args = scatter_count,
            skip_m2_filtering = true,
            run_orientation_bias_mixture_model_filter = false,
            m2_extra_args = m2_extra_args,
            split_intervals_extra_args = split_intervals_extra_args,
            batch_size = batch_size,
            num_workers = num_workers,
            gpu_count = gpu_count,
            chunk_size = chunk_size,
            test_dataset_truth_vcf = test_dataset_truth_vcf,
            test_dataset_truth_vcf_idx = test_dataset_truth_vcf_idx,

            permutect_filtering_extra_args = permutect_filtering_extra_args,
            gatk_docker = gatk_docker,
            gcs_project_for_requester_pays = gcs_project_for_requester_pays,
            gatk_override = gatk_override,
            permutect_docker = permutect_docker,
            preemptible = preemptible,
            max_retries = max_retries
    }

    call StrelkaWorkflow.Strelka {
        input:
            ref_fasta = ref_fasta,
            ref_fai = ref_fai,
            ref_dict= ref_dict,
            tumor_bam = primary_bam,
            tumor_bai = primary_bai,
            normal_bam = control_bam,
            normal_bai = control_bai,
            intervals = intervals,
            masks = masks,
            manta_extra_args = manta_extra_args,
            strelka_extra_args = strelka_extra_args,
            truth_vcf = test_dataset_truth_vcf,
            truth_vcf_idx = test_dataset_truth_vcf_idx,
            gatk_docker = gatk_docker,
            strelka_docker = strelka_docker,
            gcs_project_for_requester_pays = gcs_project_for_requester_pays
    }

    call DeepSomaticWorkflow.DeepSomatic {
        input {
        File ref_fasta
        File ref_fai
        File ref_dict

        File tumor_bam
        File tumor_bai
        File normal_bam
        File normal_bai


        model_type = deepsomatic_model_type
        intervals = intervals,
        masks = masks,
        deepsomatic_extra_args = deepsomatic_extra_args,
        truth_vcf = test_dataset_truth_vcf,
        truth_vcf_idx = test_dataset_truth_vcf_idx,

        Strgatk_docker = gatk_docker
        String deepsomatic_docker = "us.gcr.io/broad-dsde-methods/davidben/deepsomatic-gpu"
        String? gcs_project_for_requester_pays
    }


}