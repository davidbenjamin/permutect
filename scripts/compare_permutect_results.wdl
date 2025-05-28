version 1.0

workflow ComparePermutect {
    input {
		File? intervals
		File? masks
    	File permutect_vcf
    	File permutect_vcf_idx
        File other_vcf
    	File other_vcf_idx
    	File truth_vcf
    	File truth_vcf_idx

    	# runtime
    	String gatk_docker
    	File? gatk_override

        File? obscene_hack_leave_unset
    }

    call Concordance as PermutectConcordance {
        input:
            intervals = intervals,
            masks = if (defined(masks) && masks == "") then obscene_hack_leave_unset else masks,
            truth_vcf = truth_vcf,
            truth_vcf_idx = truth_vcf_idx,
            eval_vcf = permutect_vcf,
            eval_vcf_idx = permutect_vcf_idx,
            gatk_override = gatk_override,
            gatk_docker = gatk_docker
    }

    call Concordance as OtherConcordance {
        input:
            intervals = intervals,
            masks = if (defined(masks) && masks == "") then obscene_hack_leave_unset else masks,
            truth_vcf = truth_vcf,
            truth_vcf_idx = truth_vcf_idx,
            eval_vcf = other_vcf,
            eval_vcf_idx = other_vcf_idx,
            gatk_override = gatk_override,
            gatk_docker = gatk_docker
    }

    call Compare {
        input:
            permutect_ffn = PermutectConcordance.ffn,
            permutect_ffn_idx = PermutectConcordance.ffn,
            other_ffn = OtherConcordance.ffn,
            other_ffn_idx = OtherConcordance.ffn_idx,

            permutect_fp = PermutectConcordance.fp,
            permutect_fp_idx = PermutectConcordance.fp,
            other_fp = OtherConcordance.fp,
            other_fp_idx = OtherConcordance.fp_idx,

            permutect_summary = PermutectConcordance.summary,
            other_summary = OtherConcordance.summary,

            gatk_override = gatk_override,
            gatk_docker = gatk_docker
    }


    output {
        File ffn_created_by_permutect = Compare.ffn_created_by_permutect
        File ffn_created_by_permutect_idx = Compare.ffn_created_by_permutect_idx
        File ffn_fixed_by_permutect = Compare.ffn_fixed_by_permutect
        File ffn_fixed_by_permutect_idx = Compare.ffn_fixed_by_permutect_idx

        File fp_created_by_permutect = Compare.fp_created_by_permutect
        File fp_created_by_permutect_idx = Compare.fp_created_by_permutect_idx
        File fp_fixed_by_permutect = Compare.fp_fixed_by_permutect
        File fp_fixed_by_permutect_idx = Compare.fp_fixed_by_permutect_idx

        File summary = Compare.summary
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

    	File? gatk_override

    	# runtime
    	String gatk_docker
    	Int preemptible = 2
	}

    command <<<
        export GATK_LOCAL_JAR=${default="/root/gatk.jar" gatk_override}

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
    >>>

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

task Compare {
    input {
    	File permutect_ffn
        File permutect_ffn_idx
        File other_ffn
        File other_ffn_idx

        File permutect_fp
        File permutect_fp_idx
        File other_fp
        File other_fp_idx

        File permutect_summary
        File other_summary

    	File? gatk_override

    	# runtime
    	String gatk_docker
    	Int preemptible = 2
	}

    command <<<
        export GATK_LOCAL_JAR=${default="/root/gatk.jar" gatk_override}

        gatk --java-options "-Xmx2g" SelectVariants \
            -V ~{permutect_ffn} -disc ~{other_ffn} -O ffn_created_by_permutect.vcf

        gatk --java-options "-Xmx2g" SelectVariants \
            -V ~{other_ffn} -disc ~{permutect_ffn} -O ffn_fixed_by_permutect.vcf

        gatk --java-options "-Xmx2g" SelectVariants \
            -V ~{permutect_fp} -disc ~{other_fp} -O fp_created_by_permutect.vcf

        gatk --java-options "-Xmx2g" SelectVariants \
            -V ~{other_fp} -disc ~{permutect_fp} -O fp_fixed_by_permutect.vcf

        touch summary.txt
        echo "PERMUTECT SUMMARY" >> summary.txt
        cat ~{permutect_summary} >> summary.txt
        printf "\n" >> summary.txt
        echo "OTHER TOOL SUMMARY" >> summary.txt
        cat ~{other_summary} >> summary.txt
        printf "\n" >> summary.txt

        echo "Permutect FFN count:"
        grep -v '#' ~{permutect_ffn} | wc -l >> summary.txt

        echo "Other tool FFN count:"
        grep -v '#' ~{other_ffn} | wc -l >> summary.txt

        echo "Permutect FP count:"
        grep -v '#' ~{permutect_fp} | wc -l >> summary.txt

        echo "Other tool FP count:"
        grep -v '#' ~{other_fp} | wc -l >> summary.txt

        echo "Permutect created FFN count:"
        grep -v '#' ffn_created_by_permutect.vcf | wc -l >> summary.txt

        echo "Permutect fixed FFN count:"
        grep -v '#' ffn_fixed_by_permutect.vcf | wc -l >> summary.txt

        echo "Permutect created FP count:"
        grep -v '#' fp_created_by_permutect.vcf | wc -l >> summary.txt

        echo "Permutect fixed FP count:"
        grep -v '#' fp_fixed_by_permutect.vcf | wc -l >> summary.txt
    >>>

    runtime {
        memory: "5 GB"
        bootDiskSizeGb: 12
        docker: "${gatk_docker}"
        disks: "local-disk " + 100 + " HDD"
        preemptible: select_first([preemptible, 2])
    }

    output {
        File ffn_created_by_permutect = "ffn_created_by_permutect.vcf"
        File ffn_created_by_permutect_idx = "ffn_created_by_permutect.vcf.idx"
        File ffn_fixed_by_permutect = "ffn_fixed_by_permutect.vcf"
        File ffn_fixed_by_permutect_idx = "ffn_fixed_by_permutect.vcf.idx"

        File fp_created_by_permutect = "fp_created_by_permutect.vcf"
        File fp_created_by_permutect_idx = "fp_created_by_permutect.vcf.idx"
        File fp_fixed_by_permutect = "fp_fixed_by_permutect.vcf"
        File fp_fixed_by_permutect_idx = "fp_fixed_by_permutect.vcf.idx"

        File summary = "summary.txt"
    }
}