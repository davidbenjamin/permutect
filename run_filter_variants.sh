#!/bin/bash

filter_variants \
    --input vcfs/dream1-50000.vcf \
    --trained_m3_model saved/dream1-experiment-saved.pt \
    --tumor "synthetic.challenge.set1.tumor" \
    --normal "synthetic.challenge.set1.normal" \
    --output "dream1-filtered-by-m3.vcf" \
    --report_pdf dream1-experiment-testing-report.pdf
