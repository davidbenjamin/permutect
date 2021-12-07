#!/bin/bash

filter_variants \
    --input vcfs/dream1.vcf \
    --trained_m3_model saved/dream1-saved.pt \
    --tumor "synthetic.challenge.set1.tumor" \
    --normal "synthetic.challenge.set1.normal" \
    --batch_size 64 \
    --output "dream1-filtered-by-m3.vcf" \
    --report_pdf dream1-testing-report.pdf
