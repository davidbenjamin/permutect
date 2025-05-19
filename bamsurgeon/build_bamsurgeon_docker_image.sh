#!/bin/bash
# run this script in the same directory as the Dockerfile
docker build -f ./bamsurgeon.Dockerfile -t us.gcr.io/broad-dsde-methods/davidben/bam_surgeon .