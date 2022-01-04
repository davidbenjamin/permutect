#!/bin/bash
# first make sure docker is running on this computer by launching the docker app, then run this script

gcloud auth login
gcloud auth configure-docker

sudo docker build -t us.gcr.io/broad-dsde-methods/davidben/mutect3 .
docker push us.gcr.io/broad-dsde-methods/davidben/mutect3
