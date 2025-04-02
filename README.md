# Overview of Permutect
Permutect is a pipeline for calling and filtering variants from genomic sequencing data.  It takes a BAM or CRAM file (and, optionally, a matched normal sample BAM or CRAM) and outputs a filtered VCF.  Roughly speaking it comprises two parts: 1) permissive, unfiltered variant calling with Mutect2 and 2) a new deep learning-based filtering model.

The greatest advantage of Permutect over the Mutect2/FilterMutectCalls pipeline is that the latter performs poorly without a "panel of normals", which is a blacklist VCF of common errors compiled from a very large number of samples.  Since the typical user does not have sufficient data (100+ WGS samples) the best practice is to use one of the Broad Institute's publicly-available panels and hope for the best.  While this usually works for short read Illumina sequencing it fails for novel sequencing platforms and library preparation, non-human data, highly-variable parts of the genome etc.  In contrast, the Permutect model is so efficient that it can be trained on a single WGS sample with no labeled truth data without overfitting.

In addition to working on a variety of sequencing platforms Permutect is designed for both somatic and germline variant calling.  Crucially, a trained Permutect model is specific to a sequencing technology but not to any particular biological scenario.  Thus it is possible (in fact, recommended) to train the model on a germline sample and then deploy it for somatic calling or a different species.  This works due to a division of responsibility between a black box deep learning model that learns technical error modes and a probabilistic model that learns biological features such as somatic mutation rates and subclonal cell fractions.

Unlike the deep learning artifact model, the probabilistic model of biology does not have its own workflow below.  This is because it is sample-specific and rather than being trained ahead of time its parameters are learned during variant calling.

We strongly recommend running the pipeline from the [Permutect Terra workspace](https://app.terra.bio/#workspaces/broad-firecloud-dsde/Permutect).

# Pipeline
Here we described the workflows in the Terra workspace, most of which correspond to a WDL script in the /scripts directory of this repository and a command line tool in the /tools directory.

## scripts/make_training_dataset.wdl
This runs the Mutect2 WDL as a subworkflow, invoking a mode that generates a Permutect training data file as a side effect.  Most of this workflow's inputs are straightforward things like a BAM/CRAM file, a GATK docker, reference files, a genomic intervals file, and a gnomAD VCF.  (We reiterate that requiring gnomAD for training does not restrict variant calling to human data; nor does the reference used for generating training data need to match the reference used for variant calling).  Optionally, one may give a training truth VCF of known germline variants for the BAM/CRAM sample.  Without this, Permutect generates weak truth labels based on the observation that artifacts generally occur at lower allele fractions than germline variants.  This weak labeling scheme works well, but if possible it is best to train on one of the NIST Genome in a Bottle samples.

After generating Permutect training data in a plain text format, the workflow calls tools/preprocess_dataset.py to create a binary .tar.gz file for use in the model training workflow.

## scripts/train_artifact_model.wdl
This workflow takes the .tar.gz output from the training dataset workflow and produces a Permutect artifact model in PyTorch .pt format.  Most of the configuration comprises hyperparameters of the model architecture and we recommend copying these from the Terra workspace.  Feel free to contact David Benjamin (davidben@broadinstitute.org) regarding details.

## scripts/permutect.wdl, GATK Mutect2 and tools/filter_variants.py)
This workflow runs Mutect2 in a mode that generates both an unfiltered VCF and a plain text Permutect dataset.  The data format is the same as the training dataset workflow but the dataset includes all unfiltered variant calls.  It then uses the artifact model from the training workflow to compute artifact likelihoods, fits the sample-specific biology model, and generates a filtered VCF.

# Building the Permutect Docker image
First of all, I ought to say right now that all this engineering stuff is total voodoo to me copied from other people's example.  I have done my best to make it professional but I really have no idea what I'm doing.  Here is an overview of all the magical files in this repo that somehow work to make a docker image.

## requirements.txt
This is a list of required python packages along with the minimum and maximum version numbers that will work.  Some of these version constraints are frighteningly precise and frankly this file terrifies me.  I gave up on the memory-map-ninja package and so I have actually simply copied the code from their repo into this one rather than importing it as a dependency.  A better software engineer would not do this.

## setup.py
If you want to build Permutect locally, instead of within a Docker image, you must run the terminal command "pip install --no-cache-dir -r /requirements.txt".  I believe that setup.py tells pip where to look for command line programs and what names to give them.  Thus, for example, the main() method of permutect.tools.train_model becomes the command line program train_model.

## Dockerfile
Most of the commands here use requirements.txt and setup.py to install Permutect in a Docker image with the correct dependencies.  Like requirements.txt, I am afraid of this file.  The command on my laptop for building a Docker image and pushing it to the cloud is "tag='optional_tag'; docker build -t us.gcr.io/broad-dsde-methods/davidben/permutect:${tag} .; docker push us.gcr.io/broad-dsde-methods/davidben/permutect:${tag}".  This will not work for you because you don't have access to us.gcr.io/broad-dsde-methods/davidben/permutect, but I hope that you are more competent than me and understand this stuff.

In the future the Broad Institute will host a pre-built Permutect Docker image so that only advanced users who clone this repo and modify the code will ever have to build one.
