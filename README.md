# Overview of Permutect
Permutect is a pipeline for calling and filtering variants from genomic sequencing data.  It takes a BAM or CRAM file (and, optionally, a matched normal sample BAM or CRAM) and outputs a filtered VCF.  Roughly speaking it comprises two parts: 1) permissive, unfiltered variant calling with Mutect2 and 2) a new deep learning-based filtering model.

Most of the command line tools and WDL scripts in this repository pertain to training the filtering model.  In the long run we hope that most users will not need to train their own models and will instead be able to rely on pre-trained models to be supplied by the Broad Institute.  Currently, however, users are responsible for running the entire pipeline.

The greatest advantage of Permutect over the Mutect2/FilterMutectCalls pipeline is that the latter performs poorly without a "panel of normals", which is a large blacklist VCF of common errors compiled from a very large number of samples.  Since the typical user does not have sufficient data (100+ WGS samples) the best practice is to use one of the Broad Institute's publicly-available panels and hope for the best.  While this usually works for short reads Illumina sequencing it fails for novel sequencing platforms and library preparation, non-human data, highly-variable parts of the genome etc.  In contrast, the Permutect model is so efficient that it can be trained on a single WGS sample with no labeled truth data without overfitting.

In addition to working on a variety of sequencing platforms Permutect is designed for both somatic and germline variant calling.  Crucially, a trained Permutect model is specific to a sequencing technology but not to any particular biological scenario.  Thus it is possible (in fact, recommended) to train the model on a germline sample and then deploy it for somatic calling or a different species.  This works due to a division of responsibility between a black box deep learning model that learns technical error modes and a probabilistic model that learns biological features such as somatic mutation rates and subclonal cell fractions.

We strongly recommend running the pipeline from the [Permutect Terra workspace](https://app.terra.bio/#workspaces/broad-firecloud-dsde/Permutect).

# Pipeline
Here we described the workflows in the Terra workspace, most of which correspond to a WDL script in the /scripts directory of this repository and a command line tool in the /tools directory.

## permutect-training-data (not in this repo)
Unlike the workflows below, this does not correspond to a WDL here.  Rather, it runs the Mutect2 WDL in a special mode that emits a Permutect training data file as a side effect.  Most of this workflow's inputs are straightforward things like a BAM/CRAM file, a GATK docker, reference files, a genomic intervals file, and a gnomAD VCF.  (We reiterate that requiring gnomAD for training does not restrict variant calling to human data; nor does the reference used for generating training data need to match the reference used for variant calling).  Optionally, one may give a training truth VCF of known germline variants for the BAM/CRAM sample.  Without this, Permutect generates weak truth labels based on the observation that artifacts generally occur at lower allele fractions than germline variants.  This weak labeling scheme works very well, but if possible it is helpful to train on one of the NIST Genome in a Bottle samples.

## permutect-preprocess-training-data (scripts/permutect_preprocessing.wdl, tools/preprocess_dataset.py)
This takes one or more training data files from the permutect-training-data workflow, which are in a plain-text format, and combines them into a single binary .tar.gz file for later workflows.  Note that training a base model and training an artifact model both use the output of this workflow.

## permutect-train-base-model (scripts/permutect_train_base_model.wdl, tools/train_base_model.py)
The Permutect deep learning model is split into two parts, 1) a "base model" with relatively many parameters containing the bottom layers of the model that converts a set of reads and other input features into a single vector representation and 2) an artifact model that uses this representation vector to compute a log likelihood ratio that an apparent variant is actually an artifact.  This script takes the .tar.gz training data file from the permutect-preprocess-training-data workflow and produces a base model in PyTorch .pt format.  Most of the configuration comprises hyperparameters of the model architecture and we recommend copying these from the Terra workspace.  Feel free to contact David Benjamin (davidben@broadinstitute.org) regarding details.

## permutect-prune-training-dataset (scripts/permutect_pruning.wdl, tools/prune_dataset.py) (optional)
Especially when permutect-training-data is run without a training truth VCF for labels, it is helpful to prune mislabeled data before training the artifact model.  This workflow takes the .tar.gz training data from the permutect-preprocess-training-data workflow and a Permutect base model and produces a similar file in the same format with errors removed.  If the data are already clean this step has little effect, so there is no risk of over-pruning the data.  In principle one could re-train the base model after pruning but this is not necessary.  The best practice is to train a base model on unpruned data, prune the data, then train an artifact model.

## permutect-train-artifact-model (scripts/permutect_training.wdl, tools/train_model.py)
This workflow takes in a base model in .py format from the permutect-train-base-model workflow and a pruned training dataset from the permutect-prune-training-dataset workflow and produces an artifact model in Pytorch .pt format.

## permutect-call-variants (scripts/permutect.wdl, GATK Mutect2 and tools/filter_variants.py)
This workflow runs Mutect2 in a mode that generates both an unfiltered VCF and a plain text Permutect dataset.  The data format is the same as the permutect-training-data workflow but the dataset includes all unfiltered variant calls.  It then uses the base model and artifact model from previous steps to compute artifact likelihoods and combines these with a probabilistic model for biological features to obtain filtered variant calls.

# Building the Permutect Docker image
First of all, I ought to say right now that all this engineering stuff is total voodoo to me copied from other people's example.  I have done my best to make it professional but I really have no idea what I'm doing.  Here is an overview of all the magical files in this repo that somehow work to make a docker image.

## requirements.txt
This is a list of required python packages along with the minimum and maximum version numbers that will work.  Some of these version constraints are frighteningly precise and frankly this file terrifies me.  I gave up on the memory-map-ninja package and so I have actually simply copied the code from their repo into this one rather than importing it as a dependency.  A better software engineer would not do this.

## setup.py
If you want to build Permutect locally, instead of within a Docker image, you must run the terminal command "pip install --no-cache-dir -r /requirements.txt".  I believe that setup.py tells pip where to look for command line programs and what names to give them.  Thus, for example, the main() method of permutect.tools.train_model becomes the command line program train_model.

## Dockerfile
Most of the commands here use requirements.txt and setup.py to install Permutect in a Docker image with the correct dependencies.  Like requirements.txt, I am afraid of this file.  The command on my laptop for building a Docker image and pushing it to the cloud is "tag='optional_tag'; docker build -t us.gcr.io/broad-dsde-methods/davidben/permutect:${tag} .; docker push us.gcr.io/broad-dsde-methods/davidben/permutect:${tag}".  This will not work for you because you don't have access to us.gcr.io/broad-dsde-methods/davidben/permutect, but I hope that you are more competent than me and understand this stuff.

In the near future the Broad Institute will host a pre-built Permutect Docker image so that only the most advanced power users who clone this repo and modify the code will ever have to build one.





