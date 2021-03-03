### Mutect3 Prototype

## Train and Test Data

# DREAM Challenge Data
Note: while the DREAM challenge labeled truth is synthetic, Mutect3 learns truth somatic variants by downsampling germline hets and learns artifacts from actual artifacts, neither of which have anything to do with the synthetic variants and neither of which require labels.  The synthetic labels are only used in testing.

For each DREAM challenge 1-4, we have pickled tensors for the following datasets: test data (labels taken from the STATUS field of validation against the DREAM truth VCF), training data derived only from the tumor sample, training data derived only from the normal sample, and training data derived from the tumor but using the normal for better decisions about artifacts for heuristic weak labeling.  These pickled files can be passed directly to the Mutect3Dataset constructor.

The pickled files are located in the Google bucket gs://broad-dsde-methods-davidben/mutect3/dream
