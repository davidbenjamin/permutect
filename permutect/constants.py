# keys for saving and loading models etc
STATE_DICT_NAME = "model_state_dict"
OPTIMIZER_STATE_DICT_NAME = "optimizer_state_dict"
ARTIFACT_LOG_PRIORS_NAME = "artifact_log_priors"
ARTIFACT_SPECTRA_STATE_DICT_NAME = "artifact_spectra_state_dict"
HYPERPARAMS_NAME = "hyperparams"
NUM_READ_FEATURES_NAME = "num_read_features"
NUM_INFO_FEATURES_NAME = "num_info_features"
REF_SEQUENCE_LENGTH_NAME = "ref_sequence_length"
HIDDEN_LAYERS_NAME = "hidden_layers"

SOURCES_NAME = "sources"
SOURCE_NAME = "source"

# basic input and output files
INPUT_NAME = "input"
OUTPUT_NAME = "output"
TRAINING_DATASETS_NAME = "training_datasets"
TRAIN_TAR_NAME = "train_tar"
TEST_DATASET_NAME = "test_dataset"
TENSORBOARD_DIR_NAME = "tensorboard_dir"
PRETRAINED_MODEL_NAME = "pretrained_artifact_model"
ARTIFACT_MODEL_NAME = "artifact_model"

# artifact model hyperparameters
READ_LAYERS_NAME = "read_layers"
INFO_LAYERS_NAME = "info_layers"
REF_SEQ_LAYER_STRINGS_NAME = "ref_seq_layer_strings"
AGGREGATION_LAYERS_NAME = "aggregation_layers"
SELF_ATTENTION_HIDDEN_DIMENSION_NAME = "self_attention_hidden_dimension"
NUM_SELF_ATTENTION_LAYERS_NAME = "num_self_attention_layers"
NUM_ARTIFACT_CLUSTERS_NAME = "num_artifact_clusters"
DROPOUT_P_NAME = "dropout_p"
WEIGHT_DECAY_NAME = "weight_decay"
BATCH_NORMALIZE_NAME = "batch_normalize"

# training parameters
TRAINABLE_PARAMETERS_NAME = "trainable_parameters"
LEARNING_RATE_NAME = "learning_rate"
BATCH_SIZE_NAME = "batch_size"
NUM_EPOCHS_NAME = "num_epochs"
NUM_WORKERS_NAME = "num_workers"
NUM_SPECTRUM_ITERATIONS_NAME = "num_spectrum_iterations"
SPECTRUM_LEARNING_RATE_NAME = "spectrum_learning_rate"

DATASET_EDIT_TYPE_NAME = "dataset_edit"

# variant filtering parameters
INITIAL_LOG_VARIANT_PRIOR_NAME = "initial_log_variant_prior"
INITIAL_LOG_ARTIFACT_PRIOR_NAME = "initial_log_artifact_prior"
CONTIGS_TABLE_NAME = "contigs_table"
GENOMIC_SPAN_NAME = "genomic_span"
MAF_SEGMENTS_NAME = "maf_segments"
NORMAL_MAF_SEGMENTS_NAME = "normal_maf_segments"
GERMLINE_MODE_NAME = "germline_mode"
NO_GERMLINE_MODE_NAME = "no_germline_mode"
RECALL_WEIGHT_NAME = "recall_weight"
HET_BETA_NAME = "het_beta"
