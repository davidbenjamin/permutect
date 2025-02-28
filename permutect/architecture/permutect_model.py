from itertools import chain

import torch
from torch import nn, Tensor, IntTensor
from torch.utils.tensorboard import SummaryWriter
from tqdm.autonotebook import tqdm

from permutect import constants
from permutect.architecture.adversarial import Adversarial
from permutect.architecture.balancer import Balancer
from permutect.architecture.calibration import Calibration
from permutect.architecture.dna_sequence_convolution import DNASequenceConvolution
from permutect.architecture.gated_mlp import GatedRefAltMLP
from permutect.architecture.mlp import MLP
from permutect.architecture.set_pooling import SetPooling
from permutect.data.batch_indexing import BatchIndices
from permutect.data.datum import DEFAULT_GPU_FLOAT, DEFAULT_CPU_FLOAT
from permutect.data.reads_batch import ReadsBatch
from permutect.data.prefetch_generator import prefetch_generator
from permutect.metrics.evaluation_metrics import EmbeddingMetrics
from permutect.data.count_binning import alt_count_bin_index, alt_count_bin_name, MAX_ALT_COUNT
from permutect.parameters import ModelParameters
from permutect.sets.ragged_sets import RaggedSets
from permutect.misc_utils import unfreeze, freeze, gpu_if_available
from permutect.utils.enums import Variation, Epoch


class BatchOutput:
    """
    simple container class for the output of the model over a single batch
    :return:
    """
    def __init__(self, features: Tensor, uncalibrated_logits: Tensor, calibrated_logits: Tensor, weights: Tensor, source_weights: Tensor):
        self.features = features
        self.uncalibrated_logits = uncalibrated_logits
        self.calibrated_logits = calibrated_logits
        self.weights = weights
        self.source_weights = source_weights


def sums_over_chunks(tensor2d: Tensor, chunk_size: int):
    assert len(tensor2d) % chunk_size == 0
    return torch.sum(tensor2d.reshape([len(tensor2d) // chunk_size, chunk_size, -1]), dim=1)


def make_gated_ref_alt_mlp_encoder(input_dimension: int, params: ModelParameters) -> GatedRefAltMLP:
    return GatedRefAltMLP(d_model=input_dimension, d_ffn=params.self_attention_hidden_dimension, num_blocks=params.num_self_attention_layers)


class PermutectModel(torch.nn.Module):
    """
    DeepSets framework for reads and variant info.  We embed each read and concatenate the mean ref read
    embedding, mean alt read embedding, and variant info embedding, then apply an aggregation function to
    this concatenation to obtain an embedding / representation of the read set for downstream use such as
    variant filtering and clustering.

    hidden_read_layers: dimensions of layers for embedding reads, excluding input dimension, which is the
    size of each read's 1D tensor

    hidden_info_layers: dimensions of layers for embedding variant info, excluding input dimension, which is the
    size of variant info 1D tensor

    aggregation_layers: dimensions of layers for aggregation, excluding its input which is determined by the
    read and info embeddings.

    output_layers: dimensions of layers after aggregation, excluding the output dimension,
    which is 1 for a single logit representing artifact/non-artifact.  This is not part of the aggregation layers
    because we have different output layers for each variant type.
    """

    def __init__(self, params: ModelParameters, num_read_features: int, num_info_features: int, haplotypes_length: int, device=gpu_if_available()):
        super(PermutectModel, self).__init__()

        self._device = device
        self._dtype = DEFAULT_GPU_FLOAT if device != torch.device("cpu") else DEFAULT_CPU_FLOAT
        self._haplotypes_length = haplotypes_length # this is the length of ref and alt concatenated horizontally ie twice the CNN length
        self._params = params

        # embeddings of reads, info, and reference sequence prior to the transformer layers
        self.read_embedding = MLP([num_read_features] + params.read_layers, batch_normalize=params.batch_normalize, dropout_p=params.dropout_p)
        self.info_embedding = MLP([num_info_features] + params.info_layers, batch_normalize=params.batch_normalize, dropout_p=params.dropout_p)
        self.haplotypes_cnn = DNASequenceConvolution(params.ref_seq_layer_strings, haplotypes_length // 2)

        embedding_dim = self.read_embedding.output_dimension() + self.info_embedding.output_dimension() + self.haplotypes_cnn.output_dimension()

        self.ref_alt_reads_encoder = make_gated_ref_alt_mlp_encoder(embedding_dim, params)

        # after encoding alt reads (along with info and ref seq embeddings and with self-attention to ref reads)
        # aggregate encoded sets in a permutation-invariant way
        # TODO: hard-coded magic constant!!!!!
        set_pooling_hidden_layers = [-2, -2]
        self.set_pooling = SetPooling(input_dim=embedding_dim, mlp_layers=set_pooling_hidden_layers,
                                      final_mlp_layers=params.aggregation_layers, batch_normalize=params.batch_normalize, dropout_p=params.dropout_p)

        # TODO: artifact classifier hidden layers are hard-coded!!!
        # The [1] is for the output logit
        self.artifact_classifier = MLP([self.set_pooling.output_dimension()] + [-1, -1, 1], batch_normalize=params.batch_normalize,
                                       dropout_p=params.dropout_p)

        self.calibration = Calibration(params.calibration_layers)

        self.alt_count_predictor = Adversarial(MLP([self.pooling_dimension()] + [30, -1, -1, -1, 1]), adversarial_strength=0.01)
        self.alt_count_loss_func = torch.nn.MSELoss(reduction='none')

        # used for unlabeled domain adaptation -- needs to be reset depending on the number of sources, as well as
        # the particular sources used in training.  Note that we initialize as a trivial model with 1 source
        self.source_predictor = Adversarial(MLP([self.pooling_dimension()] + [1], batch_normalize=params.batch_normalize,
                dropout_p=params.dropout_p), adversarial_strength=0.01)
        self.num_sources = 1

        self.to(device=self._device, dtype=self._dtype)

    def reset_source_predictor(self, num_sources: int = 1):
        source_prediction_hidden_layers = [] if num_sources == 1 else [-1, -1]
        layers = [self.pooling_dimension()] + source_prediction_hidden_layers + [num_sources]
        self.source_predictor = Adversarial(MLP(layers, batch_normalize=self._params.batch_normalize,
            dropout_p=self._params.dropout_p), adversarial_strength=0.01).to(device=self._device, dtype=self._dtype)
        self.num_sources = num_sources

    def pooling_dimension(self) -> int:
        return self.set_pooling.output_dimension()

    def ref_alt_seq_embedding_dimension(self) -> int:
        return self.haplotypes_cnn.output_dimension()

    def haplotypes_length(self) -> int:
        return self._haplotypes_length

    def post_pooling_parameters(self):
        return chain(self.artifact_classifier.parameters(), self.calibration.parameters())

    def calibration_parameters(self):
        return self.calibration.parameters()

    def set_epoch_type(self, epoch_type: Epoch):
        if epoch_type == Epoch.TRAIN:
            self.train(True)
            unfreeze(self.parameters())
        else:
            self.train(False)
            freeze(self.parameters())

    # I really don't like the forward method of torch.nn.Module with its implicit calling that PyCharm doesn't recognize
    def forward(self, batch: ReadsBatch):
        pass

    # here 'b' is the batch index, 'r' is the flattened read index, and 'e' means an embedding dimension
    # so, for example, "re" means a 2D tensor with all reads in the batch stacked and "bre" means a 3D tensor indexed
    # first by variant within the batch, then the read within the variant
    def calculate_features(self, batch: ReadsBatch, weight_range: float = 0) -> Tensor:
        ref_counts_b, alt_counts_b = batch.get_ref_counts(), batch.get_alt_counts()
        total_ref, total_alt = torch.sum(ref_counts_b).item(), torch.sum(alt_counts_b).item()

        read_embeddings_re = self.read_embedding.forward(batch.get_reads_re().to(dtype=self._dtype))
        info_embeddings_be = self.info_embedding.forward(batch.get_info_be().to(dtype=self._dtype))
        ref_seq_embeddings_be = self.haplotypes_cnn(batch.get_one_hot_haplotypes_bcs().to(dtype=self._dtype))
        info_and_seq_be = torch.hstack((info_embeddings_be, ref_seq_embeddings_be))
        info_and_seq_re = torch.vstack((torch.repeat_interleave(info_and_seq_be, repeats=ref_counts_b, dim=0),
                                       torch.repeat_interleave(info_and_seq_be, repeats=alt_counts_b, dim=0)))
        reads_info_seq_re = torch.hstack((read_embeddings_re, info_and_seq_re))

        # TODO: might be a bug if every datum in batch has zero ref reads?
        ref_bre = RaggedSets.from_flattened_tensor_and_sizes(reads_info_seq_re[:total_ref], ref_counts_b)
        alt_bre = RaggedSets.from_flattened_tensor_and_sizes(reads_info_seq_re[total_ref:], alt_counts_b)
        _, transformed_alt_bre = self.ref_alt_reads_encoder.forward(ref_bre, alt_bre)

        # TODO: this old code has the random weighting logic which might still be valuable
        """
        transformed_alt_re = transformed_alt_bre.flattened_tensor_nf

        alt_weights_r = 1 + weight_range * (1 - 2 * torch.rand(total_alt, device=self._device, dtype=self._dtype))

        # normalize so read weights within each variant sum to 1
        alt_wt_sums_v = sums_over_rows(alt_weights_r, alt_counts)
        normalized_alt_weights_r = alt_weights_r / torch.repeat_interleave(alt_wt_sums_v, repeats=alt_counts, dim=0)

        alt_means_ve = sums_over_rows(transformed_alt_re * normalized_alt_weights_r[:,None], alt_counts)
        """
        result_be = self.set_pooling.forward(transformed_alt_bre)

        return result_be, ref_seq_embeddings_be # ref seq embeddings are useful later

    def logits_from_features(self, features_be: Tensor, ref_counts_b: IntTensor, alt_counts_b: IntTensor, var_types_b: IntTensor):
        uncalibrated_logits_b = self.artifact_classifier.forward(features_be).view(-1)
        calibrated_logits_b = self.calibration.calibrated_logits(uncalibrated_logits_b, ref_counts_b, alt_counts_b, var_types_b)
        return calibrated_logits_b, uncalibrated_logits_b

    def logits_from_reads_batch(self, features_be: Tensor, batch: ReadsBatch):
        return self.logits_from_features(features_be, batch.get_ref_counts(), batch.get_alt_counts(), batch.get_variant_types())

    def compute_source_prediction_losses(self, features_be: Tensor, batch: ReadsBatch) -> Tensor:
        if self.num_sources > 1:
            source_logits_bs = self.source_predictor.adversarial_forward(features_be)
            source_probs_bs = torch.nn.functional.softmax(source_logits_bs, dim=-1)
            source_targets_bs = torch.nn.functional.one_hot(batch.get_sources().long(), self.num_sources)
            return torch.sum(torch.square(source_probs_bs - source_targets_bs), dim=-1)
        else:
            return torch.zeros(batch.size(), device=self._device, dtype=self._dtype)

    def compute_alt_count_losses(self, features_be: Tensor, batch: ReadsBatch):
        alt_count_pred_b = torch.sigmoid(self.alt_count_predictor.adversarial_forward(features_be).view(-1))
        alt_count_target_b = batch.get_alt_counts().to(dtype=alt_count_pred_b.dtype) / MAX_ALT_COUNT
        return self.alt_count_loss_func(alt_count_pred_b, alt_count_target_b)

    def compute_batch_output(self, batch: ReadsBatch, batch_indices: BatchIndices, balancer: Balancer):
        weights_b, source_weights_b = balancer.process_batch_and_compute_weights(batch_indices)
        features_be, _ = self.calculate_features(batch, weight_range=self._params.reweighting_range)
        calibrated_logits_b, uncalibrated_logits_b = self.logits_from_reads_batch(features_be, batch)
        return BatchOutput(features=features_be, uncalibrated_logits=uncalibrated_logits_b, calibrated_logits=calibrated_logits_b,
                           weights=weights_b, source_weights=weights_b*source_weights_b)

    def make_dict_for_saving(self, artifact_log_priors=None, artifact_spectra=None):
        return {constants.STATE_DICT_NAME: self.state_dict(),
                constants.HYPERPARAMS_NAME: self._params,
                constants.NUM_READ_FEATURES_NAME: self.read_embedding.input_dimension(),
                constants.NUM_INFO_FEATURES_NAME: self.info_embedding.input_dimension(),
                constants.REF_SEQUENCE_LENGTH_NAME: self.haplotypes_length(),
                constants.ARTIFACT_LOG_PRIORS_NAME: artifact_log_priors,
                constants.ARTIFACT_SPECTRA_STATE_DICT_NAME: artifact_spectra.state_dict() if artifact_spectra is not None else None}

    # save a model, optionally with artifact log priors and spectra
    def save_model(self, path, artifact_log_priors=None, artifact_spectra=None):
        self.reset_source_predictor()   # this way it's always the same in save/load to avoid state_dict mismatches
        torch.save(self.make_dict_for_saving(artifact_log_priors, artifact_spectra), path)


def load_model(path, device: torch.device = gpu_if_available()):
    saved = torch.load(path, map_location=device)
    hyperparams = saved[constants.HYPERPARAMS_NAME]
    num_read_features = saved[constants.NUM_READ_FEATURES_NAME]
    num_info_features = saved[constants.NUM_INFO_FEATURES_NAME]
    ref_sequence_length = saved[constants.REF_SEQUENCE_LENGTH_NAME]

    model = PermutectModel(hyperparams, num_read_features=num_read_features, num_info_features=num_info_features,
                           haplotypes_length=ref_sequence_length, device=device)
    model.load_state_dict(saved[constants.STATE_DICT_NAME])

    # in case the state dict had the wrong dtype for the device we're on now eg base model was pretrained on GPU
    # and we're now on CPU
    model.to(model._dtype)
    artifact_log_priors = saved[constants.ARTIFACT_LOG_PRIORS_NAME]  # possibly None
    artifact_spectra_state_dict = saved[constants.ARTIFACT_SPECTRA_STATE_DICT_NAME]  # possibly None

    return model, artifact_log_priors, artifact_spectra_state_dict


def permute_columns_independently(mat: Tensor):
    assert mat.dim() == 2
    num_rows, num_cols = mat.size()
    weights = torch.ones(num_rows)

    result = torch.clone(mat)
    for col in range(num_cols):
        idx = torch.multinomial(weights, num_rows, replacement=True)
        result[:, col] = result[:, col][idx]
    return result


# after training for visualizing clustering etc of base model embeddings
def record_embeddings(model: PermutectModel, loader, summary_writer: SummaryWriter):
    # base_model.freeze_all() whoops -- it doesn't have freeze_all
    embedding_metrics = EmbeddingMetrics()
    ref_alt_seq_metrics = EmbeddingMetrics()

    batch: ReadsBatch
    for batch in tqdm(prefetch_generator(loader), mininterval=60, total=len(loader)):
        features_be, ref_alt_seq_embeddings_be = model.calculate_features(batch, weight_range=model._params.reweighting_range)

        features_be = features_be.cpu()
        ref_alt_seq_embeddings_be = ref_alt_seq_embeddings_be.cpu()

        labels = [("artifact" if label > 0.5 else "non-artifact") if is_labeled > 0.5 else "unlabeled" for (label, is_labeled) in
                  zip(batch.get_training_labels().tolist(), batch.get_is_labeled_mask().tolist())]
        for (metrics, embeddings) in [(embedding_metrics, features_be), (ref_alt_seq_metrics, ref_alt_seq_embeddings_be)]:
            metrics.label_metadata.extend(labels)
            metrics.correct_metadata.extend(["unknown"] * batch.size())
            metrics.type_metadata.extend([Variation(idx).name for idx in batch.get_variant_types().tolist()])
            alt_count_strings = [alt_count_bin_name(alt_count_bin_index(ac)) for ac in batch.get_alt_counts().tolist()]
            metrics.truncated_count_metadata.extend(alt_count_strings)
            metrics.features.append(embeddings)
    embedding_metrics.output_to_summary_writer(summary_writer)
    ref_alt_seq_metrics.output_to_summary_writer(summary_writer, prefix="ref and alt allele context")

