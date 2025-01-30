from itertools import chain

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm.autonotebook import tqdm

from permutect import utils, constants
from permutect.architecture.calibration import Calibration
from permutect.architecture.dna_sequence_convolution import DNASequenceConvolution
from permutect.architecture.gated_mlp import GatedRefAltMLP
from permutect.architecture.mlp import MLP
from permutect.architecture.set_pooling import SetPooling
from permutect.data.base_datum import BaseBatch, DEFAULT_GPU_FLOAT, DEFAULT_CPU_FLOAT, ArtifactBatch
from permutect.data.base_dataset import ALL_COUNTS_INDEX
from permutect.metrics.evaluation_metrics import EmbeddingMetrics, round_up_to_nearest_three, MAX_COUNT
from permutect.parameters import ModelParameters
from permutect.sets.ragged_sets import RaggedSets
from permutect.utils import Variation


def sums_over_chunks(tensor2d: torch.Tensor, chunk_size: int):
    assert len(tensor2d) % chunk_size == 0
    return torch.sum(tensor2d.reshape([len(tensor2d) // chunk_size, chunk_size, -1]), dim=1)


# note: this works for both BaseBatch/BaseDataset AND ArtifactBatch/ArtifactDataset
# if by_count is True, each count is weighted separately for balanced loss within that count
def calculate_batch_weights(batch, dataset, by_count: bool):
    # TODO: we need a parameter to control the relative weight of unlabeled loss to labeled loss
    # For batch index n, we want weight[n] = dataset.weights[alt_counts[n], labels[n], variant_types[n]]
    sources = batch.get_sources()
    counts = batch.get_alt_counts() if by_count else torch.full(size=(len(sources), ), fill_value=ALL_COUNTS_INDEX, dtype=torch.int)
    labels = batch.get_labels()
    variant_types = batch.get_variant_types()

    return utils.index_4d_array(dataset.label_balancing_weights_sclt, sources, counts, labels, variant_types)


# note: this works for both BaseBatch/BaseDataset AND ArtifactBatch/ArtifactDataset
# if by_count is True, each count is weighted separately for balanced loss within that count
def calculate_batch_source_weights(batch, dataset, by_count: bool):
    # For batch index n, we want weight[n] = dataset.source_weights[sources[n], alt_counts[n], variant_types[n]]
    sources = batch.get_sources()
    counts = batch.get_alt_counts() if by_count else torch.full(size=(len(sources), ), fill_value=ALL_COUNTS_INDEX, dtype=torch.int)
    variant_types = batch.get_variant_types()

    return utils.index_3d_array(dataset.source_balancing_weights_sct, sources, counts, variant_types)


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

    def __init__(self, params: ModelParameters, num_read_features: int, num_info_features: int, ref_sequence_length: int, device=utils.gpu_if_available()):
        super(PermutectModel, self).__init__()

        self._device = device
        self._dtype = DEFAULT_GPU_FLOAT if device != torch.device("cpu") else DEFAULT_CPU_FLOAT
        self._ref_sequence_length = ref_sequence_length
        self._params = params

        # embeddings of reads, info, and reference sequence prior to the transformer layers
        self.read_embedding = MLP([num_read_features] + params.read_layers, batch_normalize=params.batch_normalize, dropout_p=params.dropout_p)
        self.info_embedding = MLP([num_info_features] + params.info_layers, batch_normalize=params.batch_normalize, dropout_p=params.dropout_p)
        self.ref_seq_cnn = DNASequenceConvolution(params.ref_seq_layer_strings, ref_sequence_length)

        embedding_dim = self.read_embedding.output_dimension() + self.info_embedding.output_dimension() + self.ref_seq_cnn.output_dimension()

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

        # one Calibration module for each variant type; that is, calibration depends on both count and type
        self.calibration = nn.ModuleList([Calibration(params.calibration_layers) for variant_type in Variation])

        self.to(device=self._device, dtype=self._dtype)

    def pooling_dimension(self) -> int:
        return self.set_pooling.output_dimension()

    def ref_alt_seq_embedding_dimension(self) -> int:
        return self.ref_seq_cnn.output_dimension()

    def ref_sequence_length(self) -> int:
        return self._ref_sequence_length

    def post_pooling_parameters(self):
        return chain(self.artifact_classifier.parameters(), self.calibration.parameters())

    def calibration_parameters(self):
        return self.calibration.parameters()

    def final_calibration_shift_parameters(self):
        return [cal.final_adjustments for cal in self.calibration]

    def set_epoch_type(self, epoch_type: utils.Epoch):
        if epoch_type == utils.Epoch.TRAIN:
            self.train(True)
            utils.unfreeze(self.parameters())
        else:
            self.train(False)
            utils.freeze(self.parameters())

    # I really don't like the forward method of torch.nn.Module with its implicit calling that PyCharm doesn't recognize
    def forward(self, batch: BaseBatch):
        pass

    # TODO: perhaps rename to calculate_features?
    # here 'v' means "variant index within a batch", 'r' means "read index within a variant or the batch", 'e' means "index within an embedding"
    # so, for example, "re" means a 2D tensor with all reads in the batch stacked and "vre" means a 3D tensor indexed
    # first by variant within the batch, then the read
    def calculate_representations(self, batch: BaseBatch, weight_range: float = 0) -> torch.Tensor:
        ref_counts, alt_counts = batch.get_ref_counts(), batch.get_alt_counts()
        total_ref, total_alt = torch.sum(ref_counts).item(), torch.sum(alt_counts).item()

        read_embeddings_re = self.read_embedding.forward(batch.get_reads_2d().to(dtype=self._dtype))
        info_embeddings_ve = self.info_embedding.forward(batch.get_info_2d().to(dtype=self._dtype))
        ref_seq_embeddings_ve = self.ref_seq_cnn(batch.get_ref_sequences_2d().to(dtype=self._dtype))
        info_and_seq_ve = torch.hstack((info_embeddings_ve, ref_seq_embeddings_ve))
        info_and_seq_re = torch.vstack((torch.repeat_interleave(info_and_seq_ve, repeats=ref_counts, dim=0),
                                       torch.repeat_interleave(info_and_seq_ve, repeats=alt_counts, dim=0)))
        reads_info_seq_re = torch.hstack((read_embeddings_re, info_and_seq_re))

        # TODO: might be a bug if every datum in batch has zero ref reads?
        ref_bre = RaggedSets.from_flattened_tensor_and_sizes(reads_info_seq_re[:total_ref], ref_counts)
        alt_bre = RaggedSets.from_flattened_tensor_and_sizes(reads_info_seq_re[total_ref:], alt_counts)
        _, transformed_alt_bre = self.ref_alt_reads_encoder.forward(ref_bre, alt_bre)

        # TODO: this old code has the random weighting logic which might still be valuable
        """
        transformed_alt_re = transformed_alt_bre.flattened_tensor_nf

        alt_weights_r = 1 + weight_range * (1 - 2 * torch.rand(total_alt, device=self._device, dtype=self._dtype))

        # normalize so read weights within each variant sum to 1
        alt_wt_sums_v = utils.sums_over_rows(alt_weights_r, alt_counts)
        normalized_alt_weights_r = alt_weights_r / torch.repeat_interleave(alt_wt_sums_v, repeats=alt_counts, dim=0)

        alt_means_ve = utils.sums_over_rows(transformed_alt_re * normalized_alt_weights_r[:,None], alt_counts)
        """
        result_be = self.set_pooling.forward(transformed_alt_bre)

        return result_be, ref_seq_embeddings_ve # ref seq embeddings are useful later

    def logits_from_features(self, representations_2d: torch.Tensor, ref_counts: torch.IntTensor, alt_counts: torch.IntTensor,
                           variant_types: torch.IntTensor):
        uncalibrated_logits = self.artifact_classifier.forward(representations_2d).view(-1)
        calibrated_logits = torch.zeros_like(uncalibrated_logits, device=self._device)
        for n, _ in enumerate(Variation):
            mask = (variant_types == n)
            calibrated_logits += mask * self.calibration[n].forward(uncalibrated_logits, ref_counts, alt_counts)
        return calibrated_logits, uncalibrated_logits

    def logits_from_base_batch(self, representations_2d: torch.Tensor, base_batch: BaseBatch):
        return self.logits_from_features(representations_2d, base_batch.get_ref_counts(), base_batch.get_alt_counts(), base_batch.get_variant_types())

    # returns 1D tensor of length batch_size of log odds ratio (logits) between artifact and non-artifact
    def logits_from_artifact_batch(self, batch: ArtifactBatch):
        return self.logits_from_features(batch.get_representations_2d(), batch.get_ref_counts(), batch.get_alt_counts(), batch.get_variant_types())

    def make_dict_for_saving(self, artifact_log_priors=None, artifact_spectra=None):
        return {constants.STATE_DICT_NAME: self.state_dict(),
                constants.HYPERPARAMS_NAME: self._params,
                constants.NUM_READ_FEATURES_NAME: self.read_embedding.input_dimension(),
                constants.NUM_INFO_FEATURES_NAME: self.info_embedding.input_dimension(),
                constants.REF_SEQUENCE_LENGTH_NAME: self.ref_sequence_length(),
                constants.ARTIFACT_LOG_PRIORS_NAME: artifact_log_priors,
                constants.ARTIFACT_SPECTRA_STATE_DICT_NAME: artifact_spectra.state_dict() if artifact_spectra is not None else None}

    # save a model, optionally with artifact log priors and spectra
    def save_model(self, path, artifact_log_priors=None, artifact_spectra=None):
        torch.save(self.make_dict_for_saving(artifact_log_priors, artifact_spectra), path)


def load_model(path, device: torch.device = utils.gpu_if_available()):
    saved = torch.load(path, map_location=device)
    hyperparams = saved[constants.HYPERPARAMS_NAME]
    num_read_features = saved[constants.NUM_READ_FEATURES_NAME]
    num_info_features = saved[constants.NUM_INFO_FEATURES_NAME]
    ref_sequence_length = saved[constants.REF_SEQUENCE_LENGTH_NAME]

    model = PermutectModel(hyperparams, num_read_features=num_read_features, num_info_features=num_info_features,
                           ref_sequence_length=ref_sequence_length, device=device)
    model.load_state_dict(saved[constants.STATE_DICT_NAME])

    # in case the state dict had the wrong dtype for the device we're on now eg base model was pretrained on GPU
    # and we're now on CPU
    model.to(model._dtype)
    artifact_log_priors = saved[constants.ARTIFACT_LOG_PRIORS_NAME]  # possibly None
    artifact_spectra_state_dict = saved[constants.ARTIFACT_SPECTRA_STATE_DICT_NAME]  # possibly None

    return model, artifact_log_priors, artifact_spectra_state_dict


def permute_columns_independently(mat: torch.Tensor):
    assert mat.dim() == 2
    num_rows, num_cols = mat.size()
    weights = torch.ones(num_rows)

    result = torch.clone(mat)
    for col in range(num_cols):
        idx = torch.multinomial(weights, num_rows, replacement=True)
        result[:, col] = result[:, col][idx]
    return result


# after training for visualizing clustering etc of base model embeddings
def record_embeddings(base_model: PermutectModel, loader, summary_writer: SummaryWriter):
    # base_model.freeze_all() whoops -- it doesn't have freeze_all
    embedding_metrics = EmbeddingMetrics()
    ref_alt_seq_metrics = EmbeddingMetrics()

    pbar = tqdm(enumerate(loader), mininterval=60)
    for n, batch_cpu in pbar:
        batch = batch_cpu.copy_to(base_model._device, non_blocking=base_model._device.type=='cuda')
        representations, ref_alt_seq_embeddings = base_model.calculate_representations(batch, weight_range=base_model._params.reweighting_range)

        representations = representations.cpu()
        ref_alt_seq_embeddings = ref_alt_seq_embeddings.cpu()

        labels = [("artifact" if label > 0.5 else "non-artifact") if is_labeled > 0.5 else "unlabeled" for (label, is_labeled) in
                  zip(batch.get_training_labels().tolist(), batch.get_is_labeled_mask().tolist())]
        for (metrics, embeddings) in [(embedding_metrics, representations), (ref_alt_seq_metrics, ref_alt_seq_embeddings)]:
            metrics.label_metadata.extend(labels)
            metrics.correct_metadata.extend(["unknown"] * batch.size())
            metrics.type_metadata.extend([Variation(idx).name for idx in batch.get_variant_types().tolist()])
            alt_count_strings = [str(round_up_to_nearest_three(min(MAX_COUNT, ac))) for ac in batch.get_alt_counts().tolist()]
            metrics.truncated_count_metadata.extend(alt_count_strings)
            metrics.representations.append(embeddings)
    embedding_metrics.output_to_summary_writer(summary_writer)
    ref_alt_seq_metrics.output_to_summary_writer(summary_writer, prefix="ref and alt allele context")

