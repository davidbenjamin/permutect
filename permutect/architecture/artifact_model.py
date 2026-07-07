from typing import List

import torch
from torch import Tensor
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm.autonotebook import tqdm

from permutect import constants
from permutect.architecture.adversarial import Adversarial
from permutect.architecture.dna_sequence_convolution import DNASequenceConvolution
from permutect.architecture.euclidean_transformation import EuclideanTransformation
from permutect.architecture.feature_clustering import FeatureClustering
from permutect.architecture.gated_mlp import GatedRefAltMLP
from permutect.architecture.mlp import MLP
from permutect.data.batch import Batch
from permutect.data.count_binning import MAX_ALT_COUNT
from permutect.data.count_binning import alt_count_bin_index
from permutect.data.count_binning import alt_count_bin_name
from permutect.data.datum import DEFAULT_CPU_FLOAT
from permutect.data.datum import DEFAULT_GPU_FLOAT
from permutect.data.datum import Data
from permutect.data.prefetch_generator import prefetch_generator
from permutect.metrics.evaluation_metrics import EmbeddingMetrics
from permutect.misc_utils import freeze
from permutect.misc_utils import gpu_if_available
from permutect.misc_utils import unfreeze
from permutect.parameters import ModelParameters
from permutect.sets.ragged_sets import RaggedSets
from permutect.training.balancer import Balancer
from permutect.utils.enums import Epoch
from permutect.utils.enums import ParameterSet
from permutect.utils.enums import Variation

MAX_OUTLIER_LOGIT = 10
BCE = nn.BCEWithLogitsLoss(
    reduction="none"
)  # no reduction because we may want to first multiply by weights for unbalanced data


class BatchOutput:
    """
    simple container class for the output of the model over a single batch
    :return:
    """

    def __init__(
        self,
        features_be: Tensor,
        ref_features_be: Tensor,
        logits_b: Tensor,
        logits_bk: Tensor,
        weights: Tensor,
    ):
        self.features_be = features_be
        self.ref_features_be = ref_features_be
        self.logits_b = logits_b
        self.artifact_probs_b = torch.sigmoid(logits_b)
        self.logits_bk = logits_bk
        self.weights = weights
        self.outlier_binary_logits = self._outlier_binary_logits()

    def _outlier_binary_logits(self) -> Tensor:
        # columns of output.calibrated_logits_bk are nonartifact, then outlier, then artifact clusters.
        nonart_logits_bk = self.logits_bk[:, 0][:, None]
        art_logits_bk = self.logits_bk[:, 2:]
        nonoutlier_logits_bk = torch.cat((nonart_logits_bk, art_logits_bk), dim=-1)
        nonoutlier_logits_b = torch.logsumexp(nonoutlier_logits_bk, dim=-1)
        outlier_logits_b = self.logits_bk[:, 1]

        # this is a binary logit representing the probability that the datum was classified as an outlier
        # i.e. not in the nonartifact Gaussian nor the artifact distributions
        outlier_binary_logits_b = outlier_logits_b - nonoutlier_logits_b
        return outlier_binary_logits_b


class BatchLosses:
    def __init__(
        self,
        supervised_losses_b: Tensor,
        unsupervised_losses_b: Tensor,
        alt_count_losses_b: Tensor,
        total_losses_b: Tensor,
    ):
        self.supervised_losses_b = supervised_losses_b
        self.unsupervised_losses_b = unsupervised_losses_b
        self.alt_count_losses_b = alt_count_losses_b
        self.total_losses_b = total_losses_b
        self.total_loss = torch.sum(total_losses_b)


def make_gated_ref_alt_mlp_encoder(input_dimension: int, params: ModelParameters) -> GatedRefAltMLP:
    return GatedRefAltMLP(
        d_model=input_dimension,
        d_ffn=params.self_attention_hidden_dimension,
        num_blocks=params.num_self_attention_layers,
    )


class ArtifactModel(torch.nn.Module):
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

    def __init__(
        self,
        params: ModelParameters,
        num_read_features: int,
        num_info_features: int,
        haplotypes_length: int,
        device=None,
    ):
        super(ArtifactModel, self).__init__()
        if device is None:
            device = gpu_if_available()

        self._device = device
        self._dtype = DEFAULT_GPU_FLOAT if device != torch.device("cpu") else DEFAULT_CPU_FLOAT

        # this is the length of ref and alt concatenated horizontally ie twice the CNN length
        self._haplotypes_length = haplotypes_length
        self._params = params

        # embeddings of reads, info, and reference sequence prior to the transformer layers
        self.read_embedding = MLP(
            [num_read_features] + params.read_layers,
            batch_normalize=params.batch_normalize,
            dropout_p=params.dropout_p,
        )
        self.info_embedding = MLP(
            [num_info_features] + params.info_layers,
            batch_normalize=params.batch_normalize,
            dropout_p=params.dropout_p,
        )
        self.haplotypes_cnn = DNASequenceConvolution(params.ref_seq_layer_strings, haplotypes_length // 2)

        embedding_dim = (
            self.read_embedding.output_dimension()
            + self.info_embedding.output_dimension()
            + self.haplotypes_cnn.output_dimension()
        )

        # TODO: reduce dimension before sending to the encoder???
        self.ref_alt_reads_encoder = make_gated_ref_alt_mlp_encoder(embedding_dim, params)

        # reduce dimensionality of reads after gated MLP for better clustering etc
        self.reducer = MLP(
            [self.ref_alt_reads_encoder.output_dimension()] + params.aggregation_layers,
            batch_normalize=params.batch_normalize,
            dropout_p=params.dropout_p,
        )

        # Feature clustering posits nonartifact reads have a zero-centered diagonal Gaussian.
        # We shift and rotate so that the Gaussian is zero-centered and has diagonal covariance.
        # This is also useful for domain adaptation.
        self.pre_clustering_transform = EuclideanTransformation(self.reducer.output_dimension())

        self.feature_clustering = FeatureClustering(
            feature_dimension=self.reducer.output_dimension(),
            num_artifact_clusters=params.num_artifact_clusters,
        )

        self.alt_count_predictor = Adversarial(
            MLP([self.reducer.output_dimension()] + [30, -1, -1, -1, 1]), adversarial_strength=0.01
        )
        self.alt_count_loss_func = torch.nn.MSELoss(reduction="none")

        # used for unlabeled domain adaptation -- needs to be reset depending on the number of sources, as well as
        # the particular sources used in training.  Note that we initialize as a trivial model with 1 source
        self.source_predictor = Adversarial(
            MLP(
                [self.reducer.output_dimension()] + [1],
                batch_normalize=params.batch_normalize,
                dropout_p=params.dropout_p,
            ),
            adversarial_strength=0.01,
        )
        self.num_sources = 1

        self.to(device=self._device, dtype=self._dtype)

    def reset_source_predictor(self, num_sources: int = 1):
        source_prediction_hidden_layers = [] if num_sources == 1 else [-1, -1]
        layers = [self.reducer.output_dimension()] + source_prediction_hidden_layers + [num_sources]
        self.source_predictor = Adversarial(
            MLP(
                layers,
                batch_normalize=self._params.batch_normalize,
                dropout_p=self._params.dropout_p,
            ),
            adversarial_strength=0.01,
        ).to(device=self._device, dtype=self._dtype)
        self.num_sources = num_sources

    def ref_alt_seq_embedding_dimension(self) -> int:
        return self.haplotypes_cnn.output_dimension()

    def haplotypes_length(self) -> int:
        return self._haplotypes_length

    def set_epoch_type(self, epoch_type: Epoch, trainable_params: List[ParameterSet] = None):
        if epoch_type == Epoch.TRAIN:
            if trainable_params is None:
                self.train(True)
                unfreeze(self.parameters())
            else:
                freeze(self.parameters())
                for parameter_set in trainable_params:
                    unfreeze(parameter_set.get_parameters(self))
        else:
            self.train(False)
            freeze(self.parameters())

    # I really don't like the forward method of torch.nn.Module with its implicit calling that PyCharm doesn't recognize
    def forward(self, batch: Batch):
        pass

    # here 'b' is the batch index, 'r' is the flattened read index, and 'e' means an embedding dimension
    # so, for example, "re" means a 2D tensor with all reads in the batch stacked and "bre" means a 3D tensor indexed
    # first by variant within the batch, then the read within the variant
    def calculate_features(self, batch: Batch) -> tuple[RaggedSets, RaggedSets, Tensor]:
        ref_counts_b, alt_counts_b = batch.get(Data.REF_COUNT), batch.get(Data.ALT_COUNT)
        total_ref = torch.sum(ref_counts_b).item()

        read_embeddings_re = self.read_embedding.forward(batch.get_reads_re().to(dtype=self._dtype))
        info_embeddings_be = self.info_embedding.forward(batch.get_info_be().to(dtype=self._dtype))
        ref_seq_embeddings_be = self.haplotypes_cnn(batch.get_one_hot_haplotypes_bcs().to(dtype=self._dtype))
        info_and_seq_be = torch.hstack((info_embeddings_be, ref_seq_embeddings_be))

        ref_info_and_seq_re = torch.repeat_interleave(info_and_seq_be, repeats=ref_counts_b, dim=0)
        alt_info_and_seq_re = torch.repeat_interleave(info_and_seq_be, repeats=alt_counts_b, dim=0)
        info_and_seq_re = torch.vstack((ref_info_and_seq_re, alt_info_and_seq_re))
        reads_info_seq_re = torch.hstack((read_embeddings_re, info_and_seq_re))

        # TODO: might be a bug if every datum in batch has zero ref reads?
        ref_bre = RaggedSets(flattened_tensor_nf=reads_info_seq_re[:total_ref], lengths_b=ref_counts_b)
        alt_bre = RaggedSets(flattened_tensor_nf=reads_info_seq_re[total_ref:], lengths_b=alt_counts_b)
        transformed_ref_bre, transformed_alt_bre = self.ref_alt_reads_encoder.forward(ref_bre, alt_bre)

        reduced_ref_bre = transformed_ref_bre.apply_elementwise(self.reducer)
        reduced_alt_bre = transformed_alt_bre.apply_elementwise(self.reducer)

        final_transform = lambda reads_re: self.pre_clustering_transform.transform(reads_re)
        final_ref_bre = reduced_ref_bre.apply_elementwise(final_transform)
        final_alt_bre = reduced_alt_bre.apply_elementwise(final_transform)

        return final_ref_bre, final_alt_bre, ref_seq_embeddings_be  # ref seq embeddings are useful later

    def compute_alt_count_losses(self, features_be: Tensor, batch: Batch):
        alt_count_pred_b = torch.sigmoid(self.alt_count_predictor.adversarial_forward(features_be).view(-1))
        alt_count_target_b = batch.get(Data.ALT_COUNT).to(dtype=alt_count_pred_b.dtype) / MAX_ALT_COUNT
        return self.alt_count_loss_func(alt_count_pred_b, alt_count_target_b)

    def compute_batch_output(self, batch: Batch, balancer: Balancer = None):
        ref_bre, alt_bre, _ = self.calculate_features(batch)  # ragged sets of reduced and transformed reads
        logits_b, logits_bk = self.feature_clustering.calculate_logits(alt_bre, batch)

        weights_b = (
            torch.ones_like(logits_b)
            if balancer is None
            else balancer.process_batch_and_compute_weights(batch, artifact_probs_b=torch.sigmoid(logits_b).detach())
        )
        return BatchOutput(
            features_be=alt_bre.means_over_sets(),
            ref_features_be=ref_bre.means_over_sets(),
            logits_b=logits_b,
            logits_bk=logits_bk,
            weights=weights_b,
        )

    def compute_batch_losses(self, output: BatchOutput, batch: Batch):
        labels_b = batch.get_training_labels()
        is_labeled_b = batch.get_is_labeled_mask()
        supervised_losses_b = is_labeled_b * BCE(output.logits_b, labels_b)

        # Unsupervised loss encourages read embeddings to have high density in the feature clustering model.
        # We do this by penalizes the probability assigned to the outlier pseudo-cluster. Since
        # some genuine outlier data does exist, such as rare or unmodeled artifacts, we clip the outlier
        # logit to avert unduly strong influence.
        clipped_logits = torch.clip(output.outlier_binary_logits, max=MAX_OUTLIER_LOGIT)
        outlier_losses_b = BCE(clipped_logits, torch.zeros_like(output.outlier_binary_logits))

        unsupervised_losses_b = (1 - is_labeled_b) * outlier_losses_b
        alt_count_losses_b = self.compute_alt_count_losses(output.features_be, batch)

        total_losses_b = output.weights * (supervised_losses_b + unsupervised_losses_b + alt_count_losses_b)
        return BatchLosses(
            supervised_losses_b=supervised_losses_b,
            unsupervised_losses_b=unsupervised_losses_b,
            alt_count_losses_b=alt_count_losses_b,
            total_losses_b=total_losses_b,
        )

    def make_dict_for_saving(self, artifact_log_priors=None, artifact_spectra=None):
        spectra_dict = artifact_spectra.state_dict() if artifact_spectra is not None else None
        return {
            constants.STATE_DICT_NAME: self.state_dict(),
            constants.HYPERPARAMS_NAME: self._params,
            constants.NUM_READ_FEATURES_NAME: self.read_embedding.input_dimension(),
            constants.NUM_INFO_FEATURES_NAME: self.info_embedding.input_dimension(),
            constants.REF_SEQUENCE_LENGTH_NAME: self.haplotypes_length(),
            constants.ARTIFACT_LOG_PRIORS_NAME: artifact_log_priors,
            constants.ARTIFACT_SPECTRA_STATE_DICT_NAME: spectra_dict,
        }

    # save a model, optionally with artifact log priors and spectra
    def save_model(self, path, artifact_log_priors=None, artifact_spectra=None):
        self.reset_source_predictor()  # this way it's always the same in save/load to avoid state_dict mismatches
        torch.save(self.make_dict_for_saving(artifact_log_priors, artifact_spectra), path)


def load_model(path, device: torch.device = None):
    if device is None:
        device = gpu_if_available()
    saved = torch.load(path, map_location=device, weights_only=False)
    hyperparams = saved[constants.HYPERPARAMS_NAME]
    num_read_features = saved[constants.NUM_READ_FEATURES_NAME]
    num_info_features = saved[constants.NUM_INFO_FEATURES_NAME]
    ref_sequence_length = saved[constants.REF_SEQUENCE_LENGTH_NAME]

    model = ArtifactModel(
        hyperparams,
        num_read_features=num_read_features,
        num_info_features=num_info_features,
        haplotypes_length=ref_sequence_length,
        device=device,
    )
    model.load_state_dict(saved[constants.STATE_DICT_NAME])

    # in case the state dict had the wrong dtype for the device we're on now eg base model was pretrained on GPU
    # and we're now on CPU
    model.to(model._dtype)
    artifact_log_priors = saved[constants.ARTIFACT_LOG_PRIORS_NAME]  # possibly None
    artifact_spectra_state_dict = saved[constants.ARTIFACT_SPECTRA_STATE_DICT_NAME]  # possibly None

    return model, artifact_log_priors, artifact_spectra_state_dict


# after training for visualizing clustering etc of base model embeddings
@torch.no_grad()
def record_embeddings(model: ArtifactModel, loader, summary_writer: SummaryWriter):
    embedding_metrics = EmbeddingMetrics()
    ref_alt_seq_metrics = EmbeddingMetrics()

    batch: Batch
    for batch in tqdm(prefetch_generator(loader), mininterval=60, total=len(loader)):
        ref_bre, alt_bre, ref_alt_seq_embeddings_be = model.calculate_features(batch)

        alt_means_be = alt_bre.means_over_sets().cpu()
        ref_means_be = ref_bre.means_over_sets().cpu()
        ref_alt_seq_embeddings_be = ref_alt_seq_embeddings_be.cpu()

        labels = [
            ("artifact" if label > 0.5 else "non-artifact") if is_labeled > 0.5 else "unlabeled"
            for (label, is_labeled) in zip(batch.get_training_labels().tolist(), batch.get_is_labeled_mask().tolist())
        ]
        for metrics, embeddings, ref_features_be in [
            (embedding_metrics, alt_means_be, ref_means_be),
            (ref_alt_seq_metrics, ref_alt_seq_embeddings_be, None),
        ]:
            metrics.label_metadata.extend(labels)
            metrics.correct_metadata.extend(["unknown"] * batch.size())
            metrics.type_metadata.extend([Variation(idx).name for idx in batch.get(Data.VARIANT_TYPE).tolist()])
            alt_count_strings = [
                alt_count_bin_name(alt_count_bin_index(ac)) for ac in batch.get(Data.ALT_COUNT).tolist()
            ]
            metrics.truncated_count_metadata.extend(alt_count_strings)
            metrics.features.append(embeddings)
            if ref_features_be is not None:
                metrics.ref_features.append(ref_features_be)
    embedding_metrics.output_to_summary_writer(summary_writer)
    ref_alt_seq_metrics.output_to_summary_writer(summary_writer, prefix="ref and alt allele context")
