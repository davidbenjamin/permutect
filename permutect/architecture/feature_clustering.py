import torch
from torch import nn, Tensor
from torch.nn import Parameter


class FeatureClustering(nn.Module):
    def __init__(self, feature_dimension: int, num_artifact_clusters: int):
        super(FeatureClustering, self).__init__()

        self.feature_dim = feature_dimension
        self.num_artifact_clusters = num_artifact_clusters

        # the 0th cluster is non-artifact
        self.num_clusters = self.num_artifact_clusters + 1

        # num_clusters different centroids, each a vector in feature space.  Initialize even weights.
        self.centroids_ke = Parameter(torch.rand(self.num_clusters, self.feature_dim))
        self.cluster_weights_pre_softmax_k = Parameter(torch.ones(self.num_artifact_clusters))
        self.centroid_distance_normalization = Parameter(1 / torch.sqrt(torch.tensor(self.feature_dim)), requires_grad=False)

    def calculate_logits(self, features_be: Tensor):
        batch_size = len(features_be)
        centroids_bke = self.centroids_ke.view(1, self.num_clusters, self.feature_dim)
        features_bke = features_be.view(batch_size, 1, self.feature_dim)
        diff_bke = centroids_bke - features_bke
        dist_bk = torch.norm(diff_bke, dim=-1) * self.centroid_distance_normalization
        uncalibrated_logits_bk = -dist_bk

        # these are the log of weights that sum to 1
        log_artifact_cluster_weights_k = torch.log_softmax(self.cluster_weights_pre_softmax_k, dim=-1)

        # total all the artifact clusters in log space and subtract the non-artifact cluster
        uncalibrated_logits_b = torch.logsumexp(log_artifact_cluster_weights_k + uncalibrated_logits_bk[:, 1:],
                                                dim=-1) - uncalibrated_logits_bk[:, 0]
        return uncalibrated_logits_b

    # avoid implicit forward calls because PyCharm doesn't recognize them
    def forward(self, features: Tensor):
        pass