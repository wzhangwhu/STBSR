import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors, kneighbors_graph


DEFAULT_SPATIAL_RADIUS = 560
SPATIAL_RADIUS_BY_DATASET = {
    "SlideSeqV2_MouseEmbryo_E8_5": 65,
    "StereoSeq_MouseBrain": 450,
    "ARTISTA_Stage44_telencephalon": 105,
    "ARTISTA_Stage54_telencephalon": 105,
    "ARTISTA_Stage57_telencephalon": 85,
}

DEFAULT_PLOT_SPOT_SIZE = 100.0
PLOT_SPOT_SIZE_BY_DATASET = {
    "CH_D4": 250.0,
    "CH_D7": 250.0,
    "CH_D10": 250.0,
    "CH_D14": 250.0,
    "SlideSeqV2_MouseEmbryo_E8_5": 16.0,
    "StereoSeq_MouseBrain": 95.0,
    "ARTISTA_Stage44_telencephalon": 14.0,
    "ARTISTA_Stage54_telencephalon": 8.0,
    "ARTISTA_Stage57_telencephalon": 5.0,
}


def get_spatial_radius(dataset_name):
    return int(
        SPATIAL_RADIUS_BY_DATASET.get(dataset_name, DEFAULT_SPATIAL_RADIUS)
    )


def get_plot_spot_size(dataset_name):
    return float(
        PLOT_SPOT_SIZE_BY_DATASET.get(dataset_name, DEFAULT_PLOT_SPOT_SIZE)
    )


def _nan2inf(value):
    return torch.where(
        torch.isnan(value),
        torch.zeros_like(value) + np.inf,
        value,
    )


def normalize_sparse_matrix(matrix):
    """Row-normalize a sparse matrix."""
    row_sum = np.asarray(matrix.sum(1))
    inverse = np.power(row_sum, -1).flatten()
    inverse[np.isinf(inverse)] = 0.0
    return sp.diags(inverse).dot(matrix)


def sparse_mx_to_torch_sparse_tensor(sparse_matrix):
    """Convert a SciPy sparse matrix to a PyTorch sparse tensor."""
    sparse_matrix = sparse_matrix.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_matrix.row, sparse_matrix.col)).astype(np.int64)
    )
    values = torch.from_numpy(sparse_matrix.data)
    shape = torch.Size(sparse_matrix.shape)
    return torch.sparse_coo_tensor(indices, values, shape)


def spatial_construct_graph(adata, radius):
    """Construct a radius-neighbor graph from spatial coordinates."""
    coordinates = pd.DataFrame(
        np.asarray(adata.obsm["spatial"])[:, :2],
        index=adata.obs_names,
        columns=["spatial1", "spatial2"],
    )
    adjacency = np.zeros(
        (coordinates.shape[0], coordinates.shape[0]),
        dtype=np.float32,
    )
    neighbors = NearestNeighbors(radius=radius).fit(coordinates)
    indices = neighbors.radius_neighbors(
        coordinates,
        return_distance=False,
    )
    for row_index, neighbor_indices in enumerate(indices):
        adjacency[row_index, neighbor_indices] = 1.0

    graph_nei = torch.from_numpy(adjacency)
    graph_neg = torch.ones_like(graph_nei) - graph_nei
    spatial_adjacency = sp.coo_matrix(adjacency, dtype=np.float32)
    spatial_adjacency = (
        spatial_adjacency
        + spatial_adjacency.T.multiply(
            spatial_adjacency.T > spatial_adjacency
        )
        - spatial_adjacency.multiply(
            spatial_adjacency.T > spatial_adjacency
        )
    )
    return spatial_adjacency, graph_nei, graph_neg


def features_construct_graph(
    features,
    k=15,
    mode="connectivity",
    metric="cosine",
):
    """Construct a symmetric k-nearest-neighbor feature graph."""
    adjacency = kneighbors_graph(
        features,
        k + 1,
        mode=mode,
        metric=metric,
        include_self=True,
    ).toarray()
    row, column = np.diag_indices_from(adjacency)
    adjacency[row, column] = 0
    feature_adjacency = sp.coo_matrix(adjacency, dtype=np.float32)
    feature_adjacency = (
        feature_adjacency
        + feature_adjacency.T.multiply(
            feature_adjacency.T > feature_adjacency
        )
        - feature_adjacency.multiply(
            feature_adjacency.T > feature_adjacency
        )
    )
    return feature_adjacency


class NB:
    def __init__(self, theta=None, scale_factor=1.0):
        self.eps = 1e-10
        self.scale_factor = scale_factor
        self.theta = theta

    def loss(self, y_true, y_pred, mean=True):
        y_pred = y_pred * self.scale_factor
        theta = torch.minimum(
            self.theta,
            torch.tensor(1e6, device=self.theta.device),
        )
        first = (
            torch.lgamma(theta + self.eps)
            + torch.lgamma(y_true + 1.0)
            - torch.lgamma(y_true + theta + self.eps)
        )
        second = (
            (theta + y_true)
            * torch.log(1.0 + (y_pred / (theta + self.eps)))
            + y_true
            * (
                torch.log(theta + self.eps)
                - torch.log(y_pred + self.eps)
            )
        )
        result = _nan2inf(first + second)
        return torch.mean(result) if mean else result


class ZINB(NB):
    def __init__(self, pi, ridge_lambda=0.0, **kwargs):
        super().__init__(**kwargs)
        self.pi = pi
        self.ridge_lambda = ridge_lambda

    def loss(self, y_true, y_pred, mean=True):
        theta = torch.minimum(
            self.theta,
            torch.tensor(1e6, device=self.theta.device),
        )
        nb_case = (
            super().loss(y_true, y_pred, mean=False)
            - torch.log(1.0 - self.pi + self.eps)
        )
        y_pred = y_pred * self.scale_factor
        zero_nb = torch.pow(
            theta / (theta + y_pred + self.eps),
            theta,
        )
        zero_case = -torch.log(
            self.pi + ((1.0 - self.pi) * zero_nb) + self.eps
        )
        result = torch.where(y_true < 1e-8, zero_case, nb_case)
        result += self.ridge_lambda * torch.square(self.pi)
        if mean:
            result = torch.mean(result)
        return _nan2inf(result)


def dicr_loss_little_block_diagonal_precluster(
    emb1,
    emb2,
    labels,
    *,
    intra_weight=1.0,
    inter_weight=1.0,
    temperature=1.0,
    balance_classes=True,
    return_parts=False,
):
    """Compute block-diagonal cross-view consistency loss."""
    if emb1.dim() != 2 or emb2.dim() != 2:
        raise ValueError(
            f"Expected two-dimensional embeddings, got "
            f"{emb1.shape} and {emb2.shape}"
        )
    if emb1.shape[0] != emb2.shape[0]:
        raise ValueError(
            f"Embedding sizes differ: {emb1.shape[0]} and {emb2.shape[0]}"
        )
    if not isinstance(labels, torch.Tensor):
        labels = torch.tensor(labels, device=emb1.device)
    labels = labels.to(device=emb1.device).view(-1).long()

    sorted_index = torch.argsort(labels)
    first = F.normalize(emb1[sorted_index], p=2, dim=1)
    second = F.normalize(emb2[sorted_index], p=2, dim=1)
    sorted_labels = labels[sorted_index]
    similarity = torch.mm(first, second.t())
    if temperature is not None:
        similarity = similarity / max(float(temperature), 1e-6)
    same_cluster = (
        sorted_labels.unsqueeze(0) == sorted_labels.unsqueeze(1)
    )

    if balance_classes:
        intra_terms = []
        inter_terms = []
        for label in torch.unique(sorted_labels):
            current = (sorted_labels == label).nonzero(
                as_tuple=False
            ).view(-1)
            if current.numel() == 0:
                continue
            diagonal_block = similarity.index_select(
                0,
                current,
            ).index_select(1, current)
            intra_terms.append((diagonal_block - 1.0).pow(2).mean())

            other_mask = sorted_labels != label
            if other_mask.any():
                other = other_mask.nonzero(as_tuple=False).view(-1)
                off_block = similarity.index_select(
                    0,
                    current,
                ).index_select(1, other)
                inter_terms.append(off_block.pow(2).mean())

        zero = similarity.sum() * 0.0
        loss_intra = (
            torch.stack(intra_terms).mean() if intra_terms else zero
        )
        loss_inter = (
            torch.stack(inter_terms).mean() if inter_terms else zero
        )
    else:
        zero = similarity.sum() * 0.0
        loss_intra = (
            (similarity[same_cluster] - 1.0).pow(2).mean()
            if same_cluster.any()
            else zero
        )
        loss_inter = (
            similarity[~same_cluster].pow(2).mean()
            if (~same_cluster).any()
            else zero
        )

    loss = intra_weight * loss_intra + inter_weight * loss_inter
    if return_parts:
        return loss, (loss_intra.detach(), loss_inter.detach())
    return loss


def boundary_aware_regularization_loss(
    embedding,
    graph_nei,
    graph_neg,
    pos_weight=None,
):
    """Regularize neighbor and non-neighbor embedding similarities."""
    normalized = F.normalize(embedding, p=2, dim=1)
    similarity = torch.sigmoid(torch.matmul(normalized, normalized.t()))
    if pos_weight is None:
        positive_weight = graph_nei
    else:
        positive_weight = pos_weight.to(embedding.device) * graph_nei

    neighbor_loss = (
        -torch.log(similarity + 1e-10) * positive_weight
    )
    negative_loss = (
        -torch.log(1.0 - similarity + 1e-10) * graph_neg
    )
    denominator = (
        positive_weight.sum() + graph_neg.sum() + 1e-10
    )
    return (neighbor_loss.sum() + negative_loss.sum()) / denominator


def build_boundary_aware_pos_weight(
    expression,
    graph_nei,
    clip_min=0.0,
    gamma=1.0,
    smooth_eps=1e-8,
):
    """Build expression-guided weights for spatial neighbor edges."""
    if sp.issparse(expression):
        expression = expression.toarray()
    expression = np.asarray(expression, dtype=np.float32)

    similarity = np.maximum(
        cosine_similarity(expression),
        clip_min,
    )
    minimum = similarity.min()
    maximum = similarity.max()
    similarity = (
        (similarity - minimum)
        / (maximum - minimum + smooth_eps)
    )
    if gamma is not None and gamma != 1.0:
        similarity = np.power(similarity, gamma)

    if torch.is_tensor(graph_nei):
        adjacency = graph_nei.detach().cpu().numpy()
    else:
        adjacency = np.asarray(graph_nei)
    positive_weight = similarity * adjacency
    np.fill_diagonal(positive_weight, 0.0)

    return positive_weight.astype(np.float32)


def build_boundary_aware_sadj(
    spatial_adjacency,
    positive_weight,
    mix_alpha=0.5,
    sym_mode="max",
    min_weight=1e-4,
):
    """Blend the spatial graph with boundary-aware edge weights."""
    if not sp.issparse(spatial_adjacency):
        spatial_adjacency = sp.csr_matrix(
            spatial_adjacency,
            dtype=np.float32,
        )
    else:
        spatial_adjacency = spatial_adjacency.tocsr().astype(
            np.float32
        )

    base = spatial_adjacency.toarray().astype(np.float32)
    weights = np.asarray(positive_weight, dtype=np.float32)
    refined = (
        (1.0 - mix_alpha) * base
        + mix_alpha * (base * weights)
    )
    refined[refined < min_weight] = 0.0
    np.fill_diagonal(refined, 0.0)

    if sym_mode == "max":
        refined = np.maximum(refined, refined.T)
    elif sym_mode == "mean":
        refined = 0.5 * (refined + refined.T)
    else:
        raise ValueError(f"Unknown symmetrization mode: {sym_mode}")

    return sp.coo_matrix(refined, dtype=np.float32)


def build_precluster_confidence(expression, labels, eps=1e-8):
    """Estimate spot confidence from expression and precluster labels."""
    if sp.issparse(expression):
        expression = expression.toarray()
    expression = np.asarray(expression, dtype=np.float32)
    labels = np.asarray(labels)
    similarity = cosine_similarity(expression)
    confidence = np.zeros(expression.shape[0], dtype=np.float32)

    for index in range(expression.shape[0]):
        same = np.where(labels == labels[index])[0]
        different = np.where(labels != labels[index])[0]
        same = same[same != index]
        within_mean = (
            similarity[index, same].mean() if len(same) else 0.0
        )
        between_mean = (
            similarity[index, different].mean()
            if len(different)
            else 0.0
        )
        confidence[index] = within_mean - between_mean

    minimum = confidence.min()
    maximum = confidence.max()
    confidence = (
        (confidence - minimum) / (maximum - minimum + eps)
    )
    return confidence.astype(np.float32)


def refine_feature_graph_with_soft_prior(
    feature_adjacency,
    precluster_labels,
    confidence,
    beta=0.10,
    sym_mode="max",
    min_weight=1e-6,
):
    """Refine the feature graph with confidence-weighted priors."""
    if not sp.issparse(feature_adjacency):
        feature_adjacency = sp.csr_matrix(
            feature_adjacency,
            dtype=np.float32,
        )
    else:
        feature_adjacency = feature_adjacency.tocsr().astype(
            np.float32
        )

    adjacency = feature_adjacency.toarray().astype(np.float32)
    labels = np.asarray(precluster_labels)
    confidence = np.asarray(confidence, dtype=np.float32)
    same_cluster = (
        labels[:, None] == labels[None, :]
    ).astype(np.float32)
    pair_confidence = np.sqrt(np.outer(confidence, confidence))
    weights = np.where(
        same_cluster > 0,
        1.0 + beta * pair_confidence,
        1.0 - beta * pair_confidence,
    ).astype(np.float32)

    refined = adjacency * weights
    refined[adjacency <= 0] = 0.0
    refined[refined < min_weight] = 0.0
    np.fill_diagonal(refined, 0.0)
    if sym_mode == "max":
        refined = np.maximum(refined, refined.T)
    elif sym_mode == "mean":
        refined = 0.5 * (refined + refined.T)
    else:
        raise ValueError(f"Unknown symmetrization mode: {sym_mode}")

    return sp.coo_matrix(refined, dtype=np.float32)
