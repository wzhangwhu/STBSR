"""Spatial and expression graph utilities used by STBSR."""
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors, kneighbors_graph
from torch.nn import functional as F


def normalize_sparse_matrix(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(indices, values, shape)


def spatial_construct_graph(adata, radius=560):

    coor = pd.DataFrame(adata.obsm['spatial'])
    coor.index = adata.obs.index
    coor.columns = ['imagerow', 'imagecol']
    A=np.zeros((coor.shape[0],coor.shape[0]))

    nbrs = NearestNeighbors(radius=radius).fit(coor)
    distances, indices = nbrs.radius_neighbors(coor, return_distance=True)

    for it in range(indices.shape[0]):
        A[[it] * indices[it].shape[0], indices[it]]=1

    print('The graph contains %d edges, %d cells.' % (sum(sum(A)), adata.n_obs))
    print('%.4f neighbors per cell on average.' % (sum(sum(A)) / adata.n_obs))

    graph_nei = torch.from_numpy(A)

    graph_neg = torch.ones(coor.shape[0],coor.shape[0]) - graph_nei

    sadj = sp.coo_matrix(A, dtype=np.float32)
    sadj = sadj + sadj.T.multiply(sadj.T > sadj) - sadj.multiply(sadj.T > sadj)
    return sadj, graph_nei, graph_neg


def features_construct_graph(features, k=15):
    print("start features construct graph")
    A = kneighbors_graph(features, k + 1, mode="connectivity", metric="cosine", include_self=True)
    A = A.toarray()
    row, col = np.diag_indices_from(A)
    A[row, col] = 0
    fadj = sp.coo_matrix(A, dtype=np.float32)
    fadj = fadj + fadj.T.multiply(fadj.T > fadj) - fadj.multiply(fadj.T > fadj)
    return fadj


def build_boundary_aware_pos_weight(
        X,
        graph_nei,
        clip_min=0.0,
        gamma=1.0,
        smooth_eps=1e-8
):
    """Weight local spatial edges by normalized expression cosine similarity."""
    if sp.issparse(X):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float32)

    S = cosine_similarity(X)   # [-1, 1]
    S = np.maximum(S, clip_min)

    s_min, s_max = S.min(), S.max()
    S = (S - s_min) / (s_max - s_min + smooth_eps)

    if gamma is not None and gamma != 1.0:
        S = np.power(S, gamma)

    if torch.is_tensor(graph_nei):
        A = graph_nei.detach().cpu().numpy()
    else:
        A = np.asarray(graph_nei)

    pos_weight = S * A
    np.fill_diagonal(pos_weight, 0.0)

    nz = pos_weight[pos_weight > 0]
    if nz.size > 0:
        print(
            "pos_weight stats | min=%.6f mean=%.6f max=%.6f | p10=%.6f p50=%.6f p90=%.6f"
            % (
                nz.min(), nz.mean(), nz.max(),
                np.percentile(nz, 10),
                np.percentile(nz, 50),
                np.percentile(nz, 90)
            )
        )
    else:
        print("pos_weight stats | no positive edges")

    return pos_weight.astype(np.float32)


def build_boundary_aware_sadj(
        sadj,
        pos_weight,
        mix_alpha=0.5,
        min_weight=1e-4
):
    """Blend spatial adjacency with its expression-weighted counterpart."""
    if not sp.issparse(sadj):
        sadj = sp.csr_matrix(sadj, dtype=np.float32)
    else:
        sadj = sadj.tocsr().astype(np.float32)

    A_base = sadj.toarray().astype(np.float32)
    W = np.asarray(pos_weight, dtype=np.float32)

    A_new = (1.0 - mix_alpha) * A_base + mix_alpha * (A_base * W)

    A_new[A_new < min_weight] = 0.0
    np.fill_diagonal(A_new, 0.0)

    A_new = np.maximum(A_new, A_new.T)

    sadj_new = sp.coo_matrix(A_new, dtype=np.float32)
    nnz_bin = (A_new > 0).sum()
    print('boundary-aware sadj contains %d edges, %d cells.' % (nnz_bin, A_new.shape[0]))
    print('%.4f neighbors per cell on average.' % (nnz_bin / A_new.shape[0]))

    nz = A_new[A_new > 0]
    if nz.size > 0:
        print(
            'boundary-aware sadj weight stats | min=%.6f mean=%.6f max=%.6f'
            % (nz.min(), nz.mean(), nz.max())
        )

    return sadj_new


def build_precluster_confidence(X, labels, eps=1e-8):
    """Estimate confidence from within- and between-precluster expression similarity."""
    if sp.issparse(X):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float32)
    labels = np.asarray(labels)

    S = cosine_similarity(X)  # [-1, 1]
    n = X.shape[0]
    conf = np.zeros(n, dtype=np.float32)

    for i in range(n):
        same_idx = np.where(labels == labels[i])[0]
        diff_idx = np.where(labels != labels[i])[0]

        same_idx = same_idx[same_idx != i]

        same_mean = S[i, same_idx].mean() if len(same_idx) > 0 else 0.0
        diff_mean = S[i, diff_idx].mean() if len(diff_idx) > 0 else 0.0

        conf[i] = same_mean - diff_mean

    cmin, cmax = conf.min(), conf.max()
    conf = (conf - cmin) / (cmax - cmin + eps)

    print("precluster confidence | min=%.4f mean=%.4f max=%.4f" % (
        conf.min(), conf.mean(), conf.max()
    ))
    return conf.astype(np.float32)


def refine_feature_graph_with_soft_prior(
        fadj,
        precluster_labels,
        confidence,
        beta=0.10,
        min_weight=1e-6
):
    """Strengthen concordant feature edges using pairwise precluster confidence."""
    if not sp.issparse(fadj):
        fadj = sp.csr_matrix(fadj, dtype=np.float32)
    else:
        fadj = fadj.tocsr().astype(np.float32)

    A = fadj.toarray().astype(np.float32)
    labels = np.asarray(precluster_labels)
    conf = np.asarray(confidence, dtype=np.float32)

    same = (labels[:, None] == labels[None, :]).astype(np.float32)
    conf_pair = np.sqrt(np.outer(conf, conf))

    W = np.where(
        same > 0,
        1.0 + beta * conf_pair,
        1.0 - beta * conf_pair
    ).astype(np.float32)

    A_new = A * W
    A_new[A <= 0] = 0.0
    A_new[A_new < min_weight] = 0.0
    np.fill_diagonal(A_new, 0.0)

    A_new = np.maximum(A_new, A_new.T)

    fadj_new = sp.coo_matrix(A_new, dtype=np.float32)

    nz = A_new[A_new > 0]
    if nz.size > 0:
        print(
            "refined fadj | edges=%d avg_deg=%.4f | min=%.6f mean=%.6f max=%.6f"
            % ((A_new > 0).sum(), (A_new > 0).sum() / A_new.shape[0], nz.min(), nz.mean(), nz.max())
        )
    else:
        print("refined fadj | no edges")

    return fadj_new


def cross_view_consistency(spatial, feature, labels):
    """Average class-balanced cosine errors within and between preclusters."""
    if spatial.ndim != 2 or spatial.shape != feature.shape:
        raise ValueError("Cross-view embeddings must have the same two-dimensional shape.")
    labels = torch.as_tensor(labels, device=spatial.device).reshape(-1).long()
    order = torch.argsort(labels)
    grouped = labels[order]
    spatial = F.normalize(spatial[order], p=2, dim=1)
    feature = F.normalize(feature[order], p=2, dim=1)
    similarity = torch.mm(spatial, feature.t())
    within, between = [], []
    for group in torch.unique(grouped):
        members = (grouped == group).nonzero(as_tuple=False).view(-1)
        rows = similarity.index_select(0, members)
        within.append((rows.index_select(1, members) - 1.0).pow(2).mean())
        other = (grouped != group).nonzero(as_tuple=False).view(-1)
        if other.numel():
            between.append(rows.index_select(1, other).pow(2).mean())
    between_loss = torch.stack(between).mean() if between else similarity.sum() * 0.0
    return torch.stack(within).mean() + between_loss
