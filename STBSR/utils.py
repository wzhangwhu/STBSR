import sklearn
import torch
import numpy as np
import scipy.sparse as sp
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import kneighbors_graph
import pandas as pd
import scanpy as sc
import torch.nn.functional as F
EPS = 1e-15

def _nan2inf(x):
    return torch.where(torch.isnan(x), torch.zeros_like(x) + np.inf, x)

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
    return torch.sparse.FloatTensor(indices, values, shape)

def spatial_construct_graph(adata, radius=560):

    coor = pd.DataFrame(adata.obsm['spatial'])
    coor.index = adata.obs.index
    coor.columns = ['imagerow', 'imagecol']
    A=np.zeros((coor.shape[0],coor.shape[0]))

    nbrs = sklearn.neighbors.NearestNeighbors(radius=radius).fit(coor)
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


def features_construct_graph(features, k=15, pca=None, mode="connectivity", metric="cosine"):
    print("start features construct graph")
    if pca is not None:
        features = dopca(features, dim=pca).reshape(-1, 1)
    A = kneighbors_graph(features, k + 1, mode=mode,metric=metric,include_self=True)
    A = A.toarray()
    row, col = np.diag_indices_from(A)
    A[row, col] = 0
    fadj = sp.coo_matrix(A, dtype=np.float32)
    fadj = fadj + fadj.T.multiply(fadj.T > fadj) - fadj.multiply(fadj.T > fadj)
    return fadj

def get_adj(data, pca=None, k=25, mode="connectivity", metric="cosine"):
    if pca is not None:
        data = dopca(data, dim=pca)
        data = data.reshape(-1, 1)
    A = kneighbors_graph(data, k, mode=mode, metric=metric, include_self=True)
    adj = A.toarray()
    adj_n = norm_adj(adj)
    return adj, adj_n

def norm_adj(A):
    normalized_D = degree_power(A, -0.5)
    output = normalized_D.dot(A).dot(normalized_D)
    return output

def dopca(data, dim=50):
    return PCA(n_components=dim).fit_transform(data)

def degree_power(A, k):
    degrees = np.power(np.array(A.sum(1)), k).flatten()
    degrees[np.isinf(degrees)] = 0.
    if sp.issparse(A):
        D = sp.diags(degrees)
    else:
        D = np.diag(degrees)
    return D

class NB(object):
    def __init__(self, theta=None, scale_factor=1.0):
        super(NB, self).__init__()
        self.eps = 1e-10
        self.scale_factor = scale_factor
        self.theta = theta

    def loss(self, y_true, y_pred, mean=True):
        y_pred = y_pred * self.scale_factor
        theta = torch.minimum(self.theta, torch.tensor(1e6))
        t1 = torch.lgamma(theta + self.eps) + torch.lgamma(y_true + 1.0) - torch.lgamma(y_true + theta + self.eps)
        t2 = (theta + y_true) * torch.log(1.0 + (y_pred / (theta + self.eps))) + (
                y_true * (torch.log(theta + self.eps) - torch.log(y_pred + self.eps)))
        final = t1 + t2
        final = _nan2inf(final)
        if mean:
            final = torch.mean(final)
        return final


class ZINB(NB):
    def __init__(self, pi, ridge_lambda=0.0, **kwargs):
        super().__init__(**kwargs)
        self.pi = pi
        self.ridge_lambda = ridge_lambda

    def loss(self, y_true, y_pred, mean=True):
        scale_factor = self.scale_factor
        eps = self.eps
        theta = torch.minimum(self.theta, torch.tensor(1e6))
        nb_case = super().loss(y_true, y_pred, mean=False) - torch.log(1.0 - self.pi + eps)
        y_pred = y_pred * scale_factor
        zero_nb = torch.pow(theta / (theta + y_pred + eps), theta)
        zero_case = -torch.log(self.pi + ((1.0 - self.pi) * zero_nb) + eps)
        result = torch.where(torch.lt(y_true, 1e-8), zero_case, nb_case)
        ridge = self.ridge_lambda * torch.square(self.pi)
        result += ridge
        if mean:
            result = torch.mean(result)
        result = _nan2inf(result)
        return result

class Colors():
    colors = [
        "#1f77b4",
        "#ff7f0e",
        "#279e68",
        "#d62728",
        "#633194",
        "#8c564b",
        "#F73BAD",
        "#ad494a",
        "#F6E800",
        "#01F7F7",
        "#aec7e8",
        "#ffbb78",
        "#98df8a",
        "#ff9896",
        "#c5b0d5",
        "#c49c94",
        "#f7b6d2",
        "#dbdb8d",
        "#9edae5",
        "#8c6d31"]

def res_search_fixed_clus(cluster_type, adata, fixed_clus_count, increment=0.01):
    '''
                arg1(adata)[AnnData matrix]
                arg2(fixed_clus_count)[int]

                return:
                    resolution[int]
            '''
    if cluster_type == 'leiden':
        for res in sorted(list(np.arange(0.14, 2.5, increment))):#, reverse=True):
            sc.tl.leiden(adata, random_state=0, resolution=res)
            count_unique_leiden = len(pd.DataFrame(adata.obs['leiden']).leiden.unique())
            # print(res,' ' , count_unique_leiden)
            if count_unique_leiden == fixed_clus_count:
                cluster_labels = np.array(adata.obs['leiden'])
                flag=0
                break
            if count_unique_leiden > fixed_clus_count:
                cluster_labels = np.array(adata.obs['leiden'])
                flag=1
                break
    elif cluster_type == 'louvain':
        for res in sorted(list(np.arange(0.14, 2.5, increment))):#, reverse=True):
            sc.tl.louvain(adata, random_state=0, resolution=res)
            count_unique_louvain = len(pd.DataFrame(adata.obs['louvain']).louvain.unique())
            # print(res,' ' , count_unique_louvain)
            if count_unique_louvain == fixed_clus_count:
                cluster_labels = np.array(adata.obs['louvain'])
                flag = 0
                break
            if count_unique_louvain > fixed_clus_count:
                cluster_labels = np.array(adata.obs['louvain'])
                flag = 1
                break
    return cluster_labels,flag


def dicr_loss_little_block_diagonal_precluster(
    emb1: torch.Tensor,
    emb2: torch.Tensor,
    labels,
    *,
    intra_weight: float = 1.0,
    inter_weight: float = 1.0,
    temperature: float = 1.0,
    balance_classes: bool = True,
    eps: float = 1e-12,
    return_parts: bool = False,
):
    """
    基于预聚类标签的“按类分块对角化”跨视图相关损失（你训练里叫 CCR 的那项）
    目标：
      - 同类(对角块)相似度 -> 1
      - 异类(非对角块)相似度 -> 0
    S 是 N×N spot-spot 跨视图相似度矩阵：
      S[i,j] = cos(emb1_i, emb2_j)
    target 是由 labels 定义的 block diagonal：
      target[i,j] = 1 if label_i == label_j else 0
    """
    if emb1.dim() != 2 or emb2.dim() != 2:
        raise ValueError(f"emb1/emb2 must be 2D (N,D); got {emb1.shape} and {emb2.shape}")
    if emb1.shape[0] != emb2.shape[0]:
        raise ValueError(f"emb1 and emb2 must have same N; got {emb1.shape[0]} vs {emb2.shape[0]}")
    # ---- labels -> tensor on device ----
    if not isinstance(labels, torch.Tensor):
        labels = torch.tensor(labels, device=emb1.device)
    labels = labels.to(device=emb1.device)
    if labels.dim() > 1:
        labels = labels.view(-1)
    labels = labels.long()
    # ---- sort by labels so blocks are explicit ----
    sorted_idx = torch.argsort(labels)
    z1 = emb1[sorted_idx]
    z2 = emb2[sorted_idx]
    y  = labels[sorted_idx]
    # ---- N×N cross-view similarity ----
    z1 = F.normalize(z1, p=2, dim=1)
    z2 = F.normalize(z2, p=2, dim=1)
    S = torch.mm(z1, z2.t())
    if temperature is not None:
        S = S / max(float(temperature), 1e-6)
    # ---- within-class mask ----
    same = (y.unsqueeze(0) == y.unsqueeze(1))
    # ---- loss terms ----
    if balance_classes:
        uniq = torch.unique(y)
        intra_terms = []
        inter_terms = []

        for k in uniq:
            idx_k = (y == k).nonzero(as_tuple=False).view(-1)
            if idx_k.numel() == 0:
                continue

            # class-k diagonal block
            Skk = S.index_select(0, idx_k).index_select(1, idx_k)
            intra_terms.append((Skk - 1.0).pow(2).mean())

            # class-k vs all others (off-block)
            other = (y != k)
            if other.any():
                idx_o = other.nonzero(as_tuple=False).view(-1)
                Sko = S.index_select(0, idx_k).index_select(1, idx_o)
                inter_terms.append(Sko.pow(2).mean())

        zero = S.sum() * 0.0
        loss_intra = torch.stack(intra_terms).mean() if len(intra_terms) else zero
        loss_inter = torch.stack(inter_terms).mean() if len(inter_terms) else zero
    else:
        # global mean over all pairs
        loss_intra = (S[same] - 1.0).pow(2).mean() if same.any() else S.sum() * 0.0
        loss_inter = S[~same].pow(2).mean() if (~same).any() else S.sum() * 0.0

    loss = intra_weight * loss_intra + inter_weight * loss_inter
    if return_parts:
        return loss, (loss_intra.detach(), loss_inter.detach())
    return loss

def boundary_aware_regularization_loss(emb, graph_nei, graph_neg, pos_weight=None):
    """
    emb: [N, d]
    graph_nei / graph_neg: [N, N]
    pos_weight: [N, N]，邻居拉近项的权重
    """
    z = F.normalize(emb, p=2, dim=1)
    sim = torch.matmul(z, z.t())
    sim = torch.sigmoid(sim)

    if pos_weight is None:
        pos_weight = graph_nei
    else:
        pos_weight = pos_weight.to(emb.device) * graph_nei

    neigh_loss = -torch.log(sim + 1e-10) * pos_weight
    neg_loss   = -torch.log(1 - sim + 1e-10) * graph_neg

    denom = pos_weight.sum() + graph_neg.sum() + 1e-10
    return (neigh_loss.sum() + neg_loss.sum()) / denom


def build_boundary_aware_pos_weight(
        X,
        graph_nei,
        clip_min=0.0,
        gamma=1.0,
        smooth_eps=1e-8
):
    """
    基于表达相似度构建边界感知邻居权重矩阵
    X         : [N, G]
    graph_nei : [N, N] 二值邻接（numpy 或 torch）
    clip_min  : 负相似截断
    gamma     : 指数强化系数
    返回:
        pos_weight: [N, N]，仅邻居位置非零，范围约 [0,1]
    """
    if sp.issparse(X):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float32)

    # 表达余弦相似度
    S = cosine_similarity(X)   # [-1, 1]
    S = np.maximum(S, clip_min)

    # 归一化到 [0,1]
    s_min, s_max = S.min(), S.max()
    S = (S - s_min) / (s_max - s_min + smooth_eps)

    # 指数强化
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
        sym_mode="max",
        min_weight=1e-4
):
    """
    用边界感知权重重构空间传播图
    sadj       : 原始空间图 (scipy sparse, 可二值/可加权)
    pos_weight : [N,N] 的边界感知权重
    mix_alpha  : 原图与边界权重融合比例
                0.0 -> 完全原图
                1.0 -> 完全边界感知图
    """
    if not sp.issparse(sadj):
        sadj = sp.csr_matrix(sadj, dtype=np.float32)
    else:
        sadj = sadj.tocsr().astype(np.float32)

    A_base = sadj.toarray().astype(np.float32)
    W = np.asarray(pos_weight, dtype=np.float32)

    # 若原图是二值图，则 A_base 是 0/1
    # 若原图已有权重，则相当于在原权重上做边界调制
    A_new = (1.0 - mix_alpha) * A_base + mix_alpha * (A_base * W)

    # 去掉极小边
    A_new[A_new < min_weight] = 0.0
    np.fill_diagonal(A_new, 0.0)

    if sym_mode == "max":
        A_new = np.maximum(A_new, A_new.T)
    elif sym_mode == "mean":
        A_new = 0.5 * (A_new + A_new.T)
    else:
        raise ValueError(f"unknown sym_mode: {sym_mode}")

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
    """
    基于表达相似度 + 预聚类标签，给每个 spot 一个置信度 c_i ∈ [0,1]
    思想：
      - 与同簇点平均更相似 -> 置信度更高
      - 与异簇点平均更相似 -> 置信度更低
    """
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

        # 差值越大越可信
        conf[i] = same_mean - diff_mean

    # 归一化到 [0,1]
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
        sym_mode="max",
        min_weight=1e-6
):
    """
    预聚类软先验引导的特征图重加权

    fadj              : 原始 feature graph (scipy sparse)
    precluster_labels : 预聚类标签
    confidence        : 每个 spot 的置信度 c_i ∈ [0,1]
    beta              : 软重加权强度，建议 0.05~0.15
    """
    if not sp.issparse(fadj):
        fadj = sp.csr_matrix(fadj, dtype=np.float32)
    else:
        fadj = fadj.tocsr().astype(np.float32)

    A = fadj.toarray().astype(np.float32)
    labels = np.asarray(precluster_labels)
    conf = np.asarray(confidence, dtype=np.float32)

    same = (labels[:, None] == labels[None, :]).astype(np.float32)
    conf_pair = np.sqrt(np.outer(conf, conf))  # 更平滑一点

    # 同簇：1 + beta*c
    # 异簇：1 - beta*c
    W = np.where(
        same > 0,
        1.0 + beta * conf_pair,
        1.0 - beta * conf_pair
    ).astype(np.float32)

    A_new = A * W
    A_new[A <= 0] = 0.0
    A_new[A_new < min_weight] = 0.0
    np.fill_diagonal(A_new, 0.0)

    if sym_mode == "max":
        A_new = np.maximum(A_new, A_new.T)
    elif sym_mode == "mean":
        A_new = 0.5 * (A_new + A_new.T)
    else:
        raise ValueError(f"unknown sym_mode: {sym_mode}")

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

def build_boundary_score_from_expression(X, graph_nei, eps=1e-8):
    """
    用表达差异估计每个 spot 的边界分数，范围 [0,1]
    直觉：
      - 与空间邻居表达差异大 -> 更像边界点
      - 与空间邻居表达差异小 -> 更像内部点
    """
    if sp.issparse(X):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float32)

    if hasattr(graph_nei, "cpu"):
        A = graph_nei.detach().cpu().numpy()
    else:
        A = np.asarray(graph_nei)

    n = X.shape[0]
    score = np.zeros(n, dtype=np.float32)

    for i in range(n):
        nbr_idx = np.where(A[i] > 0)[0]
        nbr_idx = nbr_idx[nbr_idx != i]
        if len(nbr_idx) == 0:
            score[i] = 0.0
            continue

        diff = X[nbr_idx] - X[i]
        dist = np.sqrt((diff * diff).sum(axis=1))
        score[i] = dist.mean()

    smin, smax = score.min(), score.max()
    score = (score - smin) / (smax - smin + eps)

    print("boundary score | min=%.4f mean=%.4f max=%.4f" % (
        score.min(), score.mean(), score.max()
    ))
    return score.astype(np.float32)
