"""Unified clustering and spatial-boundary metrics for STBSR."""

from __future__ import annotations

from collections import deque

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.neighbors import NearestNeighbors


def valid_label_mask(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    gt_s = pd.Series(gt)
    pred_s = pd.Series(pred)
    return (~gt_s.isna() & ~pred_s.isna()).to_numpy()


def hungarian_acc(y_true, y_pred) -> float:
    y_true = np.asarray(y_true).astype(str)
    y_pred = np.asarray(y_pred).astype(str)
    mask = valid_label_mask(y_true, y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if y_true.size == 0:
        return np.nan
    cm = pd.crosstab(pd.Series(y_true), pd.Series(y_pred)).values
    row_ind, col_ind = linear_sum_assignment(cm, maximize=True)
    return float(cm[row_ind, col_ind].sum() / y_true.size)


def purity_score(y_true, y_pred) -> float:
    cm = pd.crosstab(pd.Series(y_true), pd.Series(y_pred)).values
    if cm.size == 0 or cm.sum() == 0:
        return np.nan
    return float(np.max(cm, axis=0).sum() / cm.sum())


def entropy_score(y_true, y_pred) -> float:
    cm = pd.crosstab(pd.Series(y_true), pd.Series(y_pred)).values.astype(float)
    if cm.size == 0 or cm.sum() == 0:
        return np.nan
    total = cm.sum()
    ent = 0.0
    for cluster in cm.T:
        cluster_total = cluster.sum()
        if cluster_total == 0:
            continue
        p = cluster / cluster_total
        ent += (-np.sum(p * np.log2(p + 1e-12))) * (cluster_total / total)
    return float(ent)


def build_spatial_edges(coords: np.ndarray, n_neighbors: int = 6) -> list[tuple[int, int]]:
    coords = np.asarray(coords, dtype=float)[:, :2]
    nn = NearestNeighbors(n_neighbors=min(n_neighbors + 1, coords.shape[0]), metric="euclidean")
    nn.fit(coords)
    _, inds = nn.kneighbors(coords)
    edges = set()
    for i in range(coords.shape[0]):
        for j in inds[i, 1:]:
            a, b = (int(i), int(j)) if i < j else (int(j), int(i))
            edges.add((a, b))
    return sorted(edges)


def edges_to_adjlist(n: int, edges: list[tuple[int, int]]) -> list[list[int]]:
    adj = [[] for _ in range(n)]
    for i, j in edges:
        adj[i].append(j)
        adj[j].append(i)
    return adj


def boundary_edges_from_labels(edges: list[tuple[int, int]], labels: np.ndarray) -> set[tuple[int, int]]:
    labels = np.asarray(labels).astype(str)
    return {edge for edge in edges if labels[edge[0]] != labels[edge[1]]}


def boundary_spots(boundary_edges: set[tuple[int, int]]) -> np.ndarray:
    spots = set()
    for i, j in boundary_edges:
        spots.add(i)
        spots.add(j)
    return np.array(sorted(spots), dtype=int)


def expand_by_hops(seed_spots: np.ndarray, adj: list[list[int]], hops: int = 1) -> np.ndarray:
    visited = set(map(int, seed_spots))
    queue = deque((int(i), 0) for i in seed_spots)
    while queue:
        node, depth = queue.popleft()
        if depth >= hops:
            continue
        for nxt in adj[node]:
            if nxt not in visited:
                visited.add(nxt)
                queue.append((nxt, depth + 1))
    return np.array(sorted(visited), dtype=int)


def map_pred_to_gt(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    gt = np.asarray(gt).astype(str)
    pred = np.asarray(pred).astype(str)
    gt_classes = np.array(sorted(pd.unique(gt)))
    pred_classes = np.array(sorted(pd.unique(pred)))
    conf = np.zeros((len(pred_classes), len(gt_classes)), dtype=int)
    gt_idx = {c: i for i, c in enumerate(gt_classes)}
    pred_idx = {c: i for i, c in enumerate(pred_classes)}
    for g, p in zip(gt, pred):
        conf[pred_idx[p], gt_idx[g]] += 1
    row_ind, col_ind = linear_sum_assignment(-conf)
    mapping = {pred_classes[r]: gt_classes[c] for r, c in zip(row_ind, col_ind)}
    for pred_class in pred_classes:
        mapping.setdefault(pred_class, gt_classes[int(np.argmax(conf[pred_idx[pred_class]]))])
    return np.array([mapping[p] for p in pred], dtype=str)


def boundary_metrics(gt: np.ndarray, pred: np.ndarray, coords: np.ndarray, n_neighbors: int = 6) -> dict:
    gt = np.asarray(gt).astype(str)
    pred = np.asarray(pred).astype(str)
    edges = build_spatial_edges(coords, n_neighbors=n_neighbors)
    adj = edges_to_adjlist(len(gt), edges)
    gt_b_edges = boundary_edges_from_labels(edges, gt)
    pred_b_edges = boundary_edges_from_labels(edges, pred)
    inter = len(gt_b_edges & pred_b_edges)
    union = len(gt_b_edges | pred_b_edges)
    precision = inter / len(pred_b_edges) if pred_b_edges else 0.0
    recall = inter / len(gt_b_edges) if gt_b_edges else 0.0
    gt_b_spots = boundary_spots(gt_b_edges)
    band = expand_by_hops(gt_b_spots, adj, hops=1)
    mapped_pred = map_pred_to_gt(gt, pred)
    return {
        "BIoU": float(inter / union) if union else 0.0,
        "BF": float(2 * precision * recall / (precision + recall)) if precision + recall else 0.0,
        "BandAcc": float(np.mean(mapped_pred[band] == gt[band])) if len(band) else np.nan,
        "BoundaryPrecision": float(precision),
        "BoundaryRecall": float(recall),
        "BoundaryEdgesGT": int(len(gt_b_edges)),
        "BoundaryEdgesPred": int(len(pred_b_edges)),
        "BoundaryBandSpots": int(len(band)),
    }


def moran_i_numeric(x: np.ndarray, edges: list[tuple[int, int]]) -> float:
    x = np.asarray(x, dtype=float)
    n = x.size
    z = x - np.mean(x)
    denom = np.sum(z * z)
    if n < 3 or denom <= 0 or not edges:
        return np.nan
    num = 0.0
    for i, j in edges:
        num += 2.0 * z[i] * z[j]
    w_sum = 2.0 * len(edges)
    return float((n / w_sum) * (num / denom))


def categorical_moran_score(labels: np.ndarray, coords: np.ndarray, n_neighbors: int = 6) -> float:
    """Prevalence-weighted Moran's I over one-hot predicted domains."""
    labels = np.asarray(labels).astype(str)
    edges = build_spatial_edges(coords, n_neighbors=n_neighbors)
    scores = []
    weights = []
    for lab in sorted(pd.unique(labels)):
        x = (labels == lab).astype(float)
        score = moran_i_numeric(x, edges)
        if np.isfinite(score):
            scores.append(score)
            weights.append(float(x.mean()))
    if not scores:
        return np.nan
    return float(np.average(scores, weights=np.asarray(weights)))


def compute_all_metrics(gt: np.ndarray, pred: np.ndarray, coords: np.ndarray, n_neighbors: int = 6) -> dict:
    gt = np.asarray(gt).astype(str)
    pred = np.asarray(pred).astype(str)
    mask = valid_label_mask(gt, pred)
    gt = gt[mask]
    pred = pred[mask]
    coords = np.asarray(coords)[mask, :2]
    out = {
        "ARI": float(adjusted_rand_score(gt, pred)) if gt.size else np.nan,
        "NMI": float(normalized_mutual_info_score(gt, pred)) if gt.size else np.nan,
        "ACC": hungarian_acc(gt, pred),
        "Purity": purity_score(gt, pred),
        "Entropy": entropy_score(gt, pred),
        "MoransScore": categorical_moran_score(pred, coords, n_neighbors=n_neighbors),
    }
    out.update(boundary_metrics(gt, pred, coords, n_neighbors=n_neighbors))
    return out

