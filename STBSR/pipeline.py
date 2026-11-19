"""Frozen joint STBSR model and two clustering readouts."""
import os

for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[variable] = "1"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import copy
import random
import time
from types import SimpleNamespace

import numpy as np
import scipy.sparse as sp
import torch
from sklearn.cluster import KMeans
from torch.nn import functional as F

from . import utils
from .models import STBSR
from .expression_pairs import expression_views, trusted_pairs
from .spatial_state import SpatialGraphState
from .stopping import Plateau


class JointSTBSR(STBSR):
    def __init__(self, genes, spatial_attention_gain):
        super().__init__(genes, 128, 64, 0.0)
        self.spatial_attention_gain = spatial_attention_gain

    def forward(self, expression, spatial_graph, expression_graph):
        spatial = self.SGCN(expression, spatial_graph)
        feature = self.FGCN(expression, expression_graph)
        meta = self._get_meta(expression.device)
        mixed = self.CGCN(expression, meta * expression_graph + (1 - meta) * spatial_graph)
        branches = torch.stack([spatial, mixed, feature], dim=1)
        gains = branches.new_tensor([self.spatial_attention_gain, 1., 1.]).reshape(1, 3, 1)
        attention = F.normalize(self.MLP_L(branches) * gains, p=2, dim=1)
        fused = torch.cat([attention[:, index] * branch for index, branch in enumerate((spatial, mixed, feature))], dim=1)
        embedding = self.proj3(fused)
        pi, dispersion, mean = self.ZINB(embedding)
        return embedding, pi, dispersion, mean, spatial, feature, mixed


def relation_arrays(data, coordinates):
    views, nearby = expression_views(SimpleNamespace(X=data.x, obsm={"spatial": coordinates}))
    trusted = trusted_pairs(views, nearby)
    near = data.graph_neighbors.cpu().numpy().astype(bool)
    away = data.graph_non_neighbors.cpu().numpy().astype(bool)
    positive, negative = (np.zeros_like(near) for _ in range(2))
    row, col = trusted["pos"]
    positive[row, col] = positive[col, row] = True
    nr, nc = trusted["neg"]
    negative[nr, nc] = negative[nc, nr] = True
    dissimilar = np.ones_like(near)
    for view in views:
        cosine = view @ view.T
        median = np.median(cosine, axis=1)
        dissimilar &= (cosine <= median[:, None]) & (cosine <= median[None, :])
    pos_weight = np.asarray(data.expression_edge_reliability, dtype=np.float32).copy()
    pos_weight[~near | dissimilar | negative] = 0.
    pos_weight[row, col] = pos_weight[col, row] = np.maximum(pos_weight[row, col], trusted["pos_w"])
    np.fill_diagonal(pos_weight, 0.)
    negative_mask = (away & dissimilar & ~positive) | negative
    np.fill_diagonal(negative_mask, False)
    assert not (negative_mask & (pos_weight > 0)).any()
    assert (pos_weight[positive] > 0).all() and negative_mask[negative].all()
    pos_mass = float(pos_weight.sum(dtype=np.float64))
    neg_count = int(negative_mask.sum())
    if pos_mass <= 0 or neg_count <= 0:
        raise ValueError("The input must provide both positive and negative relations")
    pos_weight *= .5 / pos_mass
    labels = (pos_weight > 0).astype(np.float32)
    weights = pos_weight + negative_mask.astype(np.float32) * (.5 / neg_count)
    return labels, weights, trusted["neg"]


def pair_loss(z, relations, saturate):
    labels, weights, trusted_negatives = relations
    unit = F.normalize(z, p=2, dim=1)
    logits = unit @ unit.T
    if saturate:
        row, col = trusted_negatives
        for r, c in ((row, col), (col, row)):
            values = logits[r, c]
            logits[r, c] = torch.where(values > 0, values, 0.)
    return F.binary_cross_entropy_with_logits(logits, labels, weight=weights, reduction="sum")


def scaled_cosine_error(target, reconstructed):
    valid = target.square().sum(dim=1) > 0
    if not bool(valid.any()):
        raise ValueError("SCE requires a nonzero expression row")
    a = F.normalize(target[valid], p=2, dim=1)
    b = F.normalize(reconstructed[valid], p=2, dim=1)
    return (1 - (a * b).sum(dim=1)).square().mean()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_system(data, family, cfg, seed, device):
    set_seed(seed)
    network = JointSTBSR(data.x.shape[1], cfg["spatial_attention_gain"][family]).to(device)
    graph = SpatialGraphState(data.spatial_adjacency).to(device)
    features = utils.sparse_mx_to_torch_sparse_tensor(
        utils.normalize_sparse_matrix(data.feature_adjacency + sp.eye(len(data.x)))
    ).coalesce().to(device).to_dense()
    expression = torch.as_tensor(data.x, device=device)
    return network, graph, expression, features


def make_optimizer(network, cfg):
    lr = cfg["learning_rate"]
    optimizer = torch.optim.Adam(network.parameters(), lr=lr, weight_decay=cfg["weight_decay"])
    template = dict(optimizer.param_groups[0])
    groups = {"base": [], "attention": [], "projection": []}
    for name, parameter in network.named_parameters():
        group = "attention" if name.startswith("MLP_L.") else "projection" if name.startswith("proj3.") else "base"
        groups[group].append((name, parameter))
    optimizer.param_groups = []
    for group, members in groups.items():
        factor = cfg["attention_rate_ratio"] if group == "attention" else 1.
        optimizer.param_groups.append(dict(template, params=[p for _, p in members], param_names=[n for n, _ in members],
                                     group_name=group, lr=lr * factor, update_scale=factor))
    return optimizer


def component_losses(outputs, x, data, relations, saturate):
    z, _, _, mean, spatial, feature, _ = outputs
    return (scaled_cosine_error(x, mean), pair_loss(z, relations, saturate),
            utils.dicr_loss_little_block_diagonal_precluster(spatial, feature, data.preclusters))


def classify(values, k):
    matrix = np.array(values, dtype=np.float64, order="C", copy=True)
    return KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(matrix).astype(np.int16)


def readout(embedding, mean, k):
    normalized = mean.astype(np.float64)
    normalized /= np.maximum(np.linalg.norm(normalized, axis=1, keepdims=True), 1e-12)
    return classify(embedding, k), classify(normalized, k)


def fitted_arrays(outputs):
    return dict(embedding=outputs[0].detach().cpu().numpy().copy(),
                reconstruction=outputs[3].detach().cpu().numpy().copy())


def train(data, coordinates, spec, cfg, seed, device, steps=None):
    family = spec["family"]
    w = cfg["family_weights"][family]
    weights = dict(rec=w["zinb"], pair=w["breg"], ccr=w["bdccr"])
    rules = dict(cfg["stop"], **cfg["type_stop_overrides"].get(family, {}))
    stopper = Plateau(rules, weights)
    rel = tuple(torch.as_tensor(a, device=device) for a in relation_arrays(data, coordinates))
    network, graph, x, features = make_system(data, family, cfg, seed, device)
    spatial_graph = graph.normalized().to_dense()
    optimizer = make_optimizer(network, cfg)
    saturate = family in cfg["saturate_trusted_negatives"]
    history = []
    start = time.monotonic()
    for epoch in range(1, 1 + (steps if steps is not None else cfg["maximum_epoch"])):
        if time.monotonic() - start > cfg["maximum_case_seconds"]:
            raise TimeoutError(f"Training exceeded the case time limit at update {epoch}")
        network.train()
        optimizer.zero_grad(set_to_none=True)
        rec, pair, ccr = component_losses(network(x, spatial_graph, features), x, data, rel, saturate)
        loss = sum([weights["rec"] * rec, weights["pair"] * pair, weights["ccr"] * ccr])
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite loss at update {epoch}")
        loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in network.parameters()):
            raise FloatingPointError(f"Nonfinite gradient at update {epoch}")
        optimizer.step()
        history.append(dict(CompletedSteps=epoch, Loss=float(loss.detach()), Reconstruction=float(rec.detach()),
                            Pair=float(pair.detach()), CCR=float(ccr.detach()), lambda_rec=weights["rec"],
                            lambda_pair=weights["pair"], lambda_ccr=weights["ccr"],
                            lr_base=optimizer.param_groups[0]["lr"], lr_attention=optimizer.param_groups[1]["lr"],
                            MetaRaw=float(network.meta.detach()), MetaEffective=float(network.meta.detach().clamp(0, 1))))
        if steps is None and stopper(history):
            break
    network.eval()
    state = dict(model={k: v.detach().cpu().clone() for k, v in network.state_dict().items()},
                 graph={k: v.detach().cpu().clone() for k, v in graph.state_dict().items()}, attention_history=None,
                 optimization_epoch=len(history), completed_updates=len(history), mode="eval",
                 optimizer=copy.deepcopy(optimizer.state_dict()), seed=seed, arm="fixed_reference")
    with torch.no_grad():
        arrays = fitted_arrays(network(x, spatial_graph, features))
    return state, arrays, history, stopper


def replay(data, spec, cfg, state, device):
    network, graph, x, features = make_system(data, spec["family"], cfg, state["seed"], device)
    network.load_state_dict(state["model"], strict=True)
    graph.load_state_dict(state["graph"], strict=True)
    network.eval()
    with torch.no_grad():
        return fitted_arrays(network(x, graph.normalized().to_dense(), features))
