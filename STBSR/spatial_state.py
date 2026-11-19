"""Sparse, topology-preserving spatial graph state with detached temporal feedback."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import scipy.sparse as sp
import torch
from torch import nn
from torch.nn import functional as F

from .utils import normalize_sparse_matrix, sparse_mx_to_torch_sparse_tensor


@dataclass(frozen=True)
class RefinementConfig:
    warmup: int = 100
    interval: int = 25
    ema_decay: float = 0.9
    max_step: float = 0.1
    strength: float = 0.7
    min_ratio: float = 0.5
    max_ratio: float = 1.5
    gate_evidence_fraction: float = 0.2

    def validate(self):
        if self.warmup < 0 or self.interval < 1:
            raise ValueError('warmup must be nonnegative and interval positive')
        if not 0 <= self.ema_decay < 1 or not 0 < self.max_step <= 1:
            raise ValueError('invalid EMA or update step')
        if not 0 < self.min_ratio <= 1 <= self.max_ratio or self.strength < 0:
            raise ValueError('invalid anchored graph bounds')
        if not 0 <= self.gate_evidence_fraction <= 0.5:
            raise ValueError('gate feedback must remain a minority of independent evidence')


class SpatialGraphState(nn.Module):
    """Stores each undirected spatial edge once. No distant edges can be created."""
    def __init__(self, adjacency, config: RefinementConfig | None = None):
        super().__init__()
        self.config = config or RefinementConfig()
        self.config.validate()
        adjacency = sp.csr_matrix(adjacency, dtype=np.float32).copy()
        adjacency.setdiag(0)
        adjacency.eliminate_zeros()
        if adjacency.shape[0] != adjacency.shape[1] or adjacency.shape[0] < 2:
            raise ValueError('spatial graph must be square, N >= 2')
        if not np.isfinite(adjacency.data).all() or (adjacency.data <= 0).any():
            raise ValueError('spatial weights must be finite and positive')
        if (adjacency-adjacency.T).nnz and np.max(np.abs((adjacency-adjacency.T).data)) > 1e-6:
            raise ValueError('expected symmetric audited spatial adjacency')
        upper = sp.triu(adjacency, k=1, format='coo')
        self.n_nodes = adjacency.shape[0]
        self.register_buffer('row',torch.from_numpy(upper.row.astype(np.int64)))
        self.register_buffer('col',torch.from_numpy(upper.col.astype(np.int64)))
        self.register_buffer('initial_weight',torch.from_numpy(upper.data.copy()))
        self.register_buffer('weight',torch.from_numpy(upper.data.copy()))
        self.register_buffer('ema_evidence',torch.zeros(upper.nnz))
        self.register_buffer('updates',torch.zeros((),dtype=torch.long))
        initial = sparse_mx_to_torch_sparse_tensor(normalize_sparse_matrix(adjacency+sp.eye(self.n_nodes)))
        self.register_buffer('initial_normalized',initial.coalesce())

    def node_edge_mean(self, values):
        total = values.new_zeros(self.n_nodes)
        degree = values.new_zeros(self.n_nodes)
        total.index_add_(0,self.row,values)
        total.index_add_(0,self.col,values)
        degree.index_add_(0,self.row,torch.ones_like(values))
        degree.index_add_(0,self.col,torch.ones_like(values))
        return total/degree.clamp_min(1)

    def normalized(self, message_gate=None):
        if message_gate is None and int(self.updates)==0:
            # Preserve the exact audited SciPy normalization before feedback starts.
            return self.initial_normalized
        values = self.weight
        if message_gate is not None:
            if message_gate.shape != values.shape:
                raise ValueError('message gate must match the stored undirected edges')
            values = values*message_gate
        diagonal = torch.arange(self.n_nodes,device=values.device)
        rows = torch.cat([self.row,self.col,diagonal])
        columns = torch.cat([self.col,self.row,diagonal])
        data = torch.cat([values,values,torch.ones(self.n_nodes,device=values.device,dtype=values.dtype)])
        degree = values.new_zeros(self.n_nodes).index_add(0,rows,data)
        data = data/degree[rows].clamp_min(1e-8)
        return torch.sparse_coo_tensor(torch.stack([rows,columns]),data,
                                       (self.n_nodes,self.n_nodes)).coalesce()

    @torch.no_grad()
    def refine(self, spatial_embedding, feature_embedding, completed_steps, gate_q=None):
        config = self.config
        if completed_steps < config.warmup or (completed_steps-config.warmup)%config.interval:
            return {'updated':False}
        if self.weight.numel()==0:
            return {'updated':False,'reason':'no_off_diagonal_spatial_edges'}
        if spatial_embedding.shape != feature_embedding.shape or spatial_embedding.shape[0] != self.n_nodes:
            raise ValueError('feedback embeddings must share N and latent dimension')
        spatial = F.normalize(spatial_embedding,p=2,dim=1,eps=1e-8)
        feature = F.normalize(feature_embedding,p=2,dim=1,eps=1e-8)
        ds = ((1-(spatial[self.row]*spatial[self.col]).sum(1))/2).clamp(0,1)
        df = ((1-(feature[self.row]*feature[self.col]).sum(1))/2).clamp(0,1)
        independent = (ds+df)/2
        evidence = independent
        agreement = torch.exp(-3*(ds-df).abs())
        if gate_q is not None:
            if gate_q.shape != evidence.shape:
                raise ValueError('feedback q shape mismatch')
            fraction = config.gate_evidence_fraction
            evidence = (1-fraction)*independent+fraction*gate_q.detach().clamp(0,1)
        if int(self.updates)==0:
            self.ema_evidence.copy_(evidence)
        stability = torch.exp(-5*(evidence-self.ema_evidence).abs())
        self.ema_evidence.mul_(config.ema_decay).add_(evidence,alpha=1-config.ema_decay)
        local = self.node_edge_mean(self.ema_evidence)
        centered = self.ema_evidence-(local[self.row]+local[self.col])/2
        confidence = agreement*stability
        ratio = torch.exp(-config.strength*centered*confidence).clamp(config.min_ratio,config.max_ratio)
        # Anchor to the original STBSR graph, rather than repeatedly compounding cuts.
        candidate = self.initial_weight*ratio
        ramp = min(1.0,(completed_steps-config.warmup+config.interval)/(4*config.interval))
        eta = config.max_step*ramp*confidence
        previous = self.weight.clone()
        updated = self.weight+eta*(candidate-self.weight)
        updated = updated.maximum(self.initial_weight*config.min_ratio).minimum(self.initial_weight*config.max_ratio)
        if not torch.isfinite(updated).all():
            raise FloatingPointError('nonfinite graph update')
        self.weight.copy_(updated)
        self.updates.add_(1)
        return {'updated':True,'updates':int(self.updates),
                'mean_absolute_change':float((self.weight-previous).abs().mean()),
                'min_ratio':float((self.weight/self.initial_weight).min()),
                'max_ratio':float((self.weight/self.initial_weight).max()),
                'mean_confidence':float(confidence.mean())}
