from __future__ import annotations
from dataclasses import dataclass
import time
import numpy as np
import scipy.sparse as sp
import torch
from .expression_preprocessing import normalize_adata, precluster
from . import utils

@dataclass
class TrainingData:
    x: np.ndarray
    spatial_adjacency: sp.csr_matrix
    feature_adjacency: sp.csr_matrix
    graph_neighbors: torch.Tensor
    graph_non_neighbors: torch.Tensor
    expression_edge_reliability: np.ndarray
    preclusters: np.ndarray
    genes: np.ndarray | None = None


def prepare_training_data(adata,n_clusters,spatial_radius):
    if not 1 < n_clusters < adata.n_obs:
        raise ValueError('invalid externally configured K')
    start=time.perf_counter()
    adata=normalize_adata(adata,highly_genes=3000)
    adata=precluster(adata,n_clusters)
    preprocessing_seconds=time.perf_counter()-start
    labels=adata.obs['precluster'].astype('category').cat.codes.to_numpy()
    start=time.perf_counter()
    spatial_base,neighbors,non_neighbors=utils.spatial_construct_graph(adata,radius=spatial_radius)
    feature_base=utils.features_construct_graph(adata.X,k=15)
    confidence=utils.build_precluster_confidence(adata.X,labels)
    feature=utils.refine_feature_graph_with_soft_prior(feature_base,labels,confidence,
                                                     beta=0.10,sym_mode='max',min_weight=1e-6)
    reliability=utils.build_boundary_aware_pos_weight(adata.X,neighbors,clip_min=0.0,gamma=1.1)
    spatial=utils.build_boundary_aware_sadj(spatial_base,reliability,mix_alpha=0.70,
                                          sym_mode='max',min_weight=1e-4)
    data=TrainingData(np.asarray(adata.X,dtype=np.float32),sp.csr_matrix(spatial),sp.csr_matrix(feature),
                      neighbors,non_neighbors,np.asarray(reliability,dtype=np.float32),labels,
                      adata.var_names.to_numpy().astype(str))
    timings={'PreprocessingSeconds':preprocessing_seconds,'GraphBuildSeconds':time.perf_counter()-start}
    return data,timings
