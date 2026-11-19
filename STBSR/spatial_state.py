"""Fixed spatial propagation operator used throughout training."""
import numpy as np
import scipy.sparse as sp

from .utils import normalize_sparse_matrix, sparse_mx_to_torch_sparse_tensor


def fixed_spatial_operator(adjacency):
    adjacency = sp.csr_matrix(adjacency, dtype=np.float32).copy()
    adjacency.setdiag(0)
    adjacency.eliminate_zeros()
    if adjacency.shape[0] != adjacency.shape[1] or adjacency.shape[0] < 2:
        raise ValueError("The spatial graph must be square with at least two nodes.")
    if not np.isfinite(adjacency.data).all() or (adjacency.data <= 0).any():
        raise ValueError("Spatial weights must be finite and positive.")
    difference = adjacency - adjacency.T
    if difference.nnz and np.max(np.abs(difference.data)) > 1e-6:
        raise ValueError("The spatial graph must be symmetric.")
    normalized = normalize_sparse_matrix(adjacency + sp.eye(adjacency.shape[0]))
    return sparse_mx_to_torch_sparse_tensor(normalized).coalesce()
