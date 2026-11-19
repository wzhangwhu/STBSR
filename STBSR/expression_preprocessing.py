"""Audited preprocessing; precluster never reads annotation membership."""
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
def normalize_adata(adata, highly_genes=3000):
    adata = adata.copy()
    min_cells = 50 if adata.n_obs < 10000 else 100
    sc.pp.filter_genes(adata, min_cells=min_cells)
    n_top = min(highly_genes, max(10, adata.shape[1] - 1))
    sc.pp.highly_variable_genes(adata, flavor='seurat_v3', n_top_genes=n_top)
    adata = adata[:, adata.var['highly_variable']].copy()
    if sparse.issparse(adata.X):
        adata.X = adata.X.toarray()
    sc.pp.normalize_total(adata, target_sum=10000.0)
    sc.pp.scale(adata, zero_center=False, max_value=10)
    return adata

def precluster(adata, n_class, dataset_name=''):
    n_comps = min(10, max(2, adata.n_vars - 1), max(2, adata.n_obs - 1))
    if n_comps < 2:
        adata.obs['precluster'] = pd.Categorical(np.zeros(adata.n_obs, dtype=int).astype(str))
        return adata
    sc.tl.pca(adata, svd_solver='arpack', n_comps=n_comps)
    x_pca = adata.obsm['X_pca'].copy()
    scaler = StandardScaler()
    x_pca_scaled = scaler.fit_transform(x_pca)
    spectral = KMeans(n_clusters=n_class, random_state=42, n_init=1)
    spectral_labels = spectral.fit_predict(x_pca_scaled)
    adata.obs['precluster'] = pd.Categorical(spectral_labels.astype(str))
    return adata
