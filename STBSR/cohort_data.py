"""Read counts, IDs and coordinates only; never load per-spot annotation tables."""
from pathlib import Path
import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp
import anndata as ad
from sklearn.neighbors import NearestNeighbors
from .preprocessing import prepare_training_data


def strings(values):return np.array([v.decode() if isinstance(v,bytes) else str(v) for v in values])


def matrix(node):
    if isinstance(node,h5py.Dataset):return node[:]
    encoding=node.attrs['encoding-type'];encoding=encoding.decode() if isinstance(encoding,bytes) else encoding
    constructor={'csr_matrix':sp.csr_matrix,'csc_matrix':sp.csc_matrix}[encoding]
    return constructor((node['data'][:],node['indices'][:],node['indptr'][:]),shape=tuple(node.attrs['shape']))


def unlabelled_h5ad(path,counts_layer=None):
    with h5py.File(path,'r') as f:
        x=matrix(f['X'] if counts_layer is None else f['layers'][counts_layer])
        obs=f['obs'];var=f['var']
        obs_index=obs.attrs.get('_index','_index');var_index=var.attrs.get('_index','_index')
        ids=strings(obs[obs_index][:]);genes=strings(var[var_index][:]);xy=f['obsm/spatial'][:,:2]
    data=ad.AnnData(x,obs=pd.DataFrame(index=ids),var=pd.DataFrame(index=genes))
    data.var_names_make_unique();data.obsm['spatial']=np.asarray(xy,dtype=float)
    values=x.data if sp.issparse(x) else np.asarray(x).ravel()
    assert np.isfinite(values).all() and values.min()>=0
    assert np.allclose(values,np.round(values),atol=1e-6),'Expected raw integer counts, not normalized X'
    assert len(np.unique(ids))==len(ids) and np.isfinite(xy).all()
    return data


def unlabelled_visium(path):
    import scanpy as sc
    path=Path(path);data=sc.read_10x_h5(path/'filtered_feature_bc_matrix.h5');data.var_names_make_unique()
    positions=pd.read_csv(path/'spatial/tissue_positions_list.csv',header=None,index_col=0).reindex(data.obs_names)
    assert not positions.isna().any().any()
    data=data[positions[1].to_numpy()==1].copy();data.obsm['spatial']=positions.loc[data.obs_names,[5,4]].to_numpy(float)
    data.obs=pd.DataFrame(index=data.obs_names);data.uns={};return data


def prepare(spec):
    layer=spec.get('counts_layer')
    node=spec.get('counts_node')
    if node not in (None,'X'):
        if not node.startswith('layers/'):
            raise ValueError(f'Unsupported count matrix location: {node}')
        selected=node.removeprefix('layers/')
        if layer is not None and selected!=layer:
            raise ValueError('The configured count layer and count matrix location disagree')
        layer=selected
    raw=unlabelled_visium(spec['path']) if spec['kind']=='visium' else unlabelled_h5ad(spec['path'],layer)
    xy=np.asarray(raw.obsm['spatial'],dtype=float);ids=raw.obs_names.to_numpy(dtype=str)
    distances=NearestNeighbors(n_neighbors=7).fit(xy).kneighbors()[0]
    rule=spec['radius_rule']
    if rule=='dlpfc560':radius=560.
    elif rule=='two_rings':radius=2.05*float(np.median(distances[:,0]))
    elif rule=='cell_six_context':radius=1.20*float(np.median(distances[:,5]))
    else:raise ValueError(rule)
    prepared,timing=prepare_training_data(raw,spec['given_k'],radius)
    return prepared,dict(x=prepared.x,barcodes=ids,coordinates=xy),dict(radius=radius,rule=rule,**timing)
