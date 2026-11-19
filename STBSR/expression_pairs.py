import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

SEED_VIEWS = 42

def normalized_rows(x):
    return x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-8)


def expression_views(data):
    x = np.log1p(np.asarray(data.X,dtype=np.float32))
    spatial = np.asarray(data.obsm['spatial'])
    neighbors = NearestNeighbors(n_neighbors=6).fit(spatial).kneighbors(return_distance=False)
    genes = np.random.default_rng(SEED_VIEWS).permutation(x.shape[1])
    views = []
    for hop, columns in enumerate((genes[::2], genes[1::2]),1):
        pc = PCA(n_components=20,svd_solver='randomized',random_state=SEED_VIEWS).fit_transform(x[:,columns])
        for _ in range(hop):
            pc = .25*pc + .75*pc[neighbors].mean(axis=1)
        views.append(normalized_rows(pc))
    return views,neighbors


def ranked_pairs(a,b,quantile,cap,high):
    n = len(a)
    # Both spots must put one another in the requested tail in both views.
    ix = int((n-1)*quantile)
    threshold_a = np.partition(a,ix,axis=1)[:,ix].copy()
    threshold_b = np.partition(b,ix,axis=1)[:,ix].copy()
    pieces = []
    for i in range(n):
        if high:
            mask = ((a[i]>=np.maximum(threshold_a[i],threshold_a)) &
                    (b[i]>=np.maximum(threshold_b[i],threshold_b)))
        else:
            mask = ((a[i]<=np.minimum(threshold_a[i],threshold_a)) &
                    (b[i]<=np.minimum(threshold_b[i],threshold_b)))
        mask[i] = False
        eligible = np.flatnonzero(mask)
        score = (np.minimum(a[i,eligible],b[i,eligible]) if high
                 else -np.maximum(a[i,eligible],b[i,eligible]))
        best = eligible[np.lexsort((eligible,-score))[:cap]]
        pieces.append(np.minimum(i,best)*n+np.maximum(i,best))
    return np.unique(np.concatenate(pieces))


def trusted_pairs(views,neighbors):
    a,b = (view@view.T for view in views)
    n = len(a)
    local_row = np.repeat(np.arange(n),neighbors.shape[1])
    local_col = neighbors.ravel()
    mid_a = np.median(a,axis=1)
    mid_b = np.median(b,axis=1)
    local_mask = ((a[local_row,local_col]>=np.maximum(mid_a[local_row],mid_a[local_col])) &
                  (b[local_row,local_col]>=np.maximum(mid_b[local_row],mid_b[local_col])))
    r,c = local_row[local_mask],local_col[local_mask]
    local = np.unique(np.minimum(r,c)*n+np.maximum(r,c))
    nonlocal_candidates = ranked_pairs(a,b,.95,6,high=True)
    positives = np.union1d(local,nonlocal_candidates)
    negatives = ranked_pairs(a,b,.10,12,high=False)
    # Exclude direct geometric neighbors from negative supervision.
    spatial_edges = np.unique(np.minimum(local_row,local_col)*n+np.maximum(local_row,local_col))
    negatives = np.setdiff1d(negatives,np.union1d(positives,spatial_edges),assume_unique=True)
    r,c = np.divmod(positives,n)
    weights = np.clip((1+np.minimum(a[r,c],b[r,c]))*.5,.05,1).astype(np.float32)
    assert len(positives)>n and len(negatives)>n and not np.intersect1d(positives,negatives).size
    assert np.all(r<c) and np.isfinite(weights).all()
    return {
        "pos": np.stack((r, c)),
        "pos_w": weights,
        "neg": np.stack(np.divmod(negatives, n)),
    }
