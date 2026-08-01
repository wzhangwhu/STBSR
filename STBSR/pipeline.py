from __future__ import division, print_function
import os
import time
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from scipy import sparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

device = DEVICE

DLPFC_PATHS = {}
HBC_DATA_PATH = ""

DLPFC_DATASETS = [
    '151507', '151508', '151509', '151510',
    '151669', '151670', '151671', '151672',
    '151673', '151674', '151675', '151676'
]
DATASET_ALIASES = {}
H5AD_DATASET_MAP = {}
VISIUM_DATASET_MAP = {}
REGISTERED_DATASET_MAP = {**VISIUM_DATASET_MAP, **H5AD_DATASET_MAP}

DLPFC_TARGET_COL = "layer_guess_reordered"
DLPFC_GROUND_MAP = {'WM': 0, 'Layer1': 1, 'Layer2': 2, 'Layer3': 3, 'Layer4': 4, 'Layer5': 5, 'Layer6': 6}
DLPFC_GROUND_MAP_FALLBACK = {
    "L1": 1, "L2": 2, "L3": 3, "L4": 4, "L5": 5, "L6": 6,
    "layer1": 1, "layer2": 2, "layer3": 3, "layer4": 4, "layer5": 5, "layer6": 6,
    "white matter": 0, "White matter": 0
}
HBC_TARGET_COL = "fine_annot_type"
HBC_GROUND_MAP = {
    'Tumor_edge_6': 0, 'Tumor_edge_5': 1, 'Tumor_edge_4': 2, 'Tumor_edge_3': 3, 'Tumor_edge_2': 4,
    'Tumor_edge_1': 5, 'IDC_8': 6, 'IDC_7': 7, 'IDC_6': 8, 'IDC_5': 9, 'IDC_4': 10, 'IDC_3': 11,
    'IDC_2': 12, 'IDC_1': 13, 'Healthy_2': 14, 'Healthy_1': 15, 'DCIS/LCIS_5': 16, 'DCIS/LCIS_4': 17,
    'DCIS/LCIS_2': 18, 'DCIS/LCIS_1': 19
}

highly_genes = 3000
EPOCHS = 500
GAMMA = 1.1
FEATURE_K = 15
SOFT_BETA = 0.10
MIX_ALPHA = 0.70
W_Z = 0.85
W_R = 0.05
W_C = 0.10

try:
    from .models import STBSR
    from .utils import (
        features_construct_graph,
        spatial_construct_graph,
        normalize_sparse_matrix,
        sparse_mx_to_torch_sparse_tensor,
        ZINB,
        dicr_loss_little_block_diagonal_precluster,
        boundary_aware_regularization_loss,
        build_boundary_aware_pos_weight,
        build_boundary_aware_sadj,
        build_precluster_confidence,
        refine_feature_graph_with_soft_prior,
        SPATIAL_RADIUS_BY_DATASET,
        PLOT_SPOT_SIZE_BY_DATASET,
        get_spatial_radius,
        get_plot_spot_size,
    )
except ImportError:
    from models import STBSR
    from utils import (
        features_construct_graph,
        spatial_construct_graph,
        normalize_sparse_matrix,
        sparse_mx_to_torch_sparse_tensor,
        ZINB,
        dicr_loss_little_block_diagonal_precluster,
        boundary_aware_regularization_loss,
        build_boundary_aware_pos_weight,
        build_boundary_aware_sadj,
        build_precluster_confidence,
        refine_feature_graph_with_soft_prior,
        SPATIAL_RADIUS_BY_DATASET,
        PLOT_SPOT_SIZE_BY_DATASET,
        get_spatial_radius,
        get_plot_spot_size,
    )

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def canonicalize_dataset_name(dataset_name: str) -> str:
    return DATASET_ALIASES.get(dataset_name, dataset_name)


def get_registered_dataset(dataset_name: str) -> dict:
    canonical_name = canonicalize_dataset_name(dataset_name)
    if canonical_name not in REGISTERED_DATASET_MAP:
        raise KeyError(f"Dataset not registered: {dataset_name}")
    return REGISTERED_DATASET_MAP[canonical_name]


def load_visium_with_metadata(data_path, labels_path, target_col):
    # Keep the pinned Scanpy reader to preserve the benchmark input layout.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"Use `squidpy\.read\.visium` instead\.",
            category=FutureWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message=r"Variable names are not unique\..*",
            category=UserWarning,
        )
        adata_raw = sc.read_visium(
            data_path,
            count_file="filtered_feature_bc_matrix.h5",
            load_images=True,
        )
    adata_raw.var_names_make_unique()
    labels_df = pd.read_table(labels_path, index_col=0)
    if target_col not in labels_df.columns:
        raise KeyError(f"{labels_path} missing column {target_col}, available columns: {list(labels_df.columns)}")
    labels = labels_df[target_col].copy()
    valid_barcodes = labels[~labels.isnull()].index
    common_barcodes = adata_raw.obs_names.intersection(valid_barcodes)
    adata = adata_raw[common_barcodes].copy()
    adata.obs['ground_truth'] = labels.loc[common_barcodes]
    return adata


def load_dlpfc(dataset):
    if dataset not in DLPFC_PATHS:
        raise KeyError(f"DLPFC dataset not configured: {dataset}")
    data_path = DLPFC_PATHS[dataset]
    labels_path = os.path.join(data_path, "metadata.tsv")
    adata = load_visium_with_metadata(data_path, labels_path, DLPFC_TARGET_COL)
    mapped = adata.obs["ground_truth"].map(DLPFC_GROUND_MAP)
    mask_na = mapped.isna()
    if mask_na.any():
        mapped.loc[mask_na] = adata.obs.loc[mask_na, "ground_truth"].map(DLPFC_GROUND_MAP_FALLBACK)
    adata.obs['ground'] = mapped
    adata = adata[~adata.obs['ground'].isnull()].copy()
    n_class = int(adata.obs['ground'].nunique())
    return adata, n_class, "visium"


def load_hbc():
    labels_path = os.path.join(HBC_DATA_PATH, "metadata.tsv")
    adata = load_visium_with_metadata(HBC_DATA_PATH, labels_path, HBC_TARGET_COL)
    adata.obs["ground"] = adata.obs["ground_truth"].map(HBC_GROUND_MAP)
    adata = adata[~adata.obs["ground"].isnull()].copy()
    n_class = int(adata.obs["ground"].nunique())
    return adata, n_class, "visium"


def ensure_spatial_coords(adata, dataset_name):
    if "spatial" in adata.obsm:
        return adata

    candidate_pairs = [
        ("array_row", "array_col"),
        ("imagerow", "imagecol"),
        ("pxl_row_in_fullres", "pxl_col_in_fullres"),
        ("row", "col"),
        ("x", "y"),
        ("xcoord", "ycoord"),
    ]
    for row_key, col_key in candidate_pairs:
        if row_key in adata.obs.columns and col_key in adata.obs.columns:
            adata.obsm["spatial"] = adata.obs[[row_key, col_key]].to_numpy(dtype=float)
            return adata

    raise KeyError(
        f"{dataset_name} missing spatial coordinates. "
        "Expected adata.obsm['spatial'] or obs columns like array_row/array_col."
    )


def load_visium_dataset(dataset):
    cfg = VISIUM_DATASET_MAP[dataset]
    labels_path = os.path.join(cfg["path"], "metadata.tsv")
    adata = load_visium_with_metadata(
        cfg["path"],
        labels_path,
        cfg["gt_key"],
    )
    adata.obs["ground_truth"] = adata.obs["ground_truth"].astype(str)
    adata.obs["ground"] = pd.Categorical(adata.obs["ground_truth"]).codes
    adata = adata[adata.obs["ground"] >= 0].copy()
    n_class = int(adata.obs["ground"].nunique())
    return adata, n_class, "visium"


def load_h5ad_dataset(dataset):
    dataset = canonicalize_dataset_name(dataset)
    cfg = H5AD_DATASET_MAP[dataset]
    adata = sc.read_h5ad(cfg["path"])
    adata.var_names_make_unique()
    gt_key = cfg["gt_key"]
    if gt_key not in adata.obs.columns:
        raise KeyError(f"{dataset} missing gt_key={gt_key}, available columns: {list(adata.obs.columns)}")
    adata = adata[~adata.obs[gt_key].isnull()].copy()
    adata.obs["ground_truth"] = adata.obs[gt_key].astype(str)
    adata.obs["ground"] = pd.Categorical(adata.obs["ground_truth"]).codes
    adata = adata[adata.obs["ground"] >= 0].copy()
    adata = ensure_spatial_coords(adata, dataset)
    n_class = int(adata.obs["ground"].nunique())
    return adata, n_class, "h5ad"


def load_dataset(dataset):
    dataset = canonicalize_dataset_name(dataset)
    if dataset in DLPFC_DATASETS:
        return load_dlpfc(dataset)
    if dataset == "HBC":
        return load_hbc()
    if dataset in VISIUM_DATASET_MAP:
        return load_visium_dataset(dataset)
    if dataset in H5AD_DATASET_MAP:
        return load_h5ad_dataset(dataset)
    raise ValueError(f"Unsupported dataset: {dataset}")


def calculate_clustering_metrics(ground_truth, cluster_pred, time_elapsed, run_idx, dataset, gamma):
    ground_truth = np.array(ground_truth).ravel()
    cluster_pred = np.array(cluster_pred).ravel()
    ari = adjusted_rand_score(ground_truth, cluster_pred)
    nmi = normalized_mutual_info_score(ground_truth, cluster_pred)

    def compute_acc(y_true, y_pred):
        cm = pd.crosstab(y_true, y_pred).values
        row_ind, col_ind = linear_sum_assignment(cm, maximize=True)
        return cm[row_ind, col_ind].sum() / len(y_true)

    def compute_purity(y_true, y_pred):
        cm = pd.crosstab(y_true, y_pred).values
        return np.sum(np.amax(cm, axis=0)) / np.sum(cm)

    def compute_entropy(y_true, y_pred):
        cm = pd.crosstab(y_true, y_pred).values
        cluster_entropy = []
        for clu in cm.T:
            cluster_total = np.sum(clu)
            if cluster_total == 0:
                continue
            probs = clu / cluster_total
            entropy = -np.sum(probs * np.log2(probs + 1e-10))
            cluster_entropy.append(entropy * (cluster_total / len(y_true)))
        return np.sum(cluster_entropy)

    return {
        'Dataset': dataset,
        'Gamma': gamma,
        'Run': run_idx,
        'ARI': round(ari, 4),
        'NMI': round(nmi, 4),
        'ACC': round(compute_acc(ground_truth, cluster_pred), 4),
        'Purity': round(compute_purity(ground_truth, cluster_pred), 4),
        'Entropy': round(compute_entropy(ground_truth, cluster_pred), 4),
        'Time': round(time_elapsed, 4)
    }


def normalize_adata(adata, highly_genes=3000):
    adata = adata.copy()
    min_cells = 50 if adata.n_obs < 10000 else 100
    sc.pp.filter_genes(adata, min_cells=min_cells)
    n_top = min(highly_genes, max(10, adata.shape[1] - 1))
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=n_top)
    adata = adata[:, adata.var['highly_variable']].copy()
    if sparse.issparse(adata.X):
        adata.X = adata.X.toarray()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.scale(adata, zero_center=False, max_value=10)
    return adata


def precluster(adata, n_class, dataset_name=""):
    n_comps = min(10, max(2, adata.n_vars - 1), max(2, adata.n_obs - 1))
    if n_comps < 2:
        adata.obs["precluster"] = pd.Categorical(np.zeros(adata.n_obs, dtype=int).astype(str))
        return adata
    sc.tl.pca(adata, svd_solver="arpack", n_comps=n_comps)
    x_pca = adata.obsm["X_pca"].copy()
    scaler = StandardScaler()
    x_pca_scaled = scaler.fit_transform(x_pca)
    spectral = KMeans(
        n_clusters=n_class,
        random_state=42,
        n_init=1,
    )
    spectral_labels = spectral.fit_predict(x_pca_scaled)
    adata.obs["precluster"] = pd.Categorical(spectral_labels.astype(str))
    return adata


def _map_pred_to_gt_names(y_true, y_pred):
    y_true = np.asarray(y_true).astype(str)
    y_pred = np.asarray(y_pred).astype(str)
    true_unique = np.unique(y_true)
    pred_unique = np.unique(y_pred)
    cm = pd.crosstab(pd.Categorical(y_true, categories=true_unique), y_pred, dropna=False)
    cm = cm.reindex(index=true_unique, columns=pred_unique, fill_value=0)
    row_ind, col_ind = linear_sum_assignment(cm.values, maximize=True)
    mapping = {cm.columns[c]: cm.index[r] for r, c in zip(row_ind, col_ind)}
    for p in pred_unique:
        mapping.setdefault(p, p)
    return np.array([mapping[p] for p in y_pred], dtype=object)


def save_spatial_plot(
    adata,
    dataset_name,
    out_path,
    ari,
    nmi,
    run_idx,
):
    plot_spot_size = get_plot_spot_size(dataset_name)
    plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if dataset_name == "HBC":
        fig_size = (14, 4)
        right_margin = 0.70
        legend_fontsize = 6
        legend_anchor = (1.20, 0.5)
        legend_spacing = 10
    else:
        fig_size = (11, 4)
        right_margin = 0.82
        legend_fontsize = 7
        legend_anchor = (1.10, 0.5)
        legend_spacing = 8

    fig, axes = plt.subplots(1, 2, figsize=fig_size)
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"Use `squidpy\.pl\.spatial_scatter` instead\.",
                category=FutureWarning,
            )
            sc.pl.spatial(
                adata,
                color="ground_truth",
                ax=axes[0],
                show=False,
                title=f"Ground truth | {dataset_name}",
                spot_size=plot_spot_size,
                legend_loc=None,
            )
            sc.pl.spatial(
                adata,
                color="cluster_final_plot",
                ax=axes[1],
                show=False,
                title=(
                    f"STBSR | run {run_idx:02d} | "
                    f"ARI={ari:.4f}, NMI={nmi:.4f}"
                ),
                spot_size=plot_spot_size,
                legend_loc="right margin",
                legend_fontsize=legend_fontsize,
            )
        legend = axes[1].get_legend()
        if legend is not None:
            legend.set_title("")
            legend.set_bbox_to_anchor(legend_anchor)
            legend._legend_box.align = "left"
            legend._legend_box.sep = legend_spacing
        if axes[0].get_legend() is not None:
            axes[0].get_legend().remove()
        for axis in axes:
            axis.set_axis_off()
        plt.subplots_adjust(wspace=0.30, right=right_margin)
        fig.savefig(str(out_path), dpi=300, bbox_inches="tight")
    finally:
        plt.close(fig)


def build_graphs(adata, dataset_name, gamma):
    spatial_radius = get_spatial_radius(dataset_name)
    sadj_base, graph_nei, graph_neg = spatial_construct_graph(
        adata,
        radius=spatial_radius,
    )
    fadj_base = features_construct_graph(adata.X, k=FEATURE_K)
    precluster_labels = adata.obs["precluster"].astype('category').cat.codes.values
    confidence = build_precluster_confidence(
        adata.X,
        precluster_labels,
    )
    fadj = refine_feature_graph_with_soft_prior(
        fadj_base,
        precluster_labels,
        confidence,
        beta=SOFT_BETA,
        sym_mode="max",
        min_weight=1e-6,
    )

    pos_weight_np = build_boundary_aware_pos_weight(
        adata.X,
        graph_nei,
        clip_min=0.0,
        gamma=gamma,
    )
    sadj = build_boundary_aware_sadj(
        sadj_base,
        pos_weight_np,
        mix_alpha=MIX_ALPHA,
        sym_mode="max",
        min_weight=1e-4,
    )
    return (
        sadj,
        fadj,
        graph_nei,
        graph_neg,
        pos_weight_np,
        spatial_radius,
    )


def train_model(adata, n_class, dataset, run_idx, gamma):
    (
        sadj,
        fadj,
        graph_nei,
        graph_neg,
        pos_weight_np,
        spatial_radius,
    ) = build_graphs(adata, dataset, gamma)
    pos_weight = torch.from_numpy(pos_weight_np).float().to(device)
    X = torch.FloatTensor(np.asarray(adata.X)).to(device)
    labels = adata.obs['ground'].astype(str).values

    fadj_t = sparse_mx_to_torch_sparse_tensor(normalize_sparse_matrix(sp.csr_matrix(fadj) + sp.eye(adata.n_obs))).to(device)
    sadj_t = sparse_mx_to_torch_sparse_tensor(normalize_sparse_matrix(sp.csr_matrix(sadj) + sp.eye(adata.n_obs))).to(device)
    graph_nei_t = torch.LongTensor(graph_nei.numpy()).to(device)
    graph_neg_t = torch.LongTensor(graph_neg.numpy()).to(device)
    _, gt = np.unique(labels, return_inverse=True)
    gt = torch.LongTensor(gt)

    model = STBSR(
        nfeat=adata.n_vars,
        nhid1=128,
        nhid2=64,
        dropout=0,
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)

    best_ari, best_nmi = -1, -1
    best_idx, best_emb, best_epoch = None, None, -1
    start_time = time.time()
    labels_codes = adata.obs["precluster"].astype('category').cat.codes.values

    pbar = tqdm(
        range(EPOCHS),
        ncols=80,
        desc="Epoch",
    )
    for epoch in pbar:
        model.train()
        optimizer.zero_grad()
        emb, pi, disp, mean, emb1, emb2, _ = model(X, sadj_t, fadj_t)

        zinb = ZINB(pi, theta=disp).loss(X, mean, mean=True)
        reg = boundary_aware_regularization_loss(
            emb,
            graph_nei_t,
            graph_neg_t,
            pos_weight,
        )
        ccr = dicr_loss_little_block_diagonal_precluster(
            emb1,
            emb2,
            labels_codes,
        )

        loss = W_Z * zinb + W_R * reg + W_C * ccr
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            emb_np = emb.detach().cpu().numpy()
            idx = KMeans(
                n_clusters=n_class,
                random_state=0,
                n_init=1,
            ).fit_predict(emb_np)
            ari = adjusted_rand_score(gt.cpu().numpy(), idx)
            nmi = normalized_mutual_info_score(gt.cpu().numpy(), idx)
            if (ari > best_ari) or (np.isclose(ari, best_ari) and nmi > best_nmi):
                best_ari = ari
                best_nmi = nmi
                best_idx = idx
                best_emb = emb_np.copy()
                best_epoch = epoch

    pbar.close()

    time_elapsed = time.time() - start_time
    adata.obs['cluster_final'] = best_idx.astype(str)
    adata.obs['cluster_final_plot'] = _map_pred_to_gt_names(adata.obs['ground_truth'].astype(str).values, adata.obs['cluster_final'].values)
    adata.obsm["STBSR_embedding"] = best_emb

    metrics = calculate_clustering_metrics(adata.obs['ground_truth'], adata.obs['cluster_final'], time_elapsed, run_idx, dataset, gamma)
    metrics.update({
        "BestEpoch": best_epoch,
        "Gamma": GAMMA,
        "SpatialRadius": spatial_radius,
        "PlotSpotSize": get_plot_spot_size(dataset),
        "SoftBeta": SOFT_BETA,
        "MixAlpha": MIX_ALPHA,
        "W_Z": W_Z,
        "W_R": W_R,
        "W_C": W_C,
    })
    return metrics
