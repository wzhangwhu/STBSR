"""Read reference labels after prediction and summarize a locked paper partition."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score

from metric_utils.benchmark_metric_utils import compute_all_metrics

ROOT = Path(__file__).resolve().parents[1]


def read_reference(dataset, spec, data_path):
    """Return the reference annotation indexed by original spot or cell ID."""
    if spec["family"] == "DLPFC":
        return pd.read_csv(
            data_path / f"{dataset}_truth.txt",
            sep="\t",
            names=["spot_id", "label"],
            dtype=str,
        ).set_index("spot_id")["label"]
    if dataset == "HBC":
        return pd.read_csv(data_path / "metadata.tsv", sep="\t", index_col="ID")["ground_truth"]

    import anndata as ad

    source = ad.read_h5ad(data_path, backed="r")
    try:
        column = "region" if spec["family"] == "ChickenHeart" else "Annotation"
        return source.obs[column].copy()
    finally:
        source.file.close()


def evaluate_file(path, reference):
    """Score both readouts against a single set of annotated observations."""
    if not reference.index.is_unique:
        raise ValueError("Reference IDs must be unique.")
    with np.load(path, allow_pickle=False) as data:
        ids = data["barcodes"]
        annotations = reference.reindex(ids)
        valid = annotations.notna()
        valid &= ~annotations.astype(str).str.strip().str.lower().isin(
            {"", "na", "nan", "none", "unlabelled", "unlabeled"}
        )
        mask = valid.to_numpy()
        if not mask.any():
            raise ValueError("No predicted observations match annotated reference IDs.")
        coords = data["coordinates"][mask]
        truth = annotations.to_numpy()[mask]
        return [
            dict(readout=name, n_scored=int(mask.sum()),
                 **compute_all_metrics(truth, data[name][mask], coords))
            for name in ("embedding64", "reconstruction_unit")
        ]


def paper_domains(dataset):
    """Load the assignments displayed in Fig. 2c or Fig. 4a."""
    if dataset == "151672":
        file = "fig02_151672_domains.csv"
    elif dataset == "AX_Stage57":
        file = "fig04_stage57_domains.csv"
    else:
        raise ValueError(f"No representative partition archived for {dataset}.")
    return pd.read_csv(ROOT / "benchmark" / file, dtype={"spot_id": str})


def compare_paper_ari():
    """Recompute the equal-dataset benchmark means from Supplementary Data 1."""
    values = pd.read_csv(ROOT / "benchmark" / "mean_ari_by_dataset.csv", dtype={"dataset": str})
    return values.set_index("dataset").mean().sort_values(ascending=False).rename("mean_ARI")


def plot_domains(frame, methods, size=4, max_columns=4):
    """Draw reference and archived domain assignments on the same coordinates."""
    from matplotlib import pyplot as plt

    reference = frame["reference"].fillna("Unlabelled").astype(str)
    valid = ~reference.isin(["Unlabelled", "Unlabeled"])
    classes = sorted(reference[valid].unique())
    palette = {label: plt.get_cmap("tab20")(i % 20) for i, label in enumerate(classes)}
    pairs = [("Manual annotation", reference)] + list(methods.items())
    columns = min(max_columns, len(pairs))
    rows = (len(pairs) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(3.4 * columns, 3.5 * rows), squeeze=False)
    for ax in axes.flat:
        ax.set_axis_off()
    for ax, (name, assignment) in zip(axes.flat, pairs):
        ax.scatter(frame["x"], frame["y"], c=[palette.get(x, "#d7d9dc") for x in assignment],
                   s=size, linewidths=0, rasterized=True)
        ax.set_title(name)
        ax.set_aspect("equal")
        ax.invert_yaxis()
    fig.tight_layout()
    return fig


def score_paper_domains(frame, prediction):
    valid = frame["reference"].notna() & frame["reference"].ne("Unlabelled")
    return adjusted_rand_score(frame.loc[valid, "reference"], prediction[valid]), int(valid.sum())


def display_labels(reference, clusters):
    overlap = pd.crosstab(clusters, reference)
    rows, columns = linear_sum_assignment(-overlap.to_numpy())
    mapping = dict(zip(overlap.index[rows], overlap.columns[columns]))
    return clusters.map(mapping).fillna("Unlabelled")

def stage57_boundary(frame, methods):
    """Count correctly recovered dEGC edges using the graph from Fig. 4c."""
    from scipy.spatial import cKDTree

    xy = frame[["x", "y"]].to_numpy()
    distances, neighbours = cKDTree(xy).query(xy, k=9)
    radius = 3 * np.median(distances[:, 1])
    source = np.repeat(np.arange(len(xy)), 6)
    target = neighbours[:, 1:7].ravel()
    keep = distances[:, 1:7].ravel() < radius
    edges = np.unique(np.sort(np.column_stack((source[keep], target[keep])), axis=1), axis=0)

    reference = frame["reference"].eq("dEGC").to_numpy()
    a, b = edges[reference[edges[:, 0]] != reference[edges[:, 1]]].T
    inside, outside = np.where(reference[a], a, b), np.where(reference[a], b, a)
    recovered = []
    for method in methods:
        predicted = frame[method].eq("dEGC").to_numpy()
        count = np.count_nonzero(predicted[inside] & ~predicted[outside])
        recovered.append(dict(method=method, edges=count, reference_crossing_edges=len(inside)))
    return pd.DataFrame(recovered)
