"""Read reference labels after prediction and summarize model predictions."""


import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from metric_utils.benchmark_metric_utils import compute_all_metrics

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


def plot_domains(frame, methods, size=4, max_columns=4):
    """Draw reference and predicted domain assignments on the same coordinates."""
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
        plot_palette = palette.copy()
        neutral = ("#595b60", "#777980", "#96999e", "#bcbfc4")
        unmatched = sorted(
            label for label in pd.unique(assignment)
            if isinstance(label, str) and label.startswith("Unmatched ") and label not in palette
        )
        for index, label in enumerate(unmatched):
            plot_palette[label] = neutral[index % len(neutral)]
        ax.scatter(frame["x"], frame["y"], c=[plot_palette.get(label, "#d7d9dc") for label in assignment],
                   s=size, linewidths=0, rasterized=True)
        ax.set_title(name)
        ax.set_aspect("equal")
        ax.invert_yaxis()
    fig.tight_layout()
    return fig


def display_labels(reference, clusters):
    overlap = pd.crosstab(clusters, reference)
    rows, columns = linear_sum_assignment(-overlap.to_numpy())
    mapping = {
        overlap.index[row]: overlap.columns[column]
        for row, column in zip(rows, columns, strict=True)
        if overlap.iat[row, column] > 0
    }
    unmapped = sorted(set(clusters.dropna()) - mapping.keys())
    mapping.update({cluster: f"Unmatched {index}" for index, cluster in enumerate(unmapped, 1)})
    return clusters.map(mapping).fillna("Unlabelled")
