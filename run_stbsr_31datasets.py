from __future__ import annotations

import argparse
import gc
import json
import shutil
from datetime import datetime
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import torch

from metric_utils.benchmark_metric_utils import compute_all_metrics
from STBSR import pipeline as stbsr


ROOT = Path(__file__).resolve().parent
DEFAULT_MANIFEST = ROOT / "datasets_31.csv"
DEFAULT_OUTPUT = ROOT / "outputs" / "stbsr_31datasets"
DEFAULT_DATASETS = None
DEFAULT_SEEDS = list(range(1, 11))
DEFAULT_EPOCHS = 500
METHOD = "STBSR"
EMBEDDING_KEY = "STBSR_embedding"
METADATA_KEY = "stbsr_run"

MANIFEST_COLUMNS = [
    "Order",
    "Dataset",
    "InputPath",
    "GroundTruthKey",
    "SpatialRadius",
    "PlotSpotSize",
]
SUMMARY_METRICS = [
    "ARI",
    "NMI",
    "ACC",
    "Purity",
    "Entropy",
    "Time",
    "MoransScore",
    "BIoU",
    "BF",
    "BandAcc",
]


def load_manifest(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    manifest = pd.read_csv(path, encoding="utf-8-sig")
    missing = [column for column in MANIFEST_COLUMNS if column not in manifest]
    if missing:
        raise ValueError(f"Manifest missing columns: {missing}")

    manifest = manifest[MANIFEST_COLUMNS].copy()
    manifest["Order"] = pd.to_numeric(
        manifest["Order"],
        errors="raise",
    ).astype(int)
    manifest["SpatialRadius"] = pd.to_numeric(
        manifest["SpatialRadius"],
        errors="raise",
    ).astype(int)
    manifest["PlotSpotSize"] = pd.to_numeric(
        manifest["PlotSpotSize"],
        errors="raise",
    ).astype(float)

    if manifest["Dataset"].duplicated().any():
        raise ValueError("Dataset names must be unique")
    if (manifest["SpatialRadius"] <= 0).any():
        raise ValueError("SpatialRadius must be positive")
    if (manifest["PlotSpotSize"] <= 0).any():
        raise ValueError("PlotSpotSize must be positive")
    return manifest.sort_values("Order").reset_index(drop=True)


def configure_pipeline(manifest: pd.DataFrame, epochs: int) -> None:
    records = manifest.set_index("Dataset").to_dict("index")
    required = {"HBC", *stbsr.DLPFC_DATASETS}
    missing = sorted(required - set(records))
    if missing:
        raise ValueError(f"Manifest missing required datasets: {missing}")

    stbsr.EPOCHS = int(epochs)
    stbsr.HBC_DATA_PATH = str(records["HBC"]["InputPath"])
    stbsr.HBC_TARGET_COL = str(records["HBC"]["GroundTruthKey"])
    stbsr.DLPFC_PATHS = {
        name: str(records[name]["InputPath"])
        for name in stbsr.DLPFC_DATASETS
    }
    dlpfc_keys = {
        str(records[name]["GroundTruthKey"])
        for name in stbsr.DLPFC_DATASETS
    }
    if len(dlpfc_keys) != 1:
        raise ValueError("All DLPFC datasets must use the same GroundTruthKey")
    stbsr.DLPFC_TARGET_COL = dlpfc_keys.pop()

    stbsr.VISIUM_DATASET_MAP = {}
    stbsr.H5AD_DATASET_MAP = {}
    for row in manifest.to_dict("records"):
        name = str(row["Dataset"])
        if name == "HBC" or name in stbsr.DLPFC_DATASETS:
            continue
        item = {
            "name": name,
            "path": str(row["InputPath"]),
            "gt_key": str(row["GroundTruthKey"]),
        }
        if Path(item["path"]).suffix.lower() == ".h5ad":
            stbsr.H5AD_DATASET_MAP[name] = item
        else:
            stbsr.VISIUM_DATASET_MAP[name] = item
    stbsr.REGISTERED_DATASET_MAP = {
        **stbsr.VISIUM_DATASET_MAP,
        **stbsr.H5AD_DATASET_MAP,
    }

    stbsr.SPATIAL_RADIUS_BY_DATASET.clear()
    stbsr.PLOT_SPOT_SIZE_BY_DATASET.clear()
    for row in manifest.to_dict("records"):
        name = str(row["Dataset"])
        stbsr.SPATIAL_RADIUS_BY_DATASET[name] = int(row["SpatialRadius"])
        stbsr.PLOT_SPOT_SIZE_BY_DATASET[name] = float(row["PlotSpotSize"])


def validate_inputs(manifest: pd.DataFrame, datasets: list[str]) -> None:
    known = set(manifest["Dataset"])
    unknown = sorted(set(datasets) - known)
    if unknown:
        raise ValueError(f"Datasets not present in manifest: {unknown}")
    selected = manifest[manifest["Dataset"].isin(datasets)]
    missing_paths = [
        str(path)
        for path in selected["InputPath"].map(Path)
        if not path.exists()
    ]
    if missing_paths:
        raise FileNotFoundError(f"Missing input paths: {missing_paths}")


def prepare_output(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {path}. "
            "Choose a new directory."
        )
    path.mkdir(parents=True, exist_ok=True)


def add_spatial_metrics(adata, metrics: dict[str, object]) -> None:
    extra = compute_all_metrics(
        adata.obs["ground_truth"].astype(str).to_numpy(),
        adata.obs["cluster_final"].astype(str).to_numpy(),
        np.asarray(adata.obsm["spatial"])[:, :2],
        n_neighbors=6,
    )
    for key in [
        "MoransScore",
        "BIoU",
        "BF",
        "BandAcc",
        "BoundaryPrecision",
        "BoundaryRecall",
        "BoundaryEdgesGT",
        "BoundaryEdgesPred",
        "BoundaryBandSpots",
    ]:
        metrics[key] = extra[key]


def save_compact_h5ad(
    adata,
    path: Path,
    dataset: str,
    seed: int,
    metrics: dict[str, object],
) -> None:
    obs_columns = [
        column
        for column in [
            "ground_truth",
            "ground",
            "cluster_final",
            "cluster_final_plot",
            "precluster",
            "in_tissue",
        ]
        if column in adata.obs
    ]
    result = ad.AnnData(
        X=np.zeros((adata.n_obs, 1), dtype=np.float32),
        obs=adata.obs[obs_columns].copy(),
        obsm={
            "spatial": np.asarray(
                adata.obsm["spatial"],
                dtype=np.float32,
            )[:, :2],
            EMBEDDING_KEY: np.asarray(
                adata.obsm[EMBEDDING_KEY],
                dtype=np.float32,
            ),
        },
    )
    result.uns[METADATA_KEY] = {
        "method": METHOD,
        "dataset": dataset,
        "seed": int(seed),
        "metrics": {
            key: float(value)
            for key, value in metrics.items()
            if isinstance(value, (int, float, np.integer, np.floating))
            and np.isfinite(value)
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    result.write_h5ad(path, compression="gzip")


def write_tables(rows: list[dict[str, object]], output: Path) -> None:
    runs = pd.DataFrame(rows).sort_values(["Dataset", "Seed"])
    runs.to_csv(
        output / "all_runs_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    available = [metric for metric in SUMMARY_METRICS if metric in runs]
    summary = (
        runs.groupby("Dataset", sort=False)[available]
        .agg(["mean", "std", "var", "min", "max", "count"])
        .reset_index()
    )
    summary.columns = [
        "_".join(str(item) for item in column if str(item))
        if isinstance(column, tuple)
        else str(column)
        for column in summary.columns
    ]
    summary.to_csv(
        output / "summary_mean_std_var_min_max.csv",
        index=False,
        encoding="utf-8-sig",
    )


def save_best_result(
    dataset: str,
    dataset_rows: list[dict[str, object]],
    output: Path,
) -> None:
    best = sorted(
        dataset_rows,
        key=lambda row: (-float(row["ARI"]), -float(row["NMI"]), int(row["Seed"])),
    )[0]
    source = Path(str(best["ResultH5AD"]))
    best_dir = output / "best_h5ad" / dataset
    best_dir.mkdir(parents=True, exist_ok=True)
    best_path = best_dir / (
        f"{dataset}_BEST_seed{int(best['Seed']):03d}"
        f"_ari{float(best['ARI']):.4f}"
        f"_nmi{float(best['NMI']):.4f}_STBSR.h5ad"
    )
    shutil.copy2(source, best_path)

    best_adata = ad.read_h5ad(best_path)
    plot_path = output / "spatial_png" / dataset / f"{dataset}_STBSR_best.png"
    stbsr.save_spatial_plot(
        best_adata,
        dataset,
        plot_path,
        float(best["ARI"]),
        float(best["NMI"]),
        int(best["Seed"]),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run STBSR on the registered benchmark datasets."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=DEFAULT_SEEDS,
    )
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = load_manifest(args.manifest)
    datasets = (
        manifest["Dataset"].tolist()
        if args.datasets is None
        else list(dict.fromkeys(args.datasets))
    )
    seeds = sorted(set(args.seeds))
    validate_inputs(manifest, datasets)
    configure_pipeline(manifest, args.epochs)

    selected = manifest.set_index("Dataset").loc[datasets].reset_index()
    if args.dry_run:
        return

    prepare_output(args.output_root)
    selected.to_csv(
        args.output_root / "input_manifest.csv",
        index=False,
        encoding="utf-8-sig",
    )
    config = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "method": METHOD,
        "epochs": int(args.epochs),
        "seeds": seeds,
        "datasets": datasets,
    }
    (args.output_root / "run_config.json").write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )

    rows: list[dict[str, object]] = []
    for dataset in datasets:
        adata, n_class, data_kind = stbsr.load_dataset(dataset)
        input_shape = (int(adata.n_obs), int(adata.n_vars))
        adata = stbsr.normalize_adata(adata, highly_genes=stbsr.highly_genes)
        model_shape = (int(adata.n_obs), int(adata.n_vars))
        adata = stbsr.precluster(adata, n_class, dataset_name=dataset)

        dataset_rows = []
        for seed in seeds:
            stbsr.set_seed(seed)
            run_adata = adata.copy()
            metrics = stbsr.train_model(
                run_adata,
                n_class,
                dataset,
                seed,
                stbsr.GAMMA,
            )
            add_spatial_metrics(run_adata, metrics)
            metrics.update(
                {
                    "Method": METHOD,
                    "Dataset": dataset,
                    "Seed": int(seed),
                    "InputNObs": input_shape[0],
                    "InputNVars": input_shape[1],
                    "ModelNObs": model_shape[0],
                    "ModelNVars": model_shape[1],
                    "NClass": int(n_class),
                    "DataKind": data_kind,
                }
            )
            result_path = (
                args.output_root
                / "run_h5ad"
                / dataset
                / (
                    f"{dataset}_seed{seed:03d}"
                    f"_ari{float(metrics['ARI']):.4f}"
                    f"_nmi{float(metrics['NMI']):.4f}_STBSR.h5ad"
                )
            )
            save_compact_h5ad(
                run_adata,
                result_path,
                dataset,
                seed,
                metrics,
            )
            metrics["ResultH5AD"] = str(result_path)
            rows.append(metrics)
            dataset_rows.append(metrics)
            write_tables(rows, args.output_root)
            del run_adata
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        save_best_result(dataset, dataset_rows, args.output_root)
        del adata
        gc.collect()

    expected = len(datasets) * len(seeds)
    if len(rows) != expected:
        raise RuntimeError(f"Expected {expected} results, found {len(rows)}")
    write_tables(rows, args.output_root)


if __name__ == "__main__":
    main()
