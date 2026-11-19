"""Train STBSR on the 20-dataset paper cohort and score both KMeans readouts."""

import argparse
from pathlib import Path
import subprocess
import sys

import pandas as pd

from STBSR.cli import ROOT, read_config
from STBSR.reproduction import evaluate_file, read_reference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/paper_benchmark"))
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--datasets", nargs="+", help="Default: all 20 paper datasets.")
    parser.add_argument("--seeds", nargs="+", type=int, help="Override the recorded paper seeds.")
    parser.add_argument("--device", default="cuda", choices=("cpu", "cuda"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = read_config(args.config)
    specs = config["dataset_specs"]
    datasets = args.datasets or config["datasets"]
    if len(set(datasets)) != len(datasets) or any(ds not in specs for ds in datasets):
        parser.error("Dataset names must be unique and present in the configuration.")
    if args.seeds is not None and len(set(args.seeds)) != len(args.seeds):
        parser.error("Seeds must be unique.")
    paths = {ds: (args.data_root / specs[ds]["path"]).resolve() for ds in datasets}
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        parser.error("Missing inputs:\n" + "\n".join(missing))
    overrides = config.get("benchmark_seed_overrides", {})
    seeds = {ds: args.seeds or overrides.get(ds, config["seeds"]) for ds in datasets}
    if args.dry_run:
        for dataset, path in paths.items():
            print(f"{dataset}: {path} ({len(seeds[dataset])} runs)")
        return

    output = args.output_root.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error("Choose a new or empty output directory.")
    output.mkdir(parents=True, exist_ok=True)
    results = []
    for dataset in datasets:
        replay = config.get("benchmark_replay_devices", {}).get(dataset, args.device)
        if args.device == "cpu":
            replay = "cpu"
        predictions = []
        for run, seed in enumerate(seeds[dataset], 1):
            case = output / "cases" / dataset / f"seed{seed}"
            command = [
                sys.executable, "-m", "STBSR.cli", "train", "--dataset", dataset,
                "--config", str(args.config.resolve()), "--input", str(paths[dataset]),
                "--seed", str(seed), "--device", args.device, "--replay-device", replay,
                "--out", str(case), "--compact",
            ]
            subprocess.run(command, cwd=ROOT, check=True, timeout=config["maximum_case_seconds"] + 180)
            predictions.append((run, seed, case / "predictions.npz"))
        reference = read_reference(dataset, specs[dataset], paths[dataset])
        for run, seed, path in predictions:
            results.extend(dict(dataset=dataset, run=run, seed=seed, **record)
                           for record in evaluate_file(path, reference))
        pd.DataFrame(results).to_csv(output / "per_run.csv", index=False)

    table = pd.DataFrame(results)
    metrics = ["ARI", "NMI", "ACC", "Purity", "Entropy", "MoransScore", "BIoU", "BF", "BandAcc"]
    summary = table.groupby(["dataset", "readout"], sort=False)[metrics].agg(["mean", "std", "count"])
    summary.columns = ["_".join(col) for col in summary.columns]
    summary.reset_index().to_csv(output / "by_dataset.csv", index=False)


if __name__ == "__main__":
    main()
