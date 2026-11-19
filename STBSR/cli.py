"""Train, replay, score or visualize the joint STBSR package."""
import argparse
import hashlib
import json
from pathlib import Path

from . import pipeline as joint
import numpy as np
import pandas as pd
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score
from threadpoolctl import threadpool_limits

from .cohort_data import prepare


ROOT = Path(__file__).resolve().parents[1]


def read_config(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def input_identity(dataset, data, payload):
    arrays = (data.x, np.asarray(payload["barcodes"], dtype=str), payload["coordinates"], np.asarray(data.genes, dtype=str))
    digest = hashlib.sha256()
    for value in arrays:
        x = np.ascontiguousarray(value)
        digest.update(f"{x.shape}:{x.dtype}".encode("ascii"))
        digest.update(x.view(np.uint8))
    return dict(dataset=dataset, input_sha256=digest.hexdigest())


def train_or_predict(args):
    cfg = read_config(args.config)
    spec = dict(cfg["dataset_specs"][args.dataset])
    spec["path"] = args.input if args.input is not None else args.data_root / spec["path"]
    if not spec["path"].exists():
        raise FileNotFoundError(spec["path"])
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in this Python environment")
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError(f"The output directory is not empty: {args.out}")
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    replay_threads = 2 if getattr(args, "replay_device", None) == "cpu" and device == "cuda" else 1
    with threadpool_limits(limits=1):
        data, payload, preparation = prepare(spec)
        identity = input_identity(args.dataset, data, payload)
        if args.action == "train":
            state, arrays, history, stopping = joint.train(data, payload["coordinates"], spec, cfg, args.seed, device)
            state["input_identity"] = identity
            if args.replay_device is not None and args.replay_device != device:
                if args.replay_device == "cuda" and not torch.cuda.is_available():
                    raise RuntimeError("CUDA replay requested without an available CUDA device")
                torch.set_num_threads(replay_threads)
                with threadpool_limits(limits=replay_threads):
                    arrays = joint.replay(data, spec, cfg, state, args.replay_device)
            if not args.compact:
                torch.save(state, args.out / "endpoint_model.pt")
            pd.DataFrame(history).to_csv(args.out / "loss.csv", index=False)
            receipt = dict(converged=stopping.converged, stopping_checks=stopping.checks, seed=args.seed,
                           completed_updates=state["completed_updates"])
        else:
            state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
            if "input_identity" in state and state["input_identity"] != identity:
                raise ValueError("The checkpoint input and requested dataset do not match")
            arrays = joint.replay(data, spec, cfg, state, device)
            receipt = dict(seed=state["seed"], completed_updates=state["completed_updates"])
        with threadpool_limits(limits=replay_threads):
            embedding, reconstruction = joint.readout(
                arrays["embedding"], arrays["reconstruction"], spec["given_k"]
            )
        torch.set_num_threads(1)
    path = args.out / "predictions.npz"
    pred = dict(barcodes=np.asarray(payload["barcodes"], dtype=str), coordinates=payload["coordinates"],
                embedding64=embedding, reconstruction_unit=reconstruction,
                genes=np.asarray(data.genes, dtype=str))
    if not args.compact:
        pred["embedding"] = arrays["embedding"]
    if args.save_reconstruction:
        pred["reconstructed_expression"] = arrays["reconstruction"]
    np.savez_compressed(path, **pred)
    receipt.update(**identity, device=device, replay_device=getattr(args, "replay_device", None) or device,
                   k=spec["given_k"], preparation=preparation,
                   no_annotations_read=True, prediction_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    (args.out / "run.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(path)


def load_score_inputs(args):
    with np.load(args.predictions, allow_pickle=False) as source:
        barcodes = np.asarray(source["barcodes"], dtype=str)
        coords = source["coordinates"]
        labels = {name: source[name] for name in ("embedding64", "reconstruction_unit")}
    sep = "\t" if args.labels.suffix.lower() in (".tsv", ".txt") else ","
    reference = pd.read_csv(args.labels, sep=sep, dtype={args.id_column: str})
    if not reference[args.id_column].is_unique:
        raise ValueError("Reference spot IDs must be unique")
    annotation = reference.set_index(args.id_column)[args.label_column].reindex(barcodes)
    valid = annotation.notna().to_numpy()
    if not valid.any():
        raise ValueError("No predicted spots match labelled reference spots")
    truth = np.asarray(annotation.fillna("").astype(str), dtype=str)
    scores = {key: float(adjusted_rand_score(truth[valid], value[valid])) for key, value in labels.items()}
    return coords, labels, truth, valid, scores


def score(args):
    _, _, _, valid, scores = load_score_inputs(args)
    print(json.dumps(dict(n_scored=int(valid.sum()), **scores), indent=2))


def plot(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.spatial import cKDTree

    coords, labels, truth, valid, scores = load_score_inputs(args)
    classes = sorted(set(truth[valid]))
    palette = {label: plt.get_cmap("tab20")(i % 20) for i, label in enumerate(classes)}
    palette[""] = "#d7d9dc"

    def colors(prediction):
        distinct = np.unique(prediction)
        overlap = np.array([[np.sum((prediction[valid] == cluster) & (truth[valid] == cls)) for cls in classes]
                            for cluster in distinct])
        rows, cols = linear_sum_assignment(-overlap)
        assigned = {distinct[i]: palette[classes[j]] for i, j in zip(rows, cols) if overlap[i, j] > 0}
        unmapped = iter(("#4e5868", "#b3a383", "#354946", "#7f557c") * len(distinct))
        for cluster in distinct:
            if cluster not in assigned:
                assigned[cluster] = next(unmapped)
        return [assigned[n] for n in prediction]

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.1), layout="constrained")
    plt.rcParams["pdf.fonttype"] = 42
    sizes = np.ptp(coords, axis=0)
    spacing = float(np.median(cKDTree(coords).query(coords, k=2)[0][:, 1]))
    area = float(np.clip((spacing * 130 / max(sizes) * 1.15) ** 2, .5, 80))
    titles = ("Manual annotation", f"64D embedding\nARI {scores['embedding64']:.3f}",
              f"L2-normalized reconstruction\nARI {scores['reconstruction_unit']:.3f}")
    examples = ([palette[n] for n in truth], colors(labels["embedding64"]), colors(labels["reconstruction_unit"]))
    for ax, title, color in zip(axes, titles, examples):
        ax.scatter(*coords.T, c=color, s=area, linewidths=0, rasterized=True)
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.set_title(title, fontsize=9)
        ax.set_axis_off()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf"):
        fig.savefig(args.out.with_suffix(suffix), dpi=360)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    for command in ("train", "predict"):
        p = commands.add_parser(command)
        p.add_argument("--dataset", required=True)
        p.add_argument("--config", type=Path, default=ROOT / "config.json")
        source = p.add_mutually_exclusive_group(required=True)
        source.add_argument("--data-root", type=Path)
        source.add_argument("--input", type=Path, help="Path to a Visium directory or h5ad file.")
        p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
        p.add_argument("--out", type=Path, required=True)
        p.add_argument("--save-reconstruction", action="store_true")
        p.add_argument("--compact", action="store_true", help="Keep labels and audit outputs. Skip checkpoint and latent vectors.")
        if command == "train":
            p.add_argument("--seed", type=int, default=43)
            p.add_argument("--replay-device", choices=("cpu", "cuda"),
                           help="Recompute the endpoint on this device before clustering.")
        else:
            p.add_argument("--checkpoint", type=Path, required=True)
        p.set_defaults(function=train_or_predict)
    for function in (score, plot):
        p = commands.add_parser(function.__name__)
        p.add_argument("--predictions", type=Path, required=True)
        p.add_argument("--labels", type=Path, required=True)
        p.add_argument("--id-column", default="spot_id")
        p.add_argument("--label-column", default="label")
        if function is plot:
            p.add_argument("--out", type=Path, required=True)
        p.set_defaults(function=function)
    args = parser.parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
