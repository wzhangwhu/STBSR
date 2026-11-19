"""Exercise fixed training, the HBC rule and both readouts on a small input."""
import argparse
import json
from pathlib import Path

from STBSR import pipeline as joint
import numpy as np
import scipy.sparse as sp
import torch
from threadpoolctl import threadpool_limits

from STBSR.preprocessing import TrainingData


def fixture():
    rng = np.random.default_rng(7419)
    n, g = 144, 80
    x = rng.gamma(2., 1., size=(n, g)).astype(np.float32)
    i = np.arange(n)
    groups = i // (n // 3)
    gradient = np.sin(np.linspace(0, 7, n))[:, None] * np.linspace(-10, 10, g)[None, :]
    x += gradient + 10.
    row = np.tile(i, 3)
    col = np.concatenate([(i + d) % n for d in (1, 2, 3)])
    graph = sp.csr_matrix((np.ones(len(row), np.float32), (row, col)), shape=(n, n))
    graph = graph.maximum(graph.T)
    near = graph.toarray()
    away = (1 - near - np.eye(n, dtype=np.float32)).clip(0, 1)
    d = TrainingData(x, graph, graph.copy(), torch.as_tensor(near), torch.as_tensor(away), near * .7,
                     groups, np.array([f"g{k}" for k in range(g)]))
    return d, np.column_stack((i, np.zeros(n)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    a = p.parse_args()
    cfg = json.loads(Path(__file__).with_name("config.json").read_text(encoding="utf-8"))
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    d, xy = fixture()
    with threadpool_limits(limits=1):
        labels, weights, _ = joint.relation_arrays(d, xy)
        np.testing.assert_allclose(weights[labels == 1].sum(), .5, atol=1e-6)
        np.testing.assert_allclose(weights[labels == 0].sum(), .5, atol=1e-6)
        for family in cfg["family_weights"]:
            spec = {"family": family, "given_k": 3}
            state, arrays, history, _ = joint.train(d, xy, spec, cfg, 43, a.device, steps=4)
            predicted = joint.replay(d, spec, cfg, state, a.device)
            assert len(history) == 4
            for name in arrays:
                np.testing.assert_array_equal(arrays[name], predicted[name])
            for clusters in joint.readout(arrays["embedding"], arrays["reconstruction"], 3):
                assert len(set(clusters)) == 3
    print(f"SMOKE_OK {a.device}: training, replay and both readouts, all four paper dataset families")


if __name__ == "__main__":
    main()
