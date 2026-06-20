"""
test_smoke.py
=============
Fast, dependency-light checks that exercise every module on the CPU path so the
package is verifiably correct even without a GPU.  Run with:

    python -m pytest tests/ -q
    # or, without pytest installed:
    python tests/test_smoke.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from nnscratch import NeuralNetwork, gpu
from nnscratch.data import xor_dataset, make_blobs, normalize_features, train_test_split
from nnscratch import metrics


def test_matmul_matches_numpy():
    rng = np.random.default_rng(0)
    A = rng.standard_normal((40, 30)).astype(np.float32)
    B = rng.standard_normal((30, 20)).astype(np.float32)
    ref = A @ B
    assert np.allclose(gpu.gpu_matmul(A, B), ref, atol=1e-2)
    assert np.allclose(gpu.numpy_matmul(A, B), ref, atol=1e-2)
    assert np.allclose(gpu.naive_matmul(A, B), ref, atol=1e-2)


def test_grid_rounds_up():
    # 1000 rows / 16 -> must be 63 blocks (not 62), else last rows are dropped.
    bpg, tpb = gpu.grid_2d(1000, 1000)
    assert tpb == (16, 16)
    assert bpg == (63, 63)


def test_xor_learns():
    net = NeuralNetwork([2, 2, 2], seed=1)
    net.train(xor_dataset(), l_rate=0.5, n_epoch=5000, batch_size=2,
              momentum=0.9, seed=1, verbose_every=0)
    preds = [net.predict(r[:-1]) for r in xor_dataset()]
    assert preds == [0, 1, 1, 0]


def test_gpu_batch_matches_cpu_forward():
    net = NeuralNetwork([7, 8, 3], seed=2)
    X = np.random.default_rng(1).standard_normal((25, 7)).astype(np.float32)
    cpu = np.array([net.forward_propagate(list(x)) for x in X])
    gpu_out = net.forward_batch_gpu(X)
    assert np.allclose(cpu, gpu_out, atol=1e-4)


def test_blobs_training_accuracy():
    data = normalize_features(make_blobs(180, 7, 3, spread=1.0, seed=3))
    tr, te = train_test_split(data, 0.25, seed=3)
    net = NeuralNetwork([7, 12, 3], seed=1)
    net.train(tr, l_rate=0.5, n_epoch=200, batch_size=16, momentum=0.9,
              seed=1, verbose_every=0)
    acc = metrics.evaluate(net, te)["accuracy"]
    assert acc > 0.8, f"expected >0.8 accuracy on separable blobs, got {acc}"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {fn.__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} tests passed")
    sys.exit(1 if failed else 0)
