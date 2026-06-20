"""
train.py
========
Train the from-scratch network on a real-style classification dataset
(Seeds: 7 features / 3 classes, or Heart: 13 features / 2 classes).

It exercises the full pipeline taught across the source curriculum:
  load -> normalise -> split -> mini-batch GD (+ optional momentum & LR decay)
  -> evaluate (accuracy + confusion matrix) -> GPU batched-inference check.

Run:
    python train.py                       # seeds, mini-batch + momentum
    python train.py --dataset heart
    python train.py --dataset seeds --epochs 400 --batch-size 16 \
                    --lr 0.5 --momentum 0.9 --lr-decay 0.001
"""

from __future__ import annotations

import argparse

import numpy as np

from nnscratch import NeuralNetwork, gpu
from nnscratch import data as datamod
from nnscratch import metrics


def get_dataset(name: str):
    if name == "seeds":
        return datamod.load_seeds(), "Seeds (wheat varieties)"
    if name == "heart":
        return datamod.load_heart(), "Heart disease"
    if name == "moons":
        return datamod.make_moons(n_samples=1000, noise=0.2), "Two moons"
    raise ValueError(f"unknown dataset: {name}")


def main():
    ap = argparse.ArgumentParser(description="Train NN-from-scratch on real data")
    ap.add_argument("--dataset", choices=["seeds", "heart", "moons"],
                    default="seeds")
    ap.add_argument("--hidden", type=int, nargs="+", default=[16, 8],
                    help="hidden layer sizes, e.g. --hidden 16 8")
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=0.5)
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--lr-decay", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    # --- load + preprocess ---
    raw, title = get_dataset(args.dataset)
    raw = datamod.normalize_features(raw)             # zero-mean / unit-var
    train_set, test_set = datamod.train_test_split(raw, test_ratio=0.25,
                                                   seed=args.seed)
    n_inputs = len(train_set[0]) - 1
    n_outputs = datamod.n_classes(raw)

    print("=" * 64)
    print(f" Training NN-from-scratch on: {title}")
    print("=" * 64)
    print(f" Backend          : {gpu.device_info()}")
    print(f" Samples          : {len(raw)} (train {len(train_set)} / "
          f"test {len(test_set)})")
    print(f" Features         : {n_inputs}")
    print(f" Classes          : {n_outputs}")

    # --- build network ---
    arch = [n_inputs] + list(args.hidden) + [n_outputs]
    net = NeuralNetwork(arch, seed=args.seed, activation="sigmoid")
    print("-" * 64)
    print(net.summary())
    print("-" * 64)
    print(f" Optimizer: mini-batch GD | batch={args.batch_size} | lr={args.lr} "
          f"| momentum={args.momentum} | lr_decay={args.lr_decay}")
    print(" Training...")

    # --- train ---
    history = net.train(
        train_set,
        l_rate=args.lr,
        n_epoch=args.epochs,
        n_outputs=n_outputs,
        batch_size=args.batch_size,
        momentum=args.momentum,
        lr_decay=args.lr_decay,
        seed=args.seed,
        verbose_every=max(1, args.epochs // 10),
    )

    # --- evaluate ---
    print("-" * 64)
    train_eval = metrics.evaluate(net, train_set)
    test_eval = metrics.evaluate(net, test_set)
    print(f" Final train error (SSE): {history[-1]:.4f}")
    print(f" Train accuracy : {train_eval['accuracy'] * 100:6.2f}%")
    print(f" Test  accuracy : {test_eval['accuracy'] * 100:6.2f}%")
    print(" Test confusion matrix (rows=true, cols=pred):")
    print(metrics.format_confusion_matrix(test_eval["confusion_matrix"]))

    # --- cross-check the GPU batched forward path against per-sample CPU ---
    print("-" * 64)
    X_test = np.array([r[:-1] for r in test_set], dtype=np.float32)
    y_test = np.array([int(r[-1]) for r in test_set])
    gpu_pred = net.predict_batch(X_test)
    cpu_pred = np.array([net.predict(r[:-1]) for r in test_set])
    agree = float(np.mean(gpu_pred == cpu_pred)) * 100
    gpu_acc = metrics.accuracy(y_test, gpu_pred) * 100
    print(f" GPU batched-inference accuracy : {gpu_acc:6.2f}%")
    print(f" GPU vs per-sample CPU agreement: {agree:6.2f}%")
    print("=" * 64)


if __name__ == "__main__":
    main()
