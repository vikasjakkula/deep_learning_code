"""
evaluate_model.py
=================
Load a trained model and report not just ACCURACY but CONFIDENCE -- i.e. does the
model predict with a probability close to a "sure thing" (≈1.0)?

For each input the softmax output is a probability for each class.  The model's
*confidence* is the biggest of those probabilities.  A strong, well-trained model
should be both very accurate AND very confident (its top probability sits near
1.0), while still being honest when it is actually wrong.

Run:
    python evaluate_model.py                         # uses models/best_model.npz
    python evaluate_model.py --model models/best_model.npz --dataset spiral
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from nnscratch.gpu_mlp import MLPGPU
from nnscratch import data as datamod


def build_eval_set(args):
    if args.dataset == "spiral":
        rows = datamod.make_spiral(n_samples=args.samples, n_classes=args.classes,
                                   n_features=args.features,
                                   noise=args.spiral_noise, seed=args.seed)
    elif args.dataset == "synthetic":
        rows = datamod.make_blobs(args.samples, args.features, args.classes,
                                  spread=args.spread, seed=args.seed)
    elif args.dataset == "seeds":
        rows = datamod.load_seeds()
    elif args.dataset == "heart":
        rows = datamod.load_heart()
    elif args.dataset == "moons":
        rows = datamod.make_moons(n_samples=args.samples, noise=0.25)
    else:
        raise ValueError(args.dataset)
    rows = datamod.normalize_features(rows)
    _, val = datamod.train_test_split(rows, test_ratio=0.2, seed=args.seed)
    X = np.array([r[:-1] for r in val], dtype=np.float32)
    y = np.array([int(r[-1]) for r in val])
    return X, y


def main():
    ap = argparse.ArgumentParser(description="Evaluate accuracy + confidence")
    ap.add_argument("--model", default="models/best_model.npz")
    ap.add_argument("--dataset",
                    choices=["spiral", "synthetic", "seeds", "heart", "moons"],
                    default="spiral")
    ap.add_argument("--samples", type=int, default=24000)
    ap.add_argument("--features", type=int, default=32)
    ap.add_argument("--classes", type=int, default=5)
    ap.add_argument("--spread", type=float, default=2.2)
    ap.add_argument("--spiral-noise", type=float, default=0.18)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--show", type=int, default=10, help="show N sample predictions")
    args = ap.parse_args()

    if not os.path.exists(args.model):
        raise SystemExit(f"No model at {args.model}. Train one with train_long.py.")

    net = MLPGPU.load(args.model, use_gpu=True)
    X, y = build_eval_set(args)

    probs = net.predict_proba(X)
    pred = np.argmax(probs, axis=1)
    conf = probs.max(axis=1)                 # top probability per sample
    correct = pred == y

    acc = float(np.mean(correct))
    print("=" * 64)
    print(" MODEL EVALUATION: accuracy + confidence")
    print("=" * 64)
    print(f" Model        : {args.model}")
    print(f" Architecture : {net.layer_sizes}  ({net.n_params():,} params)")
    print(f" Eval samples : {len(y)}")
    print("-" * 64)
    print(f" Accuracy                       : {acc * 100:6.2f}%")
    print(f" Mean confidence (all preds)    : {conf.mean() * 100:6.2f}%")
    print(f" Mean confidence (correct preds): {conf[correct].mean() * 100:6.2f}%")
    if (~correct).any():
        print(f" Mean confidence (wrong preds)  : {conf[~correct].mean() * 100:6.2f}%")
    print("-" * 64)
    print(" How 'sure' is it? share of predictions above a confidence bar:")
    for bar in (0.50, 0.90, 0.99, 0.999):
        share = float(np.mean(conf >= bar)) * 100
        print(f"   >= {bar*100:6.2f}% sure : {share:6.2f}% of predictions")
    print("-" * 64)
    # "near-sure" predictions and how accurate they are -- a strong model's
    # confident predictions should almost always be right.
    sure = conf >= 0.99
    if sure.any():
        sure_acc = float(np.mean(correct[sure])) * 100
        print(f" Of the {sure.mean()*100:.1f}% 'near-sure' (>=99%) predictions, "
              f"{sure_acc:.2f}% are correct.")
    print("-" * 64)
    print(f" Sample predictions (first {args.show}):")
    for i in range(min(args.show, len(y))):
        tag = "OK " if correct[i] else "XX "
        print(f"   {tag} pred={pred[i]} true={y[i]}  p(pred)={conf[i]*100:6.2f}%")
    print("=" * 64)


if __name__ == "__main__":
    main()
