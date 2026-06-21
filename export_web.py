"""
export_web.py
=============
Run the trained model and export everything the web frontend needs into
``web/data.js`` (a plain JS file, so the page works by double-clicking
``web/index.html`` -- no server, no CORS headaches).

It writes, for the saved model in ``models/best_model.npz``:
  * overall metrics (accuracy, confidence, "how sure" bars),
  * a sample of points with their TRUE class, PREDICTED class and probability,
  * a decision-boundary grid (the model's predicted class across the 2-D plane),
  * a table of example predictions.

Then it copies ``models/training_curve.png`` next to the page (if present).

Run (use the GPU venv python):
    python export_web.py
    python export_web.py --dataset spiral --classes 6 --spiral-noise 0.05
"""

from __future__ import annotations

import argparse
import json
import os
import shutil

import numpy as np

from nnscratch.gpu_mlp import MLPGPU
from nnscratch import data as datamod

HERE = os.path.dirname(os.path.abspath(__file__))
WEB = os.path.join(HERE, "web")

# Up to 8 class colours (CSS) reused by the page.
CLASS_COLORS = ["#ef4444", "#3b82f6", "#22c55e", "#f59e0b",
                "#a855f7", "#06b6d4", "#ec4899", "#84cc16"]


def build_dataset(args):
    if args.dataset == "spiral":
        rows = datamod.make_spiral(args.samples, n_classes=args.classes,
                                   n_features=2, noise=args.spiral_noise,
                                   n_turns=args.spiral_turns, seed=args.seed)
        name = f"{args.classes}-arm intertwined spiral"
    elif args.dataset == "moons":
        rows = datamod.make_moons(n_samples=args.samples, noise=0.25)
        name = "two moons"
    else:
        rows = datamod.make_blobs(args.samples, 2, args.classes,
                                  spread=args.spread, seed=args.seed)
        name = f"{args.classes} Gaussian blobs"
    rows = datamod.normalize_features(rows)
    return rows, name


def main():
    ap = argparse.ArgumentParser(description="Export model output for the web UI")
    ap.add_argument("--model", default="models/best_model.npz")
    ap.add_argument("--dataset", choices=["spiral", "moons", "blobs"],
                    default="spiral")
    ap.add_argument("--samples", type=int, default=24000)
    ap.add_argument("--classes", type=int, default=6)
    ap.add_argument("--spiral-noise", type=float, default=0.05)
    ap.add_argument("--spiral-turns", type=float, default=2.5)
    ap.add_argument("--spread", type=float, default=2.2)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--points", type=int, default=2000, help="scatter sample size")
    ap.add_argument("--grid", type=int, default=110, help="decision-grid resolution")
    args = ap.parse_args()

    if not os.path.exists(args.model):
        raise SystemExit(f"No model at {args.model}. Train one with train_long.py.")

    os.makedirs(WEB, exist_ok=True)
    net = MLPGPU.load(args.model, use_gpu=False)   # CPU forward -> light on memory
    rows, ds_name = build_dataset(args)
    data = np.array(rows, dtype=np.float32)
    X, y = data[:, :-1], data[:, -1].astype(int)

    probs = net.predict_proba(X)
    pred = np.argmax(probs, axis=1)
    conf = probs.max(axis=1)
    correct = pred == y
    n_classes = int(y.max()) + 1

    # ----- decision-boundary grid (model's predicted class across the plane) ---
    pad = 0.4
    x_min, x_max = X[:, 0].min() - pad, X[:, 0].max() + pad
    y_min, y_max = X[:, 1].min() - pad, X[:, 1].max() + pad
    gx = np.linspace(x_min, x_max, args.grid)
    gy = np.linspace(y_min, y_max, args.grid)
    mesh = np.array([[a, b] for b in gy for a in gx], dtype=np.float32)
    grid_pred = np.argmax(net.predict_proba(mesh), axis=1).astype(int)
    grid_conf = net.predict_proba(mesh).max(axis=1)

    # ----- sample of points for the scatter -----
    rng = np.random.default_rng(0)
    idx = rng.choice(len(X), size=min(args.points, len(X)), replace=False)
    points = [{"x": float(X[i, 0]), "y": float(X[i, 1]),
               "t": int(y[i]), "p": int(pred[i]),
               "c": round(float(conf[i]), 4)} for i in idx]

    # ----- metrics -----
    # keys match the web page: "50","90","99","999" (99.9%) -- note 0.999 must
    # NOT collapse onto the 0.99 key, so they are spelled out explicitly.
    bars = {key: round(float(np.mean(conf >= b)) * 100, 2)
            for key, b in [("50", 0.5), ("90", 0.9), ("99", 0.99), ("999", 0.999)]}
    sure = conf >= 0.99
    payload = {
        "project": "nnscratch",
        "dataset": ds_name,
        "architecture": net.layer_sizes,
        "n_params": int(net.n_params()),
        "n_classes": n_classes,
        "n_eval": int(len(y)),
        "accuracy": round(float(np.mean(correct)) * 100, 2),
        "mean_conf": round(float(conf.mean()) * 100, 2),
        "mean_conf_correct": round(float(conf[correct].mean()) * 100, 2),
        "sure_share": round(float(sure.mean()) * 100, 2),
        "sure_correct": round(float(np.mean(correct[sure])) * 100, 2)
        if sure.any() else 0.0,
        "bars": bars,
        "colors": CLASS_COLORS[:n_classes],
        "bounds": {"xmin": float(x_min), "xmax": float(x_max),
                   "ymin": float(y_min), "ymax": float(y_max)},
        "grid_n": args.grid,
        "grid": grid_pred.tolist(),
        "grid_conf": [round(float(c), 3) for c in grid_conf],
        "points": points,
        "examples": [
            {"x": round(float(X[i, 0]), 3), "y": round(float(X[i, 1]), 3),
             "t": int(y[i]), "p": int(pred[i]),
             "c": round(float(conf[i]) * 100, 2), "ok": bool(correct[i])}
            for i in idx[:24]
        ],
    }

    out = os.path.join(WEB, "data.js")
    with open(out, "w") as fh:
        fh.write("// Auto-generated by export_web.py -- model predictions.\n")
        fh.write("window.PRED_DATA = ")
        json.dump(payload, fh)
        fh.write(";\n")
    print(f"Wrote {out}  ({os.path.getsize(out)//1024} KB)")

    curve = os.path.join(HERE, "models", "training_curve.png")
    if os.path.exists(curve):
        shutil.copy2(curve, os.path.join(WEB, "training_curve.png"))
        print("Copied training_curve.png -> web/")
    else:
        print("(no training_curve.png yet -- run plot_training.py to add it)")

    print(f"Done. Open web/index.html in a browser.")
    print(f"  accuracy={payload['accuracy']}%  mean_conf={payload['mean_conf']}%")


if __name__ == "__main__":
    main()
