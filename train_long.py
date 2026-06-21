"""
train_long.py
=============
Long-running GPU trainer for a STRONG model.

It trains the batched, GPU-accelerated ``MLPGPU`` for as long as you let it
(a time budget and/or an epoch budget), and -- crucially -- it CHECKPOINTS the
best model to disk continuously, so leaving it running for hours is never wasted
and you can stop any time with Ctrl+C.

Heavy matrix multiplies (forward + backprop) run on the GPU via the CUDA kernels.

Examples
--------
    # train for 30 minutes on a big synthetic problem, save best model:
    python train_long.py --minutes 30

    # train for a fixed number of epochs on a deep, wide net:
    python train_long.py --epochs 5000 --hidden 256 128 64

    # train on the bundled seeds / heart / moons data instead of synthetic:
    python train_long.py --dataset seeds --minutes 5

    # resume from a saved checkpoint and keep improving it:
    python train_long.py --resume models/best_model.npz --minutes 60

Outputs (in --out, default ./models):
    best_model.npz   -> highest validation accuracy seen so far
    last_model.npz   -> most recent state (for resuming)
    training_log.csv -> epoch, train_loss, val_acc, elapsed_seconds
"""

from __future__ import annotations

import argparse
import os
import time
import signal

import numpy as np

from nnscratch import gpu
from nnscratch.gpu_mlp import MLPGPU, one_hot
from nnscratch import gpu_fast
from nnscratch import data as datamod


# Flag flipped by Ctrl+C so we stop cleanly *after* finishing the current epoch.
_STOP = {"flag": False}


def _handle_sigint(signum, frame):
    print("\n[Ctrl+C] finishing current epoch, then saving and exiting...")
    _STOP["flag"] = True


def build_dataset(args):
    """Return (X_train, Y_train, X_val, y_val, n_classes, title)."""
    if args.dataset == "spiral":
        rows = datamod.make_spiral(n_samples=args.samples,
                                   n_classes=args.classes,
                                   n_features=args.features,
                                   noise=args.spiral_noise,
                                   n_turns=args.spiral_turns, seed=args.seed)
        title = (f"Intertwined spiral ({args.samples} samples, "
                 f"{args.features} features, {args.classes} classes, "
                 f"{args.spiral_turns} turns) - HARD")
    elif args.dataset == "synthetic":
        rows = datamod.make_blobs(n_samples=args.samples,
                                  n_features=args.features,
                                  n_classes=args.classes,
                                  spread=args.spread, seed=args.seed)
        title = (f"Synthetic blobs ({args.samples} samples, "
                 f"{args.features} features, {args.classes} classes)")
    elif args.dataset == "seeds":
        rows = datamod.load_seeds(); title = "Seeds (7 feat / 3 cls)"
    elif args.dataset == "heart":
        rows = datamod.load_heart(); title = "Heart (13 feat / 2 cls)"
    elif args.dataset == "moons":
        rows = datamod.make_moons(n_samples=args.samples, noise=0.25)
        title = "Two moons (2 feat / 2 cls)"
    else:
        raise ValueError(args.dataset)

    rows = datamod.normalize_features(rows)
    train, val = datamod.train_test_split(rows, test_ratio=0.2, seed=args.seed)
    n_classes = datamod.n_classes(rows)

    Xtr = np.array([r[:-1] for r in train], dtype=np.float32)
    ytr = np.array([int(r[-1]) for r in train])
    Xva = np.array([r[:-1] for r in val], dtype=np.float32)
    yva = np.array([int(r[-1]) for r in val])
    return Xtr, one_hot(ytr, n_classes), Xva, yva, n_classes, title


def main():
    ap = argparse.ArgumentParser(description="Long-running GPU trainer")
    # stopping budget
    ap.add_argument("--minutes", type=float, default=0.0,
                    help="wall-clock budget in minutes (0 = no time limit)")
    ap.add_argument("--epochs", type=int, default=0,
                    help="max epochs (0 = no epoch limit). Use with --minutes.")
    # data
    ap.add_argument("--dataset",
                    choices=["spiral", "synthetic", "seeds", "heart", "moons"],
                    default="spiral",
                    help="default 'spiral' is a hard non-linear task that "
                         "rewards long training")
    ap.add_argument("--samples", type=int, default=24000)
    ap.add_argument("--features", type=int, default=32)
    ap.add_argument("--classes", type=int, default=5)
    ap.add_argument("--spread", type=float, default=2.2,
                    help="blob spread for --dataset synthetic (higher = harder)")
    ap.add_argument("--spiral-noise", type=float, default=0.18,
                    help="noise for --dataset spiral (higher = harder)")
    ap.add_argument("--spiral-turns", type=float, default=2.5,
                    help="how many times each spiral arm wraps (more = harder)")
    # model / optimiser
    ap.add_argument("--hidden", type=int, nargs="+", default=[256, 128, 64])
    ap.add_argument("--activation", choices=["relu", "sigmoid"], default="relu")
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--lr-decay", type=float, default=1e-4)
    ap.add_argument("--clip", type=float, default=5.0,
                    help="gradient clipping max-norm (stabilises long runs)")
    ap.add_argument("--seed", type=int, default=1)
    # engine
    ap.add_argument("--fast", action=argparse.BooleanOptionalAction, default=True,
                    help="use the device-resident GPU engine (much faster for "
                         "wide nets). --no-fast uses the simpler per-call engine.")
    # io
    ap.add_argument("--out", default="models")
    ap.add_argument("--resume", default="", help="path to a .npz checkpoint")
    ap.add_argument("--save-every", type=int, default=10,
                    help="also write last_model.npz every N epochs")
    args = ap.parse_args()

    if args.minutes <= 0 and args.epochs <= 0:
        # Safe default for "let it run a long time": 30 minutes.
        args.minutes = 30.0

    os.makedirs(args.out, exist_ok=True)
    best_path = os.path.join(args.out, "best_model.npz")
    last_path = os.path.join(args.out, "last_model.npz")
    log_path = os.path.join(args.out, "training_log.csv")

    Xtr, Ytr, Xva, yva, n_classes, title = build_dataset(args)
    n_features = Xtr.shape[1]
    arch = [n_features] + list(args.hidden) + [n_classes]

    use_fast = args.fast and gpu_fast.CUDA_AVAILABLE
    engine_name = ("FAST device-resident GPU" if use_fast
                   else ("per-call GPU" if gpu.CUDA_AVAILABLE else "NumPy CPU"))

    # build or resume model
    if use_fast:
        from numba import cuda
        if args.resume and os.path.exists(args.resume):
            net = gpu_fast.FastMLPGPU.load(args.resume, batch_size=args.batch_size)
            print(f"Resumed FAST model from {args.resume}")
        else:
            net = gpu_fast.FastMLPGPU(arch, seed=args.seed,
                                      hidden_activation=args.activation,
                                      batch_size=args.batch_size)
        # upload the whole dataset to the GPU ONCE (stays resident)
        d_X = cuda.to_device(np.ascontiguousarray(Xtr, np.float32))
        d_Y = cuda.to_device(np.ascontiguousarray(Ytr, np.float32))
        n_train = Xtr.shape[0]
        # a small fixed slice for cheap per-epoch loss reporting
        loss_X, loss_Y = Xtr[:4000], Ytr[:4000]
    else:
        if args.resume and os.path.exists(args.resume):
            net = MLPGPU.load(args.resume, use_gpu=True)
            print(f"Resumed model from {args.resume}")
        else:
            net = MLPGPU(arch, seed=args.seed,
                         hidden_activation=args.activation, use_gpu=True)

    print("=" * 70)
    print(" LONG GPU TRAINING")
    print("=" * 70)
    print(f" Backend     : {gpu.device_info()}")
    print(f" Engine      : {engine_name}")
    print(f" Dataset     : {title}")
    print(f" Train/Val   : {Xtr.shape[0]} / {Xva.shape[0]} samples")
    print(f" Architecture: {net.layer_sizes}  ({net.n_params():,} params)")
    print(f" Optimiser   : batch={args.batch_size} lr={args.lr} "
          f"momentum={args.momentum} lr_decay={args.lr_decay} act={args.activation}")
    budget = []
    if args.minutes > 0:
        budget.append(f"{args.minutes:g} min")
    if args.epochs > 0:
        budget.append(f"{args.epochs} epochs")
    print(f" Budget      : {' or '.join(budget)} (Ctrl+C to stop & save anytime)")
    print("-" * 70)

    signal.signal(signal.SIGINT, _handle_sigint)

    rng = np.random.default_rng(args.seed)
    t0 = time.perf_counter()
    deadline = t0 + args.minutes * 60 if args.minutes > 0 else float("inf")

    best_acc = -1.0
    if args.resume and os.path.exists(args.resume):
        best_acc = net.evaluate(Xva, yva)
        print(f" Resumed val accuracy: {best_acc * 100:.2f}%")

    if not os.path.exists(log_path):
        with open(log_path, "w") as fh:
            fh.write("epoch,train_loss,val_acc,elapsed_sec\n")

    epoch = 0
    try:
        while not _STOP["flag"]:
            now = time.perf_counter()
            if now >= deadline:
                print(" Time budget reached.")
                break
            if args.epochs > 0 and epoch >= args.epochs:
                print(" Epoch budget reached.")
                break

            lr = args.lr / (1.0 + args.lr_decay * epoch)
            ep_t0 = time.perf_counter()
            std_loss = None
            if use_fast:
                net.train_epoch(d_X, d_Y, n_train, lr, args.momentum,
                                args.clip, rng.permutation(n_train))
            else:
                std_loss = net.train_epoch(Xtr, Ytr, lr, args.batch_size,
                                           args.momentum, rng, max_grad_norm=args.clip)
            ep_dt = time.perf_counter() - ep_t0

            # Metrics + checkpointing.  Wrapped so a transient out-of-memory
            # (this machine's commit limit is tight) skips one report instead
            # of killing the whole run -- training simply continues.
            try:
                if use_fast:
                    val_acc, loss = net.metrics(Xva, yva, loss_X, loss_Y)
                else:
                    val_acc, loss = net.evaluate(Xva, yva), std_loss
                elapsed = time.perf_counter() - t0

                improved = val_acc > best_acc
                if improved:
                    best_acc = val_acc
                    net.save(best_path)

                with open(log_path, "a") as fh:
                    fh.write(f"{epoch},{loss:.6f},{val_acc:.6f},{elapsed:.1f}\n")

                if epoch % args.save_every == 0:
                    net.save(last_path)

                sps = Xtr.shape[0] / ep_dt
                mark = " *best*" if improved else ""
                remain = "" if deadline == float("inf") else \
                    f" | {max(0, deadline - now) / 60:5.1f} min left"
                print(f" epoch {epoch:5d} | loss {loss:.4f} | val_acc "
                      f"{val_acc * 100:6.2f}% | {ep_dt:5.2f}s "
                      f"({sps:,.0f} samp/s){remain}{mark}")
            except MemoryError:
                print(f" epoch {epoch:5d} | trained ({ep_dt:.2f}s) but skipped "
                      f"metrics/checkpoint - system low on memory")
            epoch += 1
    finally:
        net.save(last_path)
        elapsed = time.perf_counter() - t0
        print("-" * 70)
        print(f" Trained {epoch} epochs in {elapsed / 60:.1f} min")
        print(f" Best validation accuracy: {best_acc * 100:.2f}%")
        print(f" Best model : {best_path}")
        print(f" Last model : {last_path}")
        print(f" Log        : {log_path}")
        print("=" * 70)


if __name__ == "__main__":
    main()
