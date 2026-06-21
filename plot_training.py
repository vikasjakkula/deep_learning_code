"""
plot_training.py
================
Turn ``models/training_log.csv`` (written by train_long.py) into a picture of the
model getting stronger over time: training loss going DOWN and validation
accuracy going UP.

Run:
    python plot_training.py
    python plot_training.py --log models/training_log.csv --out models/training_curve.png
"""

from __future__ import annotations

import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")  # save to file without needing a display
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser(description="Plot the training learning curve")
    ap.add_argument("--log", default="models/training_log.csv")
    ap.add_argument("--out", default="models/training_curve.png")
    args = ap.parse_args()

    if not os.path.exists(args.log):
        raise SystemExit(f"No log found at {args.log}. Run train_long.py first.")

    epochs, loss, acc = [], [], []
    with open(args.log) as fh:
        for row in csv.DictReader(fh):
            epochs.append(int(row["epoch"]))
            loss.append(float(row["train_loss"]))
            acc.append(float(row["val_acc"]) * 100.0)

    if not epochs:
        raise SystemExit("Log is empty.")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax1.plot(epochs, loss, color="tab:red")
    ax1.set_title("Training loss (lower = better)")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross-entropy loss")
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, acc, color="tab:green")
    ax2.set_title("Validation accuracy (higher = better)")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_ylim(0, 100)
    ax2.grid(True, alpha=0.3)
    best = max(acc)
    best_e = epochs[acc.index(best)]
    ax2.axhline(best, color="gray", ls="--", lw=1)
    ax2.annotate(f"best {best:.1f}% @ epoch {best_e}",
                 xy=(best_e, best), xytext=(0.4, 0.1),
                 textcoords="axes fraction",
                 arrowprops=dict(arrowstyle="->", color="gray"))

    fig.suptitle("Model getting stronger over time")
    fig.tight_layout()
    fig.savefig(args.out, dpi=120)
    print(f"Saved learning curve -> {args.out}")
    print(f"Epochs: {len(epochs)} | final loss: {loss[-1]:.4f} | "
          f"best val acc: {best:.2f}%")


if __name__ == "__main__":
    main()
