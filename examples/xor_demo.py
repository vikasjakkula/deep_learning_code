"""
xor_demo.py
===========
The canonical "hello world" of from-scratch neural nets: learning XOR.

XOR is not linearly separable, so a single layer cannot solve it -- it is the
classic demonstration that you need a hidden layer + non-linear transfer.
This mirrors the reference programs in the source curriculum, now expressed
through the ``nnscratch`` package and trained with mini-batch GD + momentum.

Run:
    python examples/xor_demo.py
"""

from __future__ import annotations

import os
import sys

# Allow running directly from the examples/ folder.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nnscratch import NeuralNetwork
from nnscratch.data import xor_dataset


def main():
    dataset = xor_dataset()           # [[x1, x2, label], ...]

    # 2 inputs -> 2 hidden -> 2 outputs (one-hot), exactly like the source code.
    net = NeuralNetwork([2, 2, 2], seed=1, activation="sigmoid")
    print(net.summary())
    print("-" * 40)

    net.train(
        dataset,
        l_rate=0.5,
        n_epoch=5000,
        n_outputs=2,
        batch_size=2,      # mini-batch GD
        momentum=0.9,      # classical momentum
        seed=1,
        verbose_every=1000,
    )

    print("-" * 40)
    print("Predictions after training (Mini-batch GD + Momentum):")
    correct = 0
    for row in dataset:
        x, y_true = row[:-1], int(row[-1])
        y_pred = net.predict(x)
        correct += (y_pred == y_true)
        print(f"  {x} -> pred={y_pred}  (true={y_true})")
    print(f"Accuracy: {correct}/{len(dataset)}")


if __name__ == "__main__":
    main()
