"""
metrics.py
==========
Evaluation helpers: accuracy and a small text confusion matrix.
"""

from __future__ import annotations

from typing import List
import numpy as np


def accuracy(y_true, y_pred) -> float:
    """Fraction of correct predictions."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if len(y_true) == 0:
        return 0.0
    return float(np.mean(y_true == y_pred))


def confusion_matrix(y_true, y_pred, n_classes: int) -> np.ndarray:
    """Counts matrix ``cm[true, pred]``."""
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def format_confusion_matrix(cm: np.ndarray) -> str:
    """Pretty-print a confusion matrix for the console."""
    n = cm.shape[0]
    header = "       " + " ".join(f"P{j:<4d}" for j in range(n))
    lines = [header]
    for i in range(n):
        row = " ".join(f"{cm[i, j]:<5d}" for j in range(n))
        lines.append(f"T{i:<4d} {row}")
    return "\n".join(lines)


def evaluate(network, dataset) -> dict:
    """Run the network over a dataset and return accuracy + confusion matrix."""
    n_cls = len({int(r[-1]) for r in dataset})
    y_true = [int(r[-1]) for r in dataset]
    y_pred = [network.predict(r[:-1]) for r in dataset]
    cm = confusion_matrix(y_true, y_pred, n_cls)
    return {
        "accuracy": accuracy(y_true, y_pred),
        "confusion_matrix": cm,
        "n_classes": n_cls,
    }
