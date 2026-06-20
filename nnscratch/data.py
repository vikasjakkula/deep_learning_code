"""
data.py
=======
Self-contained dataset utilities -- loading, synthesising, normalising and
splitting -- so the training scripts run fully offline and deterministically.

Real-world datasets referenced by the brief:
  * **Seeds**  (UCI, 210 rows, 7 features, 3 wheat varieties)
  * **Heart disease** (UCI, 13 features, binary)

If a CSV for one of these is dropped into ``deep_learning_code/data/`` it is
used directly; otherwise we generate a faithful *synthetic* stand-in with the
same shape and class structure so nothing requires a network connection.
"""

from __future__ import annotations

import csv
import os
from typing import List, Tuple

import numpy as np

# .../deep_learning_code/nnscratch/data.py -> .../deep_learning_code/data
DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"
)


# ---------------------------------------------------------------------------
# Generic CSV loader  (last column = integer label)
# ---------------------------------------------------------------------------
def load_csv(path: str, has_header: bool = False) -> List[List[float]]:
    """Load a CSV into the ``[f1, ..., fn, label]`` row format used everywhere."""
    rows: List[List[float]] = []
    with open(path, newline="") as fh:
        reader = csv.reader(fh)
        if has_header:
            next(reader, None)
        for raw in reader:
            if not raw:
                continue
            rows.append([float(x) for x in raw])
    return rows


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
def normalize_features(dataset: List[List[float]]) -> List[List[float]]:
    """Standardise each feature column to zero mean / unit variance.

    Networks train far more reliably when inputs are on a common scale -- this
    is the from-scratch equivalent of sklearn's ``StandardScaler``.
    """
    data = np.array(dataset, dtype=np.float64)
    X, y = data[:, :-1], data[:, -1:]
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1.0
    X = (X - mean) / std
    return np.hstack([X, y]).tolist()


def train_test_split(dataset, test_ratio: float = 0.25, seed: int = 42
                     ) -> Tuple[list, list]:
    """Shuffle and split rows into (train, test)."""
    import random as _random
    rng = _random.Random(seed)
    data = list(dataset)
    rng.shuffle(data)
    n_test = int(len(data) * test_ratio)
    return data[n_test:], data[:n_test]


def relabel_to_contiguous(dataset: List[List[float]]) -> Tuple[List[List[float]], dict]:
    """Remap arbitrary class labels onto 0..K-1 (what the trainer expects)."""
    labels = sorted({int(row[-1]) for row in dataset})
    mapping = {lab: idx for idx, lab in enumerate(labels)}
    out = [row[:-1] + [float(mapping[int(row[-1])])] for row in dataset]
    return out, mapping


def n_classes(dataset) -> int:
    return len({int(row[-1]) for row in dataset})


# ---------------------------------------------------------------------------
# Built-in datasets (CSV if available, else synthetic)
# ---------------------------------------------------------------------------
def load_seeds(seed: int = 42) -> List[List[float]]:
    """UCI 'seeds' wheat dataset: 7 features, 3 classes.

    Reads ``data/seeds_dataset.csv`` if present, otherwise generates a
    3-cluster Gaussian stand-in with the same dimensionality.
    """
    csv_path = os.path.join(DATA_DIR, "seeds_dataset.csv")
    if os.path.exists(csv_path):
        data = load_csv(csv_path)
        data, _ = relabel_to_contiguous(data)
        return data
    return make_blobs(n_samples=210, n_features=7, n_classes=3,
                      spread=1.1, seed=seed)


def load_heart(seed: int = 42) -> List[List[float]]:
    """Heart-disease style binary dataset: 13 features, 2 classes."""
    csv_path = os.path.join(DATA_DIR, "heart.csv")
    if os.path.exists(csv_path):
        data = load_csv(csv_path, has_header=True)
        data, _ = relabel_to_contiguous(data)
        return data
    return make_blobs(n_samples=300, n_features=13, n_classes=2,
                      spread=1.4, seed=seed)


# ---------------------------------------------------------------------------
# Synthetic generators (NumPy, deterministic)
# ---------------------------------------------------------------------------
def make_blobs(n_samples: int = 300, n_features: int = 7, n_classes: int = 3,
               spread: float = 1.0, seed: int = 42) -> List[List[float]]:
    """Gaussian blobs: each class is a cloud around its own random centre."""
    rng = np.random.default_rng(seed)
    centres = rng.normal(0, 4.0, size=(n_classes, n_features))
    per = n_samples // n_classes
    rows = []
    for c in range(n_classes):
        pts = rng.normal(centres[c], spread, size=(per, n_features))
        for p in pts:
            rows.append(list(map(float, p)) + [float(c)])
    rng.shuffle(rows)
    return rows


def make_moons(n_samples: int = 1000, noise: float = 0.2, seed: int = 42
               ) -> List[List[float]]:
    """The classic two-interleaving-half-moons binary problem (2 features)."""
    rng = np.random.default_rng(seed)
    n_out = n_samples // 2
    n_in = n_samples - n_out
    t_out = np.linspace(0, np.pi, n_out)
    t_in = np.linspace(0, np.pi, n_in)
    outer = np.c_[np.cos(t_out), np.sin(t_out)]
    inner = np.c_[1 - np.cos(t_in), 1 - np.sin(t_in) - 0.5]
    X = np.vstack([outer, inner]) + rng.normal(0, noise, size=(n_samples, 2))
    y = np.array([0] * n_out + [1] * n_in)
    rows = [list(map(float, X[i])) + [float(y[i])] for i in range(n_samples)]
    rng.shuffle(rows)
    return rows


def xor_dataset() -> List[List[float]]:
    """The canonical XOR problem used throughout the source curriculum."""
    return [[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]]
