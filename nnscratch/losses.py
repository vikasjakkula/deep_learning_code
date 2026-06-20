"""
losses.py
=========
Loss functions used to score predictions and to seed backpropagation.

The from-scratch curriculum trains with sum-of-squared-error on one-hot targets
(simple, and its gradient w.r.t. the sigmoid output is just ``output - target``).
We keep that as the default and also provide cross-entropy for completeness.
"""

from __future__ import annotations

import numpy as np

_EPS = 1e-12  # guards log(0)


def mse(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Mean squared error over a batch (averaged across samples)."""
    predictions = np.atleast_2d(predictions)
    targets = np.atleast_2d(targets)
    return float(np.mean(np.sum((predictions - targets) ** 2, axis=1)))


def sse(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Sum of squared errors -- matches the ``error=`` figure printed by the
    classic from-scratch XOR trainers in the source curriculum."""
    predictions = np.atleast_2d(predictions)
    targets = np.atleast_2d(targets)
    return float(np.sum((predictions - targets) ** 2))


def mse_gradient(predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """dL/d(prediction) for squared error: simply (prediction - target).

    With a sigmoid output layer this is the ``neuron['output'] - expected``
    term you see in the reference backprop code.
    """
    return predictions - targets


def cross_entropy(probs: np.ndarray, targets: np.ndarray) -> float:
    """Categorical cross-entropy; expects softmax probabilities and one-hot targets."""
    probs = np.clip(np.atleast_2d(probs), _EPS, 1.0)
    targets = np.atleast_2d(targets)
    return float(-np.mean(np.sum(targets * np.log(probs), axis=1)))
