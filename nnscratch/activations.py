"""
activations.py
==============
Activation ("transfer") functions and their derivatives.

The KMIT Vista sessions define a neuron in two stages:
  * **Activation (linear)** :  z = W.x + b      -- handled in engine.py
  * **Transfer (non-linear)**:  a = f(z)        -- the functions below

Each activation is provided in two forms:
  * a pure-Python scalar version (matches the from-scratch teaching style), and
  * a vectorised NumPy version used by the fast batched paths.

We also expose the derivative w.r.t. the *output* where that is convenient
(sigmoid and tanh have the neat property that f'(z) can be written using f(z)),
which is exactly what backpropagation needs.
"""

from __future__ import annotations

import math
import numpy as np


# ---------------------------------------------------------------------------
# Sigmoid:  1 / (1 + e^-x)   -- the primary transfer function for this project
# ---------------------------------------------------------------------------
def sigmoid_scalar(x: float) -> float:
    """Numerically-stable scalar sigmoid (avoids overflow for large |x|)."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    ex = math.exp(x)
    return ex / (1.0 + ex)


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Vectorised sigmoid, computed in a numerically stable way."""
    out = np.empty_like(z, dtype=np.float64)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def sigmoid_derivative(output: np.ndarray) -> np.ndarray:
    """Derivative expressed via the sigmoid OUTPUT a:  f'(z) = a * (1 - a)."""
    return output * (1.0 - output)


# ---------------------------------------------------------------------------
# ReLU:  max(0, x)
# ---------------------------------------------------------------------------
def relu(z: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, z)


def relu_derivative_from_preact(z: np.ndarray) -> np.ndarray:
    """ReLU'(z) = 1 if z > 0 else 0  (computed from the pre-activation z)."""
    return (z > 0).astype(z.dtype)


# ---------------------------------------------------------------------------
# Tanh
# ---------------------------------------------------------------------------
def tanh(z: np.ndarray) -> np.ndarray:
    return np.tanh(z)


def tanh_derivative(output: np.ndarray) -> np.ndarray:
    """tanh'(z) = 1 - a^2 , using the tanh OUTPUT a."""
    return 1.0 - output ** 2


# ---------------------------------------------------------------------------
# Softmax  (row-wise, numerically stable) -- used for multi-class output probs
# ---------------------------------------------------------------------------
def softmax(z: np.ndarray) -> np.ndarray:
    z = np.atleast_2d(z)
    z = z - z.max(axis=1, keepdims=True)   # shift for stability
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


# Registry so callers can select an activation by name (used in engine.py).
ACTIVATIONS = {
    "sigmoid": (sigmoid, sigmoid_derivative),
    "tanh": (tanh, tanh_derivative),
    # ReLU's derivative needs the pre-activation, handled specially in engine.
    "relu": (relu, None),
}
