"""
gpu_mlp.py
==========
A *vectorised, batched* multi-layer perceptron whose heavy matrix multiplies --
in BOTH the forward pass and backpropagation -- run through the CUDA kernels in
``gpu.py``.  This is the module to use when you want to actually train on the GPU
for a long time and get a strong model.

How it differs from ``engine.NeuralNetwork``
-------------------------------------------
``engine.NeuralNetwork`` keeps the readable "list of neuron dictionaries"
structure and does backprop one sample at a time in pure Python -- wonderful for
learning, but CPU-bound.  ``MLPGPU`` instead stores each layer as plain weight
*matrices* and processes a whole mini-batch at once, so every expensive
``A @ W`` / ``Aᵀ @ dZ`` / ``dZ @ Wᵀ`` becomes one big matrix multiply handed to
the GPU.  The cheap elementwise bits (bias add, activation, its derivative) stay
in NumPy -- they are O(n²) next to the O(n³) matmuls, so the GPU is doing the
real work.

Architecture
------------
* Hidden layers: ReLU (default) or sigmoid.
* Output layer : softmax + cross-entropy loss (multi-class classification).
* Optimiser    : mini-batch gradient descent with momentum and LR decay.
* Everything is float32 (what the GPU likes) and reproducible from a seed.
"""

from __future__ import annotations

import time
from typing import List, Optional

import numpy as np

from . import gpu


# ---------------------------------------------------------------------------
# small numpy helpers (elementwise -> cheap, kept on CPU)
# ---------------------------------------------------------------------------
def _relu(Z):
    return np.maximum(0.0, Z)


def _relu_grad(Z):
    return (Z > 0.0).astype(np.float32)


def _sigmoid(Z):
    out = np.empty_like(Z)
    pos = Z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-Z[pos]))
    ez = np.exp(Z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def _sigmoid_grad_from_act(A):
    return A * (1.0 - A)


def _softmax(Z):
    Z = Z - Z.max(axis=1, keepdims=True)
    e = np.exp(Z)
    return e / e.sum(axis=1, keepdims=True)


class MLPGPU:
    """Batched MLP trained with GPU matrix multiplies."""

    def __init__(self, layer_sizes: List[int], seed: int = 1,
                 hidden_activation: str = "relu", use_gpu: bool = True):
        if len(layer_sizes) < 2:
            raise ValueError("need at least [n_in, n_out]")
        self.layer_sizes = list(layer_sizes)
        self.hidden_activation = hidden_activation
        self.use_gpu = use_gpu and gpu.CUDA_AVAILABLE
        self.n_layers = len(layer_sizes) - 1

        rng = np.random.default_rng(seed)
        self.W: List[np.ndarray] = []
        self.b: List[np.ndarray] = []
        for i in range(self.n_layers):
            fan_in, fan_out = layer_sizes[i], layer_sizes[i + 1]
            if hidden_activation == "relu":
                # He initialisation -- the right scale for ReLU networks.
                std = np.sqrt(2.0 / fan_in)
                Wl = rng.standard_normal((fan_in, fan_out)) * std
            else:
                # Xavier/Glorot -- the right scale for sigmoid/tanh networks.
                limit = np.sqrt(6.0 / (fan_in + fan_out))
                Wl = rng.uniform(-limit, limit, size=(fan_in, fan_out))
            self.W.append(Wl.astype(np.float32))
            self.b.append(np.zeros((fan_out,), dtype=np.float32))

        # momentum velocities (allocated lazily on first update)
        self._vW: Optional[List[np.ndarray]] = None
        self._vb: Optional[List[np.ndarray]] = None

    # ------------------------------------------------------------------
    # the one place matmuls happen -> GPU when available
    # ------------------------------------------------------------------
    def _mm(self, A, B):
        if self.use_gpu:
            return gpu.gpu_matmul(A, B)            # CUDA kernels
        return np.asarray(A, np.float32) @ np.asarray(B, np.float32)

    def _act(self, Z):
        return _relu(Z) if self.hidden_activation == "relu" else _sigmoid(Z)

    def _act_grad(self, Z, A):
        if self.hidden_activation == "relu":
            return _relu_grad(Z)
        return _sigmoid_grad_from_act(A)

    # ------------------------------------------------------------------
    # forward / backward (whole batch at once)
    # ------------------------------------------------------------------
    def forward(self, X):
        """Return (activations, pre_activations). Last activation is softmax."""
        X = np.ascontiguousarray(X, np.float32)
        A = [X]              # A[0] = inputs
        Z = []               # pre-activations per layer
        cur = X
        for li in range(self.n_layers):
            z = self._mm(cur, self.W[li]) + self.b[li]   # GPU matmul + bias
            Z.append(z)
            if li < self.n_layers - 1:
                cur = self._act(z)                       # hidden activation
            else:
                cur = _softmax(z)                        # output
            A.append(cur)
        return A, Z

    def backward(self, A, Z, Y):
        """Backprop. Returns (dW, db). Y is one-hot, shape (batch, n_out)."""
        batch = A[0].shape[0]
        dW = [None] * self.n_layers
        db = [None] * self.n_layers

        # softmax + cross-entropy -> clean gradient at the output.
        dZ = (A[-1] - Y).astype(np.float32)
        for li in reversed(range(self.n_layers)):
            A_prev = A[li]
            # grads for this layer's weights (GPU matmul on the big one)
            dW[li] = (self._mm(A_prev.T, dZ) / batch).astype(np.float32)
            db[li] = (dZ.sum(axis=0) / batch).astype(np.float32)
            if li > 0:
                dA_prev = self._mm(dZ, self.W[li].T)     # GPU matmul
                dZ = (dA_prev * self._act_grad(Z[li - 1], A[li])).astype(np.float32)
        return dW, db

    # ------------------------------------------------------------------
    # parameter update (momentum + lr)
    # ------------------------------------------------------------------
    def _step(self, dW, db, lr, momentum):
        if self._vW is None:
            self._vW = [np.zeros_like(w) for w in self.W]
            self._vb = [np.zeros_like(b) for b in self.b]
        for li in range(self.n_layers):
            if momentum > 0:
                self._vW[li] = momentum * self._vW[li] - lr * dW[li]
                self._vb[li] = momentum * self._vb[li] - lr * db[li]
                self.W[li] += self._vW[li]
                self.b[li] += self._vb[li]
            else:
                self.W[li] -= lr * dW[li]
                self.b[li] -= lr * db[li]

    # ------------------------------------------------------------------
    # one epoch
    # ------------------------------------------------------------------
    def _clip(self, dW, db, max_norm):
        """Global-norm gradient clipping -- the standard cure for the loss
        'spiking' that makes long, high-momentum runs diverge.  If the combined
        gradient is longer than ``max_norm`` we scale the whole thing down so the
        update direction is kept but the step size stays sane."""
        if not max_norm or max_norm <= 0:
            return dW, db
        sq = sum(float(np.sum(g * g)) for g in dW) + \
            sum(float(np.sum(g * g)) for g in db)
        norm = np.sqrt(sq)
        if norm > max_norm:
            scale = max_norm / (norm + 1e-12)
            dW = [g * scale for g in dW]
            db = [g * scale for g in db]
        return dW, db

    def train_epoch(self, X, Y, lr, batch_size, momentum, rng, max_grad_norm=5.0):
        n = X.shape[0]
        idx = rng.permutation(n)
        X, Y = X[idx], Y[idx]
        total_loss = 0.0
        for start in range(0, n, batch_size):
            xb = X[start:start + batch_size]
            yb = Y[start:start + batch_size]
            A, Z = self.forward(xb)
            probs = np.clip(A[-1], 1e-12, 1.0)
            total_loss += float(-np.sum(yb * np.log(probs)))
            dW, db = self.backward(A, Z, yb)
            dW, db = self._clip(dW, db, max_grad_norm)
            self._step(dW, db, lr, momentum)
        return total_loss / n

    # ------------------------------------------------------------------
    # inference / evaluation
    # ------------------------------------------------------------------
    def predict_proba(self, X, batch_size: int = 4096):
        outs = []
        for start in range(0, X.shape[0], batch_size):
            A, _ = self.forward(X[start:start + batch_size])
            outs.append(A[-1])
        return np.vstack(outs)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def evaluate(self, X, y):
        pred = self.predict(X)
        return float(np.mean(pred == np.asarray(y)))

    # ------------------------------------------------------------------
    # persistence
    # ------------------------------------------------------------------
    def save(self, path: str):
        d = {"layer_sizes": np.array(self.layer_sizes),
             "hidden_activation": self.hidden_activation}
        for i in range(self.n_layers):
            d[f"W{i}"] = self.W[i]
            d[f"b{i}"] = self.b[i]
        np.savez(path, **d)

    @classmethod
    def load(cls, path: str, use_gpu: bool = True):
        z = np.load(path, allow_pickle=True)
        layer_sizes = [int(x) for x in z["layer_sizes"]]
        act = str(z["hidden_activation"])
        net = cls(layer_sizes, hidden_activation=act, use_gpu=use_gpu)
        for i in range(net.n_layers):
            net.W[i] = z[f"W{i}"].astype(np.float32)
            net.b[i] = z[f"b{i}"].astype(np.float32)
        return net

    def n_params(self) -> int:
        return sum(w.size for w in self.W) + sum(b.size for b in self.b)


def one_hot(y, n_classes: int) -> np.ndarray:
    y = np.asarray(y, dtype=int)
    Y = np.zeros((y.shape[0], n_classes), dtype=np.float32)
    Y[np.arange(y.shape[0]), y] = 1.0
    return Y
