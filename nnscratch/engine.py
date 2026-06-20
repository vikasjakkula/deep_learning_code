"""
engine.py
=========
The ``NeuralNetwork`` class -- the readable, pure-Python heart of the framework.

Data structure (exactly the KMIT Vista convention)
--------------------------------------------------
A network is a **list of layers**.  Each layer is a **list of neuron dicts**.
Each neuron dict has a ``'weights'`` key holding ``n_inputs + 1`` numbers, where
the **last value is the bias**::

    network = [
        # hidden layer
        [ {'weights': [w0, w1, ..., bias]}, {'weights': [...]} ],
        # output layer
        [ {'weights': [...]}, {'weights': [...]} ],
    ]

During forward/backward passes each neuron dict also gains transient keys:
``'output'`` (its activation a), ``'delta'`` (its error signal), and -- when
momentum is enabled -- ``'velocity'`` (one per weight).

Two execution paths
-------------------
*   ``forward_propagate(row)`` -- the classic, one-sample-at-a-time pure-Python
    pass.  Clear and faithful to the teaching material; used during training.
*   ``forward_batch_gpu(X)`` -- packs the dict weights into matrices and runs the
    whole batch through the Numba CUDA kernels in ``gpu.py``.  Same math, vastly
    faster for large batches / wide layers.  This is where the "expressway"
    plugs into the network.

Training supports SGD, mini-batch GD, classical **momentum**, and **learning-
rate decay** -- mirroring the progression of the source curriculum.
"""

from __future__ import annotations

import math
import random
from typing import List, Dict, Optional

import numpy as np

from .activations import sigmoid_scalar, sigmoid_derivative
from . import gpu


Neuron = Dict[str, object]
Layer = List[Neuron]
Network = List[Layer]


class NeuralNetwork:
    """A fully-connected feed-forward network stored as lists of neuron dicts."""

    # ------------------------------------------------------------------
    # Construction / initialization
    # ------------------------------------------------------------------
    def __init__(self, layer_sizes: List[int], seed: int = 1,
                 activation: str = "sigmoid", init: str = "xavier"):
        """Build the network with reproducible random weights.

        Parameters
        ----------
        layer_sizes : e.g. [n_inputs, n_hidden1, ..., n_outputs]
            Number of UNITS in each layer, input layer first.  ``[2, 2, 2]`` is
            the XOR network: 2 inputs -> 2 hidden -> 2 outputs.
        seed : fixed RNG seed so every run is identical (reproducibility).
        activation : transfer function name (only "sigmoid" is fully wired for
            backprop here, matching the curriculum).
        init : "xavier" (default, symmetric Glorot-uniform -- trains deep
            sigmoid nets reliably) or "uniform01" (the plain ``random()`` in
            [0,1) used by the original from-scratch curriculum).
        """
        if len(layer_sizes) < 2:
            raise ValueError("layer_sizes needs at least [n_inputs, n_outputs]")
        self.layer_sizes = list(layer_sizes)
        self.activation = activation
        self.seed = seed
        self.init = init
        self.network: Network = self._initialize_network(layer_sizes, seed, init)

    @staticmethod
    def _initialize_network(layer_sizes: List[int], seed: int,
                            init: str = "xavier") -> Network:
        """Create layers of neuron dicts with random weights (+1 for the bias).

        A fixed seed makes initialization deterministic -- essential so the
        benchmark and training runs are reproducible.

        Why Xavier/Glorot by default?  The original curriculum draws weights from
        ``random()`` in [0, 1) -- fine for the tiny 2-2-2 XOR net, but in a
        deeper net every initial weight being positive pushes sigmoids straight
        into saturation, the gradients vanish, and a whole class can collapse.
        Glorot-uniform draws symmetric weights in ``[-L, L]`` with
        ``L = sqrt(6 / (fan_in + fan_out))``, keeping activation variance stable
        across layers so even multi-hidden-layer sigmoid nets learn well.
        """
        rng = random.Random(seed)
        network: Network = []
        # Each layer after the input gets one neuron per unit; every neuron has
        # (fan_in + 1) weights, the final one being the bias.
        for li in range(1, len(layer_sizes)):
            n_inputs = layer_sizes[li - 1]
            n_units = layer_sizes[li]
            if init == "xavier":
                limit = math.sqrt(6.0 / (n_inputs + n_units))
                layer = [
                    {"weights": [rng.uniform(-limit, limit)
                                 for _ in range(n_inputs)] + [0.0]}  # bias = 0
                    for _ in range(n_units)
                ]
            else:  # "uniform01" -- faithful to the original curriculum
                layer = [
                    {"weights": [rng.random() for _ in range(n_inputs + 1)]}
                    for _ in range(n_units)
                ]
            network.append(layer)
        return network

    # ------------------------------------------------------------------
    # Forward pass -- pure Python, one sample at a time
    # ------------------------------------------------------------------
    @staticmethod
    def activate(weights: List[float], inputs: List[float]) -> float:
        """Linear activation  z = W.x + b  (bias is the last weight)."""
        activation = weights[-1]  # bias
        for i in range(len(weights) - 1):
            activation += weights[i] * inputs[i]
        return activation

    @staticmethod
    def transfer(activation: float) -> float:
        """Non-linear transfer: the sigmoid 1 / (1 + e^-x)."""
        return sigmoid_scalar(activation)

    def forward_propagate(self, row: List[float]) -> List[float]:
        """Run one input ``row`` (features only) through every layer.

        Stores each neuron's ``'output'`` for use in backprop and returns the
        final layer's outputs.
        """
        inputs = list(row)
        for layer in self.network:
            new_inputs = []
            for neuron in layer:
                z = self.activate(neuron["weights"], inputs)
                neuron["output"] = self.transfer(z)
                new_inputs.append(neuron["output"])
            inputs = new_inputs
        return inputs

    # ------------------------------------------------------------------
    # Backward pass (backpropagation)
    # ------------------------------------------------------------------
    @staticmethod
    def transfer_derivative(output: float) -> float:
        """Sigmoid derivative in terms of its output:  a * (1 - a)."""
        return output * (1.0 - output)

    def backward_propagate_error(self, expected: List[float]) -> None:
        """Compute the ``'delta'`` for every neuron, output layer first.

        Output layer:  delta = (output - expected) * f'(output)
        Hidden layer:  delta = (sum of next-layer weight*delta) * f'(output)
        """
        for i in reversed(range(len(self.network))):
            layer = self.network[i]
            errors = []
            if i != len(self.network) - 1:
                # Hidden layer: propagate error back from the layer in front.
                for j in range(len(layer)):
                    error = 0.0
                    for neuron in self.network[i + 1]:
                        error += neuron["weights"][j] * neuron["delta"]
                    errors.append(error)
            else:
                # Output layer: gradient of squared error w.r.t. output.
                errors = [layer[j]["output"] - expected[j] for j in range(len(layer))]
            for j, neuron in enumerate(layer):
                neuron["delta"] = errors[j] * self.transfer_derivative(neuron["output"])

    # ------------------------------------------------------------------
    # Mini-batch gradient machinery
    # ------------------------------------------------------------------
    def _zero_gradients(self):
        """Gradient buffer with the same shape as the weights, all zeros."""
        return [[[0.0] * len(neuron["weights"]) for neuron in layer]
                for layer in self.network]

    def _init_momentum(self):
        """Give every neuron a zero ``'velocity'`` vector (for momentum)."""
        for layer in self.network:
            for neuron in layer:
                neuron["velocity"] = [0.0] * len(neuron["weights"])

    def _accumulate_gradients(self, row: List[float], grads) -> None:
        """Add one sample's gradients (delta * input) into the batch buffer."""
        for i in range(len(self.network)):
            inputs = row[:-1] if i == 0 else [n["output"] for n in self.network[i - 1]]
            for n_idx, neuron in enumerate(self.network[i]):
                delta = neuron["delta"]
                for j in range(len(inputs)):
                    grads[i][n_idx][j] += delta * inputs[j]
                grads[i][n_idx][-1] += delta  # bias gradient (input == 1.0)

    def _apply_gradients(self, grads, l_rate: float, batch_size: int,
                         momentum: float = 0.0) -> None:
        """Update weights from the averaged batch gradient.

        Plain SGD:        w -= l_rate * g
        With momentum:    v = momentum*v - l_rate*g ;  w += v
        """
        for i, layer in enumerate(self.network):
            for n_idx, neuron in enumerate(layer):
                for j in range(len(neuron["weights"])):
                    g = grads[i][n_idx][j] / float(batch_size)
                    if momentum > 0.0:
                        v = momentum * neuron["velocity"][j] - l_rate * g
                        neuron["velocity"][j] = v
                        neuron["weights"][j] += v
                    else:
                        neuron["weights"][j] -= l_rate * g

    # ------------------------------------------------------------------
    # Training loop  (SGD / mini-batch / momentum / LR-decay all in one)
    # ------------------------------------------------------------------
    def train(self, dataset, l_rate: float = 0.5, n_epoch: int = 5000,
              n_outputs: Optional[int] = None, batch_size: int = 1,
              momentum: float = 0.0, lr_decay: float = 0.0,
              shuffle: bool = True, seed: int = 1,
              verbose_every: int = 1000) -> List[float]:
        """Train on ``dataset`` rows of ``[f1, f2, ..., label_int]``.

        Setting ``batch_size=1`` recovers SGD; larger values give mini-batch GD.
        ``momentum>0`` enables classical momentum; ``lr_decay>0`` shrinks the
        learning rate each epoch via ``l_rate / (1 + lr_decay * epoch)``.

        Returns the per-epoch error history (sum of squared errors).
        """
        if n_outputs is None:
            n_outputs = self.layer_sizes[-1]
        if momentum > 0.0:
            self._init_momentum()

        rng = random.Random(seed)
        history: List[float] = []
        data = list(dataset)

        for epoch in range(n_epoch):
            # Learning-rate decay: large steps early, fine steps later.
            cur_lr = l_rate / (1.0 + lr_decay * epoch) if lr_decay > 0 else l_rate

            if shuffle:
                rng.shuffle(data)

            sum_error = 0.0
            for start in range(0, len(data), batch_size):
                batch = data[start:start + batch_size]
                grads = self._zero_gradients()
                for row in batch:
                    outputs = self.forward_propagate(row[:-1])
                    expected = [0.0] * n_outputs
                    expected[int(row[-1])] = 1.0
                    sum_error += sum((expected[k] - outputs[k]) ** 2
                                     for k in range(n_outputs))
                    self.backward_propagate_error(expected)
                    self._accumulate_gradients(row, grads)
                self._apply_gradients(grads, cur_lr, len(batch), momentum)

            history.append(sum_error)
            if verbose_every and (epoch % verbose_every == 0):
                print(f"  epoch={epoch:5d}  lr={cur_lr:.5f}  error={sum_error:.6f}")
        return history

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def predict(self, row: List[float]) -> int:
        """Return the class index with the highest output for one sample."""
        outputs = self.forward_propagate(row)
        return int(max(range(len(outputs)), key=lambda k: outputs[k]))

    def predict_proba(self, row: List[float]) -> List[float]:
        """Return the raw output activations for one sample."""
        return self.forward_propagate(row)

    # ------------------------------------------------------------------
    # GPU-accelerated batched forward pass
    # ------------------------------------------------------------------
    def to_matrices(self):
        """Pack the dict-of-weights into ``(W_list, b_list)`` NumPy matrices.

        For layer ``l``: ``W`` has shape (fan_in, units) and ``b`` shape (units,).
        This is the bridge between the readable dict structure and the matrix
        form the CUDA kernels consume.
        """
        weight_mats, bias_vecs = [], []
        for layer in self.network:
            units = len(layer)
            fan_in = len(layer[0]["weights"]) - 1
            W = np.empty((fan_in, units), dtype=np.float32)
            b = np.empty((units,), dtype=np.float32)
            for j, neuron in enumerate(layer):
                W[:, j] = neuron["weights"][:-1]
                b[j] = neuron["weights"][-1]
            weight_mats.append(W)
            bias_vecs.append(b)
        return weight_mats, bias_vecs

    def forward_batch_gpu(self, X: np.ndarray) -> np.ndarray:
        """Forward-propagate a whole batch ``X`` (rows = samples) on the GPU.

        Produces identical results to calling ``forward_propagate`` on each row,
        but the per-layer ``X @ W + b`` then sigmoid run as CUDA kernels.  Falls
        back to NumPy automatically when no GPU is present.
        """
        X = np.ascontiguousarray(X, dtype=np.float32)
        W_list, b_list = self.to_matrices()
        out = X
        for W, b in zip(W_list, b_list):
            out = gpu.gpu_linear_activation(out, W, b, activation="sigmoid")
        return out

    def predict_batch(self, X: np.ndarray) -> np.ndarray:
        """Class predictions for a batch using the GPU forward path."""
        probs = self.forward_batch_gpu(X)
        return np.argmax(probs, axis=1)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------
    def summary(self) -> str:
        lines = ["NeuralNetwork (dict-of-weights)"]
        lines.append(f"  architecture : {self.layer_sizes}")
        lines.append(f"  activation   : {self.activation}")
        total = 0
        for li, layer in enumerate(self.network):
            params = sum(len(n["weights"]) for n in layer)
            total += params
            lines.append(f"  layer {li}: {len(layer)} neurons, {params} params")
        lines.append(f"  total trainable params: {total}")
        return "\n".join(lines)
