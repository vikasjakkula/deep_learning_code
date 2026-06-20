"""
nnscratch
=========
A "Neural Network from Scratch" framework accelerated with Numba CUDA kernels.

The core network is pure Python (a list of layers, each a list of neuron dicts
keyed by ``'weights'``).  Heavy matrix math is offloaded to hand-written CUDA
kernels (``kernels.py`` + ``gpu.py``) that run on an NVIDIA GPU, with an
automatic NumPy/CPU fallback when no GPU is present.

Quick start
-----------
    from nnscratch import NeuralNetwork
    from nnscratch.data import xor_dataset

    net = NeuralNetwork([2, 2, 2], seed=1)
    net.train(xor_dataset(), l_rate=0.5, n_epoch=5000, batch_size=2)
    print(net.predict([1, 0]))   # -> 1
"""

from .engine import NeuralNetwork
from .kernels import CUDA_AVAILABLE, NUMBA_AVAILABLE
from . import gpu, data, metrics, activations, losses

__all__ = [
    "NeuralNetwork",
    "CUDA_AVAILABLE",
    "NUMBA_AVAILABLE",
    "gpu",
    "data",
    "metrics",
    "activations",
    "losses",
]

__version__ = "0.1.0"
