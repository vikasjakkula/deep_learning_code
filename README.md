# nnscratch — Neural Network from Scratch, accelerated with Numba CUDA

A modular, pure-Python neural-network framework that **avoids PyTorch /
TensorFlow for its core logic** and instead pushes the heavy matrix math onto an
NVIDIA GPU using hand-written **Numba CUDA kernels**.

The network *structure* is deliberately simple and readable — a **list of
layers, where each layer is a list of neuron dictionaries**, and each neuron
dict holds a `'weights'` key of length `n_inputs + 1` (the last value is the
bias). This is the KMIT Vista teaching convention. The *performance* comes from
treating big matrix multiplies as an "expressway": one GPU thread per output
cell, launched over a rounded-up 16×16 grid.

> Hardware target: **NVIDIA RTX 3050 Laptop GPU** (Ampere, CC 8.6).
> Runs anywhere thanks to an automatic NumPy/CPU fallback — only the *real*
> ~1000× speed-up needs the GPU.

This project was built from the source material in `../deep_learning`
(from-scratch SGD / mini-batch / momentum / LR-decay XOR programs + PyTorch
reference notebooks) and the sibling `everything-llms`, `minialign-rlhf-pipeline`
and `redsynth-redteam-engine` repositories.

---

## Why two execution paths?

| Concept        | "Naive" / slow lane                          | "Parallel" / expressway                          |
|----------------|----------------------------------------------|--------------------------------------------------|
| Matrix multiply| triple-nested `for` loops, **one CPU core**  | one **GPU thread per output cell**, all at once  |
| Where          | `gpu.naive_matmul`                           | `kernels.matmul_kernel` via `gpu.gpu_matmul`     |
| Forward pass   | `engine.forward_propagate` (per sample)      | `engine.forward_batch_gpu` (batched, on device)  |
| Cost model     | O(M·N·K) executed sequentially in Python     | same work spread across thousands of cores       |

The naive loop visits every output cell one after another. The CUDA kernel asks
`cuda.grid(2)` *“which (row, col) am I?”* and computes only that cell — so all
M×N cells are produced simultaneously.

---

## Project layout

```
deep_learning_code/
├── nnscratch/
│   ├── kernels.py      # @cuda.jit kernels: matmul, tiled matmul, bias, sigmoid, relu
│   ├── gpu.py          # host orchestration: to_device/copy_to_host, grid config,
│   │                   #   cuda.synchronize, naive_matmul, numpy_matmul, fallbacks
│   ├── engine.py       # NeuralNetwork (dict-of-weights): init, forward, backprop,
│   │                   #   SGD/mini-batch/momentum/LR-decay, GPU batched forward
│   ├── activations.py  # sigmoid (+ relu/tanh/softmax) and derivatives
│   ├── losses.py       # SSE / MSE / cross-entropy and gradients
│   ├── data.py         # seeds / heart / moons loaders + synthetic generators,
│   │                   #   normalisation, train/test split
│   └── metrics.py      # accuracy + confusion matrix
├── train.py            # train on seeds (7 feat / 3 cls) or heart (13 feat / 2 cls)
├── benchmark.py        # 1000×1000 matmul: naive CPU vs NumPy vs GPU
├── examples/xor_demo.py
├── tests/test_smoke.py
├── data/               # drop seeds_dataset.csv / heart.csv here to use real data
├── requirements.txt
└── setup.py
```

---

## Install

```bash
# core (CPU paths, data, benchmark reference) — works everywhere
pip install numpy

# GPU acceleration — on the RTX 3050 machine (needs NVIDIA driver + CUDA toolkit)
pip install numba
# (optionally)  pip install -e .          # installs the nnscratch package
```

> **Note on Python versions:** Numba tracks CUDA support a little behind the
> newest CPython. If `import numba` fails on a very new interpreter, use a
> Python version Numba supports (3.10–3.12 are safe) inside a venv. The CPU
> fallback runs on any version.

---

## Quick start

```python
from nnscratch import NeuralNetwork
from nnscratch.data import xor_dataset

net = NeuralNetwork([2, 2, 2], seed=1)            # 2 in → 2 hidden → 2 out
net.train(xor_dataset(), l_rate=0.5, n_epoch=5000,
          batch_size=2, momentum=0.9)             # mini-batch GD + momentum
print([net.predict(r[:-1]) for r in xor_dataset()])   # -> [0, 1, 1, 0]
```

### Run the scripts

```bash
python examples/xor_demo.py                 # learn XOR from scratch
python train.py --dataset seeds             # 7 features, 3 classes
python train.py --dataset heart --epochs 300
python benchmark.py                         # 1000x1000 GPU vs naive CPU
python tests/test_smoke.py                  # or: python -m pytest tests/ -q
```

`train.py` flags: `--hidden 16 8`, `--epochs`, `--batch-size`, `--lr`,
`--momentum`, `--lr-decay`, `--seed`.

---

## What the brief asked for — and where it lives

* **Network = list of layers; layer = list of neuron dicts; `'weights'` has
  `n_inputs+1` (bias last)** → `engine.NeuralNetwork._initialize_network`
* **Reproducible seeded init** → `seed=` argument (default Xavier; pass
  `init="uniform01"` for the original `random()` scheme)
* **Linear activation `z = W·x + b`** → `engine.activate`
* **Sigmoid transfer `1/(1+e^-x)`** → `engine.transfer` / `activations.sigmoid`
* **Forward propagation looping layer→layer** → `engine.forward_propagate`
* **Backpropagation** → `engine.backward_propagate_error` (+ mini-batch grad
  accumulation, momentum, LR-decay in `engine.train`)
* **`@cuda.jit` matmul kernel using `cuda.grid(2)`** → `kernels.matmul_kernel`
  (plus a shared-memory tiled variant `matmul_tiled_kernel`)
* **16×16 threads-per-block grid** → `kernels.TPB = 16`, `gpu.grid_2d`
* **Memory management `cuda.to_device` / `copy_to_host`** → `gpu.gpu_matmul`
* **`cuda.synchronize()` after launches** → every launcher in `gpu.py`
* **Round the grid UP (`block-1` before integer division)** → `gpu.grid_2d`
  uses `(n + tpb - 1) // tpb`, with bounds guards in every kernel
* **Naive triple-loop CPU matmul** → `gpu.naive_matmul`
* **Benchmark 1000×1000 GPU vs naive** → `benchmark.py`
* **Train on a real dataset (seeds / heart)** → `train.py` + `data.py`

---

## Notes on correctness & fairness

* The GPU batched forward pass (`forward_batch_gpu`) is cross-checked against the
  per-sample pure-Python forward pass in both `tests/` and `train.py`
  (agreement is 100%).
* `benchmark.py` times the naive loops at a smaller size and extrapolates by the
  O(n³) law (interpreted Python is genuinely slow), while NumPy and the GPU are
  always timed at the full size. It prints the GPU-vs-naive and GPU-vs-NumPy
  speed-ups and a correctness check.

## License

MIT (see `setup.py`). Educational project.
