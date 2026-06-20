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

### CPU only (works everywhere)
```bash
pip install numpy
```

### GPU — verified working recipe (no conda, no system CUDA toolkit)
Tested on **Windows 11 + RTX 3050 6GB (CC 8.6), driver 591.44 (CUDA 13.1),
Python 3.12.10**. The CUDA compiler (NVVM) and runtime come entirely from pip
wheels:

```bash
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install -U pip
.venv\Scripts\python -m pip install -r requirements-gpu.txt
# verify:
.venv\Scripts\python -c "import nnscratch; from nnscratch import gpu; print(gpu.device_info())"
# -> CUDA device: NVIDIA GeForce RTX 3050 6GB Laptop GPU (compute capability 8.6)
```

> **Why a bootstrap?** `nnscratch/_cuda_setup.py` runs automatically on import and
> wires Numba up to the pip-wheel CUDA toolkit: it sets `CUDA_HOME` to the
> `nvidia-cuda-nvcc-cu12` wheel (which carries `nvvm/bin` + `nvvm/libdevice`) and
> copies `cudart64_*.dll` next to it so Numba can read the runtime version and
> detect the GPU's compute capability. Without this, Numba 0.65 only finds CUDA
> via conda/system installs, and its newer "NVIDIA binding" path is incompatible
> with cuda-python 13.x. The bootstrap never overrides an existing `CUDA_HOME`.

> **Python version:** use **stable Python 3.12.x** for the GPU path (Numba ships
> cp312 wheels; alphas like 3.12.0a3 and very new interpreters such as 3.14 are
> not supported by Numba). The CPU fallback runs on any version.

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

## Measured results (RTX 3050, float32)

Real numbers from this machine (`python benchmark.py`):

| Size       | Naive CPU (est.) | NumPy BLAS | GPU kernels | **GPU vs Naive** |
|------------|------------------|------------|-------------|------------------|
| 1000×1000  | ~32.5 s          | 4.95 ms    | 14.7 ms     | **~2200×**       |
| 2000×2000  | ~133 s           | 29.8 ms    | 105 ms      | **~2500×**       |

The headline goal — **GPU ≫ ~1000× faster than the naive triple-loop** — is met
(~2200–2500×), with GPU output matching NumPy to float32 precision
(`max|GPU−NumPy| ≈ 2e-4`). NumPy's hand-tuned multithreaded BLAS still wins at
these sizes once host↔device copies are counted; the GPU's advantage over BLAS
grows with problem size and arithmetic intensity. Training (XOR, seeds, heart)
reaches 100% accuracy and the GPU batched-inference path agrees 100% with the
per-sample CPU forward pass.

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
