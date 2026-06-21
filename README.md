# 🧠⚡ nnscratch — a tiny brain you build yourself, made super fast by your gaming GPU

> **Explain-it-like-I'm-5:**
> A computer "brain" is just a big pile of **guesses with knobs**. You show it
> examples, it guesses, you tell it how wrong it was, and it turns its knobs a
> little to guess better next time. Do that thousands of times and it gets smart.
>
> Turning all those knobs needs a LOT of tiny sums. A normal computer does the
> sums **one at a time** (slow). Your graphics card (**GPU**) does **thousands at
> once** (fast). This project builds the brain by hand *and* teaches the GPU to do
> the sums super fast. 🚀

We do **not** use big ready-made libraries (PyTorch / TensorFlow) for the brain
itself — we build every part ourselves so you can *see* how it works. The only
"magic helper" is **Numba CUDA**, which lets us write the fast GPU sums in Python.

---

## 🍎 The big ideas, in kid words

| Fancy word | What it really means | Like… |
|---|---|---|
| **Neuron** | one little guesser with knobs (called *weights*) and one extra knob (*bias*) | a kid raising their hand with an opinion |
| **Layer** | a row of neurons working together | a row of kids in class |
| **Network** | many layers stacked, one feeding the next | grade 1 → grade 2 → grade 3 |
| **Forward pass** | put numbers in the front, an answer pops out the back | a vending machine |
| **Activation (z = W·x + b)** | mix the inputs with the knobs and add the bias | mixing ingredients |
| **Transfer / Sigmoid** | squish any number into a 0–1 "how sure am I?" | a volume dial from quiet to loud |
| **Loss** | a number for *how wrong* the guess was | your score in a game (lower = better) |
| **Backpropagation** | figure out which knob caused the mistake | "who knocked over the vase?" |
| **Gradient descent** | nudge every knob a tiny bit toward "less wrong" | walking downhill to the lowest spot |
| **Learning rate** | how big each nudge is | baby steps vs giant steps |
| **Momentum** | keep rolling the way you were going so you don't zig-zag | a ball rolling downhill |
| **Mini-batch** | learn from a small handful of examples at a time | flashcards in small stacks |
| **Epoch** | one full trip through all the examples | reading the whole book once |
| **Accuracy** | how many guesses were right | your test grade |

---

## 🐢 vs 🐇 The whole point: slow lane vs fast lane

A big "times table" (matrix multiply) is the heaviest math in a brain. We do it two ways:

| | 🐢 **Naive / slow lane** | 🐇 **GPU / fast lane** |
|---|---|---|
| How | one worker fills in every box of the answer, one box at a time | **thousands of GPU workers**, each fills in ONE box, all at once |
| In the code | `gpu.naive_matmul` (plain Python loops) | `kernels.matmul_kernel` → `gpu.gpu_matmul` |
| Speed (1000×1000) | about **32 seconds** 😴 | about **0.015 seconds** ⚡ |

> Each GPU worker just asks *"which box is mine?"* (`cuda.grid(2)`), fills only
> that box, and they all finish together. That's why it's ~**2,000× faster** than
> the slow lane on the same job.

---

## 🧩 How the brain is stored (super simple)

The whole brain is just **lists and dictionaries** — stuff you already know:

```python
network = [
    # Layer 1 (hidden): two neurons
    [ {'weights': [0.3, -0.1, 0.05]},   # last number is the BIAS
      {'weights': [0.7,  0.2, -0.4]} ],
    # Layer 2 (output): two neurons
    [ {'weights': [0.1, -0.6, 0.2]},
      {'weights': [-0.3, 0.8, 0.05]} ],
]
```

Each neuron is a little box with a `'weights'` list. The **last weight is the
bias**. That's it — no hidden magic. (Lives in `engine.py`.)

---

## 🗂️ What's in the box (project map)

```
deep_learning_code/
├── nnscratch/
│   ├── engine.py       # 🧠 the brain: build it, guess (forward), learn (backprop),
│   │                   #    train with mini-batch + momentum + learning-rate decay
│   ├── kernels.py      # ⚡ the GPU sums: matmul, faster "tiled" matmul, sigmoid, relu
│   ├── gpu.py          # 🚚 moves data to/from the GPU, sets up the workers, slow-lane matmul
│   ├── _cuda_setup.py  # 🔌 auto-connects Numba to the GPU (so it "just works")
│   ├── activations.py  # 🎚️ sigmoid / relu / tanh / softmax (the "how sure?" squishers)
│   ├── losses.py       # 📉 how-wrong scores (squared error, cross-entropy)
│   ├── data.py         # 📦 example datasets (seeds, heart, moons, XOR) + cleanup
│   └── metrics.py      # ✅ accuracy + a score grid (confusion matrix)
├── train.py            # 🏋️ teach the brain on real-ish data
├── benchmark.py        # 🏁 race the GPU vs the slow lane
├── examples/xor_demo.py# 👶 the classic "hello world" brain
└── tests/test_smoke.py # 🧪 quick checks that everything works
```

---

## ▶️ Try it (3 commands)

**1) Teach a brain the XOR puzzle:**
```bash
python examples/xor_demo.py
```
**2) Teach it to sort seeds (7 clues → 3 kinds of wheat):**
```bash
python train.py --dataset seeds
```
**3) Race the GPU against the slow lane:**
```bash
python benchmark.py
```

**4) See the predictions in a browser (web frontend):**
```bash
python export_web.py          # runs the trained model -> web/data.js
# then open web/index.html (no server needed)
```
A self-contained page that draws the model's predicted classes and learned
**decision boundary**, shows accuracy/confidence stats and an example table, and
includes a written summary of the project and of how ANNs / ML work. Match the
flags to your training, e.g. `--dataset spiral --classes 6 --spiral-noise 0.05`.

### 🏋️‍♂️ Train a STRONG model for a long time (real GPU training)

`engine.py` learns one example at a time (great for understanding, but CPU-bound).
For serious GPU training there's **`gpu_mlp.MLPGPU`** — a batched network whose
forward pass *and* backprop run as GPU matrix multiplies — plus **`train_long.py`**,
which trains for as long as you let it and **saves the best model continuously**
(so a multi-hour run is never wasted, and Ctrl+C stops cleanly).

```bash
# train for 30 minutes on a hard "spiral" problem; best model is checkpointed
.venv\Scripts\python train_long.py --minutes 30

# or run a fixed number of epochs on a big, deep network
.venv\Scripts\python train_long.py --epochs 5000 --hidden 256 128 64

# come back later and keep improving the same model
.venv\Scripts\python train_long.py --resume models/best_model.npz --minutes 60
```

It writes to `models/`: `best_model.npz` (highest accuracy so far),
`last_model.npz` (for resuming), and `training_log.csv` (the learning curve).
The default **spiral** dataset is non-linear and hard on purpose, so accuracy
keeps climbing the longer you train — that's the whole point. Key flags:
`--minutes`, `--epochs`, `--hidden`, `--batch-size`, `--lr`, `--momentum`,
`--lr-decay`, `--dataset {spiral,synthetic,seeds,heart,moons}`, `--resume`,
`--fast/--no-fast`, `--eval-every`, `--spiral-turns`, `--spiral-noise`.

**Make it harder:** more classes (arms) and more `--spiral-turns` wind the arms
more tightly, so the net needs many more epochs to separate them:

```bash
# a genuinely hard 6-arm spiral; reaches 100% + ~99.7% confidence
python train_long.py --dataset spiral --classes 6 --spiral-turns 2.5 \
    --hidden 512 512 256 --lr 0.06 --epochs 90 --fast
```

Then see the model's confidence (predicts like a near-sure event):

```bash
python evaluate_model.py --dataset spiral --classes 6 --spiral-noise 0.05
python plot_training.py        # writes models/training_curve.png
```

> **Low-memory machines:** the heavy math is on the GPU, but per-epoch metrics
> allocate a little host RAM. If your system memory is nearly full (small/disabled
> pagefile), a long run can be cut short by a memory error. Mitigations: raise
> `--eval-every` (e.g. `5`), keep `--fast` on, and close other apps or enlarge the
> Windows pagefile (System → Advanced → Performance → Virtual memory). The trainer
> already reuses buffers and skips a metric step rather than crashing.

Tiny code example:
```python
from nnscratch import NeuralNetwork
from nnscratch.data import xor_dataset

net = NeuralNetwork([2, 2, 2], seed=1)          # 2 in → 2 hidden → 2 out
net.train(xor_dataset(), l_rate=0.5, n_epoch=5000, batch_size=2, momentum=0.9)
print([net.predict(r[:-1]) for r in xor_dataset()])   # -> [0, 1, 1, 0]  🎉
```

---

## 🔧 Setup

### Easy mode (CPU — works on any computer)
```bash
pip install numpy
```
The brain still learns; the GPU parts quietly fall back to the normal computer.

### Fast mode (use your NVIDIA GPU)
Tested on **Windows 11 + RTX 3050, Python 3.12**:
```bash
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install -U pip
.venv\Scripts\python -m pip install -r requirements-gpu.txt
.venv\Scripts\python examples\xor_demo.py
```
> 💡 You need a real NVIDIA GPU and **stable Python 3.12** (not 3.14, and not an
> "alpha"). The file `_cuda_setup.py` does the annoying plumbing for you, so the
> GPU "just works" — no separate CUDA install needed.

---

## 🏁 Real results from the RTX 3050

| Job size | 🐢 Slow lane | 🐇 GPU | **GPU is this much faster** |
|---|---|---|---|
| 1000×1000 sums | ~32.5 s | 14.7 ms | **~2,200×** |
| 2000×2000 sums | ~133 s | 105 ms | **~2,500×** |

- The GPU's answers **match the normal answers exactly** (so fast *and* correct ✅).
- Training XOR, seeds, and heart all reach **100% correct**.
- The GPU's guesses agree with the slow-lane guesses **100% of the time**.

> Fun fact: at these sizes, the super-tuned NumPy library on the CPU is also very
> fast — the GPU's big win is mainly against the *naive* one-at-a-time loops, and
> it pulls further ahead as the math gets bigger.

---

## 🎓 What you learn from this project

- **How a neural network actually works** — every step by hand, no black box.
- **How GPUs make things fast** — splitting one big job into thousands of tiny
  jobs (parallel computing): `cuda.grid(2)`, 16×16 worker blocks, moving data
  to/from the GPU, and waiting for everyone to finish (`cuda.synchronize`).
- **Core machine-learning ideas** — forward pass, loss, backpropagation,
  gradient descent, mini-batches, momentum, learning-rate decay, train/test
  split, and measuring accuracy.

It's a **learning project**, not a replacement for PyTorch — but after building
this, the "real" tools stop feeling like magic. ✨

---

## 🗺️ Where each required piece lives (for graders/curious folks)

| Requirement | File |
|---|---|
| Brain = list of layers; layer = list of neuron dicts; `weights` ends in bias | `engine.py` |
| Reproducible random start (fixed seed) | `engine.py` (`seed=`) |
| `z = W·x + b` and sigmoid `1/(1+e⁻ˣ)` | `engine.py` / `activations.py` |
| Forward pass, backprop, mini-batch, momentum, LR-decay | `engine.py` |
| GPU matmul with `cuda.grid(2)`, 16×16 blocks | `kernels.py`, `gpu.py` |
| Move data with `to_device` / `copy_to_host`, then `cuda.synchronize()` | `gpu.py` |
| Round the grid **up** so any size works | `gpu.grid_2d` |
| Slow-lane triple-loop matmul | `gpu.naive_matmul` |
| 1000×1000 GPU-vs-slow benchmark | `benchmark.py` |
| Train on a real dataset (seeds / heart) | `train.py`, `data.py` |

## 📜 License
MIT — free to use and learn from. Made by **vikasjakkula** 🧑‍💻 with **Claude** 🤖.
