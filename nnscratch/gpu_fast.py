"""
gpu_fast.py
===========
A *device-resident* GPU trainer -- the fast path.

The problem with ``gpu_mlp.MLPGPU``
-----------------------------------
It calls ``gpu.gpu_matmul`` for every matrix multiply, and each of those copies
its inputs Host->Device and its result Device->Host.  For the small matrices in
an MLP, those copies dominate, so the GPU sits mostly idle (~15% utilisation).

What this module does differently
---------------------------------
Upload ONCE, compute MANY times.  The whole dataset, the weights, the momentum
buffers, and every scratch buffer live on the GPU for the entire run.  A full
forward pass + backprop is a chain of kernels that hand **device arrays**
straight to each other -- nothing comes back to the CPU between layers.  Only a
tiny per-epoch metric copy happens.  Result: far higher GPU utilisation and
faster epochs, so in a fixed time budget you train more and get a better model.

Everything is float32.  Falls back to ``gpu_mlp.MLPGPU`` (NumPy) if there's no
GPU, so callers always get a working trainer.
"""

from __future__ import annotations

import math
from typing import List

import numpy as np

from . import kernels  # importing this also runs the CUDA bootstrap
from .gpu import grid_2d

CUDA_AVAILABLE = kernels.CUDA_AVAILABLE
TPB = kernels.TPB

if CUDA_AVAILABLE:
    from numba import cuda, float32

    # ---- device kernels (all operate directly on GPU arrays) ----

    @cuda.jit
    def _mm_nn(A, B, C):                       # C = A @ B
        i, j = cuda.grid(2)
        if i < C.shape[0] and j < C.shape[1]:
            acc = float32(0.0)
            for k in range(A.shape[1]):
                acc += A[i, k] * B[k, j]
            C[i, j] = acc

    @cuda.jit
    def _mm_tn(A, B, C):                       # C = Aᵀ @ B  (A is (K,M))
        i, j = cuda.grid(2)
        if i < C.shape[0] and j < C.shape[1]:
            acc = float32(0.0)
            for k in range(A.shape[0]):
                acc += A[k, i] * B[k, j]
            C[i, j] = acc

    @cuda.jit
    def _mm_nt(A, B, C):                       # C = A @ Bᵀ  (B is (N,K))
        i, j = cuda.grid(2)
        if i < C.shape[0] and j < C.shape[1]:
            acc = float32(0.0)
            for k in range(A.shape[1]):
                acc += A[i, k] * B[j, k]
            C[i, j] = acc

    @cuda.jit
    def _add_bias(Z, b):                       # Z[i,j] += b[j]
        i, j = cuda.grid(2)
        if i < Z.shape[0] and j < Z.shape[1]:
            Z[i, j] += b[j]

    @cuda.jit
    def _relu_fwd(Z, A):                       # A = relu(Z)
        i, j = cuda.grid(2)
        if i < Z.shape[0] and j < Z.shape[1]:
            v = Z[i, j]
            A[i, j] = v if v > float32(0.0) else float32(0.0)

    @cuda.jit
    def _sigmoid_fwd(Z, A):                    # A = sigmoid(Z)
        i, j = cuda.grid(2)
        if i < Z.shape[0] and j < Z.shape[1]:
            A[i, j] = float32(1.0) / (float32(1.0) + math.exp(-Z[i, j]))

    @cuda.jit
    def _relu_grad_inplace(dZ, Z):            # dZ *= (Z > 0)
        i, j = cuda.grid(2)
        if i < dZ.shape[0] and j < dZ.shape[1]:
            if Z[i, j] <= float32(0.0):
                dZ[i, j] = float32(0.0)

    @cuda.jit
    def _sigmoid_grad_inplace(dZ, A):         # dZ *= A*(1-A)
        i, j = cuda.grid(2)
        if i < dZ.shape[0] and j < dZ.shape[1]:
            a = A[i, j]
            dZ[i, j] *= a * (float32(1.0) - a)

    @cuda.jit
    def _softmax_rows(Z):                      # in-place row softmax
        i = cuda.grid(1)
        if i < Z.shape[0]:
            m = Z[i, 0]
            for j in range(1, Z.shape[1]):
                if Z[i, j] > m:
                    m = Z[i, j]
            s = float32(0.0)
            for j in range(Z.shape[1]):
                e = math.exp(Z[i, j] - m)
                Z[i, j] = e
                s += e
            for j in range(Z.shape[1]):
                Z[i, j] /= s

    @cuda.jit
    def _subtract(A, B, C):                    # C = A - B
        i, j = cuda.grid(2)
        if i < C.shape[0] and j < C.shape[1]:
            C[i, j] = A[i, j] - B[i, j]

    @cuda.jit
    def _colsum(A, out):                       # out[j] = sum_i A[i,j]
        j = cuda.grid(1)
        if j < A.shape[1]:
            s = float32(0.0)
            for i in range(A.shape[0]):
                s += A[i, j]
            out[j] = s

    @cuda.jit
    def _sgd_W(W, dW, V, lr, momentum, inv_batch, clipv):
        i, j = cuda.grid(2)
        if i < W.shape[0] and j < W.shape[1]:
            g = dW[i, j] * inv_batch
            if clipv > float32(0.0):           # per-element gradient clipping
                if g > clipv:
                    g = clipv
                elif g < -clipv:
                    g = -clipv
            v = momentum * V[i, j] - lr * g
            V[i, j] = v
            W[i, j] += v

    @cuda.jit
    def _sgd_b(b, db, vb, lr, momentum, inv_batch, clipv):
        j = cuda.grid(1)
        if j < b.shape[0]:
            g = db[j] * inv_batch
            if clipv > float32(0.0):
                if g > clipv:
                    g = clipv
                elif g < -clipv:
                    g = -clipv
            v = momentum * vb[j] - lr * g
            vb[j] = v
            b[j] += v


def _grid1(n, tpb=256):
    return (n + tpb - 1) // tpb, tpb


class FastMLPGPU:
    """Device-resident batched MLP (softmax + cross-entropy)."""

    def __init__(self, layer_sizes: List[int], seed: int = 1,
                 hidden_activation: str = "relu", batch_size: int = 256):
        if not CUDA_AVAILABLE:
            raise RuntimeError("FastMLPGPU needs a CUDA GPU; "
                               "use gpu_mlp.MLPGPU for the CPU path.")
        self.layer_sizes = list(layer_sizes)
        self.L = len(layer_sizes) - 1
        self.hidden_activation = hidden_activation
        self.B = batch_size

        rng = np.random.default_rng(seed)
        self.W_host, self.b_host = [], []
        for i in range(self.L):
            fin, fout = layer_sizes[i], layer_sizes[i + 1]
            if hidden_activation == "relu":
                W = rng.standard_normal((fin, fout)) * np.sqrt(2.0 / fin)
            else:
                lim = np.sqrt(6.0 / (fin + fout))
                W = rng.uniform(-lim, lim, (fin, fout))
            self.W_host.append(W.astype(np.float32))
            self.b_host.append(np.zeros(fout, np.float32))

        # upload weights + momentum buffers to the device (stay there)
        self.dW = [cuda.to_device(w) for w in self.W_host]
        self.db = [cuda.to_device(b) for b in self.b_host]
        self.vW = [cuda.to_device(np.zeros_like(w)) for w in self.W_host]
        self.vb = [cuda.to_device(np.zeros_like(b)) for b in self.b_host]
        self._buffers_ready = False

    # ------------------------------------------------------------------
    def _alloc_buffers(self, B):
        ls = self.layer_sizes
        self.d_Z = [None] + [cuda.device_array((B, ls[l + 1]), np.float32)
                             for l in range(self.L)]
        self.d_A = [None] + [cuda.device_array((B, ls[l + 1]), np.float32)
                             for l in range(self.L)]
        self.d_dZ = [None] + [cuda.device_array((B, ls[l + 1]), np.float32)
                              for l in range(self.L)]
        self.d_gW = [cuda.device_array_like(w) for w in self.W_host]
        self.d_gb = [cuda.device_array_like(b) for b in self.b_host]
        self._buffers_ready = True
        self._data_n = -1  # forces data-buffer (re)allocation on first epoch

    def _ensure_data_buffers(self, n):
        """Persistent device buffers holding ONE shuffled copy of the dataset.
        Allocated once; reused every epoch so batches are zero-copy views."""
        n_used = (n // self.B) * self.B
        if getattr(self, "_data_n", -1) == n_used:
            return n_used
        self.d_Xs = cuda.device_array((n_used, self.layer_sizes[0]), np.float32)
        self.d_Ys = cuda.device_array((n_used, self.layer_sizes[-1]), np.float32)
        self.d_perm = cuda.device_array((n_used,), np.int32)
        self._data_n = n_used
        return n_used

    # ------------------------------------------------------------------
    def _forward_device(self, A0):
        """A0: device array (B, n_in). Fills d_Z/d_A; returns probs buffer."""
        A_prev = A0
        for l in range(1, self.L + 1):
            Wd = self.dW[l - 1]
            bpg, tpb = grid_2d(A_prev.shape[0], Wd.shape[1])
            # tiled (shared-memory) matmul -- faster than the naive one
            kernels.matmul_tiled_kernel[bpg, tpb](A_prev, Wd, self.d_Z[l])
            _add_bias[bpg, tpb](self.d_Z[l], self.db[l - 1])
            if l < self.L:
                if self.hidden_activation == "relu":
                    _relu_fwd[bpg, tpb](self.d_Z[l], self.d_A[l])
                else:
                    _sigmoid_fwd[bpg, tpb](self.d_Z[l], self.d_A[l])
                A_prev = self.d_A[l]
            else:
                _softmax_rows[_grid1(A_prev.shape[0])](self.d_Z[l])  # probs
        return self.d_Z[self.L]

    # ------------------------------------------------------------------
    def train_epoch(self, d_X, d_Y, n, lr, momentum, clipv, perm_host):
        """One epoch over device data. perm_host: shuffled row indices (host).

        Optimisation: we shuffle the WHOLE dataset once (a single gather kernel)
        into a persistent device buffer, then iterate batches as zero-copy slice
        views -- no per-batch allocation or host->device transfer.
        """
        B = self.B
        if not self._buffers_ready:
            self._alloc_buffers(B)
        n_used = self._ensure_data_buffers(n)
        n_batches = n_used // B

        # one gather of the full (truncated) dataset into the shuffled buffers
        self.d_perm.copy_to_device(np.ascontiguousarray(perm_host[:n_used], np.int32))
        gX = grid_2d(n_used, self.layer_sizes[0])
        gY = grid_2d(n_used, self.layer_sizes[-1])
        _gather_kernel[gX[0], gX[1]](d_X, self.d_perm, self.d_Xs)
        _gather_kernel[gY[0], gY[1]](d_Y, self.d_perm, self.d_Ys)

        for bi in range(n_batches):
            Xb = self.d_Xs[bi * B:(bi + 1) * B]   # zero-copy device views
            Yb = self.d_Ys[bi * B:(bi + 1) * B]

            probs = self._forward_device(Xb)

            # output error: dZ_L = probs - Y
            bpg, tpb = grid_2d(B, self.layer_sizes[-1])
            _subtract[bpg, tpb](probs, Yb, self.d_dZ[self.L])

            for l in range(self.L, 0, -1):
                A_prev = Xb if l == 1 else self.d_A[l - 1]
                # dW = A_prevᵀ @ dZ
                g2 = grid_2d(self.layer_sizes[l - 1], self.layer_sizes[l])
                _mm_tn[g2[0], g2[1]](A_prev, self.d_dZ[l], self.d_gW[l - 1])
                # db = colsum(dZ)
                _colsum[_grid1(self.layer_sizes[l])](self.d_dZ[l], self.d_gb[l - 1])
                if l > 1:
                    gp = grid_2d(B, self.layer_sizes[l - 1])
                    _mm_nt[gp[0], gp[1]](self.d_dZ[l], self.dW[l - 1],
                                         self.d_dZ[l - 1])
                    if self.hidden_activation == "relu":
                        _relu_grad_inplace[gp[0], gp[1]](self.d_dZ[l - 1],
                                                         self.d_Z[l - 1])
                    else:
                        _sigmoid_grad_inplace[gp[0], gp[1]](self.d_dZ[l - 1],
                                                            self.d_A[l - 1])

            inv_b = np.float32(1.0 / B)
            lr32, mom32, clip32 = np.float32(lr), np.float32(momentum), np.float32(clipv)
            for l in range(self.L):
                gW = grid_2d(self.W_host[l].shape[0], self.W_host[l].shape[1])
                _sgd_W[gW[0], gW[1]](self.dW[l], self.d_gW[l], self.vW[l],
                                     lr32, mom32, inv_b, clip32)
                _sgd_b[_grid1(self.b_host[l].shape[0])](self.db[l], self.d_gb[l],
                                                        self.vb[l], lr32, mom32,
                                                        inv_b, clip32)
        cuda.synchronize()

    # ------------------------------------------------------------------
    # metrics use host weights (cheap copy once per epoch) + NumPy forward
    # ------------------------------------------------------------------
    def sync_to_host(self):
        self.W_host = [d.copy_to_host() for d in self.dW]
        self.b_host = [d.copy_to_host() for d in self.db]

    def _np_forward(self, X):
        cur = np.ascontiguousarray(X, np.float32)
        for l in range(self.L):
            z = cur @ self.W_host[l] + self.b_host[l]
            if l < self.L - 1:
                cur = (np.maximum(0, z) if self.hidden_activation == "relu"
                       else 1.0 / (1.0 + np.exp(-z)))
            else:
                z = z - z.max(axis=1, keepdims=True)
                e = np.exp(z)
                cur = e / e.sum(axis=1, keepdims=True)
        return cur

    def evaluate(self, X, y):
        self.sync_to_host()
        pred = np.argmax(self._np_forward(X), axis=1)
        return float(np.mean(pred == np.asarray(y)))

    def loss(self, X, Y):
        self.sync_to_host()
        p = np.clip(self._np_forward(X), 1e-12, 1.0)
        return float(-np.sum(Y * np.log(p)) / X.shape[0])

    def predict_proba(self, X):
        self.sync_to_host()
        return self._np_forward(X)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def n_params(self):
        return sum(w.size for w in self.W_host) + sum(b.size for b in self.b_host)

    # ------------------------------------------------------------------
    def save(self, path):
        self.sync_to_host()
        d = {"layer_sizes": np.array(self.layer_sizes),
             "hidden_activation": self.hidden_activation}
        for i in range(self.L):
            d[f"W{i}"] = self.W_host[i]
            d[f"b{i}"] = self.b_host[i]
        np.savez(path, **d)

    @classmethod
    def load(cls, path, batch_size: int = 256):
        z = np.load(path, allow_pickle=True)
        layer_sizes = [int(x) for x in z["layer_sizes"]]
        net = cls(layer_sizes, hidden_activation=str(z["hidden_activation"]),
                  batch_size=batch_size)
        net.W_host = [z[f"W{i}"].astype(np.float32) for i in range(net.L)]
        net.b_host = [z[f"b{i}"].astype(np.float32) for i in range(net.L)]
        net.dW = [cuda.to_device(w) for w in net.W_host]
        net.db = [cuda.to_device(b) for b in net.b_host]
        return net


if CUDA_AVAILABLE:
    @cuda.jit
    def _gather_kernel(src, idx, dst):
        i, j = cuda.grid(2)
        if i < dst.shape[0] and j < dst.shape[1]:
            dst[i, j] = src[idx[i], j]

    def _gather_rows(d_src, d_idx, n_cols):
        dst = cuda.device_array((d_idx.shape[0], n_cols), np.float32)
        bpg, tpb = grid_2d(d_idx.shape[0], n_cols)
        _gather_kernel[bpg, tpb](d_src, d_idx, dst)
        return dst
