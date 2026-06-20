"""
gpu.py
======
Host-side (CPU) orchestration for the GPU kernels in ``kernels.py``.

The kernels themselves only describe what *one* GPU thread does.  This module is
the "driver": it
  1. moves matrices Host (CPU) -> Device (GPU) with ``cuda.to_device``,
  2. computes how many 16x16 thread-blocks are needed (rounding UP so any batch
     size / matrix size works),
  3. launches the kernel,
  4. calls ``cuda.synchronize()`` so the CPU waits for the GPU to finish, and
  5. copies the result back Device -> Host with ``copy_to_host()``.

It also provides the *baselines* we benchmark against:
  * ``naive_matmul``  -- pure-Python triple-nested loops (the "slow lane").
  * ``numpy_matmul``  -- NumPy's BLAS (a fair, optimised CPU reference).

Every GPU entry point degrades gracefully: if there is no CUDA device we fall
back to NumPy so training/inference still works on any laptop.
"""

from __future__ import annotations

import numpy as np

from .kernels import (
    CUDA_AVAILABLE,
    TPB,
    matmul_kernel,
    matmul_tiled_kernel,
    add_bias_kernel,
    sigmoid_kernel,
    relu_kernel,
)

if CUDA_AVAILABLE:
    from numba import cuda


# ---------------------------------------------------------------------------
# Grid configuration helper
# ---------------------------------------------------------------------------
def grid_2d(n_rows: int, n_cols: int, tpb: int = TPB):
    """Return ``(blocks_per_grid, threads_per_block)`` for a 2-D launch.

    Threads-per-block is fixed at ``tpb x tpb`` (16x16 = 256 threads).

    The crucial detail (a hard requirement of this project): we round the number
    of blocks UP using ``(n + tpb - 1) // tpb``.  Plain integer division would
    round *down* and silently drop the last partial block -- e.g. 1000 rows with
    16-row blocks needs 63 blocks (62*16 = 992 < 1000), and ``1000 // 16 == 62``
    would leave the final 8 rows uncomputed.  Adding ``tpb - 1`` before the
    floor-division guarantees full coverage for ANY size, which is exactly why
    every kernel also carries an ``if row < shape ...`` bounds guard for the
    extra threads this creates.

    Axis convention (must match the kernels): Numba's ``cuda.grid(2)`` returns
    ``(x, y)`` where ``x`` is tied to ``threadIdx.x`` / ``blockIdx.x`` and
    ``y`` to the ``.y`` indices.  The kernels read ``row, col = cuda.grid(2)``
    -- so ``x`` indexes ROWS and ``y`` indexes COLS.  Therefore the first grid
    dimension must size the ROWS and the second the COLS.
    """
    threads_per_block = (tpb, tpb)
    blocks_per_grid_x = (n_rows + tpb - 1) // tpb   # x maps to rows
    blocks_per_grid_y = (n_cols + tpb - 1) // tpb   # y maps to cols
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)
    return blocks_per_grid, threads_per_block


# ---------------------------------------------------------------------------
# The slow lane: naive pure-Python matrix multiply (NO NumPy, NO GPU)
# ---------------------------------------------------------------------------
def naive_matmul(A, B):
    """Triple-nested-loop matrix multiply on a single CPU core.

    This is intentionally the "textbook" O(M*N*K) implementation with three
    explicit Python loops and no vectorisation.  It is the reference we beat in
    ``benchmark.py``.

    Naive logic: the outer two loops visit every output cell one-at-a-time, in
    sequence, on one core.  Compare with ``gpu_matmul`` where all those cells are
    computed simultaneously across the GPU's cores.
    """
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    m, k = A.shape
    k2, n = B.shape
    if k != k2:
        raise ValueError(f"shape mismatch: {A.shape} @ {B.shape}")

    # Plain Python lists -> guarantees we are NOT secretly using NumPy's BLAS.
    a = A.tolist()
    b = B.tolist()
    c = [[0.0] * n for _ in range(m)]
    for i in range(m):          # for each output row
        ai = a[i]
        ci = c[i]
        for j in range(n):      # for each output column
            s = 0.0
            for p in range(k):  # dot product along the shared dimension
                s += ai[p] * b[p][j]
            ci[j] = s
    return np.array(c, dtype=np.float64)


# ---------------------------------------------------------------------------
# A fair CPU reference: NumPy / BLAS
# ---------------------------------------------------------------------------
def numpy_matmul(A, B):
    """Optimised CPU matrix multiply via NumPy (multi-threaded BLAS)."""
    return np.asarray(A, dtype=np.float32) @ np.asarray(B, dtype=np.float32)


# ---------------------------------------------------------------------------
# The fast lane: GPU matrix multiply
# ---------------------------------------------------------------------------
def gpu_matmul(A, B, tiled: bool = True):
    """Multiply ``A @ B`` on the GPU and return a NumPy array on the host.

    Steps mirror the canonical CUDA workflow:
        host arrays --to_device--> device arrays
        launch kernel over a rounded-up 16x16 grid
        cuda.synchronize()   (block until the GPU is done)
        device result --copy_to_host--> NumPy array

    If no CUDA device is present we transparently fall back to NumPy so callers
    never have to special-case the no-GPU machine.
    """
    A = np.ascontiguousarray(A, dtype=np.float32)
    B = np.ascontiguousarray(B, dtype=np.float32)
    m, k = A.shape
    k2, n = B.shape
    if k != k2:
        raise ValueError(f"shape mismatch: {A.shape} @ {B.shape}")

    if not CUDA_AVAILABLE:
        # Graceful CPU fallback -- identical math, just not parallel.
        return numpy_matmul(A, B)

    # 1) Host -> Device.  These transfers are why GPU only wins on BIG matrices:
    #    for tiny ones the copy cost dwarfs the compute saved.
    d_A = cuda.to_device(A)
    d_B = cuda.to_device(B)
    d_C = cuda.device_array((m, n), dtype=np.float32)

    # 2) Configure the grid (rounded UP) and launch.
    blocks_per_grid, threads_per_block = grid_2d(m, n)
    kernel = matmul_tiled_kernel if tiled else matmul_kernel
    kernel[blocks_per_grid, threads_per_block](d_A, d_B, d_C)

    # 3) Make the CPU wait for every thread to finish.  Without this, a later
    #    read or a follow-up kernel could race ahead of the unfinished compute.
    cuda.synchronize()

    # 4) Device -> Host.
    return d_C.copy_to_host()


# ---------------------------------------------------------------------------
# Fused batched linear layer + activation, entirely on the GPU
# ---------------------------------------------------------------------------
def gpu_linear_activation(X, W, b, activation: str = "sigmoid", tiled: bool = True):
    """Compute ``activation(X @ W + b)`` for a whole batch on the GPU.

    This is what makes the *batched* forward pass fast: a layer's matmul, bias
    add, and non-linear transfer all run as GPU kernels with the data kept on the
    device between steps (only one upload, one download).

    Parameters
    ----------
    X : (batch, in_features)
    W : (in_features, units)
    b : (units,)
    activation : "sigmoid" | "relu" | "linear"
    """
    X = np.ascontiguousarray(X, dtype=np.float32)
    W = np.ascontiguousarray(W, dtype=np.float32)
    b = np.ascontiguousarray(b, dtype=np.float32)
    batch, in_features = X.shape
    units = W.shape[1]

    if not CUDA_AVAILABLE:
        Z = X @ W + b
        if activation == "sigmoid":
            return 1.0 / (1.0 + np.exp(-Z))
        if activation == "relu":
            return np.maximum(0.0, Z)
        return Z

    d_X = cuda.to_device(X)
    d_W = cuda.to_device(W)
    d_b = cuda.to_device(b)
    d_Z = cuda.device_array((batch, units), dtype=np.float32)

    bpg, tpb = grid_2d(batch, units)

    # Z = X @ W
    (matmul_tiled_kernel if tiled else matmul_kernel)[bpg, tpb](d_X, d_W, d_Z)
    cuda.synchronize()

    # Z += b
    add_bias_kernel[bpg, tpb](d_Z, d_b)
    cuda.synchronize()

    # Z = activation(Z)
    if activation == "sigmoid":
        sigmoid_kernel[bpg, tpb](d_Z)
        cuda.synchronize()
    elif activation == "relu":
        relu_kernel[bpg, tpb](d_Z)
        cuda.synchronize()
    # "linear" -> no transfer kernel

    return d_Z.copy_to_host()


def device_info() -> str:
    """Human-readable one-liner about the active compute backend."""
    if not CUDA_AVAILABLE:
        return "CUDA not available -> using NumPy/CPU fallback"
    try:
        gpu = cuda.get_current_device()
        name = gpu.name.decode() if isinstance(gpu.name, bytes) else str(gpu.name)
        cc = f"{gpu.compute_capability[0]}.{gpu.compute_capability[1]}"
        return f"CUDA device: {name} (compute capability {cc})"
    except Exception as exc:  # pragma: no cover
        return f"CUDA available but device query failed: {exc}"
