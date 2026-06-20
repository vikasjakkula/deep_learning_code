"""
kernels.py
==========
Numba CUDA kernels -- the "Expressway" of the framework.

This module holds every function decorated with ``@cuda.jit``.  These are the
*device kernels*: small chunks of Python that Numba JIT-compiles to PTX and that
the NVIDIA driver then runs on thousands of GPU cores at once.

Why a separate file?
--------------------
The neural network *structure* (see ``engine.py``) is deliberately written in
plain, readable Python -- a list of layers, where every layer is a list of
neuron dictionaries.  That code is easy to reason about but slow for big matrix
math.  Whenever we need raw throughput (a 1000x1000 matrix multiply, a
batched forward pass) we hand the heavy lifting to the kernels below.

The contrast this project is built to demonstrate
-------------------------------------------------
*   **Naive logic** (``naive_matmul`` in ``gpu.py``): three nested ``for`` loops
    running on ONE CPU core.  Every output cell is computed one after another.
*   **Parallel logic** (``matmul_kernel`` below): we launch one GPU thread per
    output cell.  ``cuda.grid(2)`` tells each thread *which* (row, col) it owns,
    and they all run simultaneously.  On an RTX 3050 (2048 CUDA cores) this is
    where the ~1000x speed-up on large matrices comes from.

Hardware target: NVIDIA RTX 3050 Laptop GPU (Ampere, compute capability 8.6).

NOTE: importing this module never *requires* a GPU.  If Numba (or CUDA) is
missing we fall back to no-op decorators so the rest of the package still
imports cleanly and the pure-CPU paths keep working.  Only *launching* a kernel
needs real hardware.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Optional Numba import.
# ---------------------------------------------------------------------------
# Numba/CUDA is only present on a machine with the CUDA toolkit + drivers.
# We degrade gracefully so `import nnscratch` works everywhere (CI, laptops
# without an NVIDIA card, etc.).  `CUDA_AVAILABLE` is the single source of
# truth the rest of the package checks before trying to launch a kernel.
try:
    from numba import cuda, float32
    import math

    try:
        CUDA_AVAILABLE = cuda.is_available()
    except Exception:
        # A driver mismatch can raise rather than return False.
        CUDA_AVAILABLE = False
    NUMBA_AVAILABLE = True
except Exception:  # pragma: no cover - exercised only when numba is absent
    NUMBA_AVAILABLE = False
    CUDA_AVAILABLE = False

    # Stand-in so the @cuda.jit decorators below are still valid syntax.
    class _CudaStub:
        def jit(self, *args, **kwargs):
            def _decorator(func):
                return func

            # Support both @cuda.jit and @cuda.jit(...) usage.
            if len(args) == 1 and callable(args[0]) and not kwargs:
                return args[0]
            return _decorator

        def grid(self, n):  # never actually called without a GPU
            raise RuntimeError("cuda.grid called but Numba CUDA is unavailable")

    cuda = _CudaStub()
    float32 = float  # type: ignore


# The thread-block is a 16x16 = 256-thread square, as required by the spec.
# 256 threads/block is a sweet spot on Ampere: it keeps the SM occupied while
# leaving enough registers/shared memory per thread.  The host code in gpu.py
# imports this constant so the block size lives in exactly one place.
TPB = 16  # Threads-Per-Block, per dimension


# ---------------------------------------------------------------------------
# Kernel 1: dense matrix multiplication  C = A @ B
# ---------------------------------------------------------------------------
@cuda.jit
def matmul_kernel(A, B, C):
    """One GPU thread computes ONE element of the output matrix ``C``.

    ``cuda.grid(2)`` returns this thread's absolute (row, col) coordinate in the
    2-D launch grid -- it is shorthand for::

        row = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
        col = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

    Parameters
    ----------
    A : device array, shape (M, K)
    B : device array, shape (K, N)
    C : device array, shape (M, N)  -- written in place

    Parallel logic
    --------------
    The naive CPU version walks all M*N output cells with an outer double loop.
    Here that double loop *disappears*: each of the M*N cells is handed to its
    own thread, so they are all computed at the same time.  The only loop left
    inside a thread is the length-K dot product for its single cell.
    """
    row, col = cuda.grid(2)

    # Guard: because we round the grid size UP (see gpu.py), some threads fall
    # outside the real matrix.  They must do nothing -- otherwise they would
    # write out of bounds and corrupt memory.
    if row < C.shape[0] and col < C.shape[1]:
        acc = float32(0.0)
        for k in range(A.shape[1]):  # dot product of A's row and B's col
            acc += A[row, k] * B[k, col]
        C[row, col] = acc


# ---------------------------------------------------------------------------
# Kernel 2: tiled matrix multiplication using shared memory (advanced)
# ---------------------------------------------------------------------------
@cuda.jit
def matmul_tiled_kernel(A, B, C):
    """Shared-memory *tiled* matmul -- a faster variant of ``matmul_kernel``.

    The naive kernel re-reads A and B straight from slow global memory for every
    multiply.  Here each block cooperatively loads a 16x16 TILE of A and of B
    into fast on-chip *shared memory*, then every thread reuses those cached
    values.  This cuts global-memory traffic by ~TPB-fold and is the standard
    "CUDA matmul" taught in HPC courses.

    Both kernels produce identical results; this one is just the optimised path.
    """
    # Per-block shared scratchpads (visible to all 256 threads in the block).
    sA = cuda.shared.array(shape=(TPB, TPB), dtype=float32)
    sB = cuda.shared.array(shape=(TPB, TPB), dtype=float32)

    row, col = cuda.grid(2)
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y

    acc = float32(0.0)
    # Number of tiles we must slide across the K dimension (rounded up).
    n_tiles = (A.shape[1] + TPB - 1) // TPB

    for t in range(n_tiles):
        # --- cooperative load of one tile into shared memory ---
        a_col = t * TPB + tx
        if row < A.shape[0] and a_col < A.shape[1]:
            sA[ty, tx] = A[row, a_col]
        else:
            sA[ty, tx] = float32(0.0)

        b_row = t * TPB + ty
        if b_row < B.shape[0] and col < B.shape[1]:
            sB[ty, tx] = B[b_row, col]
        else:
            sB[ty, tx] = float32(0.0)

        # Wait until every thread has finished loading before we read the tile.
        cuda.syncthreads()

        # --- multiply the two tiles ---
        for k in range(TPB):
            acc += sA[ty, k] * sB[k, tx]

        # Wait until all threads are done with the tile before overwriting it.
        cuda.syncthreads()

    if row < C.shape[0] and col < C.shape[1]:
        C[row, col] = acc


# ---------------------------------------------------------------------------
# Kernel 3: add a bias row-vector to every row of a matrix (in place)
# ---------------------------------------------------------------------------
@cuda.jit
def add_bias_kernel(Z, bias):
    """``Z[i, j] += bias[j]`` for the whole (batch, units) matrix, in parallel.

    Used right after ``matmul_kernel`` so a full batched linear layer
    ``Z = X @ W + b`` stays entirely on the GPU.
    """
    row, col = cuda.grid(2)
    if row < Z.shape[0] and col < Z.shape[1]:
        Z[row, col] += bias[col]


# ---------------------------------------------------------------------------
# Kernel 4: elementwise sigmoid  ->  1 / (1 + e^-x)
# ---------------------------------------------------------------------------
@cuda.jit
def sigmoid_kernel(Z):
    """Apply the sigmoid transfer function to every element of ``Z`` in place.

    This is the non-linear "transfer" step from the KMIT Vista sessions, but
    vectorised: instead of looping neuron-by-neuron on the CPU we let one thread
    handle one activation value.
    """
    row, col = cuda.grid(2)
    if row < Z.shape[0] and col < Z.shape[1]:
        Z[row, col] = float32(1.0) / (float32(1.0) + math.exp(-Z[row, col]))


# ---------------------------------------------------------------------------
# Kernel 5: elementwise ReLU  ->  max(0, x)
# ---------------------------------------------------------------------------
@cuda.jit
def relu_kernel(Z):
    """Apply ReLU to every element of ``Z`` in place (a second transfer option)."""
    row, col = cuda.grid(2)
    if row < Z.shape[0] and col < Z.shape[1]:
        if Z[row, col] < float32(0.0):
            Z[row, col] = float32(0.0)
