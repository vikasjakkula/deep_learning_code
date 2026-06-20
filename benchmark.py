"""
benchmark.py
============
Demonstrate the GPU "expressway" vs. the naive CPU "slow lane".

We multiply two 1000x1000 matrices three ways and time each:

  1. **Naive CPU**  -- pure-Python triple-nested loops (one core, no NumPy).
  2. **NumPy CPU**  -- optimised multi-threaded BLAS (a fair modern baseline).
  3. **GPU (CUDA)** -- our ``matmul`` kernels on the RTX 3050.

On a CUDA laptop you should see the GPU finish a 1000x1000 multiply on the order
of ~1000x faster than the naive loops -- the headline figure of the brief.
Because the naive version is genuinely O(n^3) in interpreted Python it is slow,
so for the big size we *measure* a smaller naive run and extrapolate, while
NumPy and the GPU are always measured at the full size.

Run:
    python benchmark.py
    python benchmark.py --size 1000 --naive-size 256
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from nnscratch import gpu
from nnscratch.kernels import CUDA_AVAILABLE


def time_it(fn, *args, repeats: int = 1):
    """Return (result, best_seconds) over ``repeats`` runs."""
    best = float("inf")
    result = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        result = fn(*args)
        best = min(best, time.perf_counter() - t0)
    return result, best


def fmt(seconds: float) -> str:
    if seconds < 1e-3:
        return f"{seconds * 1e6:8.2f} us"
    if seconds < 1.0:
        return f"{seconds * 1e3:8.2f} ms"
    return f"{seconds:8.3f} s "


def main():
    ap = argparse.ArgumentParser(description="GPU vs CPU matmul benchmark")
    ap.add_argument("--size", type=int, default=1000,
                    help="N for the full N x N benchmark (default 1000)")
    ap.add_argument("--naive-size", type=int, default=256,
                    help="N used to time the naive loops, then extrapolated")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    N = args.size
    NS = min(args.naive_size, N)

    print("=" * 68)
    print(" Matrix-Multiplication Benchmark:  Naive CPU  vs  NumPy  vs  GPU")
    print("=" * 68)
    print(f" Backend: {gpu.device_info()}")
    print(f" Full benchmark size : {N} x {N}")
    print(f" Naive timing size   : {NS} x {NS} (extrapolated to {N}x{N})")
    print("-" * 68)

    rng = np.random.default_rng(args.seed)
    A = rng.standard_normal((N, N)).astype(np.float32)
    B = rng.standard_normal((N, N)).astype(np.float32)

    # --- 1) Naive pure-Python (timed small, extrapolated by the n^3 law) ---
    A_s = A[:NS, :NS].copy()
    B_s = B[:NS, :NS].copy()
    _, naive_small = time_it(gpu.naive_matmul, A_s, B_s)
    # Work scales as n^3, so multiply by (N/NS)^3 to estimate the full run.
    scale = (N / NS) ** 3
    naive_full_est = naive_small * scale
    print(f" 1) Naive  CPU (loops) : measured {fmt(naive_small)} at {NS}x{NS}")
    print(f"                          -> est. {fmt(naive_full_est)} at {N}x{N}")

    # --- 2) NumPy BLAS (full size) ---
    C_np, t_np = time_it(gpu.numpy_matmul, A, B, repeats=3)
    print(f" 2) NumPy  CPU (BLAS)  : {fmt(t_np)} at {N}x{N}")

    # --- 3) GPU kernels (full size) ---
    if CUDA_AVAILABLE:
        # Warm-up so JIT compilation is not counted in the timing.
        _ = gpu.gpu_matmul(A[:64, :64], B[:64, :64])
        C_gpu, t_gpu = time_it(gpu.gpu_matmul, A, B, repeats=3)
        backend = "GPU (CUDA kernels)"
    else:
        C_gpu, t_gpu = time_it(gpu.gpu_matmul, A, B, repeats=3)
        backend = "GPU path (NumPy fallback - no CUDA device here)"
    print(f" 3) {backend:<34}: {fmt(t_gpu)} at {N}x{N}")

    # --- correctness check (GPU path vs NumPy) ---
    max_err = float(np.max(np.abs(C_gpu - C_np)))
    print("-" * 68)
    print(f" Correctness: max|GPU - NumPy| = {max_err:.3e}  "
          f"({'OK' if max_err < 1e-2 else 'CHECK'})")

    # --- speed-ups ---
    print("-" * 68)
    print(" Speed-ups (higher = faster):")
    print(f"   GPU vs Naive (est.) : {naive_full_est / t_gpu:10.1f} x")
    print(f"   GPU vs NumPy        : {t_np / t_gpu:10.1f} x")
    print(f"   NumPy vs Naive(est) : {naive_full_est / t_np:10.1f} x")
    print("=" * 68)
    if not CUDA_AVAILABLE:
        print(" NOTE: no CUDA device detected -> the 'GPU' row used the NumPy")
        print("       fallback. Run on the RTX 3050 (with numba + CUDA toolkit)")
        print("       to see the true ~1000x GPU-vs-naive speed-up.")


if __name__ == "__main__":
    main()
