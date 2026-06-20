"""
_cuda_setup.py
==============
Make Numba find the CUDA toolkit that ships as *pip wheels* (no conda, no
system-wide CUDA install required).

Why this exists
---------------
Numba's default library loader only knows how to find the CUDA toolkit via a
conda environment, a ``CUDA_HOME`` pointing at a classic toolkit install, or a
system install.  It does NOT auto-discover the ``nvidia-*-cu12`` pip wheels.
The newer "NVIDIA binding" path *does* understand pip wheels, but it is tied to
a specific ``cuda-python`` API that broke when cuda-python moved to 13.x
(``from cuda import cuda`` -> ``from cuda.bindings import driver``).

The pragmatic, version-proof fix: the ``nvidia-cuda-nvcc-cu12`` wheel already
lays its files out exactly the way Numba's ``CUDA_HOME`` probe expects
(``<root>/nvvm/bin/nvvm64_*.dll`` and ``<root>/nvvm/libdevice/*.bc``).  So we
locate that wheel and export ``CUDA_HOME`` ourselves, *before* ``numba.cuda`` is
imported.  We also register the wheels' ``bin`` dirs as DLL search paths so the
dependent runtime DLLs resolve on Windows.

This module is imported at the top of ``kernels.py`` and is a no-op if the user
already has a working CUDA setup (we never overwrite an existing ``CUDA_HOME``).
"""

from __future__ import annotations

import os
import sys
import glob


def _site_packages_nvidia() -> str | None:
    """Return the ``site-packages/nvidia`` directory for the active interpreter."""
    candidates = [
        os.path.join(sys.prefix, "Lib", "site-packages", "nvidia"),       # Windows
        os.path.join(
            sys.prefix, "lib",
            f"python{sys.version_info.major}.{sys.version_info.minor}",
            "site-packages", "nvidia",
        ),  # Linux
    ]
    for c in candidates:
        if os.path.isdir(c):
            return c
    # Fall back to scanning sys.path entries.
    for p in sys.path:
        cand = os.path.join(p, "nvidia")
        if os.path.isdir(cand):
            return cand
    return None


def configure() -> bool:
    """Best-effort: point Numba at the pip-wheel CUDA toolkit.

    Returns True if a wheel-based toolkit was found and wired up (or was already
    configured), False if nothing could be done (caller then falls back to CPU).
    """
    # Use Numba's own (stable) ctypes loader, not the cuda-python binding -- the
    # binding path is what breaks against cuda-python 13.x.
    os.environ.setdefault("NUMBA_CUDA_USE_NVIDIA_BINDING", "0")

    nvidia = _site_packages_nvidia()
    if nvidia is None:
        return bool(os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH"))

    # Register every wheel's bin/ dir so dependent DLLs load (Windows-safe).
    if hasattr(os, "add_dll_directory"):
        for bindir in glob.glob(os.path.join(nvidia, "*", "bin")) + \
                glob.glob(os.path.join(nvidia, "*", "nvvm", "bin")):
            try:
                os.add_dll_directory(bindir)
            except (OSError, FileNotFoundError):
                pass
    # Also extend PATH for good measure.
    extra = glob.glob(os.path.join(nvidia, "*", "bin")) + \
        glob.glob(os.path.join(nvidia, "*", "nvvm", "bin"))
    if extra:
        os.environ["PATH"] = os.pathsep.join(extra + [os.environ.get("PATH", "")])

    # Point CUDA_HOME at the nvcc wheel (it carries nvvm/bin + nvvm/libdevice),
    # unless the user already set one.
    nvcc_root = os.path.join(nvidia, "cuda_nvcc")
    has_nvvm = glob.glob(os.path.join(nvcc_root, "nvvm", "bin", "nvvm*"))

    # Numba also needs to read the CUDA *runtime* (cudart) version to decide
    # which GPU compute capabilities it supports -- but cudart ships in a
    # SEPARATE wheel (cuda_runtime), so Numba's `CUDA_HOME/bin` probe misses it
    # and concludes "no supported GPUs".  Fix: copy cudart into a bin/ folder
    # under the nvcc wheel so a single CUDA_HOME exposes nvvm + libdevice +
    # cudart together.  (~0.7 MB, done once.)
    if has_nvvm:
        nvcc_bin = os.path.join(nvcc_root, "bin")
        os.makedirs(nvcc_bin, exist_ok=True)
        for cudart in glob.glob(os.path.join(nvidia, "cuda_runtime", "bin", "cudart*")):
            dest = os.path.join(nvcc_bin, os.path.basename(cudart))
            if not os.path.exists(dest):
                try:
                    import shutil
                    shutil.copy2(cudart, dest)
                except OSError:
                    pass

    if has_nvvm and not (os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")):
        os.environ["CUDA_HOME"] = nvcc_root
        os.environ["CUDA_PATH"] = nvcc_root
        return True

    return bool(os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH") or has_nvvm)


# Run on import -- must happen before `from numba import cuda` anywhere.
configure()
