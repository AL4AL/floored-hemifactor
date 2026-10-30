"""NumPy fused factor path — never materializes dense W (Phase 4 stand-in)."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from hemifactor.factorize.hemifactor import Hemifactor


def fused_matvec(hf: Hemifactor, x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).ravel()
    return hf.A @ (hf.B @ x)


def fused_batch(hf: Hemifactor, x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return (x @ hf.B.T) @ hf.A.T


def bench_fused(
    hf: Hemifactor,
    x: np.ndarray,
    repeats: int = 50,
    warmup: int = 5,
) -> dict[str, Any]:
    x = np.asarray(x, dtype=np.float64).ravel()
    for _ in range(warmup):
        fused_matvec(hf, x)
    t0 = time.perf_counter()
    for _ in range(repeats):
        fused_matvec(hf, x)
    elapsed = time.perf_counter() - t0
    factor_bytes = hf.A.nbytes + hf.B.nbytes
    return {
        "repeats": repeats,
        "seconds": elapsed,
        "us_per_call": 1e6 * elapsed / repeats,
        "factor_bytes_fp64": factor_bytes,
        "impl": "numpy_fused",
        "note": "CPU stand-in; CUDA/Triton path is Phase 4 on-device work.",
    }
