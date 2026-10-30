"""Spend bits on kept hemifactor components only."""

from __future__ import annotations

from typing import Any

import numpy as np

from hemifactor.factorize.hemifactor import Hemifactor


def uniform_quantize(arr: np.ndarray, bits: int) -> tuple[np.ndarray, dict[str, Any]]:
    """Symmetric uniform quantize to ``bits`` (2–8), dequantized float64 return."""
    if bits < 2 or bits > 16:
        raise ValueError("bits out of range")
    a = np.asarray(arr, dtype=np.float64)
    maxabs = float(np.max(np.abs(a))) + 1e-30
    qmax = (1 << (bits - 1)) - 1
    scale = maxabs / qmax
    q = np.clip(np.round(a / scale), -qmax, qmax)
    recon = q * scale
    return recon, {"bits": bits, "scale": scale, "maxabs": maxabs}


def quantize_hemifactor(hf: Hemifactor, bits: int) -> tuple[Hemifactor, dict[str, Any]]:
    A_q, meta_a = uniform_quantize(hf.A, bits)
    B_q, meta_b = uniform_quantize(hf.B, bits)
    out = Hemifactor(A=A_q, B=B_q, k=hf.k, role=hf.role, layer=hf.layer)
    return out, {"A": meta_a, "B": meta_b, "nbytes_est": out.nbytes_at_bytes_per_param(bits)}
