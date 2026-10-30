"""Distortion floor D(W, μ) proxies (Phase 1).

True D compares stored codes to a higher-rate copy. When the file already stores
Q6 for a tensor, the control bar uses a Q4_0 round-trip of the dequantized map
(gguf Python cannot pack Q4_K). Artifacts label the control codec explicitly.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def activation_mse(W: np.ndarray, W_hat: np.ndarray, x: np.ndarray) -> float:
    """E ‖(W − W_hat) x‖² for row-batch x and maps shaped (out, in)."""
    W = np.asarray(W, dtype=np.float64)
    W_hat = np.asarray(W_hat, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    if W.shape[1] != x.shape[1]:
        raise ValueError(f"W in-dim {W.shape[1]} != x {x.shape[1]}")
    y = x @ W.T
    y_hat = x @ W_hat.T
    err = y - y_hat
    return float(np.mean(np.sum(err * err, axis=1)))


def quant_roundtrip(
    W: np.ndarray,
    qtype_name: str = "Q4_0",
) -> tuple[np.ndarray, dict[str, Any]]:
    from gguf import GGMLQuantizationType
    from gguf.quants import dequantize, quantize

    qtype = getattr(GGMLQuantizationType, qtype_name)
    W32 = np.asarray(W, dtype=np.float32)
    packed = quantize(W32, qtype)
    W_hat = dequantize(packed, qtype).astype(np.float64)
    meta = {
        "codec": qtype_name,
        "note": "Control codec for D; not bitwise Q4_K from the original file.",
        "weight_mse": float(np.mean((W32.astype(np.float64) - W_hat) ** 2)),
    }
    return W_hat, meta


def distortion_vs_roundtrip(
    W: np.ndarray,
    x: np.ndarray,
    qtype_name: str = "Q4_0",
) -> dict[str, Any]:
    W_hat, meta = quant_roundtrip(W, qtype_name=qtype_name)
    d = activation_mse(W, W_hat, x)
    y_norm = float(np.mean(np.sum((x @ np.asarray(W, dtype=np.float64).T) ** 2, axis=1)))
    return {
        **meta,
        "D_activation_mse": d,
        "output_energy": y_norm,
        "D_relative": d / (y_norm + 1e-30),
        "n_samples": int(np.asarray(x).reshape(-1, x.shape[-1]).shape[0]),
    }
