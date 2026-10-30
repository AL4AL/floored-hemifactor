"""Activation sampling for μ without a full Transformer runtime.

Uses dequantized gate/up (or residual proxies from embeddings) so Phase 1/2 can
run on GGUF alone. Artifacts must record ``method`` so results are not confused
with hooked end-to-end forwards.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hemifactor.calibrate.welford import WelfordCov


def swiglu(gate: np.ndarray, up: np.ndarray) -> np.ndarray:
    """SiLU(gate) * up, row-batch."""
    sig = 1.0 / (1.0 + np.exp(-gate))
    return (gate * sig) * up


@dataclass
class ActivationBatch:
    residual: np.ndarray  # (B, d)
    ff_intermediate: np.ndarray | None = None  # (B, n_ff)


def sample_residual_gaussian(
    n: int,
    dim: int,
    rms: float = 1.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    rng = rng or np.random.default_rng(0)
    return rng.standard_normal((n, dim), dtype=np.float64) * (rms / np.sqrt(dim))


def sample_embedding_rows(
    embd: np.ndarray,
    n: int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """``embd`` is (n_embd, n_vocab) or (n_vocab, n_embd); detect orientation."""
    rng = rng or np.random.default_rng(0)
    if embd.shape[0] < embd.shape[1]:
        # (n_embd, n_vocab)
        idx = rng.integers(0, embd.shape[1], size=n)
        return np.asarray(embd[:, idx].T, dtype=np.float64)
    idx = rng.integers(0, embd.shape[0], size=n)
    return np.asarray(embd[idx], dtype=np.float64)


def mlp_intermediate(
    residual: np.ndarray,
    W_gate: np.ndarray,
    W_up: np.ndarray,
) -> np.ndarray:
    """Apply SiLU(gate)·up. GGUF dequant gives gate/up as (n_ff, n_embd)."""
    x = np.asarray(residual, dtype=np.float64)
    if W_gate.shape[1] != x.shape[1]:
        raise ValueError(f"gate in-dim {W_gate.shape[1]} != residual {x.shape[1]}")
    g = x @ W_gate.T
    u = x @ W_up.T
    return swiglu(g, u)


def accumulate_cov(
    samples: np.ndarray,
    sketch_dim: int | None = None,
    rng: np.random.Generator | None = None,
) -> dict:
    """Return dense or sketched covariance stats."""
    x = np.asarray(samples, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("samples must be (batch, dim)")
    dim = x.shape[1]
    if sketch_dim is None or sketch_dim >= dim:
        w = WelfordCov(dim)
        w.update_batch(x)
        cov = w.covariance()
        return {
            "n": w.n,
            "dim": dim,
            "mean": w.mean,
            "cov": cov,
            "participation_ratio": w.participation_ratio(),
            "sketch": False,
        }
    rng = rng or np.random.default_rng(0)
    # Gaussian sketch: S^T Cov S with S (dim, sketch)
    s = rng.standard_normal((dim, sketch_dim), dtype=np.float64) / np.sqrt(sketch_dim)
    y = x @ s
    w = WelfordCov(sketch_dim)
    w.update_batch(y)
    cov_s = w.covariance()
    # Diagonal energy proxy on full dim
    var = np.var(x, axis=0, ddof=1)
    return {
        "n": w.n,
        "dim": dim,
        "sketch_dim": sketch_dim,
        "mean": x.mean(axis=0),
        "cov_sketch": cov_s,
        "proj": s,
        "diag_var": var,
        "participation_ratio_diag": float((var.sum() ** 2) / (np.sum(var**2) + 1e-30)),
        "sketch": True,
    }
