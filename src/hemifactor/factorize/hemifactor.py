"""Floored hemifactor construction (Phase 3 stubs)."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class Hemifactor:
    """W ≈ A @ B (+ optional tail handled elsewhere)."""

    A: np.ndarray  # (m, k)
    B: np.ndarray  # (k, n)
    k: int
    role: str
    layer: int | None = None

    def nbytes_at_bytes_per_param(self, bpw: float) -> int:
        """Estimate stored bytes if factors use ``bpw`` bits per scalar element."""
        n_elem = int(self.A.size + self.B.size)
        return int(math.ceil(n_elem * bpw / 8.0))

    def matvec_fp64(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64).ravel()
        return self.A @ (self.B @ x)


def from_truncated_svd(W: np.ndarray, k: int, role: str, layer: int | None = None) -> Hemifactor:
    """Vanilla truncated SVD baseline (whitening applied by caller on W)."""
    U, S, Vt = np.linalg.svd(W, full_matrices=False)
    k = min(k, S.shape[0])
    A = U[:, :k] * S[:k]
    B = Vt[:k, :]
    return Hemifactor(A=A.astype(np.float64), B=B.astype(np.float64), k=k, role=role, layer=layer)
