"""Online covariance (Welford) for residual activations."""

from __future__ import annotations

import numpy as np


class WelfordCov:
    """Accumulate mean and covariance of row-vectors shaped (d,)."""

    def __init__(self, dim: int, dtype=np.float64):
        self.dim = dim
        self.dtype = dtype
        self.n = 0
        self.mean = np.zeros(dim, dtype=dtype)
        self._c = np.zeros((dim, dim), dtype=dtype)  # running sum of outer products of centered? use two-pass style

    def update_batch(self, x: np.ndarray) -> None:
        """x: (batch, dim) or (dim,)"""
        x = np.asarray(x, dtype=self.dtype)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if x.shape[1] != self.dim:
            raise ValueError(f"expected dim {self.dim}, got {x.shape}")
        for row in x:
            self.n += 1
            delta = row - self.mean
            self.mean += delta / self.n
            delta2 = row - self.mean
            self._c += np.outer(delta, delta2)

    def covariance(self, ddof: int = 1) -> np.ndarray:
        if self.n <= ddof:
            raise RuntimeError("not enough samples for covariance")
        return self._c / (self.n - ddof)

    def participation_ratio(self) -> float:
        cov = self.covariance()
        # tr(Σ)^2 / ||Σ||_F^2
        tr = float(np.trace(cov))
        fro = float(np.linalg.norm(cov, ord="fro"))
        if fro == 0:
            return 0.0
        return (tr * tr) / (fro * fro)
