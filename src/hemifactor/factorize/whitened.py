"""Build hemifactors from whitened truncated SVD."""

from __future__ import annotations

import numpy as np

from hemifactor.factorize.hemifactor import Hemifactor
from hemifactor.spectrum.whiten import diag_sqrt, sym_sqrt_pinv


def hemifactor_whitened(
    W: np.ndarray,
    k: int,
    role: str,
    layer: int | None = None,
    cov: np.ndarray | None = None,
    diag_var: np.ndarray | None = None,
) -> Hemifactor:
    """Rank-k factor of W in the whitened metric, mapped back to original coords.

    Fit AB ≈ W on μ by truncating SVD of W Σ^{1/2}, then B ← B_w Σ^{-1/2}.
    """
    W = np.asarray(W, dtype=np.float64)
    if cov is not None:
        s_half = sym_sqrt_pinv(cov)
        # pseudo-inverse square root
        w, q = np.linalg.eigh(np.asarray(cov, dtype=np.float64))
        w = np.clip(w, 0.0, None)
        inv_scale = np.zeros_like(w)
        mask = w > 1e-8
        inv_scale[mask] = 1.0 / np.sqrt(w[mask])
        s_inv_half = (q * inv_scale) @ q.T
        M = W @ s_half
    elif diag_var is not None:
        d = diag_sqrt(diag_var)
        M = W * d[None, :]
        s_inv_half = None
        inv_d = 1.0 / d
    else:
        M = W
        s_inv_half = None
        inv_d = None

    U, S, Vt = np.linalg.svd(M, full_matrices=False)
    k = min(k, S.shape[0])
    A = U[:, :k] * S[:k]
    Bw = Vt[:k, :]
    if cov is not None:
        B = Bw @ s_inv_half
    elif diag_var is not None:
        B = Bw * inv_d[None, :]
    else:
        B = Bw
    return Hemifactor(A=A, B=B, k=k, role=role, layer=layer)
